"""Convert ABC-130K MCAP episodes to the export contract using YAM joint FK.

The MCAP streams are asynchronous; they are resampled onto the 30 fps frame sequence.
"""
from __future__ import annotations

from functools import cached_property
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from loom_data.video import VideoWriter
from loom_data.contract import episode_record, write_frames
from loom_data.sync import frame_grid
from loom_data.robot.yam import YamFK

from loom_data.sources.base import Source, task_limit

CAMERAS = {'/top-camera': 'top', '/top-left-camera': 'top',
           '/left-wrist-camera': 'left_wrist', '/right-wrist-camera': 'right_wrist'}
STATES = {f'/{side}-{kind}-state': f'{side}_{output}'
          for side in ('left', 'right')
          for kind, output in (('arm', 'eef'), ('ee', 'gripper'))}


def messages(path, topics):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    with path.open('rb') as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            timestamp = decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
            if timestamp != message.log_time:
                raise ValueError(f'{path}: payload timestamp differs from log_time on {channel.topic}')
            yield channel.topic, timestamp, decoded


def instructions(task, annotations, origin, end):
    if not task.strip() or end <= origin:
        raise ValueError('Missing task instruction or zero-duration episode')
    # Full task fills the prefix before the first subtask; subtask transitions
    # supersede it. Clamp annotations to the sampled episode interval.
    transitions = {origin: task}
    for timestamp, text in sorted(annotations, key=lambda item: item[0]):
        if not text.strip():
            raise ValueError('Empty subtask instruction')
        if timestamp < end:
            transitions[max(origin, timestamp)] = text
    points = sorted(transitions)
    return [{'start_ns': start-origin, 'end_ns': stop-origin, 'text': transitions[start]}
            for start, stop in zip(points, points[1:] + [end])]


def convert_episode(source, destination, fk):
    state = {name: ([], []) for name in STATES.values()}
    tasks, camera_topics, camera_times = [], {}, {}
    # Read all state and every camera's acquisition times before the expensive video
    # pass: the 30 fps frame times depend on the interval all streams share.
    for topic, timestamp, decoded in messages(source / 'episode.mcap',
                                              list(STATES) + ['/instruction'] + list(CAMERAS)):
        if topic == '/instruction':
            tasks.append(decoded.data)
            continue
        if topic in CAMERAS:
            camera = CAMERAS[topic]
            if camera_topics.setdefault(camera, topic) != topic:
                raise ValueError(f'{source}: multiple source topics for {camera}')
            camera_times.setdefault(camera, []).append(timestamp)
            continue
        name = STATES[topic]
        if name.endswith('_eef'):
            value = fk.pose(name.split('_')[0], decoded.position)
        else:
            if len(decoded.position) != 1:
                raise ValueError(f'{topic}: expected one gripper value')
            value = decoded.position[0]
            if not np.isfinite(value):
                raise ValueError(f'{source.name} {topic} at {timestamp}: openness {value!r} '
                                 'is not finite')
            value = min(1.0, max(0.0, value))
        state[name][0].append(timestamp)
        state[name][1].append(value)
    if any(not times for times, _ in state.values()):
        raise ValueError(f'{source}: requires all four state streams')
    if len(tasks) != 1 or not tasks[0].strip():
        raise ValueError(f'{source}: expected one nonempty /instruction message')
    if not camera_times:
        raise ValueError(f'{source}: requires at least one camera')
    streams = [times for times, _ in state.values()] + [sorted(times) for times in camera_times.values()]
    origin = min(times[0] for times in streams)
    end = max(times[-1] for times in streams)
    grid = frame_grid(streams)
    annotations = []
    if (source / 'annotation.mcap').exists():
        annotations = [(timestamp, decoded.data) for _, timestamp, decoded in
                       messages(source / 'annotation.mcap', ['/subtask-annotation'])]
    length = write_frames(destination, state, instructions(tasks[0], annotations, origin, end), origin, grid)
    writers = {}
    with ExitStack() as stack:
        for topic, timestamp, decoded in messages(source / 'episode.mcap', list(CAMERAS)):
            camera = CAMERAS[topic]
            if camera not in writers:
                writers[camera] = VideoWriter(destination / f'{camera}.mp4', decoded.format,
                                              camera_times[camera], grid)
                stack.callback(writers[camera].close)
            writers[camera].add(decoded.data, timestamp, decoded.format)
        videos = {camera: writer.finish() for camera, writer in writers.items()}
    return episode_record(source.name, length, videos)


class Abc130k(Source):
    name = 'abc130k'
    model = 'yam'
    camera_sets = (('left_wrist', 'right_wrist', 'top'),)

    @classmethod
    def discover(cls, config, progress):
        found = []
        with progress('Scanning for episode.mcap files', unit='episode') as bar:
            for path in Path(config['source']).rglob('episode.mcap'):
                found.append(path)
                bar.update()
        tasks = [{'source': str(path.parent), 'episode_ids': [path.parent.name],
                  'inputs': [str(path), str(path.parent / 'annotation.mcap')]}
                 for path in sorted(found)[:task_limit(config)]]
        return tasks, []

    @cached_property
    def fk(self):
        return YamFK(self.model_dir)

    def episode(self, task, destination):
        return convert_episode(Path(task['source']), destination, self.fk)


SOURCE = Abc130k
