#!/usr/bin/env python3
"""Convert ABC-130K MCAP episodes to the export contract using YAM joint FK."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import tempfile

import numpy as np

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import VideoWriter, write_state, write_video_index
from scripts.robot.yam import YamFK

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = {f'/{name}-camera': name.replace('-', '_') for name in
           ('top', 'top-left', 'top-right', 'left-wrist', 'right-wrist')}
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
    videos, tasks = {}, []
    (destination / 'rgb').mkdir(parents=True)
    # Validate all state before expensive video transcoding. The second MCAP
    # pass only decodes camera topics; state samples retain their native times.
    for topic, timestamp, decoded in messages(source / 'episode.mcap', list(STATES) + ['/instruction']):
        if topic == '/instruction':
            tasks.append(decoded.data)
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
    with ExitStack() as stack:
        for topic, timestamp, decoded in messages(source / 'episode.mcap', list(CAMERAS)):
            camera = CAMERAS[topic]
            if camera not in videos:
                videos[camera] = VideoWriter(destination / 'rgb' / f'{camera}.mp4', decoded.format)
                stack.callback(videos[camera].close)
            videos[camera].add(decoded.data, timestamp, decoded.format)
        if not videos:
            raise ValueError(f'{source}: requires at least one camera')
        for writer in videos.values():
            writer.finish()
    streams = [times for times, _ in state.values()] + [video.timestamps for video in videos.values()]
    origin = min(times[0] for times in streams)
    end = max(times[-1] for times in streams)
    annotations = []
    if (source / 'annotation.mcap').exists():
        annotations = [(timestamp, decoded.data) for _, timestamp, decoded in
                       messages(source / 'annotation.mcap', ['/subtask-annotation'])]
    for name, (times, values) in state.items():
        write_state(destination / 'state' / f'{name}.parquet', times, values, origin,
                    pose=name.endswith('_eef'))
    for camera, writer in videos.items():
        write_video_index(destination / 'rgb' / f'{camera}.parquet', writer.timestamps, origin)
    return {'episode_id': source.name, 'cameras': sorted(videos),
            'instructions': instructions(tasks[0], annotations, origin, end)}


def convert(source, output, model_dir, limit=None):
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    files = sorted(source.rglob('episode.mcap'))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f'No episode.mcap found under {source}')
    names = [path.parent.name for path in files]
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', n) for n in names):
        raise ValueError('Episode directory names must be unique safe identifiers')
    fk = YamFK(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the dataset only when all episodes succeed; no partial manifest or
    # orphan episode is exposed at the requested output path on failure.
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as metadata:
            for index, path in enumerate(files, 1):
                print(f'[{index}/{len(files)}] {path.parent.name}', flush=True)
                record = convert_episode(path.parent, staged / 'episodes' / path.parent.name, fk)
                metadata.write(json.dumps(record, ensure_ascii=False) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {len(files)} episodes to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/raw/ABC-130K')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/export/ABC-130K')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'assets/robot_models/yam')
    parser.add_argument('--limit', type=int, help='convert only the first N sorted episodes')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        convert(args.input_dir, args.output_dir, args.model_dir, args.limit)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
