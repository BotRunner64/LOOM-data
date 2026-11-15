"""Convert a local MolmoAct2-BimanualYAM LeRobot v3 dataset or episode subset."""
from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from loom_data.video import copy_h264_video
from loom_data.contract import seconds_to_ns, write_state, write_video_index
from loom_data.sources.common import task_instructions
from loom_data.robot.yam import YamFK
from loom_data.sources.lerobot_v3 import LeRobotV3Reader, subset_tasks
from loom_data.video import extract_video_segment

from loom_data.sources.base import Source, task_limit

CAMERAS = {'observation.images.top': 'top',
           'observation.images.left': 'left_wrist',
           'observation.images.right': 'right_wrist'}
STATE_NAMES = [name for side in ('left', 'right') for name in
               [*(f'{side}_joint_{i}.pos' for i in range(6)), f'{side}_gripper.pos']]
COLUMNS = ('observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index')


def validate_source(info):
    if info.get('robot_type') != 'bi_yam_follower' or info.get('codebase_version') != 'v3.0':
        raise ValueError('Expected MolmoAct2 bi_yam_follower LeRobot v3.0')
    features = info.get('features', {})
    state = features.get('observation.state', {})
    if state.get('shape') != [14] or state.get('names') != STATE_NAMES:
        raise ValueError('Unsupported MolmoAct2 observation.state layout')
    if not all(features.get(key, {}).get('dtype') == 'video' for key in CAMERAS):
        raise ValueError('Expected all three MolmoAct2 video features')


def load_tasks(directory):
    tasks, annotated = {}, {}
    for row in pq.read_table(directory / 'tasks.parquet',
                             columns=['task_index', '__index_level_0__']).to_pylist():
        index, text = row['task_index'], row['__index_level_0__']
        if (type(index) is not int or index < 0 or index in tasks
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid/duplicate task_index or empty task text')
        tasks[index] = text
    path = directory / 'tasks_annotated.parquet'
    if path.exists():
        seen = set()
        for row in pq.read_table(path, columns=['episode_index', 'task']).to_pylist():
            index, text = row['episode_index'], row['task']
            if type(index) is not int or index < 0 or index in seen:
                raise ValueError('Invalid/duplicate annotation episode_index')
            seen.add(index)
            # The dataset card explicitly permits fallback for invalid annotations.
            if isinstance(text, str) and text.strip():
                annotated[index] = text
    return tasks, annotated


def read_episode(source, tasks, annotated, fk):
    metadata = json.loads((source / 'episode.json').read_text(encoding='utf-8'))
    rows = pq.read_table(source / 'data.parquet', columns=list(COLUMNS)).to_pydict()
    index = metadata.get('episode_index')
    if type(index) is not int or source.name != f'episode_{index:06d}':
        raise ValueError(f'{source}: inconsistent episode identity')
    return read_rows(source, metadata, rows, tasks, annotated, fk)


def read_rows(source, metadata, rows, tasks, annotated, fk):
    count = len(rows['timestamp'])
    index = metadata.get('episode_index')
    start, stop = metadata.get('dataset_from_index'), metadata.get('dataset_to_index')
    if (type(index) is not int or index < 0
            or type(start) is not int or type(stop) is not int or start < 0
            or count == 0 or metadata.get('length') != count or stop - start != count
            or any(i != index for i in rows['episode_index'])):
        raise ValueError(f'{source}: inconsistent episode identity/length')
    if rows['frame_index'] != list(range(count)) or rows['index'] != list(range(start, stop)):
        raise ValueError(f'{source}: noncontiguous frame/index; cannot map video frames')
    if any(t is None for t in rows['timestamp']):
        raise ValueError(f'{source}: null timestamp')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if times[0] < 0 or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{source}: negative or decreasing timestamps')
    state = np.asarray(rows['observation.state'], dtype=np.float64)
    if state.shape != (count, 14) or not np.isfinite(state).all():
        raise ValueError(f'{source}: expected finite N x 14 observation.state')
    fallback = task_instructions(rows['task_index'], tasks, times)
    if set(metadata.get('tasks', [])) != {tasks[i] for i in rows['task_index']}:
        raise ValueError(f'{source}: episode tasks disagree with task_index')
    annotation = ([{'start_ns': 0, 'end_ns': times[-1] - times[0], 'text': annotated[index]}]
                  if index in annotated else fallback)
    if set(metadata.get('videos', {})) != set(CAMERAS):
        raise ValueError(f'{source}: expected all three camera views')
    streams = {}
    for side, offset in (('left', 0), ('right', 7)):
        streams[f'{side}_eef'] = [fk.pose(side, q) for q in state[:, offset:offset + 6]]
        streams[f'{side}_gripper'] = np.clip(state[:, offset + 6], 0, 1)
    return times, streams, annotation


def convert_episode(source, destination, tasks, annotated, fk):
    times, streams, annotation = read_episode(source, tasks, annotated, fk)
    return write_episode(source.name, destination, times, streams, annotation,
                         lambda key, target: copy_h264_video(
                             source / 'videos' / f'{key}.mp4', target, len(times)))


def write_episode(episode_id, destination, times, streams, annotation, video):
    # This source supplies one shared timestamp per recorded observation row.
    # The segmented videos preserve that row order; container PTS are irrelevant.
    for name, values in streams.items():
        write_state(destination / 'state' / f'{name}.parquet', times, values, times[0],
                    pose=name.endswith('_eef'))
    for key, camera in CAMERAS.items():
        print(f'  {camera}', flush=True)
        target = destination / 'rgb' / camera
        video(key, target.with_suffix('.mp4'))
        write_video_index(target.with_suffix('.parquet'), times, times[0])
    return {'episode_id': episode_id, 'cameras': sorted(CAMERAS.values()),
            'instructions': annotation}


class DatasetReader(LeRobotV3Reader):
    def __init__(self, source):
        # Validate decoded segment lengths per episode: a bad source range must
        # not prevent preparing or converting the rest of the full dataset.
        super().__init__(source, CAMERAS, COLUMNS, validate_source, check_video_length=False)
        self.tasks, self.annotated = load_tasks(source / 'meta')

    def convert_episode(self, index, destination, fk):
        row = self.episodes[index]
        for key, video in row['videos'].items():
            frames = (video['to_timestamp'] - video['from_timestamp']) * self.fps
            if abs(frames - row['length']) > .01:
                raise ValueError(f'{self.source}: episode {index} {key} video range implies '
                                 f'{frames:g} frames; expected {row["length"]}')
        rows = self.read_rows(index)
        times, streams, annotation = read_rows(self.source, row, rows, self.tasks, self.annotated, fk)
        return write_episode(f'episode_{index:06d}', destination, times, streams, annotation,
                             lambda key, target: extract_video_segment(
                                 row['videos'][key], target, len(times), self.fps))


class MolmoAct2(Source):
    name = 'molmoact2'
    model = 'yam'

    @classmethod
    def discover(cls, config, progress):
        source, limit = Path(config['source']), task_limit(config)
        if (source / 'source').is_dir():  # episode subset written by the downloader
            meta = source / 'source/meta'
            validate_source(json.loads((meta / 'info.json').read_text(encoding='utf-8')))
            load_tasks(meta)
            shared = [meta / name for name in ('info.json', 'tasks.parquet', 'tasks_annotated.parquet')]
            return subset_tasks(source, CAMERAS, limit, progress), shared
        reader = DatasetReader(source)
        shared = [source / 'meta' / name for name in ('info.json', 'tasks.parquet', 'tasks_annotated.parquet')]
        shared.extend(reader.metadata_files)
        indices = sorted(reader.episodes)[:limit]
        tasks = []
        with progress('Discovering MolmoAct2 episodes', len(indices), unit='episode') as bar:
            for index in indices:
                row = reader.episodes[index]
                inputs = {row['data_path'], *(video['path'] for video in row['videos'].values())}
                tasks.append({'source': str(source), 'layout': 'lerobot_v3', 'index': index,
                              'episode_ids': [f'episode_{index:06d}'], 'inputs': sorted(inputs)})
                bar.update()
        return tasks, shared

    @cached_property
    def fk(self):
        # Use each fixed arm base, without assuming the ABC dual-arm mounting is
        # MolmoAct2's calibration. The contract permits independent fixed references.
        return YamFK(self.model_dir, arm_local=True)

    @cached_property
    def reader(self):
        return DatasetReader(self.root)

    @cached_property
    def tasks(self):
        return load_tasks(self.root / 'source/meta')

    def episode(self, task, destination):
        if task.get('layout') == 'lerobot_v3':
            return self.reader.convert_episode(task['index'], destination, self.fk)
        tasks, annotated = self.tasks
        return convert_episode(Path(task['source']), destination, tasks, annotated, self.fk)


SOURCE = MolmoAct2
