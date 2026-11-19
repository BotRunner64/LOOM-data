"""Convert HiFi-UMI-2K local LeRobot v3 shards or an extracted episode subset."""
from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow.parquet as pq

from loom_data.geometry import eef_offset as validate_eef_offset
from loom_data.contract import episode_record, seconds_to_ns, write_frames
from loom_data.sync import table_grid
from loom_data.sources.common import task_instructions as instructions

from loom_data.sources.lerobot_v3 import LeRobotV3Reader, selected_cameras, subset_tasks, video_features
from loom_data.video import extract_video_segment, transcode_video

from loom_data.sources.base import CONTRACT_DROPS, Source, task_limit
from loom_data.sources.policy import SourceDataError

# Body-fixed virtual fingertip -> nominal rear/pivot datum, from the vendor's
# approximate 12 cm finger length. Tracking paper + a real opening-only sample
# support a rigid raw frame; see docs/sources/hifi_umi.md for evidence and remaining limits.
DEFAULT_EEF_OFFSET = (-0.12, 0., 0.)
# Dataset convention: the stored angle is the single-finger angle, not doubled.
DEFAULT_GRIPPER_OPEN_RAD = float(np.deg2rad(35.))
CAMERAS = {'observation.images.head_main': 'top',
           'observation.images.left_hand_up': 'left_wrist',
           'observation.images.right_hand_up': 'right_wrist'}
COLUMNS = ('observation.state', 'observation.state_valid', 'timestamp',
           'frame_index', 'episode_index', 'index', 'task_index', 'valid.frame')


def validate_source(info, modality):
    """Refuse other embodiments/layouts instead of guessing their semantics."""
    if info.get('robot_type') != 'simpleai_umi_4.0' or info.get('codebase_version') != 'v3.0':
        raise ValueError('Expected HiFi-UMI simpleai_umi_4.0 LeRobot v3.0')
    layout = info.get('state_layout', {})
    expected = {'feature': 'observation.state', 'order': ['right', 'left'],
                'rotation_6d_layout': 'first_two_rows', 'gripper_unit': 'rad',
                'per_side': ['x', 'y', 'z', *(f'rot6d_{i}' for i in range(6)),
                             'gripper_angle_rad']}
    if any(layout.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported HiFi-UMI state layout')
    features = info.get('features', {})
    if features.get('observation.state', {}).get('shape') != [20]:
        raise ValueError('Expected a 20-dimensional observation.state')
    for side, offset in (('right', 0), ('left', 10)):
        for kind, start, end in (('pose', offset, offset + 9),
                                 ('gripper', offset + 9, offset + 10)):
            field = modality.get('state', {}).get(f'{side}_eef_{kind}', {})
            expected = {'start': start, 'end': end, 'absolute': True,
                        'original_key': 'observation.state'}
            expected.update({'rotation_type': 'rotation_6d',
                             'rotation_6d_layout': 'first_two_rows'} if kind == 'pose'
                            else {'unit': 'rad'})
            if any(field.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Unsupported modality for {side} {kind}')
    return video_features(CAMERAS, features)


def mounting_offsets(left_eef_offset, right_eef_offset):
    return {side: validate_eef_offset(DEFAULT_EEF_OFFSET if value is None else value,
                                     f'{side} EEF offset').tolist()
            for side, value in (('left', left_eef_offset), ('right', right_eef_offset))}


def poses_from_rows(values, eef_offset=DEFAULT_EEF_OFFSET):
    """Move the body-fixed virtual tip to the mounting datum; never use aperture."""
    offset = validate_eef_offset(eef_offset)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError('Expected finite N x 9 source poses')
    first, second = values[:, 3:6].copy(), values[:, 6:9].copy()
    if not (np.allclose(np.sum(first * first, axis=1), 1, atol=1e-5, rtol=0)
            and np.allclose(np.sum(second * second, axis=1), 1, atol=1e-5, rtol=0)
            and np.allclose(np.sum(first * second, axis=1), 0, atol=1e-5, rtol=0)):
        raise ValueError('Invalid source rotation6D rows')
    # Remove float32 roundoff only, after rejecting malformed rotations.
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    rotation = np.stack((first, second, np.cross(first, second)), axis=1)
    position = values[:, :3] + np.einsum('nij,j->ni', rotation, offset)
    return np.concatenate((position, rotation.transpose(0, 2, 1).reshape(-1, 9)), axis=1)


def validate_gripper_range(closed_rad, open_rad):
    if not np.isfinite([closed_rad, open_rad]).all() or open_rad <= closed_rad:
        raise ValueError('Gripper range requires finite open_rad > closed_rad')


def openness(angles, closed_rad=0., open_rad=DEFAULT_GRIPPER_OPEN_RAD):
    validate_gripper_range(closed_rad, open_rad)
    angles = np.asarray(angles, dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError('Nonfinite gripper angle')
    return np.clip((angles - closed_rad) / (open_rad - closed_rad), 0, 1)


def declared_fps(info):
    """The source's nominal frame rate, when its metadata states one."""
    fps = info.get('fps')
    return fps if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0 else None


def load_tasks(path):
    tasks = {}
    for row in pq.read_table(path, columns=['task_index', 'task']).to_pylist():
        index, text = row['task_index'], row['task']
        if (type(index) is not int or index in tasks
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid/duplicate task_index or empty task text')
        tasks[index] = text
    return tasks


def read_episode(source, tasks, closed_rad, open_rad, eef_offsets=None):
    metadata = json.loads((source / 'episode.json').read_text(encoding='utf-8'))
    rows = pq.read_table(source / 'data.parquet', columns=list(COLUMNS)).to_pydict()
    index = metadata.get('episode_index')
    if type(index) is not int or source.name != f'episode_{index:06d}':
        raise ValueError(f'{source}: inconsistent episode identity')
    return read_rows(source, metadata, rows, tasks, closed_rad, open_rad, eef_offsets)


def read_rows(source, metadata, rows, tasks, closed_rad, open_rad, eef_offsets=None):
    offsets = mounting_offsets((eef_offsets or {}).get('left'), (eef_offsets or {}).get('right'))
    count = len(rows['timestamp'])
    index = metadata.get('episode_index')
    start, stop = metadata.get('dataset_from_index'), metadata.get('dataset_to_index')
    if (type(index) is not int or index < 0
            or type(start) is not int or type(stop) is not int or start < 0
            or count == 0 or metadata.get('length') != count or stop - start != count
            or any(i != index for i in rows['episode_index'])):
        raise ValueError(f'{source}: inconsistent episode identity/length')
    if (rows['frame_index'] != list(range(count))
            or rows['index'] != list(range(start, stop))):
        raise ValueError(f'{source}: noncontiguous frame/index; cannot map video frames')
    if any(t is None for t in rows['timestamp']):
        raise ValueError(f'{source}: null timestamp')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if times[0] < 0 or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{source}: negative or decreasing timestamps')
    if any(value is not True for value in rows['valid.frame']):
        raise ValueError(f'{source}: valid.frame contains invalid samples')
    valid = rows['observation.state_valid']
    if any(row is None or len(row) != 20 or any(v is not True for v in row) for row in valid):
        raise ValueError(f'{source}: observation.state_valid contains invalid samples')
    state = np.asarray(rows['observation.state'], dtype=np.float64)
    if state.shape != (count, 20) or not np.isfinite(state).all():
        raise ValueError(f'{source}: expected finite N x 20 observation.state')
    annotation = instructions(rows['task_index'], tasks, times)
    if set(metadata.get('tasks', [])) != {tasks[i] for i in rows['task_index']}:
        raise ValueError(f'{source}: episode tasks disagree with task_index')
    cameras = selected_cameras(CAMERAS, metadata.get('videos'))
    streams = {}
    for side, offset in (('right', 0), ('left', 10)):
        streams[f'{side}_eef'] = poses_from_rows(state[:, offset:offset + 9], offsets[side])
        streams[f'{side}_gripper'] = openness(state[:, offset + 9], closed_rad, open_rad)
    return times, streams, annotation, cameras


def convert_episode(source, destination, tasks, closed_rad, open_rad, eef_offsets=None, fps=None):
    times, streams, annotation, cameras = read_episode(source, tasks, closed_rad, open_rad, eef_offsets)
    return write_episode(source.name, destination, times, streams, annotation, cameras,
                         lambda key, target, grid: transcode_video(
                             source / 'videos' / f'{key}.mp4', target, times, grid), fps)


def write_episode(episode_id, destination, times, streams, annotation, cameras, video, fps=None):
    # The source supplies one shared timestamp per recorded row, so state and images
    # are already aligned. Rows pass through at a nominal 30 fps; otherwise they are resampled.
    grid = table_grid(times, fps)
    length = write_frames(destination, {name: (times, values) for name, values in streams.items()},
                          annotation, times[0], grid)
    videos = {}
    for key, camera in cameras.items():
        print(f'  {camera}', flush=True)
        videos[camera] = video(key, destination / f'{camera}.mp4', grid)
    return episode_record(episode_id, length, videos)


def discover_shards(source):
    if (source / 'meta/info.json').is_file():
        shards = [source]
    else:
        shards = sorted(p for p in source.glob('chunk-*/part-*') if p.is_dir())
    if not shards:
        raise ValueError(f'{source}: expected chunk-XXXX/part-YYYY shards or source/ and episodes/ subset')
    for shard in shards:
        if not (re.fullmatch(r'chunk-\d+', shard.parent.name)
                and re.fullmatch(r'part-\d+', shard.name)):
            raise ValueError(f'Invalid HiFi-UMI shard name: {shard}')
    return shards


def shard_episode_id(shard, index):
    return f'{shard.parent.name}_{shard.name}_episode_{index:06d}'.replace('-', '_')


class ShardReader(LeRobotV3Reader):
    def __init__(self, source):
        super().__init__(source, COLUMNS,
                         lambda info: validate_source(info, json.loads(
                             (source / 'meta/modality.json').read_text(encoding='utf-8'))))
        self.tasks = load_tasks(source / 'meta/tasks.parquet')

    def convert_episode(self, index, destination, closed_rad, open_rad, eef_offsets=None):
        row = self.episodes[index]
        rows = self.read_rows(index)
        times, streams, annotation, cameras = read_rows(self.source, row, rows, self.tasks, closed_rad, open_rad, eef_offsets)
        episode_id = shard_episode_id(self.source, index)
        return write_episode(episode_id, destination, times, streams, annotation, cameras,
                             lambda key, target, grid: extract_video_segment(
                                 row['videos'][key], target, times, self.fps, grid), self.fps)


class HifiUmi(Source):
    name = 'hifi_umi'
    camera_sets = (('left_wrist', 'right_wrist', 'top'),)
    OPTIONS = {
        'left_eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                                help='left mounting origin in the source virtual fingertip axes, metres (default -0.12 0 0)'),
        'right_eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                                 help='right mounting origin in the source virtual fingertip axes, metres (default -0.12 0 0)'),
        'gripper_open_rad': dict(type=float, help='fully open single-finger angle (default 35 degrees)'),
        'gripper_closed_rad': dict(type=float, help='closed angle (default 0)'),
    }

    @classmethod
    def make_params(cls, options):
        opened = DEFAULT_GRIPPER_OPEN_RAD if options['gripper_open_rad'] is None else options['gripper_open_rad']
        closed = 0. if options['gripper_closed_rad'] is None else options['gripper_closed_rad']
        validate_gripper_range(closed, opened)
        return {'gripper_open_rad': opened, 'gripper_closed_rad': closed,
                'eef_offsets': mounting_offsets(options['left_eef_offset'], options['right_eef_offset'])}

    @classmethod
    def discover(cls, config, progress):
        source, limit = Path(config['source']), task_limit(config)
        if (source / 'source').is_dir():  # episode subset written by the downloader
            meta = source / 'source'
            validate_source(json.loads((meta / 'info.json').read_text(encoding='utf-8')),
                            json.loads((meta / 'modality.json').read_text(encoding='utf-8')))
            load_tasks(meta / 'tasks.parquet')
            shared = [meta / name for name in ('info.json', 'tasks.parquet', 'modality.json')]
            return subset_tasks(source, CAMERAS, limit, progress), shared
        tasks, shared = [], []
        shards = discover_shards(source)
        with progress('Indexing HiFi-UMI shards', len(shards), unit='shard') as bar:
            for shard in shards:
                reader = ShardReader(shard)
                indices = sorted(reader.episodes)[:limit]
                if indices:
                    shared.extend(shard / 'meta' / name for name in ('info.json', 'modality.json', 'tasks.parquet'))
                    tasks.append({'source': str(shard), 'layout': 'lerobot_v3_shard',
                                  'episode_ids': [shard_episode_id(shard, i) for i in indices],
                                  'inputs': [str(p) for p in reader.inputs(indices)]})
                bar.update()
                if limit is not None:
                    limit -= len(indices)
                    if limit == 0:
                        break
        return tasks, shared

    @cached_property
    def tasks(self):
        return load_tasks(self.root / 'source/tasks.parquet')

    @cached_property
    def fps(self):
        return declared_fps(json.loads((self.root / 'source/info.json').read_text(encoding='utf-8')))

    def episode(self, task, destination):
        params = self.params
        return convert_episode(Path(task['source']), destination, self.tasks,
                               params['gripper_closed_rad'], params['gripper_open_rad'], params['eef_offsets'],
                               self.fps)

    def process(self, task, pending, staging, store, failed):
        if task.get('layout') != 'lerobot_v3_shard':
            return super().process(task, pending, staging, store, failed)
        # One task is a whole shard: its frame table is loaded once for all episodes.
        source, params = Path(task['source']), self.params
        reader = ShardReader(source)
        indices = {shard_episode_id(source, i): i for i in reader.episodes}
        for episode_id in pending:
            with tempfile.TemporaryDirectory(prefix='episode-', dir=staging) as temporary:
                destination = Path(temporary) / episode_id
                try:
                    record = reader.convert_episode(indices[episode_id], destination, params['gripper_closed_rad'],
                                                    params['gripper_open_rad'], params['eef_offsets'])
                    self.commit(store, episode_id, destination, record)
                except SourceDataError as exc:
                    if exc.reason in CONTRACT_DROPS:
                        store.skip(episode_id, exc.reason, detail=str(exc))
                    else:
                        failed(episode_id, exc)
                except Exception as exc:
                    failed(episode_id, exc)


SOURCE = HifiUmi
