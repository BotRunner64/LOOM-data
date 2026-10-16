#!/usr/bin/env python3
"""Convert AgiBot World Beta HDF5/MP4 episodes to the export contract."""
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import islice
from pathlib import Path

import numpy as np

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import (
    normalized_gripper, pose_matrix, rotation9d, transcode_video,
    tool_transform, write_state, write_video_index,
)
from scripts.convert.source_policy import SourceDataError, source_value, stationary_base

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = 'agibot-world/AgiBotWorld-Beta'
CAMERAS = {f'{name}_color.mp4': name for name in (
    'head', 'hand_left', 'hand_right', 'head_center_fisheye', 'head_left_fisheye',
    'head_right_fisheye', 'back_left_fisheye', 'back_right_fisheye',
)}
# Provisional long-finger TCP from official G1_120s closed-pad geometry.
DEFAULT_TCP_OFFSET = (0., 0., 0.207056)
# G1_120s installs its right gripper with a pi rotation about source flange Z.
DEFAULT_LEFT_TOOL_RPY = (math.pi, -math.pi / 2, 0.)
DEFAULT_RIGHT_TOOL_RPY = (math.pi, -math.pi / 2, math.pi)
# Measured values increase when the gripper closes, verified with wrist RGB
# and control transitions. These nominal endpoints are not tip-width mm.
DEFAULT_GRIPPER_OPEN = 35.
DEFAULT_GRIPPER_CLOSED = 120.


def load_annotations(source, task_id):
    path = source / f'task_info/task_{task_id}.json'
    try:
        rows = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise SourceDataError('missing_annotation', f'Missing task annotation: {path}') from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SourceDataError('invalid_annotation', f'{path}: invalid annotation JSON') from exc
    if not isinstance(rows, list):
        raise SourceDataError('invalid_annotation', f'{path}: expected a list of episodes')
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SourceDataError('invalid_annotation', f'{path}: expected episode objects')
        index = row.get('episode_id')
        if type(index) is not int or index < 0:
            raise SourceDataError('invalid_annotation', f'{path}: invalid episode ID')
        if index not in result:
            result[index] = {'annotation': row, 'repairs': []}
        elif result[index].get('annotation') == row:
            result[index]['repairs'] = ['deduplicated_annotation']
        else:
            # Conflicts reject this episode only; unrelated rows remain usable.
            result[index] = {'drop': {'reason': 'conflicting_annotation',
                                     'detail': f'{path}: conflicting episode {index}'}}
    return result


def episode_keys(source):
    """Sort one task at a time, without traversing every video/depth directory."""
    manifest_path = source / 'subset_manifest.json'
    if manifest_path.exists():
        from scripts.download.subset_common import load_manifest
        manifest = load_manifest(manifest_path, REPO_ID)
        episodes = sorted((e['task_id'], e['episode_id']) for e in manifest['episodes'])
        if len(set(episodes)) != len(episodes):
            raise ValueError('Duplicate Beta episode IDs in manifest')
        yield from episodes
        return
    state_root = source / 'proprio_stats'
    if not state_root.is_dir():
        raise ValueError(f'Missing Beta state root: {state_root}')
    with os.scandir(state_root) as entries:
        tasks = sorted((e.name for e in entries if e.is_dir()), key=lambda n: (len(n), n))
    for task in tasks:
        if not task.isdecimal():
            raise ValueError(f'Invalid Beta task directory: {task}')
        with os.scandir(state_root / task) as entries:
            episodes = sorted((e.name for e in entries if e.is_dir()), key=lambda n: (len(n), n))
        for episode in episodes:
            if not episode.isdecimal():
                raise ValueError(f'Invalid Beta episode directory: {task}/{episode}')
            yield int(task), int(episode)


def iter_discover(source):
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f'Expected a Beta data root: {source}')
    previous_task, annotations, annotation_error = None, {}, None
    for task, episode in episode_keys(source):
        if type(task) is not int or type(episode) is not int or min(task, episode) < 0:
            raise ValueError('Invalid Beta task or episode ID')
        if task != previous_task:
            previous_task, annotation_error = task, None
            try:
                annotations = load_annotations(source, task)
            except SourceDataError as exc:
                annotations, annotation_error = {}, exc
        state = source / f'proprio_stats/{task}/{episode}/proprio_stats.h5'
        video_dir = source / f'observations/{task}/{episode}/videos'
        reasons, details = [], []
        if not state.is_file():
            reasons.append('missing_state'); details.append(f'Missing state: {state}')
        paths = sorted(video_dir.glob('*.mp4'))
        if not paths:
            reasons.append('missing_rgb'); details.append(f'Missing RGB: {video_dir}')
        elif any(p.name not in CAMERAS for p in paths):
            reasons.append('unknown_camera'); details.append(f'Unknown camera filename: {video_dir}')
        annotation = annotations.get(episode)
        if annotation_error is not None:
            reasons.append(annotation_error.reason); details.append(str(annotation_error))
        elif annotation is None:
            reasons.append('missing_annotation'); details.append(f'Missing episode annotation: {task}/{episode}')
        elif 'drop' in annotation:
            reasons.append(annotation['drop']['reason']); details.append(annotation['drop']['detail'])
        item = {'state': state, 'videos': {CAMERAS[p.name]: p for p in paths if p.name in CAMERAS},
                'annotation_path': source / f'task_info/task_{task}.json', 'episode_index': episode}
        if reasons:
            item['drop'] = {'reason': reasons[0], 'reasons': reasons, 'detail': '; '.join(details)}
        else:
            item.update(annotation)
        yield f'task_{task}_episode_{episode}', item


def discover(source, limit=None):
    items = list(islice(iter_discover(source), limit))
    if not items:
        raise ValueError('No Beta episodes found')
    return items


def read_state(path, *, with_repairs=False):
    import h5py
    try:
        with h5py.File(path, 'r') as handle:
            times = np.asarray(handle['timestamp'])
            fields = {key: np.asarray(handle[f'state/{key}']) for key in (
                'end/position', 'end/orientation', 'effector/position', 'robot/position', 'robot/orientation')}
            commands = np.asarray(handle['action/robot/velocity']) if 'action/robot/velocity' in handle else None
    except KeyError as exc:
        raise SourceDataError('missing_state_field', f'{path}: {exc}') from exc
    except FileNotFoundError as exc:
        raise SourceDataError('missing_state', f'Missing state: {path}') from exc
    except OSError as exc:
        corruption = ('file signature not found', 'truncated file', 'bad object header',
                      'incorrect metadata checksum', 'bad symbol table')
        if exc.errno is None and any(marker in str(exc).lower() for marker in corruption):
            raise SourceDataError('invalid_hdf5', f'{path}: {exc}') from exc
        raise  # HDF5 can report storage failures without setting Python errno.
    if (times.ndim != 1 or times.dtype != np.dtype('int64') or len(times) < 2
            or times[0] < 0 or np.any(times[1:] < times[:-1]) or times[-1] <= times[0]):
        raise SourceDataError('invalid_timestamp', f'{path}: expected nondecreasing int64 nanosecond acquisition times')
    shapes = {'end/position': (len(times), 2, 3), 'end/orientation': (len(times), 2, 4),
              'effector/position': (len(times), 2), 'robot/position': (len(times), 3),
              'robot/orientation': (len(times), 4)}
    for key, shape in shapes.items():
        if (fields[key].shape != shape or fields[key].dtype.kind not in 'fiub'
                or not np.isfinite(fields[key]).all()):
            reason = 'unsupported_effector' if key == 'effector/position' and fields[key].shape == (len(times), 12) else 'invalid_state'
            raise SourceDataError(reason, f'{path}: invalid {key}; expected finite {shape}, got {fields[key].shape}')
    fields['robot/position'], fields['robot/orientation'], repairs = stationary_base(
        fields['robot/position'], fields['robot/orientation'], commands)
    if with_repairs:
        return times.tolist(), fields, repairs
    return times.tolist(), fields


def poses(fields, transforms):
    result = {'left_eef': [], 'right_eef': []}
    for i in range(len(fields['end/position'])):
        world = source_value('invalid_base_pose', pose_matrix, fields['robot/position'][i], fields['robot/orientation'][i])
        for k, side in enumerate(('left', 'right')):
            flange = source_value('invalid_flange_pose', pose_matrix, fields['end/position'][i, k], fields['end/orientation'][i, k])
            # Recorded end poses already include the two-DoF waist, validated
            # against independent G1 FK. Adding torso FK again is incorrect.
            result[f'{side}_eef'].append(rotation9d(world @ flange @ transforms[k]))
    return result


def instructions(annotation, times):
    task = annotation.get('task_name', '')
    if not isinstance(task, str) or not task.strip():
        raise SourceDataError('invalid_instruction', 'Missing episode task_name')
    origin, end = times[0], times[-1]
    if end <= origin:
        raise ValueError('Zero-duration instruction clock')
    labels = annotation.get('label_info', annotation.get('lable_info', {}))
    if not isinstance(labels, dict):
        raise SourceDataError('invalid_instruction', 'Expected label_info object')
    segments = labels.get('action_config', [])
    if not isinstance(segments, list):
        raise SourceDataError('invalid_instruction', 'Expected action_config list')
    selected = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise SourceDataError('invalid_instruction', 'Expected instruction segment object')
        text = segment.get('action_text', '')
        if not isinstance(text, str):
            raise SourceDataError('invalid_instruction', 'Instruction text must be a string')
        if not text.strip():
            continue
        start, stop = segment.get('start_frame'), segment.get('end_frame')
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(times):
            raise SourceDataError('invalid_instruction', 'Instruction frame interval is out of bounds')
        selected.append((start, stop, text.strip()))
    output, cursor, previous_stop = [], 0, 0
    for start, stop, text in sorted(selected):
        if start < previous_stop:
            raise SourceDataError('invalid_instruction', 'Overlapping action_config instruction intervals')
        begin_ns = int(times[start]) - origin
        stop_ns = int(times[stop] if stop < len(times) else times[-1]) - origin
        if begin_ns > cursor:
            output.append({'start_ns': cursor, 'end_ns': begin_ns, 'text': task.strip()})
        if stop_ns > begin_ns:
            output.append({'start_ns': begin_ns, 'end_ns': stop_ns, 'text': text})
        cursor, previous_stop = stop_ns, stop
    if cursor < end-origin:
        output.append({'start_ns': cursor, 'end_ns': end-origin, 'text': task.strip()})
    return output


def convert_episode(source, destination, transforms=None,
                    gripper_closed=DEFAULT_GRIPPER_CLOSED, gripper_open=DEFAULT_GRIPPER_OPEN):
    if 'drop' in source:
        drop = source['drop']
        raise SourceDataError(drop['reason'], drop['detail'], reasons=drop.get('reasons'))
    transforms = transforms if transforms is not None else (
        tool_transform(DEFAULT_TCP_OFFSET, DEFAULT_LEFT_TOOL_RPY),
        tool_transform(DEFAULT_TCP_OFFSET, DEFAULT_RIGHT_TOOL_RPY))
    normalized_gripper(0., gripper_closed, gripper_open)  # Validate configuration before source classification.
    times, fields, repairs = read_state(source['state'], with_repairs=True)
    values = poses(fields, transforms)
    for k, side in enumerate(('left', 'right')):
        values[f'{side}_gripper'] = [source_value('invalid_gripper', normalized_gripper, v, gripper_closed, gripper_open)
                                   for v in fields['effector/position'][:, k]]
    record = {'episode_id': destination.name, 'cameras': sorted(source['videos']),
              'instructions': instructions(source['annotation'], times),
              '_conversion': {'repairs': source.get('repairs', []) + repairs}}
    for name, samples in values.items():
        write_state(destination / 'state' / f'{name}.parquet', times, samples, times[0], pose=name.endswith('_eef'))
    (destination / 'rgb').mkdir()
    for camera, path in source['videos'].items():
        transcode_video(path, destination / 'rgb' / f'{camera}.mp4', len(times))
        # Upstream row/video mapping is by display-frame index. Use acquisition
        # timestamp, never MP4 PTS or frame_index / 30.
        write_video_index(destination / 'rgb' / f'{camera}.parquet', times, times[0])
    return record


def convert(source, output, limit=None, tcp_offset=DEFAULT_TCP_OFFSET,
            left_tool_rpy=DEFAULT_LEFT_TOOL_RPY, right_tool_rpy=DEFAULT_RIGHT_TOOL_RPY,
            gripper_closed=DEFAULT_GRIPPER_CLOSED, gripper_open=DEFAULT_GRIPPER_OPEN):
    from scripts.convert.parallel import run_native
    from scripts.convert.parallel_adapters import make_config
    config = make_config('agibotworld_beta', source, limit=limit, tcp_offset=tcp_offset,
                         left_tool_rpy=left_tool_rpy, right_tool_rpy=right_tool_rpy,
                         gripper_closed=gripper_closed, gripper_open=gripper_open)
    run_native(config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/raw/AgiBotWorld-Beta')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/processed/AgiBotWorld-Beta')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--tcp-offset', nargs=3, type=float, default=DEFAULT_TCP_OFFSET, help='TCP in source flange axes, metres; nominal long-finger geometry by default')
    parser.add_argument('--left-tool-rpy', nargs=3, type=float, default=DEFAULT_LEFT_TOOL_RPY)
    parser.add_argument('--right-tool-rpy', nargs=3, type=float, default=DEFAULT_RIGHT_TOOL_RPY)
    parser.add_argument('--gripper-open', type=float, default=DEFAULT_GRIPPER_OPEN, help='measured source value at full open; provisional default 35')
    parser.add_argument('--gripper-closed', type=float, default=DEFAULT_GRIPPER_CLOSED, help='measured source value at full close; provisional default 120')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        convert(args.input_dir, args.output_dir, args.limit, args.tcp_offset,
                args.left_tool_rpy, args.right_tool_rpy, args.gripper_closed, args.gripper_open)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
