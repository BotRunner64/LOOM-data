#!/usr/bin/env python3
"""Convert extracted Galaxea R1 Lite / R1 Pro LeRobot v2.1 tasks to the export contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow.parquet as pq

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import (
    pose_matrix, seconds_to_ns, transcode_video, write_state, write_video_index,
)
from scripts.robot.kinematics import fk_poses
from scripts.robot.urdf_model import Robot, parse_urdf

ROOT = Path(__file__).resolve().parents[2]
SIDES = ('left', 'right')
CAMERAS = {f'observation.images.{name}_rgb': name for name in (
    'head', 'head_right', 'left_wrist', 'right_wrist',
)}
# Common physical TCP: midpoint of the G1 finger tips (geometric estimate).
# G1 front face -> finger origin 0.03689 m -> tip 0.041465 m.
TIP_DISTANCE = 0.03689 + 0.041465
DEFAULT_TCP_OFFSETS = {'r1lite': (0.08165 + TIP_DISTANCE, 0., 0.),
                       'r1pro': (0., 0., -TIP_DISTANCE)}
# Columns are contract X (approach), Y, Z (back/camera side) in native ee_pose.
TOOL_AXES = {'r1lite': np.eye(3),
             'r1pro': np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])}
COLUMNS = ['timestamp', 'frame_index', 'episode_index', 'task_index', 'coarse_task_index',
           'observation.state.torso', 'action.chassis.velocities'] + [
    f'observation.state.{side}_{kind}' for side in SIDES
    for kind in ('arm', 'ee_pose', 'gripper')
]
EMPTY_LABELS = {'', 'null', 'none', 'nan'}
QUALITY_LABELS = {'qualified', 'unqualified', '合格', '不合格'}


def validate_source(info):
    robot_type = info.get('robot_type')
    if info.get('codebase_version') != 'v2.1' or robot_type not in DEFAULT_TCP_OFFSETS:
        raise ValueError('Expected Galaxea LeRobot v2.1 with robot_type r1lite or r1pro')
    dof = 6 if robot_type == 'r1lite' else 7
    features = info['features']
    expected = {'observation.state.torso': [4]}
    command_shape = features.get('action.chassis.velocities', {}).get('shape')
    if command_shape not in ([3], [6]):
        raise ValueError('Expected 3D planar or 6D twist chassis command dimensions')
    for side in SIDES:
        expected.update({f'observation.state.{side}_arm': [dof],
                         f'observation.state.{side}_ee_pose': [7],
                         f'observation.state.{side}_gripper': [1]})
        pose_names = [f'/motion_control/pose_ee_arm_{side}.pose.{part}' for part in (
            'position.x', 'position.y', 'position.z',
            'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w')]
        if features.get(f'observation.state.{side}_ee_pose', {}).get('names') != pose_names:
            raise ValueError(f'Unexpected {side} ee_pose component order')
    for key, shape in expected.items():
        if features.get(key, {}).get('shape') != shape:
            raise ValueError(f'{robot_type}: unexpected dimensions for {key}, expected {shape}')
    cameras = {}
    for key, feature in features.items():
        if not key.startswith('observation.images.'):
            continue
        video_info = feature.get('info', feature.get('video_info', {}))
        if video_info.get('video.is_depth_map', False):
            continue
        if key not in CAMERAS or feature.get('dtype') != 'video':
            raise ValueError(f'Unsupported RGB camera feature: {key}')
        cameras[key] = CAMERAS[key]
    if not cameras:
        raise ValueError('No RGB cameras in source metadata')
    return robot_type, cameras


def finite_array(values, shape, label):
    try:
        data = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Expected finite {label} with shape {shape}') from exc
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f'Expected finite {label} with shape {shape}')
    return data


def openness(values, count):
    data = np.asarray(values, dtype=np.float64)
    # Actual Parquet uses scalars; accept the declared singleton vector too.
    if data.shape == (count, 1):
        data = data[:, 0]
    data = finite_array(data, (count,), 'measured gripper stroke')
    return np.clip(data / 100., 0., 1.)


class GalaxeaKinematics:
    """Recorded ee_pose + measured torso FK, in a stationary chassis frame."""

    def __init__(self, model_dir, robot_type, tcp_offset=None):
        if robot_type not in DEFAULT_TCP_OFFSETS:
            raise ValueError(f'Unsupported robot_type: {robot_type}')
        self.robot_type = robot_type
        manifest = json.loads((model_dir / 'robot.json').read_text())
        if manifest['robot'] != f'galaxea_{robot_type}':
            raise ValueError(f'{robot_type}: wrong model package')
        model = parse_urdf(model_dir / manifest['urdf'])
        self.torso_count = 3 if robot_type == 'r1lite' else 4
        self.torso_link = f'torso_link{self.torso_count}'
        parents = {j.child: j for j in model.joints}
        chain, link = [], self.torso_link
        while link in parents:
            joint = parents[link]
            chain.append(joint)
            link = joint.parent
        if link != 'base_link' or {j.name for j in chain if j.type != 'fixed'} != {
                f'torso_joint{i}' for i in range(1, self.torso_count + 1)}:
            raise ValueError(f'{robot_type}: unexpected torso kinematic chain')
        self.torso = Robot(model.name, tuple([link] + [j.child for j in reversed(chain)]), tuple(chain))
        offset = DEFAULT_TCP_OFFSETS[robot_type] if tcp_offset is None else tcp_offset
        self.tcp = np.append(finite_array(offset, (3,), 'TCP offset in metres'), 1.)
        self.axes = TOOL_AXES[robot_type]

    def poses(self, torso_values, native_poses):
        torso_values = finite_array(torso_values, (4,), 'torso state')
        if self.robot_type == 'r1lite' and abs(torso_values[3]) > 1e-6:
            raise ValueError('R1 Lite fourth torso component must be padding zero')
        values = {f'torso_joint{i+1}': float(v) for i, v in enumerate(torso_values[:self.torso_count])}
        torso = fk_poses(self.torso, values)[self.torso_link]
        result = []
        for native in native_poses:
            native = finite_array(native, (7,), 'native ee_pose')
            transform = torso @ pose_matrix(native[:3], native[3:])
            rotation = transform[:3, :3] @ self.axes
            position = (transform @ self.tcp)[:3]
            result.append(np.concatenate((position, rotation.T.reshape(-1))))
        return result


def instruction_segments(times, fine_indices, coarse_indices, tasks):
    if not times or times[-1] <= times[0] or len(fine_indices) != len(times) or len(coarse_indices) != len(times):
        raise ValueError('Invalid instruction timeline or zero-duration episode')

    def label(index):
        if type(index) is not int or index not in tasks:
            raise ValueError(f'Unknown instruction task_index: {index}')
        value = tasks[index]
        if value is None:
            return ''
        if not isinstance(value, str):
            raise ValueError(f'Invalid task text: {index}')
        text = value.strip()
        if text.lower() in QUALITY_LABELS:
            raise ValueError(f'Quality label used as instruction: {index}')
        return '' if text.lower() in EMPTY_LABELS else text

    texts = []
    for fine, coarse in zip(fine_indices, coarse_indices):
        text = label(fine) or label(coarse)
        if not text:
            raise ValueError('Neither fine nor coarse task contains instruction text')
        texts.append(text)
    # A label applies until the next row's time. Duplicate timestamps have no
    # positive-duration interval; retain the last label at that instant.
    output = []
    for i in range(len(times) - 1):
        start, end = times[i] - times[0], times[i+1] - times[0]
        if end < start:
            raise ValueError('Decreasing instruction timestamps')
        if end == start:
            continue
        if output and output[-1]['text'] == texts[i] and output[-1]['end_ns'] == start:
            output[-1]['end_ns'] = end
        else:
            output.append({'start_ns': start, 'end_ns': end, 'text': texts[i]})
    return output


def read_jsonl(path, key):
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = record[key]
        if type(index) is not int or index < 0 or index in records:
            raise ValueError(f'{path}: invalid or duplicate {key}: {index}')
        records[index] = record
    if not records:
        raise ValueError(f'{path}: empty metadata')
    return records


def source_path(directory, template, index, chunks_size, video_key=None):
    relative = Path(template.format(episode_chunk=index // chunks_size,
                                    episode_index=index, video_key=video_key))
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f'Source path escapes task directory: {relative}')
    return path


def is_stationary(commands, count, dimensions=None):
    """Accept zero chassis commands throughout; assume a fixed base for export.

    This selection rule does not establish measured physical immobility.
    Invalid commands are skipped; wheel feedback is not part of the selection.
    """
    try:
        commands = np.asarray(commands, dtype=float)
        if dimensions is None:
            dimensions = commands.shape[1] if commands.ndim == 2 else None
        if dimensions not in (3, 6):
            return False
        commands = finite_array(commands, (count, dimensions), 'chassis commands')
    except (ValueError, TypeError):
        return False
    return count > 0 and bool(np.all(np.abs(commands) <= 1e-6))


def prepare_episode(path, index, metadata, tasks, kinematics, command_dimensions=6):
    rows = pq.read_table(path, columns=COLUMNS).to_pydict()
    count = len(rows['timestamp'])
    if count == 0 or metadata['length'] != count:
        raise ValueError(f'{path}: empty episode or metadata length mismatch')
    if rows['episode_index'] != [index] * count or rows['frame_index'] != list(range(count)):
        raise ValueError(f'{path}: incorrect episode_index or nonconsecutive frame_index')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if times[-1] <= times[0] or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{path}: unordered or zero-duration timestamps')
    if not is_stationary(rows['action.chassis.velocities'], count, command_dimensions):
        return None
    dof = 6 if kinematics.robot_type == 'r1lite' else 7
    values = {f'{side}_eef': [] for side in SIDES}
    for side in SIDES:
        finite_array(rows[f'observation.state.{side}_arm'], (count, dof), f'{side} arm state')
        values[f'{side}_gripper'] = openness(rows[f'observation.state.{side}_gripper'], count)
    for i, torso in enumerate(rows['observation.state.torso']):
        poses = kinematics.poses(torso, [rows[f'observation.state.{s}_ee_pose'][i] for s in SIDES])
        for side, pose in zip(SIDES, poses):
            values[f'{side}_eef'].append(pose)
    instructions = instruction_segments(times, rows['task_index'], rows['coarse_task_index'], tasks)
    return times, values, instructions


def convert_episode(directory, index, metadata, tasks, kinematics, info, destination):
    _, cameras = validate_source(info)
    chunks_size = info['chunks_size']
    path = source_path(directory, info['data_path'], index, chunks_size)
    prepared = prepare_episode(path, index, metadata, tasks, kinematics,
                               info['features']['action.chassis.velocities']['shape'][0])
    if prepared is None:
        return None
    times, values, instructions = prepared
    for name, samples in values.items():
        write_state(destination / 'state' / f'{name}.parquet', times, samples, times[0],
                    pose=name.endswith('_eef'))
    rgb = destination / 'rgb'
    rgb.mkdir()
    for key, camera in sorted(cameras.items()):
        video = source_path(directory, info['video_path'], index, chunks_size, key)
        transcode_video(video, rgb / f'{camera}.mp4', len(times))
        write_video_index(rgb / f'{camera}.parquet', times, times[0])
    return {'episode_id': destination.name, 'cameras': sorted(cameras.values()),
            'instructions': instructions}


def discover_tasks(source):
    if (source / 'meta/info.json').is_file():
        return [source]
    directories = sorted({p.parent.parent for p in source.rglob('meta/info.json')})
    if not directories:
        raise ValueError(f'No extracted LeRobot tasks in {source}; run galaxea_subset.py extract first')
    return directories


def convert(source, output, models_dir=ROOT / 'assets/robot_models', *,
            robot_type=None, limit=None, limit_episodes=None, tcp_offsets=None):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    for value in (limit, limit_episodes):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Limits must be positive integers')
    if robot_type is not None and robot_type not in DEFAULT_TCP_OFFSETS:
        raise ValueError(f'Unsupported robot_type filter: {robot_type}')
    selected = []
    for directory in discover_tasks(source):
        info = json.loads((directory / 'meta/info.json').read_text())
        if robot_type is None or info.get('robot_type') == robot_type:
            selected.append((directory, info))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError('No tasks match the selected robot_type')
    output.parent.mkdir(parents=True, exist_ok=True)
    count, skipped, seen, models = 0, 0, set(), {}
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        (staged / 'episodes').mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as manifest:
            for directory, info in selected:
                kind, cameras = validate_source(info)
                if kind not in models:
                    models[kind] = GalaxeaKinematics(models_dir / f'galaxea_{kind}', kind,
                                                    (tcp_offsets or {}).get(kind))
                prefix = directory.name
                if not re.fullmatch(r'[A-Za-z0-9_\-]+', prefix):
                    raise ValueError(f'Unsafe task directory name: {prefix}')
                metadata = read_jsonl(directory / 'meta/episodes.jsonl', 'episode_index')
                tasks = {i: r['task'] for i, r in read_jsonl(directory / 'meta/tasks.jsonl', 'task_index').items()}
                if len(metadata) != info['total_episodes']:
                    raise ValueError(f'{directory}: episode metadata count mismatch')
                chunks_size = info['chunks_size']
                if type(chunks_size) is not int or chunks_size < 1:
                    raise ValueError('Invalid chunks_size')
                for index, episode in sorted(metadata.items()):
                    episode_id = f'{prefix}_episode_{index:06d}'
                    if episode_id in seen:
                        raise ValueError(f'Duplicate episode ID: {episode_id}')
                    seen.add(episode_id)
                    record = convert_episode(directory, index, episode, tasks, models[kind], info,
                                             staged / 'episodes' / episode_id)
                    if record is None:
                        skipped += 1
                        print(f'Exported {count}; skipped {skipped}', flush=True)
                        continue
                    manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                    count += 1
                    print(f'Exported {count}; skipped {skipped}', flush=True)
                    if limit_episodes is not None and count >= limit_episodes:
                        break
                if limit_episodes is not None and count >= limit_episodes:
                    break
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {count}; skipped {skipped}. Output: {output}', flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/raw/Galaxea-Open-World-Dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/processed/Galaxea-Open-World-Dataset')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'assets/robot_models')
    parser.add_argument('--robot-type', choices=tuple(DEFAULT_TCP_OFFSETS), help='only convert this embodiment')
    parser.add_argument('--limit', type=int, help='first N sorted tasks after filtering')
    parser.add_argument('--limit-episodes', type=int, help='first N accepted stationary episodes across selected tasks')
    for kind in DEFAULT_TCP_OFFSETS:
        parser.add_argument(f'--{kind}-tcp-offset', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                            help=f'calibrated TCP offset in native {kind} ee_pose, metres; default is geometric estimate')
    args = parser.parse_args()
    try:
        convert(args.input_dir, args.output_dir, args.models_dir,
                robot_type=args.robot_type, limit=args.limit, limit_episodes=args.limit_episodes,
                tcp_offsets={kind: getattr(args, f'{kind}_tcp_offset') for kind in DEFAULT_TCP_OFFSETS})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
