"""Convert extracted Galaxea R1 Lite / R1 Pro LeRobot v2.1 tasks to the export contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from loom_data.geometry import pose_matrix
from loom_data.contract import seconds_to_ns, write_state, write_video_index
from loom_data.video import transcode_video
from loom_data.robot.kinematics import fk_poses
from loom_data.robot.urdf_model import Robot, parse_urdf

from loom_data.sources.base import Source, task_limit

SIDES = ('left', 'right')
CAMERAS = {'observation.images.head_rgb': 'top',
           'observation.images.left_wrist_rgb': 'left_wrist',
           'observation.images.right_wrist_rgb': 'right_wrist'}
# Mount datum is arm_link6 (Lite) / arm_link7 (Pro). The Pro native
# gripper_base_link is translated [-0.0295, 0, -0.16065] from its mount,
# with identical axes; invert that fixed translation to recover the mount.
DEFAULT_EEF_OFFSETS = {'r1lite': (0., 0., 0.),
                       'r1pro': (0.0295, 0., 0.16065)}
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
    if info.get('codebase_version') != 'v2.1' or robot_type not in DEFAULT_EEF_OFFSETS:
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
        if key not in CAMERAS:
            continue
        video_info = feature.get('info', feature.get('video_info', {}))
        if video_info.get('video.is_depth_map', False):
            continue
        if feature.get('dtype') != 'video':
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

    def __init__(self, model_dir, robot_type, eef_offset=None):
        if robot_type not in DEFAULT_EEF_OFFSETS:
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
        offset = DEFAULT_EEF_OFFSETS[robot_type] if eef_offset is None else eef_offset
        self.eef = np.append(finite_array(offset, (3,), 'EEF mounting offset in metres'), 1.)
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
            position = (transform @ self.eef)[:3]
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


class Galaxea(Source):
    name = 'galaxea'
    model = ''  # the parent of galaxea_r1lite/ and galaxea_r1pro/
    policy = 'skip'
    skip_reason = 'Galaxea chassis commands do not meet the stationary selection rule'
    OPTIONS = {
        'robot_type': dict(choices=tuple(DEFAULT_EEF_OFFSETS), help='only convert this embodiment'),
        'r1lite_eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                                  help='mounting origin in native r1lite ee_pose, metres (default: model mounting joint)'),
        'r1pro_eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                                 help='mounting origin in native r1pro ee_pose, metres (default: model mounting joint)'),
    }

    @classmethod
    def make_params(cls, options):
        robot_type = options['robot_type']
        if robot_type is not None and robot_type not in DEFAULT_EEF_OFFSETS:
            raise ValueError(f'Unsupported robot_type: {robot_type}')
        return {'robot_type': robot_type,
                'eef_offsets': {kind: list(options[f'{kind}_eef_offset'] if options[f'{kind}_eef_offset'] is not None
                                           else DEFAULT_EEF_OFFSETS[kind]) for kind in ('r1lite', 'r1pro')}}

    @classmethod
    def discover(cls, config, progress):
        limit, robot_type = task_limit(config), config['params']['robot_type']
        tasks, shared = [], []
        directories = discover_tasks(Path(config['source']))
        with progress('Discovering tasks', len(directories), unit='task') as bar:
            for directory in directories:
                if limit is not None and len(tasks) >= limit:
                    break
                bar.update()
                info = json.loads((directory / 'meta/info.json').read_text(encoding='utf-8'))
                if robot_type is not None and info.get('robot_type') != robot_type:
                    continue
                _, cameras = validate_source(info)
                metadata = read_jsonl(directory / 'meta/episodes.jsonl', 'episode_index')
                read_jsonl(directory / 'meta/tasks.jsonl', 'task_index')
                chunks = info['chunks_size']
                if len(metadata) != info['total_episodes'] or type(chunks) is not int or chunks < 1:
                    raise ValueError(f'{directory}: invalid episode metadata/chunks_size')
                shared.extend(directory / 'meta' / name for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'))
                remaining = None if limit is None else limit - len(tasks)
                for index in sorted(metadata)[:remaining]:
                    paths = [source_path(directory, info['data_path'], index, chunks)]
                    paths += [source_path(directory, info['video_path'], index, chunks, key) for key in cameras]
                    tasks.append({'source': str(directory), 'index': index,
                                  'episode_ids': [f'{directory.name}_episode_{index:06d}'],
                                  'inputs': [str(p) for p in paths]})
        return tasks, shared

    def __init__(self, config):
        super().__init__(config)
        self.directories, self.kinematics = {}, {}

    def episode(self, task, destination):
        directory, index = Path(task['source']), task['index']
        if directory not in self.directories:
            self.directories[directory] = (
                json.loads((directory / 'meta/info.json').read_text(encoding='utf-8')),
                read_jsonl(directory / 'meta/episodes.jsonl', 'episode_index'),
                {i: r['task'] for i, r in read_jsonl(directory / 'meta/tasks.jsonl', 'task_index').items()})
        info, metadata, tasks = self.directories[directory]
        kind, _ = validate_source(info)
        if kind not in self.kinematics:
            self.kinematics[kind] = GalaxeaKinematics(self.model_dir / f'galaxea_{kind}', kind,
                                                      self.params['eef_offsets'][kind])
        return convert_episode(directory, index, metadata[index], tasks, self.kinematics[kind], info, destination)


SOURCE = Galaxea
