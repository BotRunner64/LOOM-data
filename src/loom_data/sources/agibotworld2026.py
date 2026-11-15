"""Convert AgiBot World 2026 LeRobot archives to the export contract."""
from __future__ import annotations

from functools import cached_property
import io
import json
import math
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
import pyarrow.parquet as pq

from loom_data.geometry import pose_matrix
from loom_data.contract import seconds_to_ns, write_state, write_video_index
from loom_data.video import transcode_video
from loom_data.robot.kinematics import fk_poses
from loom_data.robot.urdf_model import Robot, parse_urdf

from loom_data.sources.base import Source

STATE_FIELDS = ('state/left_effector/position', 'state/right_effector/position',
                'state/end/arm_position', 'state/end/arm_orientation', 'state/waist/position',
                'state/robot/position', 'state/robot/orientation')
# Recorded flange origin is the gripper mounting datum; keep the tool axes.
DEFAULT_EEF_OFFSET = (0., 0., 0.)
CAMERA_PREFIX = 'observation.images.'
ARCHIVE_NAME = re.compile(r'^(\d+)_(\d+)\.tar\.gz$')
PARQUET_NAME = re.compile(r'^data/data/chunk-\d+/(episode_\d+)\.parquet$')
VIDEO_NAME = re.compile(r'^data/videos/chunk-\d+/(observation\.images\.[^/]+)/(episode_\d+)\.mp4$')


def camera_id(key):
    if not key.startswith(CAMERA_PREFIX):
        raise ValueError(f'Invalid camera key: {key}')
    name = key[len(CAMERA_PREFIX):]
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError(f'Invalid camera identifier: {key}')
    return name


def source_fields(info):
    features = info['features']
    if info.get('robot_type') != 'g2a':
        raise ValueError('Expected AgiBot G2 (robot_type g2a)')
    fields = features['observation.state']['field_descriptions']
    indices = {key: fields[key]['indices'] for key in STATE_FIELDS}
    lengths = (1, 1, 6, 8, 5, 3, 4)
    for key, count in zip(STATE_FIELDS, lengths):
        allowed = (0, count) if key.startswith('state/robot/') else (count,)
        if len(indices[key]) not in allowed:
            raise ValueError(f'Unexpected source dimensions for {key}')
    if bool(indices['state/robot/position']) != bool(indices['state/robot/orientation']):
        raise ValueError('Robot position and orientation must be present together')
    cameras = {key: camera_id(key) for key, feature in features.items()
               if key.startswith(CAMERA_PREFIX) and not feature.get('video_info', {}).get('video.is_depth_map', False)}
    if not cameras or len(set(cameras.values())) != len(cameras):
        raise ValueError('Expected unique RGB camera IDs')
    return indices, cameras


class G2FK:
    """Torso FK plus recorded flange poses; never recompute the arm with crsB."""

    def __init__(self, directory, eef_offset=DEFAULT_EEF_OFFSET):
        manifest = json.loads((directory / 'robot.json').read_text())
        if manifest['robot'] != 'agibot_g2':
            raise ValueError('AgiBot World 2026 requires the G2 model')
        model = parse_urdf(directory / manifest['urdf'])
        parents = {joint.child: joint for joint in model.joints}
        chain, link = [], 'arm_base_link'
        while link in parents:
            joint = parents[link]
            chain.append(joint)
            link = joint.parent
        if link != 'base_link' or {j.name for j in chain if j.type != 'fixed'} != {f'body_joint{i}' for i in range(1, 6)}:
            raise ValueError('Expected G2 base_link to arm_base_link torso chain')
        self.robot = Robot(model.name, tuple([link] + [j.child for j in reversed(chain)]), tuple(chain))
        offset = np.asarray(eef_offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('EEF mounting offset must contain three finite metres')
        self.eef = np.append(offset, 1.)
        # Columns are contract X, Y, Z expressed in the SOURCE FLANGE:
        # approach +Z, lateral -Y, back (wrist camera side) +X.
        self.axes = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])

    def poses(self, state, indices, stationary_base=False):
        def field(key):
            return np.asarray([state[i] for i in indices[key]], dtype=float)

        waist = field('state/waist/position')
        if not np.isfinite(waist).all():
            raise ValueError('Nonfinite measured waist state')
        values = dict(zip((f'body_joint{i}' for i in range(1, 6)), waist))
        torso = fk_poses(self.robot, values)['arm_base_link']
        if indices['state/robot/position']:
            world = pose_matrix(field('state/robot/position'), field('state/robot/orientation'))
        elif stationary_base:
            world = np.eye(4)
        else:
            raise ValueError('Mobile base pose is absent without verified stationary-base data')
        result = []
        positions = field('state/end/arm_position').reshape(2, 3)
        orientations = field('state/end/arm_orientation').reshape(2, 4)
        for position, orientation in zip(positions, orientations):
            transform = world @ torso @ pose_matrix(position, orientation)
            rotation = transform[:3, :3] @ self.axes
            point = transform @ self.eef
            result.append(np.concatenate((point[:3], rotation.T.reshape(-1))).tolist())
        return result


def openness(value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite measured gripper position')
    # Checked against wrist RGB: 0 open, negative values close the fingers.
    return min(1., max(0., 1. + float(value) / 0.91))


def verify_stationary_base(info, actions):
    """Accept an absent base pose only for source episodes with zero base commands."""
    try:
        indices = info['features']['action']['field_descriptions']['action/robot/velocity']['indices']
    except KeyError as exc:
        raise ValueError('Missing base pose and base velocity field') from exc
    if not indices or not actions:
        raise ValueError('Missing base pose and base velocity samples')
    commands = np.asarray([[action[i] for i in indices] for action in actions], dtype=float)
    if not np.isfinite(commands).all() or np.any(np.abs(commands) > 1e-6):
        raise ValueError('Missing base pose with nonzero or nonfinite base velocity commands')


def instructions(info, episode_index, task, times):
    if not task or not task.strip() or len(times) < 2 or times[-1] <= times[0]:
        raise ValueError('Missing task text or zero-duration episode')
    end = times[-1] - times[0]
    segments = info.get('instruction_segments', {}).get(str(episode_index), [])
    # The "default" track is the step-level, nonoverlapping instruction lane.
    # Other tracks (per-arm and subtask) can overlap it.
    selected = sorted((s for s in segments if str(s.get('track', '')).lower() == 'default'
                       and str(s.get('instruction', '')).strip()),
                      key=lambda s: s['start_frame_index'])
    output = []
    cursor = 0
    for segment in selected:
        start = int(segment['start_frame_index'])
        stop = int(segment['end_frame_index'])
        if not 0 <= start < stop <= len(times):
            raise ValueError('Instruction frame interval is out of bounds')
        begin_ns = times[start] - times[0]
        stop_ns = (times[stop] if stop < len(times) else times[-1]) - times[0]
        if begin_ns < cursor:
            raise ValueError('Overlapping default instruction segments')
        if begin_ns > cursor:
            output.append({'start_ns': cursor, 'end_ns': begin_ns, 'text': task})
        if stop_ns > begin_ns:
            output.append({'start_ns': begin_ns, 'end_ns': stop_ns,
                           'text': segment['instruction'].strip()})
        cursor = stop_ns
    if cursor < end:
        output.append({'start_ns': cursor, 'end_ns': end, 'text': task.strip()})
    return output


def read_episode(member, archive, info, indices, fk, destination):
    data = io.BytesIO(archive.extractfile(member).read())
    table = pq.read_table(
        data, columns=['observation.state', 'timestamp', 'frame_index', 'episode_index'])
    rows = table.to_pydict()
    episode_index = int(member.name.rsplit('_', 1)[-1].split('.')[0])
    if not rows['timestamp'] or any(v != episode_index for v in rows['episode_index']):
        raise ValueError(f'{member.name}: invalid episode rows')
    if rows['frame_index'] != list(range(len(rows['timestamp']))):
        raise ValueError(f'{member.name}: frame_index is not consecutive')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{member.name}: timestamps decrease')
    if times[0] != 0:
        # Source LeRobot timestamps are already relative to each episode.
        raise ValueError(f'{member.name}: expected episode-relative timestamp starting at zero')
    base_indices = indices['state/robot/position'] + indices['state/robot/orientation']
    # Stationary episodes can use zero position AND zero quaternion as a
    # missing-pose marker, either throughout or partway through an episode.
    missing_frames = [all(state[i] == 0 for i in base_indices)
                      for state in rows['observation.state']]
    missing_base_pose = all(missing_frames)
    pose_indices = indices
    if any(missing_frames):
        data.seek(0)
        actions = pq.read_table(data, columns=['action']).to_pydict()['action']
        verify_stationary_base(info, actions)
    if missing_base_pose:
        pose_indices = {**indices, 'state/robot/position': [], 'state/robot/orientation': []}
    elif any(missing_frames):
        # Keep the measured reference frame rather than jumping to identity
        # when odometry drops out. Leading gaps use the first recorded pose.
        reference = next(state for state, missing in zip(rows['observation.state'], missing_frames)
                         if not missing)
        for state, missing in zip(rows['observation.state'], missing_frames):
            if missing:
                for i in base_indices:
                    state[i] = reference[i]
            else:
                reference = state
    values = {'left_eef': [], 'right_eef': [], 'left_gripper': [], 'right_gripper': []}
    for state in rows['observation.state']:
        left, right = fk.poses(state, pose_indices, missing_base_pose)
        values['left_eef'].append(left)
        values['right_eef'].append(right)
        values['left_gripper'].append(openness(state[indices['state/left_effector/position'][0]]))
        values['right_gripper'].append(openness(state[indices['state/right_effector/position'][0]]))
    for name, samples in values.items():
        write_state(destination / 'state' / f'{name}.parquet', times, samples, times[0],
                    pose=name.endswith('_eef'))
    return episode_index, times


def archive_index(path):
    """Read metadata and parquet member names in streaming order, never extract a tar tree."""
    match = ARCHIVE_NAME.fullmatch(path.name)
    if not match or not re.fullmatch(r'task_\d+', path.parent.name):
        raise ValueError(f'Unexpected AgiBot archive path: {path}')
    prefix = f'{path.parent.name}_{match.group(1)}_{match.group(2)}'
    info, metadata, entries = None, {}, {}
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name.startswith('data/videos/'):
                break
            if member.name == 'data/meta/info.json':
                info = json.load(archive.extractfile(member))
            elif member.name == 'data/meta/episodes.jsonl':
                for line in archive.extractfile(member):
                    record = json.loads(line)
                    index = record['episode_index']
                    if type(index) is not int or index < 0 or index in metadata:
                        raise ValueError(f'{path}: invalid/duplicate metadata episode index')
                    metadata[index] = record
            elif (match := PARQUET_NAME.fullmatch(member.name)):
                local = match.group(1)
                index = int(local.split('_')[1])
                if index in entries:
                    raise ValueError(f'{path}: duplicate parquet episode index')
                entries[index] = f'{prefix}_{local}'
    if (info is None or not entries or len(entries) != info['total_episodes']
            or set(entries) != set(metadata)):
        raise ValueError(f'{path}: missing or inconsistent episode metadata/data')
    source_fields(info)
    return info, metadata, entries


def convert_archive(path, staged, fk, episode_limit=None, *, episode_ids=None,
                    on_episode=None, on_error=None):
    """Stream one archive, optionally committing successes as each episode finishes.

    The standalone caller keeps all-or-nothing dataset publication. Parallel
    callers pass only unfinished IDs and persist successful episodes immediately.
    """
    staging = staged / 'episodes'
    staging.mkdir(parents=True, exist_ok=True)
    info, metadata, entries = archive_index(path)
    indices, cameras = source_fields(info)
    pending = set(episode_ids if episode_ids is not None else list(entries.values())[:episode_limit])
    selected = {index: episode_id for index, episode_id in entries.items() if episode_id in pending}
    if set(selected.values()) != set(pending):
        raise ValueError(f'{path}: archive episode identities changed')
    ready, broken, completed = {}, set(), {}

    def fail(index, exc):
        if index not in broken:
            broken.add(index)
            if on_error is None:
                raise exc
            on_error(selected[index], exc)

    # Scan the archive once for state; each successful episode remains independent.
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name.startswith('data/videos/'):
                break
            match = PARQUET_NAME.fullmatch(member.name)
            if not match:
                continue
            index = int(match.group(1).split('_')[1])
            if index not in selected:
                continue
            destination = staging / selected[index]
            try:
                _, times = read_episode(member, archive, info, indices, fk, destination)
                record = {'episode_id': selected[index], 'cameras': sorted(cameras.values()),
                          'instructions': instructions(info, index, metadata[index]['tasks'][0], times)}
                ready[index] = (times, record)
            except Exception as exc:
                fail(index, exc)
    seen = {index: set() for index in ready}
    committed = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            match = VIDEO_NAME.fullmatch(member.name)
            if not match or match.group(1) not in cameras:
                continue
            key, local = match.groups()
            index = int(local.split('_')[1])
            if index not in ready or index in broken or index in committed:
                continue
            times, record = ready[index]
            rgb = staging / selected[index] / 'rgb'
            try:
                if key in seen[index]:
                    raise ValueError(f'{path}: duplicate video {member.name}')
                rgb.mkdir(exist_ok=True)
                temporary = staging / 'source.mp4'
                with temporary.open('wb') as handle:
                    shutil.copyfileobj(archive.extractfile(member), handle)
                try:
                    transcode_video(temporary, rgb / f'{cameras[key]}.mp4', len(times))
                finally:
                    temporary.unlink(missing_ok=True)
                write_video_index(rgb / f'{cameras[key]}.parquet', times, times[0])
                seen[index].add(key)
                if seen[index] == set(cameras):
                    if on_episode is not None:
                        on_episode(record, staging / selected[index])
                    completed[index] = record
                    committed.add(index)
            except Exception as exc:
                fail(index, exc)
            if len(committed | broken) == len(selected):
                break
    for index in selected.keys() - committed - broken:
        fail(index, ValueError(f'{path}: missing state or RGB for {selected[index]}'))
    return [completed[index] for index in sorted(completed)]


class AgiBotWorld2026(Source):
    name = 'agibotworld2026'
    title = 'AgiBotWorld2026'
    model = 'agibot_g2'
    accepts_file = True  # one .tar.gz archive
    OPTIONS = {
        'eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                           help='mounting origin in source tool axes, metres (default 0 0 0)'),
    }

    @classmethod
    def make_params(cls, options):
        offset = options['eef_offset']
        return {'eef_offset': list(offset if offset is not None else DEFAULT_EEF_OFFSET)}

    @classmethod
    def discover(cls, config, progress):
        # Here --limit counts archives; --limit-episodes counts episodes across them.
        source = Path(config['source'])
        if source.is_dir():
            archives = []
            with progress('Scanning for archives', unit='archive') as bar:
                for path in source.rglob('*.tar.gz'):
                    archives.append(path)
                    bar.update()
            paths = sorted(archives)
        else:
            paths = [source]
        selected = paths if config['limit'] is None else paths[:config['limit']]
        tasks, count = [], 0
        with progress('Indexing archives', len(selected), unit='archive') as bar:
            for path in selected:
                _, _, entries = archive_index(path)
                tasks.append({'source': str(path), 'episode_ids': list(entries.values()), 'inputs': [str(path)]})
                count += len(entries)
                bar.update()
                if config['limit_episodes'] is not None and count >= config['limit_episodes']:
                    break
        return tasks, []

    @cached_property
    def fk(self):
        return G2FK(self.model_dir, self.params['eef_offset'])

    def process(self, task, pending, staging, store, failed):
        # One task is a whole archive, streamed once; episodes commit as they complete.
        convert_archive(Path(task['source']), staging, self.fk, episode_ids=pending,
                        on_episode=lambda record, directory: store.commit(record['episode_id'], directory, record),
                        on_error=failed)


SOURCE = AgiBotWorld2026
