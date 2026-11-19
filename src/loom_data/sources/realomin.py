"""Convert dual-gripper RealOmin MCAPs, resampling their asynchronous streams to 30 fps."""
from __future__ import annotations

from functools import cached_property
from contextlib import ExitStack
import hashlib
import os
from itertools import islice
from pathlib import Path

from loom_data.video import VideoWriter
from loom_data.sources.common import normalized_gripper
from loom_data.geometry import eef_offset, pose_matrix, tool_transform
from loom_data.contract import encode_pose, episode_record, write_frames
from loom_data.sync import frame_grid
from loom_data.sources.policy import SourceDataError, source_value

from loom_data.sources.base import Source, task_limit

REPO_ID = 'genrobot2025/10Kh-RealOmin-OpenData'
STATES = {f'/robot{i}/{topic}': f'{side}_{kind}'
          for i, side in enumerate(('left', 'right'))
          for topic, kind in (('vio/eef_pose', 'eef'), ('sensor/magnetic_encoder', 'gripper'))}
CAMERAS = {'/robot0/sensor/camera0/compressed': 'left_wrist',
           '/robot1/sensor/camera0/compressed': 'right_wrist'}
# Nominal V3 rear flange, transferred from the vendor's paired controller model
# via the drive-pivot datum, then expressed about camera0's raw EE origin.
# Reproduce with `loom-data robot realomin-mount`; model hashes and assumptions
# are recorded in docs/sources/realomin.md and the geometry fixture.
DEFAULT_EEF_OFFSET = (-0.068277831741, 0.000372712242, -0.098051703299)
DEFAULT_TOOL_RPY = (0., -0.261799387799149, 0.)
DEFAULT_GRIPPER_OPEN = 0.103
DEFAULT_GRIPPER_CLOSED = 0.


def messages(path, topics):
    from google.protobuf.message import DecodeError
    from mcap.exceptions import EndOfFile, InvalidMagic, RecordLengthLimitExceeded
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    try:
        with path.open('rb') as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            for _, channel, _, decoded in reader.iter_decoded_messages(topics=topics):
                fields = decoded.DESCRIPTOR.fields_by_name if hasattr(decoded, 'DESCRIPTOR') else None
                if fields is not None and 'header' not in fields:
                    raise SourceDataError('missing_header', f'{path}: no acquisition header on {channel.topic}')
                if fields is not None:
                    required = ({'frame_id', 'pose'} if channel.topic.endswith('/eef_pose') else
                                {'value'} if channel.topic in STATES else {'format', 'data'})
                    if not required <= fields.keys():
                        raise SourceDataError('missing_state_field', f'{path}: incomplete schema on {channel.topic}')
                if not decoded.HasField('header'):
                    raise SourceDataError('missing_header', f'{path}: missing acquisition header on {channel.topic}')
                timestamp = int(decoded.header.timestamp)
                if not 0 < timestamp < 2**63:
                    raise SourceDataError('invalid_timestamp', f'{path}: invalid acquisition timestamp on {channel.topic}')
                yield channel.topic, timestamp, decoded
    except FileNotFoundError as exc:
        raise SourceDataError('missing_mcap', f'Missing MCAP: {path}') from exc
    except (EndOfFile, InvalidMagic, RecordLengthLimitExceeded, DecodeError) as exc:
        raise SourceDataError('invalid_mcap', f'{path}: {exc}') from exc


def iter_discover(source, instruction=None):
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f'Expected a RealOmin data root: {source}')
    manifest_path = source / 'subset_manifest.json'
    if manifest_path.exists():
        from loom_data.download.hub import load_manifest
        manifest = load_manifest(manifest_path, REPO_ID)
        paths = iter(sorted(source / f['path'] for f in manifest['files'] if f['path'].endswith('.mcap')))
    else:
        def walk():
            def on_error(exc):
                raise exc
            for directory, folders, files in os.walk(source, onerror=on_error):
                folders[:] = sorted(d for d in folders if not d.startswith('.'))
                for name in sorted(files):
                    if name.endswith('.mcap'):
                        yield Path(directory) / name
        paths = walk()
    for path in paths:
        relative = path.relative_to(source)
        if '..' in relative.parts:
            raise ValueError(f'Unsafe MCAP path: {path}')
        task = instruction
        if task is None:
            skill = next((part for part in reversed(relative.parts[:-1]) if not part.isdecimal()), None)
            if not skill:
                raise ValueError(f'{path}: cannot infer skill from path; supply --instruction')
            task = skill.replace('_', ' ').strip()
        if not isinstance(task, str) or not task.strip():
            raise ValueError('Instruction must be nonempty')
        digest = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16]
        item = {'path': path, 'instruction': task.strip()}
        if not path.is_file():
            item['drop'] = {'reason': 'missing_mcap', 'detail': f'Missing MCAP: {path}'}
        yield f'realomin_{digest}', item


def discover(source, instruction=None, limit=None):
    items = list(islice(iter_discover(source, instruction), limit))
    if not items:
        raise ValueError(f'No MCAP trajectories found under {source}')
    return items


def convert_episode(source, destination, transform=None,
                    gripper_closed=DEFAULT_GRIPPER_CLOSED, gripper_open=DEFAULT_GRIPPER_OPEN):
    transform = tool_transform(DEFAULT_EEF_OFFSET, DEFAULT_TOOL_RPY) if transform is None else transform
    normalized_gripper(0., gripper_closed, gripper_open)
    state = {name: ([], []) for name in STATES.values()}
    camera_times = {}
    # Source channel is authoritative: right-hand header.topic_name can say robot0.
    # Camera acquisition times are read here too: the 30 fps frame times depend on every stream.
    for topic, timestamp, decoded in messages(Path(source['path']), list(STATES) + list(CAMERAS)):
        if topic in CAMERAS:
            camera_times.setdefault(CAMERAS[topic], []).append(timestamp)
            continue
        name = STATES[topic]
        if name.endswith('_eef'):
            if decoded.frame_id != 'world' or not decoded.HasField('pose'):
                raise SourceDataError('invalid_eef_pose', f'{topic}: expected a pose in the fixed world frame')
            p, q = decoded.pose.position, decoded.pose.orientation
            value = encode_pose(source_value('invalid_eef_pose', pose_matrix, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)) @ transform)
        else:
            value = source_value('invalid_gripper', normalized_gripper, decoded.value, gripper_closed, gripper_open)
        state[name][0].append(timestamp)
        state[name][1].append(value)
    if any(not times for times, _ in state.values()):
        raise SourceDataError('missing_state_stream', f"{source['path']}: requires all four native state streams")
    for times, _ in state.values():
        if any(b < a for a, b in zip(times, times[1:])):
            raise SourceDataError('invalid_timestamp', 'Decreasing state acquisition timestamps')
    if not camera_times:
        raise SourceDataError('missing_rgb', 'Requires at least one RGB camera')
    streams = [t for t, _ in state.values()] + [sorted(t) for t in camera_times.values()]
    origin, end = min(t[0] for t in streams), max(t[-1] for t in streams)
    if end <= origin:
        raise SourceDataError('invalid_timestamp', 'Zero-duration episode')
    grid = frame_grid(streams)
    length = write_frames(destination, state,
                          [{'start_ns': 0, 'end_ns': end-origin, 'text': source['instruction']}], origin, grid)
    writers = {}
    with ExitStack() as stack:
        for topic, timestamp, decoded in messages(Path(source['path']), list(CAMERAS)):
            camera = CAMERAS[topic]
            if camera not in writers:
                writers[camera] = VideoWriter(destination / f'{camera}.mp4', decoded.format,
                                              camera_times[camera], grid)
                stack.callback(writers[camera].close)
            writers[camera].add(decoded.data, timestamp, decoded.format)
        videos = {camera: writer.finish() for camera, writer in writers.items()}
    return episode_record(destination.name, length, videos)


class RealOmin(Source):
    name = 'realomin'
    policy = 'drop'
    camera_sets = (('left_wrist', 'right_wrist'),)
    OPTIONS = {
        'eef_offset': dict(type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                           help='mounting origin in source raw EE axes, metres (default: nominal V3 rear flange)'),
        'tool_rpy': dict(type=float, nargs=3, metavar=('R', 'P', 'Y'),
                         help='contract tool axes in the source body, RPY radians'),
        'gripper_open': dict(type=float, help='measured source value at full open'),
        'gripper_closed': dict(type=float, help='measured source value at full close'),
        'instruction': dict(help='task text for every episode, instead of the skill name in its path'),
    }

    @classmethod
    def make_params(cls, options):
        params = {'data_policy_version': 1,
                  'eef_offset': eef_offset(options['eef_offset'] if options['eef_offset'] is not None
                                           else DEFAULT_EEF_OFFSET).tolist(),
                  'gripper_open': DEFAULT_GRIPPER_OPEN if options['gripper_open'] is None else options['gripper_open'],
                  'gripper_closed': DEFAULT_GRIPPER_CLOSED if options['gripper_closed'] is None else options['gripper_closed'],
                  'tool_rpy': list(options['tool_rpy'] if options['tool_rpy'] is not None else DEFAULT_TOOL_RPY),
                  'instruction': options['instruction']}
        normalized_gripper(0., params['gripper_closed'], params['gripper_open'])
        if params['instruction'] is not None and (not isinstance(params['instruction'], str)
                                                  or not params['instruction'].strip()):
            raise ValueError('Instruction override must be nonempty')
        tool_transform(params['eef_offset'], params['tool_rpy'])
        return params

    @classmethod
    def discover(cls, config, progress):
        source = Path(config['source'])
        shared = [source / 'subset_manifest.json'] if (source / 'subset_manifest.json').exists() else []
        tasks = []
        with progress('Discovering native-pose episodes', unit='episode') as bar:
            for episode_id, item in islice(iter_discover(source, config['params']['instruction']), task_limit(config)):
                tasks.append({'source': str(item['path']), 'instruction': item['instruction'],
                              'episode_ids': [episode_id], 'inputs': [str(item['path'])]})
                if 'drop' in item:
                    tasks[-1]['drop'] = item['drop']
                bar.update()
        return tasks, shared

    @cached_property
    def transform(self):
        return tool_transform(self.params['eef_offset'], self.params['tool_rpy'])

    def episode(self, task, destination):
        return convert_episode({'path': Path(task['source']), 'instruction': task['instruction']}, destination,
                               self.transform, self.params['gripper_closed'], self.params['gripper_open'])


SOURCE = RealOmin
