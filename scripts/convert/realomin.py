#!/usr/bin/env python3
"""Convert dual-gripper RealOmin MCAPs, preserving each stream's native clock."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import os
from itertools import islice
from pathlib import Path
import re

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import (
    VideoWriter, normalized_gripper, pose_matrix, rotation9d,
    tool_transform, write_state, write_video_index,
)
from scripts.convert.source_policy import SourceDataError, source_value

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = 'genrobot2025/10Kh-RealOmin-OpenData'
STATES = {f'/robot{i}/{topic}': f'{side}_{kind}'
          for i, side in enumerate(('left', 'right'))
          for topic, kind in (('vio/eef_pose', 'eef'), ('sensor/magnetic_encoder', 'gripper'))}
CAMERA = re.compile(r'^/robot([01])/sensor/camera(\d+)/compressed$')
# Nominal closed inner finger-tip midpoint from the supplied V3 URDF + meshes.
# VIO-to-URDF registration is not a measured calibration; see source.md.
DEFAULT_TCP_OFFSET = (0.14405256, 0.00049106, -0.03820249)
DEFAULT_TOOL_RPY = (0., -0.261799387799149, 0.)
DEFAULT_GRIPPER_OPEN = 0.103
DEFAULT_GRIPPER_CLOSED = 0.
DEFAULT_INPUT = ROOT / 'dataset/raw/genrobot-realomin-10kh'


def camera_id(topic):
    match = CAMERA.fullmatch(topic)
    if not match:
        raise ValueError(f'Unsupported camera topic: {topic}')
    side = ('left', 'right')[int(match[1])]
    return f'{side}_wrist' + (f'_camera{match[2]}' if int(match[2]) else '')


def messages(path, *, video=False):
    from google.protobuf.message import DecodeError
    from mcap.exceptions import EndOfFile, InvalidMagic, RecordLengthLimitExceeded
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    try:
        with path.open('rb') as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            if summary is not None:
                available = {c.topic for c in summary.channels.values()}
            else:
                available = {c.topic for _, c, _ in reader.iter_messages()}
            topics = sorted(t for t in available if CAMERA.fullmatch(t)) if video else list(STATES)
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
        from scripts.download.subset_common import load_manifest
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
    transform = tool_transform(DEFAULT_TCP_OFFSET, DEFAULT_TOOL_RPY) if transform is None else transform
    normalized_gripper(0., gripper_closed, gripper_open)
    state = {name: ([], []) for name in STATES.values()}
    # Source channel is authoritative: right-hand header.topic_name can say robot0.
    for topic, timestamp, decoded in messages(Path(source['path'])):
        name = STATES[topic]
        if name.endswith('_eef'):
            if decoded.frame_id != 'world' or not decoded.HasField('pose'):
                raise SourceDataError('invalid_eef_pose', f'{topic}: expected a pose in the fixed world frame')
            p, q = decoded.pose.position, decoded.pose.orientation
            value = rotation9d(source_value('invalid_eef_pose', pose_matrix, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)) @ transform)
        else:
            value = source_value('invalid_gripper', normalized_gripper, decoded.value, gripper_closed, gripper_open)
        state[name][0].append(timestamp)
        state[name][1].append(value)
    if any(not times for times, _ in state.values()):
        raise SourceDataError('missing_state_stream', f"{source['path']}: requires all four native state streams")
    for times, _ in state.values():
        if any(b < a for a, b in zip(times, times[1:])):
            raise SourceDataError('invalid_timestamp', 'Decreasing state acquisition timestamps')
    videos = {}
    (destination / 'rgb').mkdir(parents=True)
    with ExitStack() as stack:
        for topic, timestamp, decoded in messages(Path(source['path']), video=True):
            camera = camera_id(topic)
            if camera not in videos:
                videos[camera] = VideoWriter(destination / 'rgb' / f'{camera}.mp4', decoded.format)
                stack.callback(videos[camera].close)
            videos[camera].add(decoded.data, timestamp, decoded.format)
        if not videos:
            raise SourceDataError('missing_rgb', 'Requires at least one RGB camera')
        for writer in videos.values():
            writer.finish()
    streams = [t for t, _ in state.values()] + [v.timestamps for v in videos.values()]
    origin, end = min(t[0] for t in streams), max(t[-1] for t in streams)
    if end <= origin:
        raise SourceDataError('invalid_timestamp', 'Zero-duration episode')
    for name, (times, values) in state.items():
        write_state(destination / 'state' / f'{name}.parquet', times, values, origin, pose=name.endswith('_eef'))
    for camera, writer in videos.items():
        write_video_index(destination / 'rgb' / f'{camera}.parquet', writer.timestamps, origin)
    return {'episode_id': destination.name, 'cameras': sorted(videos),
            'instructions': [{'start_ns': 0, 'end_ns': end-origin, 'text': source['instruction']}]}


def convert(source, output, limit=None, tcp_offset=DEFAULT_TCP_OFFSET, tool_rpy=DEFAULT_TOOL_RPY,
            gripper_closed=DEFAULT_GRIPPER_CLOSED, gripper_open=DEFAULT_GRIPPER_OPEN, instruction=None):
    from scripts.convert.parallel import run_native
    from scripts.convert.parallel_adapters import make_config
    config = make_config('realomin', source, limit=limit, tcp_offset=tcp_offset, tool_rpy=tool_rpy,
                         gripper_closed=gripper_closed, gripper_open=gripper_open, instruction=instruction)
    run_native(config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT if DEFAULT_INPUT.exists() else ROOT / 'dataset/raw/10Kh-RealOmin-OpenData')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/processed/10Kh-RealOmin-OpenData')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--tcp-offset', nargs=3, type=float, default=DEFAULT_TCP_OFFSET, help='TCP in source VIO body axes, metres; nominal V3 geometry by default')
    parser.add_argument('--tool-rpy', nargs=3, type=float, default=DEFAULT_TOOL_RPY, help='contract tool axes in source body, RPY radians')
    parser.add_argument('--gripper-open', type=float, default=DEFAULT_GRIPPER_OPEN)
    parser.add_argument('--gripper-closed', type=float, default=DEFAULT_GRIPPER_CLOSED)
    parser.add_argument('--instruction', help='Override path-derived skill text for all selected episodes')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        convert(args.input_dir, args.output_dir, args.limit, args.tcp_offset, args.tool_rpy,
                args.gripper_closed, args.gripper_open, args.instruction)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
