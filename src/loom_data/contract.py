"""The export contract in code: names, schemas, writers and validation.

This is the only module that knows the on-disk format defined in
docs/contract/export-contract.md. Converters write through it; the checker
and the viewer take their names from it.
"""
import math
import os
import re
import stat

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Recorded in resume plans to prevent mixing incompatible exports.
VERSION = 3

MANIFEST = 'episodes.jsonl'
EPISODES = 'episodes'
SIDES = ('left', 'right')
STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
EPISODE_ID = re.compile(r'[A-Za-z0-9_-]+')
CAMERA_IDS = ('left_wrist', 'right_wrist', 'top')

POSE = pa.list_(pa.float64(), 12)
OPENNESS = pa.float32()
TIMESTAMP = pa.field('timestamp_ns', pa.int64(), nullable=False)
FRAME_SCHEMA = pa.schema([pa.field('frame_index', pa.int64(), nullable=False), TIMESTAMP])


def is_pose(name):
    return name.endswith('_eef')


def state_field(name):
    """Value column of a state table: (column name, Arrow type)."""
    return ('pose', POSE) if is_pose(name) else ('openness', OPENNESS)


def state_schema(name):
    column, dtype = state_field(name)
    return pa.schema([TIMESTAMP, pa.field(column, dtype, nullable=False)])


def encode_pose(transform):
    """4x4 T_reference_eef -> [x, y, z, R column-major]."""
    return np.concatenate((transform[:3, 3], transform[:3, :3].T.reshape(-1))).tolist()


def pose_rotations(poses):
    """N x 12 contract poses -> N rotation matrices."""
    return np.asarray(poses, dtype=np.float64)[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)


def check_rotations(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid rotation matrix')


def seconds_to_ns(seconds):
    if not math.isfinite(seconds):
        raise ValueError(f'Nonfinite source timestamp: {seconds}')
    scaled = float(seconds) * 1_000_000_000
    if not math.isfinite(scaled):
        raise ValueError('Source timestamp exceeds int64 nanoseconds')
    value = round(scaled)
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError('Source timestamp exceeds int64 nanoseconds')
    return value


def relative_times(timestamps, origin):
    """Timestamps on the episode clock: relative to the shared origin, ordered."""
    values = [int(t) - int(origin) for t in timestamps]
    if not values or values[0] < 0 or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError('Empty, negative or unordered sample timestamps')
    return pa.array(values, type=pa.int64())


def write_state(path, timestamps, values, origin, *, pose=False):
    name, dtype = ('pose', POSE) if pose else ('openness', OPENNESS)
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite {name}: {path}')
    if pose:
        if data.shape != (len(timestamps), 12):
            raise ValueError('Expected N x 12 poses')
        check_rotations(pose_rotations(data))
    elif data.shape != (len(timestamps),) or np.any((data < 0) | (data > 1)):
        raise ValueError('Expected scalar gripper openness in [0,1]')
    schema = pa.schema([TIMESTAMP, pa.field(name, dtype, nullable=False)])
    table = pa.Table.from_arrays([relative_times(timestamps, origin),
                                 pa.array(data.tolist(), type=dtype)], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_video_index(path, timestamps, origin):
    pq.write_table(pa.Table.from_arrays([
        pa.array(range(len(timestamps)), type=pa.int64()),
        relative_times(timestamps, origin)], schema=FRAME_SCHEMA), path)


def validate_record(record, episode_id):
    """One episodes.jsonl object."""
    if not isinstance(record, dict) or record.get('episode_id') != episode_id:
        raise ValueError(f'{episode_id}: invalid manifest identity')
    cameras = record.get('cameras')
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(c, str) or c not in CAMERA_IDS for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError(f'{episode_id}: invalid cameras')
    instructions = record.get('instructions')
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f'{episode_id}: missing instructions')
    previous = 0
    for instruction in instructions:
        if not isinstance(instruction, dict):
            raise ValueError(f'{episode_id}: invalid instruction')
        start, end, text = (instruction.get(k) for k in ('start_ns', 'end_ns', 'text'))
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end < 2**63
                or not isinstance(text, str) or not text.strip()):
            raise ValueError(f'{episode_id}: invalid instruction interval/text')
        previous = end


def file_sizes(directory, record):
    """Exact file set of one episode with sizes; cheap enough for resume checks."""
    expected = {f'state/{name}.parquet' for name in STATE_NAMES}
    expected.update(f'rgb/{camera}.{ext}' for camera in record['cameras'] for ext in ('mp4', 'parquet'))
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError(f'{directory}: invalid episode directory')
    with os.scandir(directory) as entries:
        if {entry.name for entry in entries} != {'state', 'rgb'}:
            raise ValueError(f'{directory}: invalid episode directory')
    files = {}
    for kind in ('state', 'rgb'):
        folder = directory / kind
        if not stat.S_ISDIR(folder.lstat().st_mode):
            raise ValueError(f'{folder}: invalid stream directory')
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                    raise ValueError(f'{folder / entry.name}: invalid stream file')
                files[f'{kind}/{entry.name}'] = info.st_size
    if set(files) != expected:
        raise ValueError(f'{directory}: stream files differ from manifest')
    return files


def validate_episode(directory, record):
    """Check a finished episode directory against the contract; return its file sizes."""
    import av
    validate_record(record, directory.name)
    sizes = file_sizes(directory, record)
    first, last = [], []
    for relative in sorted(sizes):
        if not relative.endswith('.parquet'):
            continue
        path = directory / relative
        table = pq.read_table(path)
        video = relative.startswith('rgb/')
        pose = is_pose(path.stem)
        field, dtype = ('frame_index', pa.int64()) if video else state_field(path.stem)
        if (set(table.column_names) != {'timestamp_ns', field}
                or table.schema.field('timestamp_ns').type != pa.int64()
                or table.schema.field(field).type != dtype
                or any(c.null_count for c in table.columns) or not table.num_rows
                or (not video and pose and table.schema.field('pose').nullable)):
            raise ValueError(f'{path}: invalid schema or empty/null samples')
        times = table['timestamp_ns'].to_numpy()
        if times[0] < 0 or np.any(times[1:] < times[:-1]):
            raise ValueError(f'{path}: invalid timestamps')
        first.append(int(times[0]))
        last.append(int(times[-1]))
        values = np.asarray(table[field].to_pylist())
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: nonfinite values')
        if video:
            if not np.array_equal(values, np.arange(len(times))):
                raise ValueError(f'{path}: invalid frame indices')
            # Converters already decode every input frame and flush the encoder.
            # Verify the resulting MP4 sample count without decoding it a second time.
            with av.open(str(path.with_suffix('.mp4'))) as reader:
                if (len(reader.streams) != 1 or len(reader.streams.video) != 1
                        or reader.streams.video[0].codec_context.name != 'h264'
                        or 'mp4' not in reader.format.name.split(',')
                        or reader.streams.video[0].frames != len(times)):
                    raise ValueError(f'{path}: video format/frame count differs from index')
        elif pose:
            try:
                check_rotations(pose_rotations(values))
            except ValueError as exc:
                raise ValueError(f'{path}: invalid rotation matrix') from exc
        elif np.any((values < 0) | (values > 1)):
            raise ValueError(f'{path}: invalid gripper openness')
    if min(first) != 0 or record['instructions'][-1]['end_ns'] > max(last):
        raise ValueError(f'{directory}: invalid shared origin/instruction extent')
    return sizes
