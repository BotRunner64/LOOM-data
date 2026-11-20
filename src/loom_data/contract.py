"""The export contract in code: names, schemas, writers, readers and validation.

This is the only module that knows the on-disk format defined in
docs/contract/export-contract.md: a LeRobot v3.0 dataset per camera layout.

Converters stage one directory per episode (`frames.parquet` plus one MP4 per
camera, already synchronized at 30 fps); `build_dataset` assembles staged
episodes into the published dataset. The checker and the viewer read it back
through `Dataset`.
"""
from fractions import Fraction
import json
import math
import os
import re
import shutil
import stat

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from loom_data import sync
from loom_data.sync import FPS, frame_ns

# Recorded in resume plans to prevent mixing incompatible exports.
VERSION = 5
VIDEO_GOP = 2

SIDES = ('left', 'right')
STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
EPISODE_ID = re.compile(r'[A-Za-z0-9_-]+')
CAMERA_IDS = ('left_wrist', 'right_wrist', 'top')
LAYOUT_ORDER = ('top', 'left_wrist', 'right_wrist')
LAYOUT = re.compile(r'[a-z_]+(\+[a-z_]+)*_\d+x\d+(\+\d+x\d+)*')

STATE_KEY = 'observation.state'
STATE_DIM = 26
# Where each stream lives inside observation.state.
STATE_SLICES = {'left_eef': slice(0, 12), 'left_gripper': slice(12, 13),
                'right_eef': slice(13, 25), 'right_gripper': slice(25, 26)}
POSE_COMPONENTS = ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21', 'r02', 'r12', 'r22')
STATE_COMPONENTS = tuple(
    name for side in SIDES
    for name in (*(f'{side}_eef.{c}' for c in POSE_COMPONENTS), f'{side}_gripper.openness'))
STATE = pa.list_(pa.float32(), STATE_DIM)

# One export resolution per aspect ratio (section 5): width 640 where that gives a whole,
# even height; a ratio that does not scale to one keeps its native resolution.
RESOLUTIONS = {(4, 3): (640, 480), (16, 9): (640, 360), (8, 5): (640, 400), (16, 13): (640, 520),
               (5, 4): (640, 512), (40, 33): (640, 528), (53, 30): (848, 480)}

FRAMES = 'frames.parquet'
FRAME_SCHEMA = pa.schema([pa.field(STATE_KEY, STATE, nullable=False),
                          pa.field('task', pa.string(), nullable=False)])
DATA_SCHEMA = pa.schema([pa.field(STATE_KEY, STATE, nullable=False),
                         pa.field('timestamp', pa.float32(), nullable=False),
                         *(pa.field(name, pa.int64(), nullable=False)
                           for name in ('frame_index', 'episode_index', 'index', 'task_index'))])
DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
VIDEO_PATH = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
EPISODES_PATH = 'meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
CHUNK_SIZE = 1000              # files per chunk directory
DATA_FILE_ROWS = 500_000       # frames per data file before starting the next one
EPISODE_FILE_ROWS = 100_000    # episodes per metadata file


def is_pose(name):
    return name.endswith('_eef')


def video_key(camera):
    return f'observation.images.{camera}'


def encode_pose(transform):
    """4x4 T_reference_eef -> [x, y, z, R column-major]."""
    return np.concatenate((transform[:3, 3], transform[:3, :3].T.reshape(-1))).tolist()


def pose_rotations(poses):
    """N x 12 contract poses -> N rotation matrices."""
    return np.asarray(poses, dtype=np.float64)[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)


def check_rotations(rotations, tolerance=1e-6):
    rotations = np.asarray(rotations, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=tolerance, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=tolerance, rtol=0)):
        raise ValueError('Invalid rotation matrix')


def check_state(state):
    """N x 26 observation.state as stored: finite, float32-orthonormal poses, openness in [0,1]."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != STATE_DIM or not len(state) or not np.isfinite(state).all():
        raise ValueError(f'Expected finite N x {STATE_DIM} {STATE_KEY}')
    for name in STATE_NAMES:
        values = state[:, STATE_SLICES[name]]
        if is_pose(name):
            check_rotations(pose_rotations(values), 1e-5)
        elif np.any((values < 0) | (values > 1)):
            raise ValueError('Gripper openness outside [0,1]')
    return state


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


def export_size(width, height):
    """Export resolution of a source frame: the one registered for its aspect ratio."""
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError(f'Invalid source frame size: {width}x{height}')
    ratio = Fraction(width, height)
    size = RESOLUTIONS.get((ratio.numerator, ratio.denominator))
    if size is None:
        raise ValueError(f'Aspect ratio {ratio.numerator}:{ratio.denominator} ({width}x{height}) has no export '
                         'resolution; register it in the export contract and contract.RESOLUTIONS first')
    return size


def layout_name(cameras, sizes):
    """Deterministic dataset name of a camera set and its export resolutions."""
    cameras = [camera for camera in LAYOUT_ORDER if camera in cameras]
    found = [f'{sizes[camera][0]}x{sizes[camera][1]}' for camera in cameras]
    return '+'.join(cameras) + '_' + (found[0] if len(set(found)) == 1 else '+'.join(found))


def validate_instructions(instructions):
    """Source instruction intervals on the episode clock: ordered, disjoint, nonempty text."""
    if not isinstance(instructions, list) or not instructions:
        raise ValueError('Missing instructions')
    previous = 0
    for instruction in instructions:
        if not isinstance(instruction, dict):
            raise ValueError('Invalid instruction')
        start, end, text = (instruction.get(k) for k in ('start_ns', 'end_ns', 'text'))
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end < 2**63
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid instruction interval/text')
        previous = end


def frame_tasks(instructions, times):
    """Instruction text in force at each frame time (ns on the episode clock)."""
    validate_instructions(instructions)
    starts = np.array([i['start_ns'] for i in instructions], dtype=np.int64)
    index = np.clip(np.searchsorted(starts, times, side='right') - 1, 0, len(instructions) - 1)
    return [instructions[i]['text'] for i in index]


def stream_values(name, times, values):
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite {name}')
    if is_pose(name):
        if data.shape != (len(times), 12):
            raise ValueError('Expected N x 12 poses')
        check_rotations(pose_rotations(data))
    elif data.shape != (len(times),) or np.any((data < 0) | (data > 1)):
        raise ValueError('Expected scalar gripper openness in [0,1]')
    return data


def write_frames(destination, streams, instructions, origin, grid=None):
    """Synchronize the four state streams and the instructions; write frames.parquet.

    `streams` maps each STATE_NAMES entry to (timestamps ns, values). `grid`
    holds the frame times from `sync.frame_grid`; None passes the rows of a
    source that already is a 30 fps frame table through unchanged, and then
    every stream must carry the same timestamps. `instructions` are intervals
    relative to `origin`. Returns the frame count.
    """
    if set(streams) != set(STATE_NAMES):
        raise ValueError('Expected the four state streams')
    columns = {}
    shared = None
    for name in STATE_NAMES:
        times, values = streams[name]
        times = sync.ordered(times, name)
        data = stream_values(name, times, values)
        if grid is None:
            if shared is not None and not np.array_equal(times, shared):
                raise ValueError('Rows can only pass through when all streams share their timestamps')
            shared = times
        elif is_pose(name):
            data = sync.resample_poses(times, data, grid, name)
        else:
            data = np.clip(sync.resample_scalar(times, data, grid, name), 0., 1.)
        columns[name] = data
    frame_times = shared if grid is None else np.asarray(grid, dtype=np.int64)
    if len(frame_times) < 2 or frame_times[0] < origin:
        raise ValueError('An episode needs at least two frames at or after its time origin')
    state = np.empty((len(frame_times), STATE_DIM), dtype=np.float32)
    for name, data in columns.items():
        state[:, STATE_SLICES[name]] = data.reshape(len(frame_times), -1)
    check_state(state)
    tasks = frame_tasks(instructions, frame_times - int(origin))
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(
        [pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), STATE_DIM),
         pa.array(tasks, type=pa.string())], schema=FRAME_SCHEMA), destination / FRAMES)
    return len(frame_times)


def episode_record(episode_id, length, videos, repairs=None):
    """Record of a staged episode. `videos` maps camera -> the dict `FrameSink.finish` returns."""
    record = {'episode_id': episode_id, 'cameras': sorted(videos), 'length': length,
              'sizes': {camera: list(video['size']) for camera, video in sorted(videos.items())},
              'image_stats': {camera: video['stats'] for camera, video in sorted(videos.items())}}
    if repairs:
        record['_conversion'] = {'repairs': repairs}
    return record


def validate_record(record, episode_id):
    """One staged-episode record."""
    if not isinstance(record, dict) or record.get('episode_id') != episode_id:
        raise ValueError(f'{episode_id}: invalid record identity')
    cameras = record.get('cameras')
    if (not isinstance(cameras, list) or not cameras or cameras != sorted(set(cameras))
            or any(c not in CAMERA_IDS for c in cameras)):
        raise ValueError(f'{episode_id}: invalid cameras')
    if type(record.get('length')) is not int or record['length'] < 2:
        raise ValueError(f'{episode_id}: invalid length')
    sizes, stats = record.get('sizes'), record.get('image_stats')
    if (not isinstance(sizes, dict) or set(sizes) != set(cameras)
            or any(tuple(size) not in RESOLUTIONS.values() for size in sizes.values())):
        raise ValueError(f'{episode_id}: invalid camera resolutions')
    if not isinstance(stats, dict) or set(stats) != set(cameras):
        raise ValueError(f'{episode_id}: invalid image statistics')
    for value in stats.values():
        if (not isinstance(value, dict) or type(value.get('count')) is not int or value['count'] < 1
                or any(not isinstance(value.get(k), list) or len(value[k]) != 3
                       or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value[k])
                       for k in ('min', 'max', 'sum', 'sumsq'))):
            raise ValueError(f'{episode_id}: invalid image statistics')


def file_sizes(directory, record):
    """Exact file set of one staged episode with sizes; cheap enough for resume checks."""
    expected = {FRAMES, *(f'{camera}.mp4' for camera in record['cameras'])}
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError(f'{directory}: invalid episode directory')
    files = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                raise ValueError(f'{directory / entry.name}: invalid episode file')
            files[entry.name] = info.st_size
    if set(files) != expected:
        raise ValueError(f'{directory}: episode files differ from record')
    return files


def validate_episode(directory, record):
    """Check a staged episode directory against the contract; return its file sizes."""
    import av
    validate_record(record, directory.name)
    sizes = file_sizes(directory, record)
    table = pq.read_table(directory / FRAMES)
    if (table.schema.names != FRAME_SCHEMA.names
            or any(table.schema.field(f.name).type != f.type for f in FRAME_SCHEMA)
            or any(c.null_count for c in table.columns) or table.num_rows != record['length']):
        raise ValueError(f'{directory}: invalid frame table schema, nulls or length')
    try:
        check_state(table[STATE_KEY].to_pylist())
    except ValueError as exc:
        raise ValueError(f'{directory}: {exc}') from exc
    if any(not text.strip() for text in table['task'].unique().to_pylist()):
        raise ValueError(f'{directory}: empty instruction')
    for camera in record['cameras']:
        path = directory / f'{camera}.mp4'
        # Converters decode every input frame and flush the encoder; verify the
        # resulting MP4 without decoding it a second time.
        with av.open(str(path)) as reader:
            if len(reader.streams) != 1 or len(reader.streams.video) != 1:
                raise ValueError(f'{path}: expected a video-only file')
            video = reader.streams.video[0]
            if (video.codec_context.name != 'h264' or 'mp4' not in reader.format.name.split(',')
                    or video.codec_context.pix_fmt != 'yuv420p' or video.average_rate != FPS
                    or video.frames != record['length']
                    or [video.width, video.height] != list(record['sizes'][camera])):
                raise ValueError(f'{path}: video format, rate, size or frame count differs from record')
    return sizes


class Moments:
    """Running min/max/mean/std of one feature."""

    def __init__(self):
        self.count, self.low, self.high, self.total, self.squares = 0, None, None, 0., 0.

    def add(self, low, high, total, squares, count):
        low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        self.low = low if self.low is None else np.minimum(self.low, low)
        self.high = high if self.high is None else np.maximum(self.high, high)
        self.total = self.total + np.asarray(total, dtype=np.float64)
        self.squares = self.squares + np.asarray(squares, dtype=np.float64)
        self.count += count

    def add_values(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
        self.add(values.min(axis=0), values.max(axis=0), values.sum(axis=0), (values ** 2).sum(axis=0), len(values))

    def result(self, shape=None):
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.squares / self.count - mean ** 2, 0.))
        return {'min': np.reshape(self.low, shape or self.low.shape).tolist(),
                'max': np.reshape(self.high, shape or self.high.shape).tolist(),
                'mean': np.reshape(mean, shape or mean.shape).tolist(),
                'std': np.reshape(std, shape or std.shape).tolist(), 'count': [self.count]}


def link(source, target):
    """Hard-link a staged video into the dataset; copy where the filesystem cannot link."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def features(cameras, sizes):
    result = {STATE_KEY: {'dtype': 'float32', 'shape': [STATE_DIM], 'names': list(STATE_COMPONENTS)}}
    for camera in sorted(cameras):
        width, height = sizes[camera]
        result[video_key(camera)] = {
            'dtype': 'video', 'shape': [height, width, 3], 'names': ['height', 'width', 'channels'],
            'info': {'video.height': height, 'video.width': width, 'video.codec': 'h264',
                     'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'video.fps': FPS,
                     'video.channels': 3, 'has_audio': False}}
    result['timestamp'] = {'dtype': 'float32', 'shape': [1], 'names': None}
    for name in ('frame_index', 'episode_index', 'index', 'task_index'):
        result[name] = {'dtype': 'int64', 'shape': [1], 'names': None}
    return result


def write_tasks(path, tasks):
    """meta/tasks.parquet as pandas writes it: the task text is the index."""
    pandas = {'index_columns': ['task'],
              'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode',
                                  'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}],
              'columns': [{'name': 'task_index', 'field_name': 'task_index', 'pandas_type': 'int64',
                           'numpy_type': 'int64', 'metadata': None},
                          {'name': 'task', 'field_name': 'task', 'pandas_type': 'unicode',
                           'numpy_type': 'object', 'metadata': None}],
              'creator': {'library': 'pyarrow', 'version': pa.__version__}, 'pandas_version': '2.0.0'}
    schema = pa.schema([pa.field('task_index', pa.int64()), pa.field('task', pa.string())],
                       metadata={'pandas': json.dumps(pandas)})
    pq.write_table(pa.Table.from_arrays([pa.array(list(tasks.values()), type=pa.int64()),
                                         pa.array(list(tasks), type=pa.string())], schema=schema), path)


def build_dataset(root, robot_type, episodes, progress=None):
    """Assemble staged episodes sharing one layout into a LeRobot v3.0 dataset at `root`.

    `episodes` is the ordered list of (record, staged directory). Frame tables
    are merged into shared data files. Each episode keeps its own video file,
    so staged MP4s are linked in place without re-encoding.
    """
    record = episodes[0][0]
    cameras, sizes = record['cameras'], record['sizes']
    if any(r['cameras'] != cameras or r['sizes'] != sizes for r, _ in episodes):
        raise ValueError('Episodes of one dataset must share their camera layout')
    root.mkdir(parents=True)
    tasks, rows, index, writer, file_index, file_rows = {}, [], 0, None, -1, 0
    moments = {name: Moments() for name in (STATE_KEY, 'timestamp', 'frame_index', 'episode_index',
                                             'index', 'task_index', *map(video_key, cameras))}
    try:
        for episode_index, (record, directory) in enumerate(episodes):
            table = pq.read_table(directory / FRAMES)
            length = table.num_rows
            if length != record['length']:
                raise ValueError(f'{directory}: frame table length differs from record')
            texts = table['task'].to_pylist()
            task_index = [tasks.setdefault(text, len(tasks)) for text in texts]
            if writer is None or (file_rows and file_rows + length > DATA_FILE_ROWS):
                if writer is not None:
                    writer.close()
                file_index, file_rows = file_index + 1, 0
                path = root / DATA_PATH.format(chunk_index=file_index // CHUNK_SIZE,
                                               file_index=file_index % CHUNK_SIZE)
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, DATA_SCHEMA)
            frames = np.arange(length, dtype=np.int64)
            columns = {'timestamp': (frames / FPS).astype(np.float32), 'frame_index': frames,
                       'episode_index': np.full(length, episode_index, dtype=np.int64),
                       'index': index + frames, 'task_index': np.array(task_index, dtype=np.int64)}
            # One row group per episode, so readers can fetch an episode without the rest of the file.
            writer.write_table(pa.Table.from_arrays(
                [table[STATE_KEY].combine_chunks(), *(pa.array(columns[f.name], type=f.type)
                                                      for f in list(DATA_SCHEMA)[1:])], schema=DATA_SCHEMA))
            moments[STATE_KEY].add_values(np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float64))
            for name, values in columns.items():
                moments[name].add_values(values)
            row = {'episode_index': episode_index, 'episode_id': record['episode_id'],
                   'tasks': list(dict.fromkeys(texts)), 'length': length,
                   'dataset_from_index': index, 'dataset_to_index': index + length,
                   'data/chunk_index': file_index // CHUNK_SIZE, 'data/file_index': file_index % CHUNK_SIZE,
                   'meta/episodes/chunk_index': episode_index // EPISODE_FILE_ROWS // CHUNK_SIZE,
                   'meta/episodes/file_index': episode_index // EPISODE_FILE_ROWS % CHUNK_SIZE}
            for camera in cameras:
                key = video_key(camera)
                chunk, file = episode_index // CHUNK_SIZE, episode_index % CHUNK_SIZE
                link(directory / f'{camera}.mp4', root / VIDEO_PATH.format(
                    video_key=key, chunk_index=chunk, file_index=file))
                row.update({f'videos/{key}/chunk_index': chunk, f'videos/{key}/file_index': file,
                            f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': length / FPS})
                image = record['image_stats'][camera]
                moments[key].add(image['min'], image['max'], image['sum'], image['sumsq'], image['count'])
            rows.append(row)
            index += length
            file_rows += length
            if progress is not None:
                progress.update()
    finally:
        if writer is not None:
            writer.close()
    for start in range(0, len(rows), EPISODE_FILE_ROWS):
        number = start // EPISODE_FILE_ROWS
        path = root / EPISODES_PATH.format(chunk_index=number // CHUNK_SIZE, file_index=number % CHUNK_SIZE)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows[start:start + EPISODE_FILE_ROWS]), path)
    write_tasks(root / 'meta/tasks.parquet', tasks)
    stats = {name: value.result((3, 1, 1) if name.startswith('observation.images.') else None)
             for name, value in moments.items()}
    for camera in cameras:
        # Image statistics are sampled per pixel; LeRobot counts images. Quantiles are not computed.
        stats[video_key(camera)]['count'] = [index]
    (root / 'meta/stats.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    info = {'codebase_version': 'v3.0', 'robot_type': robot_type, 'total_episodes': len(rows),
            'total_frames': index, 'total_tasks': len(tasks), 'chunks_size': CHUNK_SIZE,
            'data_files_size_in_mb': 100, 'video_files_size_in_mb': 200, 'fps': FPS,
            'splits': {'train': f'0:{len(rows)}'}, 'data_path': DATA_PATH, 'video_path': VIDEO_PATH,
            'features': features(cameras, sizes)}
    (root / 'meta/info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return info


def is_dataset(path):
    return (path / 'meta/info.json').is_file()


def discover_datasets(paths):
    """Dataset roots under the given paths: a root, a source folder of layouts, or the export root."""
    roots = []
    for path in paths:
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f'{path}: dataset directory does not exist')
        candidates = [path] if is_dataset(path) else []
        for depth in range(2):
            if candidates:
                break
            level = sorted(p for p in path.glob('/'.join('*' * (depth + 1))) if p.is_dir())
            candidates = [p for p in level if is_dataset(p)]
        if not candidates:
            raise ValueError(f'{path}: no exported datasets (meta/info.json) found')
        for candidate in candidates:
            if candidate.resolve() not in roots:
                roots.append(candidate.resolve())
    return roots


class Dataset:
    """Read access to one exported dataset."""

    def __init__(self, root):
        self.root = root
        try:
            self.info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'{root}: not an exported dataset: {exc}') from exc
        if not isinstance(self.info, dict) or not isinstance(self.info.get('features'), dict):
            raise ValueError(f'{root}: not an exported dataset: meta/info.json is not a dataset description')
        feature = self.info['features'].get(STATE_KEY, {})
        if (self.info.get('codebase_version') != 'v3.0' or self.info.get('fps') != FPS
                or feature.get('shape') != [STATE_DIM] or feature.get('names') != list(STATE_COMPONENTS)):
            raise ValueError(f'{root}: not a contract dataset (LeRobot v3.0, {FPS} fps, {STATE_DIM}-d state)')
        self.cameras = [camera for camera in CAMERA_IDS if video_key(camera) in self.info['features']]
        if not self.cameras:
            raise ValueError(f'{root}: no contract cameras')
        table = pq.read_table(root / 'meta/tasks.parquet', columns=['task_index', 'task'])
        self.tasks = dict(zip(table['task_index'].to_pylist(), table['task'].to_pylist()))
        self.metadata_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
        episodes = []
        for path in self.metadata_files:
            episodes += pq.read_table(path).to_pylist()
        episodes.sort(key=lambda row: row['episode_index'])
        identifiers = [row['episode_id'] for row in episodes]
        if (not episodes or [row['episode_index'] for row in episodes] != list(range(len(episodes)))
                or len(episodes) != self.info.get('total_episodes') or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(i, str) or not EPISODE_ID.fullmatch(i) for i in identifiers)):
            raise ValueError(f'{root}: invalid, duplicate or missing episode metadata')
        self.episodes = episodes

    def data_path(self, episode):
        return self.root / self.info['data_path'].format(
            chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])

    def video(self, episode, camera):
        """(path, index of the episode's first frame inside that file)."""
        key = video_key(camera)
        path = self.root / self.info['video_path'].format(
            video_key=key, chunk_index=episode[f'videos/{key}/chunk_index'],
            file_index=episode[f'videos/{key}/file_index'])
        return path, round(episode[f'videos/{key}/from_timestamp'] * FPS)

    def frames(self, episode):
        """(N x 26 float64 state, per-frame instruction text) of one episode."""
        table = pq.read_table(self.data_path(episode), columns=[STATE_KEY, 'frame_index', 'task_index'],
                              filters=[('episode_index', '=', episode['episode_index'])])
        if table.num_rows != episode['length'] or table['frame_index'].to_pylist() != list(range(table.num_rows)):
            raise ValueError(f"{episode['episode_id']}: frame rows differ from episode metadata")
        state = check_state(table[STATE_KEY].to_pylist())
        return state, [self.tasks[i] for i in table['task_index'].to_pylist()]


def frame_times(count):
    """Frame times of an episode in ns: frame k is at k / 30 s."""
    return np.array([frame_ns(k) for k in range(count)], dtype=np.int64)


def instruction_intervals(texts):
    """Per-frame instruction text -> [{start_ns, end_ns, text}] runs; the last run ends at the last frame."""
    times = frame_times(len(texts))
    starts = [0] + [k for k in range(1, len(texts)) if texts[k] != texts[k - 1]]
    return [{'start_ns': int(times[start]), 'end_ns': int(times[end if end < len(texts) else -1]),
             'text': texts[start]} for start, end in zip(starts, starts[1:] + [len(texts)])]
