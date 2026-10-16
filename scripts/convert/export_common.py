"""Shared writers for the export contract."""
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import re
import tempfile
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.convert.source_policy import SourceDataError


@contextmanager
def source_video(path):
    """Classify input-open errors without masking output/disk/codec failures."""
    import av
    try:
        reader = av.open(str(path))
    except av.InvalidDataError as exc:
        raise SourceDataError('invalid_video', f'{path}: {exc}') from exc
    with reader:
        yield reader


def decode_source(decoder, packet=None, *, container=False):
    import av
    try:
        yield from decoder.decode(video=0) if container else decoder.decode(packet)
    except av.InvalidDataError as exc:
        raise SourceDataError('invalid_video', str(exc)) from exc


def configure_codec(context):
    """Parallel workers use synchronous single-thread codecs, including lookahead.

    Multiple ABC camera contexts stay open at once. A budget per codec would
    multiply that budget by the camera count; parallelism belongs to workers.
    Standalone converters retain their previous codec defaults.
    """
    if os.environ.get('ROBOT_DATA_HUB_SERIAL_CODECS') == '1':
        context.thread_count = 1
        context.thread_type = 'SLICE'
        if context.name == 'libx264':
            context.options = {**context.options, 'x264-params':
                               'threads=1:lookahead-threads=1:sync-lookahead=0'}


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


def copy_h264_video(source, destination, expected_frames):
    """Copy an already compliant MP4 after checking every display frame.

    Container PTS are deliberately not used as acquisition timestamps.
    """
    import av
    with av.open(str(source)) as reader:
        if (len(reader.streams) != 1 or len(reader.streams.video) != 1
                or reader.streams.video[0].codec_context.name != 'h264'
                or 'mp4' not in reader.format.name.split(',')):
            raise ValueError(f'{source}: expected a video-only H.264 MP4')
        configure_codec(reader.streams.video[0].codec_context)
        count, size = 0, None
        for frame in reader.decode(video=0):
            current_size = (frame.width, frame.height)
            if size is not None and size != current_size:
                raise ValueError(f'{source}: video resolution changed')
            size = current_size
            count += 1
        if count != expected_frames or count == 0:
            raise ValueError(f'{source}: expected {expected_frames} frames, decoded {count}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def relative_times(timestamps, origin):
    values = [int(t) - int(origin) for t in timestamps]
    if not values or values[0] < 0 or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError('Empty, negative or unordered sample timestamps')
    return pa.array(values, type=pa.int64())


def write_state(path, timestamps, values, origin, *, pose=False):
    name = 'pose' if pose else 'openness'
    dtype = pa.list_(pa.float64(), 12) if pose else pa.float32()
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite {name}: {path}')
    if pose:
        if data.shape != (len(timestamps), 12):
            raise ValueError('Expected N x 12 poses')
        validate_rotation_matrices(data[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1))
    elif data.shape != (len(timestamps),) or np.any((data < 0) | (data > 1)):
        raise ValueError('Expected scalar gripper openness in [0,1]')
    schema = pa.schema([pa.field('timestamp_ns', pa.int64(), nullable=False),
                        pa.field(name, dtype, nullable=False)])
    table = pa.Table.from_arrays([relative_times(timestamps, origin),
                                 pa.array(data.tolist(), type=dtype)], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_video_index(path, timestamps, origin):
    schema = pa.schema([pa.field('frame_index', pa.int64(), nullable=False),
                        pa.field('timestamp_ns', pa.int64(), nullable=False)])
    pq.write_table(pa.Table.from_arrays([
        pa.array(range(len(timestamps)), type=pa.int64()),
        relative_times(timestamps, origin)], schema=schema), path)


def create_video_decoder(codec):
    import av
    if codec not in ('h264', 'h265'):
        raise SourceDataError('unsupported_video_codec', f'Unsupported source video codec: {codec}')
    decoder = av.CodecContext.create('hevc' if codec == 'h265' else codec, 'r')
    configure_codec(decoder)
    decoder.open()
    return decoder


class VideoWriter:
    """Decode one Annex B access unit per sample, preserving PTS through reorder.

    MP4 uses an arbitrary 30 Hz container clock; actual times live in Parquet.
    Only compressed samples/timestamps are queued, never a whole decoded video.
    """
    def __init__(self, path: Path, codec: str):
        import av
        if codec not in ('h264', 'h265'):
            raise SourceDataError('unsupported_video_codec', f'Unsupported source video codec: {codec}')
        self.codec = codec
        self.decoder = create_video_decoder(codec)
        self.output = av.open(str(path), 'w')
        self.stream = None
        self.source_times = []
        self.timestamps = []
        self.seen = set()

    def add(self, data, timestamp, codec):
        import av
        if codec != self.codec or not data:
            raise SourceDataError('invalid_video', 'Video codec changed or empty access unit')
        packet = av.Packet(data)
        packet.pts = len(self.source_times)
        packet.time_base = Fraction(1, 30)
        self.source_times.append(timestamp)
        for frame in decode_source(self.decoder, packet):
            self._frame(frame)

    def _frame(self, frame):
        index = frame.pts
        if index is None or index in self.seen or not 0 <= index < len(self.source_times):
            raise SourceDataError('video_frame_mapping', 'Cannot map decoded frame to a unique MCAP sample')
        timestamp = self.source_times[index]
        if self.timestamps and timestamp < self.timestamps[-1]:
            raise SourceDataError('invalid_timestamp', 'Video display order has decreasing acquisition timestamps')
        self.seen.add(index)
        if self.stream is None:
            self.stream = self.output.add_stream('libx264', rate=30)
            self.stream.width, self.stream.height = frame.width, frame.height
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {'crf': '18', 'preset': 'fast'}
            configure_codec(self.stream.codec_context)
        elif (frame.width, frame.height) != (self.stream.width, self.stream.height):
            raise SourceDataError('invalid_video', 'Video resolution changed within episode')
        frame.pts = len(self.timestamps)
        frame.time_base = Fraction(1, 30)
        self.timestamps.append(timestamp)
        for packet in self.stream.encode(frame):
            self.output.mux(packet)

    def finish(self):
        for frame in decode_source(self.decoder):
            self._frame(frame)
        if not self.timestamps or len(self.timestamps) != len(self.source_times):
            raise SourceDataError('video_frame_count', 'Video frame count differs from MCAP sample count')
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()

    def close(self):
        self.output.close()


def pose_matrix(position, quaternion):
    """Source position in metres and quaternion in xyzw order."""
    position, quat = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError('Invalid or nonfinite source pose')
    norm = np.linalg.norm(quat)
    if norm < 1e-6 or abs(norm - 1) > 0.1:
        raise ValueError('Invalid source quaternion')
    x, y, z, w = quat / norm
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    transform[:3, 3] = position
    return transform


def tool_transform(offset, rpy):
    """Fixed source-tool -> contract TCP transform; translation in source axes."""
    from scripts.robot.kinematics import rpy_matrix
    offset, rpy = np.asarray(offset, dtype=float), np.asarray(rpy, dtype=float)
    if offset.shape != (3,) or rpy.shape != (3,) or not np.isfinite([offset, rpy]).all():
        raise ValueError('Tool offset and RPY require three finite values each')
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_matrix(rpy)
    matrix[:3, 3] = offset
    return matrix


def validate_rotation_matrices(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    identity = np.eye(3)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), identity, atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid rotation matrix')


def rotation9d(transform):
    """Position followed by the complete rotation matrix in column order."""
    return np.concatenate((transform[:3, 3], transform[:3, :3].T.reshape(-1))).tolist()


def normalized_gripper(value, closed, opened):
    if not all(math.isfinite(v) for v in (value, closed, opened)) or closed == opened:
        raise ValueError('Nonfinite gripper value or invalid open/closed range')
    return min(1., max(0., (float(value) - closed) / (opened - closed)))


def export_episodes(items, output, converter):
    """Publish a new dataset atomically after all selected episodes validate.

    ``items`` contains (stable episode_id, source) pairs. The callback accepts
    (source, destination) and returns the contract metadata record.
    """
    from scripts.convert.export_validation import validate_episode
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    names = [name for name, _ in items]
    if (not names or len(set(names)) != len(names)
            or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', name) for name in names)):
        raise ValueError('Expected nonempty, unique, safe episode IDs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as metadata:
            for i, (episode_id, source) in enumerate(items, 1):
                print(f'[{i}/{len(items)}] {episode_id}', flush=True)
                destination = staged / 'episodes' / episode_id
                record = converter(source, destination)
                validate_episode(destination, record)
                metadata.write(json.dumps(record, ensure_ascii=False) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {len(items)} episodes to {output}', flush=True)


def transcode_video(source, output, expected_frames):
    import av
    from fractions import Fraction

    if expected_frames < 1:
        raise ValueError('Expected a nonempty source video')
    count = 0
    with source_video(source) as reader, av.open(str(output), 'w') as writer:
        if len(reader.streams.video) != 1:
            raise SourceDataError('invalid_video', f'{source}: expected one video stream')
        stream = writer.add_stream('libx264', rate=30)
        stream.options = {'crf': '18', 'preset': 'fast'}
        configure_codec(reader.streams.video[0].codec_context)
        configure_codec(stream.codec_context)
        for frame in decode_source(reader, container=True):
            if count == 0:
                stream.width, stream.height = frame.width, frame.height
                stream.pix_fmt = 'yuv420p'
            elif (frame.width, frame.height) != (stream.width, stream.height):
                raise SourceDataError('invalid_video', f'{source}: video resolution changed')
            frame.pts = count
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                writer.mux(packet)
            count += 1
        if count != expected_frames:
            raise SourceDataError('video_frame_count', f'{source}: decoded {count} frames; expected {expected_frames}')
        for packet in stream.encode():
            writer.mux(packet)


def task_instructions(task_indices, tasks, times):
    if not times or times[-1] <= times[0]:
        raise ValueError('Empty or zero-duration episode')
    transitions = {}
    previous = None
    for index, timestamp in zip(task_indices, times):
        if type(index) is not int or index not in tasks:
            raise ValueError(f'Unknown task_index: {index!r}')
        text = tasks[index]
        if text != previous:
            transitions[timestamp] = text
            previous = text
    starts = sorted(transitions)
    return [{'start_ns': start - times[0], 'end_ns': end - times[0],
             'text': transitions[start]}
            for start, end in zip(starts, starts[1:] + [times[-1]]) if end > start]
