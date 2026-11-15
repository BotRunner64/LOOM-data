"""Video decoding and H.264 encoding shared by the converters.

MP4 files only carry frames at an arbitrary container rate; acquisition
times live in the Parquet frame index (export contract, section 5).
"""
from contextlib import contextmanager
from fractions import Fraction
import math
import os
from pathlib import Path
import shutil

from loom_data.sources.policy import SourceDataError


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
    if os.environ.get('LOOM_DATA_SERIAL_CODECS') == '1':
        context.thread_count = 1
        context.thread_type = 'SLICE'
        if context.name == 'libx264':
            context.options = {**context.options, 'x264-params':
                               'threads=1:lookahead-threads=1:sync-lookahead=0'}


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


def transcode_video(source, output, expected_frames):
    import av

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


def extract_video_segment(video, destination, expected_frames, fps):
    """Decode from a preceding keyframe, select display frames, and encode H.264."""
    import av

    start, end = video['from_timestamp'], video['to_timestamp']
    rate = Fraction(str(fps))
    tolerance = min(1e-3, .01 / fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(video['path']) as reader, av.open(str(destination), 'w') as writer:
        if len(reader.streams.video) != 1:
            raise ValueError(f'{video["path"]}: expected one video stream')
        source = reader.streams.video[0]
        configure_codec(source.codec_context)
        reader.seek(math.floor(start / source.time_base), stream=source, backward=True, any_frame=False)
        encoder = writer.add_stream('libx264', rate=rate)
        encoder.width, encoder.height = source.width, source.height
        encoder.pix_fmt = 'yuv420p'
        encoder.options = {'crf': '18', 'preset': 'fast'}
        configure_codec(encoder.codec_context)
        count = 0
        for frame in reader.decode(source):
            timestamp = frame.time
            if timestamp is None:
                raise ValueError(f'{video["path"]}: video frame has no timestamp')
            if timestamp < start - tolerance:
                continue
            if timestamp >= end - tolerance:
                break
            if (frame.width, frame.height) != (encoder.width, encoder.height):
                raise ValueError(f'{video["path"]}: video resolution changed')
            # Reject missing/extra frames even when their total count happens to match.
            if abs(timestamp - (start + count / fps)) > tolerance:
                raise ValueError(f'{video["path"]}: noncontiguous video segment at frame {count}')
            frame = frame.reformat(format='yuv420p')
            frame.pts, frame.time_base = count, 1 / rate
            for packet in encoder.encode(frame):
                writer.mux(packet)
            count += 1
        if count != expected_frames:
            raise ValueError(f'{video["path"]}: expected {expected_frames} frames, decoded {count}')
        for packet in encoder.encode():
            writer.mux(packet)
