"""Video decoding and synchronized H.264 encoding shared by the converters.

Every exported video is constant 30 fps at the contract resolution of its
aspect ratio: output frame k is the image of frame-table row k (export
contract, sections 3 and 5).
"""
from contextlib import contextmanager
from fractions import Fraction
import math
import os
from pathlib import Path

import numpy as np

from loom_data import contract, sync

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


class FrameSink:
    """Encode source frames, given in display order, onto the 30 fps frame sequence.

    `times` are the acquisition times of the source frames. `grid` holds the
    frame times; each takes the nearest source frame. `grid=None` passes a
    source that already is the frame sequence through one to one.
    """
    def __init__(self, path, times, grid=None):
        import av
        self.times = [int(t) for t in times]
        self.select = (np.arange(len(self.times)) if grid is None else sync.nearest(self.times, grid))
        self.count = self.written = 0
        self.stream = self.source_size = None
        self.moments = contract.Moments()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output = av.open(str(path), 'w')

    def add(self, frame):
        index = self.count
        self.count += 1
        if index >= len(self.times):
            raise SourceDataError('video_frame_count', f'Decoded more than the expected {len(self.times)} frames')
        if self.stream is None:
            self.source_size = (frame.width, frame.height)
            self.stream = self.output.add_stream('libx264', rate=sync.FPS)
            self.stream.width, self.stream.height = contract.export_size(*self.source_size)
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {'crf': '18', 'preset': 'fast'}
            configure_codec(self.stream.codec_context)
        elif (frame.width, frame.height) != self.source_size:
            raise SourceDataError('invalid_video', 'Video resolution changed within episode')
        scaled = None
        while self.written < len(self.select) and self.select[self.written] == index:
            if scaled is None:
                scaled = frame.reformat(width=self.stream.width, height=self.stream.height, format='yuv420p')
            if self.written % sync.FPS == 0:  # one frame per second feeds the image statistics
                pixels = scaled.reformat(width=160, height=max(2, self.stream.height // 4),
                                         format='rgb24').to_ndarray().reshape(-1, 3) / 255.
                self.moments.add_values(pixels)
            scaled.pts = self.written
            scaled.time_base = Fraction(1, sync.FPS)
            for packet in self.stream.encode(scaled):
                self.output.mux(packet)
            self.written += 1

    def finish(self):
        """Flush and close; return the exported size and image statistics for the episode record."""
        if self.count != len(self.times) or self.written != len(self.select) or not self.written:
            raise SourceDataError('video_frame_count',
                                  f'Decoded {self.count} frames; expected {len(self.times)}')
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()
        m = self.moments
        return {'size': [self.stream.width, self.stream.height],
                'stats': {'min': m.low.tolist(), 'max': m.high.tolist(), 'sum': m.total.tolist(),
                          'sumsq': m.squares.tolist(), 'count': m.count}}

    def close(self):
        self.output.close()


def encode_frames(path, frames, times, grid=None):
    """Encode an iterable of display-order frames through a FrameSink."""
    sink = FrameSink(path, times, grid)
    try:
        for frame in frames:
            sink.add(frame)
        return sink.finish()
    finally:
        sink.close()


def create_video_decoder(codec):
    import av
    if codec not in ('h264', 'h265'):
        raise SourceDataError('unsupported_video_codec', f'Unsupported source video codec: {codec}')
    decoder = av.CodecContext.create('hevc' if codec == 'h265' else codec, 'r')
    configure_codec(decoder)
    decoder.open()
    return decoder


class VideoWriter:
    """Decode one Annex B access unit per MCAP message and encode the synchronized video.

    `times` lists the acquisition time of every message of this camera, in
    message order, collected before decoding. Only compressed samples are
    queued, never a whole decoded video.
    """
    def __init__(self, path: Path, codec: str, times, grid):
        self.codec = codec
        self.decoder = create_video_decoder(codec)
        self.source_times = [int(t) for t in times]
        self.sink = FrameSink(path, sorted(self.source_times), grid)
        self.index = 0
        self.seen = set()

    def add(self, data, timestamp, codec):
        import av
        if codec != self.codec or not data:
            raise SourceDataError('invalid_video', 'Video codec changed or empty access unit')
        if self.index >= len(self.source_times) or self.source_times[self.index] != timestamp:
            raise SourceDataError('video_frame_mapping', 'Camera messages changed between passes')
        packet = av.Packet(data)
        packet.pts = self.index
        packet.time_base = Fraction(1, 30)
        self.index += 1
        for frame in decode_source(self.decoder, packet):
            self._frame(frame)

    def _frame(self, frame):
        index = frame.pts
        if index is None or index in self.seen or not 0 <= index < len(self.source_times):
            raise SourceDataError('video_frame_mapping', 'Cannot map decoded frame to a unique MCAP sample')
        self.seen.add(index)
        if self.sink.count >= len(self.sink.times) or self.source_times[index] != self.sink.times[self.sink.count]:
            raise SourceDataError('invalid_timestamp', 'Video display order has decreasing acquisition timestamps')
        self.sink.add(frame)

    def finish(self):
        for frame in decode_source(self.decoder):
            self._frame(frame)
        if self.index != len(self.source_times):
            raise SourceDataError('video_frame_count', 'Video frame count differs from MCAP sample count')
        return self.sink.finish()

    def close(self):
        self.sink.close()


def file_frames(source):
    """Display-order frames of a single-video-stream file."""
    with source_video(source) as reader:
        if len(reader.streams.video) != 1:
            raise SourceDataError('invalid_video', f'{source}: expected one video stream')
        configure_codec(reader.streams.video[0].codec_context)
        yield from decode_source(reader, container=True)


def transcode_video(source, output, times, grid=None):
    """Encode a whole source file whose frame i was acquired at times[i]."""
    if len(times) < 1:
        raise ValueError('Expected a nonempty source video')
    try:
        return encode_frames(output, file_frames(source), times, grid)
    except SourceDataError as exc:
        raise SourceDataError(exc.reason, f'{source}: {exc}', reasons=exc.reasons) from exc


def segment_frames(video, expected_frames, fps):
    """Display frames of one episode inside a shared video file, checked for contiguity."""
    import av

    start, end = video['from_timestamp'], video['to_timestamp']
    tolerance = min(1e-3, .01 / fps)
    with av.open(video['path']) as reader:
        if len(reader.streams.video) != 1:
            raise ValueError(f'{video["path"]}: expected one video stream')
        source = reader.streams.video[0]
        configure_codec(source.codec_context)
        reader.seek(math.floor(start / source.time_base), stream=source, backward=True, any_frame=False)
        count = 0
        for frame in reader.decode(source):
            timestamp = frame.time
            if timestamp is None:
                raise ValueError(f'{video["path"]}: video frame has no timestamp')
            if timestamp < start - tolerance:
                continue
            if timestamp >= end - tolerance:
                break
            # Reject missing/extra frames even when their total count happens to match.
            if abs(timestamp - (start + count / fps)) > tolerance:
                raise ValueError(f'{video["path"]}: noncontiguous video segment at frame {count}')
            yield frame
            count += 1
        if count != expected_frames:
            raise ValueError(f'{video["path"]}: expected {expected_frames} frames, decoded {count}')


def extract_video_segment(video, destination, times, fps, grid=None):
    """Encode one episode's segment of a shared video file; segment frame i was acquired at times[i]."""
    return encode_frames(destination, segment_frames(video, len(times), fps), times, grid)
