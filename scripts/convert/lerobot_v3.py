"""Local LeRobot v3 frame tables, episode metadata, and frame-accurate video cuts."""
from fractions import Fraction
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from scripts.convert.export_common import configure_codec


def source_path(shard, template, row, prefix, video_key=None):
    indices = [row[f'{prefix}/{name}_index'] for name in ('chunk', 'file')]
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError(f'{shard}: invalid {prefix} chunk/file indices')
    path = shard / template.format(chunk_index=indices[0], file_index=indices[1], video_key=video_key)
    if not path.resolve().is_relative_to(shard.resolve()):
        raise ValueError(f'{shard}: source path escapes shard: {path}')
    return path


class LeRobotV3Reader:
    """Keep metadata and at most one projected frame table for the current shard."""

    def __init__(self, source, cameras, columns, validate, *, check_video_length=True):
        self.columns = list(columns)
        self.source = source
        meta = source / 'meta'
        self.info = json.loads((meta / 'info.json').read_text(encoding='utf-8'))
        validate(self.info)
        self.fps = self.info.get('fps')
        if not isinstance(self.fps, (float, int)) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f'{source}: invalid FPS')
        self.metadata_files = sorted((meta / 'episodes').rglob('*.parquet'))
        if not self.metadata_files:
            raise ValueError(f'{source}: no episode metadata parquet files')
        columns = ['episode_index', 'length', 'dataset_from_index', 'dataset_to_index', 'tasks',
                   'data/chunk_index', 'data/file_index']
        columns += [f'videos/{key}/{field}' for key in cameras for field in
                    ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
        self.episodes = {}
        self.paths = {}
        for path in self.metadata_files:
            for row in pq.read_table(path, columns=columns).to_pylist():
                index, length = row['episode_index'], row['length']
                start, stop = row['dataset_from_index'], row['dataset_to_index']
                if (type(index) is not int or index < 0 or index in self.episodes
                        or type(length) is not int or length < 1
                        or type(start) is not int or type(stop) is not int
                        or start < 0 or stop - start != length):
                    raise ValueError(f'{path}: invalid/duplicate episode identity or length')
                videos = {}
                for key in cameras:
                    prefix = f'videos/{key}'
                    begin, end = row[f'{prefix}/from_timestamp'], row[f'{prefix}/to_timestamp']
                    if (not isinstance(begin, (int, float)) or not isinstance(end, (int, float))
                            or not math.isfinite(begin) or not math.isfinite(end) or begin < 0
                            or end <= begin
                            or (check_video_length and abs((end - begin) * self.fps - length) > .01)):
                        raise ValueError(f'{path}: invalid video range for episode {index}: {key}')
                    videos[key] = {'path': self.path(self.info['video_path'], row, prefix, key),
                                   'from_timestamp': begin, 'to_timestamp': end}
                row['videos'] = videos
                row['data_path'] = self.path(self.info['data_path'], row, 'data')
                self.episodes[index] = row
        if not self.episodes or len(self.episodes) != self.info.get('total_episodes'):
            raise ValueError(f'{source}: episode metadata count disagrees with info.json')
        self.data_path = self.table = None

    def path(self, template, row, prefix, video_key=None):
        indices = (row[f'{prefix}/chunk_index'], row[f'{prefix}/file_index'])
        if any(type(i) is not int or i < 0 for i in indices):
            raise ValueError(f'{self.source}: invalid {prefix} chunk/file indices')
        key = (template, *indices, video_key)
        if key not in self.paths:
            self.paths[key] = str(source_path(self.source, template, row, prefix, video_key))
        return self.paths[key]

    def inputs(self, indices):
        paths = {str(p) for p in self.metadata_files}
        for index in indices:
            row = self.episodes[index]
            paths.add(row['data_path'])
            paths.update(v['path'] for v in row['videos'].values())
        return [Path(p) for p in sorted(paths)]

    def read_rows(self, index):
        row = self.episodes[index]
        path = row['data_path']
        if path != self.data_path:
            # Release the preceding file before loading another, even across file boundaries.
            self.table = None
            self.data_path = None
            self.table = pq.read_table(path, columns=self.columns)
            self.data_path = path
        if len(self.table) == 0:
            raise ValueError(f'{path}: empty frame table')
        start = row['dataset_from_index'] - self.table['index'][0].as_py()
        if start < 0 or start + row['length'] > len(self.table):
            raise ValueError(f'{path}: episode {index} range is outside its data file')
        rows = self.table.slice(start, row['length']).to_pydict()
        return rows


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
