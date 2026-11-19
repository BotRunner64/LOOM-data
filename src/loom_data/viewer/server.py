"""Local viewer for the exported dataset contract."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from fractions import Fraction
import os
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from loom_data.contract import STATE_NAMES, STATE_SLICES, Dataset, frame_times, instruction_intervals, is_pose

HERE = Path(__file__).resolve().parent
EPISODE_PAGE_SIZE = 100
MAX_EPISODE_PAGE_SIZE = 200


class EpisodeIndex:
    """Episode lookup of one exported dataset; the export tree is only read."""

    def __init__(self, root):
        self.dataset = Dataset(root)
        self.by_id = {episode['episode_id']: episode for episode in self.dataset.episodes}
        self.total = len(self.by_id)

    def page(self, query):
        offset, limit = episode_page_bounds(query)
        return {'episodes': [e['episode_id'] for e in self.dataset.episodes[offset:offset + limit]],
                'total': self.total, 'offset': offset, 'limit': limit}

    def record(self, episode_id):
        return self.by_id[episode_id]

    def episode_id_at(self, episode_index):
        episodes = self.dataset.episodes
        return episodes[episode_index]['episode_id'] if 0 <= episode_index < len(episodes) else None

    def close(self):
        pass


def episode_page_bounds(query):
    offset = int(query.get('offset', ['0'])[0])
    limit = int(query.get('limit', [str(EPISODE_PAGE_SIZE)])[0])
    if offset < 0 or not 1 <= limit <= MAX_EPISODE_PAGE_SIZE:
        raise ValueError('Invalid episode page offset or limit')
    return offset, limit


def read_anomalies(root, path, episode_id_at=None):
    """Resolve the anomaly record contract without importing detection rules."""
    if episode_id_at is None:
        episode_id_at = EpisodeIndex(root).episode_id_at
    records = []
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('Expected an anomaly object')
                for field in ('index', 'start_ns', 'end_ns'):
                    if type(record.get(field)) is not int or record[field] < 0:
                        raise ValueError(f'{field} must be a nonnegative integer')
                episode_id = episode_id_at(record['index'])
                if episode_id is None:
                    raise ValueError('index does not address an episode of the dataset')
                if record.get('stream') not in STATE_NAMES:
                    raise ValueError('Invalid stream')
                if record['end_ns'] < record['start_ns']:
                    raise ValueError('end_ns must be >= start_ns')
                if not isinstance(record.get('reason'), str) or not record['reason'].strip():
                    raise ValueError('reason must be a nonempty string')
                records.append({key: record[key] for key in
                                ('index', 'stream', 'start_ns', 'end_ns', 'reason')})
                records[-1]['episode_id'] = episode_id
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    return records


def read_episode(dataset, episode):
    """One episode as the page consumes it: every stream on the shared 30 fps frame clock."""
    state, texts = dataset.frames(episode)
    times = frame_times(len(state)).tolist()
    streams = {}
    for name in STATE_NAMES:
        values = state[:, STATE_SLICES[name]]
        streams[name] = {'times': times, 'values': (values if is_pose(name) else values[:, 0]).tolist()}
    cameras = {}
    for camera in dataset.cameras:
        if not dataset.video(episode, camera)[0].is_file():
            raise ValueError(f'Missing camera video: {camera}')
        cameras[camera] = {'times': times, 'frames': len(times)}
    return {'episode_id': episode['episode_id'], 'instructions': instruction_intervals(texts),
            'streams': streams, 'cameras': cameras, 'end_ns': times[-1]}


class FrameReader:
    """Decode by display-order frame index; exported videos hold one frame per 1/30 s."""

    def __init__(self):
        self.lock = Lock()
        self.path = None
        self.container = None
        self.iterator = None
        self.index = -1
        self.last_frame = None
        self.last_images = {}
        self.seeked = False

    def close(self):
        if self.container is not None:
            self.container.close()
        self.path = self.container = self.iterator = self.last_frame = None
        self.last_images = {}
        self.index = -1
        self.seeked = False

    def _open(self, path):
        self.close()
        import av
        self.container = av.open(str(path))
        stream = self.container.streams.video[0]
        stream.thread_type = 'AUTO'
        # Avoid one decoder spawning a thread for every host CPU. Multi-camera episodes
        # otherwise create hundreds of FFmpeg threads and can exceed small container limits.
        stream.thread_count = min(4, os.cpu_count() or 1)
        self.iterator = self.container.decode(video=0)
        self.path = path

    def _seek(self, index):
        stream = self.container.streams.video[0]
        # Exported MP4s have one display frame per tick of their 30 Hz clock.
        # Other videos retain the sequential, display-order fallback.
        if (getattr(stream, 'average_rate', None) != 30
                or not getattr(stream, 'time_base', None)
                or not hasattr(self.container, 'seek')):
            return False
        origin = stream.start_time or 0
        target = origin + int(Fraction(index, 30) / stream.time_base)
        self.container.seek(target, stream=stream, backward=True)
        self.iterator = self.container.decode(video=0)
        self.index = -1
        self.seeked = True
        return True

    def get(self, path, index, width):
        with self.lock:
            return self._get(path, index, width)

    def _get(self, path, index, width):
        if self.path != path:
            self._open(path)
        restarted = self.index >= 0 and (index < self.index or index - self.index > 120)
        if restarted:
            # Repeated seeks on one FFmpeg decoder retain substantial native buffers.
            # Reopen before discontinuities so scrub traffic cannot grow without bound.
            self._open(path)
        if restarted or index - self.index > 120:
            self._seek(index)
        while self.index < index:
            frame = next(self.iterator)
            if self.seeked:
                stream = self.container.streams.video[0]
                if frame.pts is None or frame.time_base is None:
                    self._open(path)
                    continue
                ordinal = round((frame.pts - (stream.start_time or 0)) * frame.time_base * 30)
                if ordinal < 0 or (self.index >= 0 and ordinal != self.index + 1) or ordinal > index:
                    self._open(path)
                    continue
                self.index = ordinal
            else:
                self.index += 1
            if self.index == index:
                self.last_frame = frame
                self.last_images = {}
                self.seeked = False
        if width not in self.last_images:
            frame = self.last_frame
            if frame.width > width:
                height = max(1, round(frame.height * width / frame.width))
                frame = frame.reformat(width=width, height=height, format='rgb24')
            image = frame.to_image()
            output = BytesIO()
            image.save(output, format='JPEG', quality=80)
            self.last_images[width] = output.getvalue()
        return self.last_images[width]


def serve(root, host, port, anomalies_path=None):
    episode_index = EpisodeIndex(root)
    dataset = episode_index.dataset
    try:
        if anomalies_path is None and (root / 'anomalies.jsonl').is_file():
            anomalies_path = root / 'anomalies.jsonl'
        anomalies = read_anomalies(root, anomalies_path, episode_index.episode_id_at) \
            if anomalies_path is not None else None
    except Exception:
        episode_index.close()
        raise
    readers = {}
    readers_lock = Lock()
    static_files = {
        '/': ((HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8'),
        **{'/' + name: ((HERE / name).read_bytes(), 'text/javascript; charset=utf-8')
           for name in ('pose3d.js', 'state_curves.js', 'navigation.js', 'anomalies.js', 'episodes.js', 'vendor/three.module.min.js',
                        'vendor/three.core.min.js')},
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def reply(self, status, body, content_type):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The browser may cancel an image request while scrubbing.
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in static_files:
                    body, content_type = static_files[parsed.path]
                    self.reply(200, body, content_type)
                    return
                if parsed.path in ('/api/dataset', '/api/episodes'):
                    payload = episode_index.page(query)
                    if parsed.path == '/api/dataset':
                        payload['anomalies'] = anomalies
                elif parsed.path == '/api/episode':
                    episode_id = query['id'][0]
                    payload = read_episode(dataset, episode_index.record(episode_id))
                    if any(record['end_ns'] > payload['end_ns'] for record in (anomalies or [])
                           if record['episode_id'] == episode_id):
                        raise ValueError(f'Anomaly interval exceeds episode duration: {episode_id}')
                elif parsed.path == '/api/frame':
                    episode_id = query['id'][0]
                    camera = query['camera'][0]
                    episode = episode_index.record(episode_id)
                    if camera not in dataset.cameras:
                        raise KeyError(camera)
                    index = int(query['index'][0])
                    if not 0 <= index < episode['length']:
                        raise ValueError('Frame index outside the episode')
                    width = int(query.get('width', ['1280'])[0])
                    if not 64 <= width <= 2048:
                        raise ValueError('Invalid frame width')
                    path, first = dataset.video(episode, camera)
                    index += first
                    with readers_lock:
                        reader = readers.get(path)
                        if reader is None:
                            reader = readers[path] = FrameReader()
                    self.reply(200, reader.get(path, index, width), 'image/jpeg')
                    return
                else:
                    self.reply(404, b'Not found', 'text/plain; charset=utf-8')
                    return
                self.reply(200, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                           'application/json; charset=utf-8')
            except (KeyError, IndexError, ValueError, FileNotFoundError,
                    StopIteration, ImportError) as exc:
                self.reply(400, str(exc).encode('utf-8'), 'text/plain; charset=utf-8')

    try:
        with ThreadingHTTPServer((host, port), Handler) as server:
            print(f'Export viewer: http://{host}:{server.server_port}/')
            server.serve_forever()
    finally:
        for reader in readers.values():
            reader.close()
        episode_index.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data view', description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', type=Path, help='Exported dataset root')
    parser.add_argument('--anomalies', type=Path,
                        help='Anomaly JSONL report (default: <dataset>/anomalies.jsonl if present)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        serve(args.dataset.resolve(), args.host, args.port, args.anomalies)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
