import json
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from support import IDENTITY_POSE, make_dataset
from loom_data import contract
from loom_data.sync import frame_ns
from loom_data.viewer.server import (
    EpisodeIndex, FrameReader, read_anomalies, read_episode, serve,
)

TIMES = [0, 33_333_333, 66_666_667, 100_000_000]


def still(*episode_ids):
    """Minimal two-frame episodes."""
    return {episode_id: {'left_gripper': np.zeros(2)} for episode_id in episode_ids}


class ExportViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.template = Path(temporary.name).resolve() / 'export'
        cls.poses = np.tile(IDENTITY_POSE, (4, 1))
        cls.poses[:, 0] = [0, .5, 1, 1.5]
        make_dataset(cls.template, {'sample': {
            'left_eef': cls.poses, 'right_gripper': [0, .25, .5, 1], 'tasks': ['pick', 'pick', 'place', 'place']}})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'export'
        shutil.copytree(self.template, self.root)

    def rebuild(self, episodes):
        """Replace the fixture by a dataset with the given episodes."""
        shutil.rmtree(self.root)
        return make_dataset(self.root, episodes)

    def rewrite_episodes(self, change):
        path = self.root / 'meta/episodes/chunk-000/file-000.parquet'
        rows = change(pq.read_table(path).to_pylist())
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)

    def report(self, records):
        path = self.root / 'anomalies.jsonl'
        path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        return path

    def request(self, path, report=None):
        results = []

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def serve_forever(self):
                for item in [path] if isinstance(path, str) else path:
                    handler = self.handler.__new__(self.handler)
                    handler.path = item
                    handler.reply = lambda *args: results.append(args)
                    handler.do_GET()

        with patch('loom_data.viewer.server.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765, report)
        return results[0] if isinstance(path, str) else results

    def test_anomalies_resolve_episode_indices_and_preserve_unknown_reasons(self):
        self.rebuild(still('sample', 'other'))
        records = [
            {'episode_index': 1, 'reason': 'arbitrary_future_rule', 'stream': 'left_eef',
             'start_frame': 10, 'end_frame': 20},
            {'episode_index': 0, 'reason': '<uninterpreted reason>', 'stream': None,
             'start_frame': 30, 'end_frame': 31},
        ]
        path = self.report(records + records[:1])
        result = read_anomalies(self.root, path)
        self.assertEqual(result, [dict(records[0], episode_id='other'),
                                  dict(records[1], episode_id='sample'),
                                  dict(records[0], episode_id='other')])
        # Blank report lines are skipped but still counted when locating an error.
        path.write_text('\n' + json.dumps(records[1]) + '\n\n' + json.dumps({**records[0], 'episode_index': 2}) + '\n')
        with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:4: episode_index does not address an episode'):
            read_anomalies(self.root, path)

    def test_invalid_anomaly_records_report_source_line(self):
        valid = {'episode_index': 0, 'reason': 'custom', 'stream': 'left_eef', 'start_frame': 10,
                 'end_frame': 20}
        for change in ({'episode_index': True}, {'episode_index': 1}, {'episode_index': -1}, {'stream': 'camera'},
                       {'start_frame': -1}, {'end_frame': 9}, {'end_frame': 10}, {'end_frame': 20.0},
                       {'reason': ''}, {'reason': None}):
            with self.subTest(change=change):
                path = self.report([valid, {**valid, **change}])
                with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:2:'):
                    read_anomalies(self.root, path)
        path = self.report([valid, {key: value for key, value in valid.items() if key != 'stream'}])
        with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:2: Invalid stream'):
            read_anomalies(self.root, path)
        for malformed in ('[]', '{', '{}'):
            path.write_text(malformed)
            with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:1:'):
                read_anomalies(self.root, path)

    def test_optional_empty_and_explicit_reports(self):
        status, body, _ = self.request('/api/dataset')
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)['anomalies'])
        self.report([])
        self.assertEqual(json.loads(self.request('/api/dataset')[1])['anomalies'], [])
        record = {'episode_index': 0, 'reason': 'another_rule', 'stream': 'left_eef', 'start_frame': 1,
                  'end_frame': 2}
        report = self.report([record]).rename(self.root / 'custom.jsonl')
        payload = json.loads(self.request('/api/dataset', report)[1])
        self.assertEqual(payload['anomalies'], [dict(record, episode_id='sample')])
        with self.assertRaises(FileNotFoundError):
            self.request('/api/dataset', self.root / 'missing.jsonl')

    def test_serves_state_curve_module(self):
        status, body, content_type = self.request('/state_curves.js')
        self.assertEqual(status, 200)
        self.assertIn(b'export class StateCurves', body)
        self.assertEqual(content_type, 'text/javascript; charset=utf-8')
        status, body, content_type = self.request('/navigation.js')
        self.assertEqual(status, 200)
        self.assertIn(b'export function adjacentSampleTime', body)
        self.assertEqual(content_type, 'text/javascript; charset=utf-8')

    def test_episode_pages_bound_initial_payload_and_preserve_dataset_order(self):
        # Ids are deliberately not in sorted order: pages follow episode_index.
        identifiers = [f'episode_{(index * 7) % 251:04d}' for index in range(251)]
        self.rebuild(still(*identifiers))
        initial, middle, last, beyond = (json.loads(body) for _, body, _ in self.request(
            ['/api/dataset', '/api/episodes?offset=100', '/api/episodes?offset=200', '/api/episodes?offset=251']))
        self.assertEqual(initial['episodes'], identifiers[:100])
        self.assertEqual((initial['total'], initial['offset'], initial['limit']), (251, 0, 100))
        self.assertIsNone(initial['anomalies'])
        self.assertEqual(middle['episodes'], identifiers[100:200])
        self.assertNotIn('anomalies', middle)
        self.assertEqual(last['episodes'], identifiers[200:])
        self.assertEqual(beyond['episodes'], [])

    def test_episode_index_follows_episode_index_order(self):
        dataset = self.rebuild(still('first', 'second', 'third'))
        index = EpisodeIndex(self.root)
        self.addCleanup(index.close)
        self.assertEqual(index.total, 3)
        self.assertEqual(index.page({'offset': ['1'], 'limit': ['1']}), {
            'episodes': ['second'], 'total': 3, 'offset': 1, 'limit': 1,
        })
        self.assertEqual([index.episode_id_at(i) for i in range(3)], ['first', 'second', 'third'])
        self.assertIsNone(index.episode_id_at(3))
        self.assertIsNone(index.episode_id_at(-1))
        self.assertEqual(index.record('third'), dataset.episodes[2])
        self.assertEqual(index.record('third')['episode_index'], 2)
        with self.assertRaises(KeyError):
            index.record('fourth')

    def test_episode_pages_respect_custom_limits(self):
        self.rebuild(still('Alpha_1', 'other', 'alpha_2', 'alpha_3'))
        page = json.loads(self.request('/api/episodes?offset=1&limit=1')[1])
        self.assertEqual(page, {'episodes': ['other'], 'total': 4, 'offset': 1, 'limit': 1})
        empty = json.loads(self.request('/api/episodes?offset=4&limit=1')[1])
        self.assertEqual(empty['episodes'], [])
        self.assertEqual(empty['total'], 4)

    def test_rejects_invalid_episode_page_parameters(self):
        paths = [path + '?' + query for path in ('/api/dataset', '/api/episodes')
                 for query in ('offset=-1', 'offset=abc', 'limit=0', 'limit=201', 'limit=abc')]
        for path, (status, _, _) in zip(paths, self.request(paths)):
            with self.subTest(path=path):
                self.assertEqual(status, 400)
        self.assertEqual(self.request('/api/episodes?limit=200')[0], 200)

    def test_rejects_report_beyond_episode_length_when_loading_episode(self):
        # A record may end one past the last frame, but no further.
        record = {'episode_index': 0, 'reason': 'custom', 'stream': None, 'start_frame': 1,
                  'end_frame': len(TIMES)}
        self.report([record])
        self.assertEqual(self.request('/api/episode?id=sample')[0], 200)
        self.report([{**record, 'end_frame': len(TIMES) + 1}])
        status, body, _ = self.request('/api/episode?id=sample')
        self.assertEqual(status, 400)
        self.assertIn(b'exceeds episode length', body)

    def test_episode_payload_is_derived_from_the_frame_table(self):
        index = EpisodeIndex(self.root)
        self.addCleanup(index.close)
        self.assertEqual(index.page({})['episodes'], ['sample'])
        episode = index.record('sample')
        self.assertEqual((episode['episode_id'], episode['episode_index'], episode['length']), ('sample', 0, 4))
        self.assertEqual(TIMES, [frame_ns(k) for k in range(4)])
        data = read_episode(index.dataset, episode)
        self.assertEqual(set(data), {'episode_id', 'instructions', 'streams', 'cameras', 'end_ns'})
        self.assertEqual(data['episode_id'], 'sample')
        self.assertEqual(data['end_ns'], TIMES[-1])
        # Runs of the per-frame task; the last one ends at the last frame, where the page still shows it.
        self.assertEqual(data['instructions'], [
            {'start_ns': 0, 'end_ns': TIMES[2], 'text': 'pick'},
            {'start_ns': TIMES[2], 'end_ns': TIMES[3], 'text': 'place'}])
        self.assertEqual(data['cameras'], {'top': {'times': TIMES, 'frames': 4}})
        self.assertEqual(list(data['streams']), list(contract.STATE_NAMES))
        for stream in data['streams'].values():
            self.assertEqual(stream['times'], TIMES)
        self.assertEqual(data['streams']['left_eef']['values'], self.poses.tolist())
        self.assertEqual(data['streams']['right_eef']['values'], [IDENTITY_POSE] * 4)
        self.assertEqual(data['streams']['left_gripper']['values'], [0.] * 4)
        self.assertEqual(data['streams']['right_gripper']['values'], [0., .25, .5, 1.])
        # The HTTP endpoint serves exactly this payload as JSON.
        status, body, content_type = self.request('/api/episode?id=sample')
        self.assertEqual((status, content_type), (200, 'application/json; charset=utf-8'))
        self.assertEqual(json.loads(body), data)

    def test_episode_payload_only_holds_the_requested_episode(self):
        dataset = self.rebuild({
            'short': {'left_gripper': [0, 1], 'tasks': 'open'},
            'long': {'left_gripper': [1, .5, 0], 'tasks': ['close', 'close', 'hold']}})
        short, long = (read_episode(dataset, episode) for episode in dataset.episodes)
        self.assertEqual((short['episode_id'], short['end_ns']), ('short', TIMES[1]))
        self.assertEqual(short['streams']['left_gripper'], {'times': TIMES[:2], 'values': [0., 1.]})
        self.assertEqual(short['instructions'], [{'start_ns': 0, 'end_ns': TIMES[1], 'text': 'open'}])
        self.assertEqual((long['episode_id'], long['end_ns']), ('long', TIMES[2]))
        self.assertEqual(long['streams']['left_gripper'], {'times': TIMES[:3], 'values': [1., .5, 0.]})
        self.assertEqual(long['cameras']['top'], {'times': TIMES[:3], 'frames': 3})
        self.assertEqual([i['text'] for i in long['instructions']], ['close', 'hold'])

    def test_rejects_missing_camera_video_and_inconsistent_frame_rows(self):
        dataset = contract.Dataset(self.root)
        episode = dataset.episodes[0]
        video, _ = dataset.video(episode, 'top')
        saved = video.read_bytes()
        video.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing camera video: top'):
            read_episode(dataset, episode)
        status, body, _ = self.request('/api/episode?id=sample')
        self.assertEqual(status, 400)
        self.assertIn(b'Missing camera video', body)
        video.write_bytes(saved)
        self.assertEqual(read_episode(dataset, episode)['cameras']['top']['frames'], 4)
        # Episode metadata that disagrees with the frame table is not papered over.
        with self.assertRaisesRegex(ValueError, 'frame rows differ from episode metadata'):
            read_episode(dataset, {**episode, 'length': 5})
        status, body, _ = self.request('/api/episode?id=unknown')
        self.assertEqual(status, 400)

    def test_rejects_invalid_episode_metadata(self):
        def escape(rows):
            rows[0]['episode_id'] = '../raw'
            return rows

        self.rewrite_episodes(escape)
        with self.assertRaisesRegex(ValueError, 'episode metadata'):
            EpisodeIndex(self.root)
        with self.assertRaisesRegex(ValueError, 'episode metadata'):
            self.request('/api/dataset')
        info = self.root / 'meta/info.json'
        info.write_text(info.read_text().replace('"fps": 30', '"fps": 60'))
        with self.assertRaisesRegex(ValueError, 'not a contract dataset'):
            EpisodeIndex(self.root)

    def test_frame_requests_are_validated(self):
        paths = {
            '/api/frame?id=sample&camera=top&index=0&width=64': 200,
            '/api/frame?id=sample&camera=top&index=3&width=2048': 200,
            '/api/frame?id=sample&camera=top&index=4': 400,
            '/api/frame?id=sample&camera=top&index=-1': 400,
            '/api/frame?id=sample&camera=top&index=one': 400,
            '/api/frame?id=sample&camera=top': 400,
            '/api/frame?id=sample&camera=left_wrist&index=0': 400,
            '/api/frame?id=sample&camera=../meta&index=0': 400,
            '/api/frame?id=sample&index=0': 400,
            '/api/frame?id=other&camera=top&index=0': 400,
            '/api/frame?camera=top&index=0': 400,
            '/api/frame?id=sample&camera=top&index=0&width=63': 400,
            '/api/frame?id=sample&camera=top&index=0&width=2049': 400,
            '/api/frame?id=sample&camera=top&index=0&width=wide': 400,
        }
        for path, (status, body, content_type) in zip(paths, self.request(list(paths))):
            with self.subTest(path=path):
                self.assertEqual(status, paths[path], body[:200])
                self.assertEqual(content_type, 'image/jpeg' if status == 200 else 'text/plain; charset=utf-8')

    def test_frames_decode_from_the_episodes_own_video(self):
        from PIL import Image
        # Fixture videos are uniform gray; frame k of every episode has level k.
        dataset = self.rebuild({'first': {'left_gripper': np.zeros(3)}, 'second': {'left_gripper': np.zeros(200)}})
        results = self.request([f'/api/frame?id=second&camera=top&index={index}&width=64'
                                for index in (0, 199, 150, 151, 20)]
                               + ['/api/frame?id=first&camera=top&index=2&width=64',
                                  '/api/frame?id=first&camera=top&index=3&width=64',
                                  '/api/frame?id=second&camera=top&index=100'])
        levels = []
        for status, body, content_type in results[:6] + results[7:]:
            self.assertEqual((status, content_type), (200, 'image/jpeg'))
            image = Image.open(BytesIO(body))
            levels.append(np.asarray(image.convert('RGB'), dtype=float).mean())
            size = image.size
        np.testing.assert_allclose(levels, [0, 199, 150, 151, 20, 2, 100], atol=3)
        self.assertEqual(results[6][0], 400)  # "first" has three frames even though "second" has more
        self.assertEqual(size, tuple(dataset.info['features']['observation.images.top']['shape'][1::-1]))
        self.assertEqual(Image.open(BytesIO(results[0][1])).size[0], 64)

    def test_video_range_requests_serve_original_mp4_bytes(self):
        dataset = contract.Dataset(self.root)
        video, _ = dataset.video(dataset.episodes[0], 'top')
        content = video.read_bytes()
        size = len(content)
        with patch('loom_data.viewer.server.ThreadingHTTPServer') as server:
            serve(self.root, '127.0.0.1', 8765)
        handler_class = server.call_args.args[1]
        cases = [
            (None, 200, None, content),
            ('bytes=0-15', 206, f'bytes 0-15/{size}', content[:16]),
            ('bytes=-8', 206, f'bytes {size - 8}-{size - 1}/{size}', content[-8:]),
            (f'bytes={size - 8}-', 206, f'bytes {size - 8}-{size - 1}/{size}', content[-8:]),
            (f'bytes={size}-', 416, f'bytes */{size}', b''),
            ('bytes=4-1', 416, f'bytes */{size}', b''),
        ]
        for header, status, span, body in cases:
            with self.subTest(header=header):
                handler = handler_class.__new__(handler_class)
                handler.path = '/api/video?id=sample&camera=top'
                handler.headers = {} if header is None else {'Range': header}
                headers = {}
                handler.send_header = headers.__setitem__
                handler.send_response = Mock()
                handler.end_headers = Mock()
                handler.wfile = BytesIO()
                handler.do_GET()
                handler.send_response.assert_called_once_with(status)
                self.assertEqual(headers.get('Content-Range'), span)
                self.assertEqual(int(headers['Content-Length']), len(body))
                self.assertEqual(handler.wfile.getvalue(), body)

    def test_frame_requests_map_to_the_episode_video_file_and_frame(self):
        dataset = self.rebuild(still('first', 'second'))
        files = [dataset.video(episode, 'top') for episode in dataset.episodes]
        self.assertNotEqual(files[0][0], files[1][0])
        self.assertEqual([first for _, first in files], [0, 0])
        calls = []

        def get(reader, path, index, width):
            calls.append((path, index, width))
            return b'jpeg'

        with patch.object(FrameReader, 'get', get), patch.object(
                contract.Dataset, 'video', lambda dataset, episode, camera: (
                    files[episode['episode_index']][0], 7 * episode['episode_index'])):
            results = self.request(['/api/frame?id=second&camera=top&index=1&width=320',
                                    '/api/frame?id=first&camera=top&index=1'])
        self.assertEqual(results, [(200, b'jpeg', 'image/jpeg')] * 2)
        # A video file may hold several episodes: the episode's first frame offsets the index.
        self.assertEqual(calls, [(files[1][0], 8, 320), (files[0][0], 1, 1280)])

    def test_cancelled_browser_response_is_ignored(self):
        captured = {}

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                captured['handler'] = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def serve_forever(self):
                handler = captured['handler'].__new__(captured['handler'])
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                handler.wfile = SimpleNamespace(write=Mock(side_effect=BrokenPipeError))
                handler.reply(200, b'image', 'image/jpeg')

        with patch('loom_data.viewer.server.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765)

    def test_frame_reader_uses_threaded_decode_and_requested_width(self):
        class FakeImage:
            def save(self, output, **_):
                output.write(b'jpeg')

        class FakeFrame:
            width = 1920
            height = 1200

            def __init__(self):
                self.reformat = Mock(return_value=SimpleNamespace(to_image=lambda: FakeImage()))

        stream = SimpleNamespace(thread_type=None)
        frame = FakeFrame()
        container = SimpleNamespace(
            streams=SimpleNamespace(video=[stream]),
            decode=Mock(return_value=iter([frame])),
            close=Mock(),
        )
        fake_av = SimpleNamespace(open=Mock(return_value=container))
        reader = FrameReader()
        with patch.dict(sys.modules, {'av': fake_av}):
            self.assertEqual(reader.get(Path('video.mp4'), 0, 320), b'jpeg')
        self.assertEqual(stream.thread_type, 'AUTO')
        self.assertGreaterEqual(stream.thread_count, 1)
        self.assertLessEqual(stream.thread_count, 4)
        frame.reformat.assert_called_once_with(width=320, height=200, format='rgb24')
        reader.close()

    def test_frame_reader_seeks_to_display_index_for_distant_scrubs(self):
        decoded = []

        class FakeFrame:
            width = 320
            height = 180
            time_base = Fraction(1, 30)

            def __init__(self, index):
                self.pts = index

            def to_image(self):
                index = self.pts
                return SimpleNamespace(save=lambda output, **_: output.write(str(index).encode()))

        class FakeContainer:
            def __init__(self):
                self.cursor = 0
                self.streams = SimpleNamespace(video=[SimpleNamespace(
                    thread_type=None, average_rate=Fraction(30),
                    time_base=Fraction(1, 30), start_time=0)])

            def seek(self, target, **_):
                self.cursor = target // 10 * 10

            def decode(self, **_):
                while True:
                    index = self.cursor
                    self.cursor += 1
                    decoded.append(index)
                    yield FakeFrame(index)

            def close(self):
                pass

        fake_av = SimpleNamespace(open=lambda _: FakeContainer())
        reader = FrameReader()
        with patch.dict(sys.modules, {'av': fake_av}):
            self.assertEqual(reader.get(Path('video.mp4'), 450, 320), b'450')
            self.assertEqual(reader.get(Path('video.mp4'), 27, 320), b'27')
        self.assertLess(len(decoded), 25)
        reader.close()

    def test_serves_local_3d_modules_without_exposing_other_files(self):
        results = {}

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def serve_forever(self):
                for path in ('/', '/pose3d.js', '/anomalies.js', '/episodes.js', '/vendor/three.module.min.js',
                             '/vendor/three.core.min.js', '/vendor/../../export_viewer.py'):
                    handler = self.handler.__new__(self.handler)
                    handler.path = path
                    handler.reply = Mock()
                    handler.do_GET()
                    results[path] = handler.reply.call_args.args

        with patch('loom_data.viewer.server.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765)
        self.assertIn(b'type="module"', results['/'][1])
        for path in ('/pose3d.js', '/anomalies.js', '/episodes.js', '/vendor/three.module.min.js', '/vendor/three.core.min.js'):
            status, body, content_type = results[path]
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertTrue(content_type.startswith('text/javascript'))
        self.assertIn(b'three.core.min.js', results['/vendor/three.module.min.js'][1])
        self.assertEqual(results['/vendor/../../export_viewer.py'][0], 404)
