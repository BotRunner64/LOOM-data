import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.viewer.export_viewer import (
    FrameReader, ManifestIndex, read_anomalies, read_dataset, read_episode, serve,
)


def exported_dataset(tmp_path):
    root = tmp_path / 'export'
    episode = root / 'episodes' / 'sample'
    (episode / 'state').mkdir(parents=True)
    (episode / 'rgb').mkdir()
    record = {'episode_id': 'sample', 'cameras': ['top'],
              'instructions': [{'start_ns': 0, 'end_ns': 30, 'text': 'pick'}]}
    (root / 'episodes.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    for name in ('left_eef', 'right_eef', 'left_gripper', 'right_gripper'):
        pose = name.endswith('_eef')
        values = [[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]] * 2 if pose else [0., 1.]
        table = pa.table({'timestamp_ns': pa.array([0, 20], type=pa.int64()),
                          'pose' if pose else 'openness': pa.array(
                              values, type=pa.list_(pa.float64(), 12) if pose else pa.float32())})
        pq.write_table(table, episode / 'state' / f'{name}.parquet')
    pq.write_table(pa.table({'frame_index': pa.array([0, 1], type=pa.int64()),
                             'timestamp_ns': pa.array([10, 30], type=pa.int64())}),
                   episode / 'rgb' / 'top.parquet')
    (episode / 'rgb' / 'top.mp4').write_bytes(b'placeholder')
    return root, record


class ExportViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root, self.record = exported_dataset(Path(self.temporary.name))

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
                handler = self.handler.__new__(self.handler)
                handler.path = path
                handler.reply = lambda *args: results.append(args)
                handler.do_GET()

        with patch('scripts.viewer.export_viewer.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765, report)
        return results[0]

    def test_anomalies_resolve_physical_lines_and_preserve_unknown_reasons(self):
        manifest = self.root / 'episodes.jsonl'
        other = {**self.record, 'episode_id': 'other'}
        manifest.write_text('\n' + json.dumps(self.record) + '\n\n' + json.dumps(other) + '\n')
        records = [
            {'index': 3, 'stream': 'left_eef', 'start_ns': 10, 'end_ns': 20,
             'reason': 'arbitrary_future_rule'},
            {'index': 1, 'stream': 'right_gripper', 'start_ns': 30, 'end_ns': 30,
             'reason': '<uninterpreted reason>'},
        ]
        path = self.report(records + records[:1])
        result = read_anomalies(self.root, path)
        self.assertEqual(result, [dict(records[0], episode_id='other'),
                                  dict(records[1], episode_id='sample'),
                                  dict(records[0], episode_id='other')])
        self.report([{**records[0], 'index': 2}])
        with self.assertRaisesRegex(ValueError, 'manifest record'):
            read_anomalies(self.root, path)

    def test_invalid_anomaly_records_report_source_line(self):
        valid = {'index': 0, 'stream': 'left_eef', 'start_ns': 10, 'end_ns': 20,
                 'reason': 'custom'}
        for change in ({'index': True}, {'index': 1}, {'stream': 'camera'},
                       {'start_ns': -1}, {'end_ns': 9}, {'end_ns': 20.0},
                       {'reason': ''}, {'reason': None}):
            with self.subTest(change=change):
                path = self.report([valid, {**valid, **change}])
                with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:2:'):
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
        record = {'index': 0, 'stream': 'left_eef', 'start_ns': 30, 'end_ns': 30,
                  'reason': 'another_rule'}
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

    def test_episode_pages_bound_initial_payload_and_preserve_manifest_order(self):
        records = [{**self.record, 'episode_id': f'episode_{index:04d}'} for index in range(251)]
        (self.root / 'episodes.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        initial = json.loads(self.request('/api/dataset')[1])
        self.assertEqual(initial['episodes'], [record['episode_id'] for record in records[:100]])
        self.assertEqual((initial['total'], initial['offset'], initial['limit']), (251, 0, 100))
        self.assertIsNone(initial['anomalies'])
        middle = json.loads(self.request('/api/episodes?offset=100')[1])
        self.assertEqual(middle['episodes'], [record['episode_id'] for record in records[100:200]])
        self.assertNotIn('anomalies', middle)
        last = json.loads(self.request('/api/episodes?offset=200')[1])
        self.assertEqual(last['episodes'], [record['episode_id'] for record in records[200:]])
        self.assertEqual(json.loads(self.request('/api/episodes?offset=251')[1])['episodes'], [])

    def test_disk_manifest_index_preserves_physical_and_manifest_order(self):
        records = [{**self.record, 'episode_id': episode_id}
                   for episode_id in ('first', 'second', 'third')]
        (self.root / 'episodes.jsonl').write_text(
            '\n' + json.dumps(records[0]) + '\n\n' + json.dumps(records[1]) + '\n' +
            json.dumps(records[2]) + '\n', encoding='utf-8')
        index = ManifestIndex(self.root)
        self.addCleanup(index.close)
        self.assertEqual(index.total, 3)
        self.assertEqual(index.page({'offset': ['1'], 'limit': ['1']}), {
            'episodes': ['second'], 'total': 3, 'offset': 1, 'limit': 1,
        })
        self.assertEqual(index.episode_id_at(1), 'first')
        self.assertIsNone(index.episode_id_at(2))
        self.assertEqual(index.episode_id_at(3), 'second')
        self.assertEqual(index.record('third')['episode_id'], 'third')

    def test_episode_pages_respect_custom_limits(self):
        records = [{**self.record, 'episode_id': episode_id}
                   for episode_id in ('Alpha_1', 'other', 'alpha_2', 'alpha_3')]
        (self.root / 'episodes.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
        page = json.loads(self.request('/api/episodes?offset=1&limit=1')[1])
        self.assertEqual(page, {'episodes': ['other'], 'total': 4, 'offset': 1, 'limit': 1})
        empty = json.loads(self.request('/api/episodes?offset=4&limit=1')[1])
        self.assertEqual(empty['episodes'], [])
        self.assertEqual(empty['total'], 4)

    def test_rejects_invalid_episode_page_parameters(self):
        for path in ('/api/dataset', '/api/episodes'):
            for query in ('offset=-1', 'offset=abc', 'limit=0', 'limit=201', 'limit=abc'):
                with self.subTest(path=path, query=query):
                    self.assertEqual(self.request(path + '?' + query)[0], 400)

    def test_rejects_report_beyond_episode_duration_when_loading_episode(self):
        record = {'index': 0, 'stream': 'left_eef', 'start_ns': 30, 'end_ns': 30,
                  'reason': 'custom'}
        self.report([record])
        self.assertEqual(self.request('/api/episode?id=sample')[0], 200)
        self.report([{**record, 'end_ns': 31}])
        status, body, _ = self.request('/api/episode?id=sample')
        self.assertEqual(status, 400)
        self.assertIn(b'exceeds episode duration', body)

    def test_reads_only_export_contract(self):
        episodes = read_dataset(self.root)
        self.assertEqual(episodes, {'sample': self.record})
        data = read_episode(self.root, self.record)
        self.assertEqual(data['end_ns'], 30)
        self.assertEqual(data['cameras']['top'], {'times': [10, 30], 'frames': 2})
        self.assertEqual(data['streams']['left_eef']['values'][0][3:9],
                         [1., 0., 0., 0., 1., 0.])

    def test_rejects_missing_or_unordered_camera_index(self):
        index = self.root / 'episodes' / 'sample' / 'rgb' / 'top.parquet'
        pq.write_table(pa.table({'frame_index': [0, 2], 'timestamp_ns': [10, 5]}), index)
        with self.assertRaisesRegex(ValueError, 'Invalid camera stream'):
            read_episode(self.root, self.record)

    def test_rejects_episode_path_escape(self):
        (self.root / 'episodes.jsonl').write_text(
            '{"episode_id":"../raw","cameras":["top"],"instructions":[]}\n',
            encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'episode_id'):
            read_dataset(self.root)

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

        with patch('scripts.viewer.export_viewer.ThreadingHTTPServer', FakeServer):
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

        with patch('scripts.viewer.export_viewer.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765)
        self.assertIn(b'type="module"', results['/'][1])
        for path in ('/pose3d.js', '/anomalies.js', '/episodes.js', '/vendor/three.module.min.js', '/vendor/three.core.min.js'):
            status, body, content_type = results[path]
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertTrue(content_type.startswith('text/javascript'))
        self.assertIn(b'three.core.min.js', results['/vendor/three.module.min.js'][1])
        self.assertEqual(results['/vendor/../../export_viewer.py'][0], 404)
