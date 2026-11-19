import contextlib
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from support import make_dataset
from loom_data.check.anomaly_stats import main, read_episode_ids, summarize_dataset

LAYOUT = 'top_640x480'


class AnomalyStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.template = Path(temporary.name).resolve()
        make_dataset(cls.template / 'example', {'one': {'left_gripper': np.zeros(2)},
                                                'two': {'left_gripper': np.zeros(2)}})
        make_dataset(cls.template / 'missing', {'one': {'left_gripper': np.zeros(2)}})

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.root = self.parent / 'example'
        shutil.copytree(self.template / 'example', self.root)
        # Records address episodes by episode_index: 0 is "one", 1 is "two".
        self.records = [dict(index=0, stream=stream, start_ns=2_000_000_000,
                             end_ns=15_000_000_000, reason='internal_idle')
                        for stream in ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')]
        self.records += [dict(index=1, stream='left_eef', start_ns=0,
                              end_ns=4_000_000_000, reason='boundary_idle'),
                         dict(index=1, stream='left_eef', start_ns=6_000_000_001,
                              end_ns=6_030_000_001, reason='state_jump')]
        self.write_report()

    def write_report(self):
        (self.root / 'anomalies.jsonl').write_text(
            '\n' + '\n'.join(json.dumps(record) for record in self.records), encoding='utf-8')

    def add_missing(self):
        return Path(shutil.copytree(self.template / 'missing', self.parent / 'missing'))

    def rewrite_episodes(self, change):
        path = self.root / 'meta/episodes/chunk-000/file-000.parquet'
        rows = change(pq.read_table(path).to_pylist())
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)

    def cli_report(self, *args):
        output = StringIO()
        with contextlib.redirect_stdout(output):
            main([*map(str, args), '--json'])
        return json.loads(output.getvalue())

    def test_episode_indices_and_deduplication_across_streams_and_reasons(self):
        self.assertEqual(read_episode_ids(self.root), {0: 'one', 1: 'two'})
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['dataset'], 'example')
        self.assertEqual(summary['episode_count'], 2)
        self.assertEqual(summary['record_count'], 6)
        self.assertEqual(summary['affected_episode_count'], 2)
        self.assertEqual(summary['affected_episode_rate'], 1)
        self.assertEqual(summary['by_reason']['internal_idle']['record_count'], 4)
        self.assertEqual(summary['by_reason']['internal_idle']['affected_episode_count'], 1)
        self.assertEqual(summary['by_reason']['internal_idle']['affected_episode_rate'], .5)
        self.assertEqual(summary['by_reason']['boundary_idle']['affected_episode_count'], 1)
        self.assertEqual(summary['by_reason']['state_jump']['affected_episode_count'], 1)

    def test_layout_dataset_is_named_after_its_source(self):
        root = self.parent / 'Source-A' / LAYOUT
        root.parent.mkdir()
        self.root.rename(root)
        summary = summarize_dataset(root)
        self.assertEqual(summary['dataset'], f'Source-A/{LAYOUT}')
        self.assertEqual(summary['record_count'], 6)
        # The source folder and the export root above it both lead to the layout dataset.
        for path in (root.parent, self.parent):
            report = self.cli_report(path)
            self.assertEqual([dataset['path'] for dataset in report['datasets']], [str(root)])
            self.assertEqual(report['total']['affected_episode_count'], 2)

    def test_empty_report_and_missing_report_are_distinct(self):
        report = self.root / 'anomalies.jsonl'
        report.write_text('\n')
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['report_status'], 'available')
        self.assertEqual(summary['record_count'], 0)
        self.assertEqual(summary['affected_episode_rate'], 0)
        report.unlink()
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['report_status'], 'missing')
        self.assertIsNone(summary['record_count'])
        self.assertIsNone(summary['affected_episode_rate'])

    def test_total_without_any_report_has_no_rate(self):
        (self.root / 'anomalies.jsonl').unlink()
        total = self.cli_report(self.root)['total']
        self.assertEqual(total['episode_count'], 2)
        self.assertEqual(total['reported_episode_count'], 0)
        self.assertEqual(total['record_count'], 0)
        self.assertIsNone(total['affected_episode_rate'])
        self.assertEqual(total['by_reason'], {})

    def test_multiple_datasets_total_excludes_missing_reports_from_rate(self):
        missing = self.add_missing()
        report = self.cli_report(self.parent, self.root)
        self.assertEqual(len(report['datasets']), 2)
        self.assertEqual(report['total']['episode_count'], 3)
        self.assertEqual(report['total']['reported_episode_count'], 2)
        self.assertEqual(report['total']['missing_report_count'], 1)
        self.assertEqual(report['total']['affected_episode_rate'], 1)
        (missing / 'anomalies.jsonl').write_text(json.dumps(self.records[0]))
        report = self.cli_report(self.parent)
        self.assertEqual(report['total']['affected_episode_count'], 3)
        self.assertEqual(report['total']['by_reason']['internal_idle']['affected_episode_count'], 2)
        # Indices are per dataset: episode 1 exists in "example" but not in the one-episode dataset.
        (missing / 'anomalies.jsonl').write_text(json.dumps(self.records[-1]))
        with self.assertRaisesRegex(ValueError, 'missing/anomalies.jsonl:1: index does not address an episode'):
            summarize_dataset(missing)

    def test_custom_report_through_the_cli(self):
        custom = self.parent / 'custom.jsonl'
        custom.write_text(json.dumps(self.records[-1]))
        result = subprocess.run(
            [sys.executable, '-m', 'loom_data', 'anomaly-stats', str(self.root), '--anomalies', str(custom), '--json'],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['datasets'][0]['report_path'], str(custom))
        self.assertEqual(report['total']['record_count'], 1)
        self.assertEqual(report['total']['by_reason']['state_jump']['affected_episode_count'], 1)
        self.add_missing()
        result = subprocess.run(
            [sys.executable, '-m', 'loom_data', 'anomaly-stats', str(self.parent), '--anomalies', str(custom)],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--anomalies requires a single dataset', result.stderr)

    def test_invalid_reports_identify_line(self):
        for change in ({'index': True}, {'index': 2}, {'index': -1}, {'index': 'one'}, {'start_ns': -1},
                       {'end_ns': False}, {'end_ns': 1}, {'stream': []}, {'stream': 'top'},
                       {'reason': ''}, {'reason': None}):
            with self.subTest(change=change):
                self.records = [self.records[0] | change]
                self.write_report()
                with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:2:'):
                    summarize_dataset(self.root)
                # Restore a valid record for the next case.
                self.records = [dict(index=1, stream='left_eef', start_ns=2_000_000_000,
                                     end_ns=15_000_000_000, reason='internal_idle')]
        self.write_report()
        self.assertEqual(summarize_dataset(self.root)['record_count'], 1)

    def test_invalid_datasets_are_rejected(self):
        info = self.root / 'meta/info.json'
        metadata = self.root / 'meta/episodes/chunk-000/file-000.parquet'
        saved = info.read_text(), metadata.read_bytes()

        def unsafe(rows):
            rows[1]['episode_id'] = '../bad'
            return rows

        def duplicate(rows):
            rows[1]['episode_id'] = 'one'
            return rows

        def unordered(rows):
            rows[1]['episode_index'] = 3
            return rows

        for change in (unsafe, duplicate, unordered, lambda rows: rows[:1], lambda rows: []):
            self.rewrite_episodes(change)
            with self.subTest(change=change), self.assertRaisesRegex(
                    ValueError, 'example: invalid, duplicate or missing episode metadata'):
                summarize_dataset(self.root)
            metadata.write_bytes(saved[1])
        for text, message in (('{bad json}', 'example: not an exported dataset'),
                              (saved[0].replace('"fps": 30', '"fps": 10'), 'example: not a contract dataset'),
                              (saved[0].replace('"total_episodes": 2', '"total_episodes": 3'),
                               'example: invalid, duplicate or missing episode metadata')):
            self.assertNotEqual(text, saved[0])
            info.write_text(text)
            with self.subTest(text=text[:20]), self.assertRaisesRegex(ValueError, message):
                summarize_dataset(self.root)
        info.write_text(saved[0])
        self.assertEqual(summarize_dataset(self.root)['record_count'], 6)

    def test_cli_rejects_invalid_inputs_without_partial_output(self):
        for args in ([str(self.parent / 'not-found')],
                     [str(self.root), '--anomalies', str(self.parent / 'not-found.jsonl')]):
            output = StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.getvalue(), '')
        (self.root / 'anomalies.jsonl').write_text('invalid')
        output = StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.getvalue(), '')

    def test_interrupt_is_clean(self):
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()), patch(
                'loom_data.check.anomaly_stats.summarize_dataset', side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 130)


if __name__ == '__main__':
    unittest.main()
