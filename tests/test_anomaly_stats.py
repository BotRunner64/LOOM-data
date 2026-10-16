import contextlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.check.anomaly_stats import main, summarize_dataset


class AnomalyStatsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / 'example'
        self.root.mkdir()
        # Physical indices 1 and 3, with no state files or videos available.
        (self.root / 'episodes.jsonl').write_text(
            '\n{"episode_id":"one"}\n\n{"episode_id":"two"}\n', encoding='utf-8')
        self.records = [dict(index=1, stream=stream, start_ns=2_000_000_000,
                             end_ns=15_000_000_000, reason='internal_idle')
                        for stream in ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')]
        self.records += [dict(index=3, stream='left_eef', start_ns=0,
                              end_ns=4_000_000_000, reason='boundary_idle'),
                         dict(index=3, stream='left_eef', start_ns=6_000_000_001,
                              end_ns=6_030_000_001, reason='state_jump')]
        self.write_report()

    def write_report(self):
        (self.root / 'anomalies.jsonl').write_text(
            '\n' + '\n'.join(json.dumps(record) for record in self.records), encoding='utf-8')

    def cli_report(self, *args):
        output = StringIO()
        with contextlib.redirect_stdout(output):
            main([*map(str, args), '--json'])
        return json.loads(output.getvalue())

    def test_physical_indices_and_deduplication_across_streams_and_reasons(self):
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['record_count'], 6)
        self.assertEqual(summary['affected_episode_count'], 2)
        self.assertEqual(summary['affected_episode_rate'], 1)
        self.assertEqual(summary['by_reason']['internal_idle']['record_count'], 4)
        self.assertEqual(summary['by_reason']['internal_idle']['affected_episode_count'], 1)
        self.assertEqual(summary['by_reason']['boundary_idle']['affected_episode_count'], 1)
        self.assertEqual(summary['by_reason']['state_jump']['affected_episode_count'], 1)

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

    def test_empty_manifest_has_no_rate(self):
        (self.root / 'episodes.jsonl').write_text('\n')
        (self.root / 'anomalies.jsonl').write_text('')
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['episode_count'], 0)
        self.assertIsNone(summary['affected_episode_rate'])

    def test_multiple_datasets_total_excludes_missing_reports_from_rate(self):
        missing = self.parent / 'missing'
        missing.mkdir()
        (missing / 'episodes.jsonl').write_text('{"episode_id":"one"}\n')
        report = self.cli_report(self.parent, self.root)
        self.assertEqual(len(report['datasets']), 2)
        self.assertEqual(report['total']['episode_count'], 3)
        self.assertEqual(report['total']['reported_episode_count'], 2)
        self.assertEqual(report['total']['missing_report_count'], 1)
        self.assertEqual(report['total']['affected_episode_rate'], 1)
        (missing / 'anomalies.jsonl').write_text(json.dumps(self.records[0] | {'index': 0}))
        report = self.cli_report(self.parent)
        self.assertEqual(report['total']['affected_episode_count'], 3)
        self.assertEqual(report['total']['by_reason']['internal_idle']['affected_episode_count'], 2)

    def test_custom_report_and_both_entrypoints(self):
        custom = self.parent / 'custom.jsonl'
        custom.write_text(json.dumps(self.records[-1]))
        for entry in (['scripts/check/anomaly_stats.py'], ['-m', 'scripts.check.anomaly_stats']):
            with self.subTest(entry=entry):
                result = subprocess.run(
                    [sys.executable, *entry, str(self.root), '--anomalies', str(custom),
                     '--json'], cwd=Path(__file__).resolve().parents[1],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report['total']['record_count'], 1)
                self.assertEqual(report['total']['by_reason']['state_jump']['affected_episode_count'], 1)

    def test_invalid_reports_identify_line(self):
        for change in ({'index': True}, {'index': 2}, {'start_ns': -1}, {'end_ns': False},
                       {'end_ns': 1}, {'stream': []}, {'reason': ''}, {'reason': None}):
            with self.subTest(change=change):
                self.records = [self.records[0] | change]
                self.write_report()
                with self.assertRaisesRegex(ValueError, 'anomalies.jsonl:2:'):
                    summarize_dataset(self.root)
                # Restore a valid record for the next case.
                self.records = [dict(index=1, stream='left_eef', start_ns=2_000_000_000,
                                     end_ns=15_000_000_000, reason='internal_idle')]

    def test_invalid_manifests_and_malformed_json(self):
        for text in ('\n[]', '\n{}', '\n{"episode_id":"../bad"}', '\n{bad json}',
                     '\n{"episode_id":"one"}\n{"episode_id":"one"}'):
            with self.subTest(text=text):
                (self.root / 'episodes.jsonl').write_text(text)
                with self.assertRaisesRegex(ValueError, 'episodes.jsonl:'):
                    summarize_dataset(self.root)

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
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 2)

    def test_interrupt_is_clean(self):
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()), patch(
                'scripts.check.anomaly_stats.summarize_dataset', side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 130)

if __name__ == '__main__':
    unittest.main()
