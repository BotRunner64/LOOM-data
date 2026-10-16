import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check.dataset_stats import (
    discover_datasets, human_duration, main, summarize_dataset,
)


class DatasetStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / 'export'
        self.root.mkdir()
        # Only a manifest is needed: media and episode directories are never read.
        self.records = []
        self.add_episode('one', 3_000_000_001)

    def write_manifest(self):
        (self.root / 'episodes.jsonl').write_text(
            '\n' + '\n\n'.join(json.dumps(record) for record in self.records) + '\n',
            encoding='utf-8')

    def add_episode(self, episode_id, end_ns):
        self.records.append({'episode_id': episode_id, 'cameras': ['top'],
                             'instructions': [{'start_ns': 0, 'end_ns': end_ns,
                                               'text': 'task'}]})
        self.write_manifest()

    def test_sums_manifest_spans_before_truncating_subseconds(self):
        self.add_episode('two', 7_000_000_999)
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['episode_count'], 2)
        self.assertEqual(summary['duration_ns'], 10_000_001_000)
        self.assertEqual(summary['duration_seconds'], 10)
        self.assertIsInstance(summary['duration_seconds'], int)
        self.assertEqual(summary['duration_source'], 'instructions')
        self.assertNotIn('disk_usage_bytes', summary)

    def test_multiple_instructions_use_final_end_instead_of_summing_segments(self):
        self.records[0]['instructions'] = [
            {'start_ns': 10, 'end_ns': 1_000_000_000},
            {'start_ns': 2_000_000_000, 'end_ns': 3_000_000_001},
        ]
        self.write_manifest()
        self.assertEqual(summarize_dataset(self.root)['duration_ns'], 3_000_000_001)

    def test_invalid_records_identify_manifest_line(self):
        for record in ([], {}, {**self.records[0], 'episode_id': '../outside'},
                       {**self.records[0], 'episode_id': ''},
                       {**self.records[0], 'episode_id': 'bad\\name'},
                       {**self.records[0], 'episode_id': 'bad\0name'}):
            with self.subTest(record=record):
                self.records = [record]
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'episodes.jsonl:2:'):
                    summarize_dataset(self.root)
        (self.root / 'episodes.jsonl').write_text('\n{invalid json}')
        with self.assertRaisesRegex(ValueError, 'episodes.jsonl:2:'):
            summarize_dataset(self.root)

    def test_duplicate_episode_ids_are_rejected(self):
        self.records.append(self.records[0])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate episode_id: one'):
            summarize_dataset(self.root)

    def test_invalid_instruction_intervals_are_rejected(self):
        original = self.records[0]
        for instructions in (None, [], [None], [{'start_ns': 0, 'end_ns': True}],
                             [{'start_ns': True, 'end_ns': 2}],
                             [{'start_ns': -1, 'end_ns': 2}],
                             [{'start_ns': 2, 'end_ns': 2}],
                             [{'start_ns': 0, 'end_ns': 2**63}],
                             [{'start_ns': 0, 'end_ns': 2}, {'start_ns': 1, 'end_ns': 3}]):
            with self.subTest(instructions=instructions):
                self.records = [{**original, 'instructions': instructions}]
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'episodes.jsonl:2:'):
                    summarize_dataset(self.root)

    def test_empty_manifest(self):
        (self.root / 'episodes.jsonl').write_text('\n\n')
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['episode_count'], 0)
        self.assertEqual(summary['duration_ns'], 0)

    def test_discovery_deduplicates_overlapping_paths_and_rejects_missing_datasets(self):
        self.assertEqual(discover_datasets([self.parent, self.root]), [self.root])
        with self.assertRaisesRegex(ValueError, 'directory does not exist'):
            discover_datasets([self.parent / 'missing'])
        empty = self.parent / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'no exported datasets'):
            discover_datasets([empty])

    def test_cli_json_only_reads_manifests_without_progress(self):
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress), patch(
                'pathlib.Path.iterdir',
                side_effect=AssertionError('must not enumerate episode directories')), patch(
                'os.scandir', side_effect=AssertionError('must not scan storage')):
            main([str(self.root), str(self.root), '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual(len(report['datasets']), 1)
        self.assertEqual(report['total']['episode_count'], 1)
        self.assertEqual(report['total']['duration_ns'], 3_000_000_001)
        self.assertEqual(report['total']['duration_seconds'], 3)
        self.assertEqual(report['total']['duration_source'], 'instructions')
        self.assertNotIn('disk_usage_bytes', report['total'])
        self.assertNotIn('disk_usage_bytes', report['datasets'][0])
        self.assertEqual(progress.getvalue(), '')

    def test_cli_totals_across_multiple_datasets(self):
        other = self.parent / 'other'
        other.mkdir()
        record = {**self.records[0], 'instructions': [{'start_ns': 0, 'end_ns': 7_999_999_999}]}
        (other / 'episodes.jsonl').write_text(json.dumps(record))
        output = StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(StringIO()):
            main([str(self.parent), '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual(len(report['datasets']), 2)
        self.assertEqual(report['total']['episode_count'], 2)
        self.assertEqual(report['total']['duration_ns'], 11_000_000_000)
        self.assertEqual(report['total']['duration_seconds'], 11)

    def test_cli_displays_time_with_instruction_note_without_progress(self):
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress):
            main([str(self.root)])
        self.assertIn('Time (HH:MM:SS)', output.getvalue())
        self.assertIn('Note: Time comes from instructions', output.getvalue())
        self.assertIn('TOTAL', output.getvalue())
        self.assertNotIn('Disk usage', output.getvalue())
        self.assertEqual(progress.getvalue(), '')

    def test_cli_interrupt_exits_cleanly_without_report(self):
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress), patch(
                'scripts.check.dataset_stats.summarize_dataset', side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 130)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('Interrupted; dataset files were not modified.', progress.getvalue())

    def test_cli_rejects_invalid_manifest_without_partial_report(self):
        (self.root / 'episodes.jsonl').write_text('{invalid json}')
        output, errors = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('episodes.jsonl:1:', errors.getvalue())

    def test_duration_format_truncates_subseconds_and_supports_long_datasets(self):
        self.assertEqual(human_duration(999_999_999), '00:00:00')
        self.assertEqual(human_duration(59_999_999_999), '00:00:59')
        self.assertEqual(human_duration(100 * 3600 * 1_000_000_000), '100:00:00')


if __name__ == '__main__':
    unittest.main()
