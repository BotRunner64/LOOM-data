import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from loom_data.check.dataset_stats import dataset_name, human_duration, main, summarize_dataset
from loom_data.contract import discover_datasets

LAYOUT = 'top+left_wrist+right_wrist_640x480'


def write_info(root, episodes, frames, **changes):
    """Statistics read nothing but meta/info.json, so that is the whole fixture."""
    (root / 'meta').mkdir(parents=True, exist_ok=True)
    info = {'codebase_version': 'v3.0', 'fps': 30, 'total_episodes': episodes, 'total_frames': frames, **changes}
    (root / 'meta/info.json').write_text(json.dumps(info), encoding='utf-8')
    return root


class DatasetStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        # 91 frames at 30 fps: 3.033... s.
        self.root = write_info(self.parent / 'export', 1, 91)

    def test_duration_is_the_frame_count_at_30_fps_truncating_subseconds_last(self):
        write_info(self.root, 2, 301)
        summary = summarize_dataset(self.root)
        self.assertEqual(summary['dataset'], 'export')
        self.assertEqual(summary['path'], str(self.root))
        self.assertEqual(summary['episode_count'], 2)
        self.assertEqual(summary['frame_count'], 301)
        self.assertEqual(summary['duration_ns'], 10_033_333_333)
        self.assertEqual(summary['duration_seconds'], 10)
        self.assertIsInstance(summary['duration_seconds'], int)
        self.assertEqual(summary['duration_source'], 'frames')
        self.assertNotIn('disk_usage_bytes', summary)
        # Whole seconds are exact, and long datasets do not lose precision.
        self.assertEqual(summarize_dataset(write_info(self.root, 1, 300))['duration_ns'], 10_000_000_000)
        frames = 30 * 3600 * 100_000 + 1
        self.assertEqual(summarize_dataset(write_info(self.root, 7, frames))['duration_ns'],
                         360_000_000 * 1_000_000_000 + 33_333_333)

    def test_layout_datasets_are_named_after_their_source(self):
        for layout in (LAYOUT, 'top_640x360', 'top+left_wrist+right_wrist_640x360+640x480+640x480'):
            root = write_info(self.parent / 'Source-A' / layout, 1, 30)
            self.assertEqual(dataset_name(root), f'Source-A/{layout}')
            self.assertEqual(summarize_dataset(root)['dataset'], f'Source-A/{layout}')
        for name in ('export', 'top', 'Top_640x480', 'top_640', '640x480'):
            self.assertEqual(dataset_name(self.parent / 'Source-A' / name), name)

    def test_invalid_info_identifies_the_dataset(self):
        info = self.root / 'meta/info.json'
        for changes in ({'fps': 25}, {'fps': None}, {'total_episodes': None}, {'total_episodes': True},
                        {'total_episodes': -1}, {'total_episodes': 1.0}, {'total_frames': None},
                        {'total_frames': '91'}, {'total_frames': 90.5}, {'total_frames': 0},
                        {'total_episodes': 92}):
            with self.subTest(changes=changes):
                write_info(self.root, 1, 91, **changes)
                with self.assertRaisesRegex(ValueError, 'export: not a 30 fps contract dataset or invalid totals'):
                    summarize_dataset(self.root)
        for text in ('{invalid json}', '', '[]', 'null', '3'):
            with self.subTest(text=text):
                info.write_text(text)
                with self.assertRaisesRegex(ValueError, 'export: .*meta/info.json'):
                    summarize_dataset(self.root)
        info.unlink()
        with self.assertRaisesRegex(ValueError, 'export: cannot read meta/info.json'):
            summarize_dataset(self.root)

    def test_empty_dataset(self):
        summary = summarize_dataset(write_info(self.root, 0, 0))
        self.assertEqual(summary['episode_count'], 0)
        self.assertEqual(summary['duration_ns'], 0)

    def test_discovery_accepts_dataset_source_and_export_roots(self):
        export = self.parent / 'tree'
        first = write_info(export / 'Source-A' / LAYOUT, 1, 30)
        second = write_info(export / 'Source-A' / 'top_640x360', 1, 30)
        third = write_info(export / 'Source-B' / LAYOUT, 1, 30)
        (export / 'Source-B' / 'notes').mkdir()
        (export / 'README.md').write_text('not a dataset')
        self.assertEqual(discover_datasets([first]), [first])
        self.assertEqual(discover_datasets([export / 'Source-A']), [first, second])
        self.assertEqual(discover_datasets([export]), [first, second, third])
        # Every path is resolved on its own; a dataset reached twice is listed once, in first-seen order.
        self.assertEqual(discover_datasets([third, export, export / 'Source-A', second]), [third, first, second])
        self.assertEqual(discover_datasets([self.parent, self.root]), [self.root])

    def test_discovery_rejects_missing_and_empty_paths(self):
        with self.assertRaisesRegex(ValueError, 'directory does not exist'):
            discover_datasets([self.parent / 'missing'])
        with self.assertRaisesRegex(ValueError, 'directory does not exist'):
            discover_datasets([self.root / 'meta/info.json'])
        empty = self.parent / 'empty'
        (empty / 'a' / 'b' / 'c' / 'meta').mkdir(parents=True)
        (empty / 'a' / 'b' / 'c' / 'meta/info.json').write_text('{}')  # deeper than an export root
        (empty / 'meta').mkdir()
        (empty / 'a' / 'meta' / 'info.json').mkdir(parents=True)        # not a file
        with self.assertRaisesRegex(ValueError, 'no exported datasets'):
            discover_datasets([empty])
        with self.assertRaisesRegex(ValueError, 'no exported datasets'):
            discover_datasets([self.root, empty])

    def test_cli_json_only_reads_info_without_progress(self):
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress), patch(
                'pathlib.Path.iterdir',
                side_effect=AssertionError('must not enumerate dataset directories')), patch(
                'os.scandir', side_effect=AssertionError('must not scan storage')), patch(
                'pyarrow.parquet.read_table', side_effect=AssertionError('must not read tables')):
            main([str(self.root), str(self.root), '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual(len(report['datasets']), 1)
        self.assertEqual(report['datasets'][0]['frame_count'], 91)
        self.assertEqual(report['total']['episode_count'], 1)
        self.assertEqual(report['total']['duration_ns'], 3_033_333_333)
        self.assertEqual(report['total']['duration_seconds'], 3)
        self.assertEqual(report['total']['duration_source'], 'frames')
        self.assertNotIn('disk_usage_bytes', report['total'])
        self.assertNotIn('disk_usage_bytes', report['datasets'][0])
        self.assertEqual(progress.getvalue(), '')

    def test_cli_totals_across_multiple_datasets_sum_before_truncating(self):
        write_info(self.parent / 'other', 2, 239)
        output = StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(StringIO()):
            main([str(self.parent), '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual([dataset['dataset'] for dataset in report['datasets']], ['export', 'other'])
        self.assertEqual([dataset['duration_seconds'] for dataset in report['datasets']], [3, 7])
        self.assertEqual(report['total']['episode_count'], 3)
        self.assertEqual(report['total']['duration_ns'], 11_000_000_000 - 1)  # 330 frames, floored per dataset
        self.assertEqual(report['total']['duration_seconds'], 10)

    def test_cli_displays_time_with_frame_note_without_progress(self):
        write_info(self.parent / 'Source-A' / LAYOUT, 3, 30 * 3661)
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress):
            main([str(self.root), str(self.parent / 'Source-A')])
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0].split(), ['Dataset', 'Episodes', 'Time', '(HH:MM:SS)'])
        self.assertEqual(lines[1].split(), ['export', '1', '00:00:03'])
        self.assertEqual(lines[2].split(), [f'Source-A/{LAYOUT}', '3', '01:01:01'])
        self.assertEqual(lines[3].split(), ['TOTAL', '4', '01:01:04'])
        self.assertIn('Note: Time is the frame count at 30 fps.', output.getvalue())
        self.assertNotIn('Disk usage', output.getvalue())
        self.assertEqual(progress.getvalue(), '')

    def test_cli_interrupt_exits_cleanly_without_report(self):
        output, progress = StringIO(), StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress), patch(
                'loom_data.check.dataset_stats.summarize_dataset', side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as raised:
                main([str(self.root)])
        self.assertEqual(raised.exception.code, 130)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('Interrupted; dataset files were not modified.', progress.getvalue())

    def test_cli_rejects_invalid_info_without_partial_report(self):
        other = write_info(self.parent / 'other', 1, 30)
        for text in ('{invalid json}', '[]', json.dumps({'fps': 30, 'total_episodes': 2, 'total_frames': 1})):
            (other / 'meta/info.json').write_text(text)
            output, errors = StringIO(), StringIO()
            with self.subTest(text=text), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as raised:
                    main([str(self.root), str(other)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.getvalue(), '')
            self.assertIn(f'{other}: ', errors.getvalue())

    def test_duration_format_truncates_subseconds_and_supports_long_datasets(self):
        self.assertEqual(human_duration(999_999_999), '00:00:00')
        self.assertEqual(human_duration(59_999_999_999), '00:00:59')
        self.assertEqual(human_duration(100 * 3600 * 1_000_000_000), '100:00:00')


if __name__ == '__main__':
    unittest.main()
