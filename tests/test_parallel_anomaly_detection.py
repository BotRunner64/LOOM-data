import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.check import anomaly_detection as detector
from scripts.check import parallel
from scripts.convert.export_common import write_state, write_video_index


class ParallelDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dataset'
        self.root.mkdir()
        self.output = self.root / 'anomalies.jsonl'
        self.records = []
        # Blank lines preserve physical indices 1, 3, 5, 7, 9. One episode
        # produces no anomalies, which must still count as committed work.
        for i in range(5):
            base = self.root / 'episodes' / f'episode-{i}'
            times = np.array([0, 2 if i == 2 else 5_000_000_000], dtype=np.int64)
            poses = np.tile([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.], (2, 1))
            for name in detector.STATE_NAMES:
                write_state(base / 'state' / f'{name}.parquet', times,
                            poses if name.endswith('_eef') else np.zeros(2), 0,
                            pose=name.endswith('_eef'))
            (base / 'rgb').mkdir()
            write_video_index(base / 'rgb/top.parquet', times, 0)
            self.records.append({'episode_id': f'episode-{i}', 'cameras': ['top']})
        self.manifest = self.root / 'episodes.jsonl'
        self.manifest.write_text(''.join('\n' + json.dumps(r) + '\n' for r in self.records))

    def cli(self, *args, env=None, direct=False):
        entry = ['scripts/check/parallel.py'] if direct else ['-m', 'scripts.check.parallel']
        return subprocess.run([sys.executable, *entry, *args, '--input-dir', str(self.root)],
                              cwd=parallel.ROOT, env=env, capture_output=True, text=True, check=False)

    def test_local_cli_matches_serial_and_repeat_skips_all_results(self):
        serial = self.root / 'serial.jsonl'
        self.assertEqual(detector.detect_dataset(self.root, serial), (5, 8))
        result = self.cli('run', '--workers', '3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_bytes(), serial.read_bytes())
        store = parallel.Store(self.root, self.output)
        stamps = {p: p.stat().st_mtime_ns for p in (store.work / 'records').iterdir()}
        self.assertEqual(store.result(store.plan['tasks'][2]), [])
        result = self.cli('run', '--workers', '2', direct=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Already complete', result.stdout)
        self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_resume_changes_worker_count_and_keeps_committed_episodes(self):
        store = parallel.prepare(self.root)
        self.assertEqual(parallel.worker(self.root, worker_id=0, worker_count=3), 0)
        self.assertEqual(store.audit(), (2, 0, 3))
        committed = {p: p.read_bytes() for p in (store.work / 'records').iterdir()}
        with self.assertRaisesRegex(parallel.StateError, 'Incomplete detection'):
            parallel.finalize(self.root)
        self.assertFalse(self.output.exists())
        temporary = store.work / 'records/.interrupted.tmp'
        temporary.write_text('partial write')
        report_temporary = self.output.with_name(f'.{self.output.name}.interrupted.tmp')
        report_temporary.write_text('partial report')
        parallel.prepare(self.root)
        self.assertFalse(temporary.exists())
        self.assertFalse(report_temporary.exists())
        original = detector.read_episode
        called = []

        def read(root, record):
            called.append(record['episode_id'])
            return original(root, record)

        with patch.object(detector, 'read_episode', side_effect=read):
            for rank in range(2):
                self.assertEqual(parallel.worker(self.root, worker_id=rank, worker_count=2), 0)
        self.assertCountEqual(called, ['episode-1', 'episode-2', 'episode-4'])
        self.assertEqual(committed, {p: p.read_bytes() for p in committed})
        self.assertEqual(parallel.finalize(self.root), (5, 8))
        self.assertEqual([json.loads(line)['index'] for line in self.output.read_text().splitlines()],
                         [1, 1, 3, 3, 7, 7, 9, 9])

    def test_plan_is_a_single_sqlite_database(self):
        store = parallel.prepare(self.root)
        self.assertEqual({p.name for p in store.work.iterdir()}, {'plan.sqlite', 'records', 'errors', 'logs'})
        with sqlite3.connect(store.database) as connection:
            metadata = json.loads(connection.execute('SELECT body FROM metadata WHERE id=1').fetchone()[0])
            self.assertEqual(metadata['config'], parallel.asdict(detector.Config()))
            self.assertEqual(metadata['task_count'], 5)
            self.assertEqual(connection.execute('SELECT count(*) FROM tasks').fetchone()[0], 5)
        stamp = store.database.stat().st_mtime_ns
        parallel.prepare(self.root)
        self.assertEqual(store.database.stat().st_mtime_ns, stamp)

    def test_worker_fetches_only_assigned_tasks_and_empty_ranks_work(self):
        store = parallel.prepare(self.root)
        decoded = []
        loads = json.loads

        def load(value):
            result = loads(value)
            if isinstance(result, dict) and set(result) == {'index', 'inputs', 'record'}:
                decoded.append(result['index'])
            return result

        with patch.object(parallel.json, 'loads', side_effect=load):
            self.assertEqual(parallel.worker(self.root, worker_id=1, worker_count=3), 0)
        self.assertEqual(decoded, [3, 9])
        self.assertEqual(parallel.worker(self.root, worker_id=6, worker_count=8), 0)
        self.assertEqual(store.audit(), (2, 0, 3))

    def test_metadata_and_task_corruption_block_detection_and_publication(self):
        store = parallel.prepare(self.root)
        saved = store.database.read_bytes()
        cases = (
            ("UPDATE metadata SET body=replace(body, '0.01', '0.02')", 'Damaged detection plan metadata'),
            ('DELETE FROM tasks WHERE ordinal=0', 'Missing plan tasks'),
            ("UPDATE tasks SET body=body || ' ' WHERE ordinal=0", 'Damaged plan task checksum'),
        )
        for sql, message in cases:
            with self.subTest(sql=sql):
                store.database.write_bytes(saved)
                with sqlite3.connect(store.database) as connection:
                    connection.execute(sql)
                with self.assertRaisesRegex(parallel.StateError, message):
                    parallel.worker(self.root)
                with self.assertRaisesRegex(parallel.StateError, message):
                    parallel.finalize(self.root)
                self.assertFalse(self.output.exists())

    def test_task_digest_detects_rewritten_or_added_tasks(self):
        store = parallel.prepare(self.root)
        with sqlite3.connect(store.database) as connection:
            value = json.loads(connection.execute('SELECT body FROM tasks WHERE ordinal=0').fetchone()[0])
            value['record']['extra'] = 'changed'
            connection.execute('UPDATE tasks SET body=?, checksum=? WHERE ordinal=0',
                               (parallel.encoded(value), parallel.digest(value)))
        with self.assertRaisesRegex(parallel.StateError, 'Damaged plan tasks digest'):
            parallel.finalize(self.root)
        with sqlite3.connect(store.database) as connection:
            connection.execute("INSERT INTO tasks VALUES (5, 'foreign', '{}', '')")
        with self.assertRaisesRegex(parallel.StateError, 'Unexpected plan tasks'):
            parallel.finalize(self.root)

    def test_interrupted_plan_build_is_safe_to_retry(self):
        stamp = parallel.input_stamp

        def interrupted(path):
            if Path(path).parent.parent.name == 'episode-2':
                raise KeyboardInterrupt()
            return stamp(path)

        with patch.object(parallel, 'input_stamp', side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            parallel.prepare(self.root)
        work = parallel.work_directory(self.output)
        self.assertEqual(list(work.iterdir()), [])
        store = parallel.prepare(self.root)
        self.assertEqual(store.audit(), (0, 0, 5))
        self.assertEqual(parallel.worker(self.root), 0)
        self.assertEqual(parallel.finalize(self.root), (5, 8))

    def test_plan_publish_interruption_keeps_a_usable_database(self):
        with patch.object(parallel, 'sync_directory', side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            parallel.prepare(self.root)
        store = parallel.Store(self.root, self.output)
        self.assertEqual(len(store.plan['tasks']), 5)
        stamp = store.database.read_bytes()
        parallel.prepare(self.root)
        self.assertEqual(store.database.read_bytes(), stamp)

    def test_episode_failure_preserves_old_report_and_retry_completes(self):
        self.output.write_text('previous report\n')
        store = parallel.prepare(self.root)
        original = detector.read_episode

        def read(root, record):
            if record['episode_id'] == 'episode-1':
                raise OSError('temporary read failure')
            return original(root, record)

        with patch.object(detector, 'read_episode', side_effect=read):
            self.assertEqual(parallel.worker(self.root), 1)
        self.assertEqual(store.audit(), (4, 1, 0))
        error = store.work / 'errors/3.err'
        self.assertIn('temporary read failure', error.read_text())
        with self.assertRaisesRegex(parallel.StateError, '1 failed'):
            parallel.finalize(self.root)
        self.assertEqual(self.output.read_text(), 'previous report\n')
        parallel.prepare(self.root)
        self.assertEqual(parallel.worker(self.root), 0)
        self.assertFalse(error.exists())
        self.assertEqual(parallel.finalize(self.root), (5, 8))

    def test_audit_checks_unfinished_inputs_concurrently(self):
        store = parallel.prepare(self.root)
        original = parallel.check_inputs
        ready = threading.Barrier(len(store.plan['tasks']))

        def check(stamps):
            ready.wait(timeout=10)
            original(stamps)

        with patch.object(parallel, 'check_inputs', side_effect=check):
            self.assertEqual(store.audit(), (0, 0, 5))
        path = Path(store.plan['tasks'][-1]['inputs'][-1]['path'])
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            store.audit()

    def test_audit_counts_errors_only_for_uncommitted_results(self):
        store = parallel.prepare(self.root)
        parallel.worker(self.root, worker_id=0, worker_count=5)
        (store.work / 'errors/1.err').write_text('stale failure')
        (store.work / 'errors/3.err').write_text('failed episode')
        (store.work / 'errors/5.err').symlink_to(store.work / 'missing-error')
        self.assertEqual(store.audit(), (1, 1, 3))
        with self.assertRaisesRegex(parallel.StateError, '1 failed, 3 unfinished'):
            store.audit(complete=True)

    def test_audit_rejects_result_symlinks_including_dangling_links(self):
        store = parallel.prepare(self.root)
        parallel.worker(self.root, worker_id=0, worker_count=5)
        path = store.work / 'records/1.json'
        saved = store.work / 'saved-result.json'
        path.rename(saved)
        for target in (saved, store.work / 'missing-result'):
            with self.subTest(target=target):
                path.symlink_to(target)
                try:
                    with self.assertRaisesRegex(parallel.StateError, 'Symlink result'):
                        store.audit()
                finally:
                    path.unlink()
        saved.rename(path)
        self.assertEqual(store.audit(), (1, 0, 4))

    def test_result_commit_interruption_and_report_publish_interruption(self):
        store = parallel.prepare(self.root)
        original = parallel.atomic_json

        def interrupted(path, value):
            original(path, value)
            raise KeyboardInterrupt()

        with patch.object(parallel, 'atomic_json', side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            parallel.worker(self.root)
        self.assertEqual(store.audit(), (1, 0, 4))
        parallel.prepare(self.root)
        self.assertEqual(parallel.worker(self.root), 0)
        # A crash after report rename but before published.json can be finalized again.
        with patch.object(parallel, 'atomic_json', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            parallel.finalize(self.root)
        report = self.output.read_bytes()
        self.assertFalse((store.work / 'published.json').exists())
        self.assertEqual(parallel.finalize(self.root), (5, 8))
        self.assertEqual(self.output.read_bytes(), report)

    def test_rejects_changed_thresholds_inputs_and_detector(self):
        store = parallel.prepare(self.root)
        with self.assertRaisesRegex(parallel.StateError, 'thresholds changed'):
            parallel.prepare(self.root, config=replace(detector.Config(), max_boundary_idle_s=6))
        with (patch.object(parallel, 'file_digest', return_value='different code'),
              self.assertRaisesRegex(parallel.StateError, 'code changed')):
            parallel.worker(self.root)
        path = Path(store.plan['tasks'][0]['inputs'][0]['path'])
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            parallel.prepare(self.root)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            parallel.worker(self.root)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            parallel.finalize(self.root)

    def test_damaged_result_and_foreign_records_block_publication(self):
        store = parallel.prepare(self.root)
        parallel.worker(self.root)
        path = store.work / 'records/1.json'
        original = path.read_bytes()
        value = json.loads(original)
        value['results'] = []
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(parallel.StateError, 'Damaged result'):
            parallel.finalize(self.root)
        path.write_bytes(original)
        (store.work / 'records/999.json').write_text('{}')
        with self.assertRaisesRegex(parallel.StateError, 'Unexpected files'):
            parallel.finalize(self.root)
        self.assertFalse(self.output.exists())

    def test_published_report_edit_is_detected(self):
        parallel.prepare(self.root)
        parallel.worker(self.root)
        parallel.finalize(self.root)
        self.output.write_text('changed\n')
        with self.assertRaisesRegex(parallel.StateError, 'Published report changed'):
            parallel.prepare(self.root)
        with self.assertRaisesRegex(parallel.StateError, 'Published report changed'):
            parallel.finalize(self.root)

    def test_slurm_phases_and_threshold_overrides(self):
        result = self.cli('prepare', '--max-boundary-idle-s', '6', direct=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for rank in range(2):
            env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_NTASKS='2')
            result = self.cli('worker', env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = self.cli('finalize')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), '')
        result = self.cli('prepare', '--max-boundary-idle-s', '6',
                          env=dict(os.environ, SLURM_PROCID='0', SLURM_NTASKS='2'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require one process', result.stderr)

    def test_invalid_worker_environment_is_logged(self):
        store = parallel.prepare(self.root)
        result = self.cli('worker', env=dict(os.environ, SLURM_PROCID='bad', SLURM_NTASKS='2'))
        self.assertNotEqual(result.returncode, 0)
        logs = list((store.work / 'logs').iterdir())
        self.assertEqual(len(logs), 1)
        self.assertIn('Traceback', logs[0].read_text())

    def test_worker_before_prepare_leaves_diagnostics_and_prepare_recovers(self):
        result = self.cli('worker', '--worker-id', '0', '--worker-count', '1')
        self.assertNotEqual(result.returncode, 0)
        logs = list((parallel.work_directory(self.output) / 'logs').iterdir())
        self.assertEqual(len(logs), 1)
        self.assertIn('plan.sqlite', logs[0].read_text())
        store = parallel.prepare(self.root)
        self.assertEqual(list((store.work / 'logs').iterdir()), [])
        self.assertEqual(store.audit(), (0, 0, 5))

    def test_protects_input_paths_and_rejects_duplicate_or_empty_manifest(self):
        for output in (self.manifest, self.root / 'episodes/report.jsonl', self.root / 'report.txt'):
            with self.assertRaises(ValueError):
                parallel.prepare(self.root, output)
        self.manifest.write_text(json.dumps(self.records[0]) + '\n' + json.dumps(self.records[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate episode_id'):
            parallel.prepare(self.root)
        self.manifest.write_text('\n')
        with self.assertRaisesRegex(ValueError, 'empty'):
            parallel.prepare(self.root)

    def test_work_directory_is_outside_repository_exports(self):
        report = parallel.ROOT / 'dataset/export/ABC-130K/anomalies.jsonl'
        self.assertEqual(parallel.work_directory(report),
                         parallel.ROOT / 'dataset/work/ABC-130K/anomalies.jsonl.anomaly-work')
        self.assertEqual(parallel.work_directory(self.output),
                         self.output.with_name('anomalies.jsonl.anomaly-work'))


if __name__ == '__main__':
    unittest.main()
