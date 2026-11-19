import json
import os
import shutil
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
import pyarrow as pa
import pyarrow.parquet as pq

from loom_data import paths
from support import expected_report, make_dataset
from loom_data.check import anomalies as detector
from loom_data.check import job as parallel


class ParallelDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.template = Path(temp.name).resolve() / 'dataset'
        # Episodes 0, 1, 3 and 4 are still for exactly 5 s (two boundary records each).
        # Episode 2 is too short for any anomaly, which must still count as committed work.
        make_dataset(cls.template, {f'episode-{i}': {'left_gripper': np.zeros(3 if i == 2 else 151)}
                                    for i in range(5)})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'dataset'
        shutil.copytree(self.template, self.root)
        self.output = self.root / 'anomalies.jsonl'
        self.info = self.root / 'meta/info.json'
        self.data = self.root / 'data/chunk-000/file-000.parquet'
        self.metadata = self.root / 'meta/episodes/chunk-000/file-000.parquet'

    def rewrite_episodes(self, change):
        """Rewrite the episode metadata rows; keep info.json's total in step."""
        rows = change(pq.read_table(self.metadata).to_pylist())
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(self.metadata)), self.metadata)
        info = json.loads(self.info.read_text())
        info['total_episodes'] = len(rows)
        self.info.write_text(json.dumps(info))

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, '-m', 'loom_data', 'check', args[0], str(self.root), *args[1:]],
                              env=env, capture_output=True, text=True, check=False)

    def test_local_cli_matches_the_rules_and_repeat_skips_all_results(self):
        expected = expected_report(self.root)
        self.assertEqual(len(expected.splitlines()), 8)
        result = self.cli('run', '--workers', '3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), expected)
        store = parallel.Store(self.root, self.output)
        stamps = {p: p.stat().st_mtime_ns for p in (store.work / 'records').iterdir()}
        self.assertEqual(store.result(store.plan['tasks'][2]), [])
        result = self.cli('run', '--workers', '2')
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

        def read(dataset, episode):
            called.append(episode['episode_id'])
            return original(dataset, episode)

        with patch.object(detector, 'read_episode', side_effect=read):
            for rank in range(2):
                self.assertEqual(parallel.worker(self.root, worker_id=rank, worker_count=2), 0)
        self.assertCountEqual(called, ['episode-1', 'episode-2', 'episode-4'])
        self.assertEqual(committed, {p: p.read_bytes() for p in committed})
        self.assertEqual(parallel.finalize(self.root), (5, 8))
        self.assertEqual([json.loads(line)['index'] for line in self.output.read_text().splitlines()],
                         [0, 0, 1, 1, 3, 3, 4, 4])
        self.assertEqual(self.output.read_text(), expected_report(self.root))

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
        self.assertEqual(decoded, [1, 4])
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
            # info.json is stamped first; the episodes' data file while tasks stream into the plan.
            if Path(path) == self.data:
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

        def read(dataset, episode):
            if episode['episode_id'] == 'episode-1':
                raise OSError('temporary read failure')
            return original(dataset, episode)

        with patch.object(detector, 'read_episode', side_effect=read):
            self.assertEqual(parallel.worker(self.root), 1)
        self.assertEqual(store.audit(), (4, 1, 0))
        error = store.work / 'errors/1.err'
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
        (store.work / 'errors/0.err').write_text('stale failure')
        (store.work / 'errors/1.err').write_text('failed episode')
        (store.work / 'errors/2.err').symlink_to(store.work / 'missing-error')
        self.assertEqual(store.audit(), (1, 1, 3))
        with self.assertRaisesRegex(parallel.StateError, '1 failed, 3 unfinished'):
            store.audit(complete=True)

    def test_audit_rejects_result_symlinks_including_dangling_links(self):
        store = parallel.prepare(self.root)
        parallel.worker(self.root, worker_id=0, worker_count=5)
        path = store.work / 'records/0.json'
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
        self.assertEqual(path, self.data)
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
        path = store.work / 'records/0.json'
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
        result = self.cli('prepare', '--max-boundary-idle-s', '6')
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

    def test_tasks_address_episodes_by_episode_index_and_stamp_their_data_file(self):
        store = parallel.prepare(self.root)
        self.assertEqual(list(store.plan['tasks']), [
            {'index': i, 'record': {'episode_id': f'episode-{i}'}, 'inputs': [parallel.input_stamp(self.data)]}
            for i in range(5)])
        self.assertEqual(store.plan['manifest'], parallel.input_stamp(self.info))

    def test_rebuilt_dataset_is_rejected_through_info_json(self):
        store = parallel.prepare(self.root)
        self.assertEqual(parallel.worker(self.root, worker_id=0, worker_count=2), 0)
        self.info.write_text(self.info.read_text() + '\n')
        for call in (lambda: parallel.prepare(self.root), lambda: parallel.worker(self.root),
                     lambda: parallel.finalize(self.root)):
            with self.assertRaisesRegex(ValueError, 'Source changed since prepare.*info.json'):
                call()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list((store.work / 'records').iterdir())), 3)

    def test_worker_refuses_episodes_that_no_longer_match_the_plan(self):
        store = parallel.prepare(self.root)
        saved = self.metadata.read_bytes(), self.info.read_bytes()
        stamp = os.stat(self.info)

        def rename(rows):
            rows[3]['episode_id'] = 'renamed'
            return rows

        # Episode metadata rewritten behind an unchanged info.json stamp.
        self.rewrite_episodes(rename)
        self.info.write_bytes(saved[1])
        os.utime(self.info, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(parallel.worker(self.root), 1)
        self.assertEqual(store.audit(), (4, 1, 0))
        self.assertIn('Dataset episodes changed since prepare', (store.work / 'errors/3.err').read_text())
        with self.assertRaisesRegex(parallel.StateError, '1 failed'):
            parallel.finalize(self.root)
        self.metadata.write_bytes(saved[0])
        self.assertEqual(parallel.worker(self.root), 0)
        self.assertEqual(parallel.finalize(self.root), (5, 8))

    def test_protects_input_paths_and_rejects_invalid_duplicate_or_empty_datasets(self):
        for output in (self.info, self.root / 'meta/report.jsonl', self.data.with_name('report.jsonl'),
                       self.root / 'videos/report.jsonl', self.root / 'report.txt'):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'Output must be'):
                parallel.prepare(self.root, output)
        work = parallel.work_directory(self.output)
        saved = self.metadata.read_bytes(), self.info.read_bytes()

        def duplicate(rows):
            rows[4]['episode_id'] = rows[0]['episode_id']
            return rows

        def gap(rows):
            rows[4]['episode_index'] = 7
            return rows

        def unsafe(rows):
            rows[1]['episode_id'] = '../outside'
            return rows

        for change in (duplicate, gap, unsafe, lambda rows: rows[:4] + rows[3:4], lambda rows: []):
            self.rewrite_episodes(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'invalid, duplicate or missing'):
                parallel.prepare(self.root)
            self.metadata.write_bytes(saved[0])
        # info.json disagreeing with the episode metadata, or not describing a contract dataset.
        for old, new, message in (('"total_episodes": 5', '"total_episodes": 4', 'invalid, duplicate or missing'),
                                  ('"fps": 30', '"fps": 15', 'not a contract dataset'),
                                  ('"v3.0"', '"v2.1"', 'not a contract dataset'),
                                  ('{', '', 'not an exported dataset')):
            text = saved[1].decode()
            self.assertIn(old, text)
            self.info.write_text(text.replace(old, new, 1))
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, message):
                parallel.prepare(self.root)
        self.info.unlink()
        with self.assertRaisesRegex(ValueError, 'not an exported dataset'):
            parallel.prepare(self.root)
        self.assertFalse((work / 'plan.sqlite').exists())
        self.assertFalse(self.output.exists())

    def test_work_directory_is_outside_repository_exports(self):
        report = paths.ROOT / 'dataset/export/ABC-130K/anomalies.jsonl'
        self.assertEqual(parallel.work_directory(report),
                         paths.ROOT / 'dataset/work/ABC-130K/anomalies.jsonl.anomaly-work')
        self.assertEqual(parallel.work_directory(self.output),
                         self.output.with_name('anomalies.jsonl.anomaly-work'))


if __name__ == '__main__':
    unittest.main()
