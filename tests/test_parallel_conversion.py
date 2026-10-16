"""Real media fixtures plus crashes at persistence boundaries."""
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert import abc130k, agibotworld2026, export_common, parallel
from scripts.convert.parallel_adapters import Adapter, make_config
from test_convert_hifi_umi import make_source as hifi_source, make_shard as hifi_shard
from test_convert_molmoact2 import make_source as molmo_source, make_full_source as molmo_full_source
import test_convert_galaxea as galaxea_tests


class InputStampTests(unittest.TestCase):
    def test_file_links_missing_paths_and_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.write_bytes(b'input')
            info = source.stat()
            expected = {'path': str(source), 'size': 5, 'mtime_ns': info.st_mtime_ns}
            self.assertEqual(parallel.input_stamp(source), expected)
            link = root / 'link'
            link.symlink_to(source)
            self.assertEqual(parallel.input_stamp(link), dict(expected, path=str(link)))
            source.unlink()
            loop = root / 'loop'
            loop.symlink_to(loop)
            for path in (source, link, loop):
                self.assertEqual(parallel.input_stamp(path),
                                 {'path': str(path), 'size': None, 'mtime_ns': None})
            with self.assertRaisesRegex(ValueError, 'Expected source file'):
                parallel.input_stamp(root)
            source.write_bytes(b'input')
            missing = source / 'child'
            self.assertEqual(parallel.input_stamp(missing),
                             {'path': str(missing), 'size': None, 'mtime_ns': None})


def add_hifi_episode(source, index):
    original = source / 'episodes/episode_000007'
    destination = source / 'episodes' / f'episode_{index:06d}'
    shutil.copytree(original, destination)
    meta = json.loads((destination / 'episode.json').read_text())
    meta['episode_index'] = index
    (destination / 'episode.json').write_text(json.dumps(meta))
    table = pq.read_table(destination / 'data.parquet')
    column = table.schema.get_field_index('episode_index')
    table = table.set_column(column, 'episode_index', pa.array([index] * len(table)))
    pq.write_table(table, destination / 'data.parquet')


def manifest(output):
    return [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]


class HifiShardTests(unittest.TestCase):
    def test_full_shard_resume_keeps_committed_episodes_after_video_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'raw', root / 'out'
            hifi_shard(source)
            hifi_shard(source, chunk=1)
            config = make_config('hifi_umi', source)
            store = parallel.prepare(config, output)
            self.assertEqual(len(store.plan['tasks']), 2)
            self.assertEqual(len(store.ids), 4)
            from scripts.convert import hifi_umi
            extract = hifi_umi.extract_video_segment

            def interrupted(video, *args):
                if video['from_timestamp'] > 0:
                    raise ValueError('transient decode error')
                return extract(video, *args)

            with patch.object(hifi_umi, 'extract_video_segment', side_effect=interrupted):
                self.assertEqual(parallel.worker(output, 0, 2), 1)
            self.assertEqual(parallel.worker(output, 1, 2), 0)
            committed = 'chunk_0000_part_0000_episode_000007'
            self.assertIsNotNone(store.state(committed))
            self.assertIsNone(store.state('chunk_0000_part_0000_episode_000008'))
            self.assertFalse(any((store.work / 'staging').iterdir()))
            parallel.prepare(config, output)
            self.assertIsNotNone(store.state(committed))
            self.assertEqual(len(store.audit()[0]), 3)
            converted = []
            convert = hifi_umi.ShardReader.convert_episode

            def resumed(reader, index, *args):
                converted.append(index)
                return convert(reader, index, *args)

            with patch.object(hifi_umi.ShardReader, 'convert_episode', resumed):
                self.assertEqual(parallel.worker(output, 0, 1), 0)
            self.assertEqual(converted, [8])
            parallel.finalize(output)
            self.assertEqual(len(manifest(output)), 4)

    def test_shard_workers_publish_and_resume_with_global_episode_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hifi_shard(root / 'raw')
            hifi_shard(root / 'raw', chunk=1)
            config = make_config('hifi_umi', root / 'raw', limit_episodes=3)
            store = parallel.prepare(config, root / 'out')
            self.assertEqual([len(t['episode_ids']) for t in store.plan['tasks']], [2, 1])
            for rank in range(2):
                self.assertEqual(parallel.worker(root / 'out', rank, 2), 0)
            parallel.finalize(root / 'out')
            self.assertEqual(len(manifest(root / 'out')), 3)
            self.assertEqual(len({r['episode_id'] for r in manifest(root / 'out')}), 3)
            parallel.prepare(config, root / 'out')
            with patch.object(Adapter, 'shard', side_effect=AssertionError('must skip published episodes')):
                self.assertEqual(parallel.worker(root / 'out', 0, 1), 0)
            parallel.finalize(root / 'out')


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / 'raw', self.root / 'export'
        hifi_source(self.source)
        self.config = make_config('hifi_umi', self.source)
        self.store = parallel.prepare(self.config, self.output)
        self.episode_id = self.store.ids[0]

    def finish(self, count=1):
        for rank in range(count):
            self.assertEqual(parallel.worker(self.output, rank, count), 0)
        parallel.finalize(self.output)
        self.assertTrue(parallel.Store.load(self.output).published())

    def test_repository_export_keeps_work_separate(self):
        output = self.root / 'dataset/export/hifi'
        work = self.root / 'dataset/work/hifi.work'
        with patch.object(parallel, 'ROOT', self.root):
            store = parallel.prepare(self.config, output)
            self.assertEqual(store.work, work)
            self.assertEqual(parallel.worker(output, 0, 1), 0)
            parallel.finalize(output)
            self.assertTrue(parallel.Store.load(output).published())
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_record_then_directory_crash_boundaries(self):
        original_atomic = parallel.atomic_json
        original_rename = Path.rename
        for boundary in ('before_record', 'after_record', 'before_directory', 'after_directory'):
            output = self.root / boundary
            store = parallel.prepare(self.config, output)

            def atomic(path, value, boundary=boundary):
                is_record = path.parent.name == 'records'
                if is_record and boundary == 'before_record':
                    raise KeyboardInterrupt('simulated worker death')
                original_atomic(path, value)
                if is_record and boundary == 'after_record':
                    raise KeyboardInterrupt('simulated worker death')

            def rename(path, target, boundary=boundary, store=store):
                is_episode = Path(target).parent == store.dataset / 'episodes'
                if is_episode and boundary == 'before_directory':
                    raise KeyboardInterrupt('simulated worker death')
                result = original_rename(path, target)
                if is_episode and boundary == 'after_directory':
                    raise KeyboardInterrupt('simulated worker death')
                return result

            with self.subTest(boundary=boundary), patch.object(parallel, 'atomic_json', side_effect=atomic), patch.object(Path, 'rename', rename):
                with self.assertRaises(KeyboardInterrupt):
                    parallel.worker(output, 0, 1)
            self.assertFalse(output.exists())
            record = store.work / 'records' / f'{self.episode_id}.json'
            self.assertEqual(record.exists(), boundary != 'before_record')
            stale = store.work / 'staging' / 'killed-attempt'
            stale.mkdir()
            (stale / 'partial.mp4').write_bytes(b'partial')
            parallel.prepare(self.config, output)
            self.assertFalse(stale.exists())
            with patch.object(Adapter, 'episode', autospec=True, side_effect=Adapter.episode) as convert:
                for rank in range(3):
                    self.assertEqual(parallel.worker(output, rank, 3), 0)
                if boundary == 'after_directory':
                    convert.assert_not_called()
            parallel.finalize(output)
            self.assertEqual(len(manifest(output)), 1)

    def test_worker_count_32_then_8_then_3_preserves_commits(self):
        # A new output gets a fixed 35-task plan; the existing one remains frozen.
        for index in range(8, 42):
            add_hifi_episode(self.source, index)
        output = self.root / 'many'
        store = parallel.prepare(self.config, output)
        self.assertEqual(len(store.ids), 35)
        self.assertEqual(parallel.worker(output, 0, 32), 0)
        before = {p: p.stat().st_mtime_ns for p in (store.dataset / 'episodes').rglob('*') if p.is_file()}
        self.assertEqual(len(list((store.dataset / 'episodes').iterdir())), 2)
        parallel.prepare(self.config, output)
        for rank in (0, 1, 2):
            self.assertEqual(parallel.worker(output, rank, 8), 0)
        parallel.prepare(self.config, output)
        for rank in range(3):
            self.assertEqual(parallel.worker(output, rank, 3), 0)
        self.assertTrue(all(p.stat().st_mtime_ns == stamp for p, stamp in before.items()))
        parallel.finalize(output)
        records = manifest(output)
        self.assertEqual([r['episode_id'] for r in records], store.ids)
        self.assertEqual(len(records), len({r['episode_id'] for r in records}))

    def test_failed_episode_does_not_block_later_tasks_and_retries(self):
        add_hifi_episode(self.source, 8)
        output = self.root / 'retry'
        with patch('sys.stdout', new_callable=io.StringIO) as initial:
            store = parallel.prepare(self.config, output)
        self.assertIn('0 exported, 0 skipped, 0 failed, 2 unfinished', initial.getvalue())
        convert = Adapter.episode

        def fail_first(adapter, task, destination):
            if destination.name == self.episode_id:
                raise ValueError('injected conversion error')
            return convert(adapter, task, destination)

        with patch.object(Adapter, 'episode', fail_first):
            self.assertEqual(parallel.worker(output, 0, 1), 1)
        self.assertIsNone(store.state(self.episode_id))
        self.assertIsNotNone(store.state('episode_000008'))
        self.assertIn('injected conversion error', (store.work / 'errors' / f'{self.episode_id}.err').read_text())
        with patch('sys.stdout', new_callable=io.StringIO) as resumed:
            parallel.prepare(self.config, output)
        self.assertIn('1 exported, 0 skipped, 1 failed, 0 unfinished', resumed.getvalue())
        for rank in range(2):
            self.assertEqual(parallel.worker(output, rank, 2), 0)
        self.assertEqual(list((store.work / 'errors').iterdir()), [])
        # A crash after committing the episode but before removing its old error
        # must not make the completed episode count as failed.
        (store.work / 'errors' / f'{self.episode_id}.err').write_text('stale conversion error')
        with patch('sys.stdout', new_callable=io.StringIO) as completed:
            parallel.prepare(self.config, output)
        self.assertIn('2 exported, 0 skipped, 0 failed, 0 unfinished', completed.getvalue())
        (store.work / 'errors' / f'{self.episode_id}.err').unlink()
        parallel.finalize(output)
        self.assertEqual(len(manifest(output)), 2)
        self.assertEqual(list((store.work / 'errors').iterdir()), [])

    def test_finalize_skips_failed_episodes(self):
        add_hifi_episode(self.source, 8)
        output = self.root / 'skip_failed'
        store = parallel.prepare(self.config, output)
        with patch.object(Adapter, 'episode', side_effect=ValueError('bad source')):
            self.assertEqual(parallel.worker(output, 0, 2), 1)
        self.assertEqual(parallel.worker(output, 1, 2), 0)
        parallel.finalize(output)
        self.assertEqual([r['episode_id'] for r in manifest(output)], ['episode_000008'])
        self.assertTrue((store.work / 'errors' / f'{self.episode_id}.err').is_file())
        self.assertTrue(store.published())
        parallel.finalize(output)

    def test_finalize_interruption_and_already_published_noop(self):
        self.assertEqual(parallel.worker(self.output, 0, 1), 0)
        original_atomic = parallel.atomic_text

        def stop_after_manifest(path, text):
            original_atomic(path, text)
            if path.name == 'episodes.jsonl':
                raise KeyboardInterrupt()

        with patch.object(parallel, 'atomic_text', side_effect=stop_after_manifest):
            with self.assertRaises(KeyboardInterrupt):
                parallel.finalize(self.output)
        parallel.prepare(self.config, self.output)
        original_rename = Path.rename

        def stop_after_publish(path, destination):
            result = original_rename(path, destination)
            if destination == self.output:
                raise KeyboardInterrupt()
            return result

        with patch.object(Path, 'rename', stop_after_publish):
            with self.assertRaises(KeyboardInterrupt):
                parallel.finalize(self.output)
        with patch.object(parallel.subprocess, 'Popen') as spawn:
            parallel.run(self.config, self.output, workers=7)
            spawn.assert_not_called()
        self.assertTrue(parallel.Store.load(self.output).published())
        self.assertFalse(self.store.dataset.exists())

    def test_corrupt_or_unexpected_results_are_rejected(self):
        self.assertEqual(parallel.worker(self.output, 0, 1), 0)
        record_path = self.store.work / 'records' / f'{self.episode_id}.json'
        original = record_path.read_bytes()
        record_path.unlink()
        with self.assertRaisesRegex(parallel.StateError, 'without record'):
            parallel.prepare(self.config, self.output)
        record_path.write_bytes(b'{')
        with self.assertRaisesRegex(parallel.StateError, 'Cannot read'):
            parallel.prepare(self.config, self.output)
        record_path.write_bytes(original)
        extra = self.store.dataset / 'episodes' / 'unexpected'
        extra.mkdir()
        with self.assertRaisesRegex(parallel.StateError, 'Unexpected episode'):
            parallel.finalize(self.output)
        extra.rmdir()
        next((self.store.dataset / 'episodes' / self.episode_id / 'rgb').glob('*.mp4')).unlink()
        with self.assertRaisesRegex(parallel.StateError, 'damaged recovery'):
            parallel.prepare(self.config, self.output)

    def test_config_and_source_changes_reject_resume(self):
        changed = copy.deepcopy(self.config)
        changed['params']['gripper_open_rad'] += .1
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            parallel.prepare(changed, self.output)
        source_file = self.source / 'episodes' / self.episode_id / 'episode.json'
        source_file.write_text(source_file.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            parallel.prepare(self.config, self.output)

    def test_audit_rejects_symlinks_including_dangling_links(self):
        self.assertEqual(parallel.worker(self.output, 0, 1), 0)
        episode = self.store.dataset / 'episodes' / self.episode_id
        paths = [self.store.work / 'records' / f'{self.episode_id}.json',
                 episode, episode / 'state', episode / 'rgb',
                 next((episode / 'rgb').glob('*.mp4'))]
        for path in paths:
            saved = self.root / 'saved'
            path.rename(saved)
            try:
                for target in (saved, self.root / 'does-not-exist'):
                    with self.subTest(path=path, dangling=not target.exists()):
                        path.symlink_to(target)
                        try:
                            with self.assertRaises(parallel.StateError):
                                self.store.audit()
                        finally:
                            path.unlink()
            finally:
                saved.rename(path)
        self.assertEqual(len(self.store.audit()[0]), 1)

    def test_audit_rejects_changed_sizes_empty_and_extra_files(self):
        self.assertEqual(parallel.worker(self.output, 0, 1), 0)
        rgb = self.store.dataset / 'episodes' / self.episode_id / 'rgb'
        video = next(rgb.glob('*.mp4'))
        original = video.read_bytes()
        for contents in (b'', original + b'changed'):
            with self.subTest(empty=not contents):
                video.write_bytes(contents)
                with self.assertRaisesRegex(parallel.StateError, 'damaged recovery'):
                    self.store.audit()
        video.write_bytes(original)
        extra = rgb / 'extra.mp4'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(parallel.StateError, 'damaged recovery'):
            self.store.audit()
        extra.unlink()
        self.assertEqual(len(self.store.audit()[0]), 1)

    def test_plan_is_frozen_and_unknown_outputs_are_rejected(self):
        add_hifi_episode(self.source, 8)
        self.assertEqual(len(parallel.prepare(self.config, self.output).ids), 1)
        unknown = self.root / 'unknown'
        unknown.mkdir()
        with self.assertRaisesRegex(parallel.StateError, 'without a matching'):
            parallel.prepare(self.config, unknown)
        plan = self.store.work / 'plan.json'
        data = json.loads(plan.read_text())
        data['tasks'].reverse()
        data['config']['limit'] = 2
        plan.write_text(json.dumps(data))
        with self.assertRaisesRegex(parallel.StateError, 'checksum'):
            parallel.Store.load(self.output)

    def test_real_local_processes_and_idempotent_cli(self):
        add_hifi_episode(self.source, 8)
        output = self.root / 'processes'
        command = [sys.executable, '-m', 'scripts.convert.parallel', 'run', '--dataset', 'hifi_umi',
                   '--input-dir', str(self.source), '--output-dir', str(output)]
        for count in (2, 3):
            result = subprocess.run([*command, '--workers', str(count)], cwd=parallel.ROOT,
                                    capture_output=True, text=True, timeout=90, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(manifest(output)), 2)
        logs = list(output.with_name(output.name + '.work').glob('logs/worker-*.err'))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all('Worker exit code: 0' in p.read_text() for p in logs))

    def test_cli_input_check_failure_keeps_task_and_traceback(self):
        source_file = self.source / 'episodes' / self.episode_id / 'episode.json'
        source_file.write_text(source_file.read_text() + ' ')
        command = [sys.executable, '-m', 'scripts.convert.parallel', 'worker',
                   '--output-dir', str(self.output), '--worker-id', '0', '--worker-count', '1']
        for _ in range(2):
            result = subprocess.run(command, cwd=parallel.ROOT, capture_output=True,
                                    text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
        logs = list((self.store.work / 'logs').glob('worker-*.err'))
        self.assertEqual(len(logs), 2)  # Retries must preserve earlier diagnostics.
        for path in logs:
            text = path.read_text()
            for expected in ('host=', 'pid=', 'Worker 0/1', 'Task 0:',
                             self.episode_id, 'Traceback', 'Source changed since prepare'):
                self.assertIn(expected, text)
        self.assertEqual(list((self.store.work / 'errors').iterdir()), [])

    def test_cli_missing_slurm_environment_is_logged_before_worker_starts(self):
        output = self.root / 'not_prepared'
        env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.convert.parallel', 'worker', '--output-dir', str(output)],
            env=env, cwd=parallel.ROOT, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        log, = output.with_name(output.name + '.work').glob('logs/worker-*.err')
        self.assertIn("KeyError: 'SLURM_PROCID'", log.read_text())
        self.assertIn('Traceback', log.read_text())
        # A diagnostics-only directory must not prevent a subsequent prepare.
        parallel.prepare(self.config, output)

    def test_cli_dependency_failure_and_interrupt_are_logged(self):
        code = '''
import sys
from unittest.mock import patch
from scripts.convert import parallel
kind = sys.argv.pop()
error = ImportError('injected dependency failure') if kind == 'import' else KeyboardInterrupt()
with patch.object(parallel, 'configure_worker', side_effect=error):
    sys.exit(parallel.main())
'''
        for kind, status, expected in (('import', 1, 'ImportError: injected dependency failure'),
                                       ('interrupt', 130, 'KeyboardInterrupt')):
            output = self.root / kind
            result = subprocess.run(
                [sys.executable, '-c', code, 'worker', '--output-dir', str(output),
                 '--worker-id', '0', '--worker-count', '1', kind],
                cwd=parallel.ROOT, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, status, result.stderr)
            log, = output.with_name(output.name + '.work').glob('logs/worker-*.err')
            self.assertIn(expected, log.read_text())
            self.assertIn('Traceback', log.read_text())

    def test_cli_native_abort_keeps_fault_stack(self):
        code = '''
import os, resource, sys
from unittest.mock import patch
from scripts.convert import parallel
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with patch.object(parallel, 'configure_worker', side_effect=os.abort):
    sys.exit(parallel.main())
'''
        result = subprocess.run(
            [sys.executable, '-c', code, 'worker', '--output-dir', str(self.output),
             '--worker-id', '0', '--worker-count', '1'],
            cwd=parallel.ROOT, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr)
        log, = (self.store.work / 'logs').glob('worker-*.err')
        self.assertIn('Fatal Python error: Aborted', log.read_text())
        self.assertIn('parallel.py', log.read_text())

    def test_sigkill_leaves_an_attempt_that_next_run_recovers(self):
        # Kill after persisting metadata; no Python finally blocks can clean up.
        code = '''
import os, signal, sys
from unittest.mock import patch
from scripts.convert import parallel
original = parallel.atomic_json
def committed_record(path, value):
    original(path, value)
    if path.parent.name == 'records':
        os.kill(os.getpid(), signal.SIGKILL)
with patch.object(parallel, 'atomic_json', side_effect=committed_record):
    parallel.worker(sys.argv[1], 0, 1)
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.output)], cwd=parallel.ROOT,
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stdout + result.stderr)
        self.assertTrue((self.store.work / 'records' / f'{self.episode_id}.json').exists())
        self.assertTrue(list((self.store.work / 'staging').iterdir()))
        self.assertIsNone(self.store.state(self.episode_id))
        parallel.run(self.config, self.output, workers=2)
        self.assertEqual(len(manifest(self.output)), 1)
        self.assertEqual(list((self.store.work / 'staging').iterdir()), [])

    def test_external_launcher_three_phase_cli_and_changed_worker_count(self):
        add_hifi_episode(self.source, 8)
        output = self.root / 'external_launcher'
        prepare_args = ['prepare', '--dataset', 'hifi_umi', '--input-dir', str(self.source),
                        '--output-dir', str(output)]

        def cli(args, rank=None, count=None):
            env = dict(os.environ)
            env.pop('SLURM_PROCID', None)
            env.pop('SLURM_NTASKS', None)
            if rank is not None:
                env.update(SLURM_PROCID=str(rank), SLURM_NTASKS=str(count))
            return subprocess.run([sys.executable, '-m', 'scripts.convert.parallel', *args],
                                  env=env, cwd=parallel.ROOT, capture_output=True,
                                  text=True, timeout=60, check=False)

        self.assertEqual(cli(prepare_args).returncode, 0)
        worker_args = ['worker', '--output-dir', str(output)]
        finalize_args = ['finalize', '--output-dir', str(output)]
        self.assertEqual(cli(worker_args, rank=0, count=2).returncode, 0)
        self.assertNotEqual(cli(finalize_args).returncode, 0)
        self.assertFalse(output.exists())
        self.assertEqual(cli(prepare_args).returncode, 0)
        for rank in range(3):
            result = cli(worker_args, rank=rank, count=3)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(cli(finalize_args).returncode, 0)
        self.assertEqual(len(manifest(output)), 2)
        before = {p: p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
        self.assertEqual(cli(prepare_args).returncode, 0)
        for rank in range(4):
            self.assertEqual(cli(worker_args, rank=rank, count=4).returncode, 0)
        self.assertEqual(cli(finalize_args).returncode, 0)
        self.assertTrue(all(p.stat().st_mtime_ns == stamp for p, stamp in before.items()))

    def test_prepare_and_finalize_refuse_multiple_slurm_tasks(self):
        with patch.dict(os.environ, {'SLURM_PROCID': '0', 'SLURM_NTASKS': '8'}):
            with self.assertRaisesRegex(ValueError, 'require one process'):
                parallel.prepare(self.config, self.output)
            with self.assertRaisesRegex(ValueError, 'require one process'):
                parallel.finalize(self.output)
            with self.assertRaisesRegex(ValueError, 'require one process'):
                parallel.run(self.config, self.output)


class SourceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_molmo_real_conversion_matches_standalone(self):
        from scripts.convert import molmoact2
        source = self.root / 'raw'
        molmo_source(source)
        output = self.root / 'parallel'
        config = make_config('molmoact2', source)
        parallel.prepare(config, output)
        self.assertEqual(parallel.worker(output, 0, 1), 0)
        parallel.finalize(output)
        reference = self.root / 'standalone'
        molmoact2.convert(source, reference)
        self.assertEqual(manifest(output), manifest(reference))
        for path in reference.rglob('*.parquet'):
            self.assertTrue(pq.read_table(path).equals(pq.read_table(output / path.relative_to(reference))))

    @unittest.skipUnless('libsvtav1' in av.codecs_available, 'AV1 encoder required')
    def test_molmo_full_dataset_resume_and_standalone_match(self):
        from scripts.convert import molmoact2
        source = molmo_full_source(self.root / 'raw')
        output = self.root / 'parallel'
        config = make_config('molmoact2', source)
        store = parallel.prepare(config, output)
        self.assertEqual(len(store.plan['tasks']), 3)
        self.assertTrue(all(t['layout'] == 'lerobot_v3' and len(t['inputs']) == 4 for t in store.plan['tasks']))
        self.assertIn(str(source / 'meta/episodes/chunk-001/file-000.parquet'),
                      [p['path'] for p in store.plan['shared_inputs']])
        convert = molmoact2.DatasetReader.convert_episode

        def interrupted(reader, index, *args):
            if index == 8:
                raise ValueError('transient AV1 error')
            return convert(reader, index, *args)

        with patch.object(molmoact2.DatasetReader, 'convert_episode', interrupted):
            self.assertEqual(parallel.worker(output, 0, 2), 0)
            self.assertEqual(parallel.worker(output, 1, 2), 1)
        self.assertIsNotNone(store.state('episode_000007'))
        self.assertIsNotNone(store.state('episode_000009'))
        self.assertIsNone(store.state('episode_000008'))
        parallel.prepare(config, output)
        converted = []

        def resumed(reader, index, *args):
            converted.append(index)
            return convert(reader, index, *args)

        with patch.object(molmoact2.DatasetReader, 'convert_episode', resumed):
            self.assertEqual(parallel.worker(output, 0, 1), 0)
        self.assertEqual(converted, [8])
        parallel.finalize(output)
        reference = self.root / 'standalone'
        molmoact2.convert(source, reference)
        self.assertEqual(manifest(output), manifest(reference))
        for path in reference.rglob('*.parquet'):
            self.assertTrue(pq.read_table(path).equals(pq.read_table(output / path.relative_to(reference))))

    @unittest.skipUnless('libsvtav1' in av.codecs_available, 'AV1 encoder required')
    def test_molmo_full_dataset_limits_and_bad_video_range_are_episode_local(self):
        from scripts.convert import molmoact2
        source = molmo_full_source(self.root / 'raw')
        meta = source / 'meta/episodes/chunk-000/file-000.parquet'
        rows = pq.read_table(meta).to_pylist()
        for key in molmoact2.CAMERAS:
            rows[0][f'videos/{key}/to_timestamp'] += .04
        pq.write_table(pa.Table.from_pylist(rows), meta)
        output = self.root / 'parallel'
        config = make_config('molmoact2', source, limit=3, limit_episodes=2)
        store = parallel.prepare(config, output)
        self.assertEqual(store.ids, ['episode_000007', 'episode_000008'])
        self.assertEqual(parallel.worker(output, 0, 1), 1)
        self.assertIsNone(store.state('episode_000007'))
        self.assertIsNotNone(store.state('episode_000008'))
        self.assertFalse(output.exists())

    def test_galaxea_skip_and_both_embodiments(self):
        fixture = galaxea_tests.GalaxeaExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture.source
        output = self.root / 'both'
        config = make_config('galaxea', source)
        parallel.prepare(config, output)
        for rank in range(2):
            self.assertEqual(parallel.worker(output, rank, 2), 0)
        parallel.finalize(output)
        self.assertEqual(len(manifest(output)), 2)
        path = next((source / 'r1lite').glob('data/*/*.parquet'))
        rows = pq.read_table(path).to_pydict()
        rows['action.chassis.velocities'][1][0] = .1
        pq.write_table(pa.table(rows), path)
        skipped_output = self.root / 'skip'
        store = parallel.prepare(config, skipped_output)
        self.assertEqual(parallel.worker(skipped_output, 0, 1), 0)
        self.assertEqual(store.state('r1lite_episode_000000')['outcome'], 'skipped')
        with patch.object(Adapter, 'episode', side_effect=AssertionError('must not reconvert')):
            self.assertEqual(parallel.worker(skipped_output, 0, 1), 0)
        parallel.finalize(skipped_output)
        self.assertEqual([r['episode_id'] for r in manifest(skipped_output)], ['r1pro_episode_000000'])

    def test_abc_mcap_adapter_with_real_codec(self):
        source = self.root / 'raw'
        episode = source / 'episode_1'
        episode.mkdir(parents=True)
        (episode / 'episode.mcap').touch()
        context = av.CodecContext.create('libx264', 'w')
        context.width = context.height = 32
        context.pix_fmt = 'yuv420p'
        context.time_base = Fraction(1, 30)
        context.options = {'bf': '0', 'threads': '1'}
        packets = []
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 70, np.uint8), format='rgb24')
            frame.pts = index
            packets.extend(context.encode(frame))
        packets.extend(context.encode(None))
        records = [('/instruction', 0, SimpleNamespace(data='Pick'))]
        for topic in abc130k.STATES:
            for time in (0, 100000000):
                records.append((topic, time, SimpleNamespace(position=[0.] * 6 if 'arm' in topic else [.5])))
        for index, packet in enumerate(packets):
            records.append(('/top-camera', index * 50000000, SimpleNamespace(data=bytes(packet), format='h264')))
        output = self.root / 'out'
        parallel.prepare(make_config('abc130k', source), output)
        with patch.object(abc130k, 'messages', side_effect=lambda path, topics: (r for r in records if r[0] in topics)):
            self.assertEqual(parallel.worker(output, 0, 1), 0)
        parallel.finalize(output)
        self.assertEqual(manifest(output)[0]['cameras'], ['top'])

    def test_plan_rejects_duplicate_ids_before_creating_output(self):
        source = self.root / 'raw'
        for group in ('a', 'b'):
            episode = source / group / 'episode_1'
            episode.mkdir(parents=True)
            (episode / 'episode.mcap').touch()
        output = self.root / 'out'
        with self.assertRaisesRegex(parallel.StateError, 'unique safe'):
            parallel.prepare(make_config('abc130k', source), output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name('out.work').exists())

    def test_serial_codec_configuration_applies_to_encoder_and_decoder(self):
        with patch.dict(os.environ, {'ROBOT_DATA_HUB_SERIAL_CODECS': '1'}):
            decoder = export_common.create_video_decoder('h264')
            self.assertEqual(decoder.thread_count, 1)
            encoder = av.CodecContext.create('libx264', 'w')
            export_common.configure_codec(encoder)
            self.assertEqual(encoder.thread_count, 1)
            self.assertIn('sync-lookahead=0', encoder.options['x264-params'])


def agibot_source(root):
    sample = json.loads((Path(__file__).parent / 'fixtures/agibotworld2026_pose_samples.json').read_text())['samples'][0]
    state, fields = [], {}
    for key in agibotworld2026.STATE_FIELDS:
        fields[key] = {'indices': list(range(len(state), len(state) + len(sample[key])))}
        state.extend(sample[key])
    info = {'robot_type': 'g2a', 'total_episodes': 2, 'features': {
        'observation.state': {'field_descriptions': fields},
        'observation.images.hand_left': {'video_info': {'video.is_depth_map': False}},
        'action': {'field_descriptions': {'action/robot/velocity': {'indices': [0, 1, 2]}}},
    }}
    video = root / 'source.mp4'
    with av.open(str(video), 'w') as writer:
        stream = writer.add_stream('libx264', rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        stream.thread_count = 1
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 70, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    archive_path = root / 'raw/task_1/0_1.tar.gz'
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, 'w:gz') as archive:
        def add(name, data):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        add('data/meta/info.json', json.dumps(info).encode())
        add('data/meta/episodes.jsonl', ''.join(json.dumps({'episode_index': i, 'tasks': ['Pick']}) + '\n' for i in range(2)).encode())
        for index in range(2):
            table = pa.table({'observation.state': [state] * 3, 'timestamp': [0., .1, .2],
                              'frame_index': [0, 1, 2], 'episode_index': [index] * 3, 'action': [[0.] * 3] * 3})
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            add(f'data/data/chunk-000/episode_{index:06d}.parquet', buffer.getvalue())
        for index in range(2):
            add(f'data/videos/chunk-000/observation.images.hand_left/episode_{index:06d}.mp4', video.read_bytes())
    return archive_path


class ArchiveResumeTests(unittest.TestCase):
    def test_archive_commits_each_episode_and_resumes_without_reconverting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = agibot_source(root)
            output = root / 'out'
            config = make_config('agibotworld2026', source)
            store = parallel.prepare(config, output)
            self.assertEqual(len(store.plan['tasks']), 1)
            self.assertEqual(len(store.ids), 2)
            original = agibotworld2026.transcode_video
            calls = []

            def interrupted(source, destination, count):
                calls.append(destination)
                if len(calls) == 2:
                    raise KeyboardInterrupt('second episode interrupted')
                return original(source, destination, count)

            with patch.object(agibotworld2026, 'transcode_video', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    parallel.worker(output, 0, 1)
            self.assertIsNotNone(store.state(store.ids[0]))
            self.assertIsNone(store.state(store.ids[1]))
            parallel.prepare(config, output)
            with patch.object(agibotworld2026, 'read_episode', wraps=agibotworld2026.read_episode) as read, patch.object(agibotworld2026, 'transcode_video', wraps=original) as transcode:
                for rank in range(3):
                    self.assertEqual(parallel.worker(output, rank, 3), 0)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(transcode.call_count, 1)
            parallel.finalize(output)
            reference = root / 'reference'
            agibotworld2026.convert(source, reference, Path(config['model_dir']))
            self.assertEqual(manifest(output), manifest(reference))

    def test_archive_failure_continues_and_global_limit_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = agibot_source(root)
            output = root / 'out'
            config = make_config('agibotworld2026', source)
            store = parallel.prepare(config, output)
            original = agibotworld2026.transcode_video

            def fail_first(source, destination, count):
                if destination.parent.parent.name == store.ids[0]:
                    raise ValueError('injected broken video')
                return original(source, destination, count)

            with patch.object(agibotworld2026, 'transcode_video', side_effect=fail_first):
                self.assertEqual(parallel.worker(output, 0, 1), 1)
            self.assertIsNone(store.state(store.ids[0]))
            self.assertIsNotNone(store.state(store.ids[1]))
            parallel.prepare(config, output)
            self.assertEqual(parallel.worker(output, 0, 2), 0)
            parallel.finalize(output)
            self.assertEqual(len(manifest(output)), 2)
            limited_output = root / 'limited'
            limited = make_config('agibotworld2026', source, limit_episodes=1)
            limited_store = parallel.prepare(limited, limited_output)
            self.assertEqual(len(limited_store.ids), 1)
            self.assertEqual(parallel.worker(limited_output, 0, 2), 0)
            parallel.finalize(limited_output)
            self.assertEqual(len(manifest(limited_output)), 1)
            # The global episode limit must not inspect later unselected archives.
            later = source.with_name('2_3.tar.gz')
            later.write_bytes(b'not a tar archive')
            limited_tree = make_config('agibotworld2026', source.parent, limit_episodes=1)
            tree_store = parallel.prepare(limited_tree, root / 'limited_tree')
            self.assertEqual(len(tree_store.plan['tasks']), 1)
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                parallel.prepare(config, limited_output)


if __name__ == '__main__':
    unittest.main()
