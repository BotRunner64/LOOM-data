"""Planning-time estimates, reproducible subsets, and unchanged worker execution."""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.convert import parallel
from scripts.convert.parallel_adapters import make_config
from scripts.convert.storage_budget import sample_tasks, storage_bytes
from test_convert_hifi_umi import make_source, make_shard
from test_parallel_conversion import add_hifi_episode, manifest


def task(ids, path, size):
    return {'episode_ids': ids, 'source': path,
            'inputs': [{'path': path, 'size': size, 'mtime_ns': 1}]}


def sampling_config(budget, seed=0, ratio=1.):
    return {'storage_budget': budget, 'sample_seed': seed, 'storage_estimate_ratio': ratio}


class StorageValueTests(unittest.TestCase):
    def test_sizes_and_invalid_values(self):
        for value, expected in [('1GB', 10**9), ('1GiB', 2**30), (' 1.5 mib ', 1572864),
                                ('42', 42), (42, 42), ('1B', 1), ('0.001KB', 1)]:
            self.assertEqual(storage_bytes(value), expected)
        for value in [0, -1, True, 1.5, '0GB', '-1GB', 'inf', '1e3', '1XB', '0.1B', '1G']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage_bytes(value)


class SamplingTests(unittest.TestCase):
    def test_shared_files_and_archive_episode_costs_are_apportioned(self):
        tasks = [task(['a'], '/shared.mp4', 1000), task(['b'], '/shared.mp4', 1000),
                 task(['c'], '/c.mp4', 600), task(['d', 'e'], '/archive.tar.gz', 800)]
        original = copy.deepcopy(tasks)
        selected, summary = sample_tasks(sampling_config(2400), tasks)
        self.assertEqual(selected, original)
        self.assertEqual(summary['estimated_output_bytes'], 2400)
        self.assertEqual(summary['selected_episodes'], 5)
        self.assertEqual(summary['candidate_episodes'], 5)
        selected, summary = sample_tasks(sampling_config(900), tasks)
        self.assertGreater(summary['selected_episodes'], 0)
        self.assertLess(summary['selected_episodes'], 5)
        self.assertLessEqual(summary['estimated_output_bytes'], 900)
        self.assertEqual(tasks, original)
        costs = {'a': 500, 'b': 500, 'c': 600, 'd': 400, 'e': 400}
        self.assertEqual(sum(costs[i] for t in selected for i in t['episode_ids']),
                         summary['estimated_output_bytes'])
        # The source groups and their original ordering remain usable by workers.
        selected_ids = {i for t in selected for i in t['episode_ids']}
        self.assertEqual([t['source'] for t in selected],
                         [t['source'] for t in tasks if selected_ids.intersection(t['episode_ids'])])

    def test_seed_is_reproducible_and_changes_the_subset(self):
        tasks = [task([f'ep_{i}'], f'/{i}.mp4', 100) for i in range(20)]
        first = sample_tasks(sampling_config(500, seed=42), tasks)
        self.assertEqual(first, sample_tasks(sampling_config(500, seed=42), tasks))
        self.assertNotEqual(first[0], sample_tasks(sampling_config(500, seed=43), tasks)[0])
        self.assertEqual(first[1]['selected_episodes'], 5)

    def test_partial_archive_selection_and_estimate_ratio(self):
        tasks = [task([f'ep_{i}' for i in range(5)], '/archive.tar.gz', 1000)]
        selected, summary = sample_tasks(sampling_config(400), tasks)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]['episode_ids']), 2)
        self.assertEqual(summary['estimated_output_bytes'], 400)
        selected, summary = sample_tasks(sampling_config(400, ratio=2), tasks)
        self.assertEqual(len(selected[0]['episode_ids']), 1)
        self.assertEqual(summary['estimated_output_bytes'], 400)

    def test_skip_oversized_candidates_and_report_too_small_budget(self):
        tasks = [task(['large'], '/large', 10000), task(['small'], '/small', 100)]
        selected, _ = sample_tasks(sampling_config(100), tasks)
        self.assertEqual(selected[0]['episode_ids'], ['small'])
        with self.assertRaisesRegex(ValueError, 'too small for one estimated episode'):
            sample_tasks(sampling_config(99), tasks)
        self.assertEqual(sample_tasks({}, tasks), (tasks, None))


class StorageBudgetPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw'
        make_source(self.source)

    def config(self, **kwargs):
        return make_config('hifi_umi', self.source, **kwargs)

    def test_workers_export_the_fixed_subset_even_if_actual_output_exceeds_estimate(self):
        for index in range(8, 14):
            add_hifi_episode(self.source, index)
        # Deliberately underestimate: workers must still process the entire plan.
        config = self.config(storage_budget=60, storage_estimate_ratio=.001, sample_seed=42)
        serial_output = self.root / 'serial'
        serial = parallel.prepare(config, serial_output)
        self.assertGreater(len(serial.ids), 0)
        self.assertLess(len(serial.ids), 7)
        self.assertEqual(serial.plan['storage_sampling']['selected_episodes'], len(serial.ids))
        self.assertEqual(parallel.worker(serial_output, 0, 1), 0)
        parallel.finalize(serial_output)
        parallel_output = self.root / 'parallel'
        parallel.run(config, parallel_output, workers=3)
        multi = parallel.Store.load(parallel_output)
        self.assertEqual(serial.ids, multi.ids)
        self.assertEqual(manifest(serial_output), manifest(parallel_output))
        self.assertEqual([r['episode_id'] for r in manifest(serial_output)], serial.ids)
        actual = sum(p.stat().st_size for p in serial_output.rglob('*') if p.is_file())
        self.assertGreater(actual, config['storage_budget'])
        self.assertFalse((serial.work / 'storage_budget.json').exists())
        self.assertFalse((serial.work / 'storage_budget.lock').exists())
        parallel.prepare(config, serial_output)
        with patch('scripts.convert.parallel_adapters.discover', side_effect=AssertionError('must reuse plan')):
            parallel.prepare(config, serial_output)
        before = {p: p.stat().st_mtime_ns for p in serial_output.rglob('*') if p.is_file()}
        parallel.run(config, serial_output, workers=2)
        self.assertTrue(all(p.stat().st_mtime_ns == stamp for p, stamp in before.items()))

    def test_resume_keeps_sample_when_worker_count_changes(self):
        for index in range(8, 14):
            add_hifi_episode(self.source, index)
        config = self.config(storage_budget=60, storage_estimate_ratio=.001)
        output = self.root / 'resume'
        store = parallel.prepare(config, output)
        self.assertEqual(parallel.worker(output, 0, 2), 0)
        before = {p: p.stat().st_mtime_ns for p in store.dataset.rglob('*') if p.is_file()}
        resumed = parallel.prepare(config, output)
        self.assertEqual(store.ids, resumed.ids)
        self.assertEqual(parallel.worker(output, 0, 1), 0)
        self.assertTrue(all(p.stat().st_mtime_ns == stamp for p, stamp in before.items()))
        parallel.finalize(output)
        self.assertEqual([r['episode_id'] for r in manifest(output)], store.ids)
        for key, value in [('storage_budget', 200), ('sample_seed', 43), ('storage_estimate_ratio', 2.)]:
            with self.assertRaisesRegex(ValueError, 'Incompatible conversion config'):
                parallel.prepare(dict(config, **{key: value}), output)

    def test_full_shard_plan_contains_only_sampled_episode_ids(self):
        source = self.root / 'shards'
        make_shard(source)
        config = make_config('hifi_umi', source, storage_budget=1, storage_estimate_ratio=.000001)
        store = parallel.prepare(config, self.root / 'shard-out')
        self.assertEqual(len(store.ids), 1)
        self.assertEqual(store.plan['storage_sampling']['candidate_episodes'], 2)
        self.assertEqual(parallel.worker(store.output, 0, 1), 0)
        parallel.finalize(store.output)
        self.assertEqual([r['episode_id'] for r in manifest(store.output)], store.ids)

    def test_all_source_clis_forward_budget_options(self):
        for dataset in parallel.DATASETS:
            with self.subTest(dataset=dataset), patch.object(parallel, 'run_native') as run:
                module = importlib.import_module(f'scripts.convert.{dataset}')
                argv = [dataset, '--input-dir', str(self.source), '--output-dir', str(self.root / dataset),
                        '--storage-budget', '1.5GiB', '--sample-seed', '42', '--storage-estimate-ratio', '2']
                with patch.object(sys, 'argv', argv):
                    module.main()
                config, output = run.call_args.args
                self.assertEqual(config['dataset'], dataset)
                self.assertEqual(config['storage_budget'], 1610612736)
                self.assertEqual(config['sample_seed'], 42)
                self.assertEqual(config['storage_estimate_ratio'], 2.)
                self.assertEqual(output, self.root / dataset)

    def test_invalid_estimate_ratio_and_budget_only_flags(self):
        for ratio in (0, -1, True, float('inf'), float('nan')):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'positive and finite'):
                self.config(storage_budget=100, storage_estimate_ratio=ratio)
        for kwargs in ({'sample_seed': 1}, {'storage_estimate_ratio': 2}):
            with self.assertRaisesRegex(ValueError, 'require --storage-budget'):
                self.config(**kwargs)


if __name__ == '__main__':
    unittest.main()
