"""Mount-origin migration: rigid transforms, every UMI entry point and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from scripts.convert import hifi_umi, parallel, realomin
from scripts.convert.export_validation import validate_episode
from scripts.convert.parallel_adapters import make_config
from scripts.robot.kinematics import rpy_matrix
from test_convert_hifi_umi import make_source, make_shard


class MountOriginTests(unittest.TestCase):
    def test_tip_to_mount_translation_rotates_with_source_tool_and_preserves_axes(self):
        rotations = [rpy_matrix([.2, -.4, .7]), rpy_matrix([-.6, .3, -1.1])]
        position, offset = np.array([1., 2., 3.]), np.array([-.14, .02, .03])
        source = [np.r_[position, rotation[:2].ravel()] for rotation in rotations]
        poses = hifi_umi.poses_from_rows(source, offset)
        for pose, rotation in zip(poses, rotations):
            world_tip, tip_mount = np.eye(4), np.eye(4)
            world_tip[:3, :3], world_tip[:3, 3] = rotation, position
            tip_mount[:3, 3] = offset
            expected = world_tip @ tip_mount
            np.testing.assert_allclose(pose[:3], expected[:3, 3], atol=1e-12)
            np.testing.assert_allclose(pose[3:].reshape(3, 3).T, rotation, atol=1e-12)
        self.assertGreater(np.linalg.norm(poses[1, :3] - poses[0, :3]), .01)

    def test_invalid_mount_offsets_fail_before_reading_or_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            source.mkdir()
            for dataset, kwargs in (
                    ('hifi_umi', {'left_eef_offset': [0, 0]}),
                    ('hifi_umi', {'left_eef_offset': [0, 0, 0], 'right_eef_offset': [0, 0, float('nan')]}),
                    ('realomin', {'eef_offset': [0, float('inf'), 0]})):
                with self.subTest(dataset=dataset, kwargs=kwargs), self.assertRaises(ValueError):
                    make_config(dataset, source, **kwargs)
            # Both UMI sources now have explicit, documented nominal defaults.
            config = make_config('realomin', source)
            np.testing.assert_allclose(config['params']['eef_offset'], realomin.DEFAULT_EEF_OFFSET)
            hifi_config = make_config('hifi_umi', source, left_eef_offset=[-.15, 0, 0])
            self.assertEqual(hifi_config['params']['eef_offsets'],
                             {'left': [-.15, 0, 0], 'right': [-.12, 0, 0]})
            with self.assertRaisesRegex(ValueError, 'three finite metres'):
                hifi_umi.convert(source, root / 'output', left_eef_offset=[0, 0, float('inf')])
            self.assertFalse((root / 'output').exists())
            self.assertFalse((root / 'episode').exists())

    def test_real_opening_sample_uses_fixed_offset_without_angle_dependent_correction(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/hifi_umi_rigid_frame_samples.json').read_text())
        state = np.asarray([row['observation.state'] for row in fixture['rows']])
        pose = hifi_umi.poses_from_rows(state[:, :9])
        self.assertGreater(abs(state[-1, 9] - state[0, 9]), .37)
        self.assertLess(np.linalg.norm(state[-1, :3] - state[0, :3]), .0006)
        # Reframing this near-stationary body must not introduce the centimetre
        # shift that aperture-dependent fingertip kinematics would manufacture.
        self.assertLess(np.linalg.norm(pose[-1, :3] - pose[0, :3]), .002)
        raw_first_column = np.column_stack((state[:, 3], state[:, 6],
                                           np.cross(state[:, 3:6], state[:, 6:9])[:, 0]))
        np.testing.assert_allclose(pose[:, :3], state[:, :3] - .12 * raw_first_column, atol=1e-7)

    def test_nominal_hifi_defaults_match_subset_shard_parallel_and_cli(self):
        import subprocess
        import sys
        expected = {'left': [3.88, 5., 6.], 'right': [1., 1.88, 3.]}
        for layout in ('subset', 'shard'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (make_source if layout == 'subset' else make_shard)(root / 'raw')
                hifi_umi.convert(root / 'raw', root / 'serial')
                config = make_config('hifi_umi', root / 'raw')
                self.assertEqual(config['params']['eef_offsets'],
                                 {'left': [-.12, 0., 0.], 'right': [-.12, 0., 0.]})
                parallel.prepare(config, root / 'parallel')
                self.assertEqual(parallel.worker(root / 'parallel', 0, 1), 0)
                parallel.finalize(root / 'parallel')
                records = [json.loads(line) for line in (root / 'serial/episodes.jsonl').read_text().splitlines()]
                for record in records:
                    relative = Path('episodes') / record['episode_id']
                    for output in ('serial', 'parallel'):
                        validate_episode(root / output / relative, record)
                        for side in ('left', 'right'):
                            table = pq.read_table(root / output / relative / 'state' / f'{side}_eef.parquet')
                            # Fixture aperture varies widely while its raw pose
                            # stays fixed; only the gripper output should vary.
                            np.testing.assert_allclose(np.array(table['pose'].to_pylist())[:, :3],
                                                       np.tile(expected[side], (table.num_rows, 1)), atol=1e-7)
                if layout == 'subset':
                    command = [sys.executable, '-m', 'scripts.convert.hifi_umi',
                               '--input-dir', str(root / 'raw'), '--output-dir', str(root / 'cli')]
                    result = subprocess.run(command, cwd=hifi_umi.ROOT, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    for side in ('left', 'right'):
                        path = relative / 'state' / f'{side}_eef.parquet'
                        self.assertEqual(pq.read_table(root / 'cli' / path).to_pydict(),
                                         pq.read_table(root / 'serial' / path).to_pydict())

    def test_nonzero_left_right_mounts_match_in_subset_shard_and_parallel_exports(self):
        offsets = {'left_eef_offset': [-.11, .02, .03], 'right_eef_offset': [-.15, -.01, .04]}
        # Fixture rotations: left +90 about X, right +90 about Z.
        expected = {'left': [3.89, 4.97, 6.02], 'right': [1.01, 1.85, 3.04]}
        for layout in ('subset', 'shard'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'raw'
                (make_source if layout == 'subset' else make_shard)(source)
                originals = {p: p.read_bytes() for p in source.rglob('*.parquet')}
                serial, concurrent = root / 'serial', root / 'parallel'
                hifi_umi.convert(source, serial, **offsets)
                config = make_config('hifi_umi', source, **offsets)
                parallel.prepare(config, concurrent)
                self.assertEqual(parallel.worker(concurrent, 0, 1), 0)
                parallel.finalize(concurrent)
                serial_records = (serial / 'episodes.jsonl').read_text()
                self.assertEqual([json.loads(line) for line in serial_records.splitlines()],
                                 [json.loads(line) for line in (concurrent / 'episodes.jsonl').read_text().splitlines()])
                for line in serial_records.splitlines():
                    record = json.loads(line)
                    episode = Path('episodes') / record['episode_id']
                    validate_episode(serial / episode, record)
                    validate_episode(concurrent / episode, record)
                    for side in ('left', 'right'):
                        relative = episode / 'state' / f'{side}_eef.parquet'
                        table = pq.read_table(serial / relative)
                        self.assertEqual(table.to_pydict(), pq.read_table(concurrent / relative).to_pydict())
                        np.testing.assert_allclose(np.array(table['pose'].to_pylist())[:, :3],
                                                   np.tile(expected[side], (table.num_rows, 1)), atol=1e-7)
                self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))

    def test_old_tcp_plans_are_rejected_by_prepare_worker_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            config = make_config('hifi_umi', root / 'raw',
                                 left_eef_offset=[-.1, 0, 0], right_eef_offset=[-.1, 0, 0])
            store = parallel.prepare(config, root / 'out')
            old = copy.deepcopy(store.plan)
            old['config']['params'].pop('export_contract_version')
            old['id'] = parallel.digest({key: value for key, value in old.items() if key != 'id'})
            (store.work / 'plan.json').write_text(json.dumps(old))
            for action in (lambda: parallel.prepare(config, root / 'out'),
                           lambda: parallel.worker(root / 'out', 0, 1),
                           lambda: parallel.finalize(root / 'out')):
                with self.assertRaisesRegex(parallel.StateError, 'Incompatible export contract'):
                    action()
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
