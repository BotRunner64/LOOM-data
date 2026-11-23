from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from support import IDENTITY_POSE, check, expected_report, make_dataset
from loom_data import contract
from loom_data.check import anomalies as script


def poses(count, positions=None, angles=None):
    """An EEF stream of `count` frames: x positions in metres and rotations about z in degrees."""
    angles = np.deg2rad(np.zeros(count) if angles is None else angles)
    values = np.zeros((count, 12))
    if positions is not None:
        values[:, 0] = positions
    values[:, 3] = np.cos(angles)
    values[:, 4] = np.sin(angles)
    values[:, 6] = -np.sin(angles)
    values[:, 7] = np.cos(angles)
    values[:, 11] = 1
    return script.StateStream(values, script.rotation_quaternions(values))


def gripper(openness):
    return script.StateStream(np.asarray(openness, dtype=float))


def still(count=600):
    return {name: gripper(np.zeros(count)) if 'gripper' in name else poses(count)
            for name in script.STATE_NAMES}


def detect(streams, reason, config=script.Config()):
    return [row for row in script.detect_episode(3, streams, config) if row['reason'] == reason]


def record(reason, start, end, stream=None):
    return dict(episode_index=3, reason=reason, stream=stream, start_frame=start, end_frame=end)


def pulse(count, start, length=1, height=1.):
    frames = np.arange(count)
    return height * ((frames >= start) & (frames < start + length))


class DetectionTests(unittest.TestCase):
    def test_boundary_idle_ends_when_any_stream_moves(self):
        frames = np.arange(600)
        streams = still()
        streams['left_eef'] = poses(600, positions=.1*(frames >= 150) + .1*(frames >= 420))
        streams['right_gripper'] = gripper(pulse(600, 210, 270))
        rows = detect(streams, 'boundary_idle')
        # Head: nothing moved before frame 150. Tail: the gripper last differs from the final frame at 479.
        self.assertEqual(rows, [record('boundary_idle', 0, 150), record('boundary_idle', 480, 600)])
        self.assertEqual([list(row) for row in rows], [list(script.FIELDS)] * 2)

    def test_boundary_idle_is_longer_than_the_limit_and_static_episodes_get_one_record(self):
        self.assertEqual(detect(still(600), 'boundary_idle'), [record('boundary_idle', 0, 600)])
        self.assertEqual(detect(still(600), 'internal_idle'), [])
        # 90 frames last exactly 3 s, which does not exceed the limit.
        self.assertEqual(detect(still(90), 'boundary_idle'), [])
        self.assertEqual(detect(still(91), 'boundary_idle'), [record('boundary_idle', 0, 91)])
        streams = still(200)
        streams['left_gripper'] = gripper(np.arange(200) >= 90)
        self.assertEqual(detect(streams, 'boundary_idle'), [record('boundary_idle', 90, 200)])
        streams['left_gripper'] = gripper(np.arange(200) >= 91)
        self.assertEqual(detect(streams, 'boundary_idle'),
                         [record('boundary_idle', 0, 91), record('boundary_idle', 91, 200)])

    def test_boundary_orientation_openness_and_overrides(self):
        frames = np.arange(600)
        steps = (frames >= 30).astype(float) + (frames >= 570)
        streams = still()
        streams['right_eef'] = poses(600, angles=6*steps)
        self.assertEqual(detect(streams, 'boundary_idle'), [])
        config = replace(script.Config(), orientation_tolerance_deg=20)
        self.assertEqual(detect(streams, 'boundary_idle', config), [record('boundary_idle', 0, 600)])
        streams['left_gripper'] = gripper(.1*steps)
        self.assertEqual(detect(streams, 'boundary_idle', config), [])

    def test_boundary_tail_tolerates_drift_within_tolerance_of_the_last_frame(self):
        frames = np.arange(600)
        # A reach, then 1.5 cm of drift over the last 10 s: the last 200 frames stay within 1 cm of the end.
        positions = np.where(frames < 300, frames*.001, .3 + .015*(frames-300)/299)
        streams = still()
        streams['left_eef'] = poses(600, positions=positions)
        self.assertEqual(detect(streams, 'boundary_idle'), [record('boundary_idle', 400, 600)])

    def test_eef_spikes_of_one_to_four_frames_are_reported_at_their_frames(self):
        for length in range(1, 6):
            streams = still(120)
            streams['left_eef'] = poses(120, positions=pulse(120, 40, length, .1))
            streams['right_eef'] = poses(120, angles=pulse(120, 70, length, 40))
            with self.subTest(length=length):
                expected = [record('eef_spike', 40, 40+length, 'left_eef'),
                            record('eef_spike', 70, 70+length, 'right_eef')]
                self.assertEqual(detect(streams, 'eef_spike'), expected if length <= 4 else [])

    def test_gripper_excursions_are_not_spikes(self):
        for length in range(1, 5):
            streams = still(120)
            for name in ('left_gripper', 'right_gripper'):
                streams[name] = gripper(pulse(120, 40, length))
            with self.subTest(length=length):
                self.assertEqual(script.detect_episode(3, streams), [])

    def test_resampled_spike_ramps_belong_to_the_excursion(self):
        # A native 15 fps one-sample spike is a triangle after interpolation to 30 fps.
        positions = np.zeros(60)
        positions[20:23] = [.025, .05, .025]
        streams = still(60)
        streams['left_eef'] = poses(60, positions=positions)
        self.assertEqual(detect(streams, 'eef_spike'), [record('eef_spike', 20, 23, 'left_eef')])
        # Ramp frames within tolerance of the stable pose are not part of the excursion.
        positions[20:23] = [.008, .05, .008]
        streams['left_eef'] = poses(60, positions=positions)
        self.assertEqual(detect(streams, 'eef_spike'), [record('eef_spike', 21, 22, 'left_eef')])

    def test_spike_amplitude_is_strict_and_configurable(self):
        streams = still(60)
        for height, config, count in ((.015, script.Config(), 0), (.02, script.Config(), 0),
                                      (.021, script.Config(), 1),
                                      (.02, replace(script.Config(), position_spike_min_m=.019), 1)):
            streams['left_eef'] = poses(60, positions=pulse(60, 30, 2, height))
            with self.subTest(height=height):
                self.assertEqual(len(detect(streams, 'eef_spike', config)), count)
        streams['left_eef'] = poses(60, angles=pulse(60, 30, 2, 9))
        self.assertEqual(detect(streams, 'eef_spike'), [])
        config = replace(script.Config(), orientation_spike_min_deg=8)
        self.assertEqual(len(detect(streams, 'eef_spike', config)), 1)

    def test_spike_needs_three_stable_frames_on_both_sides(self):
        for start, length, reported in ((3, 1, False), (4, 1, True), (56, 1, True), (57, 1, False),
                                        (53, 4, True), (54, 4, False)):
            streams = still(60)
            streams['left_eef'] = poses(60, positions=pulse(60, start, length, .1))
            with self.subTest(start=start, length=length):
                self.assertEqual(len(detect(streams, 'eef_spike')), int(reported))
        frames = np.arange(60)
        for name, positions in (('moving before', .02*np.minimum(frames, 29) + pulse(60, 30, 1, .1)),
                                ('moving after', .02*np.maximum(frames-31, 0) + pulse(60, 30, 1, .1)),
                                ('step', .1*(frames >= 30)),
                                ('no return', pulse(60, 30, 2, .1) + .015*(frames >= 32)),
                                ('second excursion in context', pulse(60, 30, 1, .1) + pulse(60, 33, 1, .1))):
            streams = still(60)
            streams['left_eef'] = poses(60, positions=positions)
            with self.subTest(name=name):
                self.assertEqual(detect(streams, 'eef_spike'), [])

    def test_position_and_orientation_spikes_share_one_record_per_arm(self):
        streams = still(120)
        streams['left_eef'] = poses(120, positions=pulse(120, 40, 2, .1), angles=pulse(120, 40, 2, 40))
        streams['right_eef'] = poses(120, positions=pulse(120, 40, 1, .1) + pulse(120, 80, 3, .1))
        self.assertEqual(detect(streams, 'eef_spike'), [
            record('eef_spike', 40, 42, 'left_eef'), record('eef_spike', 40, 41, 'right_eef'),
            record('eef_spike', 80, 83, 'right_eef')])
        # Orientation moving through the excursion is motion, not a spike.
        streams = still(120)
        streams['left_eef'] = poses(120, positions=pulse(120, 40, 2, .1), angles=3.*np.arange(120))
        self.assertEqual(detect(streams, 'eef_spike'), [])

    def test_reviewed_speed_peaks_are_ignored(self):
        for increments in ([.03035, .00574, .02090, .03106, .02144, .00350, .02381],
                           [.01683, .01678, .04332, .05160, .03968, .01350, .01701],
                           [.01253, .01082, .00537, .03150, .01937, .00422, .01426]):
            delta = np.r_[np.repeat(increments[0], 12), increments, np.repeat(increments[-1], 12)]
            positions = np.r_[0, np.cumsum(delta)]
            streams = still(len(positions))
            streams['left_eef'] = poses(len(positions), positions=positions)
            self.assertEqual(detect(streams, 'eef_spike'), [])

    def test_rotations_are_invariant_to_fixed_frames_and_quaternion_sign(self):
        data = poses(120, angles=170+pulse(120, 60, 1, 40))
        fixed = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        for columns in (slice(3, 6), slice(6, 9), slice(9, 12)):
            data.values[:, columns] = data.values[:, columns] @ fixed.T
        data.rotations = script.rotation_quaternions(data.values)
        streams = still(120)
        streams['left_eef'] = data
        self.assertEqual(detect(streams, 'eef_spike'), [record('eef_spike', 60, 61, 'left_eef')])
        data.rotations[::2] *= -1
        self.assertEqual(detect(streams, 'eef_spike'), [record('eef_spike', 60, 61, 'left_eef')])
        streams['left_eef'] = poses(120, angles=170+15.*np.arange(120))
        self.assertEqual(detect(streams, 'eef_spike'), [])

    def test_rotations_near_half_turn_on_each_axis(self):
        matrices = []
        for rotation in (np.eye(3), np.diag([1, -1, -1]),
                         np.diag([-1, 1, -1]), np.diag([-1, -1, 1])):
            matrices.append(np.r_[np.zeros(3), rotation.T.reshape(-1)])
        quaternions = script.rotation_quaternions(np.array(matrices))
        np.testing.assert_allclose(quaternions, np.eye(4))
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions[0], quaternions[1:]), 180)
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions, -quaternions), 0)

    def test_internal_idle_reports_one_record_per_pause(self):
        frames = np.arange(900)
        streams = still(900)
        streams['left_eef'] = poses(900, positions=.1*(frames >= 30) + .1*(frames >= 450))
        streams['right_gripper'] = gripper(.2*(frames >= 60) + .2*(frames >= 450) + .2*(frames >= 700))
        # Nothing moves between the gripper's change at frame 60 and both changes at frame 450.
        self.assertEqual(detect(streams, 'internal_idle'), [record('internal_idle', 60, 450)])
        self.assertEqual(detect(streams, 'internal_idle', replace(script.Config(), max_internal_idle_s=13)), [])
        streams['left_gripper'] = gripper(frames % 2)
        self.assertEqual(detect(streams, 'internal_idle'), [])

    def test_internal_idle_is_longer_than_the_limit_and_excludes_the_first_and_last_segment(self):
        frames = np.arange(700)
        for end, reported in ((330, False), (331, True)):
            streams = still(700)
            streams['left_gripper'] = gripper(.2*(frames >= 30) + .2*(frames >= end) + .2*(frames >= end + 200))
            with self.subTest(end=end):
                self.assertEqual(detect(streams, 'internal_idle'),
                                 [record('internal_idle', 30, end)] if reported else [])
        streams = still(700)
        streams['left_gripper'] = gripper(frames >= 350)
        self.assertEqual(detect(streams, 'internal_idle'), [])
        self.assertEqual(detect(streams, 'boundary_idle'),
                         [record('boundary_idle', 0, 350), record('boundary_idle', 350, 700)])

    def test_internal_idle_does_not_mistake_smooth_motion_for_a_pause(self):
        frames = np.arange(901)
        middle = np.clip(frames/30-1, 0, 28)
        for name, kind, speed, step in (('left_eef', 'positions', .02, .1), ('right_eef', 'angles', 10., 20.),
                                        ('left_gripper', None, .01, .1), ('right_gripper', None, .01, .1)):
            values = middle*speed + step*(frames >= 30) + step*(frames >= 870)
            streams = still(901)
            streams[name] = gripper(values) if kind is None else poses(901, **{kind: values})
            with self.subTest(name=name):
                self.assertEqual(detect(streams, 'internal_idle'), [])

    def test_internal_idle_retains_real_pause_with_subthreshold_noise(self):
        frames = np.arange(901)
        signal = .1*(frames >= 30) + .1*(frames >= 600)
        streams = still(901)
        streams['left_eef'] = poses(901, positions=signal + .001*np.sin(frames/30*4))
        self.assertEqual(detect(streams, 'internal_idle'), [record('internal_idle', 30, 600)])

    def test_records_are_ordered_by_start_frame(self):
        frames = np.arange(900)
        streams = still(900)
        streams['left_gripper'] = gripper(.2*(frames >= 100) + .2*(frames >= 500) + .2*(frames >= 700))
        streams['right_eef'] = poses(900, positions=pulse(900, 300, 2, .1))
        rows = script.detect_episode(3, streams)
        self.assertEqual(rows, [record('boundary_idle', 0, 100), record('eef_spike', 300, 302, 'right_eef'),
                                record('boundary_idle', 700, 900)])
        streams['right_eef'] = poses(900)
        self.assertEqual(script.detect_episode(3, streams), [
            record('boundary_idle', 0, 100), record('internal_idle', 100, 500), record('boundary_idle', 700, 900)])

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in ({'max_boundary_idle_s': -1}, {'position_spike_min_m': float('nan')},
                       {'openness_tolerance': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                script.Config(**kwargs)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'dataset'
        # Episode 0 is too short for any rule; episode 1 is still for 151 frames, just over 5 s.
        self.dataset = make_dataset(self.root, {'short': {'left_gripper': np.zeros(3)},
                                                'sample': {'left_gripper': np.zeros(151)}})
        self.data = self.dataset.data_path(self.dataset.episodes[0])

    def rewrite_state(self, change):
        """Rewrite the frame table after `change(state N x 26, columns)` edited it in place."""
        table = pq.read_table(self.data)
        state = np.array(table[contract.STATE_KEY].to_pylist(), dtype=np.float32)
        columns = {name: np.array(table[name].to_pylist()) for name in table.schema.names[1:]}
        keep = change(state, columns)
        if keep is not None:
            state, columns = state[keep], {name: values[keep] for name, values in columns.items()}
        pq.write_table(pa.Table.from_arrays(
            [pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), contract.STATE_DIM),
             *(pa.array(columns[f.name], type=f.type) for f in list(contract.DATA_SCHEMA)[1:])],
            schema=contract.DATA_SCHEMA), self.data)

    def test_read_episode_has_one_row_per_frame_in_every_stream(self):
        poses = np.tile(IDENTITY_POSE, (4, 1))
        poses[:, 0] = [0, .1, .2, .3]
        root = self.root.with_name('streams')
        dataset = make_dataset(root, {'still': {'left_gripper': np.zeros(2)},
                                      'moving': {'left_eef': poses, 'right_gripper': [0, .25, .5, 1]}})
        streams = script.read_episode(dataset, dataset.episodes[1])
        self.assertEqual(list(streams), list(script.STATE_NAMES))
        for name, stream in streams.items():
            self.assertEqual(len(stream.values), 4)
            self.assertEqual(stream.rotations is None, 'gripper' in name)
        np.testing.assert_allclose(streams['left_eef'].values, poses, atol=1e-6)
        np.testing.assert_allclose(streams['right_eef'].values, np.tile(IDENTITY_POSE, (4, 1)))
        np.testing.assert_allclose(streams['left_eef'].rotations, np.tile([1., 0, 0, 0], (4, 1)))
        np.testing.assert_allclose(streams['right_gripper'].values, [0, .25, .5, 1])
        np.testing.assert_allclose(streams['left_gripper'].values, 0)
        # The other episode sharing the data file is not mixed in.
        self.assertEqual(len(script.read_episode(dataset, dataset.episodes[0])['left_eef'].values), 2)

    def test_report_uses_episode_index_and_frame_ranges_without_video_decode(self):
        with patch('av.open', side_effect=AssertionError('detection must not open videos')):
            self.assertEqual(check(self.root), (2, 1))
        report = (self.root / 'anomalies.jsonl').read_text()
        self.assertEqual(report, '{"episode_index":1,"reason":"boundary_idle","stream":null,'
                                 '"start_frame":0,"end_frame":151}\n')

    def test_exported_spike_is_reported_for_the_arm_only(self):
        openness = np.zeros(61)
        openness[30] = .6
        poses = np.tile(IDENTITY_POSE, (61, 1))
        poses[30, 0] = .1
        root = self.root.with_name('spike')
        make_dataset(root, {'spike': {'right_eef': poses, 'left_gripper': openness}})
        self.assertEqual(check(root), (1, 1))
        report = (root / 'anomalies.jsonl').read_text()
        self.assertEqual(report, expected_report(root))
        self.assertEqual([json.loads(line) for line in report.splitlines()], [
            dict(episode_index=0, reason='eef_spike', stream='right_eef', start_frame=30, end_frame=31)])

    def test_failure_preserves_previous_report_and_removes_temporary_file(self):
        output = self.root / 'report.jsonl'
        output.write_text('previous\n')
        info = self.root / 'meta/info.json'
        saved = info.read_text()
        info.write_text(saved.replace('"fps": 30', '"fps": 25'))
        with self.assertRaisesRegex(ValueError, 'not a contract dataset'):
            check(self.root, output)
        info.write_text('{bad json')
        with self.assertRaisesRegex(ValueError, 'not an exported dataset'):
            check(self.root, output)
        # An episode that cannot be read fails the run after the plan exists.
        info.write_text(saved)
        self.rewrite_state(lambda state, columns: state.__setitem__((slice(None), 12), 2))
        with self.assertRaisesRegex(RuntimeError, 'Detection failed'):
            check(self.root, output)
        self.assertEqual(output.read_text(), 'previous\n')
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_rejects_invalid_state_and_protects_inputs(self):
        saved = self.data.read_bytes()

        def nonfinite(state, columns):
            state[1, 12] = np.nan

        def openness(state, columns):
            state[-1, 25] = 2

        def rotation(state, columns):
            state[2, 3] = .5

        def missing_frame(state, columns):
            return np.arange(len(state)) != 100

        def unordered_frames(state, columns):
            columns['frame_index'][[3, 4]] = columns['frame_index'][[4, 3]]

        for change in (nonfinite, openness, rotation, missing_frame, unordered_frames):
            self.data.write_bytes(saved)
            self.rewrite_state(change)
            with self.subTest(change=change.__name__), self.assertRaises((ValueError, RuntimeError)):
                check(self.root)
            self.assertFalse((self.root / 'anomalies.jsonl').exists())
        self.data.write_bytes(saved)
        for output in (self.root / 'meta/info.json', self.root / 'meta/report.jsonl',
                       self.data.with_name('report.jsonl'), self.root / 'videos/report.jsonl',
                       self.root / 'state.parquet'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                check(self.root, output)
            self.assertFalse(output.exists() and output.suffix == '.jsonl')
        self.assertEqual(check(self.root, self.root.parent / 'elsewhere.jsonl'), (2, 1))

    def test_cli_and_threshold_override(self):
        result = subprocess.run([sys.executable, '-m', 'loom_data', 'check', 'run', str(self.root),
                                 '--max-boundary-idle-s', '6'], capture_output=True, text=True,
                                env=dict(os.environ, LOOM_DATA_NO_PROGRESS='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Checked 2 episodes; wrote 0 anomalies', result.stdout)
        self.assertEqual((self.root / 'anomalies.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
