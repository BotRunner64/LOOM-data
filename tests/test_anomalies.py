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
from loom_data.sync import frame_ns


def stream(seconds, positions=None, angles=None, openness=None):
    times = np.rint(np.asarray(seconds) * 1e9).astype(np.int64)
    if openness is not None:
        return script.StateStream(times, np.asarray(openness, dtype=float))
    n = len(times)
    angles = np.deg2rad(np.zeros(n) if angles is None else angles)
    values = np.zeros((n, 12))
    if positions is not None:
        p = np.asarray(positions)
        if p.ndim == 1:
            values[:, 0] = p
        else:
            values[:, :3] = p
    values[:, 3] = np.cos(angles)
    values[:, 4] = np.sin(angles)
    values[:, 6] = -np.sin(angles)
    values[:, 7] = np.cos(angles)
    values[:, 11] = 1
    return script.StateStream(times, values, script.rotation_quaternions(values))


def still(end=20):
    return {name: stream([0, end], **({'openness': [0, 0]} if 'gripper' in name else {}))
            for name in script.STATE_NAMES}


def detect(streams, reason, end=20, config=script.Config()):
    return [row for row in script.detect_episode(3, streams, int(end*1e9), config)
            if row['reason'] == reason]


class DetectionTests(unittest.TestCase):
    def test_boundary_uses_minimum_across_streams_and_final_reference(self):
        streams = still()
        streams['left_eef'] = stream([1, 4, 5, 14, 15, 18], positions=[0, 0, .1, .2, .3, .3])
        streams['right_gripper'] = stream([2, 7, 16, 19], openness=[0, 1, 0, 0])
        self.assertEqual(detect(streams, 'boundary_idle'), [
            dict(index=3, stream='left_eef', start_ns=0, end_ns=5_000_000_000, reason='boundary_idle'),
            dict(index=3, stream='left_eef', start_ns=14_000_000_000, end_ns=20_000_000_000,
                 reason='boundary_idle')])

    def test_boundary_all_static_two_records_and_ties_are_stable(self):
        rows = detect(still(), 'boundary_idle')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['stream'] == 'left_eef' and row['start_ns'] == 0
                            and row['end_ns'] == 20_000_000_000 for row in rows))
        self.assertEqual(detect(still(), 'internal_idle'), [])
        self.assertEqual(detect(still(3), 'boundary_idle', end=3), [])

    def test_boundary_orientation_openness_and_overrides(self):
        streams = still()
        streams['right_eef'] = stream([0, 1, 19, 20], angles=[0, 6, 12, 18])
        self.assertEqual(detect(streams, 'boundary_idle'), [])
        config = replace(script.Config(), orientation_tolerance_deg=20)
        self.assertEqual(len(detect(streams, 'boundary_idle', config=config)), 2)
        streams['left_gripper'] = stream([0, 1, 19, 20], openness=[0, .1, .2, .3])
        self.assertEqual(detect(streams, 'boundary_idle', config=config), [])

    def test_isolated_one_sample_spikes_across_sample_rates_and_channels(self):
        for hz in (25, 30, 100, 300):
            times = np.arange(hz+1)/hz
            sample = hz//2
            signal = np.arange(hz+1) == sample
            streams = still(1)
            streams['left_eef'] = stream(times, positions=.1*signal)
            streams['right_eef'] = stream(times, angles=40*signal)
            for name in ('left_gripper', 'right_gripper'):
                streams[name] = stream(times, openness=.6*signal)
            with self.subTest(hz=hz):
                rows = detect(streams, 'state_jump', end=1)
                self.assertEqual([r['stream'] for r in rows], list(script.STATE_NAMES))
                self.assertTrue(all(r['start_ns'] == round(times[sample]*1e9)
                                    and r['end_ns'] == round(times[sample+1]*1e9) for r in rows))

    def test_steps_and_multi_sample_excursions_are_ignored(self):
        for hz in (30, 100, 300):
            times = np.arange(hz+1)/hz
            for signal in ((times >= .5), ((times >= .4) & (times < .4+2/hz))):
                streams = still(1)
                streams['left_eef'] = stream(times, positions=.1*signal)
                streams['right_eef'] = stream(times, angles=40*signal)
                streams['left_gripper'] = stream(times, openness=.6*signal)
                with self.subTest(hz=hz):
                    self.assertEqual(detect(streams, 'state_jump'), [])

    def test_resolved_fast_ramps_and_excursions_are_ignored(self):
        for hz in (30, 60, 100, 250, 1000):
            times = np.arange(hz+1)/hz
            for duration in (max(.05, 2/hz), .1, .2):
                phase = np.clip((times-.4)/duration, 0, 1)
                excursion = np.clip((times-.4)/max(duration, 4/hz), 0, 1)
                for shape in (phase, (1-np.cos(np.pi*phase))/2, np.sin(np.pi*excursion)**2):
                    for sign in (-1, 1):
                        streams = still(1)
                        streams['left_eef'] = stream(times, positions=sign*.04*shape)
                        streams['right_eef'] = stream(times, angles=sign*40*shape)
                        with self.subTest(hz=hz, duration=duration, sign=sign):
                            self.assertEqual(detect(streams, 'state_jump'), [])

    def test_reviewed_speed_peaks_are_ignored(self):
        for hz, increments in (
                (25, [.03035, .00574, .02090, .03106, .02144, .00350, .02381]),
                (25, [.01683, .01678, .04332, .05160, .03968, .01350, .01701]),
                (30, [.01253, .01082, .00537, .03150, .01937, .00422, .01426])):
            delta = np.r_[np.repeat(increments[0], 12), increments, np.repeat(increments[-1], 12)]
            positions = np.r_[0, np.cumsum(delta)]
            streams = still(2)
            streams['left_eef'] = stream(np.arange(len(positions))/hz, positions=positions)
            with self.subTest(hz=hz):
                self.assertEqual(detect(streams, 'state_jump'), [])

    def test_context_motion_prevents_reporting_a_turnaround(self):
        times = np.arange(201)/100
        signal = np.arange(201) == 100
        for speed in (.02, .2):
            streams = still(2)
            streams['left_eef'] = stream(times, positions=speed*times+.1*signal)
            with self.subTest(speed=speed):
                self.assertEqual(len(detect(streams, 'state_jump')), int(speed == .02))

    def test_spike_and_stability_gates_must_match_the_same_component(self):
        times = np.arange(201)/100
        signal = np.arange(201) == 100
        streams = still(2)
        # The position spike has moving position context; a stationary
        # orientation cannot be used to pass its stability gate.
        streams['left_eef'] = stream(times, positions=.5*times+.1*signal, angles=2*signal)
        self.assertEqual(detect(streams, 'state_jump'), [])
        streams['left_eef'] = stream(times, positions=.5*times, angles=40*signal)
        self.assertEqual(len(detect(streams, 'state_jump')), 1)

    def test_spike_subtracts_neighbour_drift_and_uses_strict_amplitude(self):
        times = np.arange(301)/300
        signal = np.arange(301) == 150
        streams = still(1)
        streams['left_eef'] = stream(times, positions=.02*signal)
        self.assertEqual(detect(streams, 'state_jump'), [])
        config = replace(script.Config(), position_jump_min_m=.019)
        self.assertEqual(len(detect(streams, 'state_jump', config=config)), 1)
        # A 2.2 cm excursion followed by a permanent 0.5 cm shift has
        # insufficient excess amplitude after subtracting neighbour drift.
        positions = .022*signal+.005*(times > .5)
        streams['left_eef'] = stream(times, positions=positions)
        self.assertEqual(detect(streams, 'state_jump'), [])

    def test_duplicate_times_match_the_last_visible_sample(self):
        times = np.arange(301)/300
        duplicate = np.repeat(times, 2)
        positions = np.zeros(len(duplicate))
        positions[300] = 10
        streams = still(1)
        streams['left_eef'] = stream(duplicate, positions=positions)
        self.assertEqual(detect(streams, 'state_jump'), [])
        positions[301] = .1
        streams['left_eef'] = stream(duplicate, positions=positions)
        self.assertEqual(len(detect(streams, 'state_jump')), 1)

    def test_missing_context_short_streams_and_recording_gaps_are_ignored(self):
        streams = still(1)
        streams['left_eef'] = stream([0, .01, .02], positions=[0, 10, 0])
        self.assertEqual(detect(streams, 'state_jump'), [])
        times = np.arange(301)/300
        for sample in (3, 150, 297):
            signal = np.arange(301) == sample
            data = stream(times, positions=.1*signal)
            if sample == 150:
                data.times[151:] += 100_000_000
            streams['left_eef'] = data
            self.assertEqual(detect(streams, 'state_jump'), [])

    def test_noise_irregular_sampling_and_hidden_values_do_not_trigger(self):
        rng = np.random.default_rng(42)
        times = np.r_[0, np.cumsum(rng.uniform(.002, .005, 300))]
        held = np.floor(times/.02)*.02
        streams = still(2)
        streams['left_eef'] = stream(times, positions=.02*held+rng.uniform(-.001, .001, len(times)),
                                     angles=2*held+rng.uniform(-.2, .2, len(times)))
        self.assertEqual(detect(streams, 'state_jump'), [])
        sample = np.searchsorted(times, .5)
        streams['left_eef'].values[sample, 0] += .1
        rows = detect(streams, 'state_jump')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['start_ns'], streams['left_eef'].times[sample])

    def test_rotations_are_invariant_to_fixed_frames_and_quaternion_sign(self):
        times = np.arange(201)/100
        signal = np.arange(201) == 100
        data = stream(times, angles=170+40*signal)
        fixed = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        data.values[:, 3:6] = data.values[:, 3:6] @ fixed.T
        data.values[:, 6:9] = data.values[:, 6:9] @ fixed.T
        data.values[:, 9:12] = data.values[:, 9:12] @ fixed.T
        data.rotations = script.rotation_quaternions(data.values)
        streams = still(2)
        streams['left_eef'] = data
        expected = detect(streams, 'state_jump')
        self.assertEqual(len(expected), 1)
        data.rotations[::2] *= -1
        self.assertEqual(detect(streams, 'state_jump'), expected)
        streams['left_eef'] = stream(times, angles=170+500*times)
        self.assertEqual(detect(streams, 'state_jump'), [])

    def test_two_separated_spikes_merge_channels_without_duplicate_events(self):
        times = np.arange(601)/300
        signal = (np.arange(601) == 150) | (np.arange(601) == 450)
        streams = still(2)
        streams['left_eef'] = stream(times, positions=.1*signal, angles=40*signal)
        rows = detect(streams, 'state_jump')
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['start_ns'] for r in rows], [500_000_000, 1_500_000_000])

    def test_gripper_fast_continuous_open_close_is_not_a_jump(self):
        for hz in (30, 100, 300):
            times = np.arange(hz+1)/hz
            # Resolve the fastest transition with at least two intervals.
            for duration in (max(.04, 2/hz), .08, .15):
                phase = np.clip((times-.4)/duration, 0, 1)
                for signal in (phase, (1-np.cos(np.pi*phase))/2):
                    for name in ('left_gripper', 'right_gripper'):
                        for values in (signal, 1-signal):
                            with self.subTest(hz=hz, duration=duration, name=name,
                                              opening=values[-1] > values[0]):
                                streams = still(1)
                                streams[name] = stream(times, openness=values)
                                self.assertEqual(detect(streams, 'state_jump'), [])

    def test_gripper_observed_velocity_peaks_use_local_motion(self):
        # Native 30 Hz samples from reported continuous opening events.
        for peak in ([.427, .535, .748, .857, .905, .957],
                     [.332, .427, .754, .847, .897, .950],
                     [.0747, .3264, .638, .7104, .7176, .7186],
                     [.989, .976, .961, .939, .635, .587],
                     [.581, .584, .821, .868, .908, .934]):
            for values in (np.array(peak), 1-np.array(peak)):
                streams = still(1)
                streams['left_gripper'] = stream(np.arange(31)/30,
                    openness=np.r_[np.repeat(values[0], 12), values, np.repeat(values[-1], 13)])
                self.assertEqual(detect(streams, 'state_jump'), [])

    def test_rotations_near_half_turn_on_each_axis(self):
        poses = []
        for rotation in (np.eye(3), np.diag([1, -1, -1]),
                         np.diag([-1, 1, -1]), np.diag([-1, -1, 1])):
            poses.append(np.r_[np.zeros(3), rotation.T.reshape(-1)])
        quaternions = script.rotation_quaternions(np.array(poses))
        np.testing.assert_allclose(quaternions, np.eye(4))
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions[0], quaternions[1:]), 180)
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions, -quaternions), 0)

    def test_internal_idle_merges_native_changes_and_emits_four_records(self):
        streams = still(30)
        streams['left_eef'] = stream([0, 1, 5, 15, 30], positions=[0, .1, .1, .2, .2])
        streams['right_gripper'] = stream([0, 2, 15, 29], openness=[0, .2, .4, .4])
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual([r['stream'] for r in rows], list(script.STATE_NAMES))
        self.assertTrue(all(r['start_ns'] == 2_000_000_000 and r['end_ns'] == 15_000_000_000
                            for r in rows))
        self.assertEqual(detect(streams, 'internal_idle', end=30, config=replace(
            script.Config(), max_internal_idle_s=13)), [])
        streams['left_gripper'] = stream(np.arange(31), openness=np.arange(31) % 2)
        self.assertEqual(detect(streams, 'internal_idle', end=30), [])

    def test_internal_idle_does_not_mistake_smooth_motion_for_a_pause(self):
        # Large steps bracket smooth motion, so the old adjacent-frame rule
        # incorrectly reported the entire middle as idle.
        for hz in (30, 100, 300):
            times = np.arange(30*hz+1)/hz
            middle = np.clip(times-1, 0, 28)
            for name, channel, speed, step in (
                    ('left_eef', 'positions', .02, .1),
                    ('right_eef', 'angles', 10., 20.),
                    ('left_gripper', 'openness', .01, .1),
                    ('right_gripper', 'openness', .01, .1)):
                with self.subTest(hz=hz, channel=channel, name=name):
                    streams = still(30)
                    values = middle*speed + step*(times >= 1) + step*(times >= 29)
                    streams[name] = stream(times, **{channel: values})
                    self.assertEqual(detect(streams, 'internal_idle', end=30), [])

    def test_internal_idle_retains_real_pause_with_subthreshold_noise(self):
        times = np.arange(901)/30
        streams = still(30)
        signal = .1*(times >= 1) + .1*(times >= 20)
        streams['left_eef'] = stream(times, positions=signal+.001*np.sin(times*4))
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['start_ns'] == 1_000_000_000 and r['end_ns'] == 20_000_000_000
                            for r in rows))

    def test_internal_idle_ignores_hidden_duplicate_samples(self):
        streams = still(30)
        times = np.repeat(np.arange(31), 2)
        positions = .1*(times >= 1) + .1*(times >= 20)
        positions[::2] += 10  # Viewer only displays the final value at each time.
        streams['left_eef'] = stream(times, positions=positions)
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['start_ns'] == 1_000_000_000 and r['end_ns'] == 20_000_000_000
                            for r in rows))

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in ({'max_boundary_idle_s': -1}, {'position_jump_min_m': float('nan')},
                       {'openness_tolerance': float('inf')}, {'jump_max_duration_s': 0},
                       {'jump_context_s': .01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                script.Config(**kwargs)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'dataset'
        # Episode 0 is too short for any rule; episode 1 is still for exactly 5 s (frames 0..150).
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

    def test_read_episode_puts_every_stream_on_the_frame_clock(self):
        poses = np.tile(IDENTITY_POSE, (4, 1))
        poses[:, 0] = [0, .1, .2, .3]
        root = self.root.with_name('streams')
        dataset = make_dataset(root, {'still': {'left_gripper': np.zeros(2)},
                                      'moving': {'left_eef': poses, 'right_gripper': [0, .25, .5, 1]}})
        streams, end_ns = script.read_episode(dataset, dataset.episodes[1])
        times = [frame_ns(k) for k in range(4)]
        self.assertEqual(times, [0, 33_333_333, 66_666_667, 100_000_000])
        self.assertEqual(end_ns, times[-1])
        self.assertEqual(list(streams), list(script.STATE_NAMES))
        for name, stream in streams.items():
            self.assertEqual(stream.times.dtype, np.int64)
            self.assertEqual(stream.times.tolist(), times)
            self.assertEqual(stream.rotations is None, 'gripper' in name)
        np.testing.assert_allclose(streams['left_eef'].values, poses, atol=1e-6)
        np.testing.assert_allclose(streams['right_eef'].values, np.tile(IDENTITY_POSE, (4, 1)))
        np.testing.assert_allclose(streams['left_eef'].rotations, np.tile([1., 0, 0, 0], (4, 1)))
        np.testing.assert_allclose(streams['right_gripper'].values, [0, .25, .5, 1])
        np.testing.assert_allclose(streams['left_gripper'].values, 0)
        # The other episode sharing the data file is not mixed in.
        self.assertEqual(script.read_episode(dataset, dataset.episodes[0])[1], frame_ns(1))

    def test_report_uses_episode_index_and_last_frame_time_without_video_decode(self):
        with patch('av.open', side_effect=AssertionError('detection must not open videos')):
            self.assertEqual(check(self.root), (2, 2))
        rows = [json.loads(line) for line in (self.root / 'anomalies.jsonl').read_text().splitlines()]
        self.assertEqual(frame_ns(150), 5_000_000_000)
        self.assertEqual(rows, [dict(index=1, stream='left_eef', start_ns=0,
                                     end_ns=5_000_000_000, reason='boundary_idle')] * 2)

    def test_exported_one_frame_spike_is_reported_at_its_frame_times(self):
        openness = np.zeros(61)
        openness[30] = .6
        poses = np.tile(IDENTITY_POSE, (61, 1))
        poses[30, 0] = .1
        root = self.root.with_name('spike')
        make_dataset(root, {'spike': {'right_eef': poses, 'left_gripper': openness}})
        self.assertEqual(check(root), (1, 2))
        report = (root / 'anomalies.jsonl').read_text()
        self.assertEqual(report, expected_report(root))
        self.assertEqual([json.loads(line) for line in report.splitlines()], [
            dict(index=0, stream=name, start_ns=frame_ns(30), end_ns=frame_ns(31), reason='state_jump')
            for name in ('right_eef', 'left_gripper')])

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
        self.assertEqual(check(self.root, self.root.parent / 'elsewhere.jsonl'), (2, 2))

    def test_cli_and_threshold_override(self):
        result = subprocess.run([sys.executable, '-m', 'loom_data', 'check', 'run', str(self.root),
                                 '--max-boundary-idle-s', '5'], capture_output=True, text=True,
                                env=dict(os.environ, LOOM_DATA_NO_PROGRESS='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Checked 2 episodes; wrote 0 anomalies', result.stdout)
        self.assertEqual((self.root / 'anomalies.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
