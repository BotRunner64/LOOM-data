from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check import anomaly_detection as script
from scripts.convert.export_common import write_state, write_video_index


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
        self.root = Path(self.temp.name)
        self.base = self.root / 'episodes/sample'
        for name, data in still(2).items():
            write_state(self.base / 'state' / f'{name}.parquet', data.times, data.values, 0,
                        pose=name.endswith('_eef'))
        (self.base / 'rgb').mkdir()
        write_video_index(self.base / 'rgb/top.parquet', [0, 5_000_000_000], 0)
        self.record = {'episode_id': 'sample', 'cameras': ['top'],
                       'instructions': [{'start_ns': 0, 'end_ns': 99_000_000_000, 'text': 'test'}]}
        self.manifest = self.root / 'episodes.jsonl'
        self.manifest.write_text('\n' + json.dumps(self.record) + '\n', encoding='utf-8')

    def test_jsonl_physical_line_index_camera_endpoint_and_no_video_decode(self):
        self.assertEqual(script.detect_dataset(self.root), (1, 2))
        rows = [json.loads(line) for line in (self.root / 'anomalies.jsonl').read_text().splitlines()]
        self.assertTrue(all(row == dict(index=1, stream='left_eef', start_ns=0,
                                       end_ns=5_000_000_000, reason='boundary_idle') for row in rows))

    def test_failure_preserves_previous_report_and_removes_temporary_file(self):
        output = self.root / 'report.jsonl'
        output.write_text('previous\n')
        self.manifest.write_text(json.dumps(self.record) + '\n{}\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            script.detect_dataset(self.root, output)
        self.assertEqual(output.read_text(), 'previous\n')
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_rejects_invalid_state_and_protects_inputs(self):
        for times, values in (([2, 1], [0, 0]), ([0, 1], [float('nan'), 0]),
                              ([0, 1], [0, 2]), ([0, None], [0, 0])):
            pq.write_table(pa.table({'timestamp_ns': pa.array(times, type=pa.int64()),
                                     'openness': values}), self.base / 'state/left_gripper.parquet')
            with self.subTest(times=times, values=values), self.assertRaises(ValueError):
                script.detect_dataset(self.root)
        for output in (self.manifest, self.base / 'report.jsonl', self.root / 'state.parquet'):
            with self.assertRaises(ValueError):
                script.detect_dataset(self.root, output)

    def test_cli_both_entrypoints_and_threshold_override(self):
        repo = Path(__file__).resolve().parents[1]
        for entry in (['scripts/check/anomaly_detection.py'], ['-m', 'scripts.check.anomaly_detection']):
            result = subprocess.run([sys.executable, *entry, str(self.root), '--max-boundary-idle-s', '5'],
                                    cwd=repo, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('wrote 0 anomalies', result.stdout)
            self.assertEqual((self.root / 'anomalies.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
