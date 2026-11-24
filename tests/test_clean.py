from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np

from support import IDENTITY_POSE, check, make_dataset
from loom_data import contract, resumable
from loom_data.check import anomalies, clean as script
from loom_data.contract import STATE_SLICES

LEFT, RIGHT_GRIPPER = STATE_SLICES['left_eef'], STATE_SLICES['right_gripper'].start


def episode(count, positions=None, angles=None, openness=None):
    """N x 26 state: the left arm moves along x and turns about z; the right gripper opens."""
    state = np.zeros((count, 26))
    for name in ('left_eef', 'right_eef'):
        state[:, STATE_SLICES[name]] = IDENTITY_POSE
    if positions is not None:
        state[:, 0] = positions
    if angles is not None:
        angles = np.deg2rad(angles)
        state[:, 3], state[:, 4], state[:, 6], state[:, 7] = np.cos(angles), np.sin(angles), -np.sin(angles), np.cos(angles)
    if openness is not None:
        state[:, RIGHT_GRIPPER] = openness
    return state


def busy(count):
    """Gripper openness that changes every half second, so the episode never idles."""
    return .5 * (np.arange(count) // 15 % 2)


def steps(count, *frames):
    """Gripper openness that steps up at each of the given frames."""
    return .2 * sum(np.arange(count) >= frame for frame in frames)


def spikes(count, *frames, height=.1, length=1):
    values = np.zeros(count)
    for frame in frames:
        values[frame:frame + length] = height
    return values


def arm(positions):
    poses = np.tile(IDENTITY_POSE, (len(positions), 1))
    poses[:, 0] = positions
    return poses


class CleanStateTests(unittest.TestCase):
    def test_an_episode_without_anomalies_is_kept_unchanged(self):
        state = episode(120, positions=.001 * np.arange(120), openness=busy(120))
        result = script.clean_state(state)
        self.assertEqual((result.start, result.end, result.repaired), (0, 120, 0))
        np.testing.assert_array_equal(result.state, state)

    def test_spikes_are_interpolated_between_the_frames_around_them(self):
        frames = np.arange(120)
        positions = np.where(frames < 40, 0., .006)
        angles = np.where(frames < 40, 0., 3.)
        positions[40:42], angles[40:42] = .1, 40.
        state = episode(120, positions=positions, angles=angles, openness=busy(120))
        result = script.clean_state(state)
        self.assertEqual((result.start, result.end, result.repaired), (0, 120, 1))
        np.testing.assert_allclose(result.state[38:44, 0], [0, 0, .002, .004, .006, .006], atol=1e-12)
        turned = np.degrees(np.arctan2(result.state[38:44, 4], result.state[38:44, 3]))
        np.testing.assert_allclose(turned, [0, 0, 1, 2, 3, 3], atol=1e-9)
        contract.check_state(result.state)
        # Only the spike frames of that arm change.
        untouched = np.ones(state.shape, dtype=bool)
        untouched[40:42, LEFT] = False
        np.testing.assert_array_equal(result.state[untouched], state[untouched])

    def test_an_episode_with_too_many_spikes_is_dropped(self):
        frames = (10, 20, 30, 40, 50, 60)
        five = episode(120, positions=spikes(120, *frames[:5]), openness=busy(120))
        self.assertEqual(script.clean_state(five).repaired, 5)
        six = episode(120, positions=spikes(120, *frames), openness=busy(120))
        self.assertEqual(script.clean_state(six), 'eef_spike')
        self.assertEqual(script.clean_state(six, max_spikes=6).repaired, 6)
        self.assertEqual(script.clean_state(five, max_spikes=0), 'eef_spike')
        # Both arms count towards the limit.
        six = episode(120, positions=spikes(120, *frames[:3]), openness=busy(120))
        six[:, STATE_SLICES['right_eef'].start] = spikes(120, *frames[3:])
        self.assertEqual(script.clean_state(six), 'eef_spike')

    def test_an_internal_pause_drops_the_episode(self):
        state = episode(600, openness=steps(600, 30, 400, 580))
        self.assertEqual(script.clean_state(state), 'internal_idle')
        self.assertEqual(script.clean_state(state, replace(anomalies.Config(), max_internal_idle_s=13)).end, 600)

    def test_idling_is_judged_after_spikes_are_repaired(self):
        # The spike splits a 15.7 s pause into two shorter ones, which detection alone does not report.
        state = episode(900, positions=spikes(900, 250), openness=steps(900, 30, 500, 700, 880))
        streams = anomalies.episode_streams(state)
        self.assertEqual([row['reason'] for row in anomalies.detect_episode(0, streams)], ['eef_spike'])
        self.assertEqual(script.clean_state(state), 'internal_idle')
        # Likewise a spike no longer ends the idle head early.
        state = episode(400, positions=spikes(400, 60), openness=steps(400, 200) + busy(400) * (np.arange(400) >= 200))
        result = script.clean_state(state)
        self.assertEqual((result.start, result.end, result.repaired), (170, 400, 1))

    def test_idle_ends_are_trimmed_to_one_second(self):
        result = script.clean_state(episode(600, openness=steps(600, 150, 300, 480)))
        self.assertEqual((result.start, result.end), (120, 510))
        self.assertEqual(script.clean_state(episode(600, openness=steps(600, 150, 300, 480)), keep_frames=0)[1:3],
                         (150, 480))
        # An end that idles for 3 s or less is left alone.
        result = script.clean_state(episode(600, openness=steps(600, 90, 300, 480)))
        self.assertEqual((result.start, result.end), (0, 510))
        result = script.clean_state(episode(600, openness=steps(600, 150, 300, 510)))
        self.assertEqual((result.start, result.end), (120, 600))

    def test_episodes_that_only_idle_are_dropped(self):
        self.assertEqual(script.clean_state(episode(600)), 'boundary_idle')
        # One step between two idle ends: every frame belongs to an end that would be trimmed.
        self.assertEqual(script.clean_state(episode(600, openness=steps(600, 300))), 'boundary_idle')
        # A short episode that never moves is not reported, yet has nothing to keep either.
        self.assertEqual(script.clean_state(episode(60)), 'boundary_idle')


class VideoTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)

    def video(self, name, keys, count=40):
        """A video whose keyframes are exactly the frames in `keys`."""
        path = self.directory / name
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('libx264', rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
            stream.options = {'bf': '0', 'x264-params': 'keyint=1000:min-keyint=1:scenecut=0'}
            for index in range(count):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 5 * index, np.uint8), format='rgb24')
                frame.pts, frame.time_base = index, Fraction(1, 30)
                frame.pict_type = av.video.frame.PictureType.I if index in keys else av.video.frame.PictureType.P
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        return path

    def frames(self, path):
        with av.open(str(path)) as reader:
            return [frame.to_ndarray(format='rgb24') for frame in reader.decode(video=0)]

    def test_cut_point_is_the_latest_keyframe_shared_by_all_videos(self):
        even = self.video('even.mp4', range(0, 40, 2))
        odd = self.video('odd.mp4', [0, *range(1, 40, 2)])
        sparse = self.video('sparse.mp4', [0, 12, 13, 30])
        self.assertEqual(script.common_keyframe([even], 21), 20)
        self.assertEqual(script.common_keyframe([even], 20), 20)
        self.assertEqual(script.common_keyframe([odd], 20), 19)
        # Videos that never share a keyframe again can only be cut at frame 0.
        self.assertEqual(script.common_keyframe([even, odd], 20), 0)
        self.assertEqual(script.common_keyframe([even, sparse], 20), 12)
        self.assertEqual(script.common_keyframe([odd, sparse], 20), 13)
        with self.assertRaisesRegex(ValueError, 'shorter than its episode'):
            script.common_keyframe([even], 40)

    def test_copy_keeps_the_frames_bit_exact(self):
        source = self.video('source.mp4', range(0, 40, 2))
        target = self.directory / 'copy.mp4'
        script.copy_frames(source, target, 12, 31)
        expected, copied = self.frames(source)[12:31], self.frames(target)
        self.assertEqual(len(copied), 19)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(expected, copied)))
        with av.open(str(target)) as reader:
            stream = reader.streams.video[0]
            self.assertEqual((stream.frames, stream.average_rate, len(reader.streams)), (19, 30, 1))
        with self.assertRaisesRegex(ValueError, 'frame 13 is not a keyframe'):
            script.copy_frames(source, self.directory / 'bad.mp4', 13, 31)
        with self.assertRaisesRegex(ValueError, 'shorter than its episode'):
            script.copy_frames(source, self.directory / 'long.mp4', 12, 41)


class CleanDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.parent = Path(temp.name).resolve()
        cls.root = cls.parent / 'source'
        spiky = np.zeros(120)
        spiky[60:62] = .2
        cls.source = make_dataset(cls.root, {
            'busy': {'right_gripper': busy(120)},
            'static': {'right_gripper': np.zeros(151)},
            'trim': {'right_gripper': steps(400, 150, 250), 'tasks': ['reach'] * 200 + ['place'] * 200},
            'pause': {'right_gripper': steps(600, 30, 400, 580)},
            'spike': {'left_eef': arm(spiky), 'right_gripper': busy(120)},
            'glitchy': {'left_eef': arm(spikes(120, 10, 20, 30, 40, 50, 60)), 'right_gripper': busy(120)},
        }, cameras=('top', 'left_wrist'))
        cls.output = cls.parent / 'cleaned'
        cls.summary = script.clean(cls.root, cls.output)
        cls.cleaned = contract.Dataset(cls.output)

    def source_episode(self, episode_id):
        return next(e for e in self.source.episodes if e['episode_id'] == episode_id)

    def cleaned_episode(self, episode_id):
        return next(e for e in self.cleaned.episodes if e['episode_id'] == episode_id)

    def decoded(self, dataset, episode, camera):
        with av.open(str(dataset.video(episode, camera)[0])) as reader:
            return [frame.to_ndarray(format='rgb24') for frame in reader.decode(video=0)]

    def test_dropped_episodes_are_left_out_and_the_rest_renumbered(self):
        self.assertEqual([(e['episode_index'], e['episode_id']) for e in self.cleaned.episodes],
                         [(0, 'busy'), (1, 'trim'), (2, 'spike')])
        start = self.cleaned_range('trim')[0]
        self.assertEqual(self.summary, {
            'episodes': 6, 'kept': 3, 'repaired_episodes': 1, 'repaired_spikes': 1, 'trimmed_episodes': 1,
            'removed_frames': 400 - (280 - start), 'dropped': 3, 'dropped_boundary_idle': 1,
            'dropped_internal_idle': 1, 'dropped_eef_spike': 1})
        self.assertEqual(self.cleaned.info['total_episodes'], 3)
        self.assertEqual(self.cleaned.info['total_frames'], 120 + (280 - start) + 120)
        self.assertEqual(self.cleaned.info['robot_type'], self.source.info['robot_type'])
        self.assertEqual(self.cleaned.cameras, self.source.cameras)

    def cleaned_range(self, episode_id):
        log = [json.loads(line) for line in (self.output / script.LOG).read_text().splitlines()]
        entry = next(e for e in log if e['episode_id'] == episode_id)
        return entry['start_frame'], entry['end_frame']

    def test_log_lists_every_dropped_or_changed_episode(self):
        log = [json.loads(line) for line in (self.output / script.LOG).read_text().splitlines()]
        start = self.cleaned_range('trim')[0]
        self.assertEqual(log, [
            {'episode_id': 'static', 'source_episode_index': 1, 'action': 'dropped', 'reason': 'boundary_idle'},
            {'episode_id': 'trim', 'source_episode_index': 2, 'action': 'kept', 'repaired_spikes': 0,
             'start_frame': start, 'end_frame': 280},
            {'episode_id': 'pause', 'source_episode_index': 3, 'action': 'dropped', 'reason': 'internal_idle'},
            {'episode_id': 'spike', 'source_episode_index': 4, 'action': 'kept', 'repaired_spikes': 1,
             'start_frame': 0, 'end_frame': 120},
            {'episode_id': 'glitchy', 'source_episode_index': 5, 'action': 'dropped', 'reason': 'eef_spike'}])

    def test_trimmed_episode_keeps_state_instructions_and_exact_video_frames(self):
        start, end = self.cleaned_range('trim')
        # One second of idle is kept at each end; the head starts on a keyframe at or just before it.
        self.assertEqual(end, 280)
        self.assertIn(start, (119, 120))
        source, cleaned = self.source_episode('trim'), self.cleaned_episode('trim')
        self.assertEqual(cleaned['length'], end - start)
        state, tasks = self.source.frames(source)
        trimmed_state, trimmed_tasks = self.cleaned.frames(cleaned)
        np.testing.assert_array_equal(trimmed_state, state[start:end])
        self.assertEqual(trimmed_tasks, tasks[start:end])
        self.assertEqual(cleaned['tasks'], ['reach', 'place'])
        for camera in self.cleaned.cameras:
            path, first = self.cleaned.video(cleaned, camera)
            self.assertEqual(first, 0)
            expected, copied = self.decoded(self.source, source, camera)[start:end], self.decoded(self.cleaned, cleaned, camera)
            self.assertEqual(len(copied), end - start)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(expected, copied)))
            with av.open(str(path)) as reader:
                self.assertEqual((reader.streams.video[0].frames, reader.streams.video[0].average_rate),
                                 (end - start, 30))

    def test_unchanged_videos_are_linked_not_copied(self):
        for episode_id in ('busy', 'spike'):
            for camera in self.cleaned.cameras:
                source = self.source.video(self.source_episode(episode_id), camera)[0]
                cleaned = self.cleaned.video(self.cleaned_episode(episode_id), camera)[0]
                self.assertTrue(os.path.samefile(source, cleaned))
        np.testing.assert_array_equal(self.cleaned.frames(self.cleaned_episode('busy'))[0],
                                      self.source.frames(self.source_episode('busy'))[0])

    def test_spike_is_repaired_in_the_frame_table(self):
        state = self.source.frames(self.source_episode('spike'))[0]
        repaired = self.cleaned.frames(self.cleaned_episode('spike'))[0]
        np.testing.assert_allclose(state[60:62, 0], .2, atol=1e-6)
        np.testing.assert_allclose(repaired[:, LEFT], np.tile(IDENTITY_POSE, (120, 1)), atol=1e-7)
        np.testing.assert_array_equal(np.delete(repaired, np.r_[LEFT], axis=1), np.delete(state, np.r_[LEFT], axis=1))

    def test_statistics_are_recomputed_for_state_and_reused_for_images(self):
        source = json.loads((self.root / 'meta/stats.json').read_text())
        cleaned = json.loads((self.output / 'meta/stats.json').read_text())
        total = self.cleaned.info['total_frames']
        for camera in self.cleaned.cameras:
            key = contract.video_key(camera)
            self.assertEqual({name: cleaned[key][name] for name in ('min', 'max', 'mean', 'std')},
                             {name: source[key][name] for name in ('min', 'max', 'mean', 'std')})
            self.assertEqual(cleaned[key]['count'], [total])
        self.assertEqual(cleaned[contract.STATE_KEY]['count'], [total])
        # The repaired spike no longer shows in the state statistics.
        self.assertAlmostEqual(source[contract.STATE_KEY]['max'][0], .2, places=6)
        self.assertEqual(cleaned[contract.STATE_KEY]['max'][0], 0)

    def test_cleaned_dataset_has_no_anomalies_and_the_source_is_untouched(self):
        self.assertEqual(check(self.output, self.parent / 'cleaned.jsonl'), (3, 0))
        self.assertEqual(contract.Dataset(self.root).info, self.source.info)
        self.assertEqual(len(contract.Dataset(self.root).episodes), 6)
        self.assertFalse(resumable.work_directory(self.output, '.clean-work').exists())

    def test_rejects_unsafe_outputs_and_publishes_nothing_on_failure(self):
        for output, message in ((self.output, 'Output exists'), (self.root / 'cleaned', 'outside the source dataset'),
                                (self.parent, 'Output exists')):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, message):
                script.clean(self.root, output)
        output = self.parent / 'nothing'
        with self.assertRaisesRegex(ValueError, 'Every episode was dropped'):
            script.clean(self.root, output, max_spikes=0, config=replace(anomalies.Config(), max_internal_idle_s=0,
                                                                         max_boundary_idle_s=0), keep_idle_s=0)
        self.assertFalse(output.exists())
        self.assertFalse(resumable.work_directory(output, '.clean-work').exists())

    def test_cli_and_threshold_overrides(self):
        output = self.parent / 'cli'
        result = subprocess.run([sys.executable, '-m', 'loom_data', 'clean', str(self.root), '--output', str(output),
                                 '--max-spikes', '6', '--max-internal-idle-s', '13', '--keep-idle-s', '0'],
                                capture_output=True, text=True, env=dict(os.environ, LOOM_DATA_NO_PROGRESS='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Kept 5 of 6 episodes: repaired 7 spikes in 2, trimmed 1 (', result.stdout)
        self.assertIn('Dropped 1: 1 boundary_idle, 0 internal_idle, 0 eef_spike', result.stdout)
        self.assertEqual([e['episode_id'] for e in contract.Dataset(output).episodes],
                         ['busy', 'trim', 'pause', 'spike', 'glitchy'])
        result = subprocess.run([sys.executable, '-m', 'loom_data', 'clean', str(self.root), '--output', str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Error: Output exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
