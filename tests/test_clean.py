from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from support import IDENTITY_POSE, check, make_dataset
from loom_data import contract, resumable
from loom_data.check import anomalies, clean as script, job as check_job
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


def tree(*roots):
    """Relative path and size of every file under the given directories."""
    return [(str(path.relative_to(root.parent)), path.stat().st_size)
            for root in roots for path in sorted(root.rglob('*')) if path.is_file()]


def decoded(path):
    with av.open(str(path)) as reader:
        return [frame.to_ndarray(format='rgb24') for frame in reader.decode(video=0)]


class CleanDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.parent = Path(temp.name).resolve()
        cls.template = cls.parent / 'template'
        spiky = np.zeros(120)
        spiky[60:62] = .2
        source = make_dataset(cls.template, {
            'busy': {'right_gripper': busy(120)},
            'static': {'right_gripper': np.zeros(151)},
            'trim': {'right_gripper': steps(400, 150, 250), 'tasks': ['reach'] * 200 + ['place'] * 200},
            'pause': {'right_gripper': steps(600, 30, 400, 580)},
            'spike': {'left_eef': arm(spiky), 'right_gripper': busy(120)},
            'glitchy': {'left_eef': arm(spikes(120, 10, 20, 30, 40, 50, 60)), 'right_gripper': busy(120)},
        }, cameras=('top', 'left_wrist'))
        # What the dataset held before it is cleaned in place.
        cls.cameras, cls.info = source.cameras, source.info
        cls.stats = json.loads((cls.template / 'meta/stats.json').read_text())
        cls.frames = {e['episode_id']: source.frames(e) for e in source.episodes}
        cls.videos = {(e['episode_id'], camera): decoded(source.video(e, camera)[0])
                      for e in source.episodes if e['episode_id'] in ('trim', 'spike') for camera in source.cameras}
        cls.root = cls.copy('dataset')
        cls.removed = cls.parent / 'dataset.removed'
        cls.report = check(cls.root)
        cls.stale = check_job.work_directory(cls.root / 'anomalies.jsonl')
        cls.stale_existed = cls.stale.is_dir()
        cls.summary = script.clean(cls.root)
        cls.cleaned, cls.originals = contract.Dataset(cls.root), contract.Dataset(cls.removed)

    @classmethod
    def copy(cls, name):
        return Path(shutil.copytree(cls.template, cls.parent / name))

    def episode(self, dataset, episode_id):
        return next(e for e in dataset.episodes if e['episode_id'] == episode_id)

    def log(self):
        return [json.loads(line) for line in (self.removed / script.LOG).read_text().splitlines()]

    def test_dataset_is_cleaned_in_place_and_renumbered(self):
        self.assertEqual([(e['episode_index'], e['episode_id']) for e in self.cleaned.episodes],
                         [(0, 'busy'), (1, 'trim'), (2, 'spike')])
        start = next(e for e in self.log() if e['episode_id'] == 'trim')['start_frame']
        self.assertEqual(self.summary, {
            'episodes': 6, 'changed': 5, 'removed': str(self.removed), 'kept': 3, 'repaired_episodes': 1,
            'repaired_spikes': 1, 'trimmed_episodes': 1, 'removed_frames': 400 - (280 - start), 'dropped': 3,
            'dropped_boundary_idle': 1, 'dropped_internal_idle': 1, 'dropped_eef_spike': 1})
        self.assertEqual(self.cleaned.info['total_frames'], 120 + (280 - start) + 120)
        self.assertEqual(self.cleaned.info['robot_type'], self.info['robot_type'])
        self.assertEqual(self.cleaned.cameras, self.cameras)
        # Video files follow the new episode numbering and nothing else is left behind.
        for camera in self.cameras:
            directory = self.root / 'videos' / contract.video_key(camera) / 'chunk-000'
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             ['file-000.mp4', 'file-001.mp4', 'file-002.mp4'])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['data', 'meta', 'videos'])
        self.assertFalse(resumable.work_directory(self.root, '.clean-work').exists())

    def test_originals_of_every_dropped_or_changed_episode_form_a_dataset(self):
        self.assertEqual([e['episode_id'] for e in self.originals.episodes],
                         ['static', 'trim', 'pause', 'spike', 'glitchy'])
        self.assertEqual(self.originals.info['robot_type'], self.info['robot_type'])
        for episode in self.originals.episodes:
            state, tasks = self.originals.frames(episode)
            np.testing.assert_array_equal(state, self.frames[episode['episode_id']][0])
            self.assertEqual(tasks, self.frames[episode['episode_id']][1])
            for camera in self.cameras:
                with av.open(str(self.originals.video(episode, camera)[0])) as reader:
                    self.assertEqual(reader.streams.video[0].frames, episode['length'])
        for key in self.videos:
            original = decoded(self.originals.video(self.episode(self.originals, key[0]), key[1])[0])
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.videos[key], original, strict=True)))
        # The originals still show the anomalies; the cleaned dataset does not.
        self.assertEqual(check(self.removed)[0], 5)
        self.assertGreater(check(self.removed)[1], 5)
        self.assertEqual(check(self.root, self.parent / 'cleaned.jsonl'), (3, 0))

    def test_no_file_is_shared_between_or_within_the_datasets(self):
        videos = [p for root in (self.root, self.removed) for p in root.rglob('*.mp4')]
        self.assertEqual(len(videos), (3 + 5) * 2)
        self.assertTrue(all(path.stat().st_nlink == 1 for path in videos))
        self.assertEqual(len({(path.stat().st_dev, path.stat().st_ino) for path in videos}), len(videos))

    def test_log_lists_every_dropped_or_changed_episode(self):
        start = next(e for e in self.log() if e['episode_id'] == 'trim')['start_frame']
        self.assertEqual(self.log(), [
            {'episode_id': 'static', 'source_episode_index': 1, 'action': 'dropped', 'reason': 'boundary_idle'},
            {'episode_id': 'trim', 'source_episode_index': 2, 'action': 'kept', 'repaired_spikes': 0,
             'start_frame': start, 'end_frame': 280},
            {'episode_id': 'pause', 'source_episode_index': 3, 'action': 'dropped', 'reason': 'internal_idle'},
            {'episode_id': 'spike', 'source_episode_index': 4, 'action': 'kept', 'repaired_spikes': 1,
             'start_frame': 0, 'end_frame': 120},
            {'episode_id': 'glitchy', 'source_episode_index': 5, 'action': 'dropped', 'reason': 'eef_spike'}])

    def test_trimmed_episode_keeps_state_instructions_and_exact_video_frames(self):
        entry = next(e for e in self.log() if e['episode_id'] == 'trim')
        start, end = entry['start_frame'], entry['end_frame']
        # One second of idle is kept at each end; the head starts on a keyframe at or just before it.
        self.assertEqual(end, 280)
        self.assertIn(start, (119, 120))
        cleaned = self.episode(self.cleaned, 'trim')
        self.assertEqual(cleaned['length'], end - start)
        state, tasks = self.frames['trim']
        trimmed_state, trimmed_tasks = self.cleaned.frames(cleaned)
        np.testing.assert_array_equal(trimmed_state, state[start:end])
        self.assertEqual(trimmed_tasks, tasks[start:end])
        self.assertEqual(cleaned['tasks'], ['reach', 'place'])
        for camera in self.cameras:
            path, first = self.cleaned.video(cleaned, camera)
            self.assertEqual(first, 0)
            copied = decoded(path)
            self.assertTrue(all(np.array_equal(a, b)
                                for a, b in zip(self.videos['trim', camera][start:end], copied, strict=True)))
            with av.open(str(path)) as reader:
                self.assertEqual((reader.streams.video[0].frames, reader.streams.video[0].average_rate),
                                 (end - start, 30))

    def test_spike_is_repaired_in_the_frame_table_and_keeps_its_video(self):
        state = self.frames['spike'][0]
        repaired = self.cleaned.frames(self.episode(self.cleaned, 'spike'))[0]
        np.testing.assert_allclose(state[60:62, 0], .2, atol=1e-6)
        np.testing.assert_allclose(repaired[:, LEFT], np.tile(IDENTITY_POSE, (120, 1)), atol=1e-7)
        np.testing.assert_array_equal(np.delete(repaired, np.r_[LEFT], axis=1), np.delete(state, np.r_[LEFT], axis=1))
        np.testing.assert_array_equal(self.cleaned.frames(self.episode(self.cleaned, 'busy'))[0], self.frames['busy'][0])
        for camera in self.cameras:
            kept = decoded(self.cleaned.video(self.episode(self.cleaned, 'spike'), camera)[0])
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.videos['spike', camera], kept, strict=True)))

    def test_statistics_are_recomputed_for_state_and_reused_for_images(self):
        for root in (self.root, self.removed):
            stats = json.loads((root / 'meta/stats.json').read_text())
            total = contract.Dataset(root).info['total_frames']
            for camera in self.cameras:
                key = contract.video_key(camera)
                self.assertEqual({name: stats[key][name] for name in ('min', 'max', 'mean', 'std')},
                                 {name: self.stats[key][name] for name in ('min', 'max', 'mean', 'std')})
                self.assertEqual(stats[key]['count'], [total])
            self.assertEqual(stats[contract.STATE_KEY]['count'], [total])
        # The repaired spike no longer shows in the state statistics of the cleaned dataset.
        self.assertAlmostEqual(self.stats[contract.STATE_KEY]['max'][0], .2, places=6)
        self.assertEqual(json.loads((self.root / 'meta/stats.json').read_text())[contract.STATE_KEY]['max'][0], 0)

    def test_stale_anomaly_report_moves_out_with_its_recovery_state(self):
        self.assertEqual(self.report[0], 6)
        self.assertTrue(self.stale_existed)
        self.assertFalse((self.root / 'anomalies.jsonl').exists())
        self.assertFalse(self.stale.exists())
        self.assertEqual(len((self.removed / script.OLD_REPORT).read_text().splitlines()), self.report[1])

    def test_a_cleaned_dataset_is_not_cleaned_again(self):
        before = tree(self.root, self.removed)
        with self.assertRaisesRegex(ValueError, 'Removed directory exists'):
            script.clean(self.root)
        # With its originals elsewhere, a second pass finds nothing left to clean.
        self.assertIsNone(script.clean(self.root, self.parent / 'second.removed'))
        self.assertFalse((self.parent / 'second.removed').exists())
        self.assertEqual(tree(self.root, self.removed), before)

    def test_rejects_unsafe_removed_directories_and_leaves_the_dataset_alone_on_failure(self):
        root = self.copy('untouched')
        before = tree(root)
        for removed, message in ((root / 'removed', 'outside the dataset'), (self.parent, 'Removed directory exists')):
            with self.subTest(removed=removed), self.assertRaisesRegex(ValueError, message):
                script.clean(root, removed)
        with self.assertRaisesRegex(ValueError, 'Every episode would be dropped'):
            script.clean(root, max_spikes=0, keep_idle_s=0,
                         config=replace(anomalies.Config(), max_internal_idle_s=0, max_boundary_idle_s=0))
        with patch.object(script, 'copy_frames', side_effect=OSError('disk full')), self.assertRaises(OSError):
            script.clean(root)
        self.assertEqual(tree(root), before)
        self.assertFalse(root.with_name('untouched.removed').exists())
        self.assertFalse(resumable.work_directory(root, '.clean-work').exists())

    def test_interrupted_replacement_is_finished_by_running_again(self):
        expected = [(path.replace('dataset', 'crash', 1), size) for path, size in tree(self.root, self.removed)
                    if not path.endswith(script.OLD_REPORT)]
        moves = []
        with patch.object(script, 'move', side_effect=lambda *args: moves.append(args) or os.rename(*args)):
            script.clean(self.copy('count'))
        # 16 videos, then three directory swaps and the removed directory.
        self.assertEqual(len(moves), 16 + 7)
        for crash_after in (0, 1, 9, 16, 17, 20, 22):
            root = self.copy('crash')
            done = []

            def move(source, target):
                if len(done) == crash_after:
                    raise RuntimeError('power loss')
                done.append(os.rename(source, target))

            with self.subTest(crash_after=crash_after):
                with patch.object(script, 'move', side_effect=move), self.assertRaisesRegex(RuntimeError, 'power loss'):
                    script.clean(root)
                # Any arguments finish the interrupted run; they are not applied anew.
                summary = script.clean(root, max_spikes=0)
                self.assertTrue(summary.pop('resumed'))
                self.assertEqual(summary, {**self.summary, 'removed': str(root.with_name('crash.removed'))})
                self.assertEqual(tree(root, root.with_name('crash.removed')), expected)
                self.assertEqual(check(root), (3, 0))
                self.assertFalse(resumable.work_directory(root, '.clean-work').exists())
            shutil.rmtree(root)
            shutil.rmtree(root.with_name('crash.removed'))
            shutil.rmtree(check_job.work_directory(root / 'anomalies.jsonl'), ignore_errors=True)

    def test_cli_and_threshold_overrides(self):
        root, removed = self.copy('cli'), self.parent / 'elsewhere'
        command = [sys.executable, '-m', 'loom_data', 'clean', str(root), '--max-spikes', '6',
                   '--max-internal-idle-s', '13', '--keep-idle-s', '0.5', '--removed']
        result = subprocess.run([*command, str(removed)], capture_output=True, text=True,
                                env=dict(os.environ, LOOM_DATA_NO_PROGRESS='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Kept 5 of 6 episodes: repaired 7 spikes in 2, trimmed 1 (', result.stdout)
        self.assertIn('Dropped 1: 1 boundary_idle, 0 internal_idle, 0 eef_spike', result.stdout)
        self.assertIn(f'Originals of 4 episodes: {removed}', result.stdout)
        self.assertEqual([e['episode_id'] for e in contract.Dataset(root).episodes],
                         ['busy', 'trim', 'pause', 'spike', 'glitchy'])
        self.assertEqual([e['episode_id'] for e in contract.Dataset(removed).episodes],
                         ['static', 'trim', 'spike', 'glitchy'])
        result = subprocess.run([*command, str(removed)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Error: Removed directory exists', result.stderr)
        result = subprocess.run([*command, str(self.parent / 'other')], capture_output=True, text=True,
                                env=dict(os.environ, LOOM_DATA_NO_PROGRESS='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Nothing to clean; the dataset is unchanged', result.stdout)


if __name__ == '__main__':
    unittest.main()
