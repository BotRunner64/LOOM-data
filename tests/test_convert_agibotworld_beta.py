import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import h5py
import numpy as np

from support import (assert_conversion_fails, convert, datasets, episode_errors, manifest, read_export,
                     video_means, write_video)
from loom_data import sync
from loom_data.convert import job
from loom_data.sources import agibotworld_beta as script
from loom_data.geometry import pose_matrix, tool_transform

LAYOUT = 'top+left_wrist+right_wrist_640x480'
GRAY_OFFSETS = {'top': 0, 'left_wrist': 20, 'right_wrist': 40}  # tells the cameras apart
SEGMENTS = [{'start_frame': 0, 'end_frame': 1, 'action_text': 'Pick'},
            {'start_frame': 1, 'end_frame': 2, 'action_text': 'Twist'}]


def frame_grays(count):
    return [round(i * 200 / (count - 1)) for i in range(count)]


def make_source(root, episode=123, *, offsets=(0, 33_234_567, 66_765_432), end_position=None,
                end_orientation=None, cameras=tuple(script.CAMERAS), size=(64, 48), segments=SEGMENTS):
    """One Beta episode of task 694 under `root`; returns its state file.

    The default is a convertible episode: three cameras, 4:3 video, and jittery
    acquisition times at a nominal 30 fps (`offsets`, ns from the first row).
    """
    count = len(offsets)
    state = root/f'proprio_stats/694/{episode}/proprio_stats.h5'
    state.parent.mkdir(parents=True)
    origin = 1_739_382_608_553_487_000
    with h5py.File(state, 'w') as h:
        h['timestamp'] = np.array([origin + offset for offset in offsets], dtype='int64')
        h['state/end/position'] = np.zeros((count, 2, 3)) if end_position is None else np.asarray(end_position, float)
        h['state/end/orientation'] = (np.tile([0., 0., 0., 1.], (count, 2, 1)) if end_orientation is None
                                      else np.asarray(end_orientation, float))
        # Left closes (35 -> 120) while right opens.
        h['state/effector/position'] = np.column_stack((np.linspace(35., 120., count), np.linspace(120., 35., count)))
        h['state/robot/position'] = np.tile([1., 2., 0.], (count, 1))
        h['state/robot/orientation'] = np.tile([0., 0., 0., 1.], (count, 1))
    for name in cameras:
        write_video(root/f'observations/694/{episode}/videos/{name}',
                    [g + GRAY_OFFSETS[script.CAMERAS[name]] for g in frame_grays(count)], size=size)
    annotation = {'episode_id': episode, 'task_name': 'Twist cap', 'label_info': {'action_config': segments}}
    path = root/'task_info/task_694.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps((json.loads(path.read_text()) if path.exists() else []) + [annotation]))
    return state


def skips(output):
    """Episode id -> recorded reason of every episode a conversion left out."""
    records = [json.loads(p.read_text()) for p in sorted((job.work_directory(output)/'records').glob('*.json'))]
    return {r['episode_id']: r['reason'] for r in records if r['outcome'] == 'skipped'}


def expected_pose(position, degrees, side):
    """Contract pose of a flange at `position`, turned `degrees` about z, on the fixture's base at (1, 2, 0)."""
    half = math.radians(degrees) / 2
    flange = pose_matrix(position, [0., 0., math.sin(half), math.cos(half)])
    tool = tool_transform(script.DEFAULT_EEF_OFFSET, getattr(script, f'DEFAULT_{side.upper()}_TOOL_RPY'))
    transform = pose_matrix([1., 2., 0.], [0., 0., 0., 1.]) @ flange @ tool
    return np.concatenate((transform[:3, 3], transform[:3, :3].T.reshape(-1)))


class BetaMappingTests(unittest.TestCase):
    def test_real_flange_poses_match_independent_official_fk_in_base(self):
        samples = json.loads((Path(__file__).parent/'fixtures/agibotworld_beta_pose_samples.json').read_text())['samples']
        fields = {name: np.array([s[name] for s in samples]) for name in (
            'end/position', 'end/orientation', 'robot/position', 'robot/orientation')}
        # Identity mobile pose and identity tool isolate the recorded reference
        # frame; independent FK includes measured waist and arm joint values.
        fields['robot/position'][:] = 0
        fields['robot/orientation'][:] = [0, 0, 0, 1]
        poses = script.poses(fields, (np.eye(4), np.eye(4)))
        for k, side in enumerate(('left', 'right')):
            for sample, pose in zip(samples, poses[f'{side}_eef']):
                expected = np.array(sample['independent_base_flange'][k])
                np.testing.assert_allclose(pose[:3], expected[:3,3], atol=5e-6, rtol=0)
                np.testing.assert_allclose(pose[3:6], expected[:3,0], atol=3e-5, rtol=0)
                np.testing.assert_allclose(pose[6:9], expected[:3,1], atol=3e-5, rtol=0)

    def test_mobile_base_and_mount_compose_in_correct_order(self):
        fields = {'end/position': np.array([[[1,0,0], [1,0,0]]]),
                  'end/orientation': np.tile([0,0,0,1], (1,2,1)),
                  'robot/position': np.array([[2,3,0]]),
                  'robot/orientation': np.array([[0,0,math.sqrt(.5),math.sqrt(.5)]])}
        transforms = (tool_transform([0,0,.2], script.DEFAULT_LEFT_TOOL_RPY),
                      tool_transform([0,0,.2], script.DEFAULT_RIGHT_TOOL_RPY))
        poses = script.poses(fields, transforms)
        for side in ('left', 'right'):
            np.testing.assert_allclose(poses[f'{side}_eef'][0][:3], [2,4,.2], atol=1e-12)
        np.testing.assert_allclose(poses['left_eef'][0][3:6], [0,0,1], atol=1e-12)
        # Mounted palm backs are opposite in source flange coordinates.
        for transform, sign in zip(transforms, (1,-1)):
            np.testing.assert_allclose(transform[:3,2], [sign,0,0], atol=1e-12)
            np.testing.assert_allclose(np.cross(transform[:3,2],transform[:3,0]),transform[:3,1])

    def test_segment_boundaries_use_native_times_fill_gaps_and_reject_overlap(self):
        row = {'task_name': 'Whole task', 'label_info': {'action_config': [
            {'start_frame': 1, 'end_frame': 3, 'action_text': 'Pick'},
        ]}}
        times = [1000, 1031, 1068, 1102]
        # The segment covers frames 1 and 2; frame 3, the last one, falls back to the task.
        self.assertEqual(script.instructions(row,times), [
            {'start_ns':0,'end_ns':31,'text':'Whole task'},
            {'start_ns':31,'end_ns':102,'text':'Pick'},
            {'start_ns':102,'end_ns':103,'text':'Whole task'},
        ])
        row['label_info']['action_config'][0]['end_frame']=4
        self.assertEqual(script.instructions(row,times)[-1], {'start_ns':31,'end_ns':102,'text':'Pick'})
        self.assertEqual(script.instructions({'task_name': 'Whole task'},times),
                         [{'start_ns':0,'end_ns':102,'text':'Whole task'}])
        row['label_info']['action_config'].append({'start_frame':2,'end_frame':3,'action_text':'Overlap'})
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            script.instructions(row,times)

    def test_gripper_reading_increases_on_closure(self):
        self.assertEqual(script.normalized_gripper(35,120,35),1)
        self.assertEqual(script.normalized_gripper(77.5,120,35),.5)
        self.assertEqual(script.normalized_gripper(120,120,35),0)
        self.assertEqual(script.normalized_gripper(121,120,35),0)
        with self.assertRaises(ValueError):script.normalized_gripper(float('nan'),120,35)


class BetaExportTests(unittest.TestCase):
    def assert_video(self, episode, frames, size=(640, 480)):
        """Every camera shows source frame `frames[k]` at frame k."""
        grays = frame_grays(max(frames) + 1)
        for camera, (path, first) in episode['videos'].items():
            with av.open(str(path)) as reader:
                self.assertEqual(len(reader.streams), 1)
                stream = reader.streams.video[0]
                self.assertEqual((stream.codec_context.name, stream.codec_context.pix_fmt), ('h264', 'yuv420p'))
                self.assertEqual((stream.width, stream.height, stream.average_rate), (*size, 30))
            np.testing.assert_allclose(video_means(path, first, episode['length']),
                                       [grays[i] + GRAY_OFFSETS[camera] for i in frames], atol=4)
            self.assertEqual(len(video_means(path)), episode['length'])

    def test_nominal_30_fps_rows_pass_through_one_to_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            (root/'raw/observations/694/123/videos/back_left_fisheye_color.mp4').write_bytes(b'unused')
            convert('agibotworld_beta', root/'raw',root/'out')
            self.assertEqual(list(datasets(root/'out')), [LAYOUT])
            dataset = datasets(root/'out')[LAYOUT]
            self.assertEqual((dataset.info['codebase_version'], dataset.info['fps']), ('v3.0', 30))
            episode, = read_export(root/'out')
            self.assertEqual((episode['episode_id'], episode['length']), ('task_694_episode_123', 3))
            self.assertEqual(sorted(episode['cameras']), ['left_wrist', 'right_wrist', 'top'])
            # The jittery acquisition times (31.2 / 37.5 ms steps) are dropped: row k is frame k.
            for side in ('left', 'right'):
                np.testing.assert_allclose(episode[f'{side}_eef'], [expected_pose([0, 0, 0], 0, side)] * 3,
                                           atol=1e-6, rtol=0)
            np.testing.assert_allclose(episode['left_gripper'], [1., .5, 0.], atol=1e-7, rtol=0)
            np.testing.assert_allclose(episode['right_gripper'], [0., .5, 1.], atol=1e-7, rtol=0)
            # Segments cover frames 0 and 1; the uncovered last frame falls back to the task name.
            self.assertEqual(episode['tasks'], ['Pick', 'Twist', 'Twist cap'])
            self.assertEqual(episode['instructions'], [
                {'start_ns': 0, 'end_ns': 33_333_333, 'text': 'Pick'},
                {'start_ns': 33_333_333, 'end_ns': 66_666_667, 'text': 'Twist'},
                {'start_ns': 66_666_667, 'end_ns': 66_666_667, 'text': 'Twist cap'}])
            self.assert_video(episode, [0, 1, 2])

    def test_rows_at_another_rate_are_resampled_by_acquisition_time(self):
        # No declared rate: a 15 fps median row interval sends the episode through resampling.
        angles = [0., 30., 60., 90.]
        left = [[.1 * i, 0., 0.] for i in range(4)]
        right = [[0., .2 * i, .3] for i in range(4)]
        half = [math.radians(a) / 2 for a in angles]
        orientation = [[[0., 0., math.sin(h), math.cos(h)]] * 2 for h in half]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            make_source(root/'raw', offsets=(0, 66_666_667, 133_333_333, 200_000_000),
                        end_position=np.stack((left, right), axis=1), end_orientation=orientation, size=(64, 36))
            convert('agibotworld_beta', root/'raw',root/'out')
            self.assertEqual(list(datasets(root/'out')), ['top+left_wrist+right_wrist_640x360'])
            episode, = read_export(root/'out')
            self.assertEqual(episode['length'], 7)
            for side, positions in (('left', left), ('right', right)):
                # Frames 0, 2, 4, 6 sit on source rows; the odd ones are half-way in position and angle.
                expected = [expected_pose(np.add(positions[k // 2], positions[(k + 1) // 2]) / 2, 15. * k, side)
                            for k in range(7)]
                np.testing.assert_allclose(episode[f'{side}_eef'], expected, atol=1e-6, rtol=0)
            np.testing.assert_allclose(episode['left_gripper'], np.linspace(1., 0., 7), atol=1e-6, rtol=0)
            np.testing.assert_allclose(episode['right_gripper'], np.linspace(0., 1., 7), atol=1e-6, rtol=0)
            # Source rows are Pick, Twist, task, task; each holds for two frames.
            self.assertEqual(episode['tasks'], ['Pick', 'Pick', 'Twist', 'Twist'] + ['Twist cap'] * 3)
            # Frames midway between two source frames repeat the earlier one.
            self.assert_video(episode, [0, 0, 1, 1, 2, 2, 3], (640, 360))

    def test_a_hole_in_resampled_rows_is_dropped_as_a_sampling_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            # 15 fps rows with a hole wider than the state limit: no state to interpolate and no RGB
            # frame near the frames inside it.
            hole = sync.STATE_GAP_NS // 66_666_667 + 2
            make_source(root/'raw', 124, offsets=[i * 66_666_667 for i in (*range(4), *range(3 + hole, 7 + hole))])
            convert('agibotworld_beta', root/'raw',root/'out')
            self.assertEqual(skips(root/'out'), {'task_694_episode_124': 'sampling_gap'})
            self.assertEqual(manifest(root/'out'), [{'episode_id': 'task_694_episode_123'}])
            summary = json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual(summary['drop_reasons'], {'sampling_gap': 1})

    def test_other_camera_sets_are_skipped_as_unsupported_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            make_source(root/'raw', 124, cameras=('head_color.mp4', 'hand_left_color.mp4'))  # no right wrist
            convert('agibotworld_beta', root/'raw',root/'out')
            self.assertEqual(skips(root/'out'), {'task_694_episode_124': 'unsupported_layout'})
            self.assertEqual(episode_errors(root/'out'), '')
            self.assertEqual(list(datasets(root/'out')), [LAYOUT])
            self.assertEqual(manifest(root/'out'), [{'episode_id': 'task_694_episode_123'}])

    def test_dexhands_invalid_times_and_nonfinite_state_are_rejected(self):
        for invalid in ('dexhand','timestamp','nonfinite'):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as temporary:
                state=make_source(Path(temporary))
                with h5py.File(state,'a') as h:
                    if invalid=='dexhand':
                        del h['state/effector/position'];h['state/effector/position']=np.zeros((3,12))
                    elif invalid=='timestamp':h['timestamp'][2]=h['timestamp'][0]
                    else:h['state/end/position'][0,0,0]=float('nan')
                with self.assertRaises(ValueError):script.read_state(state)

    def test_video_failure_does_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            with patch.object(script,'transcode_video',side_effect=ValueError('frame mismatch')):
                assert_conversion_fails(self, 'Conversion failed', 'agibotworld_beta', root/'raw',root/'out')
            self.assertIn('frame mismatch', (root/'out.work/errors/task_694_episode_123.err').read_text())
            self.assertFalse((root/'out').exists())

    def test_an_unregistered_aspect_ratio_fails_instead_of_being_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw', size=(32, 32))
            assert_conversion_fails(self, 'Aspect ratio 1:1', 'agibotworld_beta', root/'raw',root/'out')
            self.assertEqual(skips(root/'out'), {})

    def test_parallel_adapter_publishes_and_resumes(self):
        from loom_data.convert import job as parallel
        from loom_data.convert.config import make_config
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            config=make_config('agibotworld_beta',root/'raw')
            parallel.prepare(config,root/'out')
            self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            parallel.prepare(config,root/'out')
            with patch.object(script,'convert_episode',side_effect=AssertionError('must skip committed episode')):
                self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            self.assertEqual(manifest(root/'out'), [{'episode_id': 'task_694_episode_123'}])


if __name__=='__main__':unittest.main()
