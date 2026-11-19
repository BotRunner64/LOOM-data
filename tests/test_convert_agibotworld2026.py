import math
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from support import (MODELS, assert_conversion_fails, convert, datasets, episode_errors, manifest, read_export,
                     video_means, write_video)
from loom_data.convert import job
from loom_data.sources.agibotworld2026 import (
    CAMERAS, G2FK, STATE_FIELDS, pose_matrix,
    instructions, openness, read_episode, seconds_to_ns, source_fields,
    verify_stationary_base,
)

SAMPLES = json.loads((Path(__file__).parent / 'fixtures/agibotworld2026_pose_samples.json').read_text())['samples']


class AgibotWorld2026MappingTests(unittest.TestCase):
    def test_source_seconds_become_exact_nanoseconds_for_resampling(self):
        self.assertEqual(seconds_to_ns(0.03333333507180214), 33_333_335)
        with self.assertRaises(ValueError):
            seconds_to_ns(math.nan)

    def test_gripper_zero_is_visually_open_and_negative_closes(self):
        self.assertEqual(openness(0), 1)
        self.assertEqual(openness(-0.455), 0.5)
        self.assertEqual(openness(-0.91), 0)
        self.assertEqual(openness(-1), 0)
        with self.assertRaises(ValueError):
            openness(math.inf)

    def test_depth_is_excluded_and_missing_odometry_is_preserved(self):
        names = (
            'state/left_effector/position', 'state/right_effector/position',
            'state/end/arm_position', 'state/end/arm_orientation', 'state/waist/position',
            'state/robot/position', 'state/robot/orientation',
        )
        lengths = (1, 1, 6, 8, 5, 0, 0)
        fields = {}
        offset = 0
        for name, length in zip(names, lengths):
            fields[name] = {'indices': list(range(offset, offset + length))}
            offset += length
        info = {'robot_type': 'g2a', 'features': {
            'observation.state': {'field_descriptions': fields},
            'observation.images.fisheye': {'video_info': {'video.is_depth_map': False}},
            'observation.images.top_head': {'video_info': {'video.is_depth_map': True}},
            'observation.images.head_depth': {'video_info': {'video.is_depth_map': True}},
            'observation.images.hand_left': {'video_info': {'video.is_depth_map': False}},
        }}
        indices, cameras = source_fields(info)
        self.assertEqual(indices['state/robot/position'], [])
        self.assertEqual(cameras, {'observation.images.hand_left': 'left_wrist'})

    def test_instruction_segments_use_sample_times_and_fill_gaps(self):
        info = {'instruction_segments': {'3': [
            {'track': 'default', 'instruction': 'Pick', 'start_frame_index': 1,
             'end_frame_index': 3},
            {'track': 'Left arm', 'instruction': 'Overlapping arm label',
             'start_frame_index': 0, 'end_frame_index': 4},
        ]}}
        times = [0, 31, 68, 102]
        # The segment covers frames 1 and 2; frame 3, the last one, falls back to the task.
        self.assertEqual(instructions(info, 3, 'Whole task', times), [
            {'start_ns': 0, 'end_ns': 31, 'text': 'Whole task'},
            {'start_ns': 31, 'end_ns': 102, 'text': 'Pick'},
            {'start_ns': 102, 'end_ns': 103, 'text': 'Whole task'},
        ])
        info['instruction_segments']['3'][0]['end_frame_index'] = 4
        self.assertEqual(instructions(info, 3, 'Whole task', times)[-1],
                         {'start_ns': 31, 'end_ns': 102, 'text': 'Pick'})
        self.assertEqual(instructions(info, 4, 'Whole task', times),
                         [{'start_ns': 0, 'end_ns': 102, 'text': 'Whole task'}])

    def test_missing_base_pose_requires_zero_base_commands(self):
        info = {'features': {'action': {'field_descriptions': {
            'action/robot/velocity': {'indices': [1, 3]},
        }}}}
        verify_stationary_base(info, [[4, 0, 5, 0], [6, 0, 7, 0]])
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            verify_stationary_base(info, [[4, 0, 5, 0], [6, 0, 7, 0.1]])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            verify_stationary_base(info, [[4, 0, 5, math.nan]])


@unittest.skipUnless((MODELS / 'agibot_g2/robot.json').exists(), 'G2 model assets absent')
class AgibotWorld2026PoseTests(unittest.TestCase):
    def setUp(self):
        self.fk = G2FK(MODELS / 'agibot_g2')
        self.samples = SAMPLES

    @staticmethod
    def pack(sample):
        state, indices = [], {}
        for key in STATE_FIELDS:
            indices[key] = list(range(len(state), len(state) + len(sample[key])))
            state.extend(sample[key])
        return state, indices

    def test_contract_axes_in_source_flange(self):
        np.testing.assert_array_equal(self.fk.axes[:, 0], [0, 0, 1])
        np.testing.assert_array_equal(self.fk.axes[:, 2], [1, 0, 0])
        np.testing.assert_array_equal(self.fk.axes[:, 1], np.cross(self.fk.axes[:, 2], self.fk.axes[:, 0]))
        self.assertAlmostEqual(np.linalg.det(self.fk.axes), 1)

    def test_origin_stays_at_recorded_flange_without_finger_length(self):
        from loom_data.robot.kinematics import fk_poses
        for sample in self.samples:
            state, indices = self.pack(sample)
            waist = dict(zip((f'body_joint{i}' for i in range(1, 6)), sample['state/waist/position']))
            torso = fk_poses(self.fk.robot, waist)['arm_base_link']
            for i, pose in enumerate(self.fk.poses(state, indices, True)):
                flange = np.asarray(sample['state/end/arm_position']).reshape(2, 3)[i]
                np.testing.assert_allclose(pose[:3], torso[:3, :3] @ flange + torso[:3, 3], atol=1e-12)

    def test_recorded_wrist_cameras_are_rigidly_attached_to_exported_tools(self):
        # Real source rows with independently recorded wrist extrinsics. Wrong
        # arm geometry, torso omission, frame order or quaternion order break
        # this invariant as the arms and torso move through the episode.
        relative = [[], []]
        for sample in self.samples:
            state, indices = self.pack(sample)
            for i, (side, pose) in enumerate(zip(('left', 'right'), self.fk.poses(state, indices, True))):
                pose = np.asarray(pose)
                rotation = pose[3:12].reshape(3, 3).T
                prefix = f'extrinsic_end_T_hand_{side}_rgbd_aligned/'
                camera_p = np.asarray(sample[prefix + 'translation_vector'])
                camera_r = np.asarray(sample[prefix + 'rotation_matrix']).reshape(3, 3)
                relative[i].append(np.concatenate((rotation.T @ (camera_p - pose[:3]), (rotation.T @ camera_r).ravel())))
        for samples in relative:
            np.testing.assert_allclose(samples, np.broadcast_to(samples[0], np.shape(samples)), atol=2e-6, rtol=0)

    def test_mobile_base_is_composed_on_the_left(self):
        sample = self.samples[2].copy()
        state, indices = self.pack(sample)
        fixed = self.fk.poses(state, indices, True)
        sample['state/robot/position'] = [2, -1, 0.3]
        sample['state/robot/orientation'] = [0, 0, math.sqrt(0.5), math.sqrt(0.5)]
        world = pose_matrix(sample['state/robot/position'], sample['state/robot/orientation'])
        state, indices = self.pack(sample)
        for old, new in zip(fixed, self.fk.poses(state, indices)):
            np.testing.assert_allclose(new[:3], world[:3, :3] @ old[:3] + world[:3, 3])
            np.testing.assert_allclose(new[3:6], world[:3, :3] @ old[3:6])
            np.testing.assert_allclose(new[6:9], world[:3, :3] @ old[6:9])
        sample['state/end/arm_orientation'] = [0] * 8
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            self.fk.poses(*self.pack(sample))

    def test_zero_base_pose_episode_uses_verified_fixed_base(self):
        sample = self.samples[0].copy()
        expected = self.fk.poses(*self.pack(sample), True)
        sample['state/robot/position'] = [0.] * 3
        sample['state/robot/orientation'] = [0.] * 4
        state, indices = self.pack(sample)
        original_indices = {key: value[:] for key, value in indices.items()}
        _, times, values = self.read_source_episode([state, state], indices, [[0., 0.]] * 2)
        self.assertEqual(times, [0, 100_000_000])
        for side, pose in zip(('left', 'right'), expected):
            np.testing.assert_allclose(values[f'{side}_eef'], [pose, pose])
        self.assertEqual(indices, original_indices)

    def test_zero_base_pose_rejects_movement_and_invalid_partial_odometry(self):
        sample = self.samples[0].copy()
        sample['state/robot/position'] = [0.] * 3
        sample['state/robot/orientation'] = [0.] * 4
        state, indices = self.pack(sample)
        corrupted = state.copy()
        corrupted[indices['state/robot/orientation'][-1]] = 0.5
        displaced = state.copy()
        displaced[indices['state/robot/position'][0]] = 0.1
        for command in (0.1, math.nan):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'base velocity'):
                self.read_source_episode([state, state], indices, [[0., 0.], [command, 0.]])
        for states in ([corrupted, state], [displaced, displaced]):
            with self.subTest(states=states), self.assertRaisesRegex(ValueError, 'quaternion'):
                self.read_source_episode(states, indices, [[0., 0.]] * 2)

    def test_partial_zero_base_pose_preserves_last_recorded_reference(self):
        sample = self.samples[0].copy()
        sample['state/robot/position'] = [2., -1., 0.3]
        sample['state/robot/orientation'] = [0., 0., math.sin(0.2), math.cos(0.2)]
        first, indices = self.pack(sample)
        sample['state/robot/position'] = [2.01, -1., 0.3]
        last, _ = self.pack(sample)
        sample['state/robot/position'] = [0.] * 3
        sample['state/robot/orientation'] = [0.] * 4
        missing, _ = self.pack(sample)
        first_poses = self.fk.poses(first, indices)
        last_poses = self.fk.poses(last, indices)
        states = [missing, first, missing, last, missing]
        _, _, values = self.read_source_episode(states, indices, [[0., 0.]] * 5)
        for i, side in enumerate(('left', 'right')):
            np.testing.assert_allclose(values[f'{side}_eef'], [first_poses[i]] * 3 + [last_poses[i]] * 2)
        with self.assertRaisesRegex(ValueError, 'base velocity'):
            self.read_source_episode(states, indices, [[0., 0.]] * 4 + [[0.1, 0.]])

    def test_recorded_base_pose_preserves_motion_without_action_column(self):
        sample = self.samples[0].copy()
        sample['state/robot/position'] = [2., -1., 0.3]
        sample['state/robot/orientation'] = [0., 0., 0., 1.]
        state, indices = self.pack(sample)
        expected = self.fk.poses(state, indices)
        _, _, values = self.read_source_episode([state, state], indices, None)
        for side, pose in zip(('left', 'right'), expected):
            np.testing.assert_allclose(values[f'{side}_eef'], [pose, pose])

    def read_source_episode(self, states, indices, actions):
        columns = {'observation.state': states, 'timestamp': [i * 0.1 for i in range(len(states))],
                   'frame_index': list(range(len(states))), 'episode_index': [0] * len(states)}
        if actions is not None:
            columns['action'] = actions
        data = io.BytesIO()
        pq.write_table(pa.table(columns), data)
        archive = SimpleNamespace(extractfile=lambda member: io.BytesIO(data.getvalue()))
        info = {'features': {'action': {'field_descriptions': {
            'action/robot/velocity': {'indices': [0, 1]},
        }}}}
        return read_episode(SimpleNamespace(name='data/data/chunk-000/episode_000000.parquet'),
                            archive, info, indices, self.fk)


LAYOUT = 'top+left_wrist+right_wrist_640x480'
GRAY_OFFSETS = {'top': 0, 'left_wrist': 20, 'right_wrist': 40}  # tells the cameras apart
SEGMENT = {'track': 'default', 'instruction': 'Pick', 'start_frame_index': 1, 'end_frame_index': 2}


def packed(sample):
    """One fixture sample as an observation.state row and its field indices."""
    state, indices = [], {}
    for key in STATE_FIELDS:
        indices[key] = list(range(len(state), len(state) + len(sample[key])))
        state.extend(sample[key])
    return state, indices


def frame_grays(count):
    return [round(i * 200 / (count - 1)) for i in range(count)]


def write_archive(path, *, fps=30, timestamps=(0., .031, .07), episodes=2, video_keys=tuple(CAMERAS),
                  missing_video=None, size=(64, 48)):
    """An AgiBot World 2026 archive whose episodes all hold the first fixture samples, one per row.

    Episode 0 has a default-track 'Pick' segment on row 1; the others only have their task.
    """
    count = len(timestamps)
    states = [packed(sample)[0] for sample in SAMPLES[:count]]
    info = {'robot_type': 'g2a', 'fps': fps, 'total_episodes': episodes, 'features': {
        'observation.state': {'field_descriptions': {key: {'indices': value}
                                                     for key, value in packed(SAMPLES[0])[1].items()}},
        **{key: {'video_info': {'video.is_depth_map': False}} for key in video_keys},
        'observation.images.head_depth': {'video_info': {'video.is_depth_map': True}},
        'action': {'field_descriptions': {'action/robot/velocity': {'indices': [0, 1, 2]}}},
    }, 'instruction_segments': {'0': [SEGMENT, {**SEGMENT, 'track': 'Left arm', 'instruction': 'Arm label',
                                                 'start_frame_index': 0, 'end_frame_index': count}]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary, tarfile.open(path, 'w:gz') as archive:
        def add(name, data):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        add('data/meta/info.json', json.dumps(info).encode())
        add('data/meta/episodes.jsonl', ''.join(
            json.dumps({'episode_index': i, 'tasks': [f'Task {i}']}) + '\n' for i in range(episodes)).encode())
        for index in range(episodes):
            table = pa.table({'observation.state': states, 'timestamp': list(timestamps),
                              'frame_index': list(range(count)), 'episode_index': [index] * count,
                              'action': [[0.] * 3] * count})
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            add(f'data/data/chunk-000/episode_{index:06d}.parquet', buffer.getvalue())
        for key in video_keys:
            video = write_video(Path(temporary) / f'{key}.mp4',
                                [g + GRAY_OFFSETS[CAMERAS[key]] for g in frame_grays(count)], size=size, rate=fps)
            for index in range(episodes):
                if (key, index) != missing_video:
                    add(f'data/videos/chunk-000/{key}/episode_{index:06d}.mp4', video.read_bytes())
        add('data/videos/chunk-000/observation.images.head_depth/episode_000000.mp4', b'not decoded')
    return path


def skips(output):
    """Episode id -> recorded reason of every episode a conversion left out."""
    records = [json.loads(p.read_text()) for p in sorted((job.work_directory(output)/'records').glob('*.json'))]
    return {r['episode_id']: r['reason'] for r in records if r['outcome'] == 'skipped'}


def turn(start, end, weight):
    """Rotation `weight` of the way from `start` to `end` along the shortest arc (axis-angle, not quaternions)."""
    relative = start.T @ end
    angle = math.acos(max(-1., min(1., (np.trace(relative) - 1) / 2)))
    if angle < 1e-12:
        return start
    axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]]) / (2 * math.sin(angle))
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    step = weight * angle
    return start @ (np.eye(3) + math.sin(step) * skew + (1 - math.cos(step)) * skew @ skew)


def blend(start, end, weight):
    """Contract pose `weight` of the way between two contract poses."""
    start, end = np.asarray(start), np.asarray(end)
    rotation = turn(start[3:].reshape(3, 3).T, end[3:].reshape(3, 3).T, weight)
    return np.concatenate((start[:3] * (1 - weight) + end[:3] * weight, rotation.T.reshape(-1)))


@unittest.skipUnless((MODELS / 'agibot_g2/robot.json').exists(), 'G2 model assets absent')
class AgibotWorld2026ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'out'
        fk = G2FK(MODELS / 'agibot_g2')
        # Contract poses and openness of the fixture rows, computed without the converter's episode reader.
        self.poses = [fk.poses(*packed(sample), True) for sample in SAMPLES[:3]]
        self.openness = {side: [openness(sample[f'state/{side}_effector/position'][0]) for sample in SAMPLES[:3]]
                         for side in ('left', 'right')}

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

    def test_declared_30_fps_archive_passes_rows_through(self):
        archive = write_archive(self.root / 'raw/task_7/3_1.tar.gz')
        convert('agibotworld2026', archive, self.output)
        self.assertEqual(list(datasets(self.output)), [LAYOUT])
        dataset = datasets(self.output)[LAYOUT]
        self.assertEqual((dataset.info['codebase_version'], dataset.info['fps'], dataset.info['robot_type']),
                         ('v3.0', 30, 'agibotworld2026'))
        episodes = read_export(self.output)
        self.assertEqual([e['episode_id'] for e in episodes],
                         ['task_7_3_1_episode_000000', 'task_7_3_1_episode_000001'])
        for episode in episodes:
            self.assertEqual(episode['length'], 3)
            self.assertEqual(sorted(episode['cameras']), ['left_wrist', 'right_wrist', 'top'])  # depth left out
            for k, side in enumerate(('left', 'right')):
                # Jittery row times (31 / 39 ms steps) change nothing: row k is frame k.
                np.testing.assert_allclose(episode[f'{side}_eef'], [pose[k] for pose in self.poses],
                                           atol=1e-6, rtol=0)
                np.testing.assert_allclose(episode[f'{side}_gripper'], self.openness[side], atol=1e-7, rtol=0)
            self.assert_video(episode, [0, 1, 2])
        # Only the default track is used; the overlapping per-arm track is ignored.
        self.assertEqual(episodes[0]['tasks'], ['Task 0', 'Pick', 'Task 0'])
        self.assertEqual(episodes[1]['tasks'], ['Task 1'] * 3)
        self.assertEqual(episodes[1]['instructions'], [{'start_ns': 0, 'end_ns': 66_666_667, 'text': 'Task 1'}])
        self.assertEqual(sorted(dataset.tasks.values()), ['Pick', 'Task 0', 'Task 1'])

    def test_another_declared_rate_is_resampled_with_repeated_video_frames(self):
        archive = write_archive(self.root / 'raw/task_7/3_1.tar.gz', fps=10, timestamps=(0., .1, .2),
                                episodes=1, size=(64, 40))
        convert('agibotworld2026', archive, self.output)
        self.assertEqual(list(datasets(self.output)), ['top+left_wrist+right_wrist_640x400'])
        episode, = read_export(self.output)
        self.assertEqual(episode['length'], 7)  # 0.2 s of source rows -> frames 0..6
        for k, side in enumerate(('left', 'right')):
            rows = [pose[k] for pose in self.poses]
            # Frame j is j/3 of a source interval after row 0: on a row every third frame, between otherwise.
            expected = [blend(rows[min(j // 3, 1)], rows[min(j // 3, 1) + 1], j / 3 - min(j // 3, 1))
                        for j in range(7)]
            np.testing.assert_allclose(expected[3], rows[1], atol=1e-12)
            np.testing.assert_allclose(episode[f'{side}_eef'], expected, atol=1e-6, rtol=0)
            np.testing.assert_allclose(episode[f'{side}_gripper'],
                                       np.interp(np.arange(7) / 3, [0, 1, 2], self.openness[side]),
                                       atol=1e-6, rtol=0)
        # The Pick segment covers source row 1, i.e. 100 ms <= t < 200 ms.
        self.assertEqual(episode['tasks'], ['Task 0'] * 3 + ['Pick'] * 3 + ['Task 0'])
        # Each frame takes the nearest of the 10 fps source frames.
        self.assert_video(episode, [0, 0, 1, 1, 1, 2, 2], (640, 400))

    def test_rows_too_sparse_to_resample_are_skipped_as_a_sampling_gap(self):
        write_archive(self.root / 'raw/task_7/3_1.tar.gz', episodes=1)
        write_archive(self.root / 'raw/task_7/4_1.tar.gz', fps=5, timestamps=(0., .2, .4))  # 200 ms between rows
        convert('agibotworld2026', self.root / 'raw', self.output)
        self.assertEqual(skips(self.output), {f'task_7_4_1_episode_00000{i}': 'sampling_gap' for i in range(2)})
        self.assertEqual(episode_errors(self.output), '')  # a contract drop, not a failure, even for a strict source
        self.assertEqual(manifest(self.output), [{'episode_id': 'task_7_3_1_episode_000000'}])

    def test_other_camera_sets_are_skipped_as_unsupported_layout(self):
        write_archive(self.root / 'raw/task_7/3_1.tar.gz', episodes=1)
        write_archive(self.root / 'raw/task_7/4_1.tar.gz', episodes=1,
                      video_keys=('observation.images.top_head', 'observation.images.hand_left'))
        convert('agibotworld2026', self.root / 'raw', self.output)
        self.assertEqual(skips(self.output), {'task_7_4_1_episode_000000': 'unsupported_layout'})
        self.assertEqual(list(datasets(self.output)), [LAYOUT])
        self.assertEqual(manifest(self.output), [{'episode_id': 'task_7_3_1_episode_000000'}])

    def test_a_declared_camera_without_its_video_fails_the_episode(self):
        archive = write_archive(self.root / 'raw/task_7/3_1.tar.gz',
                                missing_video=('observation.images.hand_right', 1))
        assert_conversion_fails(self, 'missing state or RGB for task_7_3_1_episode_000001',
                                'agibotworld2026', archive, self.output)
        self.assertEqual(skips(self.output), {})


if __name__ == '__main__':
    unittest.main()
