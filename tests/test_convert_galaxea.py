import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import support
from support import MODELS, assert_conversion_fails, datasets, episode_errors, read_export, video_means, write_video
from loom_data.convert import job
from loom_data.sources.galaxea import (
    DEFAULT_EEF_OFFSETS, TOOL_AXES, Galaxea, GalaxeaKinematics,
    instruction_segments, is_stationary, openness, source_path, validate_source,
)
from loom_data.geometry import pose_matrix
from loom_data.robot.kinematics import fk_poses
from loom_data.robot.urdf_model import parse_urdf

VIDEO_KEYS = {'observation.images.head_rgb': 'top',
              'observation.images.left_wrist_rgb': 'left_wrist',
              'observation.images.right_wrist_rgb': 'right_wrist'}


def source_info(kind, fps=30, video_keys=tuple(VIDEO_KEYS)):
    features = {'observation.state.torso': {'shape': [4]},
                'action.chassis.velocities': {'shape': [6]},
                'observation.state.chassis.velocities': {'shape': [3]}}
    for key in video_keys:
        features[key] = {'dtype': 'video', 'info': {'video.is_depth_map': False}}
    for side in ('left', 'right'):
        for key, size in [('arm', 6 if kind == 'r1lite' else 7), ('gripper', 1), ('ee_pose', 7)]:
            features[f'observation.state.{side}_{key}'] = {'shape': [size]}
        features[f'observation.state.{side}_ee_pose']['names'] = [
            f'/motion_control/pose_ee_arm_{side}.pose.{part}' for part in (
                'position.x', 'position.y', 'position.z',
                'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w')]
    return {'codebase_version': 'v2.1', 'robot_type': kind, 'fps': fps, 'features': features,
            'total_episodes': 1, 'chunks_size': 1000,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}


class GalaxeaMappingTests(unittest.TestCase):
    def test_embodiment_and_component_order_are_checked(self):
        for kind in DEFAULT_EEF_OFFSETS:
            info = source_info(kind)
            info['features']['observation.images.head_right_rgb'] = {'dtype': 'video'}
            self.assertEqual(validate_source(info), (kind, VIDEO_KEYS))
            info['features']['observation.state.left_arm']['shape'] = [8]
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                validate_source(info)
        info = source_info('r1lite')
        info['features']['observation.state.right_ee_pose']['names'].reverse()
        with self.assertRaisesRegex(ValueError, 'component order'):
            validate_source(info)
        info = source_info('r1lite')
        info['robot_type'] = 'r1'
        with self.assertRaisesRegex(ValueError, 'robot_type'):
            validate_source(info)

    def test_planar_chassis_commands_from_server_release(self):
        for kind in DEFAULT_EEF_OFFSETS:
            info = source_info(kind)
            info['features']['action.chassis.velocities']['shape'] = [3]
            self.assertEqual(validate_source(info)[0], kind)
            info['features']['action.chassis.velocities']['shape'] = [4]
            with self.assertRaisesRegex(ValueError, 'chassis command dimensions'):
                validate_source(info)
        self.assertTrue(is_stationary([[0., 0., 0.]] * 3, 3, 3))
        for component in range(3):
            commands = [[0.] * 3 for _ in range(3)]
            commands[1][component] = .01
            self.assertFalse(is_stationary(commands, 3, 3))
        for commands in ([[0.] * 6] * 3, [[0.] * 2] * 3,
                         [[0., 0., math.nan]] * 3, [[0., 0., math.inf]] * 3,
                         [[0., 0., 0.], [0., 0.], [0., 0., 0.]]):
            self.assertFalse(is_stationary(commands, 3, 3))

    def test_gripper_mm_normalization_and_nonfinite_rejection(self):
        np.testing.assert_allclose(openness([-1, 0, 50, 100, 101], 5), [0, 0, .5, 1, 1])
        np.testing.assert_allclose(openness([[25], [75]], 2), [.25, .75])
        for value in (math.nan, math.inf, None):
            with self.assertRaises(ValueError):
                openness([value], 1)

    def test_instructions_use_native_times_and_coarse_fallback(self):
        tasks = {0: 'null', 1: 'Pick', 2: 'Place', 3: 'Whole task'}
        self.assertEqual(instruction_segments([100, 130, 210, 280, 400],
                                             [0, 1, 1, 2, 2], [3]*5, tasks), [
            {'start_ns': 0, 'end_ns': 30, 'text': 'Whole task'},
            {'start_ns': 30, 'end_ns': 180, 'text': 'Pick'},
            {'start_ns': 180, 'end_ns': 300, 'text': 'Place'},
        ])
        self.assertEqual(instruction_segments([100, 100, 170], [1, 2, 2], [3]*3, tasks), [
            {'start_ns': 0, 'end_ns': 70, 'text': 'Place'}])
        # A label that starts on the last row still covers that row's frame.
        self.assertEqual(instruction_segments([100, 130, 210], [1, 1, 2], [3]*3, tasks), [
            {'start_ns': 0, 'end_ns': 110, 'text': 'Pick'},
            {'start_ns': 110, 'end_ns': 111, 'text': 'Place'}])
        with self.assertRaisesRegex(ValueError, 'Quality'):
            instruction_segments([0, 1], [0, 0], [1, 1], {0: 'qualified', 1: 'Pick'})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            instruction_segments([0, 1], [99, 99], [1, 1], tasks)
        with self.assertRaisesRegex(ValueError, 'zero-duration'):
            instruction_segments([0, 0], [1, 1], [3, 3], tasks)

    def test_stationary_screen_requires_finite_zero_commands_throughout(self):
        self.assertTrue(is_stationary([[0.]*6]*3, 3))
        self.assertTrue(is_stationary([[1e-7]*6]*3, 3))
        for component in range(6):
            commands = [[0.]*6 for _ in range(3)]
            commands[1][component] = -.1
            self.assertFalse(is_stationary(commands, 3))
        for commands in (None, [], [[math.nan]*6]*3, [[math.inf]*6]*3):
            self.assertFalse(is_stationary(commands, 3))
        self.assertFalse(is_stationary([], 0))

    def test_task_paths_cannot_escape(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            source_path(Path('/tmp/task'), '../outside', 0, 1000)

    def test_axes_are_right_handed_and_match_physical_tool_directions(self):
        np.testing.assert_array_equal(TOOL_AXES['r1lite'][:, 0], [1, 0, 0])
        np.testing.assert_array_equal(TOOL_AXES['r1lite'][:, 2], [0, 0, 1])
        np.testing.assert_array_equal(TOOL_AXES['r1pro'][:, 0], [0, 0, -1])
        np.testing.assert_array_equal(TOOL_AXES['r1pro'][:, 2], [1, 0, 0])
        for axes in TOOL_AXES.values():
            np.testing.assert_array_equal(np.cross(axes[:, 2], axes[:, 0]), axes[:, 1])
            self.assertAlmostEqual(np.linalg.det(axes), 1.)


@unittest.skipUnless(all((MODELS / f'galaxea_{k}/robot.json').exists()
                         for k in DEFAULT_EEF_OFFSETS), 'Galaxea model assets absent')
class GalaxeaPoseTests(unittest.TestCase):
    def test_recorded_poses_match_distinct_model_frames(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/galaxea_pose_samples.json').read_text())
        for kind in DEFAULT_EEF_OFFSETS:
            directory = MODELS / f'galaxea_{kind}'
            model = parse_urdf(directory / 'robot.urdf')
            converter = GalaxeaKinematics(directory, kind)
            for sample in (s for s in fixture['samples'] if s['robot_type'] == kind):
                state = sample['state']
                q = {f'arm_{side}_joint{i+1}': v for side in ('left', 'right')
                     for i, v in enumerate(state[f'observation.state.{side}_arm'])}
                q.update({f'torso_joint{i+1}': v for i, v in enumerate(state['observation.state.torso'][:converter.torso_count])})
                model_poses = fk_poses(model, q)
                native = [state[f'observation.state.{side}_ee_pose'] for side in ('left', 'right')]
                exported = converter.poses(state['observation.state.torso'], native)
                for side, pose, raw in zip(('left', 'right'), exported, native):
                    link = f'arm_{side}_link6' if kind == 'r1lite' else f'arm_{side}_gripper_base_link'
                    expected = model_poses[link]
                    # Joint and pose topics have small source synchronization errors.
                    mount_link = f'arm_{side}_link{6 if kind == "r1lite" else 7}'
                    np.testing.assert_allclose(pose[:3], model_poses[mount_link][:3, 3], atol=.01, rtol=0)
                    rotation = expected[:3, :3] @ converter.axes
                    np.testing.assert_allclose(pose[3:6], rotation[:, 0], atol=.03, rtol=0)
                    np.testing.assert_allclose(pose[6:9], rotation[:, 1], atol=.03, rtol=0)
                    # Check the exact composition independently of joint feedback.
                    combined = model_poses[converter.torso_link] @ pose_matrix(raw[:3], raw[3:])
                    np.testing.assert_allclose(pose[:3], (combined @ converter.eef)[:3], atol=1e-12)

    def test_torso_motion_changes_reference_pose_and_mount_rotates_with_tool(self):
        for kind in DEFAULT_EEF_OFFSETS:
            fk = GalaxeaKinematics(MODELS / f'galaxea_{kind}', kind)
            raw = [[.2, .3, .4, 0, 0, 0, 1]]*2
            before = fk.poses([0, 0, 0, 0], raw)[0]
            moved = fk.poses([.2, -.1, .3, .1 if kind == 'r1pro' else 0], raw)[0]
            self.assertGreater(np.linalg.norm(before[:3] - moved[:3]), .01)
            bad = copy.deepcopy(raw);bad[0][3:] = [0]*4
            with self.assertRaisesRegex(ValueError, 'quaternion'):
                fk.poses([0]*4, bad)
        with self.assertRaisesRegex(ValueError, 'padding'):
            GalaxeaKinematics(MODELS/'galaxea_r1lite', 'r1lite').poses([0, 0, 0, .1], raw)


TASKS = {0: 'null', 1: '收纳@Put away.', 2: 'Pick', 3: 'Place'}
WHOLE = TASKS[1]
SKIPPED = Galaxea.skip_reason
LAYOUT = 'top+left_wrist+right_wrist_640x480'
GRAY_OFFSETS = {'top': 0, 'left_wrist': 20, 'right_wrist': 40}  # tells the cameras apart


def native_pose(index, step_degrees=10.):
    """Row `index` of the fixture trajectory: the tool moves along x and turns about native z."""
    half = math.radians(step_degrees * index) / 2
    return [.2 + .03 * index, .3, .4, 0., 0., math.sin(half), math.cos(half)]


def torso_state(kind):
    return [.1, -.2, .3, .1 if kind == 'r1pro' else 0.]


def write_task(directory, kind, *, fps=30, timestamps=(10., 10.04, 10.11), grippers=(0., 50., 100.),
               task_indices=None, step_degrees=10., video_keys=tuple(VIDEO_KEYS), size=(64, 48), video_frames=None):
    """One extracted Galaxea task with a single stationary episode. Returns the frame gray levels of `top`."""
    shutil.rmtree(directory, ignore_errors=True)
    count = len(timestamps)
    (directory/'meta').mkdir(parents=True)
    info = source_info(kind, fps, video_keys)
    (directory/'meta/info.json').write_text(json.dumps(info))
    (directory/'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': count})+'\n')
    (directory/'meta/tasks.jsonl').write_text(
        '\n'.join(json.dumps({'task_index': i, 'task': text}) for i, text in TASKS.items())+'\n')
    data = {'timestamp': list(timestamps), 'frame_index': list(range(count)), 'episode_index': [0]*count,
            'task_index': list(task_indices or [0]*count), 'coarse_task_index': [1]*count,
            'observation.state.torso': [torso_state(kind)]*count, 'action.chassis.velocities': [[0.]*6]*count,
            'observation.state.chassis.velocities': [[0.]*3]*count}
    for side in ('left', 'right'):
        data[f'observation.state.{side}_arm'] = [[0.]*(6 if kind == 'r1lite' else 7)]*count
        data[f'observation.state.{side}_ee_pose'] = [native_pose(i, step_degrees) for i in range(count)]
        data[f'observation.state.{side}_gripper'] = list(grippers)
    path = source_path(directory, info['data_path'], 0, 1000)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(data), path)
    grays = [round(i * 200 / (count - 1)) for i in range(count)][:video_frames]
    for key in video_keys:
        write_video(source_path(directory, info['video_path'], 0, 1000, key),
                    [g + GRAY_OFFSETS[VIDEO_KEYS[key]] for g in grays], size=size, rate=fps)
    return grays


def convert(source, output, **options):
    """Convert and return the number of exported episodes."""
    support.convert('galaxea', source, output, **options)
    return len(support.manifest(output))


def skips(output):
    """Episode id -> recorded reason of every episode a conversion left out."""
    records = [json.loads(p.read_text()) for p in sorted((job.work_directory(output)/'records').glob('*.json'))]
    return {r['episode_id']: r['reason'] for r in records if r['outcome'] == 'skipped'}


def expected_poses(kind, count, step_degrees=10.):
    """Contract poses of the fixture rows, straight from the kinematics (both arms record the same pose)."""
    kinematics = GalaxeaKinematics(MODELS/f'galaxea_{kind}', kind)
    return np.array([kinematics.poses(torso_state(kind), [native_pose(i, step_degrees)])[0] for i in range(count)])


@unittest.skipUnless(all((MODELS / f'galaxea_{k}/robot.json').exists()
                         for k in DEFAULT_EEF_OFFSETS), 'Galaxea model assets absent')
class GalaxeaExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.grays = {kind: write_task(self.source/kind, kind) for kind in DEFAULT_EEF_OFFSETS}

    def assert_video(self, episode, grays, size):
        for camera, (path, first) in episode['videos'].items():
            with av.open(str(path)) as reader:
                self.assertEqual(len(reader.streams), 1)
                stream = reader.streams.video[0]
                self.assertEqual((stream.codec_context.name, stream.codec_context.pix_fmt), ('h264', 'yuv420p'))
                self.assertEqual((stream.width, stream.height, stream.average_rate), (*size, 30))
            np.testing.assert_allclose(video_means(path, first, episode['length']),
                                       [g + GRAY_OFFSETS[camera] for g in grays], atol=4)
            self.assertEqual(len(video_means(path)), episode['length'])  # one video file per episode

    def test_two_embodiments_export_the_same_contract(self):
        # Declared 30 fps: rows pass through one to one, whatever their jittery timestamps say.
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 2)
        self.assertEqual(list(datasets(output)), [LAYOUT])
        dataset = datasets(output)[LAYOUT]
        self.assertEqual((dataset.info['codebase_version'], dataset.info['fps'], dataset.info['robot_type']),
                         ('v3.0', 30, 'galaxea'))
        self.assertEqual({p.name for p in (output/LAYOUT).iterdir()}, {'meta', 'data', 'videos'})
        episodes = read_export(output)
        self.assertEqual([e['episode_id'] for e in episodes], ['r1lite_episode_000000', 'r1pro_episode_000000'])
        for kind, episode in zip(('r1lite', 'r1pro'), episodes):
            self.assertEqual(episode['length'], 3)
            self.assertEqual(sorted(episode['cameras']), ['left_wrist', 'right_wrist', 'top'])
            poses = expected_poses(kind, 3)
            for side in ('left', 'right'):
                np.testing.assert_allclose(episode[f'{side}_eef'], poses, atol=1e-6, rtol=0)
                np.testing.assert_allclose(episode[f'{side}_gripper'], [0., .5, 1.], atol=1e-7, rtol=0)
            self.assertEqual(episode['instructions'], [{'start_ns': 0, 'end_ns': 66_666_667, 'text': WHOLE}])
            self.assert_video(episode, self.grays[kind], (640, 480))
        # The two embodiments mount their tools differently, so the same native pose exports differently.
        self.assertGreater(np.abs(episodes[0]['left_eef'] - episodes[1]['left_eef']).max(), .01)
        table = pq.read_table(dataset.data_path(dataset.episodes[1]))
        self.assertEqual(table.schema.field('observation.state').type, pa.list_(pa.float32(), 26))
        np.testing.assert_allclose(table['timestamp'].to_pylist(), [0, 1/30, 2/30] * 2, atol=1e-7)
        self.assertEqual(table['frame_index'].to_pylist(), [0, 1, 2] * 2)
        # Rerunning a finished conversion changes nothing.
        self.assertEqual(convert(self.source, output), 2)

    def test_15_fps_rows_are_resampled_onto_30_fps_with_repeated_video_frames(self):
        grays = {kind: write_task(self.source/kind, kind, fps=15, timestamps=[10. + i/15 for i in range(4)],
                                  grippers=(0., 40., 100., 60.), task_indices=(0, 2, 2, 3),
                                  step_degrees=30., size=(64, 36)) for kind in DEFAULT_EEF_OFFSETS}
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 2)
        self.assertEqual(list(datasets(output)), ['top+left_wrist+right_wrist_640x360'])
        for kind, episode in zip(('r1lite', 'r1pro'), read_export(output)):
            self.assertEqual(episode['length'], 7)  # 0.2 s of source rows -> frames 0..6
            rows = expected_poses(kind, 4, 30.)
            # Half-way rotations, computed by the kinematics at the half-way angle: slerp must agree.
            between = expected_poses(kind, 7, 15.)
            for side in ('left', 'right'):
                pose = episode[f'{side}_eef']
                np.testing.assert_allclose(pose[::2], rows, atol=1e-6, rtol=0)             # frames on a source row
                np.testing.assert_allclose(pose[1::2, :3], (rows[:-1, :3] + rows[1:, :3]) / 2, atol=1e-6, rtol=0)
                np.testing.assert_allclose(pose[1::2, 3:], between[1::2, 3:], atol=1e-6, rtol=0)
                np.testing.assert_allclose(episode[f'{side}_gripper'], [0., .2, .4, .7, 1., .8, .6],
                                           atol=1e-6, rtol=0)
            # A row's label holds until the next row, 1/15 s later.
            self.assertEqual(episode['tasks'], [WHOLE, WHOLE, 'Pick', 'Pick', 'Pick', 'Pick', 'Place'])
            # A frame midway between two source frames shows the earlier one.
            self.assert_video(episode, [grays[kind][i] for i in (0, 0, 1, 1, 2, 2, 3)], (640, 360))

    def test_instruction_segments_land_on_their_frames(self):
        write_task(self.source/'r1lite', 'r1lite', timestamps=(10., 10.031, 10.07, 10.1, 10.135, 10.166),
                   grippers=(0.,)*6, task_indices=(0, 0, 2, 2, 3, 3))
        write_task(self.source/'r1pro', 'r1pro', task_indices=(0, 0, 3))  # a label starting on the last row
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 2)
        lite, pro = read_export(output)
        self.assertEqual(lite['tasks'], [WHOLE, WHOLE, 'Pick', 'Pick', 'Place', 'Place'])
        self.assertEqual(lite['instructions'], [
            {'start_ns': 0, 'end_ns': 66_666_667, 'text': WHOLE},
            {'start_ns': 66_666_667, 'end_ns': 133_333_333, 'text': 'Pick'},
            {'start_ns': 133_333_333, 'end_ns': 166_666_667, 'text': 'Place'}])
        self.assertEqual(pro['tasks'], [WHOLE, WHOLE, 'Place'])
        tasks = datasets(output)[LAYOUT].tasks
        self.assertEqual(sorted(tasks.values()), sorted([WHOLE, 'Pick', 'Place']))

    def test_a_gap_in_the_source_rows_is_skipped_with_a_reason(self):
        # 15 fps rows with three rows missing: frames inside the 267 ms hole have no state to interpolate.
        write_task(self.source/'r1pro', 'r1pro', fps=15, timestamps=[10. + i/15 for i in (0, 1, 5, 6)],
                   grippers=(0.,)*4)
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 1)
        self.assertEqual(skips(output), {'r1pro_episode_000000': 'sampling_gap'})
        self.assertEqual(episode_errors(output), '')
        self.assertEqual(read_export(output)[0]['episode_id'], 'r1lite_episode_000000')

    def test_other_camera_sets_are_skipped_as_unsupported_layout(self):
        write_task(self.source/'r1pro', 'r1pro', video_keys=tuple(VIDEO_KEYS)[:2])  # no right wrist camera
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 1)
        self.assertEqual(skips(output), {'r1pro_episode_000000': 'unsupported_layout'})
        self.assertEqual(list(datasets(output)), [LAYOUT])
        self.assertEqual(read_export(output)[0]['episode_id'], 'r1lite_episode_000000')

    def test_missing_camera_is_not_published(self):
        next((self.source/'r1pro').glob('videos/*/*/*.mp4')).unlink()
        output = self.root/'export'
        assert_conversion_fails(self, 'Error', 'galaxea', self.source, output)
        self.assertEqual(skips(output), {})

    def test_motion_is_skipped_and_limit_counts_candidates(self):
        path = next((self.source/'r1lite').glob('data/*/*.parquet'))
        rows = pq.read_table(path).to_pydict()
        rows['action.chassis.velocities'][1][0] = .1
        pq.write_table(pa.table(rows), path)
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 1)
        self.assertEqual(read_export(output)[0]['episode_id'], 'r1pro_episode_000000')
        self.assertEqual(skips(output), {'r1lite_episode_000000': SKIPPED})
        self.assertEqual({p.name for p in output.iterdir()}, {LAYOUT})
        # The limit selects candidates before conversion, so a skipped episode still uses it up.
        assert_conversion_fails(self, 'No episode was exported', 'galaxea', self.source, self.root/'limited',
                                limit_episodes=1)
        self.assertEqual(skips(self.root/'limited'), {'r1lite_episode_000000': SKIPPED})

    def test_invalid_commands_are_skipped_and_nothing_is_published(self):
        for path in self.source.glob('*/data/*/*.parquet'):
            rows = pq.read_table(path).to_pydict()
            rows['action.chassis.velocities'][1][0] = math.nan
            pq.write_table(pa.table(rows), path)
        output = self.root/'export'
        assert_conversion_fails(self, 'No episode was exported', 'galaxea', self.source, output)
        self.assertEqual(skips(output), {f'{kind}_episode_000000': SKIPPED for kind in DEFAULT_EEF_OFFSETS})
        self.assertEqual(episode_errors(output), '')  # skipped by the selection rule, not failed

    def test_wheel_feedback_is_not_required_or_used_for_selection(self):
        for kind in DEFAULT_EEF_OFFSETS:
            directory = self.source / kind
            path = next(directory.glob('data/*/*.parquet'))
            rows = pq.read_table(path).to_pydict()
            if kind == 'r1lite':
                rows['observation.state.chassis.velocities'] = [[99., math.nan, -99.]]*3
            else:
                del rows['observation.state.chassis.velocities']
            pq.write_table(pa.table(rows), path)
            info_path = directory / 'meta/info.json'
            info = json.loads(info_path.read_text())
            del info['features']['observation.state.chassis.velocities']
            info_path.write_text(json.dumps(info))
        self.assertEqual(convert(self.source, self.root/'export'), 2)

    def test_bad_timestamps_indices_and_video_count_are_rejected(self):
        path = next((self.source/'r1lite').glob('data/*/*.parquet'))
        original = pq.read_table(path).to_pydict()
        for column, bad, message in [('timestamp', [10., 9., 11.], 'unordered'),
                                     ('frame_index', [0, 2, 3], 'frame_index')]:
            rows = copy.deepcopy(original); rows[column] = bad
            pq.write_table(pa.table(rows), path)
            assert_conversion_fails(self, message, 'galaxea', self.source, self.root/'export')
        # A video with fewer frames than rows cannot be mapped row by row.
        write_task(self.source/'r1lite', 'r1lite', video_frames=2)
        assert_conversion_fails(self, 'Decoded 2 frames; expected 3', 'galaxea', self.source, self.root/'export')
        self.assertEqual(skips(self.root/'export'), {})

    def test_an_unregistered_aspect_ratio_is_rejected(self):
        write_task(self.source/'r1lite', 'r1lite', size=(32, 32))
        assert_conversion_fails(self, 'Aspect ratio 1:1', 'galaxea', self.source, self.root/'export')


if __name__ == '__main__':
    unittest.main()
