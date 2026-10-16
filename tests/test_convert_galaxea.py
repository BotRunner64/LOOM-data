import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert.galaxea import (
    ROOT, DEFAULT_TCP_OFFSETS, TOOL_AXES, GalaxeaKinematics, convert,
    instruction_segments, is_stationary, openness, source_path, validate_source,
)
from scripts.convert.export_common import pose_matrix, transcode_video
from scripts.robot.kinematics import fk_poses
from scripts.robot.urdf_model import parse_urdf


def source_info(kind):
    features = {'observation.state.torso': {'shape': [4]},
                'action.chassis.velocities': {'shape': [6]},
                'observation.state.chassis.velocities': {'shape': [3]},
                'observation.images.head_rgb': {'dtype': 'video', 'info': {'video.is_depth_map': False}}}
    for side in ('left', 'right'):
        for key, size in [('arm', 6 if kind == 'r1lite' else 7), ('gripper', 1), ('ee_pose', 7)]:
            features[f'observation.state.{side}_{key}'] = {'shape': [size]}
        features[f'observation.state.{side}_ee_pose']['names'] = [
            f'/motion_control/pose_ee_arm_{side}.pose.{part}' for part in (
                'position.x', 'position.y', 'position.z',
                'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w')]
    return {'codebase_version': 'v2.1', 'robot_type': kind, 'features': features,
            'total_episodes': 1, 'chunks_size': 1000,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}


class GalaxeaMappingTests(unittest.TestCase):
    def test_embodiment_and_component_order_are_checked(self):
        for kind in DEFAULT_TCP_OFFSETS:
            info = source_info(kind)
            self.assertEqual(validate_source(info), (kind, {'observation.images.head_rgb': 'head'}))
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
        for kind in DEFAULT_TCP_OFFSETS:
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


@unittest.skipUnless(all((ROOT / f'assets/robot_models/galaxea_{k}/robot.json').exists()
                         for k in DEFAULT_TCP_OFFSETS), 'Galaxea model assets absent')
class GalaxeaPoseTests(unittest.TestCase):
    def test_recorded_poses_match_distinct_model_frames(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/galaxea_pose_samples.json').read_text())
        for kind in DEFAULT_TCP_OFFSETS:
            directory = ROOT / f'assets/robot_models/galaxea_{kind}'
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
                    np.testing.assert_allclose(pose[:3], (expected @ converter.tcp)[:3], atol=.01, rtol=0)
                    rotation = expected[:3, :3] @ converter.axes
                    np.testing.assert_allclose(pose[3:6], rotation[:, 0], atol=.03, rtol=0)
                    np.testing.assert_allclose(pose[6:9], rotation[:, 1], atol=.03, rtol=0)
                    # Check the exact composition independently of joint feedback.
                    combined = model_poses[converter.torso_link] @ pose_matrix(raw[:3], raw[3:])
                    np.testing.assert_allclose(pose[:3], (combined @ converter.tcp)[:3], atol=1e-12)

    def test_torso_motion_changes_reference_pose_and_tcp_rotates_with_tool(self):
        for kind in DEFAULT_TCP_OFFSETS:
            fk = GalaxeaKinematics(ROOT / f'assets/robot_models/galaxea_{kind}', kind)
            raw = [[.2, .3, .4, 0, 0, 0, 1]]*2
            before = fk.poses([0, 0, 0, 0], raw)[0]
            moved = fk.poses([.2, -.1, .3, .1 if kind == 'r1pro' else 0], raw)[0]
            self.assertGreater(np.linalg.norm(before[:3] - moved[:3]), .01)
            bad = copy.deepcopy(raw);bad[0][3:] = [0]*4
            with self.assertRaisesRegex(ValueError, 'quaternion'):
                fk.poses([0]*4, bad)
        with self.assertRaisesRegex(ValueError, 'padding'):
            GalaxeaKinematics(ROOT/'assets/robot_models/galaxea_r1lite', 'r1lite').poses([0, 0, 0, .1], raw)


@unittest.skipUnless(all((ROOT / f'assets/robot_models/galaxea_{k}/robot.json').exists()
                         for k in DEFAULT_TCP_OFFSETS), 'Galaxea model assets absent')
class GalaxeaExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        for kind in DEFAULT_TCP_OFFSETS:
            directory = self.source / kind
            (directory/'meta').mkdir(parents=True)
            info = source_info(kind)
            (directory/'meta/info.json').write_text(json.dumps(info))
            (directory/'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 3})+'\n')
            (directory/'meta/tasks.jsonl').write_text('\n'.join(json.dumps(r) for r in [
                {'task_index': 0, 'task': 'null'}, {'task_index': 1, 'task': '收纳@Put away.'}])+'\n')
            data = {'timestamp': [10., 10.04, 10.11], 'frame_index': [0, 1, 2], 'episode_index': [0]*3,
                    'task_index': [0]*3, 'coarse_task_index': [1]*3,
                    'observation.state.torso': [[0.]*4]*3, 'action.chassis.velocities': [[0.]*6]*3,
                    'observation.state.chassis.velocities': [[0.]*3]*3}
            for side in ('left', 'right'):
                data[f'observation.state.{side}_arm'] = [[0.]*(6 if kind=='r1lite' else 7)]*3
                data[f'observation.state.{side}_ee_pose'] = [[.2, .3, .4, 0., 0., 0., 1.]]*3
                data[f'observation.state.{side}_gripper'] = [0., 50., 100.]
            path = source_path(directory, info['data_path'], 0, 1000)
            path.parent.mkdir(parents=True)
            pq.write_table(pa.table(data), path)
            video = source_path(directory, info['video_path'], 0, 1000, 'observation.images.head_rgb')
            video.parent.mkdir(parents=True)
            with av.open(str(video), 'w') as writer:
                stream = writer.add_stream('mpeg4', rate=7)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for i in range(3):
                    frame = av.VideoFrame.from_ndarray(np.full((32,32,3), i*70, np.uint8), format='rgb24')
                    for packet in stream.encode(frame):writer.mux(packet)
                for packet in stream.encode():writer.mux(packet)

    def test_two_embodiments_export_the_same_contract(self):
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 2)
        records = [json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['episode_id'] for r in records}), 2)
        for record in records:
            directory = output/'episodes'/record['episode_id']
            self.assertEqual(record['instructions'], [{'start_ns': 0, 'end_ns': 110000000, 'text': '收纳@Put away.'}])
            for side in ('left', 'right'):
                pose = pq.read_table(directory/'state'/f'{side}_eef.parquet')
                self.assertEqual(pose.schema.field('pose').type, pa.list_(pa.float64(), 12))
                self.assertFalse(pose.schema.field('pose').nullable)
                self.assertEqual(pose['timestamp_ns'].to_pylist(), [0, 40000000, 110000000])
                grip = pq.read_table(directory/'state'/f'{side}_gripper.parquet')
                self.assertEqual(grip['openness'].to_pylist(), [0., .5, 1.])
                self.assertEqual(grip.schema.field('openness').type, pa.float32())
            index = pq.read_table(directory/'rgb/head.parquet').to_pydict()
            self.assertEqual(index, {'frame_index': [0,1,2], 'timestamp_ns': [0,40000000,110000000]})
            with av.open(str(directory/'rgb/head.mp4')) as reader:
                self.assertEqual(len(reader.streams), 1)
                self.assertEqual(reader.streams.video[0].codec_context.name, 'h264')
                frames = list(reader.decode(video=0))
                self.assertEqual(len(frames), 3)
                self.assertLess(frames[0].to_ndarray().mean(), frames[-1].to_ndarray().mean())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            convert(self.source, output)

    def test_missing_camera_rolls_back_the_entire_export(self):
        next((self.source/'r1pro').glob('videos/*/*/*.mp4')).unlink()
        output = self.root/'export'
        with self.assertRaises(OSError):
            convert(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.export-*')))

    def test_motion_is_skipped_and_limit_counts_only_accepted_episodes(self):
        path = next((self.source/'r1lite').glob('data/*/*.parquet'))
        rows = pq.read_table(path).to_pydict()
        rows['action.chassis.velocities'][1][0] = .1
        pq.write_table(pa.table(rows), path)
        output = self.root/'export'
        self.assertEqual(convert(self.source, output, limit_episodes=1), 1)
        records = [json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines()]
        self.assertEqual(records[0]['episode_id'], 'r1pro_episode_000000')
        self.assertEqual({p.name for p in output.iterdir()}, {'episodes', 'episodes.jsonl'})
        self.assertFalse((output/'episodes/r1lite_episode_000000').exists())

    def test_invalid_commands_are_skipped_without_reason_files(self):
        for path in self.source.glob('*/data/*/*.parquet'):
            rows = pq.read_table(path).to_pydict()
            rows['action.chassis.velocities'][1][0] = math.nan
            pq.write_table(pa.table(rows), path)
        output = self.root/'export'
        self.assertEqual(convert(self.source, output), 0)
        self.assertEqual((output/'episodes.jsonl').read_text(), '')
        self.assertEqual(list((output/'episodes').iterdir()), [])
        self.assertEqual({p.name for p in output.iterdir()}, {'episodes', 'episodes.jsonl'})

    def test_wheel_feedback_is_not_required_or_used_for_selection(self):
        for kind in DEFAULT_TCP_OFFSETS:
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
        for column, bad, message in [('timestamp', [10.,9.,11.], 'unordered'),
                                      ('frame_index', [0,2,3], 'frame_index')]:
            rows = copy.deepcopy(original); rows[column] = bad
            pq.write_table(pa.table(rows), path)
            with self.assertRaisesRegex(ValueError, message):
                convert(self.source, self.root/'export')
        video = next((self.source/'r1lite').glob('videos/*/*/*.mp4'))
        with self.assertRaisesRegex(ValueError, 'expected 4'):
            transcode_video(video, self.root/'mismatch.mp4', 4)


if __name__ == '__main__':
    unittest.main()
