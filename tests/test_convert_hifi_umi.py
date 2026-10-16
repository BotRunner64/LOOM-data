import json
from fractions import Fraction
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert import hifi_umi as script


def make_source(root, frames=4):
    metadata_dir = root / 'source'
    metadata_dir.mkdir(parents=True)
    info = {
        'robot_type': 'simpleai_umi_4.0', 'codebase_version': 'v3.0',
        'state_layout': {'feature': 'observation.state', 'order': ['right', 'left'],
                         'rotation_6d_layout': 'first_two_rows', 'gripper_unit': 'rad',
                         'per_side': ['x', 'y', 'z', *(f'rot6d_{i}' for i in range(6)),
                                      'gripper_angle_rad']},
        'features': {'observation.state': {'shape': [20]},
                     **{key: {'dtype': 'video'} for key in script.CAMERAS}},
    }
    modality = {'state': {}}
    for side, offset in [('right', 0), ('left', 10)]:
        for kind, start, end in [('pose', offset, offset + 9), ('gripper', offset + 9, offset + 10)]:
            field = {'start': start, 'end': end, 'absolute': True, 'original_key': 'observation.state'}
            field.update({'rotation_type': 'rotation_6d', 'rotation_6d_layout': 'first_two_rows'}
                         if kind == 'pose' else {'unit': 'rad'})
            modality['state'][f'{side}_eef_{kind}'] = field
    (metadata_dir / 'info.json').write_text(json.dumps(info))
    (metadata_dir / 'modality.json').write_text(json.dumps(modality))
    pq.write_table(pa.table({'task_index': [0, 3], 'task': ['拿起杯子', 'Put down']}),
                   metadata_dir / 'tasks.parquet')
    episode = root / 'episodes/episode_000007'
    (episode / 'videos').mkdir(parents=True)
    # Right rotation is +90 degrees about Z. Left is +90 degrees about X.
    right = [1, 2, 3, 0, -1, 0, 1, 0, 0]
    left = [4, 5, 6, 1, 0, 0, 0, 0, -1]
    table = pa.table({
        'observation.state': pa.array([right + [x * script.DEFAULT_GRIPPER_OPEN_RAD]
                                      + left + [y * script.DEFAULT_GRIPPER_OPEN_RAD] for x, y in
                                       zip([-.1, .5, 1.2, 1], [1, .75, .25, 0])],
                                      type=pa.list_(pa.float32(), 20)),
        'observation.state_valid': pa.array([[True] * 20] * 4, type=pa.list_(pa.bool_(), 20)),
        # Not frame_index / FPS, and all streams start away from zero.
        'timestamp': pa.array([2., 2.031, 2.079, 2.141], type=pa.float32()),
        'frame_index': [0, 1, 2, 3], 'episode_index': [7] * 4,
        'index': [100, 101, 102, 103], 'task_index': [0, 0, 3, 3],
        'valid.frame': [True] * 4,
        # Deliberately unusable actions: the converter must never read them.
        'action': [[float('nan')] * 20] * 4,
    })
    pq.write_table(table, episode / 'data.parquet')
    (episode / 'episode.json').write_text(json.dumps({
        'episode_index': 7, 'length': 4, 'dataset_from_index': 100, 'dataset_to_index': 104,
        'tasks': ['拿起杯子', 'Put down'], 'videos': {key: {} for key in script.CAMERAS},
    }))
    for key in script.CAMERAS:
        with av.open(str(episode / 'videos' / f'{key}.mp4'), 'w') as output:
            stream = output.add_stream('libx264', rate=25)
            stream.width = stream.height = 32
            stream.pix_fmt = 'yuv420p'
            for i in range(frames):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 60, np.uint8), format='rgb24')
                frame.pts = i
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return episode


def make_shard(root, chunk=0, part=0, video_frames=8):
    """Two episodes sharing frame/video files, with a cut between keyframes."""
    shard = root / f'chunk-{chunk:04d}' / f'part-{part:04d}'
    with tempfile.TemporaryDirectory() as directory:
        subset = Path(directory)
        episode = make_source(subset)
        shutil.copytree(subset / 'source', shard / 'meta')
        info = json.loads((shard / 'meta/info.json').read_text())
        info.update(fps=25., total_episodes=2,
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                    video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        (shard / 'meta/info.json').write_text(json.dumps(info))
        first = pq.read_table(episode / 'data.parquet')
        second = first.set_column(first.schema.get_field_index('episode_index'), 'episode_index', pa.array([8] * 4))
        second = second.set_column(second.schema.get_field_index('index'), 'index', pa.array([104, 105, 106, 107]))
        data = shard / 'data/chunk-002/file-003.parquet'
        data.parent.mkdir(parents=True)
        pq.write_table(pa.concat_tables([first, second]), data, row_group_size=5)
        for number, index in enumerate((7, 8)):
            row = {'episode_index': index, 'length': 4, 'dataset_from_index': 100 + number * 4,
                   'dataset_to_index': 104 + number * 4, 'tasks': ['拿起杯子', 'Put down'],
                   'data/chunk_index': 2, 'data/file_index': 3}
            for key in script.CAMERAS:
                row.update({f'videos/{key}/chunk_index': 2, f'videos/{key}/file_index': 3,
                            f'videos/{key}/from_timestamp': number * .16,
                            f'videos/{key}/to_timestamp': (number + 1) * .16})
            metadata = shard / f'meta/episodes/chunk-{number:03d}/file-000.parquet'
            metadata.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([row]), metadata)
        original = None
        for key in script.CAMERAS:
            video = shard / f'videos/{key}/chunk-002/file-003.mp4'
            video.parent.mkdir(parents=True)
            if original is not None:
                shutil.copyfile(original, video)
                continue
            original = video
            with av.open(str(video), 'w') as output:
                stream = output.add_stream('libx264', rate=25)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                stream.options = {'g': '250', 'sc_threshold': '0'}
                for i in range(video_frames):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 28, np.uint8), format='rgb24')
                    frame.pts, frame.time_base = i, Fraction(1, 25)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
    return shard


class HifiUmiTests(unittest.TestCase):
    def test_full_shards_cut_nonkeyframe_boundaries_and_keep_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for chunk, part in ((0, 0), (0, 1)):
                make_shard(root / 'raw', chunk, part)
            script.convert(root / 'raw', root / 'out')
            records = [json.loads(line) for line in (root / 'out/episodes.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode_id'] for r in records], [
                'chunk_0000_part_0000_episode_000007', 'chunk_0000_part_0000_episode_000008',
                'chunk_0000_part_0001_episode_000007', 'chunk_0000_part_0001_episode_000008'])
            for number, record in enumerate(records):
                destination = root / 'out/episodes' / record['episode_id']
                times = pq.read_table(destination / 'state/right_eef.parquet')['timestamp_ns'].to_pylist()
                self.assertEqual(times, [script.seconds_to_ns(t) - 2_000_000_000 for t in
                                         np.array([2., 2.031, 2.079, 2.141], dtype=np.float32)])
                for camera in script.CAMERAS.values():
                    path = destination / 'rgb' / f'{camera}.mp4'
                    with av.open(str(path)) as video:
                        self.assertEqual(len(video.streams), 1)
                        self.assertEqual(video.streams.video[0].codec_context.name, 'h264')
                        frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 4)
                    np.testing.assert_allclose([f.to_ndarray(format='rgb24').mean() for f in frames],
                                               np.arange((number % 2) * 4, (number % 2) * 4 + 4) * 28, atol=4)
                    self.assertEqual(pq.read_table(path.with_suffix('.parquet'))['timestamp_ns'].to_pylist(), times)

    def test_shard_reader_reuses_projected_table_and_standalone_global_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = make_shard(root / 'raw')
            make_shard(root / 'raw', 1)
            reader = script.ShardReader(shard)
            with patch.object(script.pq, 'read_table', wraps=pq.read_table) as read:
                for index in (7, 8):
                    reader.convert_episode(index, root / str(index), 0., script.DEFAULT_GRIPPER_OPEN_RAD)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(read.call_args.kwargs['columns'], list(script.COLUMNS))
            script.convert(root / 'raw', root / 'out', limit=3)
            self.assertEqual(len((root / 'out/episodes.jsonl').read_text().splitlines()), 3)

    def test_short_shard_video_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shard(root / 'raw', video_frames=7)
            with self.assertRaisesRegex(ValueError, 'expected 4 frames, decoded 3'):
                script.convert(root / 'raw', root / 'out')
            self.assertFalse((root / 'out').exists())
            self.assertFalse(list(root.glob('.out-*')))

    def test_shard_metadata_rejects_duplicate_indices_and_bad_video_range(self):
        for bad_range in (False, True):
            with self.subTest(bad_range=bad_range), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shard = make_shard(root)
                path = shard / 'meta/episodes/chunk-001/file-000.parquet'
                row = pq.read_table(path).to_pylist()[0]
                if bad_range:
                    row[f'videos/{next(iter(script.CAMERAS))}/to_timestamp'] += .04
                else:
                    row['episode_index'] = 7
                pq.write_table(pa.Table.from_pylist([row]), path)
                with self.assertRaises(ValueError):
                    script.ShardReader(shard)

    def test_rotation_is_repacked_not_transposed_and_preserves_body_delta(self):
        # A non-axis-aligned rotation catches mistakes hidden by identity poses.
        from scripts.robot.kinematics import rpy_matrix
        rotation = rpy_matrix([.2, -.4, .7])
        source = np.r_[1, 2, 3, rotation[:2].ravel()]
        pose = script.poses_from_rows([source])[0]
        np.testing.assert_allclose(pose, np.r_[1, 2, 3, rotation.T.reshape(-1)], atol=1e-12)
        reconstructed = pose[3:12].reshape(3, 3).T
        np.testing.assert_allclose(reconstructed.T @ [1, 0, 0], rotation.T @ [1, 0, 0])
        for bad in ([0] * 9, [1, 2, 3, 1, 0, 0, 1, 0, 0], [float('nan')] * 9):
            with self.assertRaises(ValueError):
                script.poses_from_rows([bad])

    def test_gripper_uses_fixed_endpoints_and_rejects_nonfinite(self):
        np.testing.assert_allclose(script.openness(np.deg2rad([0, 17.5, 35, 70])), [0, .5, 1, 1])
        np.testing.assert_allclose(script.openness([-.2, .1, .4, .7, 1], .1, .7), [0, 0, .5, 1, 1])
        for closed, opened in [(1, 1), (1, 0), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                script.openness([.2], closed, opened)
        for angle in [float('nan'), float('inf'), float('-inf')]:
            with self.assertRaises(ValueError):
                script.openness([angle], 0, 1)

    def test_end_to_end_six_videos_schema_and_native_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            output = root / 'out'
            script.convert(root / 'raw', output)
            records = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record['cameras'], sorted(script.CAMERAS.values()))
            exported = output / 'episodes' / record['episode_id']
            times = [script.seconds_to_ns(t) - 2_000_000_000 for t in
                     pq.read_table(episode / 'data.parquet')['timestamp'].to_pylist()]
            self.assertEqual(record['instructions'], [
                {'start_ns': 0, 'end_ns': times[2], 'text': '拿起杯子'},
                {'start_ns': times[2], 'end_ns': times[-1], 'text': 'Put down'},
            ])
            for side in ['right', 'left']:
                pose = pq.read_table(exported / 'state' / f'{side}_eef.parquet')
                self.assertEqual(pose.schema.field('pose').type, pa.list_(pa.float64(), 12))
                self.assertFalse(pose.schema.field('pose').nullable)
                self.assertEqual(pose['timestamp_ns'].to_pylist(), times)
                gripper = pq.read_table(exported / 'state' / f'{side}_gripper.parquet')
                self.assertEqual(gripper.schema.field('openness').type, pa.float32())
                np.testing.assert_allclose(gripper['openness'].to_pylist(),
                                           [0, .5, 1, 1] if side == 'right' else [1, .75, .25, 0],
                                           atol=1e-7)
            self.assertEqual(pq.read_table(exported / 'state/right_eef.parquet')['pose'][0].as_py(),
                             [1, 2, 3, 0, 1, 0, -1, 0, 0, 0, 0, 1])
            for key, camera in script.CAMERAS.items():
                video = exported / 'rgb' / f'{camera}.mp4'
                self.assertEqual(video.read_bytes(), (episode / 'videos' / f'{key}.mp4').read_bytes())
                table = pq.read_table(video.with_suffix('.parquet'))
                self.assertEqual(table['frame_index'].to_pylist(), [0, 1, 2, 3])
                self.assertEqual(table['timestamp_ns'].to_pylist(), times)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                script.convert(root / 'raw', output, gripper_open_rad=1.)

    def test_bad_inputs_fail_without_publishing(self):
        mutations = {
            'invalid pose mask': ('observation.state_valid', [[False] + [True] * 19] * 4),
            'null mask': ('observation.state_valid', [[None] + [True] * 19] * 4),
            'invalid frame': ('valid.frame', [True, False, True, True]),
            'null frame': ('valid.frame', [True, None, True, True]),
            'bad ordering': ('timestamp', [0, .2, .1, .3]),
            'bad frame index': ('frame_index', [0, 2, 1, 3]),
            'mixed episodes': ('episode_index', [7, 7, 8, 7]),
            'unknown task': ('task_index', [0, 0, 2, 2]),
            'nonfinite state': ('observation.state', [[float('nan')] * 20] * 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            table = pq.read_table(episode / 'data.parquet')
            for name, (column, values) in mutations.items():
                with self.subTest(name=name):
                    changed = table.set_column(table.schema.get_field_index(column), table.schema.field(column),
                                               pa.array(values, type=table.schema.field(column).type))
                    pq.write_table(changed, episode / 'data.parquet')
                    with patch.object(script, 'copy_h264_video') as video:
                        with self.assertRaises(ValueError):
                            script.convert(root / 'raw', root / 'out', gripper_open_rad=1.)
                        video.assert_not_called()
                    self.assertFalse((root / 'out').exists())
                    self.assertFalse(list(root.glob('.out-*')))

    def test_video_frame_mismatch_and_missing_camera_roll_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'short', frames=3)
            episode = make_source(root / 'raw')
            first = episode / 'videos' / f'{next(iter(script.CAMERAS))}.mp4'
            original = first.read_bytes()
            first.write_bytes((root / 'short/episodes/episode_000007/videos' / first.name).read_bytes())
            with self.assertRaisesRegex(ValueError, 'expected 4 frames, decoded 3'):
                script.convert(root / 'raw', root / 'out', gripper_open_rad=1.)
            self.assertFalse((root / 'out').exists())
            first.write_bytes(original)
            (episode / 'videos' / f'{list(script.CAMERAS)[-1]}.mp4').unlink()
            with self.assertRaises(FileNotFoundError):
                script.convert(root / 'raw', root / 'out', gripper_open_rad=1.)
            self.assertFalse((root / 'out').exists())
            self.assertFalse(list(root.glob('.out-*')))

    def test_missing_episode_metadata_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            (root / 'raw/episodes/episode_000008').mkdir()
            with self.assertRaises(FileNotFoundError):
                script.convert(root / 'raw', root / 'out', gripper_open_rad=1.)
            self.assertFalse((root / 'out').exists())

    def test_source_layout_and_time_validation(self):
        for value in [float('nan'), float('inf'), 1e300, 1e20]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                script.seconds_to_ns(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            info = json.loads((root / 'raw/source/info.json').read_text())
            modality = json.loads((root / 'raw/source/modality.json').read_text())
            script.validate_source(info, modality)
            info['state_layout']['order'] = ['left', 'right']
            with self.assertRaisesRegex(ValueError, 'state layout'):
                script.validate_source(info, modality)
            info['state_layout']['order'] = ['right', 'left']
            modality['state']['right_eef_pose']['rotation_6d_layout'] = 'first_two_columns'
            with self.assertRaisesRegex(ValueError, 'modality'):
                script.validate_source(info, modality)

    def test_instructions_duplicates_and_final_transition(self):
        tasks = {1: 'A', 2: 'B', 3: 'C'}
        self.assertEqual(script.instructions([1, 2, 3, 1], tasks, [10, 20, 20, 40]), [
            {'start_ns': 0, 'end_ns': 10, 'text': 'A'},
            {'start_ns': 10, 'end_ns': 30, 'text': 'C'},
        ])
        with self.assertRaises(ValueError):
            script.instructions([1], tasks, [0])


if __name__ == '__main__':
    unittest.main()
