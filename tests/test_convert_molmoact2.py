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

from scripts.convert import molmoact2 as script
from scripts.robot.kinematics import fk_poses
from scripts.robot.urdf_model import parse_urdf

MODEL = script.ROOT / 'assets/robot_models/yam'


def make_source(root, frames=4):
    meta = root / 'source/meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'robot_type': 'bi_yam_follower', 'codebase_version': 'v3.0',
        'features': {'observation.state': {'shape': [14], 'names': script.STATE_NAMES},
                     **{key: {'dtype': 'video'} for key in script.CAMERAS}},
    }))
    pq.write_table(pa.table({'task_index': [0, 3], '__index_level_0__': ['拿起杯子', 'Put down']}),
                   meta / 'tasks.parquet')
    pq.write_table(pa.table({'episode_index': [7], 'task': ['整理杯子']}),
                   meta / 'tasks_annotated.parquet')
    episode = root / 'episodes/episode_000007'
    (episode / 'videos').mkdir(parents=True)
    pq.write_table(pa.table({
        'observation.state': pa.array([
            [.2, -.3, .5, -.6, .4, .1, a, -.1, .4, -.2, .5, -.3, .6, b]
            for a, b in zip([-.1, .25, 1.2, 1], [1, .75, .5, 0])],
            type=pa.list_(pa.float32(), 14)),
        # Nonzero origin, irregular intervals; container FPS must not determine time.
        'timestamp': pa.array([2., 2.031, 2.079, 2.141], type=pa.float32()),
        'frame_index': [0, 1, 2, 3], 'episode_index': [7] * 4,
        'index': [100, 101, 102, 103], 'task_index': [0, 0, 3, 3],
        'action': [[float('nan')] * 14] * 4,
    }), episode / 'data.parquet')
    (episode / 'episode.json').write_text(json.dumps({
        'episode_index': 7, 'length': 4, 'dataset_from_index': 100, 'dataset_to_index': 104,
        'tasks': ['拿起杯子', 'Put down'], 'task': 'stale cached annotation',
        'videos': {key: {'from_timestamp': i * 100, 'to_timestamp': i * 100 + 1}
                   for i, key in enumerate(script.CAMERAS)},
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


def make_full_source(root, video_frames=12):
    """Three episodes, two frame files, and AV1 camera streams with different offsets."""
    with tempfile.TemporaryDirectory() as directory:
        subset = Path(directory)
        episode = make_source(subset)
        shutil.copytree(subset / 'source/meta', root / 'meta')
        info = json.loads((root / 'meta/info.json').read_text())
        info.update(fps=25, total_episodes=3,
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                    video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        (root / 'meta/info.json').write_text(json.dumps(info))
        table = pq.read_table(episode / 'data.parquet')
        tables, metadata = [], []
        for number, index in enumerate((7, 8, 9)):
            changed = table.set_column(table.schema.get_field_index('episode_index'), 'episode_index',
                                       pa.array([index] * 4))
            changed = changed.set_column(changed.schema.get_field_index('index'), 'index',
                                         pa.array(list(range(100 + number * 4, 104 + number * 4))))
            tables.append(changed)
            row = {'episode_index': index, 'length': 4, 'dataset_from_index': 100 + number * 4,
                   'dataset_to_index': 104 + number * 4, 'tasks': ['拿起杯子', 'Put down'],
                   'data/chunk_index': 2, 'data/file_index': 3 if number < 2 else 4}
            for camera_number, key in enumerate(script.CAMERAS):
                row.update({f'videos/{key}/chunk_index': 2, f'videos/{key}/file_index': 5,
                            f'videos/{key}/from_timestamp': (number * 4 + camera_number) / 25,
                            f'videos/{key}/to_timestamp': ((number + 1) * 4 + camera_number) / 25})
            metadata.append(row)
        for file_index, data in ((3, pa.concat_tables(tables[:2])), (4, tables[2])):
            path = root / f'data/chunk-002/file-{file_index:03d}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(data, path, row_group_size=5)
        for number, rows in enumerate((metadata[:2], metadata[2:])):
            path = root / f'meta/episodes/chunk-{number:03d}/file-000.parquet'
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
        pq.write_table(pa.table({'episode_index': [7, 8, 9], 'task': ['整理杯子', None, ' ']}),
                       root / 'meta/tasks_annotated.parquet')
        for camera_number, key in enumerate(script.CAMERAS):
            path = root / f'videos/{key}/chunk-002/file-005.mp4'
            path.parent.mkdir(parents=True)
            with av.open(str(path), 'w') as output:
                stream = output.add_stream('libsvtav1', rate=25)
                stream.width = stream.height = 64
                stream.pix_fmt = 'yuv420p'
                stream.thread_count = 1
                stream.options = {'preset': '13', 'crf': '30', 'svtav1-params': 'lp=1'}
                for i in range(video_frames + camera_number):
                    value = max(0, i - camera_number) * 20
                    frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), value, np.uint8), format='rgb24')
                    frame.pts, frame.time_base = i, Fraction(1, 25)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
    return root


class MolmoAct2Tests(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists() and 'libsvtav1' in av.codecs_available, 'YAM assets and AV1 encoder required')
    def test_full_dataset_av1_cuts_camera_offsets_annotations_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_full_source(root / 'raw')
            script.convert(source, root / 'out', limit=2)
            records = [json.loads(line) for line in (root / 'out/episodes.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode_id'] for r in records], ['episode_000007', 'episode_000008'])
            self.assertEqual(records[0]['instructions'][0]['text'], '整理杯子')
            self.assertEqual([r['text'] for r in records[1]['instructions']], ['拿起杯子', 'Put down'])
            for number, record in enumerate(records):
                exported = root / 'out/episodes' / record['episode_id']
                times = pq.read_table(exported / 'state/left_eef.parquet')['timestamp_ns'].to_pylist()
                expected_times = [script.seconds_to_ns(t) - 2_000_000_000 for t in
                                  np.array([2., 2.031, 2.079, 2.141], dtype=np.float32)]
                self.assertEqual(times, expected_times)
                for camera in script.CAMERAS.values():
                    path = exported / 'rgb' / f'{camera}.mp4'
                    with av.open(str(path)) as video:
                        self.assertEqual(video.streams.video[0].codec_context.name, 'h264')
                        self.assertEqual(len(video.streams.audio), 0)
                        frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 4)
                    np.testing.assert_allclose([f.to_ndarray(format='rgb24').mean() for f in frames],
                                               np.arange(number * 4, number * 4 + 4) * 20, atol=4)
                    self.assertEqual(pq.read_table(path.with_suffix('.parquet'))['timestamp_ns'].to_pylist(), times)

    @unittest.skipUnless('libsvtav1' in av.codecs_available, 'AV1 encoder required')
    def test_full_dataset_reads_shared_data_once_and_switches_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_full_source(Path(directory) / 'raw')
            reader = script.DatasetReader(source)
            with patch.object(script.pq, 'read_table', wraps=pq.read_table) as read:
                for index in (7, 8, 9):
                    rows = reader.read_rows(index)
                    self.assertEqual(rows['episode_index'], [index] * 4)
                self.assertEqual(read.call_count, 2)
                self.assertEqual(read.call_args.kwargs['columns'], list(script.COLUMNS))

    @unittest.skipUnless(MODEL.exists() and 'libsvtav1' in av.codecs_available, 'YAM assets and AV1 encoder required')
    def test_short_av1_segment_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_full_source(root / 'raw', video_frames=11)
            with self.assertRaisesRegex(ValueError, 'expected 4 frames, decoded 3'):
                script.convert(source, root / 'out')
            self.assertFalse((root / 'out').exists())
            self.assertFalse(list(root.glob('.out-*')))

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_full_export_schema_state_time_annotations_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            script.convert(root / 'raw', root / 'out')
            records = [json.loads(line) for line in (root / 'out/episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            exported = root / 'out/episodes' / episode.name
            source = pq.read_table(episode / 'data.parquet').to_pydict()
            times = [script.seconds_to_ns(t) - 2_000_000_000 for t in source['timestamp']]
            self.assertEqual(record, {'episode_id': episode.name, 'cameras': ['left_wrist', 'right_wrist', 'top'],
                                     'instructions': [{'start_ns': 0, 'end_ns': times[-1], 'text': '整理杯子'}]})
            model = parse_urdf(MODEL / 'robot.urdf')
            for side, offset, grips in [('left', 0, [0, .25, 1, 1]), ('right', 7, [1, .75, .5, 0])]:
                poses = pq.read_table(exported / 'state' / f'{side}_eef.parquet')
                self.assertEqual(poses['timestamp_ns'].to_pylist(), times)
                self.assertEqual(poses.schema.field('pose').type, pa.list_(pa.float64(), 12))
                self.assertFalse(poses.schema.field('pose').nullable)
                # Independent expected FK: remove world mounting and apply TCP/axes at wrist.
                q = source['observation.state'][0][offset:offset + 6]
                transforms = fk_poses(model, {f'arm_{side}_joint{i+1}': value for i, value in enumerate(q)})
                wrist = np.linalg.inv(transforms[f'arm_{side}_base_link']) @ transforms[f'arm_{side}_link6']
                expected = np.r_[wrist[:3, 3] + .1347 * wrist[:3, 2], wrist[:3, 2], wrist[:3, 0], wrist[:3, 1]]
                np.testing.assert_allclose(poses['pose'][0].as_py(), expected, atol=1e-9)
                gripper = pq.read_table(exported / 'state' / f'{side}_gripper.parquet')
                self.assertEqual(gripper['timestamp_ns'].to_pylist(), times)
                self.assertEqual(gripper['openness'].type, pa.float32())
                self.assertEqual(gripper['openness'].to_pylist(), grips)
            for key, camera in script.CAMERAS.items():
                video = exported / 'rgb' / f'{camera}.mp4'
                self.assertEqual(video.read_bytes(), (episode / 'videos' / f'{key}.mp4').read_bytes())
                table = pq.read_table(exported / 'rgb' / f'{camera}.parquet')
                self.assertEqual(table['timestamp_ns'].to_pylist(), times)
                self.assertEqual(table['frame_index'].to_pylist(), list(range(4)))
                with av.open(str(video)) as reader:
                    self.assertEqual(reader.streams.video[0].codec_context.name, 'h264')
                    self.assertEqual(len(reader.streams.audio), 0)
                    for i, frame in enumerate(reader.decode(video=0)):
                        self.assertAlmostEqual(frame.to_ndarray(format='rgb24').mean(), i * 60, delta=3)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                script.convert(root / 'raw', root / 'out')

    def test_annotation_fallback_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root)
            meta = root / 'source/meta'
            for annotation in (None, '', '  '):
                pq.write_table(pa.table({'episode_index': [7], 'task': [annotation]}), meta / 'tasks_annotated.parquet')
                tasks, annotated = script.load_tasks(meta)
                with patch.object(script, 'YamFK') as fk:
                    times, _, instruction = script.read_episode(episode, tasks, annotated, fk)
                self.assertEqual(instruction, [
                    {'start_ns': 0, 'end_ns': times[2] - times[0], 'text': '拿起杯子'},
                    {'start_ns': times[2] - times[0], 'end_ns': times[-1] - times[0], 'text': 'Put down'}])
            (meta / 'tasks_annotated.parquet').unlink()
            self.assertEqual(script.load_tasks(meta)[1], {})
            pq.write_table(pa.table({'episode_index': [7, 7], 'task': ['A', 'B']}), meta / 'tasks_annotated.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                script.load_tasks(meta)

    def test_bad_rows_fail_before_copying_video(self):
        mutations = [
            ('timestamp', [2., 1., 3., 4.]), ('timestamp', [2.] * 4),
            ('timestamp', [None, 2., 3., 4.]), ('timestamp', [0., float('inf'), 2., 3.]),
            ('frame_index', [0, 2, 1, 3]), ('index', [100, 101, 101, 103]),
            ('episode_index', [7, 7, 8, 7]), ('task_index', [0, 0, 99, 3]),
            ('observation.state', [[0.] * 13] * 4),
            ('observation.state', [[0.] * 6 + [float('nan')] + [0.] * 7] * 4),
            ('observation.state', [[float('inf')] + [0.] * 13] * 4),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = make_source(root / 'raw')
                rows = pq.read_table(episode / 'data.parquet').to_pydict()
                rows[key] = value
                pq.write_table(pa.table(rows), episode / 'data.parquet')
                with patch.object(script, 'YamFK'), patch.object(script, 'copy_h264_video') as video:
                    with self.assertRaises(ValueError):
                        script.convert(root / 'raw', root / 'out')
                    video.assert_not_called()
                self.assertFalse((root / 'out').exists())
                self.assertFalse(list(root.glob('.out-*')))

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_failed_video_or_missing_episode_does_not_publish(self):
        for missing_episode in (False, True):
            with self.subTest(missing_episode=missing_episode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_source(root / 'raw', frames=4 if missing_episode else 3)
                if missing_episode:
                    (root / 'raw/episodes/episode_000008').mkdir()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    script.convert(root / 'raw', root / 'out')
                self.assertFalse((root / 'out').exists())
                self.assertFalse(list(root.glob('.out-*')))

    def test_wrong_layout_and_invalid_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            path = root / 'raw/source/meta/info.json'
            info = json.loads(path.read_text())
            script.validate_source(info)
            info['features']['observation.state']['names'].reverse()
            with self.assertRaisesRegex(ValueError, 'layout'):
                script.validate_source(info)
            for limit in (0, -1, True):
                with self.assertRaisesRegex(ValueError, 'limit'):
                    script.convert(root / 'raw', root / 'out', limit=limit)


if __name__ == '__main__':
    unittest.main()
