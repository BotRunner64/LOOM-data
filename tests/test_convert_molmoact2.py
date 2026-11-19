import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from support import (MODELS, assert_conversion_fails, convert, datasets, manifest, read_export, video_means,
                     write_video)
from loom_data.sources import molmoact2 as script
from loom_data.robot.kinematics import fk_poses
from loom_data.robot.urdf_model import parse_urdf
from test_convert_hifi_umi import (assert_contract_video, frame_bracket, interpolated_pose, rewrite, row_times,
                                   skip_record)

MODEL = MODELS / 'yam'
LAYOUT = 'top+left_wrist+right_wrist_640x480'
JOINTS = {'left': [.2, -.3, .5, -.6, .4, .1], 'right': [-.1, .4, -.2, .5, -.3, .6]}
OPENED = {'left': [0, .25, 1, 1], 'right': [1, .75, .5, 0]}  # the fixture's gripper values, clipped to [0,1]


def make_source(root, frames=4, fps=30):
    """One-episode downloader subset: 4 rows at `fps`, `frames` video frames per camera."""
    meta = root / 'source/meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'robot_type': 'bi_yam_follower', 'codebase_version': 'v3.0', 'fps': fps,
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
            [*JOINTS['left'], a, *JOINTS['right'], b]
            for a, b in zip([-.1, .25, 1.2, 1], [1, .75, .5, 0])],
            type=pa.list_(pa.float32(), 14)),
        # Nonzero origin: the exported clock starts at frame 0, not at the source time.
        'timestamp': pa.array(row_times(4, fps), type=pa.float32()),
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
        write_video(episode / 'videos' / f'{key}.mp4', [i * 60 for i in range(frames)], rate=fps)
    return episode


def make_full_source(root, video_frames=12, fps=25):
    """Three episodes, two frame files, and AV1 camera streams with different offsets."""
    with tempfile.TemporaryDirectory() as directory:
        subset = Path(directory)
        episode = make_source(subset, fps=fps)
        shutil.copytree(subset / 'source/meta', root / 'meta')
        info = json.loads((root / 'meta/info.json').read_text())
        info.update(fps=fps, total_episodes=3,
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
                            f'videos/{key}/from_timestamp': (number * 4 + camera_number) / fps,
                            f'videos/{key}/to_timestamp': ((number + 1) * 4 + camera_number) / fps})
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
            # Each camera's stream starts `camera_number` frames late, so the cuts differ per camera.
            write_video(root / f'videos/{key}/chunk-002/file-005.mp4',
                        [max(0, i - camera_number) * 20 for i in range(video_frames + camera_number)],
                        rate=fps, codec='libsvtav1',
                        options={'preset': '13', 'crf': '30', 'svtav1-params': 'lp=1'})
    return root


def expected_pose(model, side, joints):
    """Independent FK: remove the world mounting and apply the contract axes at the mounting origin."""
    transforms = fk_poses(model, {f'arm_{side}_joint{i + 1}': value for i, value in enumerate(joints)})
    wrist = np.linalg.inv(transforms[f'arm_{side}_base_link']) @ transforms[f'arm_{side}_link6']
    return np.r_[wrist[:3, 3], wrist[:3, 2], wrist[:3, 0], wrist[:3, 1]]


class MolmoAct2Tests(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists() and 'libsvtav1' in av.codecs_available, 'YAM assets and AV1 encoder required')
    def test_full_dataset_av1_cuts_camera_offsets_annotations_and_limit(self):
        model = parse_urdf(MODEL / 'robot.urdf')
        # Source frame of each exported frame: 30 fps rows pass through one to one. 25 fps rows are
        # resampled: frames at 0, 33.3, 66.7, 100 ms take the nearest of the rows at 0, 40, 80, 120 ms,
        # and the tie at 100 ms goes to the earlier one.
        for fps, picked, tasks in ((30, [0, 1, 2, 3], ['拿起杯子', '拿起杯子', 'Put down', 'Put down']),
                                   (25, [0, 1, 2, 2], ['拿起杯子', '拿起杯子', '拿起杯子', 'Put down'])):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = make_full_source(root / 'raw', fps=fps)
                convert('molmoact2', source, root / 'out', limit=2)
                self.assertEqual(list(datasets(root / 'out')), [LAYOUT])
                episodes = read_export(root / 'out')
                self.assertEqual([e['episode_id'] for e in episodes], ['episode_000007', 'episode_000008'])
                # Episode 7 has a valid annotation; episode 8's is null, so its per-row tasks are used.
                self.assertEqual(episodes[0]['tasks'], ['整理杯子'] * 4)
                self.assertEqual(episodes[1]['tasks'], tasks)
                seconds = row_times(4, fps).astype(np.float64)
                frames = seconds[0] + np.arange(4) / 30
                for number, episode in enumerate(episodes):
                    self.assertEqual(episode['length'], 4)
                    for side in ('left', 'right'):
                        np.testing.assert_allclose(episode[f'{side}_eef'],
                                                   [expected_pose(model, side, JOINTS[side])] * 4, atol=1e-6)
                        expected = OPENED[side] if fps == 30 else np.interp(frames, seconds, OPENED[side])
                        np.testing.assert_allclose(episode[f'{side}_gripper'], expected, atol=1e-6)
                    for camera in script.CAMERAS.values():
                        path, first = episode['videos'][camera]
                        assert_contract_video(self, path, 4)
                        np.testing.assert_allclose(video_means(path, first, 4),
                                                   (np.array(picked) + number * 4) * 20, atol=4)

    @unittest.skipUnless(MODEL.exists() and 'libsvtav1' in av.codecs_available, 'YAM assets and AV1 encoder required')
    def test_full_dataset_episode_missing_a_camera_is_skipped_as_unsupported_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_full_source(root / 'raw')
            metadata = source / 'meta/episodes/chunk-001/file-000.parquet'  # holds episode 9 only
            table = pq.read_table(metadata)
            omit = [name for name in table.column_names if name.startswith('videos/observation.images.top/')]
            pq.write_table(table.drop(omit), metadata)
            convert('molmoact2', source, root / 'out')
            self.assertEqual(list(datasets(root / 'out')), [LAYOUT])
            self.assertEqual(manifest(root / 'out'), [{'episode_id': 'episode_000007'}, {'episode_id': 'episode_000008'}])
            record = skip_record(root / 'out', 'episode_000009')
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))
            self.assertIn('left_wrist+right_wrist', record['detail'])

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_subset_episode_with_another_camera_set_is_skipped_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            metadata = json.loads((episode / 'episode.json').read_text())
            del metadata['videos']['observation.images.left']
            (episode / 'episode.json').write_text(json.dumps(metadata))
            # The only episode is left out, so there is nothing to publish; the reason is on record.
            assert_conversion_fails(self, 'No episode was exported', 'molmoact2', root / 'raw', root / 'out')
            record = skip_record(root / 'out', episode.name, published=False)
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))

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
            assert_conversion_fails(self, 'expected 4 frames, decoded 3', 'molmoact2', source, root / 'out')

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_30_fps_subset_passes_rows_through_into_a_lerobot_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            convert('molmoact2', root / 'raw', root / 'out')
            self.assertEqual(sorted(p.name for p in (root / 'out').iterdir()), [LAYOUT])
            dataset = datasets(root / 'out')[LAYOUT]
            self.assertEqual((dataset.info['codebase_version'], dataset.info['fps'], dataset.info['total_frames']),
                             ('v3.0', 30, 4))
            self.assertNotIn('action', dataset.info['features'])
            table = pq.read_table(dataset.data_path(dataset.episodes[0]))
            self.assertEqual(table.schema.field('observation.state').type, pa.list_(pa.float32(), 26))
            self.assertEqual(table['frame_index'].to_pylist(), [0, 1, 2, 3])
            # Frame k is at k / 30 s: the source's own clock (starting at 2 s) is not carried over.
            np.testing.assert_allclose(table['timestamp'].to_pylist(), np.arange(4) / 30, atol=1e-7)
            episodes = read_export(root / 'out')
            self.assertEqual(len(episodes), 1)
            exported = episodes[0]
            self.assertEqual((exported['episode_id'], exported['length']), (episode.name, 4))
            self.assertEqual(exported['cameras'], ['left_wrist', 'right_wrist', 'top'])
            # The episode annotation replaces the per-row tasks and the stale cached text.
            self.assertEqual(exported['tasks'], ['整理杯子'] * 4)
            source = pq.read_table(episode / 'data.parquet').to_pydict()
            model = parse_urdf(MODEL / 'robot.urdf')
            for side, offset in (('left', 0), ('right', 7)):
                expected = [expected_pose(model, side, row[offset:offset + 6]) for row in source['observation.state']]
                np.testing.assert_allclose(exported[f'{side}_eef'], expected, atol=1e-6)
                np.testing.assert_allclose(exported[f'{side}_gripper'], OPENED[side], atol=1e-6)
            for camera in script.CAMERAS.values():
                path, first = exported['videos'][camera]
                assert_contract_video(self, path, 4)
                np.testing.assert_allclose(video_means(path, first, 4), [0, 60, 120, 180], atol=3)
            # A finished output only accepts the configuration that produced it.
            with self.assertRaisesRegex(ValueError, 'Incompatible conversion config'):
                convert('molmoact2', root / 'raw', root / 'out', limit=1)

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_25_fps_subset_is_resampled_by_its_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw', fps=25)
            # Move both arms between rows so that interpolated poses differ from every source row.
            state = np.array(pq.read_table(episode / 'data.parquet')['observation.state'].to_pylist())
            state[:, 0] += .15 * np.arange(4)
            state[:, 4] -= .2 * np.arange(4)
            state[:, 8] += .1 * np.arange(4)
            state[:, 12] += .25 * np.arange(4)
            rewrite(episode / 'data.parquet', **{'observation.state': state.tolist()})
            (root / 'raw/source/meta/tasks_annotated.parquet').unlink()
            convert('molmoact2', root / 'raw', root / 'out')
            exported = read_export(root / 'out')[0]
            # Rows at 0, 40, 80, 120 ms cover the frames at 0, 33.3, 66.7 and 100 ms; nothing is extrapolated.
            self.assertEqual(exported['length'], 4)
            source = np.array(pq.read_table(episode / 'data.parquet')['observation.state'].to_pylist())
            seconds = row_times(4, 25).astype(np.float64)
            model = parse_urdf(MODEL / 'robot.urdf')
            for side, offset in (('left', 0), ('right', 7)):
                rows = [expected_pose(model, side, row[offset:offset + 6]) for row in source]
                np.testing.assert_allclose(exported[f'{side}_eef'][0], rows[0], atol=1e-6)
                for frame in (1, 2, 3):
                    low, weight = frame_bracket(seconds, frame)
                    self.assertEqual(low, frame - 1)
                    interpolated = interpolated_pose(rows[low], rows[low + 1], weight)
                    np.testing.assert_allclose(exported[f'{side}_eef'][frame], interpolated, atol=1e-6)
                    # The interpolated pose is a new value, not a copy of a neighbouring row.
                    self.assertGreater(np.abs(interpolated - rows[low]).max(), 1e-3)
                    self.assertGreater(np.abs(interpolated - rows[low + 1]).max(), 1e-3)
                    np.testing.assert_allclose(exported[f'{side}_gripper'][frame],
                                               OPENED[side][low] * (1 - weight) + OPENED[side][low + 1] * weight,
                                               atol=1e-6)
            # The task changes at the row at 80 ms, so the frame at 66.7 ms still has the first one.
            self.assertEqual(exported['tasks'], ['拿起杯子', '拿起杯子', '拿起杯子', 'Put down'])
            for camera in script.CAMERAS.values():
                path, first = exported['videos'][camera]
                assert_contract_video(self, path, 4)
                # Nearest source frame; the frame at 100 ms is midway between rows 2 and 3 and takes the earlier.
                np.testing.assert_allclose(video_means(path, first, 4), [0, 60, 120, 120], atol=3)

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_declared_rate_decides_and_the_median_row_interval_stands_in_for_it(self):
        jitter = np.array([2., 2.031, 2.079, 2.141], dtype=np.float32)
        cases = [(30, jitter, [0, 60, 120, 180]),            # declared 30 fps: timestamps are not consulted
                 (None, row_times(4, 30), [0, 60, 120, 180]),  # undeclared, rows 1/30 s apart
                 (None, row_times(4, 25), [0, 60, 120, 120])]  # undeclared, rows 1/25 s apart: resampled
        for declared, timestamps, means in cases:
            with self.subTest(declared=declared, timestamps=timestamps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = make_source(root / 'raw')
                path = root / 'raw/source/meta/info.json'
                info = json.loads(path.read_text())
                if declared is None:
                    del info['fps']
                path.write_text(json.dumps(info))
                rewrite(episode / 'data.parquet', timestamp=timestamps)
                convert('molmoact2', root / 'raw', root / 'out')
                exported = read_export(root / 'out')[0]
                self.assertEqual(exported['length'], 4)
                if means[-1] == 180:
                    np.testing.assert_allclose(exported['left_gripper'], OPENED['left'], atol=1e-6)
                else:
                    self.assertAlmostEqual(exported['left_gripper'][1], .25 * (1 / 30) / .04, places=5)
                video, first = exported['videos']['top']
                np.testing.assert_allclose(video_means(video, first, 4), means, atol=3)

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_gap_in_resampled_rows_is_recorded_as_a_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw', fps=25)
            # No state between 40 ms and 300 ms: more than the 100 ms a frame may be interpolated across.
            rewrite(episode / 'data.parquet', timestamp=np.array([2., 2.04, 2.30, 2.34], dtype=np.float32))
            assert_conversion_fails(self, 'No episode was exported', 'molmoact2', root / 'raw', root / 'out')
            record = skip_record(root / 'out', episode.name, published=False)
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'sampling_gap'))

    def test_annotation_fallback_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root)
            meta = root / 'source/meta'
            for annotation in (None, '', '  '):
                pq.write_table(pa.table({'episode_index': [7], 'task': [annotation]}), meta / 'tasks_annotated.parquet')
                tasks, annotated = script.load_tasks(meta)
                with patch.object(script, 'YamFK') as fk:
                    times, _, instruction, _ = script.read_episode(episode, tasks, annotated, fk)
                self.assertEqual(instruction, [
                    {'start_ns': 0, 'end_ns': times[2] - times[0], 'text': '拿起杯子'},
                    {'start_ns': times[2] - times[0], 'end_ns': times[-1] - times[0] + 1, 'text': 'Put down'}])
            (meta / 'tasks_annotated.parquet').unlink()
            self.assertEqual(script.load_tasks(meta)[1], {})
            pq.write_table(pa.table({'episode_index': [7, 7], 'task': ['A', 'B']}), meta / 'tasks_annotated.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                script.load_tasks(meta)

    def test_bad_rows_fail_before_decoding_video(self):
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
                with patch.object(script, 'YamFK'), patch.object(script, 'transcode_video') as video:
                    assert_conversion_fails(self, '', 'molmoact2', root / 'raw', root / 'out')
                    video.assert_not_called()

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_failed_video_or_missing_episode_does_not_publish(self):
        for missing_episode in (False, True):
            with self.subTest(missing_episode=missing_episode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_source(root / 'raw', frames=4 if missing_episode else 3)
                if missing_episode:
                    (root / 'raw/episodes/episode_000008').mkdir()
                assert_conversion_fails(self, '' if missing_episode else 'Decoded 3 frames; expected 4',
                                        'molmoact2', root / 'raw', root / 'out')

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
                assert_conversion_fails(self, 'Limits must be positive', 'molmoact2', root / 'raw', root / 'out', limit=limit)


if __name__ == '__main__':
    unittest.main()
