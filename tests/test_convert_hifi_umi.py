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

from support import assert_conversion_fails, convert, datasets, manifest, read_export, video_means, write_video
from loom_data.convert import job
from loom_data.sources import hifi_umi as script

LAYOUT = 'top+left_wrist+right_wrist_640x480'
ZERO_OFFSETS = {'left_eef_offset': [0., 0., 0.], 'right_eef_offset': [0., 0., 0.]}
TASKS = ['拿起杯子', 'Put down']


def row_times(count, fps):
    """Row timestamps as the fixtures store them: float32 seconds, away from zero."""
    return (2 + np.arange(count) / fps).astype(np.float32)


def make_source(root, frames=4, fps=30):
    """One-episode downloader subset: 4 rows at `fps`, `frames` video frames per camera."""
    metadata_dir = root / 'source'
    metadata_dir.mkdir(parents=True)
    info = {
        'robot_type': 'simpleai_umi_4.0', 'codebase_version': 'v3.0', 'fps': fps,
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
    pq.write_table(pa.table({'task_index': [0, 3], 'task': TASKS}),
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
        # All streams start away from zero.
        'timestamp': pa.array(row_times(4, fps), type=pa.float32()),
        'frame_index': [0, 1, 2, 3], 'episode_index': [7] * 4,
        'index': [100, 101, 102, 103], 'task_index': [0, 0, 3, 3],
        'valid.frame': [True] * 4,
        # Deliberately unusable actions: the converter must never read them.
        'action': [[float('nan')] * 20] * 4,
    })
    pq.write_table(table, episode / 'data.parquet')
    (episode / 'episode.json').write_text(json.dumps({
        'episode_index': 7, 'length': 4, 'dataset_from_index': 100, 'dataset_to_index': 104,
        'tasks': TASKS, 'videos': {key: {} for key in script.CAMERAS},
    }))
    for key in script.CAMERAS:
        write_video(episode / 'videos' / f'{key}.mp4', [i * 60 for i in range(frames)], rate=fps)
    return episode


def make_shard(root, chunk=0, part=0, video_frames=8, fps=25):
    """Two episodes sharing frame/video files, with a cut between keyframes."""
    shard = root / f'chunk-{chunk:04d}' / f'part-{part:04d}'
    with tempfile.TemporaryDirectory() as directory:
        subset = Path(directory)
        episode = make_source(subset, fps=fps)
        shutil.copytree(subset / 'source', shard / 'meta')
        info = json.loads((shard / 'meta/info.json').read_text())
        info.update(fps=float(fps), total_episodes=2,
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
                   'dataset_to_index': 104 + number * 4, 'tasks': TASKS,
                   'data/chunk_index': 2, 'data/file_index': 3}
            for key in script.CAMERAS:
                row.update({f'videos/{key}/chunk_index': 2, f'videos/{key}/file_index': 3,
                            f'videos/{key}/from_timestamp': number * 4 / fps,
                            f'videos/{key}/to_timestamp': (number + 1) * 4 / fps})
            metadata = shard / f'meta/episodes/chunk-{number:03d}/file-000.parquet'
            metadata.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([row]), metadata)
        original = None
        for key in script.CAMERAS:
            video = shard / f'videos/{key}/chunk-002/file-003.mp4'
            if original is not None:
                video.parent.mkdir(parents=True)
                shutil.copyfile(original, video)
                continue
            original = write_video(video, [i * 28 for i in range(video_frames)], rate=fps,
                                   options={'g': '250', 'sc_threshold': '0'})
    return shard


def rotation(axis, degrees):
    """Rotation matrix about a coordinate axis (0, 1 or 2)."""
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    result = np.eye(3)
    result[i, i] = result[j, j] = c
    result[i, j], result[j, i] = -s, s
    return result


def interpolated_pose(a, b, weight):
    """Expected contract pose between poses `a` and `b`: position lerp, rotation along the geodesic.

    Written with the axis-angle form R_a exp(w log(R_a^T R_b)), independently of the converter's quaternions.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    first, second = a[3:].reshape(3, 3).T, b[3:].reshape(3, 3).T
    relative = first.T @ second
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    step = np.eye(3)
    if angle > 1e-9:
        axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                         relative[1, 0] - relative[0, 1]]) / (2 * np.sin(angle))
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        step = np.eye(3) + np.sin(weight * angle) * skew + (1 - np.cos(weight * angle)) * skew @ skew
    return np.r_[a[:3] * (1 - weight) + b[:3] * weight, (first @ step).T.reshape(-1)]


def frame_bracket(seconds, frame):
    """Rows around 30 fps frame `frame` of a table sampled at `seconds`, and the later row's weight."""
    seconds = np.asarray(seconds, dtype=np.float64)
    time = seconds[0] + frame / 30
    low = int(np.searchsorted(seconds, time, side='right')) - 1
    low = min(max(low, 0), len(seconds) - 2)
    return low, (time - seconds[low]) / (seconds[low + 1] - seconds[low])


def skip_record(output, episode_id, published=True):
    return job.Store.load(output).state(episode_id, published=published)


def assert_contract_video(case, path, length, size=(640, 480)):
    with av.open(str(path)) as reader:
        case.assertEqual(len(reader.streams), 1)
        video = reader.streams.video[0]
        case.assertEqual((video.codec_context.name, video.codec_context.pix_fmt), ('h264', 'yuv420p'))
        case.assertEqual((video.width, video.height), size)
        case.assertEqual(video.average_rate, 30)
        times = [frame.time for frame in reader.decode(video=0)]
    np.testing.assert_allclose(times, np.arange(length) / 30, atol=1e-4)


def rewrite(path, **columns):
    """Replace columns of a fixture Parquet file, keeping their types."""
    table = pq.read_table(path)
    for name, values in columns.items():
        field = table.schema.field(name)
        table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
    pq.write_table(table, path)


class HifiUmiTests(unittest.TestCase):
    def test_subset_selects_main_views_and_skips_a_camera_set_it_does_not_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            metadata = episode / 'episode.json'
            row = json.loads(metadata.read_text())
            # Auxiliary views are ignored; the three main views alone make the layout.
            row['videos']['observation.images.right_hand_down'] = {}
            metadata.write_text(json.dumps(row))
            convert('hifi_umi', root / 'raw', root / 'out')
            self.assertEqual(list(datasets(root / 'out')), [LAYOUT])
            self.assertEqual(read_export(root / 'out')[0]['cameras'], ['left_wrist', 'right_wrist', 'top'])

            # One wrist only: not an exported layout, so the episode is left out with a reason.
            row['videos'] = {'observation.images.right_hand_up': {},
                             'observation.images.right_hand_down': {}}
            metadata.write_text(json.dumps(row))
            assert_conversion_fails(self, 'No episode was exported', 'hifi_umi', root / 'raw', root / 'one')
            record = skip_record(root / 'one', episode.name, published=False)
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))

            del row['videos']['observation.images.right_hand_up']
            metadata.write_text(json.dumps(row))
            assert_conversion_fails(self, 'No supported RGB cameras', 'hifi_umi', root / 'raw', root / 'empty')

    def test_shard_episode_missing_a_camera_is_skipped_as_unsupported_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = make_shard(root / 'raw')
            metadata = shard / 'meta/episodes/chunk-001/file-000.parquet'
            table = pq.read_table(metadata)
            omit = [name for name in table.column_names if name.startswith('videos/observation.images.head_main/')]
            pq.write_table(table.drop(omit), metadata)
            convert('hifi_umi', root / 'raw', root / 'out')
            self.assertEqual(list(datasets(root / 'out')), [LAYOUT])
            self.assertEqual(manifest(root / 'out'), [{'episode_id': 'chunk_0000_part_0000_episode_000007'}])
            record = skip_record(root / 'out', 'chunk_0000_part_0000_episode_000008')
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))
            self.assertIn('left_wrist+right_wrist', record['detail'])
            self.assertEqual(skip_record(root / 'out', 'chunk_0000_part_0000_episode_000007')['outcome'], 'exported')

    def test_full_shards_cut_nonkeyframe_boundaries_and_keep_unique_ids(self):
        # Source frame of each exported frame: 30 fps rows pass through one to one. 25 fps rows are
        # resampled: frames at 0, 33.3, 66.7, 100 ms take the nearest of the rows at 0, 40, 80, 120 ms,
        # and the tie at 100 ms goes to the earlier one.
        for fps, picked, tasks in ((30, [0, 1, 2, 3], [0, 0, 1, 1]), (25, [0, 1, 2, 2], [0, 0, 0, 1])):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for chunk, part in ((0, 0), (0, 1)):
                    make_shard(root / 'raw', chunk, part, fps=fps)
                convert('hifi_umi', root / 'raw', root / 'out', **ZERO_OFFSETS)
                self.assertEqual(list(datasets(root / 'out')), [LAYOUT])
                self.assertEqual([r['episode_id'] for r in manifest(root / 'out')], [
                    'chunk_0000_part_0000_episode_000007', 'chunk_0000_part_0000_episode_000008',
                    'chunk_0000_part_0001_episode_000007', 'chunk_0000_part_0001_episode_000008'])
                seconds = row_times(4, fps).astype(np.float64)
                frames = seconds[0] + np.arange(4) / 30
                for number, episode in enumerate(read_export(root / 'out')):
                    self.assertEqual(episode['length'], 4)
                    self.assertEqual(episode['tasks'], [TASKS[i] for i in tasks])
                    np.testing.assert_allclose(episode['right_eef'], [[1, 2, 3, 0, 1, 0, -1, 0, 0, 0, 0, 1]] * 4,
                                               atol=1e-6)
                    np.testing.assert_allclose(episode['left_eef'], [[4, 5, 6, 1, 0, 0, 0, 0, 1, 0, -1, 0]] * 4,
                                               atol=1e-6)
                    for side, opened in (('right', [0, .5, 1, 1]), ('left', [1, .75, .25, 0])):
                        expected = opened if fps == 30 else np.interp(frames, seconds, opened)
                        np.testing.assert_allclose(episode[f'{side}_gripper'], expected, atol=1e-6)
                    for camera in script.CAMERAS.values():
                        path, first = episode['videos'][camera]
                        assert_contract_video(self, path, 4)
                        np.testing.assert_allclose(video_means(path, first, 4),
                                                   (np.array(picked) + (number % 2) * 4) * 28, atol=4)

    def test_shard_reader_reuses_projected_table_and_standalone_global_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = make_shard(root / 'raw')
            make_shard(root / 'raw', 1)
            reader = script.ShardReader(shard)
            with patch.object(script.pq, 'read_table', wraps=pq.read_table) as read:
                for index in (7, 8):
                    record = reader.convert_episode(index, root / str(index), 0., script.DEFAULT_GRIPPER_OPEN_RAD,
                                                    eef_offsets={'left': [0., 0., 0.], 'right': [0., 0., 0.]})
                    self.assertEqual((record['length'], record['cameras']), (4, ['left_wrist', 'right_wrist', 'top']))
                self.assertEqual(read.call_count, 1)
                self.assertEqual(read.call_args.kwargs['columns'], list(script.COLUMNS))
            convert('hifi_umi', root / 'raw', root / 'out', limit=3, **ZERO_OFFSETS)
            self.assertEqual(len(manifest(root / 'out')), 3)

    def test_short_shard_video_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_shard(root / 'raw', video_frames=7)
            assert_conversion_fails(self, 'expected 4 frames, decoded 3', 'hifi_umi', root / 'raw', root / 'out',
                                    **ZERO_OFFSETS)

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
        from loom_data.robot.kinematics import rpy_matrix
        rotation = rpy_matrix([.2, -.4, .7])
        source = np.r_[1, 2, 3, rotation[:2].ravel()]
        pose = script.poses_from_rows([source], eef_offset=[0., 0., 0.])[0]
        np.testing.assert_allclose(pose, np.r_[1, 2, 3, rotation.T.reshape(-1)], atol=1e-12)
        reconstructed = pose[3:12].reshape(3, 3).T
        np.testing.assert_allclose(reconstructed.T @ [1, 0, 0], rotation.T @ [1, 0, 0])
        for bad in ([0] * 9, [1, 2, 3, 1, 0, 0, 1, 0, 0], [float('nan')] * 9):
            with self.assertRaises(ValueError):
                script.poses_from_rows([bad], eef_offset=[0., 0., 0.])

    def test_gripper_uses_fixed_endpoints_and_rejects_nonfinite(self):
        np.testing.assert_allclose(script.openness(np.deg2rad([0, 17.5, 35, 70])), [0, .5, 1, 1])
        np.testing.assert_allclose(script.openness([-.2, .1, .4, .7, 1], .1, .7), [0, 0, .5, 1, 1])
        for closed, opened in [(1, 1), (1, 0), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                script.openness([.2], closed, opened)
        for angle in [float('nan'), float('inf'), float('-inf')]:
            with self.assertRaises(ValueError):
                script.openness([angle], 0, 1)

    def test_30_fps_subset_passes_rows_through_into_a_lerobot_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            output = root / 'out'
            convert('hifi_umi', root / 'raw', output, **ZERO_OFFSETS)
            self.assertEqual(sorted(p.name for p in output.iterdir()), [LAYOUT])
            dataset = datasets(output)[LAYOUT]
            self.assertEqual((dataset.info['codebase_version'], dataset.info['fps'], dataset.info['total_frames']),
                             ('v3.0', 30, 4))
            self.assertNotIn('action', dataset.info['features'])
            table = pq.read_table(dataset.data_path(dataset.episodes[0]))
            self.assertEqual(table.schema.field('observation.state').type, pa.list_(pa.float32(), 26))
            self.assertEqual(table['frame_index'].to_pylist(), [0, 1, 2, 3])
            # Frame k is at k / 30 s: the source's own clock (starting at 2 s) is not carried over.
            np.testing.assert_allclose(table['timestamp'].to_pylist(), np.arange(4) / 30, atol=1e-7)
            episodes = read_export(output)
            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertEqual((episode['episode_id'], episode['length']), ('episode_000007', 4))
            self.assertEqual(episode['cameras'], ['left_wrist', 'right_wrist', 'top'])
            # Each row keeps its own instruction.
            self.assertEqual(episode['tasks'], ['拿起杯子', '拿起杯子', 'Put down', 'Put down'])
            np.testing.assert_allclose(episode['right_eef'], [[1, 2, 3, 0, 1, 0, -1, 0, 0, 0, 0, 1]] * 4, atol=1e-6)
            np.testing.assert_allclose(episode['left_eef'], [[4, 5, 6, 1, 0, 0, 0, 0, 1, 0, -1, 0]] * 4, atol=1e-6)
            np.testing.assert_allclose(episode['right_gripper'], [0, .5, 1, 1], atol=1e-6)
            np.testing.assert_allclose(episode['left_gripper'], [1, .75, .25, 0], atol=1e-6)
            for camera in script.CAMERAS.values():
                path, first = episode['videos'][camera]
                assert_contract_video(self, path, 4)
                np.testing.assert_allclose(video_means(path, first, 4), [0, 60, 120, 180], atol=3)
            # A finished output only accepts the configuration that produced it.
            with self.assertRaisesRegex(ValueError, 'Incompatible conversion config'):
                convert('hifi_umi', root / 'raw', output, gripper_open_rad=1., **ZERO_OFFSETS)

    def test_25_fps_subset_is_resampled_by_its_timestamps(self):
        offsets = {'left': [-.11, .02, .03], 'right': [-.15, -.01, .04]}
        positions = {'right': [[1 + .1 * k, 2, 3 - .05 * k] for k in range(4)], 'left': [[4, 5 + .2 * k, 6] for k in range(4)]}
        rotations = {'right': [rotation(2, 20 * k) for k in range(4)], 'left': [rotation(0, 90 + 10 * k) for k in range(4)]}
        opened = {'right': [0, .5, 1, 1], 'left': [1, .75, .25, 0]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw', fps=25)
            state = [[*positions['right'][k], *rotations['right'][k][:2].ravel(),
                      opened['right'][k] * script.DEFAULT_GRIPPER_OPEN_RAD,
                      *positions['left'][k], *rotations['left'][k][:2].ravel(),
                      opened['left'][k] * script.DEFAULT_GRIPPER_OPEN_RAD] for k in range(4)]
            rewrite(episode / 'data.parquet', **{'observation.state': state})
            convert('hifi_umi', root / 'raw', root / 'out',
                    left_eef_offset=offsets['left'], right_eef_offset=offsets['right'])
            exported = read_export(root / 'out')[0]
            # Rows at 0, 40, 80, 120 ms cover the frames at 0, 33.3, 66.7 and 100 ms; nothing is extrapolated.
            self.assertEqual(exported['length'], 4)
            seconds = row_times(4, 25).astype(np.float64)
            for side in ('left', 'right'):
                # The mounting offset is applied to each source row; frames interpolate the mounted poses.
                rows = [np.r_[np.array(p) + r @ offsets[side], r.T.reshape(-1)]
                        for p, r in zip(positions[side], rotations[side])]
                np.testing.assert_allclose(exported[f'{side}_eef'][0], rows[0], atol=1e-6)
                for frame in (1, 2, 3):
                    low, weight = frame_bracket(seconds, frame)
                    self.assertEqual(low, frame - 1)
                    np.testing.assert_allclose(exported[f'{side}_eef'][frame],
                                               interpolated_pose(rows[low], rows[low + 1], weight), atol=1e-6)
                    np.testing.assert_allclose(exported[f'{side}_gripper'][frame],
                                               opened[side][low] * (1 - weight) + opened[side][low + 1] * weight,
                                               atol=1e-6)
            # The instruction changes at the row at 80 ms, so the frame at 66.7 ms still has the first one.
            self.assertEqual(exported['tasks'], ['拿起杯子', '拿起杯子', '拿起杯子', 'Put down'])
            for camera in script.CAMERAS.values():
                path, first = exported['videos'][camera]
                assert_contract_video(self, path, 4)
                # Nearest source frame; the frame at 100 ms is midway between rows 2 and 3 and takes the earlier.
                np.testing.assert_allclose(video_means(path, first, 4), [0, 60, 120, 120], atol=3)

    def test_declared_rate_decides_and_the_median_row_interval_stands_in_for_it(self):
        jitter = np.array([2., 2.031, 2.079, 2.141], dtype=np.float32)
        cases = [(30, jitter, [0, 60, 120, 180]),            # declared 30 fps: timestamps are not consulted
                 (None, row_times(4, 30), [0, 60, 120, 180]),  # undeclared, rows 1/30 s apart
                 (None, row_times(4, 25), [0, 60, 120, 120])]  # undeclared, rows 1/25 s apart: resampled
        for declared, timestamps, means in cases:
            with self.subTest(declared=declared, timestamps=timestamps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = make_source(root / 'raw')
                info = json.loads((root / 'raw/source/info.json').read_text())
                if declared is None:
                    del info['fps']
                (root / 'raw/source/info.json').write_text(json.dumps(info))
                rewrite(episode / 'data.parquet', timestamp=timestamps)
                convert('hifi_umi', root / 'raw', root / 'out', **ZERO_OFFSETS)
                exported = read_export(root / 'out')[0]
                self.assertEqual(exported['length'], 4)
                if means[-1] == 180:
                    np.testing.assert_allclose(exported['right_gripper'], [0, .5, 1, 1], atol=1e-6)
                else:
                    self.assertAlmostEqual(exported['right_gripper'][1], .5 * (1 / 30) / .04, places=5)
                path, first = exported['videos']['top']
                np.testing.assert_allclose(video_means(path, first, 4), means, atol=3)

    def test_gap_in_resampled_rows_drops_only_that_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = make_shard(root / 'raw')
            # Episode 8 has no state between 40 ms and 300 ms: more than the 100 ms a frame may span.
            rewrite(shard / 'data/chunk-002/file-003.parquet',
                    timestamp=np.r_[row_times(4, 25), np.array([2., 2.04, 2.30, 2.34], dtype=np.float32)])
            convert('hifi_umi', root / 'raw', root / 'out', **ZERO_OFFSETS)
            self.assertEqual(manifest(root / 'out'), [{'episode_id': 'chunk_0000_part_0000_episode_000007'}])
            record = skip_record(root / 'out', 'chunk_0000_part_0000_episode_000008')
            self.assertEqual((record['outcome'], record['reason']), ('skipped', 'sampling_gap'))

    def test_export_resolution_follows_each_camera_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            top = episode / 'videos/observation.images.head_main.mp4'
            write_video(top, [i * 60 for i in range(4)], size=(96, 54))
            convert('hifi_umi', root / 'raw', root / 'out', **ZERO_OFFSETS)
            self.assertEqual(list(datasets(root / 'out')), ['top+left_wrist+right_wrist_640x360+640x480+640x480'])
            videos = read_export(root / 'out')[0]['videos']
            assert_contract_video(self, videos['top'][0], 4, (640, 360))
            assert_contract_video(self, videos['left_wrist'][0], 4, (640, 480))
            # A ratio without a registered export resolution is refused, not cropped or padded.
            write_video(top, [i * 60 for i in range(4)], size=(48, 48))
            assert_conversion_fails(self, 'has no export resolution', 'hifi_umi', root / 'raw', root / 'square',
                                    **ZERO_OFFSETS)

    def test_bad_inputs_fail_without_publishing(self):
        mutations = {
            'invalid pose mask': ('observation.state_valid', [[False] + [True] * 19] * 4),
            'null mask': ('observation.state_valid', [[None] + [True] * 19] * 4),
            'invalid frame': ('valid.frame', [True, False, True, True]),
            'null frame': ('valid.frame', [True, None, True, True]),
            'bad ordering': ('timestamp', [0, .2, .1, .3]),
            'zero duration': ('timestamp', [2.] * 4),
            'bad frame index': ('frame_index', [0, 2, 1, 3]),
            'mixed episodes': ('episode_index', [7, 7, 8, 7]),
            'unknown task': ('task_index', [0, 0, 2, 2]),
            'nonfinite state': ('observation.state', [[float('nan')] * 20] * 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            original = (episode / 'data.parquet').read_bytes()
            for name, (column, values) in mutations.items():
                with self.subTest(name=name):
                    (episode / 'data.parquet').write_bytes(original)
                    rewrite(episode / 'data.parquet', **{column: values})
                    # Rows are rejected before any video is decoded.
                    with patch.object(script, 'transcode_video') as video:
                        assert_conversion_fails(self, '', 'hifi_umi', root / 'raw', root / 'out',
                                                gripper_open_rad=1., **ZERO_OFFSETS)
                        video.assert_not_called()

    def test_video_frame_mismatch_and_missing_camera_roll_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'short', frames=3)
            episode = make_source(root / 'raw')
            first = episode / 'videos' / f'{next(iter(script.CAMERAS))}.mp4'
            original = first.read_bytes()
            first.write_bytes((root / 'short/episodes/episode_000007/videos' / first.name).read_bytes())
            assert_conversion_fails(self, 'Decoded 3 frames; expected 4', 'hifi_umi', root / 'raw', root / 'out',
                                    gripper_open_rad=1., **ZERO_OFFSETS)
            first.write_bytes(original)
            (episode / 'videos' / f'{list(script.CAMERAS)[-1]}.mp4').unlink()
            assert_conversion_fails(self, '', 'hifi_umi', root / 'raw', root / 'out',
                                    gripper_open_rad=1., **ZERO_OFFSETS)

    def test_missing_episode_metadata_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            (root / 'raw/episodes/episode_000008').mkdir()
            assert_conversion_fails(self, '', 'hifi_umi', root / 'raw', root / 'out',
                                    gripper_open_rad=1., **ZERO_OFFSETS)

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
            # A label that first appears on the last row keeps that row's frame.
            {'start_ns': 30, 'end_ns': 31, 'text': 'A'},
        ])
        self.assertEqual(script.instructions([1, 1], tasks, [10, 40]), [{'start_ns': 0, 'end_ns': 31, 'text': 'A'}])
        with self.assertRaises(ValueError):
            script.instructions([1], tasks, [0])


if __name__ == '__main__':
    unittest.main()
