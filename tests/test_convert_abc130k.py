from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import pyarrow.parquet as pq

from support import MODELS, assert_conversion_fails, convert, datasets, episode_errors, read_export, video_means
from loom_data import contract, sync
from loom_data.sources import abc130k as script
from loom_data.sources.policy import SourceDataError
from loom_data.sync import frame_ns
from loom_data.robot.kinematics import fk_poses
from loom_data.robot.urdf_model import parse_urdf


MODEL = MODELS / 'yam'
MS = 1_000_000
T0 = 1_700_000_000_000_000_000  # a realistic absolute acquisition time, ns
FRAME = [frame_ns(k) for k in range(8)]  # offsets of 30 fps frame times


def packets(count, codec='h264', step=40, size=(64, 48)):
    """One compressed access unit per frame; frame i is uniform gray i*step."""
    context = av.CodecContext.create({'h264': 'libx264', 'h265': 'libx265'}[codec], 'w')
    context.width, context.height = size
    context.pix_fmt = 'yuv420p'
    context.time_base = Fraction(1, 30)
    context.thread_count = 1
    context.options = {'bf': '0'}
    if codec == 'h265':
        context.options['x265-params'] = 'log-level=error:pools=1'
    output = []
    for index in range(count):
        frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), index*step, dtype=np.uint8), format='rgb24')
        frame.pts = index
        output.extend(context.encode(frame))
    output.extend(context.encode(None))
    assert len(output) == count
    return [bytes(packet) for packet in output]


def rotation_z(degrees):
    """Rotation about z as the nine column-major contract components."""
    a = np.deg2rad(degrees)
    return [np.cos(a), np.sin(a), 0., -np.sin(a), np.cos(a), 0., 0., 0., 1.]


class PlanarFK:
    """Stands in for YamFK: joint 0 is the x position, joint 1 a rotation about z in degrees."""
    def pose(self, side, position):
        return [float(position[0]), 0., 0., *rotation_z(position[1])]


def episode(states, cameras, task='pick', codec='h264', step=40, size=(64, 48)):
    """MCAP records of one episode, in log-time order.

    `states` maps a state topic to [(offset ns, position list)], `cameras` a
    camera topic to the offsets of its frames; frame i of a camera is gray i*step.
    """
    records = [('/instruction', T0 - 90 * MS, SimpleNamespace(data=task))]
    for topic, samples in states.items():
        records += [(topic, T0 + offset, SimpleNamespace(position=position)) for offset, position in samples]
    for topic, offsets in cameras.items():
        records += [(topic, T0 + offset, SimpleNamespace(data=data, format=codec))
                    for offset, data in zip(offsets, packets(len(offsets), codec, step, size))]
    return sorted(records, key=lambda record: record[1])


def reading(records):
    """Patch the MCAP reader to yield `records`, filtered by topic like the real one."""
    return patch.object(script, 'messages', side_effect=lambda path, topics: iter(
        [record for record in records[path.name] if record[0] in topics]))


def staged_state(directory):
    return np.array(pq.read_table(directory / contract.FRAMES)[contract.STATE_KEY].to_pylist())


ALL_CAMERAS = ('/top-camera', '/left-wrist-camera', '/right-wrist-camera')


def on_grid(frames, cameras=ALL_CAMERAS, size=(64, 48)):
    """An episode whose every stream is sampled exactly on `frames` 30 fps frame times."""
    arm = [(FRAME[k], [float(k), 0.]) for k in range(frames)]
    gripper = [(FRAME[k], [k / 10]) for k in range(frames)]
    return episode({'/left-arm-state': arm, '/right-arm-state': arm,
                    '/left-ee-state': gripper, '/right-ee-state': gripper},
                   {topic: FRAME[:frames] for topic in cameras}, size=size)


class ExportTests(unittest.TestCase):
    def export(self, root, episodes):
        """Convert `episodes` (episode_id -> MCAP records) from root/raw to root/export."""
        for name in episodes:
            (root / 'raw' / name).mkdir(parents=True)
            (root / 'raw' / name / 'episode.mcap').touch()
        with patch.object(script, 'YamFK', return_value=PlanarFK()), patch.object(
                script, 'messages', side_effect=lambda path, topics: iter(
                    [r for r in episodes[path.parent.name] if r[0] in topics])):
            convert('abc130k', root / 'raw', root / 'export', model_dir=MODEL)

    def test_nonfinite_gripper_fails_before_video_encoding(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = [('/right-ee-state', 123, SimpleNamespace(position=[value]))]
                with patch.object(script, 'messages', return_value=iter(records)), patch.object(script, 'VideoWriter') as video:
                    with self.assertRaisesRegex(ValueError, 'is not finite'):
                        script.convert_episode(root / 'episode_1', root / 'out', None)
                    video.assert_not_called()

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_episode_is_synchronized_and_transcoded_to_h264(self):
        fk = script.YamFK(MODEL)
        for codec in ('h264', 'h265'):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # Deliberately differing stream start times and lengths. The camera is
                # the earliest stream and the instruction is earlier still; neither
                # decides where the frames start. No source pose is supplied.
                records = episode(
                    {'/left-arm-state': [(10*MS, [0]*6), (90*MS, [0]*6)],
                     '/left-ee-state': [(10*MS, [-.002]), (90*MS, [1.00232])],
                     '/right-arm-state': [(15*MS, [0]*6), (55*MS, [0]*6), (100*MS, [0]*6)],
                     '/right-ee-state': [(15*MS, [-.002]), (55*MS, [1.00232]), (100*MS, [.5])]},
                    {'/top-left-camera' if codec == 'h265' else '/top-camera': FRAME[:4]}, codec=codec, step=80)
                records.append(('/top-right-camera', T0, SimpleNamespace(data=b'unused', format=codec)))
                with reading({'episode.mcap': records}):
                    record = script.convert_episode(root / 'episode_1', root / 'episode_1_out', fk)
                self.assertEqual(sorted(p.name for p in (root / 'episode_1_out').iterdir()), ['frames.parquet', 'top.mp4'])
                self.assertEqual((record['episode_id'], record['cameras'], record['length'], record['sizes']),
                                 ('episode_1', ['top'], 3, {'top': [640, 480]}))
                contract.validate_episode(root / 'episode_1_out', {**record, 'episode_id': 'episode_1_out'})
                # Frames start at the latest first sample (15 ms) and stop before the
                # earliest last sample (90 ms): 15, 48.33 and 81.67 ms.
                state = staged_state(root / 'episode_1_out')
                self.assertEqual(state.shape, (3, 26))
                for side in ('left', 'right'):
                    np.testing.assert_allclose(state[:, contract.STATE_SLICES[f'{side}_eef']],
                                               [fk.pose(side, [0]*6)] * 3, atol=1e-6)
                # Out-of-range openness is clipped at the source samples, then interpolated.
                np.testing.assert_allclose(state[:, 12], [5/80, (115/3)/80, (215/3)/80], atol=1e-6)
                np.testing.assert_allclose(state[:, 25], [0, (100/3)/40, 1 - .5*(80/3)/45], atol=1e-6)
                self.assertEqual(pq.read_table(root / 'episode_1_out/frames.parquet')['task'].to_pylist(), ['pick'] * 3)
                with av.open(str(root / 'episode_1_out/top.mp4')) as video:
                    stream = video.streams.video[0]
                    self.assertEqual(stream.codec_context.name, 'h264')
                    self.assertEqual(stream.codec_context.pix_fmt, 'yuv420p')
                    self.assertEqual(stream.average_rate, 30)
                    self.assertEqual(len(video.streams.audio), 0)
                # Nearest source frames: 0 ms, 33 ms and 67 ms; the 100 ms frame is unused.
                np.testing.assert_allclose(video_means(root / 'episode_1_out/top.mp4'), [0, 80, 160], atol=3)

    def test_asynchronous_state_is_interpolated_onto_the_shared_30_fps_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'episode_1').mkdir()
            (root / 'episode_1/annotation.mcap').touch()
            records = episode(
                {'/left-arm-state': [(0, [0., 0.]), (100*MS, [3., 90.])],
                 '/left-ee-state': [(0, [0.]), (50*MS, [1.]), (100*MS, [.5])],
                 '/right-arm-state': [(10*MS, [1., 30.]), (110*MS, [1., 30.])],
                 '/right-ee-state': [(-20*MS, [0.]), (60*MS, [1.]), (120*MS, [.5])]},
                {'/top-camera': [k*20*MS + 5*MS for k in range(6)]})
            annotation = [('/subtask-annotation', T0 + 40*MS, SimpleNamespace(data='lift'))]
            with reading({'episode.mcap': records, 'annotation.mcap': annotation}):
                record = script.convert_episode(root / 'episode_1', root / 'out', PlanarFK())
            # Streams span -20..120 ms, but only 10..100 ms is covered by all of them:
            # frames at 10, 43.33 and 76.67 ms.
            self.assertEqual(record['length'], 3)
            state = staged_state(root / 'out')
            weights = np.array([10, 10 + 100/3, 10 + 200/3]) / 100
            for weight, row in zip(weights, state):
                np.testing.assert_allclose(row[:12], [3*weight, 0, 0, *rotation_z(90*weight)], atol=1e-6)
                np.testing.assert_allclose(row[13:25], [1, 0, 0, *rotation_z(30)], atol=1e-6)
            np.testing.assert_allclose(state[:, 12], [.2, (10 + 100/3)/50, 1 - .5*(80/3)/50], atol=1e-6)
            np.testing.assert_allclose(state[:, 25], [30/80, (190/3)/80, 1 - .5*(50/3)/60], atol=1e-6)
            # The subtask starts at 40 ms on the same clock: the first frame still shows the task.
            self.assertEqual(pq.read_table(root / 'out/frames.parquet')['task'].to_pylist(), ['pick', 'lift', 'lift'])
            # Camera frames at 5, 25, ..., 105 ms; the nearest to each frame time are 5, 45 and 85 ms.
            np.testing.assert_allclose(video_means(root / 'out/top.mp4'), [0, 80, 160], atol=3)

    def test_each_frame_takes_the_nearest_camera_frame(self):
        gripper = [(FRAME[k], [0.]) for k in range(7)]
        arm = [(FRAME[k], [0., 0.]) for k in range(7)]
        states = {'/left-arm-state': arm, '/right-arm-state': arm, '/left-ee-state': gripper, '/right-ee-state': gripper}
        # 60 fps drops every other frame; 15 fps repeats frames, a frame time midway
        # between two source frames taking the earlier one.
        cameras = {'/top-camera': FRAME[:7],
                   '/left-wrist-camera': [k * 1_000_000_000 // 60 for k in range(13)],
                   '/right-wrist-camera': [k * 1_000_000_000 // 15 for k in range(4)]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with reading({'episode.mcap': episode(states, cameras, step=20)}):
                record = script.convert_episode(root / 'episode_1', root / 'out', PlanarFK())
            self.assertEqual((record['cameras'], record['length']), (['left_wrist', 'right_wrist', 'top'], 7))
            contract.validate_episode(root / 'out', {**record, 'episode_id': 'out'})
            for camera, frames in (('top', range(7)), ('left_wrist', range(0, 13, 2)), ('right_wrist', [0, 0, 1, 1, 2, 2, 3])):
                np.testing.assert_allclose(video_means(root / f'out/{camera}.mp4'), np.array(frames) * 20, atol=3,
                                           err_msg=camera)

    def test_each_stream_bounds_the_frame_interval(self):
        # Left gripper openness is the time since T0 in units of 300 ms, so an exported
        # value reveals the source time of its frame.
        ramp = [(k*100*MS, [k/3]) for k in range(4)]
        flat = lambda start, end: [(t*MS, [0., 0.]) for t in range(start, end + 1, 50)]
        for limiting, start, end in (('none', 0, 300), ('right arm', 37, 300), ('right arm', 0, 250), ('camera', 20, 240)):
            with self.subTest(limiting=limiting, start=start, end=end), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                late = limiting == 'right arm'
                states = {'/left-arm-state': flat(0, 300), '/left-ee-state': ramp,
                          '/right-arm-state': [(start*MS, [0., 0.]), *flat(50, 250), (end*MS, [0., 0.])] if late
                          else flat(0, 300),
                          '/right-ee-state': [(t*MS, [0.]) for t in (0, 100, 200, 300)]}
                camera = (range(start, end + 1, 20) if limiting == 'camera' else range(0, 301, 20))
                with reading({'episode.mcap': episode(states, {'/top-camera': [t*MS for t in camera]}, step=10)}):
                    record = script.convert_episode(root / 'episode_1', root / 'out', PlanarFK())
                times = staged_state(root / 'out')[:, 12] * 300  # ms since T0
                self.assertEqual(record['length'], len(times))
                self.assertAlmostEqual(times[0], start, delta=1e-3)
                np.testing.assert_allclose(np.diff(times), 1000/30, atol=1e-3)
                self.assertLessEqual(times[-1], end + 1e-3)
                self.assertGreater(times[-1] + 1000/30, end)

    def test_gaps_and_disjoint_streams_are_skipped_with_a_reason(self):
        arm = [(FRAME[k], [float(k), 0.]) for k in range(4)]
        gripper = [(FRAME[k], [k / 10]) for k in range(4)]
        states = {'/left-arm-state': arm, '/right-arm-state': arm, '/left-ee-state': gripper, '/right-ee-state': gripper}
        cameras = {topic: FRAME[:4] for topic in ALL_CAMERAS}
        # Frame times spanning a little over twice the RGB limit.
        span = [frame_ns(k) for k in range(2 * sync.FRAME_GAP_NS * sync.FPS // sync.NS_PER_SECOND + 4)]
        episodes = {
            'good': episode(states, cameras),
            # Two arm samples farther apart than the state limit, with frame times between them.
            'state_gap': episode({**states, '/left-arm-state': [(0, [0., 0.]), (sync.STATE_GAP_NS + 50*MS, [1., 0.])],
                                  '/right-ee-state': [(k*50*MS, [0.]) for k in range(4)]},
                                 {topic: FRAME[:6] for topic in cameras}),
            # The wrist camera has only the first and last frame: those midway are beyond the RGB limit of both.
            'rgb_gap': episode({topic: [(t, [0.] * (2 if 'arm' in topic else 1)) for t in span] for topic in states},
                               {**{topic: span for topic in cameras}, '/left-wrist-camera': [span[0], span[-1]]}, step=10),
            # The wrist camera starts after the state ends.
            'disjoint': episode(states, {**cameras, '/right-wrist-camera': [FRAME[5], FRAME[6]]}),
            # The streams overlap for less than one frame interval.
            'one_frame': episode(states, {**cameras, '/top-camera': [FRAME[3] - 20*MS, FRAME[3] + 10*MS]}),
        }
        expected = {'state_gap': 'sampling_gap', 'rgb_gap': 'sampling_gap',
                    'disjoint': 'no_common_interval', 'one_frame': 'no_common_interval'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in episodes:
                (root / 'raw' / name).mkdir(parents=True)
                (root / 'raw' / name / 'episode.mcap').touch()
            with patch.object(script, 'YamFK', return_value=PlanarFK()), patch.object(
                    script, 'messages', side_effect=lambda path, topics: iter(
                        [r for r in episodes[path.parent.name] if r[0] in topics])):
                # Called directly, the converter classifies the episode ...
                for name, reason in expected.items():
                    with self.assertRaises(SourceDataError) as caught:
                        script.convert_episode(root / 'raw' / name, root / 'direct' / name, PlanarFK())
                    self.assertEqual(caught.exception.reason, reason, name)
                # ... and the job leaves it out although this source has no drop policy.
                convert('abc130k', root / 'raw', root / 'export', model_dir=MODEL)
            self.assertEqual(script.SOURCE.policy, 'strict')
            self.assertEqual(list(datasets(root / 'export')), ['top+left_wrist+right_wrist_640x480'])
            exported = read_export(root / 'export')
            self.assertEqual([e['episode_id'] for e in exported], ['good'])
            self.assertEqual(exported[0]['length'], 4)
            np.testing.assert_allclose(exported[0]['left_eef'][:, 0], [0, 1, 2, 3], atol=1e-6)
            np.testing.assert_allclose(exported[0]['right_gripper'], [0, .1, .2, .3], atol=1e-6)
            work = root / 'export.work'
            summary = json.loads((work / 'conversion_summary.json').read_text())
            self.assertEqual((summary['planned'], summary['exported'], summary['dropped'], summary['failed']), (5, 1, 4, 0))
            self.assertEqual(summary['drop_reasons'], {'no_common_interval': 2, 'sampling_gap': 2})
            for name, reason in expected.items():
                record = json.loads((work / 'records' / f'{name}.json').read_text())
                self.assertEqual((record['outcome'], record['reason'], record['reasons']), ('skipped', reason, [reason]))
                self.assertTrue(record['detail'].strip())
            self.assertEqual(episode_errors(root / 'export'), '')

    def test_a_short_camera_dropout_is_exported_with_the_neighbouring_image_repeated(self):
        # The top camera loses frames 3, 4 and 5 (a 133 ms hole); every other stream is complete.
        arm = [(FRAME[k], [float(k), 0.]) for k in range(8)]
        gripper = [(FRAME[k], [k / 10]) for k in range(8)]
        kept = [0, 1, 2, 6, 7]
        records = episode({'/left-arm-state': arm, '/right-arm-state': arm,
                           '/left-ee-state': gripper, '/right-ee-state': gripper},
                          {'/top-camera': [FRAME[k] for k in kept],
                           '/left-wrist-camera': FRAME, '/right-wrist-camera': FRAME}, step=30)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.export(root, {'dropout': records})
            exported, = read_export(root / 'export')
            self.assertEqual((exported['episode_id'], exported['length']), ('dropout', 8))
            self.assertEqual(episode_errors(root / 'export'), '')
            # Source image i of the top camera is gray 30*i. The image before the hole also fills
            # frames 3 and 4 (4 is midway, so it takes the earlier), the one after it frame 5.
            path, first = exported['videos']['top']
            np.testing.assert_allclose(video_means(path, first, 8), np.array([0, 1, 2, 2, 2, 3, 3, 4]) * 30, atol=3)
            for camera in ('left_wrist', 'right_wrist'):
                path, first = exported['videos'][camera]
                np.testing.assert_allclose(video_means(path, first, 8), np.arange(8) * 30, atol=3, err_msg=camera)
            # State is sampled on every frame time and does not notice the dropout.
            for side in ('left', 'right'):
                np.testing.assert_allclose(exported[f'{side}_eef'][:, 0], np.arange(8), atol=1e-6)
                np.testing.assert_allclose(exported[f'{side}_gripper'], np.arange(8) / 10, atol=1e-6)

    def test_published_dataset_is_a_30_fps_lerobot_v3_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.export(root, {'episode_a': on_grid(3), 'episode_b': on_grid(4)})
            self.assertFalse((root / 'export/episodes.jsonl').exists())
            (layout, dataset), = datasets(root / 'export').items()
            self.assertEqual(layout, 'top+left_wrist+right_wrist_640x480')
            self.assertEqual(dataset.cameras, ['left_wrist', 'right_wrist', 'top'])
            self.assertEqual((dataset.info['codebase_version'], dataset.info['fps'], dataset.info['total_frames']),
                             ('v3.0', 30, 7))
            self.assertNotIn('action', dataset.info['features'])
            exported = read_export(root / 'export')
            self.assertEqual([(e['episode_id'], e['length']) for e in exported], [('episode_a', 3), ('episode_b', 4)])
            for item in exported:
                # Samples sit on the frame times, so they pass through unchanged.
                frames = np.arange(item['length'])
                np.testing.assert_allclose(item['left_eef'][:, 0], frames, atol=1e-6)
                np.testing.assert_allclose(item['left_gripper'], frames / 10, atol=1e-6)
                self.assertEqual(item['tasks'], ['pick'] * item['length'])
                for path, first in item['videos'].values():
                    np.testing.assert_allclose(video_means(path, first, item['length']), frames * 40, atol=3)

    def test_aspect_ratios_publish_into_separate_layouts(self):
        episodes = {'standard_a': on_grid(3), 'wide': on_grid(4, size=(64, 40)), 'standard_b': on_grid(4)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.export(root, episodes)
            found = datasets(root / 'export')
            self.assertEqual(list(found), ['top+left_wrist+right_wrist_640x400', 'top+left_wrist+right_wrist_640x480'])
            self.assertEqual({layout: [e['episode_id'] for e in dataset.episodes] for layout, dataset in found.items()},
                             {'top+left_wrist+right_wrist_640x400': ['wide'],
                              'top+left_wrist+right_wrist_640x480': ['standard_a', 'standard_b']})
            for layout, height in (('top+left_wrist+right_wrist_640x400', 400), ('top+left_wrist+right_wrist_640x480', 480)):
                for camera in ('top', 'left_wrist', 'right_wrist'):
                    feature = found[layout].info['features'][contract.video_key(camera)]
                    self.assertEqual(feature['shape'], [height, 640, 3])
                    path, _ = found[layout].video(found[layout].episodes[0], camera)
                    with av.open(str(path)) as video:
                        self.assertEqual((video.streams.video[0].width, video.streams.video[0].height), (640, height))
            self.assertFalse((root / 'export.work/conversion_summary.json').exists())  # nothing was left out

    def test_other_camera_sets_are_skipped_as_unsupported_layout(self):
        self.assertEqual(script.SOURCE.camera_sets, (('left_wrist', 'right_wrist', 'top'),))
        episodes = {'complete': on_grid(3),
                    'no_right_wrist': on_grid(3, cameras=('/top-camera', '/left-wrist-camera')),
                    'top_only': on_grid(3, cameras=('/top-left-camera',))}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.export(root, episodes)
            self.assertEqual(list(datasets(root / 'export')), ['top+left_wrist+right_wrist_640x480'])
            self.assertEqual([e['episode_id'] for e in read_export(root / 'export')], ['complete'])
            work = root / 'export.work'
            summary = json.loads((work / 'conversion_summary.json').read_text())
            self.assertEqual((summary['exported'], summary['dropped'], summary['failed']), (1, 2, 0))
            self.assertEqual(summary['drop_reasons'], {'unsupported_layout': 2})
            for name, cameras in (('no_right_wrist', 'left_wrist+top'), ('top_only', 'top')):
                record = json.loads((work / 'records' / f'{name}.json').read_text())
                self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))
                self.assertIn(cameras, record['detail'])
            self.assertEqual(episode_errors(root / 'export'), '')

    def test_annotations_use_shared_clock_and_clip(self):
        self.assertEqual(script.instructions('task', [], 100, 400),
                         [{'start_ns': 0, 'end_ns': 300, 'text': 'task'}])
        self.assertEqual(script.instructions('task', [(150, 'a'), (300, 'b'), (500, 'c')], 100, 400),
                         [{'start_ns': 0, 'end_ns': 50, 'text': 'task'},
                          {'start_ns': 50, 'end_ns': 200, 'text': 'a'},
                          {'start_ns': 200, 'end_ns': 300, 'text': 'b'}])
        self.assertEqual(script.instructions('task', [(50, 'old'), (100, 'new')], 100, 400),
                         [{'start_ns': 0, 'end_ns': 300, 'text': 'new'}])

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_fk_mount_origin_and_axes(self):
        fk = script.YamFK(MODEL)
        robot = parse_urdf(MODEL / 'robot.urdf')
        for side in ('left', 'right'):
            q = [.2, -.3, .5, -.6, .4, .1]
            wrist = fk_poses(robot, {f'arm_{side}_joint{i+1}': value for i, value in enumerate(q)})[f'arm_{side}_link6']
            pose = fk.pose(side, q)
            np.testing.assert_allclose(pose[:3], wrist[:3, 3], atol=1e-9)
            np.testing.assert_allclose(pose[3:6], wrist[:3, 2], atol=1e-9)
            np.testing.assert_allclose(pose[6:9], wrist[:3, 0], atol=1e-9)
        with self.assertRaises(ValueError):
            fk.pose('left', [0] * 7)
        with self.assertRaises(ValueError):
            fk.pose('left', [float('nan')] * 6)

    def test_failed_export_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw/episode_1'
            source.mkdir(parents=True)
            (source / 'episode.mcap').touch()
            with patch.object(script, 'YamFK'), patch.object(script, 'convert_episode', side_effect=ValueError('bad data')):
                assert_conversion_fails(self, 'bad data', 'abc130k', root / 'raw', root / 'export', model_dir=MODEL)
            self.assertFalse((root / 'export').exists())
            self.assertIn('bad data', episode_errors(root / 'export'))


if __name__ == '__main__':
    unittest.main()
