from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert import abc130k as script
from scripts.convert.export_common import VideoWriter, write_state
from scripts.robot.kinematics import fk_poses
from scripts.robot.urdf_model import parse_urdf


MODEL = script.ROOT / 'assets/robot_models/yam'


class ExportTests(unittest.TestCase):
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
    def test_episode_video_and_shared_origin(self):
        import av
        for codec, encoder in [('h264', 'libx264'), ('h265', 'libx265')]:
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                context = av.CodecContext.create(encoder, 'w')
                context.width = context.height = 32
                context.pix_fmt = 'yuv420p'
                context.time_base = Fraction(1, 30)
                context.options = {'bf': '0'}
                if codec == 'h265':
                    context.options['x265-params'] = 'log-level=error:pools=1'
                packets = []
                for index in range(3):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index*80, dtype=np.uint8), format='rgb24')
                    frame.pts = index
                    packets.extend(context.encode(frame))
                packets.extend(context.encode(None))
                self.assertEqual(len(packets), 3)
                # Deliberately differing stream start times and lengths. Camera
                # is the earliest stream; instruction is earlier still and must
                # not determine the common origin. No source pose is supplied.
                records = [('/instruction', 10, SimpleNamespace(data='pick'))]
                for topic in script.STATES:
                    positions = [0]*6 if 'arm' in topic else [.5]
                    for i, timestamp in enumerate([110, 190] if 'left' in topic else [115, 155, 200]):
                        value = positions if 'arm' in topic else [[-.002], [1.00232], [.5]][i]
                        records.append((topic, timestamp, SimpleNamespace(position=value)))
                for timestamp, packet in zip([100, 140, 180], packets):
                    records.append(('/top-camera', timestamp, SimpleNamespace(data=bytes(packet), format=codec)))
                with patch.object(script, 'messages', side_effect=lambda path, topics: (
                        r for r in sorted(records, key=lambda r: r[1]) if r[0] in topics)):
                    metadata = script.convert_episode(root / 'episode_1', root / 'out', script.YamFK(MODEL))
                self.assertEqual(metadata['instructions'], [{'start_ns': 0, 'end_ns': 100, 'text': 'pick'}])
                index_table = pq.read_table(root / 'out/rgb/top.parquet')
                self.assertEqual(index_table['timestamp_ns'].to_pylist(), [0, 40, 80])
                self.assertEqual(index_table['frame_index'].to_pylist(), [0, 1, 2])
                self.assertEqual(pq.read_table(root / 'out/state/left_eef.parquet')['timestamp_ns'].to_pylist(), [10, 90])
                self.assertEqual(pq.read_table(root / 'out/state/left_gripper.parquet')['openness'].to_pylist(), [0., 1.])
                self.assertEqual(pq.read_table(root / 'out/state/right_gripper.parquet')['openness'].to_pylist(), [0., 1., .5])
                with av.open(str(root / 'out/rgb/top.mp4')) as video:
                    self.assertEqual(video.streams.video[0].codec_context.name, 'h264')
                    self.assertEqual(len(video.streams.audio), 0)
                    frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 3)
                    for i, frame in enumerate(frames):
                        self.assertAlmostEqual(float(frame.to_ndarray(format='rgb24').mean()), i*80, delta=3)

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
    def test_fk_tcp_offset_and_axes(self):
        fk = script.YamFK(MODEL)
        robot = parse_urdf(MODEL / 'robot.urdf')
        for side in ('left', 'right'):
            q = [.2, -.3, .5, -.6, .4, .1]
            wrist = fk_poses(robot, {f'arm_{side}_joint{i+1}': value for i, value in enumerate(q)})[f'arm_{side}_link6']
            pose = fk.pose(side, q)
            np.testing.assert_allclose(pose[:3], wrist[:3, 3] + wrist[:3, 2] * .1347, atol=1e-9)
            np.testing.assert_allclose(pose[3:6], wrist[:3, 2], atol=1e-9)
            np.testing.assert_allclose(pose[6:9], wrist[:3, 0], atol=1e-9)
        with self.assertRaises(ValueError):
            fk.pose('left', [0] * 7)
        with self.assertRaises(ValueError):
            fk.pose('left', [float('nan')] * 6)

    def test_state_schema_preserves_asynchronous_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.parquet'
            pose = [1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1]
            write_state(path, [105, 119], [pose, pose], 100, pose=True)
            table = pq.read_table(path)
            self.assertEqual(table['timestamp_ns'].to_pylist(), [5, 19])
            self.assertEqual(table.schema.field('pose').type, pa.list_(pa.float64(), 12))
            self.assertFalse(table.schema.field('pose').nullable)
            write_state(path, [101, 110, 120], [0, .5, 1], 100)
            self.assertEqual(pq.read_table(path)['openness'].type, pa.float32())
            for times, values in [([100], [1.1]), ([110, 100], [.1, .2]), ([100], [float('nan')])]:
                with self.assertRaises(ValueError):
                    write_state(path, times, values, 100)

    def test_failed_export_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw/episode_1'
            source.mkdir(parents=True)
            (source / 'episode.mcap').touch()
            with patch.object(script, 'YamFK'), patch.object(script, 'convert_episode', side_effect=ValueError('bad data')):
                with self.assertRaisesRegex(ValueError, 'bad data'):
                    script.convert(root / 'raw', root / 'export', MODEL)
            self.assertFalse((root / 'export').exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['raw'])


if __name__ == '__main__':
    unittest.main()
