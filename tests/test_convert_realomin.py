from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import av
import numpy as np
import pyarrow.parquet as pq

from scripts.convert import realomin as script
from scripts.convert.export_common import tool_transform
from scripts.convert.export_validation import validate_episode
from scripts.robot.realomin_mount import mounting_geometry


def geometry_urdfs(root):
    """Minimal vendor joint excerpts; no network or full asset package needed."""
    fixture = json.loads((Path(__file__).parent / 'fixtures/realomin_v3_mount_geometry.json').read_text())
    paths = []
    for kind in ('gripper', 'controller'):
        robot = ET.Element('robot', name=kind)
        ET.SubElement(robot, 'link', name='base_link')
        for name, value in fixture[f'{kind}_joints'].items():
            ET.SubElement(robot, 'link', name=name)
            joint = ET.SubElement(robot, 'joint', name=name, type=value['type'])
            ET.SubElement(joint, 'parent', link='base_link')
            ET.SubElement(joint, 'child', link=name)
            ET.SubElement(joint, 'origin', xyz=' '.join(map(str, value['xyz'])),
                          rpy=' '.join(map(str, value['rpy'])))
        path = root / f'{kind}.urdf'
        ET.ElementTree(robot).write(path)
        paths.append(path)
    return paths, fixture


def packets():
    encoder = av.CodecContext.create('libx264', 'w')
    encoder.width = encoder.height = 32
    encoder.pix_fmt = 'yuv420p'
    encoder.time_base = Fraction(1, 30)
    encoder.thread_count = 1
    encoder.options = {'bf': '0'}
    output = []
    for i in range(3):
        frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i*70, dtype=np.uint8), format='rgb24')
        frame.pts = i
        output.extend(encoder.encode(frame))
    output.extend(encoder.encode(None))
    return [bytes(p) for p in output]


class RealOminTests(unittest.TestCase):
    def test_default_rear_mount_is_reproducible_from_vendor_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, fixture = geometry_urdfs(Path(temporary))
            geometry = mounting_geometry(*paths)
            np.testing.assert_allclose(geometry['eef_offset'], fixture['mount_in_raw_eef_m'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(script.DEFAULT_EEF_OFFSET, geometry['eef_offset'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(geometry['mount_in_gripper_base_m'],
                                       fixture['mount_in_gripper_base_m'], atol=1e-12, rtol=0)
            self.assertAlmostEqual(geometry['max_pivot_residual_m'], .0001, places=12)
            # Moving the explicit rear flange moves the result. Finger lengths
            # are absent from the calculation and cannot change the mount datum.
            tree = ET.parse(paths[1])
            tree.find("./joint[@name='joint_Flange']/origin").set('xyz', '-0.077889 -0.000038794 -0.097619')
            tree.write(paths[1])
            self.assertAlmostEqual(np.linalg.norm(np.array(mounting_geometry(*paths)['eef_offset'])
                                                 - geometry['eef_offset']), .01, places=10)

    def test_nominal_mount_is_used_by_direct_serial_parallel_and_cli_exports(self):
        from scripts.convert import parallel
        from scripts.convert.parallel_adapters import make_config
        import subprocess
        import sys
        records = []
        # Two changing source rotations catch accidental world-frame addition.
        quaternions = [(0., 0., 0., 1.), (0., 0., np.sqrt(.5), np.sqrt(.5))]
        for topic, name in script.STATES.items():
            for time, quaternion in zip((15, 40), quaternions):
                value = (NS(frame_id='world', pose=NS(position=NS(x=1., y=2., z=3.),
                         orientation=NS(**dict(zip(('x', 'y', 'z', 'w'), quaternion)))),
                         HasField=lambda field: True) if name.endswith('_eef') else NS(value=.0515))
                records.append((topic, time, value))
        records.extend(('/robot0/sensor/camera0/compressed', time, NS(data=data, format='h264'))
                       for time, data in zip((10, 20, 30), packets()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'raw/clean_bowl/1/1.mcap'
            source.parent.mkdir(parents=True)
            source.touch()
            with patch.object(script, 'messages', side_effect=lambda path, video=False: iter(
                    row for row in records if bool(script.CAMERA.fullmatch(row[0])) == video)):
                direct = script.convert_episode({'path': source, 'instruction': 'clean bowl'}, root / 'direct')
                script.convert(root / 'raw', root / 'serial')
                config = make_config('realomin', root / 'raw')
                parallel.prepare(config, root / 'parallel')
                self.assertEqual(parallel.worker(root / 'parallel', 0, 1), 0)
                parallel.finalize(root / 'parallel')
            validate_episode(root / 'direct', {**direct, 'episode_id': 'direct'})
            record = json.loads((root / 'serial/episodes.jsonl').read_text())
            p = np.array([1., 2., 3.])
            # Values independently fixed by the vendor geometry fixture.
            offset = np.array([-.0682778317412809, .0003727122423727631, -.09805170329867613])
            expected_positions = [p + offset, p + [-offset[1], offset[0], offset[2]]]
            for side in ('left', 'right'):
                relative = Path('state') / f'{side}_eef.parquet'
                direct_table = pq.read_table(root / 'direct' / relative)
                np.testing.assert_allclose(np.array(direct_table['pose'].to_pylist())[:, :3], expected_positions, atol=1e-12)
                for output in ('serial', 'parallel'):
                    table = pq.read_table(root / output / 'episodes' / record['episode_id'] / relative)
                    self.assertEqual(table.to_pydict(), direct_table.to_pydict())
            # Parser must accept omission of --eef-offset, with no model files.
            bad = root / 'bad/clean_bowl/1.mcap'
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b'bad MCAP')
            result = subprocess.run([sys.executable, '-m', 'scripts.convert.realomin',
                                     '--input-dir', str(root / 'bad'), '--output-dir', str(root / 'cli')],
                                    cwd=script.ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'cli/episodes.jsonl').exists())

    def test_shared_origin_keeps_native_lengths_offsets_and_clips_only_gripper(self):
        records = []
        for topic, name in script.STATES.items():
            times = [110, 190] if name.startswith('left') else [115, 155, 200]
            for i, time in enumerate(times):
                if name.endswith('_eef'):
                    value = NS(frame_id='world', pose=NS(position=NS(x=1., y=2., z=3.),
                               orientation=NS(x=0., y=0., z=0., w=1.)), HasField=lambda field: True)
                else:
                    value = NS(value=[-.002, .105, .0515][i])
                records.append((topic, time, value))
        for time, packet in zip([100, 140, 180], packets()):
            records.append(('/robot0/sensor/camera0/compressed', time, NS(data=packet, format='h264')))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'episode_test'
            def messages(path, *, video=False):
                return iter(r for r in records if bool(script.CAMERA.fullmatch(r[0])) == video)
            transform = tool_transform([.1, 0, 0], [0, 0, 0])
            with patch.object(script, 'messages', side_effect=messages):
                record = script.convert_episode({'path': Path('sample.mcap'), 'instruction': 'clean bowl'}, destination, transform)
            validate_episode(destination, record)
            self.assertEqual(record['instructions'], [{'start_ns': 0, 'end_ns': 100, 'text': 'clean bowl'}])
            self.assertEqual(pq.read_table(destination/'rgb/left_wrist.parquet')['timestamp_ns'].to_pylist(), [0, 40, 80])
            table = pq.read_table(destination/'state/left_eef.parquet')
            self.assertEqual(table['timestamp_ns'].to_pylist(), [10, 90])
            np.testing.assert_allclose(table['pose'].to_pylist()[0], [1.1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1])
            self.assertEqual(pq.read_table(destination/'state/left_gripper.parquet')['openness'].to_pylist(), [0., 1.])
            self.assertEqual(pq.read_table(destination/'state/right_gripper.parquet')['openness'].to_pylist(), [0., 1., .5])

    def test_payload_header_nanoseconds_are_preserved_and_channel_is_authoritative(self):
        timestamp = 1_765_520_396_511_462_525
        topic = '/robot1/vio/eef_pose'
        decoded = NS(header=NS(timestamp=timestamp, topic_name='/robot0/vio/eef_pose'), HasField=lambda key: True)
        reader = NS(get_summary=lambda: NS(channels={1: NS(topic=topic)}),
                    iter_decoded_messages=lambda **kwargs: iter([(None, NS(topic=topic), NS(log_time=13), decoded)]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sample.mcap'
            path.touch()
            with patch('mcap.reader.make_reader', return_value=reader):
                records = list(script.messages(path))
        self.assertEqual(records[0][:2], (topic, timestamp))

    def test_bad_state_fails_before_expensive_video(self):
        value = NS(value=float('nan'))
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(script, 'messages', return_value=iter([('/robot0/sensor/magnetic_encoder', 100, value)])), patch.object(script, 'VideoWriter') as writer:
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    script.convert_episode({'path': Path('sample.mcap'), 'instruction': 'clean'}, Path(temporary)/'ep', transform=tool_transform([0., 0., 0.], script.DEFAULT_TOOL_RPY))
                writer.assert_not_called()

    def test_repeated_leaf_filenames_have_distinct_stable_ids_and_correct_skills(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for skill in ('clean_bowl', 'fold_towel'):
                path = root/skill/'00001/00001.mcap'
                path.parent.mkdir(parents=True)
                path.touch()
            items = script.discover(root)
            self.assertEqual(len({i for i, _ in items}), 2)
            self.assertEqual([s['instruction'] for _, s in items], ['clean bowl', 'fold towel'])
            self.assertEqual([i for i, _ in items], [i for i, _ in script.discover(root)])
            self.assertTrue(all(s['instruction']=='Override' for _, s in script.discover(root, 'Override')))

    def test_nominal_tool_axes_follow_finger_plane_and_are_right_handed(self):
        transform = tool_transform([0., 0., 0.], script.DEFAULT_TOOL_RPY)
        np.testing.assert_allclose(transform[:3, 0], [np.cos(np.pi/12), 0, np.sin(np.pi/12)])
        np.testing.assert_allclose(np.cross(transform[:3, 2], transform[:3, 0]), transform[:3, 1])
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.)

    def test_failed_conversion_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'raw/clean_bowl/1/1.mcap'
            source.parent.mkdir(parents=True)
            source.touch()
            with patch.object(script, 'convert_episode', side_effect=ValueError('bad sample')):
                with self.assertRaisesRegex(RuntimeError, 'Conversion failed'):
                    script.convert(root/'raw', root/'out', eef_offset=[0., 0., 0.])
            self.assertIn('bad sample', next((root/'out.work/errors').glob('*.err')).read_text())
            self.assertFalse((root/'out').exists())
            self.assertFalse(list(root.glob('.out-*')))

    def test_parallel_adapter_keeps_calibration_and_resumes_native_mcap(self):
        from scripts.convert import parallel
        from scripts.convert.parallel_adapters import make_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'raw/clean_bowl/1/1.mcap'
            path.parent.mkdir(parents=True)
            path.touch()
            records = []
            for topic, name in script.STATES.items():
                value = (NS(frame_id='world', pose=NS(position=NS(x=0., y=0., z=0.),
                         orientation=NS(x=0., y=0., z=0., w=1.)), HasField=lambda field: True)
                         if name.endswith('_eef') else NS(value=.0515))
                records.extend((topic, time, value) for time in (15,40))
            records.extend(('/robot0/sensor/camera0/compressed', time, NS(data=data,format='h264'))
                           for time,data in zip((10,20,30),packets()))
            config = make_config('realomin',root/'raw',eef_offset=[.2,0,0],tool_rpy=[0,0,0],instruction='Clean')
            parallel.prepare(config,root/'out')
            with patch.object(script,'messages',side_effect=lambda path,video=False: iter(
                    r for r in records if bool(script.CAMERA.fullmatch(r[0]))==video)):
                self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            import json
            record=json.loads((root/'out/episodes.jsonl').read_text())
            destination=root/'out/episodes'/record['episode_id']
            np.testing.assert_allclose(pq.read_table(destination/'state/left_eef.parquet')['pose'].to_pylist()[0][:3],[.2,0,0])
            parallel.prepare(config,root/'out')
            with patch.object(script,'convert_episode',side_effect=AssertionError('must resume')):
                self.assertEqual(parallel.worker(root/'out',0,1),0)
            with self.assertRaises(ValueError):
                make_config('realomin',root/'raw',gripper_closed=0,gripper_open=0, eef_offset=[0., 0., 0.])
            with self.assertRaises(ValueError):
                make_config('realomin',root/'raw',left_tool_rpy=[0,0,0], eef_offset=[0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
