import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq

from support import assert_conversion_fails, convert, datasets, episode_errors, read_export, video_means
from loom_data import contract, sync
from loom_data.sources import realomin as script
from loom_data.geometry import tool_transform
from loom_data.sources.policy import SourceDataError
from loom_data.sync import frame_ns
from test_convert_abc130k import packets
from loom_data.robot.realomin_mount import mounting_geometry


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


MS = 1_000_000
T0 = 1_765_520_396_511_462_525  # an acquisition header time from the real data, ns
FRAME = [frame_ns(k) for k in range(8)]  # offsets of 30 fps frame times
EEF = {'left': '/robot0/vio/eef_pose', 'right': '/robot1/vio/eef_pose'}
GRIPPER = {'left': '/robot0/sensor/magnetic_encoder', 'right': '/robot1/sensor/magnetic_encoder'}
CAMERA = {'left': '/robot0/sensor/camera0/compressed', 'right': '/robot1/sensor/camera0/compressed'}
IDENTITY = tool_transform([0., 0., 0.], [0., 0., 0.])


def pose(x=1., degrees=0.):
    """A source pose message at (x, 2, 3), rotated about world z."""
    half = np.deg2rad(degrees) / 2
    return NS(frame_id='world', pose=NS(position=NS(x=x, y=2., z=3.),
              orientation=NS(x=0., y=0., z=np.sin(half), w=np.cos(half))), HasField=lambda field: True)


def rotation_z(degrees):
    """Rotation about z as the nine column-major contract components."""
    a = np.deg2rad(degrees)
    return [np.cos(a), np.sin(a), 0., -np.sin(a), np.cos(a), 0., 0., 0., 1.]


def episode(states, cameras, step=40):
    """MCAP records of one episode.

    `states` maps a state topic to [(offset ns, message)], `cameras` a camera
    topic to the offsets of its frames; frame i of a camera is gray i*step.
    """
    records = [(topic, T0 + offset, value) for topic, samples in states.items() for offset, value in samples]
    for topic, offsets in cameras.items():
        records += [(topic, T0 + offset, NS(data=data, format='h264'))
                    for offset, data in zip(offsets, packets(len(offsets), step=step))]
    return records


def on_grid(frames, cameras=tuple(CAMERA.values())):
    """An episode whose every stream is sampled exactly on `frames` 30 fps frame times.

    The left arm moves 1 m per frame along x; openness is frame/10 on both sides.
    """
    states = {EEF['left']: [(FRAME[k], pose(float(k))) for k in range(frames)],
              EEF['right']: [(FRAME[k], pose()) for k in range(frames)]}
    for topic in GRIPPER.values():
        states[topic] = [(FRAME[k], NS(value=k / 10 * script.DEFAULT_GRIPPER_OPEN)) for k in range(frames)]
    return episode(states, {topic: FRAME[:frames] for topic in cameras})


def reading(records):
    """Patch the MCAP reader to yield `records`, filtered by topic like the real one.

    `records` is one episode's list, or a function of the MCAP path returning it.
    """
    select = records if callable(records) else lambda path: records
    return patch.object(script, 'messages', side_effect=lambda path, topics: iter(
        [record for record in select(path) if record[0] in topics]))


def staged_state(directory):
    return np.array(pq.read_table(directory / contract.FRAMES)[contract.STATE_KEY].to_pylist())


def mcap(root, skill, name='1'):
    """An empty placeholder MCAP under root/raw; its content comes from `reading`."""
    path = root / 'raw' / skill / name / f'{name}.mcap'
    path.parent.mkdir(parents=True)
    path.touch()
    return path


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
        from loom_data.convert import job as parallel
        from loom_data.convert.config import make_config
        import subprocess
        import sys
        # Two changing source rotations catch accidental world-frame addition. The samples
        # sit on frames 0 and 2, so those rows are the source samples themselves.
        states = {topic: [(FRAME[0], pose(1., 0)), (FRAME[2], pose(1., 90))] for topic in EEF.values()}
        states.update({topic: [(FRAME[0], NS(value=.0515)), (FRAME[2], NS(value=.0515))] for topic in GRIPPER.values()})
        records = episode(states, {topic: FRAME[:3] for topic in CAMERA.values()})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = mcap(root, 'clean_bowl')
            with reading(records):
                direct = script.convert_episode({'path': source, 'instruction': 'clean bowl'}, root / 'direct')
                convert('realomin', root / 'raw', root / 'serial')
                config = make_config('realomin', root / 'raw')
                parallel.prepare(config, root / 'parallel')
                self.assertEqual(parallel.worker(root / 'parallel', 0, 1), 0)
                parallel.finalize(root / 'parallel')
            contract.validate_episode(root / 'direct', direct)
            self.assertEqual((direct['cameras'], direct['length']), (['left_wrist', 'right_wrist'], 3))
            state = staged_state(root / 'direct')
            p = np.array([1., 2., 3.])
            # Values independently fixed by the vendor geometry fixture.
            offset = np.array([-.0682778317412809, .0003727122423727631, -.09805170329867613])
            expected_positions = [p + offset, p + [-offset[1], offset[0], offset[2]]]
            for side in ('left', 'right'):
                np.testing.assert_allclose(state[[0, 2], contract.STATE_SLICES[f'{side}_eef']][:, :3],
                                           expected_positions, atol=1e-6)
                np.testing.assert_allclose(state[:, contract.STATE_SLICES[f'{side}_gripper']], .5, atol=1e-6)
            for output in ('serial', 'parallel'):
                self.assertEqual(list(datasets(root / output)), ['left_wrist+right_wrist_640x480'])
                item, = read_export(root / output)
                np.testing.assert_array_equal(item['state'], state)
                self.assertEqual(item['tasks'], ['clean bowl'] * 3)
            # Parser must accept omission of --eef-offset, with no model files. The only
            # episode is unreadable, so it is dropped and nothing is left to publish.
            bad = root / 'bad/clean_bowl/1.mcap'
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b'bad MCAP')
            result = subprocess.run([sys.executable, '-m', 'loom_data', 'convert', 'run', '--dataset', 'realomin',
                                     '--input-dir', str(root / 'bad'), '--output-dir', str(root / 'cli')],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nothing to publish', result.stdout + result.stderr)
            self.assertFalse((root / 'cli').exists())
            summary = json.loads((root / 'cli.work/conversion_summary.json').read_text())
            self.assertEqual((summary['dropped'], summary['failed'], summary['drop_reasons']), (1, 0, {'invalid_mcap': 1}))

    def test_asynchronous_streams_are_resampled_onto_the_shared_30_fps_frames(self):
        # Every stream has its own start, end and rate. Offsets in ms from T0.
        states = {
            EEF['left']: [(0, pose(1., 0)), (100*MS, pose(4., 90))],
            GRIPPER['left']: [(0, NS(value=-.002)), (50*MS, NS(value=.105)), (100*MS, NS(value=.0515))],
            EEF['right']: [(10*MS, pose(1., 30)), (110*MS, pose(1., 30))],
            GRIPPER['right']: [(-20*MS, NS(value=0.)), (60*MS, NS(value=.103)), (120*MS, NS(value=.0515))]}
        cameras = {CAMERA['left']: [k*20*MS + 5*MS for k in range(6)],  # 5, 25, ..., 105 ms
                   CAMERA['right']: [0, 66*MS, 132*MS]}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'episode_test'
            with reading(episode(states, cameras)):
                record = script.convert_episode({'path': Path('sample.mcap'), 'instruction': 'clean bowl'},
                                                destination, tool_transform([.1, 0, 0], [0, 0, 0]))
            contract.validate_episode(destination, record)
            # Streams span -20..132 ms, but only 10..100 ms is covered by all of them:
            # frames at 10, 43.33 and 76.67 ms.
            self.assertEqual((record['cameras'], record['length']), (['left_wrist', 'right_wrist'], 3))
            state = staged_state(destination)
            self.assertEqual(state.shape, (3, 26))
            # The exported poses (source pose composed with the mount) are what is interpolated:
            # positions linearly between (1.1, 2, 3) and (4, 2.1, 3), rotation by slerp from 0 to 90 degrees.
            for weight, row in zip(np.array([10, 10 + 100/3, 10 + 200/3]) / 100, state):
                position = (1 - weight) * np.array([1.1, 2, 3]) + weight * np.array([4, 2.1, 3])
                np.testing.assert_allclose(row[:12], [*position, *rotation_z(90 * weight)], atol=1e-6)
                np.testing.assert_allclose(row[13:25], [1 + .1*np.cos(np.pi/6), 2 + .1*np.sin(np.pi/6), 3,
                                                        *rotation_z(30)], atol=1e-6)
            # Only the gripper samples are clipped to [0,1] (-.002 -> 0, .105 -> 1), then interpolated.
            np.testing.assert_allclose(state[:, 12], [.2, (10 + 100/3)/50, 1 - .5*(80/3)/50], atol=1e-6)
            np.testing.assert_allclose(state[:, 25], [30/80, (190/3)/80, 1 - .5*(50/3)/60], atol=1e-6)
            self.assertEqual(pq.read_table(destination / 'frames.parquet')['task'].to_pylist(), ['clean bowl'] * 3)
            # Nearest source frames: left 5, 45, 85 ms; the slower right camera repeats its 66 ms frame.
            np.testing.assert_allclose(video_means(destination / 'left_wrist.mp4'), [0, 80, 160], atol=3)
            np.testing.assert_allclose(video_means(destination / 'right_wrist.mp4'), [0, 40, 40], atol=3)

    def test_gaps_and_disjoint_streams_are_dropped_with_a_reason(self):
        still = lambda offsets: [(t, pose()) for t in offsets]
        open_ = lambda offsets: [(t, NS(value=.0515)) for t in offsets]
        states = {**{topic: still(FRAME[:4]) for topic in EEF.values()},
                  **{topic: open_(FRAME[:4]) for topic in GRIPPER.values()}}
        cameras = {topic: FRAME[:4] for topic in CAMERA.values()}
        # Frame times spanning a little over twice the RGB limit.
        span = [frame_ns(k) for k in range(2 * sync.FRAME_GAP_NS * sync.FPS // sync.NS_PER_SECOND + 4)]
        long = {**{topic: still(FRAME[:6]) for topic in EEF.values()}, **{topic: open_(FRAME[:6]) for topic in GRIPPER.values()}}
        episodes = {
            'good': on_grid(4),
            # Two pose samples farther apart than the state limit, with frame times between them.
            'state_gap': episode({**long, EEF['right']: still([0, sync.STATE_GAP_NS + 50*MS])},
                                 {topic: FRAME[:6] for topic in cameras}),
            # The left camera has only the first and last frame: those midway are beyond the RGB limit of both.
            'rgb_gap': episode({**{topic: still(span) for topic in EEF.values()},
                                **{topic: open_(span) for topic in GRIPPER.values()}},
                               {**{topic: span for topic in cameras}, CAMERA['left']: [span[0], span[-1]]}, step=10),
            # The right camera starts after the state ends.
            'disjoint': episode(states, {**cameras, CAMERA['right']: [FRAME[5], FRAME[6]]}),
        }
        expected = {'state_gap': 'sampling_gap', 'rgb_gap': 'sampling_gap', 'disjoint': 'no_common_interval'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in episodes:
                mcap(root, name)
            ids = {item['path'].parent.parent.name: episode_id for episode_id, item in script.discover(root / 'raw')}
            with reading(lambda path: episodes[path.parent.parent.name]):
                for name, reason in expected.items():
                    with self.assertRaises(SourceDataError) as caught:
                        script.convert_episode({'path': root / 'raw' / name / '1/1.mcap', 'instruction': name},
                                               root / 'direct' / name)
                    self.assertEqual(caught.exception.reason, reason, name)
                convert('realomin', root / 'raw', root / 'out')
            exported = read_export(root / 'out')
            self.assertEqual([e['episode_id'] for e in exported], [ids['good']])
            self.assertEqual((exported[0]['length'], exported[0]['tasks']), (4, ['good'] * 4))
            np.testing.assert_allclose(exported[0]['left_gripper'], [0, .1, .2, .3], atol=1e-6)
            work = root / 'out.work'
            summary = json.loads((work / 'conversion_summary.json').read_text())
            self.assertEqual((summary['planned'], summary['exported'], summary['dropped'], summary['failed']), (4, 1, 3, 0))
            self.assertEqual(summary['drop_reasons'], {'no_common_interval': 1, 'sampling_gap': 2})
            for name, reason in expected.items():
                record = json.loads((work / 'records' / f'{ids[name]}.json').read_text())
                self.assertEqual((record['outcome'], record['reason'], record['reasons']), ('skipped', reason, [reason]))
                self.assertTrue(record['detail'].strip())
            self.assertEqual(episode_errors(root / 'out'), '')

    def test_other_camera_sets_are_skipped_as_unsupported_layout(self):
        self.assertEqual(script.SOURCE.camera_sets, (('left_wrist', 'right_wrist'),))
        episodes = {'both': on_grid(3), 'left_only': on_grid(3, cameras=(CAMERA['left'],)),
                    'right_only': on_grid(3, cameras=(CAMERA['right'],))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in episodes:
                mcap(root, name)
            ids = {item['path'].parent.parent.name: episode_id for episode_id, item in script.discover(root / 'raw')}
            with reading(lambda path: episodes[path.parent.parent.name]):
                convert('realomin', root / 'raw', root / 'out')
            self.assertEqual(list(datasets(root / 'out')), ['left_wrist+right_wrist_640x480'])
            exported, = read_export(root / 'out')
            self.assertEqual((exported['episode_id'], exported['cameras']), (ids['both'], ['left_wrist', 'right_wrist']))
            # Samples sit on the frame times, so they pass through unchanged.
            np.testing.assert_allclose(exported['left_eef'][:, 0], np.arange(3) + script.DEFAULT_EEF_OFFSET[0], atol=1e-6)
            for camera in exported['cameras']:
                path, first = exported['videos'][camera]
                np.testing.assert_allclose(video_means(path, first, 3), [0, 40, 80], atol=3)
            work = root / 'out.work'
            summary = json.loads((work / 'conversion_summary.json').read_text())
            self.assertEqual((summary['exported'], summary['dropped'], summary['failed']), (1, 2, 0))
            self.assertEqual(summary['drop_reasons'], {'unsupported_layout': 2})
            for name in ('left_only', 'right_only'):
                record = json.loads((work / 'records' / f'{ids[name]}.json').read_text())
                self.assertEqual((record['outcome'], record['reason']), ('skipped', 'unsupported_layout'))
                self.assertIn(name.replace('only', 'wrist'), record['detail'])
            self.assertEqual(episode_errors(root / 'out'), '')

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
                records = list(script.messages(path, [topic]))
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
                assert_conversion_fails(self, 'Conversion failed', 'realomin', root/'raw', root/'out', eef_offset=[0., 0., 0.])
            self.assertIn('bad sample', next((root/'out.work/errors').glob('*.err')).read_text())
            self.assertFalse((root/'out').exists())

    def test_parallel_adapter_keeps_calibration_and_resumes_native_mcap(self):
        from loom_data.convert import job as parallel
        from loom_data.convert.config import make_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mcap(root, 'clean_bowl')
            config = make_config('realomin', root/'raw', eef_offset=[.2, 0, 0], tool_rpy=[0, 0, 0], instruction='Clean')
            parallel.prepare(config, root/'out')
            with reading(on_grid(3)):
                self.assertEqual(parallel.worker(root/'out', 0, 1), 0)
            parallel.finalize(root/'out')
            item, = read_export(root/'out')
            # Source positions (k, 2, 3) with the configured mount offset along the unrotated x axis.
            np.testing.assert_allclose(item['left_eef'][:, :3], [[.2, 2, 3], [1.2, 2, 3], [2.2, 2, 3]], atol=1e-6)
            self.assertEqual(item['tasks'], ['Clean'] * 3)
            parallel.prepare(config, root/'out')
            with patch.object(script, 'convert_episode', side_effect=AssertionError('must resume')):
                self.assertEqual(parallel.worker(root/'out', 0, 1), 0)
            with self.assertRaises(ValueError):
                make_config('realomin', root/'raw', gripper_closed=0, gripper_open=0, eef_offset=[0., 0., 0.])
            with self.assertRaises(ValueError):
                make_config('realomin', root/'raw', left_tool_rpy=[0, 0, 0], eef_offset=[0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
