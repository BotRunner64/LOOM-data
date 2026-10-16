import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import h5py
import numpy as np
import pyarrow.parquet as pq

from scripts.convert import agibotworld_beta as script
from scripts.convert.export_common import tool_transform
from scripts.convert.export_validation import validate_episode


def make_source(root):
    state = root/'proprio_stats/694/123/proprio_stats.h5'
    state.parent.mkdir(parents=True)
    origin = 1_739_382_608_553_487_000
    with h5py.File(state, 'w') as h:
        h['timestamp'] = np.array([origin, origin+31_234_567, origin+68_765_432], dtype='int64')
        h['state/end/position'] = np.zeros((3, 2, 3))
        h['state/end/orientation'] = np.tile([0., 0., 0., 1.], (3, 2, 1))
        h['state/effector/position'] = np.array([[35., 120.], [77.5, 77.5], [120., 35.]])
        h['state/robot/position'] = np.tile([1., 2., 0.], (3, 1))
        h['state/robot/orientation'] = np.tile([0., 0., 0., 1.], (3, 1))
    video = root/'observations/694/123/videos/head_color.mp4'
    video.parent.mkdir(parents=True)
    with av.open(str(video), 'w') as writer:
        stream = writer.add_stream('libx264', rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.thread_count = 1
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((32,32,3), i*80, dtype='uint8'), format='rgb24')
            for packet in stream.encode(frame):writer.mux(packet)
        for packet in stream.encode():writer.mux(packet)
    annotation = {'episode_id': 123, 'task_name': 'Twist cap', 'label_info': {'action_config': [
        {'start_frame': 0, 'end_frame': 1, 'action_text': 'Pick'},
        {'start_frame': 1, 'end_frame': 2, 'action_text': 'Twist'},
    ]}}
    (root/'task_info').mkdir()
    (root/'task_info/task_694.json').write_text(json.dumps([annotation]))
    return state


class BetaMappingTests(unittest.TestCase):
    def test_real_flange_poses_match_independent_official_fk_in_base(self):
        samples = json.loads((Path(__file__).parent/'fixtures/agibotworld_beta_pose_samples.json').read_text())['samples']
        fields = {name: np.array([s[name] for s in samples]) for name in (
            'end/position', 'end/orientation', 'robot/position', 'robot/orientation')}
        # Identity mobile pose and identity tool isolate the recorded reference
        # frame; independent FK includes measured waist and arm joint values.
        fields['robot/position'][:] = 0
        fields['robot/orientation'][:] = [0, 0, 0, 1]
        poses = script.poses(fields, (np.eye(4), np.eye(4)))
        for k, side in enumerate(('left', 'right')):
            for sample, pose in zip(samples, poses[f'{side}_eef']):
                expected = np.array(sample['independent_base_flange'][k])
                np.testing.assert_allclose(pose[:3], expected[:3,3], atol=5e-6, rtol=0)
                np.testing.assert_allclose(pose[3:6], expected[:3,0], atol=3e-5, rtol=0)
                np.testing.assert_allclose(pose[6:9], expected[:3,1], atol=3e-5, rtol=0)

    def test_mobile_base_and_tcp_compose_in_correct_order(self):
        fields = {'end/position': np.array([[[1,0,0], [1,0,0]]]),
                  'end/orientation': np.tile([0,0,0,1], (1,2,1)),
                  'robot/position': np.array([[2,3,0]]),
                  'robot/orientation': np.array([[0,0,math.sqrt(.5),math.sqrt(.5)]])}
        transforms = (tool_transform([0,0,.2], script.DEFAULT_LEFT_TOOL_RPY),
                      tool_transform([0,0,.2], script.DEFAULT_RIGHT_TOOL_RPY))
        poses = script.poses(fields, transforms)
        for side in ('left', 'right'):
            np.testing.assert_allclose(poses[f'{side}_eef'][0][:3], [2,4,.2], atol=1e-12)
        np.testing.assert_allclose(poses['left_eef'][0][3:6], [0,0,1], atol=1e-12)
        # Mounted palm backs are opposite in source flange coordinates.
        for transform, sign in zip(transforms, (1,-1)):
            np.testing.assert_allclose(transform[:3,2], [sign,0,0], atol=1e-12)
            np.testing.assert_allclose(np.cross(transform[:3,2],transform[:3,0]),transform[:3,1])

    def test_segment_boundaries_use_native_times_fill_gaps_and_reject_overlap(self):
        row = {'task_name': 'Whole task', 'label_info': {'action_config': [
            {'start_frame': 1, 'end_frame': 3, 'action_text': 'Pick'},
        ]}}
        times = [1000, 1031, 1068, 1102]
        self.assertEqual(script.instructions(row,times), [
            {'start_ns':0,'end_ns':31,'text':'Whole task'},
            {'start_ns':31,'end_ns':102,'text':'Pick'},
        ])
        row['label_info']['action_config'][0]['end_frame']=4
        self.assertEqual(script.instructions(row,times)[-1]['end_ns'],102)
        row['label_info']['action_config'].append({'start_frame':2,'end_frame':3,'action_text':'Overlap'})
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            script.instructions(row,times)

    def test_gripper_reading_increases_on_closure(self):
        self.assertEqual(script.normalized_gripper(35,120,35),1)
        self.assertEqual(script.normalized_gripper(77.5,120,35),.5)
        self.assertEqual(script.normalized_gripper(120,120,35),0)
        self.assertEqual(script.normalized_gripper(121,120,35),0)
        with self.assertRaises(ValueError):script.normalized_gripper(float('nan'),120,35)


class BetaExportTests(unittest.TestCase):
    def test_real_media_schema_video_count_and_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            script.convert(root/'raw',root/'out')
            record=json.loads((root/'out/episodes.jsonl').read_text())
            destination=root/'out/episodes'/record['episode_id']
            validate_episode(destination,record)
            for kind in ('state/left_eef','rgb/head'):
                self.assertEqual(pq.read_table(destination/f'{kind}.parquet')['timestamp_ns'].to_pylist(),[0,31_234_567,68_765_432])
            self.assertEqual(pq.read_table(destination/'state/left_gripper.parquet')['openness'].to_pylist(),[1.,.5,0.])
            self.assertEqual(record['instructions'][0]['end_ns'],31_234_567)

    def test_dexhands_invalid_times_and_nonfinite_state_are_rejected(self):
        for invalid in ('dexhand','timestamp','nonfinite'):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as temporary:
                state=make_source(Path(temporary))
                with h5py.File(state,'a') as h:
                    if invalid=='dexhand':
                        del h['state/effector/position'];h['state/effector/position']=np.zeros((3,12))
                    elif invalid=='timestamp':h['timestamp'][2]=h['timestamp'][0]
                    else:h['state/end/position'][0,0,0]=float('nan')
                with self.assertRaises(ValueError):script.read_state(state)

    def test_video_failure_does_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            with patch.object(script,'transcode_video',side_effect=ValueError('frame mismatch')):
                with self.assertRaisesRegex(RuntimeError,'Conversion failed'):script.convert(root/'raw',root/'out')
            self.assertIn('frame mismatch', (root/'out.work/errors/task_694_episode_123.err').read_text())
            self.assertFalse((root/'out').exists())
            self.assertFalse(list(root.glob('.out-*')))

    def test_parallel_adapter_publishes_and_resumes(self):
        from scripts.convert import parallel
        from scripts.convert.parallel_adapters import make_config
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            config=make_config('agibotworld_beta',root/'raw')
            parallel.prepare(config,root/'out')
            self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            parallel.prepare(config,root/'out')
            with patch.object(script,'convert_episode',side_effect=AssertionError('must skip committed episode')):
                self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            record=json.loads((root/'out/episodes.jsonl').read_text())
            self.assertEqual(record['episode_id'],'task_694_episode_123')


if __name__=='__main__':unittest.main()
