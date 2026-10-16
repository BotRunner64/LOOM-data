"""Data drops, stationary-base recovery and failure/resume boundaries."""
import copy
import errno
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from scripts.convert import agibotworld_beta as beta, parallel, realomin
from scripts.convert.parallel_adapters import discover as discover_plan, make_config
from scripts.convert.source_policy import SourceDataError
from test_convert_agibotworld_beta import make_source


def add_episode(root, index):
    for folder in ('proprio_stats', 'observations'):
        shutil.copytree(root/folder/'694/123', root/folder/f'694/{index}')
    path = root/'task_info/task_694.json'
    rows = json.loads(path.read_text())
    row = copy.deepcopy(rows[0]); row['episode_id'] = index
    rows.append(row); path.write_text(json.dumps(rows))


def zero_base(path, commands=True):
    with h5py.File(path, 'a') as h:
        h['state/robot/position'][:] = 0
        h['state/robot/orientation'][:] = 0
        if commands:
            h['action/robot/velocity'] = np.zeros((3, 2))


class BaseRecoveryTests(unittest.TestCase):
    def test_zero_base_uses_identity_only_with_complete_stationary_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_source(Path(temporary)); zero_base(path)
            _, fields, repairs = beta.read_state(path, with_repairs=True)
            self.assertEqual(repairs, ['stationary_base_identity'])
            np.testing.assert_array_equal(fields['robot/orientation'], [[0,0,0,1]]*3)
            np.testing.assert_allclose(beta.poses(fields, (np.eye(4), np.eye(4)))['left_eef'],
                                       [[0,0,0,1,0,0,0,1,0]]*3)
            # Recovery changes only exported arrays, never the raw HDF5.
            with h5py.File(path) as h:
                self.assertFalse(np.asarray(h['state/robot/orientation']).any())

    def test_leading_and_trailing_gaps_keep_the_recorded_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_source(Path(temporary)); zero_base(path)
            with h5py.File(path, 'a') as h:
                h['state/robot/position'][1] = [2,3,0]
                h['state/robot/orientation'][1] = [0,0,np.sqrt(.5),np.sqrt(.5)]
            _, fields, repairs = beta.read_state(path, with_repairs=True)
            self.assertEqual(repairs, ['stationary_base_fill'])
            np.testing.assert_allclose(fields['robot/position'], [[2,3,0]]*3)
            np.testing.assert_allclose(fields['robot/orientation'], [[0,0,np.sqrt(.5),np.sqrt(.5)]]*3)

    def test_missing_moving_nonfinite_and_incomplete_commands_are_rejected(self):
        for case in ('absent', 'moving', 'nan', 'short'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                path = make_source(Path(temporary)); zero_base(path, commands=False)
                with h5py.File(path, 'a') as h:
                    if case != 'absent':
                        commands = np.zeros((2 if case=='short' else 3,2))
                        if case=='moving':commands[1,0]=.01
                        if case=='nan':commands[1,0]=float('nan')
                        h['action/robot/velocity']=commands
                with self.assertRaises(SourceDataError):beta.read_state(path)

    def test_displaced_zero_quaternion_and_flange_errors_are_not_repaired(self):
        for kind in ('base', 'flange'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path=make_source(Path(temporary))
                with h5py.File(path,'a') as h:
                    h[f'state/{"robot" if kind=="base" else "end"}/orientation'][:]=0
                    h['action/robot/velocity']=np.zeros((3,2))
                _,fields=beta.read_state(path)
                with self.assertRaises(SourceDataError):beta.poses(fields,(np.eye(4),np.eye(4)))

    def test_action_end_length_does_not_affect_state_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=make_source(root/'raw')
            with h5py.File(path,'a') as h:h['action/end/position']=np.zeros((1,2,3))
            beta.convert(root/'raw',root/'out')
            self.assertEqual(len((root/'out/episodes.jsonl').read_text().splitlines()),1)


class NativeDropTests(unittest.TestCase):
    def test_hdf5_corruption_drops_but_unclassified_io_errors_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);state=make_source(root/'raw')
            state.write_bytes(b'not an HDF5 file')
            beta.convert(root/'raw',root/'out')
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual(s['drop_reasons'],{'invalid_hdf5':1})
        with patch('h5py.File',side_effect=OSError('storage read failed without errno')):
            with self.assertRaises(OSError) as error:beta.read_state(Path('source.h5'))
            self.assertNotIsInstance(error.exception,SourceDataError)

    def test_identical_annotations_are_deduplicated_and_conflicts_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw');add_episode(root/'raw',124)
            path=root/'raw/task_info/task_694.json';rows=json.loads(path.read_text())
            rows.append(copy.deepcopy(rows[0]))
            conflict=copy.deepcopy(rows[1]);conflict['task_name']='Conflicting task';rows.append(conflict)
            path.write_text(json.dumps(rows))
            beta.convert(root/'raw',root/'out')
            summary=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual((summary['planned'],summary['exported'],summary['repaired'],summary['dropped']),(2,1,1,1))
            self.assertEqual(summary['repair_reasons'],{'deduplicated_annotation':1})
            self.assertEqual(summary['drop_reasons'],{'conflicting_annotation':1})
            record=json.loads((root/'out/episodes.jsonl').read_text())
            self.assertEqual(set(record),{'episode_id','cameras','instructions'})
            plan=json.loads((root/'out.work/plan.json').read_text())
            self.assertTrue(all('annotation' not in t for t in plan['tasks']))

    def test_missing_files_are_recorded_without_running_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            (root/'raw/task_info/task_694.json').unlink()
            shutil.rmtree(root/'raw/observations')
            with patch.object(beta,'convert_episode',side_effect=AssertionError('must preflight-drop')):
                beta.convert(root/'raw',root/'out')
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual(s['dropped'],1);self.assertEqual(s['drop_fraction'],1.)
            self.assertEqual(s['drop_reasons'],{'missing_annotation':1,'missing_rgb':1})
            self.assertEqual((root/'out/episodes.jsonl').read_text(),'')

    def test_bad_episodes_drop_and_good_repaired_episode_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);state=make_source(root/'raw');zero_base(state)
            for ep in (124,125,126):add_episode(root/'raw',ep)
            with h5py.File(root/'raw/proprio_stats/694/124/proprio_stats.h5','a') as h:
                del h['state/effector/position'];h['state/effector/position']=np.zeros((3,12))
            with h5py.File(root/'raw/proprio_stats/694/125/proprio_stats.h5','a') as h:
                h['timestamp'][2]=h['timestamp'][0]
            # A real MP4/HDF5 frame-count mismatch, not an injected generic error.
            with h5py.File(root/'raw/proprio_stats/694/126/proprio_stats.h5','a') as h:
                for key in ('timestamp','state/end/position','state/end/orientation',
                            'state/effector/position','state/robot/position','state/robot/orientation',
                            'action/robot/velocity'):
                    values=np.asarray(h[key])[:2];del h[key];h[key]=values
            ann=root/'raw/task_info/task_694.json';rows=json.loads(ann.read_text())
            for row in rows:row['label_info']={}
            ann.write_text(json.dumps(rows))
            config=make_config('agibotworld_beta',root/'raw');store=parallel.prepare(config,root/'out')
            for rank in (0,1):self.assertEqual(parallel.worker(root/'out',rank,2),0)
            parallel.prepare(config,root/'out')
            with patch.object(beta,'convert_episode',side_effect=AssertionError('committed/drop records must resume')):
                self.assertEqual(parallel.worker(root/'out',0,1),0)
            parallel.finalize(root/'out')
            self.assertTrue(store.published())
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual((s['exported'],s['repaired'],s['dropped'],s['failed']),(1,1,3,0))
            self.assertEqual(s['drop_reasons'],{'invalid_timestamp':1,'unsupported_effector':1,'video_frame_count':1})
            self.assertEqual(list((root/'out.work/errors').iterdir()),[])

    def test_operational_and_programming_errors_remain_retryable(self):
        for error in (ValueError('unexpected converter bug'),OSError(errno.ENOSPC,'disk full'),ImportError('missing codec')):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);make_source(root/'raw');config=make_config('agibotworld_beta',root/'raw')
                store=parallel.prepare(config,root/'out')
                with patch.object(beta,'convert_episode',side_effect=error):
                    self.assertEqual(parallel.worker(root/'out',0,1),1)
                self.assertIsNone(store.state(store.ids[0]))
                store.audit();store.write_summary()
                summary=json.loads((root/'out.work/conversion_summary.json').read_text())
                self.assertEqual((summary['failed'],summary['dropped']),(1,0))
                self.assertFalse((root/'out').exists())
                self.assertEqual(parallel.worker(root/'out',0,1),0)
                parallel.finalize(root/'out')

    def test_finalize_omits_recorded_failures_and_remains_idempotent(self):
        for error in (ValueError('unexpected converter bug'),OSError(errno.ENOSPC,'disk full'),ImportError('missing codec')):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);make_source(root/'raw');add_episode(root/'raw',124)
                config=make_config('agibotworld_beta',root/'raw');store=parallel.prepare(config,root/'out')
                original=beta.convert_episode
                def convert(source,destination,*args,**kwargs):
                    if destination.name.endswith('_124'):raise error
                    return original(source,destination,*args,**kwargs)
                with patch.object(beta,'convert_episode',side_effect=convert):
                    self.assertEqual(parallel.worker(root/'out',0,1),1)
                failed_id='task_694_episode_124'
                self.assertIsNone(store.state(failed_id))
                error_path=store.work/'errors'/f'{failed_id}.err'
                self.assertTrue(error_path.is_file())
                parallel.finalize(root/'out')
                self.assertTrue(store.published())
                self.assertEqual([json.loads(line)['episode_id'] for line in
                                  (root/'out/episodes.jsonl').read_text().splitlines()],['task_694_episode_123'])
                summary=json.loads((store.work/'conversion_summary.json').read_text())
                self.assertEqual((summary['exported'],summary['failed'],summary['dropped'],summary['unfinished']),(1,1,0,0))
                parallel.prepare(config,root/'out')
                with patch.object(beta,'convert_episode',side_effect=AssertionError('already published')):
                    self.assertEqual(parallel.worker(root/'out',0,1),0)
                parallel.finalize(root/'out')
                self.assertTrue(error_path.is_file())

    def test_finalize_still_rejects_unfinished_episode_without_error_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            parallel.prepare(make_config('agibotworld_beta',root/'raw'),root/'out')
            with self.assertRaisesRegex(parallel.StateError,'unfinished episodes'):parallel.finalize(root/'out')
            self.assertFalse((root/'out').exists())

    def test_realomin_finalize_accepts_recorded_runtime_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'raw/clean_bowl/1/1.mcap'
            source.parent.mkdir(parents=True);source.touch()
            store=parallel.prepare(make_config('realomin',root/'raw'),root/'out')
            with patch.object(realomin,'convert_episode',side_effect=OSError(errno.ENOSPC,'disk full')):
                self.assertEqual(parallel.worker(root/'out',0,1),1)
            parallel.finalize(root/'out')
            self.assertTrue(store.published())
            self.assertEqual((root/'out/episodes.jsonl').read_text(),'')
            summary=json.loads((store.work/'conversion_summary.json').read_text())
            self.assertEqual((summary['failed'],summary['dropped'],summary['unfinished']),(1,0,0))
            self.assertTrue((store.work/'errors'/f'{store.ids[0]}.err').is_file())

    def test_old_native_policy_cannot_be_loaded_by_workers_or_finalize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            store=parallel.prepare(make_config('agibotworld_beta',root/'raw'),root/'out')
            plan=copy.deepcopy(store.plan);plan['config']['params'].pop('data_policy_version')
            plan['id']=parallel.digest({k:v for k,v in plan.items() if k!='id'})
            parallel.atomic_json(store.work/'plan.json',plan)
            with self.assertRaisesRegex(parallel.StateError,'policy changed'):parallel.worker(root/'out',0,1)
            with self.assertRaisesRegex(parallel.StateError,'policy changed'):parallel.finalize(root/'out')

    def test_interrupted_drop_record_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root/'raw')
            shutil.rmtree(root/'raw/observations')
            config=make_config('agibotworld_beta',root/'raw');store=parallel.prepare(config,root/'out')
            original=parallel.atomic_json
            def interrupted(path,value):
                original(path,value)
                if value.get('outcome')=='skipped':raise KeyboardInterrupt()
            with patch.object(parallel,'atomic_json',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):parallel.worker(root/'out',0,1)
            parallel.prepare(config,root/'out')
            with patch.object(beta,'convert_episode',side_effect=AssertionError('drop is committed')):
                self.assertEqual(parallel.worker(root/'out',0,3),0)
            parallel.finalize(root/'out');self.assertTrue(store.published())

    def test_native_local_processes_publish_drops_and_resume_with_new_worker_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);state=make_source(root/'raw');zero_base(state)
            add_episode(root/'raw',124)
            with h5py.File(root/'raw/proprio_stats/694/124/proprio_stats.h5','a') as h:
                h['state/end/position'][0,0,0]=float('nan')
            config=make_config('agibotworld_beta',root/'raw')
            parallel.run(config,root/'out',workers=2)
            parallel.run(config,root/'out',workers=3)
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual((s['exported'],s['repaired'],s['dropped']),(1,1,1))
            self.assertEqual(s['drop_reasons'],{'invalid_state':1})

    def test_limit_stops_before_reading_later_task_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);make_source(root)
            (root/'proprio_stats/999/1').mkdir(parents=True)
            original=beta.load_annotations
            def load(source,task):
                if task==999:raise AssertionError('limit must stop discovery')
                return original(source,task)
            with patch.object(beta,'load_annotations',side_effect=load):
                self.assertEqual(len(beta.discover(root,limit=1)),1)
                tasks,_=discover_plan(make_config('agibotworld_beta',root,limit_episodes=1))
                self.assertEqual(len(tasks),1)

    def test_realomin_corrupt_mcap_drops_and_stage2_keeps_scene_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'raw/Clutter Tidy-Up [Stage2]/00001/00001.mcap'
            source.parent.mkdir(parents=True);source.write_bytes(b'not an MCAP')
            self.assertEqual(realomin.discover(root/'raw')[0][1]['instruction'],'Clutter Tidy-Up [Stage2]')
            realomin.convert(root/'raw',root/'out')
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual(s['drop_reasons'],{'invalid_mcap':1})

    def test_realomin_missing_side_is_a_data_drop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'raw/clean_bowl/1/1.mcap'
            source.parent.mkdir(parents=True);source.touch()
            with patch.object(realomin,'messages',return_value=iter(())),patch.object(realomin,'VideoWriter') as video:
                realomin.convert(root/'raw',root/'out')
            video.assert_not_called()
            s=json.loads((root/'out.work/conversion_summary.json').read_text())
            self.assertEqual(s['drop_reasons'],{'missing_state_stream':1})


if __name__=='__main__':unittest.main()
