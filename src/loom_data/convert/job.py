"""The conversion job: prepare a plan, run statically assigned workers, then publish.

Every conversion goes through this job, whether one local worker or a Slurm
array. Episodes are committed one by one, so an interrupted job resumes where
it stopped. State lives in a work directory beside the output:

    plan.json          frozen config, tasks and input fingerprints
    records/<id>.json  one committed or skipped episode
    errors/<id>.err    last failure of an unfinished episode
    dataset/           committed episodes, renamed to the output by finalize
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback

from loom_data import contract, paths, resumable, sources
from loom_data.convert import config as configuration
from loom_data.convert.budget import sample_tasks
from loom_data.resumable import (
    StateError, atomic_json, atomic_text, check_inputs, configure_worker, digest, encoded, input_stamp,
    no_progress, progress, read_json, require_single_process, run_cli, run_workers, sync_directory,
    worker_diagnostics, worker_rank,
)


def work_directory(output):
    return resumable.work_directory(output, '.work')


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('version') != 1:
        raise StateError('Unsupported or damaged plan')
    if plan.get('id') != digest({k: v for k, v in plan.items() if k != 'id'}):
        raise StateError('Plan checksum mismatch')
    if plan.get('config', {}).get('params', {}).get('export_contract_version') != contract.VERSION:
        raise StateError('Incompatible export contract: old TCP plans require a new output and re-export')
    if plan.get('config', {}).get('storage_budget') is not None and not isinstance(plan.get('storage_sampling'), dict):
        raise StateError('Incompatible storage budget plan; prepare a new output directory')
    ids = []
    for task in plan['tasks']:
        if not task['episode_ids']:
            raise StateError('Empty scheduling task')
        ids.extend(task['episode_ids'])
    if (not ids or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids)):
        raise StateError('Episode IDs must be unique safe identifiers')


class Store:
    def __init__(self, output, plan):
        self.output = Path(output).resolve()
        self.work = work_directory(self.output)
        self.plan = plan
        validate_plan(plan)
        if plan['output'] != str(self.output):
            raise StateError('Plan belongs to a different output')
        self.ids = [i for task in plan['tasks'] for i in task['episode_ids']]
        self.id_set = set(self.ids)
        self.dataset = self.work / 'dataset'
        policy = sources.get(plan['config']['dataset']).policy
        self.skips = policy != 'strict'   # episodes may be left out with a recorded reason
        self.drops = policy == 'drop'     # ... because their source data is bad
        if self.drops and plan['config']['params'].get('data_policy_version') != 1:
            raise StateError('Source data policy changed; use a new output directory')

    @classmethod
    def load(cls, output):
        output = Path(output).resolve()
        return cls(output, read_json(work_directory(output) / 'plan.json'))

    def state(self, episode_id, *, published=False):
        if episode_id not in self.id_set:
            raise StateError(f'Unknown episode: {episode_id}')
        path = self.work / 'records' / f'{episode_id}.json'
        directory = (self.output if published else self.dataset) / 'episodes' / episode_id
        if path.is_symlink():
            raise StateError(f'{episode_id}: symlink recovery record')
        if not path.exists():
            if directory.exists() or directory.is_symlink():
                raise StateError(f'{episode_id}: episode exists without record')
            return None
        record = read_json(path)
        if (not isinstance(record, dict) or record.get('plan_id') != self.plan['id']
                or record.get('episode_id') != episode_id):
            raise StateError(f'{episode_id}: recovery record belongs to a different plan')
        try:
            if record.get('outcome') == 'skipped':
                if (not self.skips
                        or not isinstance(record.get('reason'), str) or not record['reason'].strip()
                        or directory.exists() or directory.is_symlink()):
                    raise ValueError('invalid skip record or unexpected episode directory')
                if self.drops:
                    reasons = record.get('reasons')
                    if (not isinstance(reasons, list) or not reasons or record['reason'] not in reasons
                            or any(not isinstance(r, str) or not r.strip() for r in reasons)
                            or len(set(reasons)) != len(reasons) or not isinstance(record.get('detail'), str)):
                        raise ValueError('invalid data-drop reasons')
                return record
            if record.get('outcome') != 'exported':
                raise ValueError('invalid record outcome')
            contract.validate_record(record.get('manifest'), episode_id)
            validate_conversion(record.get('conversion', {}))
            if (not isinstance(record.get('files'), dict) or not record['files']
                    or any(type(n) is not int or n <= 0 for n in record['files'].values())):
                raise ValueError('invalid recorded file sizes')
            if not directory.exists() and not directory.is_symlink():
                return None
            if contract.file_sizes(directory, record['manifest']) != record['files']:
                raise ValueError('committed files changed')
        except (ValueError, OSError) as exc:
            raise StateError(f'{episode_id}: damaged recovery state: {exc}') from exc
        return record

    def audit(self, *, published=False, complete=False, progress=no_progress):
        records = self.work / 'records'
        if records.is_symlink():
            raise StateError('Symlink records directory')
        record_names = {p.name for p in records.iterdir()} if records.exists() else set()
        if record_names - {f'{i}.json' for i in self.ids}:
            raise StateError('Unexpected files in records directory')
        record_ids = {name[:-5] for name in record_names}
        dataset = self.output if published else self.dataset
        episodes = dataset / 'episodes'
        if dataset.is_symlink() or episodes.is_symlink():
            raise StateError('Symlink dataset/episodes directory')
        actual = {p.name for p in episodes.iterdir()} if episodes.exists() else set()
        if actual - self.id_set:
            raise StateError('Unexpected episode directories')
        orphaned = actual - record_ids
        if orphaned:
            raise StateError(f'{min(orphaned)}: episode exists without record')
        exported, skipped, missing = [], [], []
        drop_reasons, repair_reasons = Counter(), Counter()
        repaired = 0
        def check(episode_id):
            return self.state(episode_id, published=published)

        # Bound both shared-filesystem concurrency and queued work. No workers
        # may be writing this output during audit; the directory lists are stable.
        with progress('Auditing episodes', len(self.ids), unit='episode') as bar, \
                ThreadPoolExecutor(max_workers=8) as executor:
            ids = iter(self.ids)
            while batch := list(islice(ids, 256)):
                pending = [i for i in batch if i in record_ids]
                states = dict(zip(pending, executor.map(check, pending)))
                for episode_id in batch:
                    state = states.get(episode_id)
                    if state is None:
                        missing.append(episode_id)
                    elif state['outcome'] == 'exported':
                        exported.append(state['manifest'])
                        repairs = state.get('conversion', {}).get('repairs', [])
                        repaired += bool(repairs)
                        repair_reasons.update(repairs)
                    else:
                        skipped.append(episode_id)
                        drop_reasons.update(state.get('reasons', [state['reason']]))
                    bar.update()
        error_ids = {p.stem for p in (self.work / 'errors').glob('*.err') if p.is_file()}
        failed = sum(i in error_ids for i in missing)
        self.statistics = {'planned': len(self.ids), 'exported': len(exported), 'repaired': repaired,
                           'dropped': len(skipped), 'failed': failed, 'unfinished': len(missing) - failed,
                           'drop_fraction': len(skipped) / len(self.ids),
                           'repair_fraction': repaired / len(self.ids),
                           'drop_reasons': dict(sorted(drop_reasons.items())),
                           'repair_reasons': dict(sorted(repair_reasons.items()))}
        if complete and missing:
            unfinished = [i for i in missing if not (self.work / 'errors' / f'{i}.err').is_file()]
            if unfinished:
                raise StateError(f'{len(unfinished)} unfinished episodes, including {unfinished[:3]}')
        return exported, skipped, missing

    def published(self, progress=no_progress):
        if not self.output.exists():
            return False
        if self.dataset.exists():
            raise StateError('Both published output and working dataset exist')
        if {p.name for p in self.output.iterdir()} != {'episodes', 'episodes.jsonl'}:
            raise StateError('Existing output is not a published dataset')
        expected, _, _ = self.audit(published=True, complete=True, progress=progress)
        try:
            actual = [json.loads(line) for line in (self.output / 'episodes.jsonl').read_text().splitlines()]
        except (ValueError, OSError) as exc:
            raise StateError('Damaged published manifest') from exc
        if actual != expected:
            raise StateError('Published manifest differs from recovery records')
        return True

    def commit(self, episode_id, directory, manifest):
        if self.state(episode_id) is not None:
            raise StateError(f'{episode_id}: already committed')
        if manifest.get('episode_id') != episode_id or directory.name != episode_id:
            raise StateError('Conversion returned a different episode ID')
        manifest = dict(manifest)
        conversion = manifest.pop('_conversion', {})
        validate_conversion(conversion)
        sizes = contract.validate_episode(directory, manifest)
        # Flush data before the recovery record and directory commit become durable.
        for name in sizes:
            with (directory / name).open('rb') as handle:
                os.fsync(handle.fileno())
        for folder in (directory / 'state', directory / 'rgb', directory):
            sync_directory(folder)
        atomic_json(self.work / 'records' / f'{episode_id}.json',
                    {'plan_id': self.plan['id'], 'episode_id': episode_id,
                     'outcome': 'exported', 'manifest': manifest, 'files': sizes,
                     'conversion': conversion})
        destination = self.dataset / 'episodes' / episode_id
        if destination.exists():
            raise StateError(f'{episode_id}: destination appeared during conversion')
        directory.rename(destination)
        sync_directory(destination.parent)
        sync_directory(directory.parent)
        (self.work / 'errors' / f'{episode_id}.err').unlink(missing_ok=True)
        print(f'Committed {episode_id}', flush=True)

    def skip(self, episode_id, reason, *, detail='', reasons=None):
        if not self.skips:
            raise StateError('This dataset has no legitimate skip policy')
        if self.state(episode_id) is not None:
            raise StateError(f'{episode_id}: already completed')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A skip needs an explicit reason')
        reasons = list(dict.fromkeys(reasons or [reason]))
        if (reason not in reasons or any(not isinstance(r, str) or not r.strip() for r in reasons)
                or not isinstance(detail, str)):
            raise ValueError('Invalid data-drop reasons')
        atomic_json(self.work / 'records' / f'{episode_id}.json',
                    {'plan_id': self.plan['id'], 'episode_id': episode_id,
                     'outcome': 'skipped', 'reason': reason, 'reasons': reasons, 'detail': detail})
        (self.work / 'errors' / f'{episode_id}.err').unlink(missing_ok=True)
        print(f'Skipped {episode_id}: {reason}', flush=True)

    def write_summary(self):
        if self.drops:
            atomic_json(self.work / 'conversion_summary.json', self.statistics)
            s = self.statistics
            print(f"Conversion: {s['exported']} exported ({s['repaired']} repaired), "
                  f"{s['dropped']} dropped ({s['drop_fraction']:.2%}), "
                  f"{s['failed']} failed, {s['unfinished']} unfinished; "
                  f"report: {self.work / 'conversion_summary.json'}", flush=True)

    def failure(self, episode_id, source, exc):
        message = f'{episode_id}\nInput: {source}\n' + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(message, file=sys.stderr, flush=True)
        atomic_text(self.work / 'errors' / f'{episode_id}.err', message)


def validate_conversion(value):
    if not isinstance(value, dict):
        raise StateError('Invalid conversion metadata')
    repairs = value.get('repairs', [])
    if (not isinstance(repairs, list) or any(not isinstance(r, str) or not r.strip() for r in repairs)
            or len(set(repairs)) != len(repairs)):
        raise StateError('Invalid conversion repair reasons')

def prepare(config, output):
    require_single_process()
    output = Path(output).resolve()
    work = work_directory(output)
    source = Path(config['source'])
    if output == source or output.is_relative_to(source) or work.is_relative_to(source):
        raise ValueError('Output/work directory must be outside the input source')
    model_dir = config.get('model_dir')
    if model_dir and not Path(model_dir).is_dir():
        raise ValueError(f'Model directory does not exist: {model_dir}')
    if (work / 'plan.json').exists():
        store = Store.load(output)
        if store.plan['config'] != config:
            raise ValueError('Incompatible conversion config; use a new output')
    else:
        if output.exists():
            raise StateError('Output already exists without a matching resume plan')
        if work.exists() and any(p.name not in ('plan.json', 'logs') and not p.name.endswith('.tmp') for p in work.iterdir()):
            raise StateError('Work directory has data but no plan')
        tasks, shared = configuration.discover(config, progress)
        stamp_count = len(shared) + sum(len(task['inputs']) for task in tasks)
        shared_inputs = []
        with progress('Recording input metadata', stamp_count, unit='file') as bar:
            for path in shared:
                shared_inputs.append(input_stamp(path))
                bar.update()
            for task in tasks:
                stamps = []
                for path in task['inputs']:
                    stamps.append(input_stamp(path))
                    bar.update()
                task['inputs'] = stamps
        tasks, sampling = sample_tasks(config, tasks)
        plan = {'version': 1, 'config': config, 'output': str(output),
                'tasks': tasks, 'shared_inputs': shared_inputs}
        if sampling is not None:
            plan['storage_sampling'] = sampling
        print('Computing plan checksum and writing plan...', file=sys.stderr, flush=True)
        plan['id'] = digest(plan)
        store = Store(output, plan)
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / 'plan.json', plan)
    if sampling := store.plan.get('storage_sampling'):
        print(f'Storage sampling: {sampling["selected_episodes"]}/{sampling["candidate_episodes"]} episodes; '
              f'estimated {sampling["estimated_output_bytes"]}/{sampling["budget_bytes"]} bytes '
              f'(output/source ratio {sampling["estimate_ratio"]})', flush=True)
    stamps = list(store.plan['shared_inputs'])
    for task in store.plan['tasks']:
        stamps += task['inputs']
    check_inputs(stamps, progress)
    if store.published(progress):
        store.write_summary()
        print(f'Already complete: {output}', flush=True)
        return store
    for name in ('records', 'errors', 'logs', 'staging', 'dataset/episodes'):
        directory = work / name
        if directory.is_symlink():
            raise StateError(f'Symlink working directory: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
    # Only prepare may clean attempts and diagnostics, and only after previous workers have exited.
    for name in ('logs', 'staging'):
        for path in (work / name).iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    for directory in (work, work / 'records', work / 'errors', store.dataset):
        for path in directory.glob('.*.tmp'):
            path.unlink()
    exported, skipped, missing = store.audit(progress=progress)
    store.write_summary()
    total = len(store.ids)
    done = len(exported) + len(skipped)
    failed = sum((work / 'errors' / f'{episode_id}.err').is_file() for episode_id in missing)
    print(f'Progress: {done}/{total} ({done / total:.1%}) episodes done '
          f'({len(exported)} exported, {len(skipped)} skipped, '
          f'{failed} failed, {len(missing) - failed} unfinished)', flush=True)
    print(f'Prepared {len(store.plan["tasks"])} tasks / {len(store.ids)} episodes: {work}', flush=True)
    return store


def worker(output, worker_id, worker_count, *, report=None):
    report = report or (lambda message: None)
    report(f'Worker {worker_id}/{worker_count}: configuring dependencies')
    if worker_count < 1 or not 0 <= worker_id < worker_count:
        raise ValueError('Require worker_count > 0 and 0 <= worker_id < worker_count')
    configure_worker()
    report('Loading plan')
    store = Store.load(output)
    report('Checking shared inputs and published output')
    check_inputs(store.plan['shared_inputs'])
    if store.published():
        return 0
    report('Initializing source')
    source = sources.get(store.plan['config']['dataset'])(store.plan['config'])
    failures = 0
    for index, task in enumerate(store.plan['tasks']):
        if index % worker_count != worker_id:
            continue
        report(f'Task {index}: input={task["source"]}; episodes={task["episode_ids"]}')
        check_inputs(task['inputs'])
        pending = [i for i in task['episode_ids'] if store.state(i) is None]
        if not pending:
            continue
        with tempfile.TemporaryDirectory(prefix=f'w{worker_id}-', dir=store.work / 'staging') as temporary:
            def failed(episode_id, exc):
                nonlocal failures
                if isinstance(exc, StateError):
                    raise exc
                failures += 1
                report(f'Episode {episode_id} failed: {type(exc).__name__}: {exc}')
                store.failure(episode_id, task['source'], exc)
            try:
                source.process(task, pending, Path(temporary), store, failed)
            except StateError:
                raise
            except Exception as exc:
                # Archive-level I/O errors affect only episodes not yet committed.
                for episode_id in pending:
                    if store.state(episode_id) is None:
                        failed(episode_id, exc)
    return 1 if failures else 0


def finalize(output):
    require_single_process()
    store = Store.load(output)
    print('Auditing episodes...', file=sys.stderr, flush=True)
    if store.published(progress=progress):
        store.write_summary()
        print(f'Already published: {store.output}', flush=True)
        return
    try:
        manifest, _, _ = store.audit(complete=True, progress=progress)
    except StateError:
        if hasattr(store, 'statistics'):
            store.write_summary()
        raise
    store.write_summary()
    print(f'Writing episodes.jsonl ({len(manifest)} episodes)...', file=sys.stderr, flush=True)
    atomic_text(store.dataset / 'episodes.jsonl', ''.join(encoded(r) + '\n' for r in manifest))
    if store.output.exists():
        raise StateError('Output appeared before publication')
    print('Publishing dataset...', file=sys.stderr, flush=True)
    store.output.parent.mkdir(parents=True, exist_ok=True)
    store.dataset.rename(store.output)
    sync_directory(store.output.parent)
    sync_directory(store.work)
    print(f'Published {len(manifest)} episodes: {store.output}', flush=True)


def run_serial(config, output):
    """Prepare, convert in this process, and publish."""
    store = prepare(config, output)
    if worker(store.output, 0, 1):
        store.audit()
        store.write_summary()
        raise RuntimeError(f'Conversion failed; inspect {store.work}/errors/ and rerun to resume')
    finalize(store.output)


def run(config, output, workers=1):
    """Local convenience runner. Cluster scheduling belongs to the caller."""
    require_single_process()
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    configure_worker()
    store = prepare(config, output)
    if store.published():
        return
    command = [sys.executable, '-m', 'loom_data', 'convert', 'worker', '--output-dir', str(store.output)]
    statuses = run_workers(command, workers)
    if any(statuses):
        if store.drops:
            store.audit()
            store.write_summary()
        raise RuntimeError(f'Workers failed ({statuses}); rerun the same command to resume')
    finalize(store.output)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data convert', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
            ('run', 'local: prepare, run --workers processes, finalize'),
            ('prepare', 'create or check the plan and clear interrupted attempts; one process')):
        launch = sub.add_parser(name, help=help_text)
        launch.add_argument('--dataset', choices=sources.NAMES, required=True)
        launch.add_argument('--input-dir', type=Path, help='source data (default: dataset/raw/<dataset name>)')
        launch.add_argument('--output-dir', type=Path, help='export (default: dataset/export/<dataset name>)')
        if name == 'run':
            launch.add_argument('--workers', type=int, default=1, help='local worker processes (default 1)')
        configuration.add_arguments(launch)
    child = sub.add_parser('worker', help='convert the tasks assigned to one worker; launch with srun after prepare')
    child.add_argument('--output-dir', type=Path, required=True)
    child.add_argument('--worker-id', type=int, help='default: $SLURM_PROCID')
    child.add_argument('--worker-count', type=int, help='default: $SLURM_NTASKS')
    finish = sub.add_parser('finalize', help='check every episode and publish; one process after all workers exit')
    finish.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'worker':
            output = Path(args.output_dir).resolve()
            with worker_diagnostics(work_directory(output) / 'logs', f'output={output}') as report:
                rank, count = worker_rank(args.worker_id, args.worker_count)
                status = worker(args.output_dir, rank, count, report=report)
                report(f'Worker exit code: {status}; episode errors: {work_directory(args.output_dir)}/errors/')
                return status
        if args.command == 'finalize':
            finalize(args.output_dir)
            return 0
        if args.command == 'run' and args.workers < 1:
            raise ValueError('--workers must be positive')
        title = sources.get(args.dataset).title
        if args.input_dir is None:
            args.input_dir = paths.ROOT / 'dataset/raw' / title
        if args.output_dir is None:
            args.output_dir = paths.ROOT / 'dataset/export' / title
        configure_worker()
        config = configuration.from_arguments(args)
        if args.command == 'prepare':
            prepare(config, args.output_dir)
        else:
            run(config, args.output_dir, args.workers)
    except KeyboardInterrupt:
        print('Interrupted; wait for all workers to exit, then rerun to resume.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    run_cli(main)
