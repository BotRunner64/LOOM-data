#!/usr/bin/env python3
"""The anomaly detection job: prepare a plan, run workers, then write the report.

Same shape as the conversion job. State lives in `<report>.anomaly-work`:

    plan.sqlite          one row per episode, with the detection thresholds
    records/<index>.json anomalies of one checked episode (possibly none)
    errors/<index>.err   last failure of an unfinished episode
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, fields
from itertools import islice
from pathlib import Path

from loom_data import contract
from loom_data.contract import STATE_NAMES
from loom_data import resumable
from loom_data.resumable import (
    StateError, atomic_json, atomic_text, check_inputs, configure_worker, digest, encoded, input_stamp,
    progress, read_json, require_single_process, run_cli, run_workers, sync_directory, worker_diagnostics,
    worker_rank,
)


def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def work_directory(output):
    """Keep recovery files out of repository exports and away from conversion state."""
    return resumable.work_directory(output, '.anomaly-work')


@contextmanager
def task_database(path):
    # The file is published once and only replaced while all workers are stopped.
    # immutable avoids SQLite lock/journal traffic on the shared filesystem.
    connection = None
    try:
        connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        connection.execute('PRAGMA cache_size=-2048')
        yield connection
    except sqlite3.DatabaseError as exc:
        raise StateError(f'Damaged detection plan: {path}: {exc}') from exc
    finally:
        if connection is not None:
            connection.close()


class Tasks:
    """A bounded-memory, indexed view retaining ordinal % worker_count assignment."""

    def __init__(self, path, ordinals):
        self.path = path
        self.ordinals = ordinals

    def __len__(self):
        return len(self.ordinals)

    def __getitem__(self, index):
        selection = self.ordinals[index]
        if isinstance(selection, range):
            return Tasks(self.path, selection)
        return next(iter(Tasks(self.path, range(selection, selection + 1))))

    def __iter__(self):
        with task_database(self.path) as connection:
            ordinals = iter(self.ordinals)
            while batch := list(islice(ordinals, 128)):
                if self.ordinals.step == 1:
                    rows = connection.execute(
                        'SELECT ordinal, body, checksum FROM tasks WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal',
                        (batch[0], batch[-1]))
                else:
                    marks = ','.join('?' for _ in batch)
                    rows = connection.execute(
                        f'SELECT ordinal, body, checksum FROM tasks WHERE ordinal IN ({marks}) ORDER BY ordinal', batch)
                found = 0
                for ordinal, body, checksum in rows:
                    if ordinal != batch[found]:
                        raise StateError('Missing plan tasks')
                    if hashlib.sha256(body.encode()).hexdigest() != checksum:
                        raise StateError('Damaged plan task checksum')
                    yield json.loads(body)
                    found += 1
                if found != len(batch):
                    raise StateError('Missing plan tasks')


def create_plan(work, tasks, metadata):
    """Stream tasks and metadata into the sole plan file, then publish atomically."""
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / f'.plan-{uuid.uuid4().hex}.tmp'
    connection = sqlite3.connect(temporary)
    checksum = hashlib.sha256()
    count = 0
    previous = -1
    try:
        # Durability is provided by fsync + rename after the temporary DB is closed.
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('PRAGMA synchronous=OFF')
        connection.execute('PRAGMA cache_size=-8192')
        connection.execute('CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK (id=1), body TEXT NOT NULL)')
        connection.execute('CREATE TABLE tasks (ordinal INTEGER PRIMARY KEY, episode_id TEXT UNIQUE, '
                           'body TEXT NOT NULL, checksum TEXT NOT NULL)')
        for task in tasks:
            if (not isinstance(task, dict) or type(task.get('index')) is not int
                    or task['index'] <= previous):
                raise StateError('Invalid episode task list')
            previous = task['index']
            body = encoded(task)
            try:
                connection.execute('INSERT INTO tasks VALUES (?, ?, ?, ?)',
                                   (count, task['record']['episode_id'], body,
                                    hashlib.sha256(body.encode()).hexdigest()))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'Duplicate episode_id: {task["record"]["episode_id"]}') from exc
            checksum.update(body.encode())
            checksum.update(b'\n')
            count += 1
        if not count or metadata.get('version') != 1:
            raise StateError('Invalid episode task list or plan version')
        plan = {**metadata, 'task_count': count, 'tasks_digest': checksum.hexdigest()}
        plan['id'] = digest(plan)
        connection.execute('INSERT INTO metadata VALUES (1, ?)', (encoded(plan),))
        connection.commit()
        connection.close()
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        temporary.replace(work / 'plan.sqlite')
        sync_directory(work)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get('version') != 1
            or plan.get('id') != digest({k: v for k, v in plan.items() if k != 'id'})
            or type(plan.get('task_count')) is not int or plan['task_count'] < 1
            or not isinstance(plan.get('tasks_digest'), str)):
        raise StateError('Damaged detection plan metadata')


class Store:
    def __init__(self, root, output):
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        self.work = work_directory(self.output)
        self.database = self.work / 'plan.sqlite'
        if not self.database.exists():
            raise StateError(f'Missing detection plan: {self.database}; run prepare before starting workers')
        self.database_stamp = input_stamp(self.database)
        with task_database(self.database) as connection:
            row = connection.execute('SELECT body FROM metadata WHERE id=1').fetchone()
        if row is None:
            raise StateError('Missing detection plan metadata')
        self.plan = json.loads(row[0])
        validate_plan(self.plan)
        check_inputs([self.database_stamp])
        plan = self.plan
        if plan.get('root') != str(self.root) or plan.get('output') != str(self.output):
            raise StateError('Damaged plan or plan belongs to another dataset/report')
        self.plan['tasks'] = Tasks(self.database, range(plan['task_count']))

    def check_shared(self):
        check_inputs([self.plan['manifest'], self.database_stamp])
        from loom_data.check import anomalies as detector
        if file_digest(detector.__file__) != self.plan['detector']:
            raise StateError('Detection code changed since prepare; use a new report path')

    def result(self, task):
        path = self.work / 'records' / f'{task["index"]}.json'
        if path.is_symlink():
            raise StateError(f'Symlink result: {path}')
        if not path.exists():
            return None
        value = read_json(path)
        if (not isinstance(value, dict) or value.get('plan_id') != self.plan['id']
                or value.get('index') != task['index']
                or value.get('checksum') != digest({k: v for k, v in value.items() if k != 'checksum'})
                or not isinstance(value.get('results'), list)):
            raise StateError(f'Damaged result: {path}')
        for row in value['results']:
            if (not isinstance(row, dict)
                    or set(row) != {'index', 'stream', 'start_ns', 'end_ns', 'reason'}
                    or row['index'] != task['index']
                    or row['stream'] not in STATE_NAMES
                    or row['reason'] not in ('boundary_idle', 'state_jump', 'internal_idle')
                    or type(row['start_ns']) is not int or type(row['end_ns']) is not int
                    or not 0 <= row['start_ns'] <= row['end_ns']):
                raise StateError(f'Invalid anomaly record: {path}')
        return value['results']

    def audit(self, *, complete=False):
        expected = set()
        checksum = hashlib.sha256()
        with task_database(self.database) as connection:
            if connection.execute('SELECT 1 FROM tasks WHERE ordinal >= ? LIMIT 1',
                                  (len(self.plan['tasks']),)).fetchone():
                raise StateError('Unexpected plan tasks')
        for task in self.plan['tasks']:
            expected.add(f'{task["index"]}.json')
            checksum.update(encoded(task).encode())
            checksum.update(b'\n')
        if checksum.hexdigest() != self.plan['tasks_digest']:
            raise StateError('Damaged plan tasks digest')
        records = self.work / 'records'
        if records.is_symlink():
            raise StateError('Symlink records directory')
        actual = {p.name for p in records.iterdir()} if records.exists() else set()
        if actual - expected:
            raise StateError('Unexpected files in records directory; rerun prepare after workers exit')
        errors = self.work / 'errors'
        error_names = {p.name for p in errors.iterdir()} if errors.exists() else set()

        def check(task):
            # Even unfinished episodes must retain the inputs frozen by prepare.
            check_inputs(task['inputs'])
            if f'{task["index"]}.json' in actual and self.result(task) is not None:
                return 'done'
            error_name = f'{task["index"]}.err'
            if error_name in error_names and (errors / error_name).exists():
                return 'failed'
            return 'unfinished'

        done = failed = unfinished = 0
        # Workers must have exited before audit. Reuse stable directory lists
        # and bound both shared-filesystem concurrency and queued work.
        with progress('Auditing detection', len(expected), unit='episode') as bar, \
                ThreadPoolExecutor(max_workers=8) as executor:
            tasks = iter(self.plan['tasks'])
            while batch := list(islice(tasks, 256)):
                for state in executor.map(check, batch):
                    if state == 'done':
                        done += 1
                    elif state == 'failed':
                        failed += 1
                    else:
                        unfinished += 1
                    bar.update()
        if complete and done != len(expected):
            raise StateError(f'Incomplete detection: {failed} failed, {unfinished} unfinished; '
                             'rerun workers before finalize')
        return done, failed, unfinished

    def published(self):
        path = self.work / 'published.json'
        if not path.exists():
            return False
        value = read_json(path)
        if (not isinstance(value, dict) or value.get('plan_id') != self.plan['id']
                or not self.output.is_file() or value.get('sha256') != file_digest(self.output)):
            raise StateError('Published report changed or is missing; use a new report path')
        return True


def prepare(root, output=None, config=None):
    require_single_process()
    from loom_data.check import anomalies as detector
    root = Path(root).resolve()
    output = detector.report_path(root, output)
    config = config or detector.Config()
    work = work_directory(output)
    if work == root or any(work.is_relative_to(root / name) for name in ('meta', 'data', 'videos')):
        raise ValueError('Work directory must be outside the dataset\'s meta/, data/ and videos/')
    if work.is_symlink():
        raise StateError('Symlink work directory')
    if (work / 'plan.sqlite').exists():
        print('Loading detection plan...', file=sys.stderr, flush=True)
        store = Store(root, output)
        if store.plan['config'] != asdict(config):
            raise StateError('Detection thresholds changed; use a new report path')
    else:
        if work.exists() and any(p.name != 'logs' and not p.name.endswith('.tmp') for p in work.iterdir()):
            raise StateError('Work directory is not empty but has no plan.sqlite; '
                             'use a new report path or rebuild the recovery directory')
        # info.json is rewritten whenever a dataset is rebuilt; it guards the whole plan.
        stamp = input_stamp(root / 'meta/info.json')
        dataset = contract.Dataset(root)

        def tasks():
            stamps = {}
            with progress('Preparing detection', len(dataset.episodes), unit='episode') as bar:
                for episode in dataset.episodes:
                    path = dataset.data_path(episode)
                    if path not in stamps:
                        stamps[path] = input_stamp(path)
                    yield {'index': episode['episode_index'], 'record': {'episode_id': episode['episode_id']},
                           'inputs': [stamps[path]]}
                    bar.update()
            check_inputs([stamp])

        plan = {'version': 1, 'root': str(root), 'output': str(output),
                'config': asdict(config), 'manifest': stamp,
                'detector': file_digest(detector.__file__)}
        create_plan(work, tasks(), plan)
        store = Store(root, output)
    store.check_shared()
    for name in ('records', 'errors', 'logs'):
        directory = work / name
        if directory.is_symlink():
            raise StateError(f'Symlink work subdirectory: {directory}')
        directory.mkdir(exist_ok=True)
    # Previous workers must have exited before prepare cleans interrupted writes.
    for directory in (work, work / 'records', work / 'errors'):
        for path in directory.glob('.*.tmp'):
            path.unlink()
    for path in (work / 'logs').iterdir():
        path.unlink()
    for path in output.parent.glob(f'.{output.name}.*.tmp'):
        path.unlink()
    done, failed, unfinished = store.audit()
    if store.published():
        if done != len(store.plan['tasks']):
            raise StateError('Published report has missing recovery records')
        print(f'Already complete: {output}', flush=True)
    else:
        total = len(store.plan['tasks'])
        print(f'Progress: {done}/{total} ({done / total:.1%}) episodes done '
              f'({failed} failed, {unfinished} unfinished)', flush=True)
        print(f'Prepared {total} episodes: {work}', flush=True)
    return store


def worker(root, output=None, worker_id=0, worker_count=1, *, report=None):
    if worker_count < 1 or not 0 <= worker_id < worker_count:
        raise ValueError('Require worker_count > 0 and 0 <= worker_id < worker_count')
    configure_worker()
    from loom_data.check import anomalies as detector
    root = Path(root).resolve()
    store = Store(root, detector.report_path(root, output))
    store.check_shared()
    config = detector.Config(**store.plan['config'])
    report = report or (lambda message: None)
    failures = 0
    dataset = contract.Dataset(root)
    assigned = store.plan['tasks'][worker_id::worker_count]
    with progress(f'Worker {worker_id}', len(assigned), unit='episode') as bar:
        for task in assigned:
            index = task['index']
            report(f'Episode index={index} id={task["record"]["episode_id"]}')
            check_inputs(task['inputs'])
            if store.result(task) is not None:
                bar.update()
                continue
            error = store.work / 'errors' / f'{index}.err'
            try:
                episode = dataset.episodes[index]
                if episode['episode_id'] != task['record']['episode_id']:
                    raise ValueError('Dataset episodes changed since prepare')
                streams, end_ns = detector.read_episode(dataset, episode)
                results = detector.detect_episode(index, streams, end_ns, config)
                check_inputs(task['inputs'])
            except Exception:  # noqa: BLE001 - record episode failures and continue assigned work
                failures += 1
                message = f'Episode index={index} id={task["record"]["episode_id"]}\n{traceback.format_exc()}'
                atomic_text(error, message)
                report(message)
            else:
                value = {'plan_id': store.plan['id'], 'index': index, 'results': results}
                value['checksum'] = digest(value)
                # The result rename commits an episode, including zero anomalies.
                atomic_json(store.work / 'records' / f'{index}.json', value)
                error.unlink(missing_ok=True)
            bar.update()
    store.check_shared()
    return 1 if failures else 0


def finalize(root, output=None):
    require_single_process()
    from loom_data.check import anomalies as detector
    root = Path(root).resolve()
    store = Store(root, detector.report_path(root, output))
    store.check_shared()
    store.audit(complete=True)
    episodes = len(store.plan['tasks'])
    if store.published():
        print(f'Already published: {store.output}', flush=True)
        return episodes, sum(len(store.result(t)) for t in store.plan['tasks'])
    store.output.parent.mkdir(parents=True, exist_ok=True)
    anomalies = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=store.output.parent,
                                         prefix=f'.{store.output.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            for task in store.plan['tasks']:
                for row in store.result(task):
                    # Recovery JSON sorts keys; restore the serial report's field order.
                    ordered = {key: row[key] for key in ('index', 'stream', 'start_ns', 'end_ns', 'reason')}
                    handle.write(json.dumps(ordered, separators=(',', ':')) + '\n')
                    anomalies += 1
            handle.flush()
            os.fsync(handle.fileno())
        store.check_shared()
        checksum = file_digest(temporary)
        temporary.replace(store.output)
        sync_directory(store.output.parent)
        atomic_json(store.work / 'published.json', {'plan_id': store.plan['id'], 'sha256': checksum})
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Checked {episodes} episodes; wrote {anomalies} anomalies to {store.output}', flush=True)
    return episodes, anomalies


def run_serial(root, output=None, config=None):
    """Prepare, check in this process, and write the report."""
    store = prepare(root, output, config)
    if not store.published() and worker(store.root, store.output, 0, 1):
        raise RuntimeError(f'Detection failed; inspect {store.work}/errors/ and rerun to resume')
    return finalize(store.root, store.output)


def run(root, output=None, config=None, workers=1):
    require_single_process()
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    configure_worker()
    store = prepare(root, output, config)
    if store.published():
        return finalize(store.root, store.output)
    command = [sys.executable, '-m', 'loom_data', 'check', 'worker', str(store.root), '--output', str(store.output)]
    statuses = run_workers(command, workers)
    if any(statuses):
        raise RuntimeError(f'Workers failed ({statuses}); rerun the same command to resume')
    return finalize(store.root, store.output)


def main(argv=None):
    # Limit native pools before importing the detector's NumPy and Arrow dependencies.
    configure_worker()
    from loom_data.check.anomalies import Config, report_path
    parser = argparse.ArgumentParser(prog='loom-data check', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
            ('run', 'local: prepare, run --workers processes, finalize'),
            ('prepare', 'create or check the plan; one process'),
            ('worker', 'check the episodes assigned to one worker; launch with srun after prepare'),
            ('finalize', 'write the report; one process after all workers exit')):
        command = sub.add_parser(name, help=help_text)
        command.add_argument('dataset', type=Path, help='exported dataset root')
        command.add_argument('--output', type=Path, help='JSONL report (default: DATASET/anomalies.jsonl)')
        if name in ('prepare', 'run'):
            for field in fields(Config):
                command.add_argument('--' + field.name.replace('_', '-'), type=field.type,
                                     default=field.default, help=f'default {field.default}')
        if name == 'run':
            command.add_argument('--workers', type=int, default=1, help='local worker processes (default 1)')
        if name == 'worker':
            command.add_argument('--worker-id', type=int, help='default: $SLURM_PROCID')
            command.add_argument('--worker-count', type=int, help='default: $SLURM_NTASKS')
    args = parser.parse_args(argv)
    try:
        if args.command == 'worker':
            output = report_path(args.dataset.resolve(), args.output)
            with worker_diagnostics(work_directory(output) / 'logs', f'output={output}') as report:
                rank, count = worker_rank(args.worker_id, args.worker_count)
                status = worker(args.dataset, args.output, rank, count, report=report)
                report(f'Worker exit code: {status}')
                return status
        if args.command == 'finalize':
            finalize(args.dataset, args.output)
        else:
            config = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
            if args.command == 'prepare':
                prepare(args.dataset, args.output, config)
            else:
                run(args.dataset, args.output, config, args.workers)
    except KeyboardInterrupt:
        print('Interrupted; wait for all workers to exit, then rerun to resume.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    run_cli(main)
