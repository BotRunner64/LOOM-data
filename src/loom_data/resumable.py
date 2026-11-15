"""Runtime shared by the resumable jobs (conversion and anomaly detection).

A job is prepare -> workers -> finalize. Each job keeps its own plan and
records; this module holds what both need: durable atomic writes, input
fingerprints, worker setup and diagnostics, and the local worker launcher.
"""
from contextlib import contextmanager
import errno
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid

from tqdm import tqdm

from loom_data import paths


class StateError(ValueError):
    """Inconsistent recovery state: never overwrite or silently skip it."""


def progress(label, total=None, *, unit='item'):
    """Use tqdm on stderr, refreshing redirected logs less frequently."""
    terminal = sys.stderr.isatty()
    return tqdm(desc=label, total=total, unit=unit, file=sys.stderr,
                dynamic_ncols=terminal, mininterval=.1 if terminal else 5, miniters=1,
                disable=os.environ.get('LOOM_DATA_NO_PROGRESS') == '1')


def no_progress(label, total=None, *, unit='item'):
    return tqdm(desc=label, total=total, unit=unit, disable=True)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise StateError(f'Cannot read {path}: {exc}') from exc


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path, text):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, encoded(value) + '\n')


def input_stamp(path):
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        # Match Path.exists() for missing paths and unresolvable symlinks.
        if exc.errno not in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP):
            raise
        return {'path': str(path), 'size': None, 'mtime_ns': None}
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Expected source file: {path}')
    return {'path': str(path), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


def check_inputs(stamps, progress=no_progress):
    with progress('Verifying inputs', len(stamps), unit='file') as bar:
        for stamp in stamps:
            if input_stamp(stamp['path']) != stamp:
                raise ValueError(f'Source changed since prepare: {stamp["path"]}; use a new output')
            bar.update()


def work_directory(output, suffix):
    """Recovery state lives outside the export tree: dataset/export/X -> dataset/work/X<suffix>."""
    output = Path(output).resolve()
    export_root = (paths.ROOT / 'dataset/export').resolve()
    if output != export_root and output.is_relative_to(export_root):
        output = paths.ROOT / 'dataset/work' / output.relative_to(export_root)
    return output.with_name(output.name + suffix)


def require_single_process():
    if int(os.environ.get('SLURM_NTASKS', '1')) > 1 and 'SLURM_PROCID' in os.environ:
        raise ValueError('prepare, finalize and local run require one process; launch worker with srun')


def configure_worker():
    # Set before importing numpy/Arrow/codec modules in child processes.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['LOOM_DATA_SERIAL_CODECS'] = '1'
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


@contextmanager
def worker_diagnostics(directory, label):
    """Keep CLI startup failures and native crash stacks outside episode records."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'worker-{uuid.uuid4().hex}.err'
    with path.open('x', encoding='utf-8', buffering=1) as handle:
        def report(message):
            print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}',
                  file=handle, flush=True)

        report(f'Start host={socket.gethostname()} pid={os.getpid()} {label}')
        report(f'Python: {sys.executable}; argv={sys.argv!r}')
        for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_PROCID', 'SLURM_NTASKS'):
            report(f'{key}={os.environ.get(key, "<unset>")}')
        print(f'Worker diagnostics: {path}', file=sys.stderr, flush=True)
        was_enabled = faulthandler.is_enabled()
        faulthandler.enable(file=handle)
        try:
            yield report
        except BaseException:
            report('Worker terminated by exception:')
            traceback.print_exc(file=handle)
            handle.flush()
            raise
        finally:
            faulthandler.disable()
            if was_enabled:
                faulthandler.enable()


def worker_rank(worker_id, worker_count):
    """Explicit rank, or the one srun assigns to each task."""
    rank = worker_id if worker_id is not None else int(os.environ['SLURM_PROCID'])
    count = worker_count if worker_count is not None else int(os.environ['SLURM_NTASKS'])
    return rank, count


def run_workers(command, workers):
    """Launch local worker processes and return their exit codes."""
    processes = []
    try:
        for rank in range(workers):
            processes.append(subprocess.Popen([*command, '--worker-id', str(rank), '--worker-count', str(workers)]))
        return [p.wait() for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


def run_cli(main):
    """Exit with main()'s status; SIGTERM (scancel) unwinds like Ctrl-C so state stays resumable."""
    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
