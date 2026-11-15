"""Helpers shared by the conversion tests."""
from pathlib import Path
import json
import shutil

from loom_data import paths
from loom_data.check import anomalies, job as check_job
from loom_data.convert import convert, job

MODELS = paths.MODELS

__all__ = ['MODELS', 'assert_conversion_fails', 'check', 'convert', 'episode_errors', 'expected_report', 'manifest']


def manifest(output):
    return [json.loads(line) for line in (Path(output) / 'episodes.jsonl').read_text().splitlines()]


def episode_errors(output):
    """Text of every recorded episode failure of a conversion."""
    return ''.join(p.read_text() for p in sorted(job.work_directory(output).glob('errors/*.err')))


def assert_conversion_fails(case, pattern, dataset, source, output, **options):
    """A fresh conversion must fail without publishing.

    A nonempty `pattern` must match the raised error or a recorded episode failure.
    """
    shutil.rmtree(job.work_directory(output), ignore_errors=True)
    try:
        convert(dataset, source, output, **options)
    except (ValueError, OSError, RuntimeError) as exc:
        if pattern:
            case.assertRegex(f'{type(exc).__name__}: {exc}\n{episode_errors(output)}', pattern)
    else:
        case.fail('conversion unexpectedly succeeded')
    case.assertFalse(Path(output).exists(), 'a failed conversion must not publish')


def check(root, output=None, config=None):
    """Run anomaly detection from a clean state; return (episodes, anomalies)."""
    root = Path(root).resolve()
    shutil.rmtree(check_job.work_directory(anomalies.report_path(root, output)), ignore_errors=True)
    return check_job.run_serial(root, output, config)


def expected_report(root, config=anomalies.Config()):
    """The report the job must write, computed directly from the rules."""
    rows = []
    for index, line in enumerate((Path(root) / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()):
        if line.strip():
            streams, end_ns = anomalies.read_episode(Path(root), json.loads(line))
            rows += anomalies.detect_episode(index, streams, end_ns, config)
    return ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
