"""Helpers shared by the conversion tests."""
from pathlib import Path
import json
import shutil

from loom_data import paths
from loom_data.convert import convert, job

MODELS = paths.MODELS

__all__ = ['MODELS', 'assert_conversion_fails', 'convert', 'episode_errors', 'manifest']


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
