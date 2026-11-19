"""Summarize episode counts and durations by reading each dataset's meta/info.json only."""

import argparse
import json
from pathlib import Path
import sys

from loom_data.contract import FPS, LAYOUT, discover_datasets
from loom_data.paths import EXPORT
NS_PER_SECOND = 1_000_000_000


def dataset_name(root):
    """<source>/<layout> for a layout dataset inside a source folder, otherwise the folder name."""
    return f'{root.parent.name}/{root.name}' if LAYOUT.fullmatch(root.name) else root.name


def summarize_dataset(root):
    """Read only meta/info.json: every frame lasts 1/30 s."""
    root = Path(root).resolve()
    try:
        info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'{root}: cannot read meta/info.json: {exc}') from exc
    if not isinstance(info, dict):
        raise ValueError(f'{root}: meta/info.json is not a JSON object')
    count, frames = info.get('total_episodes'), info.get('total_frames')
    if (info.get('fps') != FPS or type(count) is not int or type(frames) is not int
            or count < 0 or frames < count):
        raise ValueError(f'{root}: not a {FPS} fps contract dataset or invalid totals')
    duration_ns = frames * NS_PER_SECOND // FPS
    return {'dataset': dataset_name(root), 'path': str(root), 'episode_count': count, 'frame_count': frames,
            'duration_ns': duration_ns, 'duration_seconds': duration_ns // NS_PER_SECOND,
            'duration_source': 'frames'}


def human_duration(duration_ns):
    seconds = duration_ns // NS_PER_SECOND
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data stats', description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path,
                        help='dataset roots or their parent (default: dataset/export)')
    parser.add_argument('--json', action='store_true', help='print a JSON report to stdout')
    args = parser.parse_args(argv)
    try:
        roots = discover_datasets(args.paths or [EXPORT])
        datasets = [summarize_dataset(root) for root in roots]
    except KeyboardInterrupt:
        print('Interrupted; dataset files were not modified.', file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    totals = {key: sum(dataset[key] for dataset in datasets) for key in
              ('episode_count', 'duration_ns')}
    totals['duration_seconds'] = totals['duration_ns'] // NS_PER_SECOND
    totals['duration_source'] = 'frames'
    if args.json:
        print(json.dumps({'datasets': datasets, 'total': totals}, ensure_ascii=False, indent=2))
    else:
        rows = [('Dataset', 'Episodes', 'Time (HH:MM:SS)')]
        for dataset in datasets + [{'dataset': 'TOTAL', **totals}]:
            rows.append((dataset['dataset'], str(dataset['episode_count']),
                         human_duration(dataset['duration_ns'])))
        widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
        for row in rows:
            print('  '.join(value.ljust(width) for value, width in zip(row, widths)))
        print(f'\nNote: Time is the frame count at {FPS} fps.')
