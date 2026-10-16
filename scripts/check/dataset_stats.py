"""Summarize episode counts and instruction spans by reading exported manifests only."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
NS_PER_SECOND = 1_000_000_000


def discover_datasets(paths):
    """Accept dataset roots or their parent, preserving order and removing duplicates."""
    roots = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError(f'{path}: dataset directory does not exist')
        candidates = ([path] if (path / 'episodes.jsonl').is_file() else
                      [child for child in sorted(path.iterdir())
                       if child.is_dir() and (child / 'episodes.jsonl').is_file()])
        if not candidates:
            raise ValueError(f'{path}: no exported datasets with episodes.jsonl found')
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in roots:
                roots.append(candidate)
    return roots


def iter_manifest(root):
    """Validate and stream records without retaining instruction text for the dataset."""
    episode_ids = set()
    path = root / 'episodes.jsonl'
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected an episode object')
                episode_id = record.get('episode_id')
                if (not isinstance(episode_id, str) or episode_id in ('', '.', '..')
                        or '/' in episode_id or '\\' in episode_id or '\0' in episode_id):
                    raise ValueError('invalid episode_id')
                if episode_id in episode_ids:
                    raise ValueError(f'duplicate episode_id: {episode_id}')
                instruction_span_ns(record)
                episode_ids.add(episode_id)
                yield record
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc


def instruction_span_ns(record):
    """Return annotation extent, without assuming it covers every sample stream."""
    instructions = record.get('instructions')
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{record['episode_id']}: missing instructions")
    previous_end = 0
    for instruction in instructions:
        if not isinstance(instruction, dict):
            raise ValueError(f"{record['episode_id']}: invalid instruction")
        start, end = instruction.get('start_ns'), instruction.get('end_ns')
        if (type(start) is not int or type(end) is not int
                or not previous_end <= start < end < 2**63):
            raise ValueError(f"{record['episode_id']}: invalid instruction interval")
        previous_end = end
    return previous_end


def summarize_dataset(root):
    """Read only the manifest; annotation extent is not a stream verification."""
    root = Path(root).resolve()
    duration_ns = 0
    count = 0
    for record in iter_manifest(root):
        duration_ns += record['instructions'][-1]['end_ns']
        count += 1
    return {'dataset': root.name, 'path': str(root), 'episode_count': count,
            'duration_ns': duration_ns, 'duration_seconds': duration_ns // NS_PER_SECOND,
            'duration_source': 'instructions'}


def human_duration(duration_ns):
    seconds = duration_ns // NS_PER_SECOND
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path,
                        help='dataset roots or their parent (default: dataset/processed)')
    parser.add_argument('--json', action='store_true', help='print a JSON report to stdout')
    args = parser.parse_args(argv)
    try:
        roots = discover_datasets(args.paths or [ROOT / 'dataset/processed'])
        datasets = [summarize_dataset(root) for root in roots]
    except KeyboardInterrupt:
        print('Interrupted; dataset files were not modified.', file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    totals = {key: sum(dataset[key] for dataset in datasets) for key in
              ('episode_count', 'duration_ns')}
    totals['duration_seconds'] = totals['duration_ns'] // NS_PER_SECOND
    totals['duration_source'] = 'instructions'
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
        print('\nNote: Time comes from instructions (the final end_ns of each episode).')


if __name__ == '__main__':
    main()
