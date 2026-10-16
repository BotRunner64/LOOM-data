"""Summarize existing anomaly reports, including affected episodes by reason."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.check.dataset_stats import discover_datasets

STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
REASONS = ('boundary_idle', 'internal_idle', 'state_jump')


def read_episode_ids(root):
    """Keep physical manifest indices, including gaps caused by blank lines."""
    episodes, seen = {}, set()
    path = root / 'episodes.jsonl'
    with path.open(encoding='utf-8') as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode_id = record.get('episode_id') if isinstance(record, dict) else None
                if (not isinstance(episode_id, str) or episode_id in ('', '.', '..')
                        or any(char in episode_id for char in ('/', '\\', '\0'))):
                    raise ValueError('invalid episode_id')
                if episode_id in seen:
                    raise ValueError(f'duplicate episode_id: {episode_id}')
                episodes[index] = episode_id
                seen.add(episode_id)
            except ValueError as exc:
                raise ValueError(f'{path}:{index + 1}: {exc}') from exc
    return episodes


def iter_anomalies(path, episodes):
    """Stream and validate report records against the manifest."""
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected an anomaly object')
                for field in ('index', 'start_ns', 'end_ns'):
                    if type(record.get(field)) is not int or record[field] < 0:
                        raise ValueError(f'{field} must be a nonnegative integer')
                if record['index'] not in episodes:
                    raise ValueError('index does not address an episode manifest record')
                if record.get('stream') not in STATE_NAMES:
                    raise ValueError('invalid stream')
                if record['end_ns'] < record['start_ns']:
                    raise ValueError('end_ns must be >= start_ns')
                if not isinstance(record.get('reason'), str) or not record['reason'].strip():
                    raise ValueError('reason must be a nonempty string')
                yield {key: record[key] for key in
                       ('index', 'stream', 'start_ns', 'end_ns', 'reason')}
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc


def counts(record_count, affected_episode_count, episode_count):
    return {'record_count': record_count, 'affected_episode_count': affected_episode_count,
            'affected_episode_rate': (affected_episode_count / episode_count
                                      if episode_count else None)}


def summarize_dataset(root, report_path=None):
    root = Path(root).resolve()
    episodes = read_episode_ids(root)
    path = Path(report_path).resolve() if report_path is not None else root / 'anomalies.jsonl'
    summary = {'dataset': root.name, 'path': str(root), 'report_path': str(path),
               'episode_count': len(episodes), 'report_status': 'available',
               'by_reason': {}}
    if not path.is_file():
        if report_path is not None or path.exists():
            raise ValueError(f'{path}: anomaly report is not a file')
        summary.update(report_status='missing', record_count=None,
                       affected_episode_count=None, affected_episode_rate=None)
        return summary

    affected, groups = set(), {}
    record_count = 0
    for record in iter_anomalies(path, episodes):
        record_count += 1
        affected.add(record['index'])
        group = groups.setdefault(record['reason'], {'records': 0, 'episodes': set()})
        group['records'] += 1
        group['episodes'].add(record['index'])
    summary.update(counts(record_count, len(affected), len(episodes)))
    summary['by_reason'] = {
        name: counts(group['records'], len(group['episodes']), len(episodes))
        for name, group in sorted(groups.items())}
    return summary


def summarize_total(datasets):
    available = [dataset for dataset in datasets if dataset['report_status'] == 'available']
    reported_episodes = sum(dataset['episode_count'] for dataset in available)
    total = {'dataset_count': len(datasets), 'report_count': len(available),
             'missing_report_count': len(datasets) - len(available),
             'episode_count': sum(dataset['episode_count'] for dataset in datasets),
             'reported_episode_count': reported_episodes,
             **counts(sum(dataset['record_count'] for dataset in available),
                      sum(dataset['affected_episode_count'] for dataset in available),
                      reported_episodes)}
    groups = {}
    for dataset in available:
        for name, value in dataset['by_reason'].items():
            group = groups.setdefault(name, {'records': 0, 'episodes': 0})
            group['records'] += value['record_count']
            group['episodes'] += value['affected_episode_count']
    total['by_reason'] = {name: counts(group['records'], group['episodes'], reported_episodes)
                          for name, group in sorted(groups.items())}
    return total


def print_table(rows, *, numeric_columns=(), total_row=False):
    rows = [[str(value) if value is not None else '—' for value in row] for row in rows]
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    separator = '  '.join('─' * width for width in widths)
    for ordinal, row in enumerate(rows):
        if ordinal == 1 or (total_row and ordinal == len(rows) - 1):
            print(separator)
        print('  '.join(value.rjust(width) if index in numeric_columns else value.ljust(width)
                        for index, (value, width) in enumerate(zip(row, widths))).rstrip())


def percent(value):
    return f'{value:.2%}' if value is not None else '—'


def number(value):
    return f'{value:,}' if value is not None else '—'


def print_report(report):
    datasets, total = report['datasets'], report['total']
    reasons = list(REASONS) + sorted(set(total['by_reason']) - set(REASONS))
    rows = [('Dataset', 'Episodes', 'Records', 'Affected', 'Rate', *reasons)]
    for dataset in datasets + [{'dataset': 'TOTAL', **total}]:
        reason_counts = [
            None if dataset.get('report_status') == 'missing' else
            dataset['by_reason'].get(reason, {}).get('affected_episode_count', 0)
            for reason in reasons]
        rows.append((dataset['dataset'], number(dataset['episode_count']),
                     number(dataset['record_count']), number(dataset['affected_episode_count']),
                     percent(dataset['affected_episode_rate']), *map(number, reason_counts)))
    print_table(rows, numeric_columns=range(1, len(rows[0])), total_row=True)
    print('\nAffected and reason columns count unique episodes; reasons can overlap.')
    missing = [dataset['dataset'] for dataset in datasets if dataset['report_status'] == 'missing']
    if missing:
        print('Missing reports: ' + ', '.join(missing))
        print(f"Total rate uses {number(total['reported_episode_count'])} episodes with reports.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path,
                        help='dataset roots or their parent (default: dataset/export)')
    parser.add_argument('--anomalies', type=Path, help='custom report; requires a single dataset')
    parser.add_argument('--json', action='store_true', help='print a JSON report to stdout')
    args = parser.parse_args(argv)
    try:
        roots = discover_datasets(args.paths or [ROOT / 'dataset/export'])
        if args.anomalies is not None and len(roots) != 1:
            raise ValueError('--anomalies requires a single dataset')
        datasets = [summarize_dataset(root, args.anomalies) for root in roots]
        report = {'datasets': datasets, 'total': summarize_total(datasets)}
    except KeyboardInterrupt:
        print('Interrupted; dataset files were not modified.', file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == '__main__':
    main()
