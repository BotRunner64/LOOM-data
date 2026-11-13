"""Estimate storage and freeze a reproducible episode subset during prepare."""
from collections import Counter
from fractions import Fraction
import hashlib
import math
import re


def storage_bytes(value):
    """Parse exact bytes or a size such as 100GB / 100GiB without float rounding."""
    if type(value) is int:
        if value > 0:
            return value
    elif isinstance(value, str):
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(B|[KMGTPE]i?B)?\s*', value, re.IGNORECASE)
        if match:
            unit = (match[2] or 'B').upper()
            power = 'BKMGTPE'.index(unit[0])
            size = Fraction(match[1]) * (1024 if 'I' in unit else 1000) ** power
            if size > 0 and size.denominator == 1:
                return int(size)
    raise ValueError('Storage budget must be positive whole bytes, e.g. 100GB or 100GiB')


def add_budget_arguments(parser):
    parser.add_argument('--storage-budget', type=storage_bytes,
                        help='approximate output storage budget used to sample the plan; e.g. 100GB or 100GiB')
    parser.add_argument('--sample-seed', type=int, default=0,
                        help='episode sampling seed when --storage-budget is set (default 0)')
    parser.add_argument('--storage-estimate-ratio', type=float, default=1.,
                        help='estimated output/source byte ratio for storage sampling (default 1)')


def sample_tasks(config, tasks):
    """Select episodes by seeded hash priority, preserving source task ordering.

    File sizes come from prepare's input stamps, without decoding or conversion.
    Count each source file once, splitting shared files across referring tasks
    and each task's bytes equally across its candidate episodes. This handles
    packed videos, archives and shards without charging a full file per episode.
    """
    if config.get('storage_budget') is None:
        return tasks, None
    budget = config['storage_budget']
    ratio = Fraction(str(config['storage_estimate_ratio']))
    paths = [{stamp['path']: stamp['size'] or 0 for stamp in task['inputs']} for task in tasks]
    references = Counter(path for files in paths for path in files)
    candidates = []
    for task, files in zip(tasks, paths):
        task_bytes = sum((Fraction(size, references[path]) for path, size in files.items()), Fraction())
        episode_bytes = max(1, math.ceil(task_bytes * ratio / len(task['episode_ids'])))
        for episode_id in task['episode_ids']:
            priority = hashlib.sha256(f'{config["sample_seed"]}:{episode_id}'.encode()).digest()
            candidates.append((priority, episode_id, episode_bytes))
    selected = set()
    used = 0
    for _, episode_id, size in sorted(candidates):
        if size <= budget - used:
            selected.add(episode_id)
            used += size
    if not selected:
        smallest = min(size for _, _, size in candidates)
        raise ValueError(f'Storage budget {budget} bytes is too small for one estimated episode '
                         f'(smallest {smallest} bytes); increase --storage-budget or adjust --storage-estimate-ratio')
    sampled = []
    for task in tasks:
        ids = [episode_id for episode_id in task['episode_ids'] if episode_id in selected]
        if ids:
            sampled.append(dict(task, episode_ids=ids))
    summary = {'method': 'shared_source_bytes_per_episode', 'budget_bytes': budget,
               'estimate_ratio': config['storage_estimate_ratio'],
               'candidate_episodes': len(candidates), 'selected_episodes': len(selected),
               'estimated_candidate_bytes': sum(size for _, _, size in candidates),
               'estimated_output_bytes': used}
    return sampled, summary


def run_budgeted_cli(args, dataset):
    """Route source CLIs with a budget through the common resumable store."""
    if args.storage_budget is None:
        if args.sample_seed != 0 or args.storage_estimate_ratio != 1.:
            raise ValueError('--sample-seed and --storage-estimate-ratio require --storage-budget')
        return False
    from argparse import Namespace
    from scripts.convert.parallel import run_native
    from scripts.convert.parallel_adapters import configuration
    values = dict(vars(args), dataset=dataset)
    if 'models_dir' in values:
        values['model_dir'] = values['models_dir']
    run_native(configuration(Namespace(**values)), args.output_dir)
    return True
