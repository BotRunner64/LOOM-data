"""Conversion configuration: what a plan records about how a dataset is converted."""
import math
from pathlib import Path

from loom_data import contract, paths, sources
from loom_data.resumable import no_progress
from loom_data.convert.budget import storage_bytes


def source_options():
    """Every source-specific option: name -> (argparse kwargs, datasets that accept it)."""
    options = {}
    for name in sources.NAMES:
        for option, spec in sources.get(name).OPTIONS.items():
            if option in options:
                known, datasets = options[option]
                if {k: known.get(k) for k in ('type', 'nargs', 'choices')} != {k: spec.get(k) for k in ('type', 'nargs', 'choices')}:
                    raise TypeError(f'Sources disagree on the shape of option {option}')
                datasets.append(name)
            else:
                options[option] = (dict(spec), [name])
    return options


def add_arguments(parser):
    """Selection limits, storage budget and every source option."""
    parser.add_argument('--model-dir', type=Path,
                        help='robot model directory (default: the source\'s model under assets/robot_models)')
    parser.add_argument('--limit', type=int, help='first N episodes (first N archives for agibotworld2026)')
    parser.add_argument('--limit-episodes', type=int,
                        help='first N candidate episodes overall, counting episodes that are later skipped')
    parser.add_argument('--storage-budget', type=storage_bytes,
                        help='approximate output size used to sample episodes, e.g. 100GB or 100GiB')
    parser.add_argument('--sample-seed', type=int, default=0,
                        help='sampling seed when --storage-budget is set (default 0)')
    parser.add_argument('--storage-estimate-ratio', type=float, default=1.,
                        help='estimated output/source size ratio for --storage-budget (default 1)')
    group = parser.add_argument_group('source options')
    for option, (spec, datasets) in source_options().items():
        spec = {k: v for k, v in spec.items() if k != 'help'}
        helps = {sources.get(name).OPTIONS[option].get('help', '') for name in datasets}
        text = helps.pop() if len(helps) == 1 else 'see the source documentation'
        group.add_argument('--' + option.replace('_', '-'), help=f'[{", ".join(datasets)}] {text}', **spec)


def from_arguments(args):
    names = ('model_dir', 'limit', 'limit_episodes', 'storage_budget', 'sample_seed', 'storage_estimate_ratio',
             *source_options())
    return make_config(args.dataset, args.input_dir, **{name: getattr(args, name) for name in names})


def make_config(dataset, source, *, model_dir=None, limit=None, limit_episodes=None,
                storage_budget=None, sample_seed=0, storage_estimate_ratio=1., **options):
    """Validate and freeze a conversion request. The result is stored in the plan."""
    kind = sources.get(dataset)
    source = Path(source).resolve()
    if not source.exists() or (not kind.accepts_file and not source.is_dir()):
        raise ValueError(f'Input does not exist or has the wrong type: {source}')
    for value in (limit, limit_episodes):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Limits must be positive integers')
    for option, value in options.items():
        if value is not None and option not in kind.OPTIONS:
            raise ValueError(f'--{option.replace("_", "-")} is not an option of {dataset}')
    if kind.model is None:
        if model_dir is not None:
            raise ValueError(f'{dataset} uses recorded poses, not a robot model (--model-dir)')
    else:
        model_dir = str(Path(model_dir or paths.ROOT / 'assets/robot_models' / kind.model).resolve())
    params = {'export_contract_version': contract.VERSION,
              **kind.make_params({option: options.get(option) for option in kind.OPTIONS})}
    offsets = ([params['eef_offset']] if 'eef_offset' in params else []) + list(params.get('eef_offsets', {}).values())
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in offsets):
        raise ValueError('EEF offsets require three finite metres')
    config = {'dataset': dataset, 'source': str(source), 'model_dir': model_dir,
              'limit': limit, 'limit_episodes': limit_episodes, 'params': params}
    if storage_budget is not None:
        if type(sample_seed) is not int:
            raise ValueError('Sample seed must be an integer')
        if (isinstance(storage_estimate_ratio, bool) or not isinstance(storage_estimate_ratio, (int, float))
                or not math.isfinite(storage_estimate_ratio) or storage_estimate_ratio <= 0):
            raise ValueError('Storage estimate ratio must be positive and finite')
        config.update(storage_budget=storage_bytes(storage_budget), sample_seed=sample_seed,
                      storage_estimate_ratio=float(storage_estimate_ratio))
    elif sample_seed != 0 or storage_estimate_ratio != 1.:
        raise ValueError('--sample-seed and --storage-estimate-ratio require --storage-budget')
    return config


def discover(config, progress=no_progress):
    """All conversion tasks for a config, with --limit-episodes applied across tasks."""
    tasks, shared = sources.get(config['dataset']).discover(config, progress)
    remaining = config['limit_episodes']
    if remaining is not None:
        selected = []
        for task in tasks:
            task['episode_ids'] = task['episode_ids'][:remaining]
            selected.append(task)
            remaining -= len(task['episode_ids'])
            if remaining == 0:
                break
        tasks = selected
    if not tasks:
        raise ValueError(f'No tasks found for {config["dataset"]}: {config["source"]}')
    return tasks, sorted(set(shared))
