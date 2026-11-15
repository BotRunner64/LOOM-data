"""Download small, pinned subsets of the source datasets from the Hugging Face Hub.

Every dataset has two commands. `select` picks a subset and saves it as a
manifest pinned to a commit; `download` fetches exactly what a manifest lists.
"""
import argparse
from importlib import import_module
from pathlib import Path
import sys

from loom_data import paths, sources
from loom_data.download.hub import configure_hub, positive_int


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data download', description=__doc__.split('\n\n')[0])
    datasets = parser.add_subparsers(dest='dataset', required=True, metavar='DATASET')
    for name in sources.NAMES:
        module = import_module(f'loom_data.download.{name}')
        summary = module.__doc__.strip().splitlines()[0]
        dataset = datasets.add_parser(name, help=summary, description=module.__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
        commands = dataset.add_subparsers(dest='command', required=True)
        for command_name, function in module.COMMANDS.items():
            command = commands.add_parser(command_name)
            command.set_defaults(function=function)
            command.add_argument('--manifest', type=Path, default=paths.MANIFESTS / module.MANIFEST)
            command.add_argument('--workers', type=positive_int, default=module.WORKERS)
            command.add_argument('--retries', type=int, default=3, help='additional attempts for transient failures')
            if command_name == 'select':
                command.add_argument('--revision', default='main', help='branch, tag or commit to pin')
            else:
                command.add_argument('--output-dir', type=Path, default=paths.RAW / sources.TITLES[name])
            module.add_arguments(command, command_name)
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error('--retries must be nonnegative')
    configure_hub()
    try:
        args.function(args)
    except KeyboardInterrupt:
        print('Interrupted. Rerun the command to continue.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0
