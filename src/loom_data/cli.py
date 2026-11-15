"""loom-data: one command for every step from raw data to a checked export."""
import importlib
import sys

from loom_data.resumable import run_cli

COMMANDS = {
    'convert': ('loom_data.convert.job', 'convert a source dataset to the export contract'),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help') or argv[0] not in COMMANDS:
        width = max(map(len, COMMANDS))
        print('usage: loom-data <command> ...\n\ncommands:')
        for name, (_, summary) in COMMANDS.items():
            print(f'  {name:<{width}}  {summary}')
        print('\nRun "loom-data <command> --help" for the options of a command.')
        return 0 if argv and argv[0] in ('-h', '--help') else 2
    module = importlib.import_module(COMMANDS[argv[0]][0])
    return module.main(argv[1:])


def run():
    run_cli(main)
