"""loom-data: one command for every step from raw data to a checked export."""
import importlib
import sys

from loom_data.resumable import run_cli

COMMANDS = {
    'download': ('loom_data.download', 'download a small pinned subset of a source dataset'),
    'convert': ('loom_data.convert.job', 'convert a source dataset to the export contract'),
    'check': ('loom_data.check.job', 'detect anomalies in an exported dataset'),
    'clean': ('loom_data.check.clean', 'write a copy of an exported dataset with its anomalies handled'),
    'pack': ('loom_data.pack', 'pack the videos of exported datasets into large files for training'),
    'stats': ('loom_data.check.dataset_stats', 'episode counts and durations of exported datasets'),
    'anomaly-stats': ('loom_data.check.anomaly_stats', 'summarize anomaly reports'),
    'view': ('loom_data.viewer.server', 'browse an exported dataset in the browser'),
    'robot': ('loom_data.robot', 'robot model tools'),
}


def dispatch(prog, commands, argv):
    """Run `argv[0]` from `commands`: name -> (module with main(argv), one-line summary)."""
    if not argv or argv[0] in ('-h', '--help') or argv[0] not in commands:
        width = max(map(len, commands))
        print(f'usage: {prog} <command> ...\n\ncommands:')
        for name, (_, summary) in commands.items():
            print(f'  {name:<{width}}  {summary}')
        print(f'\nRun "{prog} <command> --help" for the options of a command.')
        return 0 if argv and argv[0] in ('-h', '--help') else 2
    return importlib.import_module(commands[argv[0]][0]).main(argv[1:])


def main(argv=None):
    return dispatch('loom-data', COMMANDS, sys.argv[1:] if argv is None else argv)


def run():
    run_cli(main)
