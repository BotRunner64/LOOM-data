"""What a data source provides to the conversion job.

A source module defines one `Source` subclass and exports it as `SOURCE`.
The class answers three questions:

- `make_params`: which source-specific options exist and what they resolve to;
- `discover`: which episodes exist under an input directory;
- `episode`: how one episode becomes a contract episode directory.

Everything else (planning, resume, validation, publication) belongs to
`loom_data.convert.job` and is the same for every source.
"""
from pathlib import Path

from loom_data.sources.policy import SourceDataError


def task_limit(config):
    """Episode cap during discovery: the tighter of --limit and --limit-episodes."""
    return min((v for v in (config['limit'], config['limit_episodes']) if v is not None), default=None)


class Source:
    name = None           # dataset key used on the command line and in plans
    model = None          # default folder under assets/robot_models ('' = that folder itself); None = no robot model
    accepts_file = False  # the input may be a single file instead of a directory
    # strict: every planned episode must convert.
    # skip:   `episode` may return None for episodes outside the source's selection rule.
    # drop:   bad source data (SourceDataError) is recorded with a reason and left out.
    policy = 'strict'
    skip_reason = ''
    # Source-specific options: option name -> argparse keyword arguments.
    OPTIONS = {}

    @classmethod
    def make_params(cls, options):
        """Resolve `options` (every OPTIONS key, None when unset) to the params stored in the plan."""
        return {}

    @classmethod
    def discover(cls, config, progress):
        """Return (tasks, shared input paths).

        A task is a JSON-serializable dict with `source`, `episode_ids` and
        `inputs` (the files whose size and mtime guard a resume), plus whatever
        the source needs to convert it later.
        """
        raise NotImplementedError

    def __init__(self, config):
        self.config = config
        self.params = config['params']
        self.root = Path(config['source'])
        self.model_dir = Path(config['model_dir']) if config['model_dir'] else None

    def episode(self, task, destination):
        """Write one episode under `destination` and return its episodes.jsonl record."""
        raise NotImplementedError

    def process(self, task, pending, staging, store, failed):
        """Convert the unfinished episodes of one task. Default: one episode per task."""
        episode_id = pending[0]
        destination = staging / episode_id
        drops = self.policy == 'drop'
        if drops and 'drop' in task:
            drop = task['drop']
            store.skip(episode_id, drop['reason'], detail=drop['detail'], reasons=drop.get('reasons'))
            return
        try:
            record = self.episode(task, destination)
        except SourceDataError as exc:
            if not drops:
                raise
            store.skip(episode_id, exc.reason, detail=str(exc), reasons=exc.reasons)
            return
        if record is None:
            store.skip(episode_id, self.skip_reason)
        else:
            store.commit(episode_id, destination, record)
