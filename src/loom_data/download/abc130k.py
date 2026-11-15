"""Select ABC-130k episodes using Hub metadata, then download the saved selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import PurePosixPath
import random

from loom_data.download.hub import (
    ENDPOINT, download_subset, format_size, list_tree, load_manifest, positive_int, resolve_revision, write_json,
)

REPO_ID = "XDOF/ABC-130k"
MANIFEST = "abc130k-raw-5-seed42.json"
WORKERS = 4


def sample_episodes(episodes: list[str], count: int, seed: int, task: str) -> list[str]:
    population = sorted(set(episodes))
    rng = random.Random(f"{seed}:{task}")
    return sorted(rng.sample(population, min(count, len(population))))


def select_task(api, revision: str, task_path: str, count: int, seed: int, retries: int):
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    task = PurePosixPath(task_path).name
    # One recursive metadata scan avoids an individual request for every selected episode.
    entries = list_tree(api, REPO_ID, revision, task_path, retries, recursive=True)
    episodes = [
        entry.path for entry in entries
        if isinstance(entry, RepoFolder)
        and PurePosixPath(entry.path).parent.as_posix() == task_path
        and PurePosixPath(entry.path).name.startswith("episode_")
    ]
    if not episodes:
        raise ValueError(f"No episode directories found in {task_path}")
    selected = sample_episodes(episodes, count, seed, task)
    wanted = set(selected)
    files = [
        {"path": entry.path, "size": entry.size}
        for entry in entries
        if isinstance(entry, RepoFile)
        and PurePosixPath(entry.path).parent.as_posix() in wanted
        and PurePosixPath(entry.path).name in ("episode.mcap", "annotation.mcap")
    ]
    paths = {file["path"] for file in files}
    for episode in selected:
        if f"{episode}/episode.mcap" not in paths:
            raise ValueError(f"Selected episode is missing episode.mcap: {episode}")
    return {
        "task": task,
        "available_episodes": len(episodes),
        "selected_episodes": selected,
    }, files


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}; use a new path to select again")
    api = HfApi(endpoint=ENDPOINT)
    revision = resolve_revision(api, REPO_ID, args.revision, args.retries)
    split_path = f"data/{args.split}"
    tasks = sorted(
        entry.path for entry in list_tree(api, REPO_ID, revision, split_path, args.retries)
        if isinstance(entry, RepoFolder)
    )
    if args.task:
        wanted = {f"{split_path}/{task}" for task in args.task}
        missing = wanted - set(tasks)
        if missing:
            raise ValueError(f"Unknown task directories: {', '.join(sorted(missing))}")
        tasks = [task for task in tasks if task in wanted]
    if not tasks:
        raise ValueError(f"No task directories found in {split_path}")
    print(f"Revision: {revision}\nScanning {len(tasks)} tasks (metadata only)", flush=True)
    selections, files = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(select_task, api, revision, task, args.per_task, args.seed, args.retries): task
            for task in tasks
        }
        try:
            for index, future in enumerate(as_completed(futures), 1):
                selection, task_files = future.result()
                selections.append(selection)
                files.extend(task_files)
                print(f"[{index}/{len(tasks)}] {selection['task']}: "
                      f"{len(selection['selected_episodes'])}/{selection['available_episodes']} episodes", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    # Preserve dataset documentation and shared metadata/model assets alongside the episodes.
    for entry in list_tree(api, REPO_ID, revision, "", args.retries):
        if isinstance(entry, RepoFile) and entry.path in ("README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"):
            files.append({"path": entry.path, "size": entry.size})
        elif isinstance(entry, RepoFolder) and entry.path in ("meta", "models", "docs"):
            files.extend(
                {"path": child.path, "size": child.size}
                for child in list_tree(api, REPO_ID, revision, entry.path, args.retries, recursive=True)
                if isinstance(child, RepoFile)
            )
    selections.sort(key=lambda selection: selection["task"])
    files.sort(key=lambda file: file["path"])
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "per_task": args.per_task,
        "seed": args.seed,
        "sampling": "sorted episode paths; random.Random(f'{seed}:{task}').sample",
        "tasks": selections,
        "files": files,
        "total_episodes": sum(len(task["selected_episodes"]) for task in selections),
        "total_bytes": sum(file["size"] for file in files),
    }
    write_json(args.manifest, manifest)
    print(f"Saved {args.manifest}: {len(selections)} tasks, {manifest['total_episodes']} episodes, "
          f"{len(files)} files, {format_size(manifest['total_bytes'])}", flush=True)


def download(args) -> None:
    download_subset(load_manifest(args.manifest, REPO_ID), args.output_dir, args.workers, args.retries, every=100)


def add_arguments(command, name) -> None:
    if name == "select":
        command.add_argument("--per-task", type=positive_int, default=30)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--split", choices=("train", "val"), default="train")
        command.add_argument("--task", action="append", help="exact task directory name; repeat to select several")


COMMANDS = {"select": select, "download": download}
