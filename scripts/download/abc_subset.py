#!/usr/bin/env python3
"""Select ABC-130k episodes using Hub metadata, then download the saved selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
import time
from typing import Callable, TypeVar

REPO_ID = "XDOF/ABC-130k"
ENDPOINT = "https://huggingface.co"
DEFAULT_MANIFEST = Path("dataset/manifests/abc130k-raw-5-seed42.json")
T = TypeVar("T")


def retry(operation: Callable[[], T], retries: int) -> T:
    """Retry transient HTTP/network errors; do not retry authentication failures."""
    import httpx

    for attempt in range(retries + 1):
        try:
            return operation()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in (408, 429) and status < 500:
                raise
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def format_size(size: int) -> str:
    return f"{size / 1024**3:.2f} GiB"


def sample_episodes(episodes: list[str], count: int, seed: int, task: str) -> list[str]:
    population = sorted(set(episodes))
    rng = random.Random(f"{seed}:{task}")
    return sorted(rng.sample(population, min(count, len(population))))


def list_tree(api, revision: str, path: str, retries: int, recursive: bool = False):
    # Materialize inside retry so errors from paginated responses are also retried.
    return retry(
        lambda: list(api.list_repo_tree(
            REPO_ID, path_in_repo=path or None, revision=revision,
            repo_type="dataset", recursive=recursive,
        )),
        retries,
    )


def select_task(api, revision: str, task_path: str, count: int, seed: int, retries: int):
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    task = PurePosixPath(task_path).name
    # One recursive metadata scan avoids an individual request for every selected episode.
    entries = list_tree(api, revision, task_path, retries, recursive=True)
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
    revision = retry(lambda: api.dataset_info(REPO_ID, revision=args.revision, expand=["sha"]).sha, args.retries)
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve the dataset revision to a commit SHA")
    split_path = f"data/{args.split}"
    tasks = sorted(
        entry.path for entry in list_tree(api, revision, split_path, args.retries)
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
    for entry in list_tree(api, revision, "", args.retries):
        if isinstance(entry, RepoFile) and entry.path in ("README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"):
            files.append({"path": entry.path, "size": entry.size})
        elif isinstance(entry, RepoFolder) and entry.path in ("meta", "models", "docs"):
            files.extend(
                {"path": child.path, "size": child.size}
                for child in list_tree(api, revision, entry.path, args.retries, recursive=True)
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


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != REPO_ID:
        raise ValueError("Unsupported manifest schema or dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest must pin a full commit SHA")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest contains no files")
    seen = set()
    for file in files:
        name = file.get("path", "")
        pure = PurePosixPath(name)
        if not name or not pure.parts or pure.is_absolute() or ".." in pure.parts or "\\" in name or pure.as_posix() != name:
            raise ValueError(f"Invalid file path: {name!r}")
        if name in seen or type(file.get("size")) is not int or file["size"] < 0:
            raise ValueError(f"Duplicate file or invalid file size: {name}")
        seen.add(name)
    if manifest.get("total_bytes") != sum(file["size"] for file in files):
        raise ValueError("Manifest total_bytes does not match its files")
    return manifest


def download_file(file: dict, revision: str, output_dir: Path, retries: int) -> None:
    from huggingface_hub import hf_hub_download

    path = Path(retry(lambda: hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=file["path"],
        revision=revision, local_dir=output_dir, endpoint=ENDPOINT,
    ), retries))
    if path.stat().st_size != file["size"]:
        # A truncated local file can still have a valid SDK cache metadata entry.
        path = Path(retry(lambda: hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=file["path"],
            revision=revision, local_dir=output_dir, endpoint=ENDPOINT, force_download=True,
        ), retries))
        if path.stat().st_size != file["size"]:
            raise ValueError(f"File size mismatch after redownload: {file['path']}")


def download(args) -> None:
    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = args.output_dir / "subset_manifest.json"
    if local_manifest.exists():
        previous = load_manifest(local_manifest)
        if (previous["revision"], previous["files"]) != (manifest["revision"], manifest["files"]):
            raise ValueError("Output directory contains a different subset; choose another --output-dir")
    write_json(local_manifest, manifest)
    files = manifest["files"]
    print(f"Downloading {len(files)} files ({format_size(manifest['total_bytes'])}) "
          f"to {args.output_dir}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, file, manifest["revision"], args.output_dir, args.retries): file
            for file in files
        }
        try:
            for index, future in enumerate(as_completed(futures), 1):
                file = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"path": file["path"], "error": str(exc)})
                    print(f"FAILED {file['path']}: {exc}", file=sys.stderr, flush=True)
                if index % 100 == 0 or index == len(files):
                    print(f"[{index}/{len(files)}] completed; {len(failures)} failures", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    write_json(args.output_dir / "download_failures.json", {"files": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed. Rerun the same download command to retry.")
    print("Download complete.", flush=True)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("select", select), ("download", download)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--workers", type=positive_int, default=4)
        command.add_argument("--retries", type=int, default=3, help="additional attempts for transient failures")
        if name == "select":
            command.add_argument("--per-task", type=positive_int, default=30)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--split", choices=("train", "val"), default="train")
            command.add_argument("--revision", default="main")
            command.add_argument("--task", action="append", help="exact task directory name; repeat to select several")
        else:
            command.add_argument("--output-dir", type=Path, default=Path("data/ABC-130k-subset"))
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    # Also route OAuth refresh to the official Hub when the shell defines a mirror endpoint.
    os.environ["HF_ENDPOINT"] = ENDPOINT
    try:
        args.function(args)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the command to continue.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
