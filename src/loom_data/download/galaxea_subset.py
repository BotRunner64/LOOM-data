#!/usr/bin/env python3
"""Select and download a small subset of Galaxea Open-World Dataset task archives.

The Hub stores this dataset as task-level ``lerobot/<Task>.tar.gz`` archives, each
being a self-contained LeRobot v2.1 dataset holding many episodes (trajectories).
A single trajectory cannot be downloaded on its own, so the subset unit is the
task archive: the smallest archives are selected for a fast, reproducible sample.
"""

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
import tarfile
import time
from typing import Callable, TypeVar

REPO_ID = "OpenGalaxea/Galaxea-Open-World-Dataset"
ENDPOINT = "https://huggingface.co"
DEFAULT_MANIFEST = Path("dataset/manifests/galaxea-open-world-5.json")
ARCHIVE_DIR = "lerobot"
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


def list_tree(api, revision: str, path: str, retries: int):
    # Materialize inside retry so errors from paginated responses are also retried.
    return retry(
        lambda: list(api.list_repo_tree(
            REPO_ID, path_in_repo=path or None, revision=revision,
            repo_type="dataset", recursive=False,
        )),
        retries,
    )


# Archive names mostly end with an optional "_<YYYYMMDD>_<NNN>" collection suffix;
# some omit the underscore before the date (e.g. "Egg_Placement20250703_002").
_ARCHIVE_SUFFIX = re.compile(r"^(?P<task>.+?)_?(?P<date>\d{8})_(?P<version>\d{3})$")


def parse_archive(path: str, size: int) -> dict | None:
    pure = PurePosixPath(path)
    if not pure.name.endswith(".tar.gz"):
        return None
    stem = pure.name[: -len(".tar.gz")]
    match = _ARCHIVE_SUFFIX.match(stem)
    if match:
        task, date, version = match.group("task"), match.group("date"), match.group("version")
    else:
        task, date, version = stem, None, None
    return {
        "path": path,
        "size": size,
        "task": stem,
        "file": pure.name,
        "date": date,
        "version": version,
    }


def sample_tasks(tasks: list[dict], count: int, seed: int, strategy: str) -> list[dict]:
    if strategy == "smallest":
        return sorted(tasks, key=lambda item: (item["size"], item["path"]))[:count]
    population = sorted(item["path"] for item in tasks)
    rng = random.Random(f"{seed}:{REPO_ID}")
    wanted = set(rng.sample(population, min(count, len(population))))
    return sorted((item for item in tasks if item["path"] in wanted), key=lambda item: item["path"])


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}; use a new path to select again")
    api = HfApi(endpoint=ENDPOINT)
    revision = retry(lambda: api.dataset_info(REPO_ID, revision=args.revision, expand=["sha"]).sha, args.retries)
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve the dataset revision to a commit SHA")
    tasks = []
    for entry in list_tree(api, revision, ARCHIVE_DIR, args.retries):
        if isinstance(entry, RepoFile) and entry.path.endswith(".tar.gz"):
            parsed = parse_archive(entry.path, entry.size)
            if parsed:
                tasks.append(parsed)
    if not tasks:
        raise ValueError(f"No task archives found under {ARCHIVE_DIR}/")
    selected = sample_tasks(tasks, args.count, args.seed, args.strategy)
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": args.count,
        "strategy": args.strategy,
        "seed": args.seed,
        "unit": "task_archive",
        "note": "Each archive is a self-contained LeRobot v2.1 task dataset with multiple episodes",
        "sampling": (
            "sorted by size ascending" if args.strategy == "smallest"
            else f"random.Random(f'{args.seed}:{REPO_ID}').sample over sorted paths"
        ),
        "available_tasks": len(tasks),
        "tasks": selected,
        "files": [{"path": item["path"], "size": item["size"]} for item in selected],
        "total_bytes": sum(item["size"] for item in selected),
    }
    write_json(args.manifest, manifest)
    print(f"Revision: {revision}", flush=True)
    print(f"{len(tasks)} task archives available", flush=True)
    for item in selected:
        print(f"  {item['path']}  ({format_size(item['size'])})", flush=True)
    print(f"Saved {args.manifest}: {len(selected)} task archives, "
          f"{format_size(manifest['total_bytes'])}", flush=True)


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
    print(f"Downloading {len(files)} task archives ({format_size(manifest['total_bytes'])}) "
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
                if index % 5 == 0 or index == len(files):
                    print(f"[{index}/{len(files)}] completed; {len(failures)} failures", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    write_json(args.output_dir / "download_failures.json", {"files": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed. Rerun the same download command to retry.")
    print("Download complete.", flush=True)


def extract_archive(file: dict, output_dir: Path) -> Path:
    archive = output_dir / file["path"]
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(output_dir, filter="data")
        except TypeError:  # Python < 3.12 has no extraction filter.
            root = output_dir.resolve()
            for member in tar.getmembers():
                target = (output_dir / member.name).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(output_dir)
    return archive


def extract(args) -> None:
    manifest = load_manifest(args.manifest)
    for index, file in enumerate(manifest["files"], 1):
        extract_archive(file, args.output_dir)
        print(f"[{index}/{len(manifest['files'])}] extracted {file['path']}", flush=True)
    print("Extraction complete.", flush=True)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("select", select), ("download", download), ("extract", extract)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--workers", type=positive_int, default=4)
        command.add_argument("--retries", type=int, default=3, help="additional attempts for transient failures")
        if name == "select":
            command.add_argument("--count", type=positive_int, default=5)
            command.add_argument("--strategy", choices=("smallest", "random"), default="smallest")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--revision", default="main")
        else:
            command.add_argument("--output-dir", type=Path,
                                 default=Path("dataset/raw/Galaxea-Open-World-Dataset"))
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
