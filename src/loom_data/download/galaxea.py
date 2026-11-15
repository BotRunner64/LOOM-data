"""Select and download a small subset of Galaxea Open-World Dataset task archives.

The Hub stores this dataset as task-level ``lerobot/<Task>.tar.gz`` archives, each
being a self-contained LeRobot v2.1 dataset holding many episodes (trajectories).
A single trajectory cannot be downloaded on its own, so the subset unit is the
task archive: the smallest archives are selected for a fast, reproducible sample.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import random
import re
import tarfile

from loom_data.download.hub import (
    ENDPOINT, download_subset, format_size, list_tree, load_manifest, positive_int, resolve_revision, write_json,
)

REPO_ID = "OpenGalaxea/Galaxea-Open-World-Dataset"
MANIFEST = "galaxea-open-world-5.json"
WORKERS = 4
ARCHIVE_DIR = "lerobot"
# Archive names mostly end with an optional "_<YYYYMMDD>_<NNN>" collection suffix;
# some omit the underscore before the date (e.g. "Egg_Placement20250703_002").
_ARCHIVE_SUFFIX = re.compile(r"^(?P<task>.+?)_?(?P<date>\d{8})_(?P<version>\d{3})$")


def parse_archive(path: str, size: int) -> dict | None:
    pure = PurePosixPath(path)
    if not pure.name.endswith(".tar.gz"):
        return None
    stem = pure.name[: -len(".tar.gz")]
    match = _ARCHIVE_SUFFIX.match(stem)
    date, version = (match.group("date"), match.group("version")) if match else (None, None)
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
    revision = resolve_revision(api, REPO_ID, args.revision, args.retries)
    tasks = []
    for entry in list_tree(api, REPO_ID, revision, ARCHIVE_DIR, args.retries):
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


def download(args) -> None:
    download_subset(load_manifest(args.manifest, REPO_ID), args.output_dir, args.workers, args.retries)


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
    manifest = load_manifest(args.manifest, REPO_ID)
    for index, file in enumerate(manifest["files"], 1):
        extract_archive(file, args.output_dir)
        print(f"[{index}/{len(manifest['files'])}] extracted {file['path']}", flush=True)
    print("Extraction complete.", flush=True)


def add_arguments(command, name) -> None:
    if name == "select":
        command.add_argument("--count", type=positive_int, default=5)
        command.add_argument("--strategy", choices=("smallest", "random"), default="smallest")
        command.add_argument("--seed", type=int, default=42)


COMMANDS = {"select": select, "download": download, "extract": extract}
