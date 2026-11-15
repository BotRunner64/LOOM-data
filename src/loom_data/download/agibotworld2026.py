"""Select and download a small subset of AgiBot World 2026 trajectories."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
import random
import re

from loom_data.download.hub import (
    ENDPOINT, download_subset, format_size, list_tree, load_manifest, positive_int, resolve_revision, write_json,
)

REPO_ID = "agibot-world/AgiBotWorld2026"
MANIFEST = "agibotworld2026-commercial-5.json"
WORKERS = 4
SCENES = {"CommercialSpaces": "商业场景", "Home": "家居场景", "Industry": "工业场景"}
_FRAME_FILE = re.compile(r"^(\d+)_(\d+)\.tar\.gz$")


def parse_trajectory(path: str, size: int) -> dict | None:
    pure = PurePosixPath(path)
    match = _FRAME_FILE.match(pure.name)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return {
        "path": path,
        "size": size,
        "task": pure.parent.name,
        "file": pure.name,
        "start_frame": start,
        "end_frame": end,
        "length": end - start + 1,
    }


def sample_trajectories(trajectories: list[dict], count: int, seed: int, strategy: str) -> list[dict]:
    if strategy == "smallest":
        return sorted(trajectories, key=lambda item: (item["size"], item["path"]))[:count]
    population = sorted(item["path"] for item in trajectories)
    rng = random.Random(f"{seed}:{REPO_ID}")
    wanted = set(rng.sample(population, min(count, len(population))))
    return sorted((item for item in trajectories if item["path"] in wanted), key=lambda item: item["path"])


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}; use a new path to select again")
    api = HfApi(endpoint=ENDPOINT)
    revision = resolve_revision(api, REPO_ID, args.revision, args.retries)
    prefix = f"{args.theme}/{args.scene}"
    entries = list_tree(api, REPO_ID, revision, prefix, args.retries, recursive=True)
    trajectories = []
    for entry in entries:
        if isinstance(entry, RepoFile) and entry.path.endswith(".tar.gz"):
            parsed = parse_trajectory(entry.path, entry.size)
            if parsed:
                trajectories.append(parsed)
    if not trajectories:
        raise ValueError(f"No trajectories found in {prefix}")
    selected = sample_trajectories(trajectories, args.count, args.seed, args.strategy)
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "theme": args.theme,
        "scene": args.scene,
        "scene_label": SCENES.get(args.scene, args.scene),
        "count": args.count,
        "strategy": args.strategy,
        "seed": args.seed,
        "sampling": (
            "sorted by size ascending" if args.strategy == "smallest"
            else f"random.Random(f'{args.seed}:{REPO_ID}').sample over sorted paths"
        ),
        "available_trajectories": len(trajectories),
        "trajectories": selected,
        "files": [{"path": item["path"], "size": item["size"]} for item in selected],
        "total_bytes": sum(item["size"] for item in selected),
    }
    write_json(args.manifest, manifest)
    print(f"Revision: {revision}", flush=True)
    print(f"Scene: {args.theme}/{args.scene} ({manifest['scene_label']}); "
          f"{len(trajectories)} available trajectories", flush=True)
    for item in selected:
        print(f"  {item['path']}  ({item['length']} frames, {format_size(item['size'])})", flush=True)
    print(f"Saved {args.manifest}: {len(selected)} trajectories, "
          f"{format_size(manifest['total_bytes'])}", flush=True)


def download(args) -> None:
    download_subset(load_manifest(args.manifest, REPO_ID), args.output_dir, args.workers, args.retries)


def add_arguments(command, name) -> None:
    if name == "select":
        command.add_argument("--theme", default="ImitationLearning",
                             choices=("ImitationLearning", "RichInteraction", "ReinforcementLearning"))
        command.add_argument("--scene", default="CommercialSpaces", choices=sorted(SCENES))
        command.add_argument("--count", type=positive_int, default=5)
        command.add_argument("--strategy", choices=("smallest", "random"), default="smallest")
        command.add_argument("--seed", type=int, default=42)


COMMANDS = {"select": select, "download": download}
