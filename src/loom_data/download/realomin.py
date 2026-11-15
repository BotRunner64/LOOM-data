"""Select and download complete MCAP trajectories from 10Kh-RealOmin-OpenData."""
from __future__ import annotations

from datetime import datetime, timezone
import random

from loom_data.download.hub import (
    ENDPOINT, download_subset, format_size, load_manifest, positive_int, resolve_revision, retry, safe_path,
    write_json,
)

REPO_ID = "genrobot2025/10Kh-RealOmin-OpenData"
MANIFEST = "realomin-clean-bowl-5.json"
WORKERS = 3


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}")
    api = HfApi(endpoint=ENDPOINT)
    revision = resolve_revision(api, REPO_ID, args.revision, args.retries)
    entries = retry(lambda: list(api.list_repo_tree(
        REPO_ID, path_in_repo=safe_path(args.prefix), repo_type="dataset",
        revision=revision, recursive=True,
    )), args.retries)
    candidates = [{"path": f.path, "size": f.size} for f in entries
                  if isinstance(f, RepoFile) and f.path.endswith(".mcap")]
    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} MCAP files available, requested {args.count}")
    if args.strategy == "smallest":
        selected = sorted(candidates, key=lambda f: (f["size"], f["path"]))[:args.count]
    else:
        selected = random.Random(f"{args.seed}:{REPO_ID}").sample(
            sorted(candidates, key=lambda f: f["path"]), args.count)
    selected = sorted(selected, key=lambda f: f["path"])
    root = retry(lambda: list(api.list_repo_tree(REPO_ID, repo_type="dataset", revision=revision)), args.retries)
    asset = next(f for f in root if f.path == "DAS_Gripper_V3.zip")
    files = selected + [{"path": asset.path, "size": asset.size}]
    manifest = {
        "schema_version": 1, "repo_id": REPO_ID, "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(), "prefix": args.prefix,
        "count": args.count, "strategy": args.strategy, "seed": args.seed,
        "available_trajectories": len(candidates), "trajectories": selected,
        "files": files, "total_bytes": sum(f["size"] for f in files),
    }
    write_json(args.manifest, manifest)
    for f in selected:
        print(f"  {f['path']} ({format_size(f['size'])})", flush=True)
    print(f"Saved {args.manifest}: {len(selected)} trajectories, {format_size(manifest['total_bytes'])}", flush=True)


def download(args) -> None:
    download_subset(load_manifest(args.manifest, REPO_ID), args.output_dir, args.workers, args.retries, every=1)


def add_arguments(command, name) -> None:
    if name == "select":
        command.add_argument("--prefix", default="Cooking_and_Kitchen_Clean/clean_bowl/00001")
        command.add_argument("--count", type=positive_int, default=5)
        command.add_argument("--strategy", choices=("smallest", "random"), default="random")
        command.add_argument("--seed", type=int, default=42)


COMMANDS = {"select": select, "download": download}
