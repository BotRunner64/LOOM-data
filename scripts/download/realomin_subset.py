#!/usr/bin/env python3
"""Select and download complete MCAP trajectories from 10Kh-RealOmin-OpenData."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.download.subset_common import (
    ENDPOINT, configure_hub, download_files, format_size, load_manifest,
    positive_int, prepare_output, retry, safe_path, write_json,
)

REPO_ID = "genrobot2025/10Kh-RealOmin-OpenData"
DEFAULT_MANIFEST = Path("dataset/manifests/realomin-clean-bowl-5.json")


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}")
    api = HfApi(endpoint=ENDPOINT)
    revision = retry(lambda: api.dataset_info(REPO_ID, revision=args.revision, expand=["sha"]).sha, args.retries)
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
    manifest = load_manifest(args.manifest, REPO_ID)
    prepare_output(manifest, args.output_dir)
    download_files(manifest, args.output_dir, args.workers, args.retries)
    write_json(args.output_dir / "download_failures.json", {"files": []})
    print("Download complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("select", select), ("download", download)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--retries", type=int, default=3)
        if name == "select":
            command.add_argument("--prefix", default="Cooking_and_Kitchen_Clean/clean_bowl/00001")
            command.add_argument("--count", type=positive_int, default=5)
            command.add_argument("--strategy", choices=("smallest", "random"), default="random")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--revision", default="main")
        else:
            command.add_argument("--output-dir", type=Path, default=Path("dataset/raw/10Kh-RealOmin-OpenData"))
            command.add_argument("--workers", type=positive_int, default=3)
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    configure_hub()
    args.function(args)


if __name__ == "__main__":
    main()
