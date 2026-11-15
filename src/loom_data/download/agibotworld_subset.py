#!/usr/bin/env python3
"""Select and download a small subset of AgiBot World 2026 trajectories."""

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

REPO_ID = "agibot-world/AgiBotWorld2026"
ENDPOINT = "https://huggingface.co"
DEFAULT_MANIFEST = Path("dataset/manifests/agibotworld2026-commercial-5.json")
SCENES = {"CommercialSpaces": "商业场景", "Home": "家居场景", "Industry": "工业场景"}
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
            repo_type="dataset", recursive=True,
        )),
        retries,
    )


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
    revision = retry(lambda: api.dataset_info(REPO_ID, revision=args.revision, expand=["sha"]).sha, args.retries)
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve the dataset revision to a commit SHA")
    prefix = f"{args.theme}/{args.scene}"
    entries = list_tree(api, revision, prefix, args.retries)
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
            command.add_argument("--theme", default="ImitationLearning",
                                 choices=("ImitationLearning", "RichInteraction", "ReinforcementLearning"))
            command.add_argument("--scene", default="CommercialSpaces", choices=sorted(SCENES))
            command.add_argument("--count", type=positive_int, default=5)
            command.add_argument("--strategy", choices=("smallest", "random"), default="smallest")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--revision", default="main")
        else:
            command.add_argument("--output-dir", type=Path, default=Path("dataset/raw/AgiBotWorld2026"))
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
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
