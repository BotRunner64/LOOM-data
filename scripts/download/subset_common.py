"""Shared helpers for pinned Hugging Face subsets (no full-repository downloads)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import re

from scripts.download.agibotworld_subset import retry, write_json, positive_int, format_size

ENDPOINT = "https://huggingface.co"


def configure_hub() -> None:
    os.environ["HF_ENDPOINT"] = ENDPOINT
    # httpx supports socks5://, whereas some shell configurations use socks://.
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(key, "").startswith("socks://"):
            os.environ[key] = os.environ[key].replace("socks://", "socks5://", 1)


def safe_path(name: str) -> str:
    pure = PurePosixPath(name)
    if (not name or pure.is_absolute() or ".." in pure.parts or "\\" in name
            or pure.as_posix() != name or name == "."):
        raise ValueError(f"Invalid relative path: {name!r}")
    return name


def load_manifest(path: Path, repo_id: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != repo_id:
        raise ValueError("Unsupported manifest schema or dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest must pin a full commit SHA")
    if not manifest.get("files"):
        raise ValueError("Manifest contains no files")
    seen = set()
    for file in manifest["files"]:
        name = safe_path(file["path"])
        if name in seen or type(file.get("size")) is not int or file["size"] < 0:
            raise ValueError(f"Duplicate file or invalid size: {name}")
        seen.add(name)
    if manifest["total_bytes"] != sum(file["size"] for file in manifest["files"]):
        raise ValueError("Manifest total_bytes does not match its files")
    return manifest


def prepare_output(manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local = output_dir / "subset_manifest.json"
    if local.exists():
        previous = json.loads(local.read_text())
        if (previous["revision"], previous["files"]) != (manifest["revision"], manifest["files"]):
            raise ValueError("Output directory contains another subset; choose another --output-dir")
    write_json(local, manifest)


def download_files(manifest: dict, output_dir: Path, workers: int, retries: int) -> None:
    from huggingface_hub import hf_hub_download

    def fetch(file: dict) -> None:
        def run(force: bool = False):
            return hf_hub_download(
                repo_id=manifest["repo_id"], filename=file["path"], repo_type="dataset",
                revision=manifest["revision"], local_dir=output_dir, endpoint=ENDPOINT,
                force_download=force,
            )
        path = Path(retry(run, retries))
        if path.stat().st_size != file["size"]:
            path = Path(retry(lambda: run(True), retries))
            if path.stat().st_size != file["size"]:
                raise ValueError(f"Size mismatch: {file['path']}")
        print(f"Downloaded {file['path']} ({format_size(file['size'])})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, manifest["files"]))
