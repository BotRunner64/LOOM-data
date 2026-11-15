"""Pinned Hugging Face subsets: shared pieces of every `select` and `download` command.

`select` resolves a revision to a commit SHA, picks a small subset and saves it
as a manifest. `download` fetches exactly the files a manifest lists, so a
subset can be reproduced later.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Callable, TypeVar

ENDPOINT = "https://huggingface.co"
T = TypeVar("T")


def configure_hub() -> None:
    # Also route OAuth refresh to the official Hub when the shell defines a mirror endpoint.
    os.environ["HF_ENDPOINT"] = ENDPOINT
    # httpx supports socks5://, whereas some shell configurations use socks://.
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(key, "").startswith("socks://"):
            os.environ[key] = os.environ[key].replace("socks://", "socks5://", 1)


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


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def safe_path(name: str) -> str:
    pure = PurePosixPath(name)
    if (not name or pure.is_absolute() or ".." in pure.parts or "\\" in name
            or pure.as_posix() != name or name == "."):
        raise ValueError(f"Invalid relative path: {name!r}")
    return name


def resolve_revision(api, repo_id: str, revision: str, retries: int) -> str:
    """Pin a branch or tag to its commit SHA."""
    sha = retry(lambda: api.dataset_info(repo_id, revision=revision, expand=["sha"]).sha, retries)
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Could not resolve the dataset revision to a commit SHA")
    return sha


def list_tree(api, repo_id: str, revision: str, path: str, retries: int, recursive: bool = False):
    # Materialize inside retry so errors from paginated responses are also retried.
    return retry(
        lambda: list(api.list_repo_tree(
            repo_id, path_in_repo=path or None, revision=revision,
            repo_type="dataset", recursive=recursive,
        )),
        retries,
    )


def load_manifest(path: Path, repo_id: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != repo_id:
        raise ValueError("Unsupported manifest schema or dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest must pin a full commit SHA")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest contains no files")
    seen = set()
    for file in files:
        name = safe_path(file.get("path", ""))
        if name in seen or type(file.get("size")) is not int or file["size"] < 0:
            raise ValueError(f"Duplicate file or invalid size: {name}")
        seen.add(name)
    if manifest.get("total_bytes") != sum(file["size"] for file in files):
        raise ValueError("Manifest total_bytes does not match its files")
    return manifest


def prepare_output(manifest: dict, output_dir: Path, key: str = "files") -> None:
    """Record the manifest beside the data; refuse to mix two subsets in one directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    local = output_dir / "subset_manifest.json"
    if local.exists():
        previous = json.loads(local.read_text())
        if (previous["revision"], previous[key]) != (manifest["revision"], manifest[key]):
            raise ValueError("Output directory contains a different subset; choose another --output-dir")
    write_json(local, manifest)


def download_file(repo_id: str, revision: str, file: dict, output_dir: Path | None, retries: int) -> Path:
    """Fetch one pinned file into `output_dir` (or the Hub cache when None); return its path."""
    from huggingface_hub import hf_hub_download

    def fetch(force: bool):
        return Path(retry(lambda: hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=file["path"], revision=revision,
            local_dir=output_dir, endpoint=ENDPOINT, force_download=force,
        ), retries))

    path = fetch(False)
    # A truncated local file can still have a valid SDK cache metadata entry.
    if path.stat().st_size != file["size"]:
        path = fetch(True)
        if path.stat().st_size != file["size"]:
            raise ValueError(f"File size mismatch after redownload: {file['path']}")
    return path


class DownloadError(RuntimeError):
    def __init__(self, failures: list[dict]):
        super().__init__(f"{len(failures)} downloads failed. Rerun the same download command to retry.")
        self.failures = failures


def download_files(manifest: dict, output_dir: Path, workers: int, retries: int, every: int = 1) -> None:
    """Download every file of a manifest; raise DownloadError after trying them all."""
    files = manifest["files"]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, manifest["repo_id"], manifest["revision"], file, output_dir, retries): file
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
                if index % every == 0 or index == len(files):
                    print(f"[{index}/{len(files)}] completed; {len(failures)} failures", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if failures:
        raise DownloadError(failures)


def download_subset(manifest: dict, output_dir: Path, workers: int, retries: int, every: int = 5) -> None:
    """The whole `download` command for subsets that are plain repository files."""
    prepare_output(manifest, output_dir)
    print(f"Downloading {len(manifest['files'])} files ({format_size(manifest['total_bytes'])}) "
          f"to {output_dir}", flush=True)
    try:
        download_files(manifest, output_dir, workers, retries, every)
    except DownloadError as exc:
        write_json(output_dir / "download_failures.json", {"files": exc.failures})
        raise
    write_json(output_dir / "download_failures.json", {"files": []})
    print("Download complete.", flush=True)
