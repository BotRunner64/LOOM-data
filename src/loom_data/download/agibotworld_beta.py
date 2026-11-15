"""Download a small, complete AgiBot World Beta subset from bounded TAR shards."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from loom_data import paths
from loom_data.download.hub import (
    ENDPOINT, download_files, format_size, load_manifest, positive_int, prepare_output, resolve_revision, retry,
    safe_path, write_json,
)

REPO_ID = "agibot-world/AgiBotWorld-Beta"
MANIFEST = "agibotworld-beta-5.json"
WORKERS = 2


def state_members(archive: Path) -> dict[tuple[int, int], dict]:
    result = {}
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[-1] == "proprio_stats.h5":
                safe_path(member.name)
                key = (int(parts[0]), int(parts[1]))
                if key in result:
                    raise ValueError(f"Duplicate state member: {key}")
                result[key] = {"member": member.name, "size": member.size,
                               "destination": f"proprio_stats/{member.name}"}
    return result


def video_members(archive: Path, task_id: int) -> dict[tuple[int, int], list[dict]]:
    result: dict[tuple[int, int], list[dict]] = {}
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[1] == "videos" and parts[2].endswith(".mp4"):
                safe_path(member.name)
                key = (task_id, int(parts[0]))
                result.setdefault(key, []).append({
                    "member": member.name, "size": member.size,
                    "destination": f"observations/{task_id}/{member.name}",
                })
    return result


def extract_members(archive: Path, members: list[dict], output_dir: Path) -> None:
    """Copy only explicitly selected regular files; never extract archive paths or links."""
    with tarfile.open(archive, "r:") as tar:
        for item in members:
            member = tar.getmember(safe_path(item["member"]))
            if not member.isfile() or member.size != item["size"]:
                raise ValueError(f"Unexpected archive member: {member.name}")
            destination = output_dir / safe_path(item["destination"])
            if destination.is_file() and destination.stat().st_size == member.size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable member: {member.name}")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if temporary.stat().st_size != member.size:
                raise ValueError(f"Truncated member: {member.name}")
            temporary.replace(destination)


def select(args) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}")
    api = HfApi(endpoint=ENDPOINT)
    revision = resolve_revision(api, REPO_ID, args.revision, args.retries)

    def tree(prefix: str) -> list[dict]:
        return [{"path": f.path, "size": f.size} for f in retry(lambda: list(api.list_repo_tree(
            REPO_ID, path_in_repo=prefix, repo_type="dataset", revision=revision, recursive=True,
        )), args.retries) if isinstance(f, RepoFile) and f.path.endswith(".tar")]

    states = tree("proprio_stats")
    if args.state_archive:
        state = next(f for f in states if f["path"] == args.state_archive)
    else:
        state = min(states, key=lambda f: (f["size"], f["path"]))
    budget = int(args.max_download_gib * 1024**3)
    if state["size"] > budget:
        raise ValueError("State shard exceeds --max-download-gib")
    staging = args.output_dir / ".archives"

    def fetch(file: dict) -> Path:
        download_files({"repo_id": REPO_ID, "revision": revision, "files": [file]}, staging, 1, args.retries)
        return staging / file["path"]

    print(f"Indexing state shard {state['path']} ({format_size(state['size'])})", flush=True)
    state_index = state_members(fetch(state))
    candidates = []
    for archive in tree("observations"):
        task = int(PurePosixPath(archive["path"]).parts[1])
        start, end = map(int, PurePosixPath(archive["path"]).stem.split("-"))
        if any(t == task and start <= e <= end for t, e in state_index):
            candidates.append(archive)
    files = [{**state, "role": "archive"}]
    selected: dict[tuple[int, int], dict] = {}
    total = state["size"]
    for archive in sorted(candidates, key=lambda f: (f["size"], f["path"])):
        if total + archive["size"] > budget:
            continue
        task = int(PurePosixPath(archive["path"]).parts[1])
        videos = video_members(fetch(archive), task)
        total += archive["size"]
        available = sorted(set(videos).intersection(state_index).difference(selected),
                           key=lambda key: (sum(v["size"] for v in videos[key]), key))
        if not available:
            print(f"No exact state/video matches in {archive['path']}; checking the next shard", flush=True)
            continue
        files.append({**archive, "role": "archive"})
        for key in available[:args.count - len(selected)]:
            selected[key] = {
                "task_id": key[0], "episode_id": key[1],
                "state": {**state_index[key], "archive": state["path"]},
                "videos": [{**v, "archive": archive["path"]} for v in videos[key]],
            }
        if len(selected) == args.count:
            break
    if len(selected) < args.count:
        raise ValueError(f"Found only {len(selected)} complete episodes within --max-download-gib; increase the budget")
    tasks = {f.path: f.size for f in retry(lambda: list(api.list_repo_tree(
        REPO_ID, path_in_repo="task_info", repo_type="dataset", revision=revision,
    )), args.retries) if isinstance(f, RepoFile)}
    for task in sorted({key[0] for key in selected}):
        path = f"task_info/task_{task}.json"
        files.append({"path": path, "size": tasks[path], "role": "metadata"})
    manifest = {
        "schema_version": 1, "repo_id": REPO_ID, "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(), "count": args.count,
        "sampling": "smallest state shard, then smallest matching observation shards and video payloads",
        "files": files, "total_bytes": sum(f["size"] for f in files),
        "episodes": [selected[key] for key in sorted(selected)],
    }
    manifest["selected_bytes"] = sum(
        e["state"]["size"] + sum(v["size"] for v in e["videos"]) for e in manifest["episodes"])
    write_json(args.manifest, manifest)
    print(f"Saved {args.manifest}: {len(selected)} episodes, source shards {format_size(manifest['total_bytes'])}, "
          f"selected payload {format_size(manifest['selected_bytes'])}", flush=True)


def download(args) -> None:
    manifest = load_manifest(args.manifest, REPO_ID)
    if len(manifest.get("episodes", [])) != manifest.get("count"):
        raise ValueError("Manifest episode count does not match")
    prepare_output(manifest, args.output_dir)
    archives = [f for f in manifest["files"] if f["role"] == "archive"]
    metadata = [f for f in manifest["files"] if f["role"] == "metadata"]
    download_files({**manifest, "files": archives}, args.output_dir / ".archives", args.workers, args.retries)
    download_files({**manifest, "files": metadata}, args.output_dir, args.workers, args.retries)
    grouped: dict[str, list[dict]] = {}
    for episode in manifest["episodes"]:
        for item in [episode["state"], *episode["videos"]]:
            grouped.setdefault(safe_path(item["archive"]), []).append(item)
    if set(grouped) != {f["path"] for f in archives}:
        raise ValueError("Episode members reference unexpected archives")
    for archive, members in grouped.items():
        extract_members(args.output_dir / ".archives" / archive, members, args.output_dir)
    # Preserve upstream annotations without restricting the task JSON to the subset.
    for episode in manifest["episodes"]:
        rows = json.loads((args.output_dir / f"task_info/task_{episode['task_id']}.json").read_text())
        if not any(int(row["episode_id"]) == episode["episode_id"] for row in rows):
            raise ValueError(f"Missing annotation for episode {episode['episode_id']}")
    write_json(args.output_dir / "download_failures.json", {"files": []})
    print(f"Download complete: {len(manifest['episodes'])} episodes (HDF5 + all RGB videos + task annotations).", flush=True)


def add_arguments(command, name) -> None:
    if name == "select":
        # Selection has to open the shards, so it downloads them into the output directory.
        command.add_argument("--output-dir", type=Path, default=paths.RAW / "AgiBotWorld-Beta")
        command.add_argument("--count", type=positive_int, default=5)
        command.add_argument("--state-archive", help="exact proprio_stats/*.tar path (default: the smallest)")
        command.add_argument("--max-download-gib", type=float, default=4,
                             help="budget for the state and observation shards downloaded while selecting")


COMMANDS = {"select": select, "download": download}
