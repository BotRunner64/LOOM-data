"""Select and download a small HiFi-UMI-2K trajectory (episode) subset.

HiFi-UMI-2K is published as hundreds of independent LeRobot v3 shards
(``chunk-XXXX/part-0000``). Every shard holds many episodes and concatenates the
six camera views of those episodes into one MP4 per view, so a single trajectory
is not a standalone file. This tool therefore picks the smallest shard, samples
a reproducible set of episodes from it, downloads only the small shard metadata
and frame table, and cuts the selected episode segments out of the remote MP4s
with HTTP range requests.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re

from loom_data.download import episodes as cutting
from loom_data.download.episodes import extract_videos, fetch_file, prepare_output, sample_episodes
from loom_data.download.hub import ENDPOINT, format_size, retry, write_json

REPO_ID = "simple-world-lab/HiFi-UMI-2K"
MANIFEST = "hifi-umi-2k-5.json"
WORKERS = 3
VIDEO_KEYS = (
    "observation.images.head_main",
    "observation.images.head_main_stereo_right",
    "observation.images.left_hand_up",
    "observation.images.left_hand_down",
    "observation.images.right_hand_up",
    "observation.images.right_hand_down",
)


def shard_name(path: str) -> str | None:
    """Return ``chunk-XXXX/part-YYYY`` for a file inside a shard, else ``None``."""
    parts = PurePosixPath(path).parts
    if len(parts) >= 2 and parts[0].startswith("chunk-") and parts[1].startswith("part-"):
        return f"{parts[0]}/{parts[1]}"
    return None


def collect_shards(siblings: list) -> list[dict]:
    """Aggregate file siblings into shard records sorted by ascending size."""
    groups: dict[str, dict] = {}
    for sibling in siblings:
        name = shard_name(sibling.rfilename)
        if name is None:
            continue
        group = groups.setdefault(name, {"shard": name, "total_bytes": 0, "file_count": 0})
        group["total_bytes"] += sibling.size or 0
        group["file_count"] += 1
    return sorted(groups.values(), key=lambda item: (item["total_bytes"], item["shard"]))


def pick_shard(shards: list[dict], name: str | None) -> dict:
    if not shards:
        raise ValueError("Dataset contains no shards")
    if name is None:
        return shards[0]
    for shard in shards:
        if shard["shard"] == name:
            return shard
    raise ValueError(f"Unknown shard: {name}")


def classify_shard_files(siblings: list, shard: str) -> dict[str, dict]:
    """Map the six schema roles of a shard to their exact sibling paths and sizes."""
    roles = {"info": [], "modality": [], "stats": [], "tasks": [], "episodes": [], "data": []}
    videos: dict[str, dict] = {}
    prefix = shard + "/"
    for sibling in siblings:
        path = sibling.rfilename
        if not path.startswith(prefix):
            continue
        size = sibling.size or 0
        if path.endswith("/meta/info.json"):
            roles["info"].append({"path": path, "size": size})
        elif path.endswith("/meta/modality.json"):
            roles["modality"].append({"path": path, "size": size})
        elif path.endswith("/meta/stats.json"):
            roles["stats"].append({"path": path, "size": size})
        elif path.endswith("/meta/tasks.parquet"):
            roles["tasks"].append({"path": path, "size": size})
        elif "/meta/episodes/" in path and path.endswith(".parquet"):
            roles["episodes"].append({"path": path, "size": size})
        elif "/data/" in path and path.endswith(".parquet"):
            roles["data"].append({"path": path, "size": size})
        elif "/videos/" in path and path.endswith(".mp4"):
            key = PurePosixPath(path).parts[3]
            videos.setdefault(key, {"path": path, "size": size})
    result: dict[str, dict] = {}
    for role, entries in roles.items():
        if len(entries) != 1:
            raise ValueError(f"Expected exactly one {role} file in {shard}, found {len(entries)}")
        result[role] = entries[0]
    missing = [key for key in VIDEO_KEYS if key not in videos]
    if missing:
        raise ValueError(f"Shard {shard} is missing camera views: {', '.join(missing)}")
    result["videos"] = {key: videos[key] for key in VIDEO_KEYS}
    return result


def select(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    if args.manifest.exists():
        raise ValueError(f"Manifest already exists: {args.manifest}; use a new path to select again")
    api = HfApi(endpoint=ENDPOINT)
    info = retry(lambda: api.dataset_info(REPO_ID, revision=args.revision, files_metadata=True), args.retries)
    revision = info.sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve the dataset revision to a commit SHA")
    shards = collect_shards(list(info.siblings))
    shard = pick_shard(shards, args.shard)
    files = classify_shard_files(list(info.siblings), shard["shard"])

    info_path = retry(lambda: hf_hub_download(
        REPO_ID, files["info"]["path"], repo_type="dataset", revision=revision, endpoint=ENDPOINT,
    ), args.retries)
    info_json = json.loads(Path(info_path).read_text())
    episodes_path = retry(lambda: hf_hub_download(
        REPO_ID, files["episodes"]["path"], repo_type="dataset", revision=revision, endpoint=ENDPOINT,
    ), args.retries)

    fps = float(info_json["fps"])
    total_frames = int(info_json["total_frames"])
    episodes = []
    for row in pq.read_table(episodes_path).to_pylist():
        videos = {}
        for key in VIDEO_KEYS:
            videos[key] = {
                "path": files["videos"][key]["path"],
                "size": files["videos"][key]["size"],
                "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
                "to_timestamp": float(row[f"videos/{key}/to_timestamp"]),
            }
        episodes.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "tasks": list(row["tasks"]),
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
            "videos": videos,
        })
    if not episodes:
        raise ValueError(f"No episodes found in {shard['shard']}")
    for episode in episodes:
        if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
            raise ValueError(f"Episode {episode['episode_index']} has inconsistent index range")
    selected = sample_episodes(episodes, args.count, args.seed, args.strategy, REPO_ID)

    estimated_video_bytes = 0
    for episode in selected:
        for key in VIDEO_KEYS:
            estimated_video_bytes += round(
                episode["videos"][key]["size"] * episode["length"] / total_frames
            )
    manifest_files = [
        {"role": role, "path": files[role]["path"], "size": files[role]["size"]}
        for role in ("info", "modality", "stats", "tasks", "episodes", "data")
    ]
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard": shard["shard"],
        "count": args.count,
        "strategy": args.strategy,
        "seed": args.seed,
        "fps": fps,
        "video_keys": list(VIDEO_KEYS),
        "sampling": (
            "sorted by episode length ascending" if args.strategy == "smallest"
            else f"random.Random(f'{args.seed}:{REPO_ID}').sample over sorted episode_index"
        ),
        "available_shards": len(shards),
        "shard_bytes": shard["total_bytes"],
        "available_episodes": len(episodes),
        "episodes": selected,
        "files": manifest_files,
        "total_bytes": sum(file["size"] for file in manifest_files),
        "estimated_video_bytes": estimated_video_bytes,
    }
    write_json(args.manifest, manifest)
    print(f"Revision: {revision}", flush=True)
    print(f"Shard: {shard['shard']} ({format_size(shard['total_bytes'])}); "
          f"{len(episodes)} episodes available", flush=True)
    for episode in selected:
        print(f"  episode_{episode['episode_index']:06d}: {episode['length']} frames, "
              f"{episode['tasks'][0] if episode['tasks'] else '(no task)'}", flush=True)
    print(f"Saved {args.manifest}: {len(selected)} trajectories, "
          f"metadata {format_size(manifest['total_bytes'])}, "
          f"estimated video {format_size(estimated_video_bytes)}", flush=True)


def load_manifest(path: Path) -> dict:
    manifest = cutting.load_manifest(path, REPO_ID, VIDEO_KEYS)
    shard = manifest.get("shard")
    if not isinstance(shard, str) or shard_name(shard) != shard:
        raise ValueError(f"Invalid shard: {shard!r}")
    return manifest


def write_episode_tables(manifest: dict, data_table, output_dir: Path) -> None:
    import pyarrow.parquet as pq

    for episode in manifest["episodes"]:
        start = episode["dataset_from_index"]
        subset = data_table.slice(start, episode["length"])
        episode_dir = output_dir / "episodes" / f"episode_{episode['episode_index']:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(subset, episode_dir / "data.parquet")
        write_json(episode_dir / "episode.json", {
            "episode_index": episode["episode_index"],
            "length": episode["length"],
            "tasks": episode["tasks"],
            "dataset_from_index": episode["dataset_from_index"],
            "dataset_to_index": episode["dataset_to_index"],
            "fps": manifest["fps"],
            "videos": episode["videos"],
        })


def download(args) -> None:
    import pyarrow.parquet as pq

    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir
    prepare_output(manifest, output_dir)

    print(f"Downloading {len(manifest['files'])} metadata files "
          f"({format_size(manifest['total_bytes'])}) for {manifest['shard']}", flush=True)
    fetched = {}
    for file in manifest["files"]:
        destination = output_dir / "source" / f"{file['role']}{PurePosixPath(file['path']).suffix}"
        fetch_file(REPO_ID, file, manifest["revision"], destination, args.retries)
        fetched[file["role"]] = destination
    write_episode_tables(manifest, pq.read_table(fetched["data"]), output_dir)
    print(f"Wrote {len(manifest['episodes'])} episode frame tables", flush=True)

    extract_videos(manifest, output_dir, args)


def add_arguments(command, name) -> None:
    cutting.add_arguments(command, name)
    if name == "select":
        command.add_argument("--shard", help="exact chunk-XXXX/part-YYYY (default: the smallest)")


COMMANDS = {"select": select, "download": download}
