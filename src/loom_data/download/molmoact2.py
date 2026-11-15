"""Select and download a small MolmoAct2-BimanualYAM trajectory (episode) subset.

MolmoAct2-BimanualYAM is a single merged LeRobot v3 dataset: episode frame tables
live in ``data/chunk-*/*.parquet`` (one file may hold several episodes), while
each camera view concatenates many episodes into one big MP4 per video file.
A single trajectory is therefore not a standalone file.

This tool samples a reproducible set of episodes, downloads only the relevant
(and small) metadata plus the data files that hold the chosen episodes, and
cuts each episode segment out of the remote MP4s with HTTP range requests.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from loom_data.download import episodes as cutting
from loom_data.download.episodes import extract_videos, fetch_file, prepare_output, sample_episodes
from loom_data.download.hub import ENDPOINT, format_size, retry, write_json

REPO_ID = "allenai/MolmoAct2-BimanualYAM-Dataset"
MANIFEST = "molmoact2-bimanualyam-5.json"
WORKERS = 3
VIDEO_KEYS = (
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
)


def repo_file_sizes(info) -> dict[str, int]:
    return {sibling.rfilename: sibling.size or 0 for sibling in info.siblings}


def episode_metadata_files(files: dict[str, int]) -> list[dict]:
    candidates = []
    for path, size in files.items():
        if re.fullmatch(r"meta/episodes/.*\.parquet", path):
            candidates.append({"path": path, "size": size})
    if not candidates:
        raise ValueError("Dataset contains no episode metadata parquet files")
    return sorted(candidates, key=lambda item: item["path"])


def _render_template(template: str, **values) -> str:
    return template.format(**values)


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
    sizes = repo_file_sizes(info)
    episode_files = episode_metadata_files(sizes)

    info_path = retry(lambda: hf_hub_download(
        REPO_ID, "meta/info.json", repo_type="dataset", revision=revision, endpoint=ENDPOINT,
    ), args.retries)
    info_json = json.loads(Path(info_path).read_text())
    fps = float(info_json["fps"])
    total_frames = int(info_json["total_frames"])
    data_path = info_json["data_path"]
    video_path = info_json["video_path"]

    tasks_path = retry(lambda: hf_hub_download(
        REPO_ID, "meta/tasks.parquet", repo_type="dataset", revision=revision, endpoint=ENDPOINT,
    ), args.retries)
    task_names = {}
    for row in pq.read_table(tasks_path).to_pylist():
        task_names[int(row["task_index"])] = row["__index_level_0__"]

    annotated = {}
    if "meta/tasks_annotated.parquet" in sizes:
        annotated_path = retry(lambda: hf_hub_download(
            REPO_ID, "meta/tasks_annotated.parquet", repo_type="dataset", revision=revision, endpoint=ENDPOINT,
        ), args.retries)
        for row in pq.read_table(annotated_path).to_pylist():
            annotated[int(row["episode_index"])] = row["task"]

    episode_columns = [
        "episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index",
        "data/chunk_index", "data/file_index",
    ]
    for key in VIDEO_KEYS:
        episode_columns += [
            f"videos/{key}/chunk_index", f"videos/{key}/file_index",
            f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp",
        ]
    episodes = []
    for file in episode_files:
        local = retry(lambda: hf_hub_download(
            REPO_ID, file["path"], repo_type="dataset", revision=revision, endpoint=ENDPOINT,
        ), args.retries)
        for row in pq.read_table(local, columns=episode_columns).to_pylist():
            videos = {}
            for key in VIDEO_KEYS:
                videos[key] = {
                    "chunk_index": int(row[f"videos/{key}/chunk_index"]),
                    "file_index": int(row[f"videos/{key}/file_index"]),
                    "path": _render_template(
                        video_path, video_key=key,
                        chunk_index=int(row[f"videos/{key}/chunk_index"]),
                        file_index=int(row[f"videos/{key}/file_index"]),
                    ),
                    "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
                    "to_timestamp": float(row[f"videos/{key}/to_timestamp"]),
                }
            data_chunk = int(row["data/chunk_index"])
            data_file = int(row["data/file_index"])
            episodes.append({
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "tasks": list(row["tasks"]),
                "task": annotated.get(int(row["episode_index"])),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "data": {
                    "chunk_index": data_chunk,
                    "file_index": data_file,
                    "path": _render_template(data_path, chunk_index=data_chunk, file_index=data_file),
                },
                "videos": videos,
            })
    if not episodes:
        raise ValueError("Dataset contains no episodes")
    for episode in episodes:
        if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
            raise ValueError(f"Episode {episode['episode_index']} has inconsistent index range")
    episodes.sort(key=lambda item: item["episode_index"])
    selected = sample_episodes(episodes, args.count, args.seed, args.strategy, REPO_ID)

    # Frames hosted by each remote video file, used for a per-segment size estimate.
    video_frames: dict[str, int] = {}
    for episode in episodes:
        for key in VIDEO_KEYS:
            path = episode["videos"][key]["path"]
            video_frames[path] = video_frames.get(path, 0) + episode["length"]

    meta_paths = ["meta/info.json", "meta/tasks.parquet"]
    if "meta/tasks_annotated.parquet" in sizes:
        meta_paths.append("meta/tasks_annotated.parquet")
    meta_paths += [file["path"] for file in episode_files]
    manifest_files = []
    for path in meta_paths:
        size = sizes.get(path)
        if size is None:
            raise ValueError(f"Missing repo file: {path}")
        manifest_files.append({"role": "meta", "path": path, "size": size})
    data_files: dict[str, dict] = {}
    for episode in selected:
        data = episode["data"]
        size = sizes.get(data["path"])
        if size is None:
            raise ValueError(f"Missing repo file: {data['path']}")
        data_files[data["path"]] = {"role": "data", "path": data["path"], "size": size}
    manifest_files.extend(sorted(data_files.values(), key=lambda item: item["path"]))

    estimated_video_bytes = 0
    for episode in selected:
        for key in VIDEO_KEYS:
            video = episode["videos"][key]
            size = sizes.get(video["path"])
            if size is None:
                raise ValueError(f"Missing repo file: {video['path']}")
            frames = video_frames.get(video["path"], 0) or total_frames
            estimated_video_bytes += round(size * episode["length"] / frames)
            video["size"] = size
        episode["data"]["size"] = data_files[episode["data"]["path"]]["size"]

    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": args.count,
        "strategy": args.strategy,
        "seed": args.seed,
        "fps": fps,
        "total_frames": total_frames,
        "video_keys": list(VIDEO_KEYS),
        "sampling": (
            "sorted by episode length ascending" if args.strategy == "smallest"
            else f"random.Random(f'{args.seed}:{REPO_ID}').sample over sorted episode_index"
        ),
        "available_episodes": len(episodes),
        "episodes": selected,
        "files": manifest_files,
        "total_bytes": sum(file["size"] for file in manifest_files),
        "estimated_video_bytes": estimated_video_bytes,
    }
    write_json(args.manifest, manifest)
    print(f"Revision: {revision}", flush=True)
    print(f"{len(episodes)} episodes available", flush=True)
    for episode in selected:
        task = episode.get("task") or (episode["tasks"][0] if episode["tasks"] else "(no task)")
        print(f"  episode_{episode['episode_index']:06d}: {episode['length']} frames, {task}", flush=True)
    print(f"Saved {args.manifest}: {len(selected)} trajectories, "
          f"metadata {format_size(manifest['total_bytes'])}, "
          f"estimated video {format_size(estimated_video_bytes)}", flush=True)


def load_manifest(path: Path) -> dict:
    return cutting.load_manifest(path, REPO_ID, VIDEO_KEYS)


def write_episode_tables(manifest: dict, data_tables: dict[str, object], output_dir: Path) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for episode in manifest["episodes"]:
        table = data_tables[episode["data"]["path"]]
        mask = pc.equal(table["episode_index"], episode["episode_index"])
        subset = table.filter(mask)
        if subset.num_rows != episode["length"]:
            raise ValueError(
                f"episode_{episode['episode_index']:06d}: expected {episode['length']} frames "
                f"in {episode['data']['path']}, found {subset.num_rows}"
            )
        episode_dir = output_dir / "episodes" / f"episode_{episode['episode_index']:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(subset, episode_dir / "data.parquet")
        write_json(episode_dir / "episode.json", {
            "episode_index": episode["episode_index"],
            "length": episode["length"],
            "tasks": episode["tasks"],
            "task": episode.get("task"),
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

    print(f"Downloading {len(manifest['files'])} metadata/data files "
          f"({format_size(manifest['total_bytes'])})", flush=True)
    data_tables: dict[str, object] = {}
    for file in manifest["files"]:
        destination = output_dir / "source" / Path(file["path"])
        fetch_file(REPO_ID, file, manifest["revision"], destination, args.retries)
        if file["role"] == "data":
            data_tables[file["path"]] = pq.read_table(destination)
    write_episode_tables(manifest, data_tables, output_dir)
    print(f"Wrote {len(manifest['episodes'])} episode frame tables", flush=True)

    extract_videos(manifest, output_dir, args)


add_arguments = cutting.add_arguments
COMMANDS = {"select": select, "download": download}
