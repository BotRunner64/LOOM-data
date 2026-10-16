#!/usr/bin/env python3
"""Select and download a small HiFi-UMI-2K trajectory (episode) subset.

HiFi-UMI-2K is published as hundreds of independent LeRobot v3 shards
(``chunk-XXXX/part-0000``). Every shard holds many episodes and concatenates the
six camera views of those episodes into one MP4 per view, so a single trajectory
is not a standalone file. This tool therefore picks the smallest shard, samples
a reproducible set of episodes from it, downloads only the small shard metadata
and frame table, and cuts the selected episode segments out of the remote MP4s
with HTTP range requests.

Because the source shards interleave episodes in one compressed stream, a
frame-accurate cut cannot reuse the source packets; each selected segment is
decoded from the nearest keyframe and re-encoded with H.264 at a quality that
matches the original bit rate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import time
from typing import Callable, TypeVar

REPO_ID = "simple-world-lab/HiFi-UMI-2K"
ENDPOINT = "https://huggingface.co"
DEFAULT_MANIFEST = Path("dataset/manifests/hifi-umi-2k-5.json")
DEFAULT_OUTPUT_DIR = Path("dataset/raw/HiFi-UMI-2K")
VIDEO_KEYS = (
    "observation.images.head_main",
    "observation.images.head_main_stereo_right",
    "observation.images.left_hand_up",
    "observation.images.left_hand_down",
    "observation.images.right_hand_up",
    "observation.images.right_hand_down",
)
# Episodes metadata timestamps are exact multiples of the frame period, so a
# tolerance far below one frame is enough to absorb float representation noise.
FRAME_EPSILON = 1e-3
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


def hf_url(path: str, revision: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(REPO_ID, path, repo_type="dataset", revision=revision, endpoint=ENDPOINT)


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


def sample_episodes(episodes: list[dict], count: int, seed: int, strategy: str) -> list[dict]:
    if strategy == "smallest":
        return sorted(episodes, key=lambda item: (item["length"], item["episode_index"]))[:count]
    population = sorted(episodes, key=lambda item: item["episode_index"])
    rng = random.Random(f"{seed}:{REPO_ID}")
    wanted = set(rng.sample([item["episode_index"] for item in population], min(count, len(population))))
    return [item for item in population if item["episode_index"] in wanted]


def frame_in_segment(time_seconds: float | None, start: float, end: float) -> bool:
    return time_seconds is not None and start - FRAME_EPSILON <= time_seconds < end - FRAME_EPSILON


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
    selected = sample_episodes(episodes, args.count, args.seed, args.strategy)

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
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != REPO_ID:
        raise ValueError("Unsupported manifest schema or dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest must pin a full commit SHA")
    shard = manifest.get("shard")
    if not isinstance(shard, str) or shard_name(shard) != shard:
        raise ValueError(f"Invalid shard: {shard!r}")
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
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest contains no episodes")
    indices = set()
    for episode in episodes:
        index = episode.get("episode_index")
        length = episode.get("length")
        videos = episode.get("videos")
        if type(index) is not int or index in indices:
            raise ValueError(f"Duplicate or invalid episode_index: {index!r}")
        indices.add(index)
        if type(length) is not int or length < 1:
            raise ValueError(f"Invalid episode length for episode {index}")
        if episode.get("dataset_to_index", 0) - episode.get("dataset_from_index", 0) != length:
            raise ValueError(f"Inconsistent index range for episode {index}")
        if not isinstance(videos, dict) or sorted(videos) != sorted(VIDEO_KEYS):
            raise ValueError(f"Episode {index} does not cover all camera views")
        for key, video in videos.items():
            if video.get("to_timestamp", 0) <= video.get("from_timestamp", 0):
                raise ValueError(f"Invalid video range for {key} in episode {index}")
    return manifest


def fetch_file(file: dict, revision: str, destination: Path, retries: int) -> None:
    from huggingface_hub import hf_hub_download

    source = Path(retry(lambda: hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=file["path"],
        revision=revision, endpoint=ENDPOINT,
    ), retries))
    if source.stat().st_size != file["size"]:
        source = Path(retry(lambda: hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=file["path"],
            revision=revision, endpoint=ENDPOINT, force_download=True,
        ), retries))
        if source.stat().st_size != file["size"]:
            raise ValueError(f"File size mismatch after redownload: {file['path']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def retry_av(operation: Callable[[], T], retries: int) -> T:
    import av

    for attempt in range(retries + 1):
        try:
            return operation()
        except (av.FFmpegError, OSError, ValueError) as exc:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def extract_segment(source_url: str, start: float, end: float, length: int, fps: float,
                    destination: Path, crf: int, preset: str, retries: int) -> None:
    """Decode ``[start, end)`` seconds of a remote MP4 and re-encode it frame-accurately."""
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    def run() -> None:
        if temporary.exists():
            temporary.unlink()
        container = av.open(source_url, options={
            "rw_timeout": "60000000",
            "reconnect": "1",
            "reconnect_streamed": "1",
            "reconnect_delay_max": "30",
        })
        try:
            stream = container.streams.video[0]
            container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
            output = av.open(str(temporary), "w", format="mp4")
            try:
                encoder = output.add_stream("libx264", rate=stream.average_rate or fps)
                encoder.width = stream.width
                encoder.height = stream.height
                encoder.pix_fmt = "yuv420p"
                encoder.options = {"crf": str(crf), "preset": preset}
                written = 0
                for frame in container.decode(stream):
                    time_seconds = frame.time
                    if time_seconds is None or time_seconds < start - FRAME_EPSILON:
                        continue
                    if time_seconds >= end - FRAME_EPSILON:
                        break
                    for packet in encoder.encode(frame.reformat(format="yuv420p")):
                        output.mux(packet)
                    written += 1
                for packet in encoder.encode():
                    output.mux(packet)
            finally:
                output.close()
        finally:
            container.close()
        if written != length:
            raise ValueError(f"expected {length} frames for {destination.name}, wrote {written}")
        temporary.replace(destination)

    try:
        retry_av(run, retries)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def local_frame_count(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def extract_video_key(key: str, manifest: dict, output_dir: Path, args) -> list[dict]:
    failures = []
    episodes = sorted(manifest["episodes"],
                      key=lambda item: item["videos"][key]["from_timestamp"])
    for episode in episodes:
        video = episode["videos"][key]
        destination = output_dir / "episodes" / f"episode_{episode['episode_index']:06d}" / "videos" / f"{key}.mp4"
        if destination.exists():
            try:
                if local_frame_count(destination) == episode["length"]:
                    continue
            except Exception:
                pass
        try:
            extract_segment(
                hf_url(video["path"], manifest["revision"]),
                video["from_timestamp"], video["to_timestamp"], episode["length"],
                manifest["fps"], destination, args.crf, args.preset, args.retries,
            )
        except Exception as exc:
            failures.append({"episode_index": episode["episode_index"], "video": key, "error": str(exc)})
            print(f"FAILED episode_{episode['episode_index']:06d}/{key}: {exc}", file=sys.stderr, flush=True)
    return failures


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
    output_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = output_dir / "subset_manifest.json"
    if local_manifest.exists():
        previous = load_manifest(local_manifest)
        if (previous["revision"], previous["episodes"]) != (manifest["revision"], manifest["episodes"]):
            raise ValueError("Output directory contains a different subset; choose another --output-dir")
    write_json(local_manifest, manifest)

    print(f"Downloading {len(manifest['files'])} metadata files "
          f"({format_size(manifest['total_bytes'])}) for {manifest['shard']}", flush=True)
    fetched = {}
    for file in manifest["files"]:
        destination = output_dir / "source" / f"{file['role']}{PurePosixPath(file['path']).suffix}"
        fetch_file(file, manifest["revision"], destination, args.retries)
        fetched[file["role"]] = destination
    write_episode_tables(manifest, pq.read_table(fetched["data"]), output_dir)
    print(f"Wrote {len(manifest['episodes'])} episode frame tables", flush=True)

    print(f"Extracting {len(manifest['episodes'])} trajectories x {len(VIDEO_KEYS)} camera views "
          f"(estimated {format_size(manifest['estimated_video_bytes'])})", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_video_key, key, manifest, output_dir, args): key
            for key in manifest["video_keys"]
        }
        try:
            for index, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                failures.extend(future.result())
                print(f"[{index}/{len(futures)}] {key}: done; {len(failures)} failures so far", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    write_json(output_dir / "download_failures.json", {"files": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} segments failed. Rerun the same download command to retry.")
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
        command.add_argument("--workers", type=positive_int, default=3)
        command.add_argument("--retries", type=int, default=3, help="additional attempts for transient failures")
        if name == "select":
            command.add_argument("--count", type=positive_int, default=5)
            command.add_argument("--strategy", choices=("smallest", "random"), default="random")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--shard", help="exact chunk-XXXX/part-YYYY; defaults to the smallest")
            command.add_argument("--revision", default="main")
        else:
            command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
            command.add_argument("--crf", type=int, default=18, help="libx264 quality; 18 matches the source bit rate")
            command.add_argument("--preset", default="medium")
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    # Also route OAuth refresh to the official Hub when the shell defines a mirror endpoint.
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
