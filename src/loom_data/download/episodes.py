"""Cut single episodes out of LeRobot v3 repositories (HiFi-UMI, MolmoAct2).

These repositories concatenate many episodes into one MP4 per camera, so one
trajectory is not a standalone file. `download` fetches the small metadata and
frame tables, then decodes each selected segment from the remote MP4 with HTTP
range requests and re-encodes it with H.264: a frame-accurate cut cannot reuse
packets of a stream that interleaves episodes.

The result is the per-episode layout the converters read:
    source/                         upstream metadata
    episodes/episode_N/data.parquet, episode.json, videos/<camera>.mp4
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Callable, TypeVar

from loom_data.download import hub
from loom_data.download.hub import ENDPOINT, format_size, write_json

# Episodes metadata timestamps are exact multiples of the frame period, so a
# tolerance far below one frame is enough to absorb float representation noise.
FRAME_EPSILON = 1e-3
T = TypeVar("T")


def hf_url(repo_id: str, path: str, revision: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id, path, repo_type="dataset", revision=revision, endpoint=ENDPOINT)


def sample_episodes(episodes: list[dict], count: int, seed: int, strategy: str, repo_id: str) -> list[dict]:
    if strategy == "smallest":
        return sorted(episodes, key=lambda item: (item["length"], item["episode_index"]))[:count]
    population = sorted(episodes, key=lambda item: item["episode_index"])
    rng = random.Random(f"{seed}:{repo_id}")
    wanted = set(rng.sample([item["episode_index"] for item in population], min(count, len(population))))
    return [item for item in population if item["episode_index"] in wanted]


def frame_in_segment(time_seconds: float | None, start: float, end: float) -> bool:
    return time_seconds is not None and start - FRAME_EPSILON <= time_seconds < end - FRAME_EPSILON


def load_manifest(path: Path, repo_id: str, video_keys: tuple[str, ...]) -> dict:
    manifest = hub.load_manifest(path, repo_id)
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
        if not isinstance(videos, dict) or sorted(videos) != sorted(video_keys):
            raise ValueError(f"Episode {index} does not cover all camera views")
        for key, video in videos.items():
            if video.get("to_timestamp", 0) <= video.get("from_timestamp", 0):
                raise ValueError(f"Invalid video range for {key} in episode {index}")
    return manifest


def fetch_file(repo_id: str, file: dict, revision: str, destination: Path, retries: int) -> None:
    """Fetch one small file through the Hub cache and copy it to `destination`."""
    source = hub.download_file(repo_id, revision, file, None, retries)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def retry_av(operation: Callable[[], T], retries: int) -> T:
    import av

    for attempt in range(retries + 1):
        try:
            return operation()
        except (av.FFmpegError, OSError, ValueError):
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
                hf_url(manifest["repo_id"], video["path"], manifest["revision"]),
                video["from_timestamp"], video["to_timestamp"], episode["length"],
                manifest["fps"], destination, args.crf, args.preset, args.retries,
            )
        except Exception as exc:
            failures.append({"episode_index": episode["episode_index"], "video": key, "error": str(exc)})
            print(f"FAILED episode_{episode['episode_index']:06d}/{key}: {exc}", file=sys.stderr, flush=True)
    return failures


def prepare_output(manifest: dict, output_dir: Path) -> None:
    hub.prepare_output(manifest, output_dir, key="episodes")


def extract_videos(manifest: dict, output_dir: Path, args) -> None:
    """Cut every selected episode out of every camera view, one worker per view."""
    print(f"Extracting {len(manifest['episodes'])} trajectories x {len(manifest['video_keys'])} camera views "
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


def add_arguments(command, name) -> None:
    if name == "select":
        command.add_argument("--count", type=hub.positive_int, default=5)
        command.add_argument("--strategy", choices=("smallest", "random"), default="random")
        command.add_argument("--seed", type=int, default=42)
    else:
        command.add_argument("--crf", type=int, default=18, help="libx264 quality; 18 matches the source bit rate")
        command.add_argument("--preset", default="medium")
