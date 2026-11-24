"""Write a cleaned copy of an exported dataset: repair EEF spikes, drop and trim idle episodes.

docs/contract/anomaly-contract.md specifies how each anomaly is handled. The
source dataset is only read. Videos are never re-encoded: an unchanged episode
links its files and a trimmed one copies the packets of the frames it keeps.
"""
import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sys
from typing import NamedTuple

import numpy as np

from loom_data import contract, resumable, sync
from loom_data.check import anomalies
from loom_data.contract import FPS, STATE_NAMES, STATE_SLICES, is_pose

KEEP_IDLE_S = 1.   # idle kept next to the motion at a trimmed end
MAX_SPIKES = 5     # an episode with more spikes is dropped instead of repaired
LOG = 'cleaning.jsonl'


class Kept(NamedTuple):
    state: np.ndarray   # N x 26, spikes repaired, not yet trimmed
    start: int          # frames [start, end) are kept
    end: int
    repaired: int       # number of spikes repaired


def clean_state(state, config=anomalies.Config(), keep_frames=round(KEEP_IDLE_S * FPS), max_spikes=MAX_SPIKES):
    """Handle the anomalies of one episode's N x 26 state: a `Kept`, or the reason the episode is dropped."""
    state = np.array(state, dtype=np.float64)
    count = len(state)
    streams = anomalies.episode_streams(state)
    spikes = [(name, start, end) for name in STATE_NAMES if is_pose(name)
              for start, end in anomalies.spike_intervals(streams[name], config)]
    if len(spikes) > max_spikes:
        return 'eef_spike'
    for name, start, end in spikes:
        columns = STATE_SLICES[name]
        # Interpolate from the frame before the spike to the frame after it.
        state[start:end, columns] = sync.resample_poses(
            np.array([start - 1, end]), state[[start - 1, end], columns], np.arange(start, end))
    if spikes:
        # A spike counts as motion, so idling is judged on the repaired state.
        streams = anomalies.episode_streams(state)
    if any(start > 0 and end < count and (end - start) / FPS > config.max_internal_idle_s
           for start, end in anomalies.idle_segments(streams, count, config)):
        return 'internal_idle'
    head, tail = anomalies.boundary_idle(streams, count, config)
    trim_head = head / FPS > config.max_boundary_idle_s
    trim_tail = (count - tail) / FPS > config.max_boundary_idle_s
    # Never moved, or every frame belongs to an idle end that would be trimmed.
    if head == count or (trim_head and trim_tail and head >= tail):
        return 'boundary_idle'
    start = max(0, head - keep_frames) if trim_head else 0
    end = min(count, tail + keep_frames) if trim_tail else count
    if end - start < 2:
        return 'boundary_idle'
    return Kept(state, start, end, len(spikes))


def packets(reader, path):
    """Video packets of an open file, one per frame, checked to be stored in display order."""
    previous = None
    for packet in reader.demux(reader.streams.video[0]):
        if packet.dts is None:
            continue
        if packet.pts is None or (previous is not None and packet.pts <= previous):
            raise ValueError(f'{path}: video frames are not stored in display order')
        previous = packet.pts
        yield packet


def common_keyframe(paths, frame):
    """The latest frame at or before `frame` where every video has a keyframe.

    A video can only be cut without re-encoding at a keyframe. Exported videos
    have one at least every other frame, and all of them at frame 0.
    """
    import av
    shared = np.ones(frame + 1, dtype=bool)
    for path in paths:
        with av.open(str(path)) as reader:
            flags = [packet.is_keyframe for packet, _ in zip(packets(reader, path), range(frame + 1))]
        if len(flags) != frame + 1:
            raise ValueError(f'{path}: video is shorter than its episode')
        shared &= flags
    if not shared[0]:
        raise ValueError(f'{paths[0]}: video does not start with a keyframe')
    return int(np.flatnonzero(shared)[-1])


def copy_frames(source, destination, start, end):
    """Copy frames [start, end) of a video file into a new file without re-encoding."""
    import av
    with av.open(str(source)) as reader, av.open(str(destination), 'w', format='mp4') as writer:
        target = writer.add_stream_from_template(reader.streams.video[0])
        origin, copied = None, 0
        for index, packet in enumerate(packets(reader, source)):
            if index >= end:
                break
            if index < start:
                continue
            if origin is None:
                if not packet.is_keyframe:
                    raise ValueError(f'{source}: frame {start} is not a keyframe')
                origin = packet.pts
            packet.pts -= origin
            packet.dts -= origin
            packet.stream = target
            writer.mux(packet)
            copied += 1
    if copied != end - start:
        raise ValueError(f'{source}: video is shorter than its episode')


def stage_episode(dataset, episode, kept, tasks, directory):
    """Stage a kept episode for `contract.build_dataset`; return the frame range actually kept."""
    videos = {}
    for camera in dataset.cameras:
        path, first = dataset.video(episode, camera)
        if first:
            raise ValueError(f'{path}: episodes share a video file; expected one file per episode')
        videos[camera] = path
    start, end = kept.start, kept.end
    if start:
        start = common_keyframe(list(videos.values()), start)
    contract.write_frame_table(directory, kept.state[start:end], tasks[start:end])
    for camera, path in videos.items():
        if (start, end) == (0, len(kept.state)):
            contract.link(path, directory / f'{camera}.mp4')
        else:
            copy_frames(path, directory / f'{camera}.mp4', start, end)
    return start, end


def clean(root, output, config=None, keep_idle_s=KEEP_IDLE_S, max_spikes=MAX_SPIKES):
    """Write the cleaned copy of dataset `root` to `output`; return the summary counts."""
    root, output = Path(root).resolve(), Path(output).resolve()
    config = config or anomalies.Config()
    if keep_idle_s < 0 or max_spikes < 0:
        raise ValueError('keep_idle_s and max_spikes must be nonnegative')
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output must be outside the source dataset')
    dataset = contract.Dataset(root)
    work = resumable.work_directory(output, '.clean-work')
    shutil.rmtree(work, ignore_errors=True)   # leftovers of an interrupted run
    sizes = {camera: dataset.info['features'][contract.video_key(camera)]['shape'][1::-1]
             for camera in dataset.cameras}
    staged, log, counts = [], [], Counter()
    try:
        with resumable.progress('Cleaning', len(dataset.episodes), unit='episode') as bar:
            for episode in dataset.episodes:
                state, tasks = dataset.frames(episode)
                result = clean_state(state, config, round(keep_idle_s * FPS), max_spikes)
                entry = {'episode_id': episode['episode_id'], 'source_episode_index': episode['episode_index']}
                if isinstance(result, str):
                    counts['dropped'] += 1
                    counts['dropped_' + result] += 1
                    log.append({**entry, 'action': 'dropped', 'reason': result})
                else:
                    directory = work / 'episodes' / episode['episode_id']
                    start, end = stage_episode(dataset, episode, result, tasks, directory)
                    staged.append(({'episode_id': episode['episode_id'], 'cameras': sorted(dataset.cameras),
                                    'length': end - start, 'sizes': sizes}, directory))
                    trimmed = (start, end) != (0, len(state))
                    counts['kept'] += 1
                    counts['repaired_episodes'] += bool(result.repaired)
                    counts['repaired_spikes'] += result.repaired
                    counts['trimmed_episodes'] += trimmed
                    counts['removed_frames'] += len(state) - (end - start)
                    if trimmed or result.repaired:
                        log.append({**entry, 'action': 'kept', 'repaired_spikes': result.repaired,
                                    'start_frame': start, 'end_frame': end})
                bar.update()
        if not staged:
            raise ValueError('Every episode was dropped; nothing to write')
        # The source's image statistics are estimates from one frame per second; the kept frames
        # are a subset of the same videos, so they are reused instead of decoding every video again.
        stats = json.loads((root / 'meta/stats.json').read_text(encoding='utf-8'))
        with resumable.progress('Assembling', len(staged), unit='episode') as bar:
            contract.build_dataset(work / 'dataset', dataset.info.get('robot_type'), staged, bar,
                                   image_stats={contract.video_key(camera): stats[contract.video_key(camera)]
                                                for camera in dataset.cameras})
        (work / 'dataset' / LOG).write_text(''.join(json.dumps(entry) + '\n' for entry in log), encoding='utf-8')
        output.parent.mkdir(parents=True, exist_ok=True)
        (work / 'dataset').rename(output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {'episodes': len(dataset.episodes), **{name: int(counts[name]) for name in (
        'kept', 'repaired_episodes', 'repaired_spikes', 'trimmed_episodes', 'removed_frames', 'dropped',
        *('dropped_' + reason for reason in anomalies.REASONS))}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data clean', description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', type=Path, help='exported dataset root')
    parser.add_argument('--output', type=Path, required=True, help='root of the cleaned dataset; must not exist')
    parser.add_argument('--keep-idle-s', type=float, default=KEEP_IDLE_S,
                        help=f'idle kept at a trimmed end (default {KEEP_IDLE_S})')
    parser.add_argument('--max-spikes', type=int, default=MAX_SPIKES,
                        help=f'drop an episode with more spikes than this (default {MAX_SPIKES})')
    for field in fields(anomalies.Config):
        parser.add_argument('--' + field.name.replace('_', '-'), type=field.type, default=field.default,
                            help=f'default {field.default}')
    args = parser.parse_args(argv)
    try:
        config = anomalies.Config(**{f.name: getattr(args, f.name) for f in fields(anomalies.Config)})
        summary = clean(args.dataset, args.output, config, args.keep_idle_s, args.max_spikes)
    except KeyboardInterrupt:
        print('Interrupted; nothing was published. Rerun to start over.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    dropped = ', '.join(f'{summary["dropped_" + reason]} {reason}' for reason in anomalies.REASONS)
    print(f'Kept {summary["kept"]} of {summary["episodes"]} episodes: repaired {summary["repaired_spikes"]} spikes '
          f'in {summary["repaired_episodes"]}, trimmed {summary["trimmed_episodes"]} '
          f'({summary["removed_frames"]} frames removed)')
    print(f'Dropped {summary["dropped"]}: {dropped}')
    print(f'Wrote {Path(args.output).resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    resumable.run_cli(main)
