"""Clean an exported dataset in place: repair EEF spikes, drop and trim idle episodes.

docs/contract/anomaly-contract.md specifies how each anomaly is handled. Every
dropped or changed episode is kept as it was in a separate dataset, the removed
directory. Videos are never re-encoded: a trimmed episode copies the packets of
the frames it keeps.

Everything new is first staged in a work directory while the dataset is only
read. The dataset is then replaced by a journal of renames; an interrupted
replacement is finished by running the command again.
"""
import argparse
from collections import Counter
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import sys
from typing import NamedTuple

import numpy as np

from loom_data import contract, paths, resumable, sync
from loom_data.check import anomalies
from loom_data.contract import FPS, STATE_NAMES, STATE_SLICES, is_pose

KEEP_IDLE_S = 1.   # idle kept next to the motion at a trimmed end
MAX_SPIKES = 5     # an episode with more spikes is dropped instead of repaired
LOG = 'cleaning.jsonl'                    # in the removed directory: what happened to each episode
OLD_REPORT = 'anomalies.before-clean.jsonl'   # in the removed directory: the dataset's stale anomaly report
JOURNAL = 'commit.json'
FILES_MOVED = 'files-moved'


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


def removed_directory(root, removed=None):
    """Where the originals of dropped and changed episodes go: dataset/export/X -> dataset/removed/X."""
    if removed is not None:
        return Path(removed).resolve()
    export = paths.EXPORT.resolve()
    if root != export and root.is_relative_to(export):
        return paths.ROOT / 'dataset/removed' / root.relative_to(export)
    return root.with_name(root.name + '.removed')


def device(path):
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


def move(source, target):
    os.rename(source, target)


def replay(moves):
    """Apply (source, target) renames in order, skipping those an interrupted run already made."""
    for source, target in moves:
        if os.path.lexists(target):
            continue
        if not os.path.lexists(source):
            raise resumable.StateError(f'Cannot finish the interrupted clean: {source} is missing')
        move(source, target)


def commit(work):
    """Replace the dataset as the journal says; safe to repeat after an interruption."""
    plan = resumable.read_json(work / JOURNAL)
    # File moves only target the staged trees, which stay in the work directory until the marker
    # is written. After it, replaying them would mistake files of the new dataset for old ones.
    if not (work / FILES_MOVED).exists():
        replay(plan['files'])
        (work / FILES_MOVED).touch()
    replay(plan['directories'])
    for path in plan['stale']:
        shutil.rmtree(path, ignore_errors=True)
    # Drop the journal before the work directory: a replay must never see half-deleted staging.
    (work / JOURNAL).unlink()
    shutil.rmtree(work, ignore_errors=True)
    return plan['summary']


def stage(dataset, work, removed, config, keep_frames, max_spikes):
    """Stage the cleaned dataset and the dataset of originals; return the journal, or None if nothing changes."""
    root = dataset.root
    sizes = {camera: dataset.info['features'][contract.video_key(camera)]['shape'][1::-1]
             for camera in dataset.cameras}
    kept, originals, log, counts = [], [], [], Counter()
    sources = {}   # (tree, episode_id, camera) -> the file that becomes this episode's video in that tree
    with resumable.progress('Cleaning', len(dataset.episodes), unit='episode') as bar:
        for episode in dataset.episodes:
            identifier = episode['episode_id']
            state, tasks = dataset.frames(episode)
            videos = {}
            for camera in dataset.cameras:
                videos[camera], first = dataset.video(episode, camera)
                if first:
                    raise ValueError(f'{videos[camera]}: episodes share a video file; expected one file per episode')
            result = clean_state(state, config, keep_frames, max_spikes)
            entry = {'episode_id': identifier, 'source_episode_index': episode['episode_index']}
            if isinstance(result, str):
                changed = True
                counts['dropped'] += 1
                counts['dropped_' + result] += 1
                log.append({**entry, 'action': 'dropped', 'reason': result})
            else:
                start, end = result.start, result.end
                if start:
                    start = common_keyframe(list(videos.values()), start)
                trimmed = (start, end) != (0, len(state))
                changed = trimmed or bool(result.repaired)
                directory = work / 'staged/new' / identifier
                contract.write_frame_table(directory, result.state[start:end], tasks[start:end])
                for camera, path in videos.items():
                    if trimmed:
                        copy_frames(path, directory / f'{camera}.mp4', start, end)
                    sources['new', identifier, camera] = directory / f'{camera}.mp4' if trimmed else path
                kept.append(({'episode_id': identifier, 'cameras': sorted(dataset.cameras), 'length': end - start,
                              'sizes': sizes}, directory))
                counts['kept'] += 1
                counts['repaired_episodes'] += bool(result.repaired)
                counts['repaired_spikes'] += result.repaired
                counts['trimmed_episodes'] += trimmed
                counts['removed_frames'] += len(state) - (end - start)
                if changed:
                    log.append({**entry, 'action': 'kept', 'repaired_spikes': result.repaired,
                                'start_frame': start, 'end_frame': end})
            if changed:
                directory = work / 'staged/removed' / identifier
                contract.write_frame_table(directory, state, tasks)
                for camera, path in videos.items():
                    if sources.get(('new', identifier, camera)) == path:
                        # Only the state was repaired: the dataset keeps this video, the originals get a copy.
                        shutil.copyfile(path, directory / f'{camera}.mp4')
                        path = directory / f'{camera}.mp4'
                    sources['removed', identifier, camera] = path
                originals.append(({'episode_id': identifier, 'cameras': sorted(dataset.cameras),
                                   'length': len(state), 'sizes': sizes}, directory))
            bar.update()
    if not originals:
        return None
    if not kept:
        raise ValueError('Every episode would be dropped; the dataset is left unchanged')
    # Image statistics are estimates from one frame per second of the same videos; they are
    # reused for both datasets instead of decoding every video again.
    stats = json.loads((root / 'meta/stats.json').read_text(encoding='utf-8'))
    stats = {contract.video_key(camera): stats[contract.video_key(camera)] for camera in dataset.cameras}
    files = []
    for tree, episodes in (('new', kept), ('removed', originals)):
        def place(directory, camera, target, tree=tree):
            target.parent.mkdir(parents=True, exist_ok=True)
            files.append((sources[tree, directory.name, camera], target))

        with resumable.progress(f'Assembling {tree}', len(episodes), unit='episode') as bar:
            contract.build_dataset(work / tree, dataset.info.get('robot_type'), episodes, bar,
                                   image_stats=stats, place=place)
    (work / 'removed' / LOG).write_text(''.join(json.dumps(entry) + '\n' for entry in log), encoding='utf-8')
    stale = []
    if (root / 'anomalies.jsonl').is_file():
        # The report addresses episodes of the dataset as it was; its recovery state goes with it.
        from loom_data.check import job
        files.append((root / 'anomalies.jsonl', work / 'removed' / OLD_REPORT))
        stale.append(job.work_directory(root / 'anomalies.jsonl'))
    (work / 'old').mkdir()
    directories = []
    for name in ('data', 'meta', 'videos'):
        directories += [(root / name, work / 'old' / name), (work / 'new' / name, root / name)]
    directories.append((work / 'removed', removed))
    summary = {'episodes': len(dataset.episodes), 'changed': len(originals), 'removed': str(removed),
               **{name: int(counts[name]) for name in (
                   'kept', 'repaired_episodes', 'repaired_spikes', 'trimmed_episodes', 'removed_frames', 'dropped',
                   *('dropped_' + reason for reason in anomalies.REASONS))}}
    return {'files': [[str(a), str(b)] for a, b in files], 'directories': [[str(a), str(b)] for a, b in directories],
            'stale': [str(path) for path in stale], 'summary': summary}


def clean(root, removed=None, config=None, keep_idle_s=KEEP_IDLE_S, max_spikes=MAX_SPIKES):
    """Clean dataset `root` in place; return the summary counts, or None if it has nothing to clean."""
    root = Path(root).resolve()
    work = resumable.work_directory(root, '.clean-work')
    if (work / JOURNAL).exists():
        return {**commit(work), 'resumed': True}
    config = config or anomalies.Config()
    if keep_idle_s < 0 or max_spikes < 0:
        raise ValueError('keep_idle_s and max_spikes must be nonnegative')
    removed = removed_directory(root, removed)
    if removed.exists():
        raise ValueError(f'Removed directory exists: {removed}; this dataset was cleaned before')
    if removed.is_relative_to(root) or root.is_relative_to(removed):
        raise ValueError('The removed directory must be outside the dataset')
    if len({device(root), device(work), device(removed)}) != 1:
        raise ValueError('The work and removed directories must be on the same filesystem as the dataset')
    dataset = contract.Dataset(root)
    shutil.rmtree(work, ignore_errors=True)   # leftovers of a run interrupted while staging
    try:
        plan = stage(dataset, work, removed, config, round(keep_idle_s * FPS), max_spikes)
        if plan is not None:
            removed.parent.mkdir(parents=True, exist_ok=True)
            resumable.atomic_json(work / JOURNAL, plan)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if plan is None:
        shutil.rmtree(work, ignore_errors=True)
        return None
    return commit(work)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data clean', description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', type=Path, help='exported dataset root; cleaned in place')
    parser.add_argument('--removed', type=Path,
                        help='where the originals of dropped and changed episodes go; must not exist '
                             '(default: dataset/removed/... for a dataset under dataset/export, else DATASET.removed)')
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
        summary = clean(args.dataset, args.removed, config, args.keep_idle_s, args.max_spikes)
    except KeyboardInterrupt:
        print('Interrupted; rerun the same command. It starts over, or finishes a replacement already under way.',
              file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    if summary is None:
        print('Nothing to clean; the dataset is unchanged', flush=True)
        return 0
    if summary.get('resumed'):
        print('Finished the interrupted clean')
    dropped = ', '.join(f'{summary["dropped_" + reason]} {reason}' for reason in anomalies.REASONS)
    print(f'Kept {summary["kept"]} of {summary["episodes"]} episodes: repaired {summary["repaired_spikes"]} spikes '
          f'in {summary["repaired_episodes"]}, trimmed {summary["trimmed_episodes"]} '
          f'({summary["removed_frames"]} frames removed)')
    print(f'Dropped {summary["dropped"]}: {dropped}')
    print(f'Originals of {summary["changed"]} episodes: {summary["removed"]}', flush=True)
    return 0


if __name__ == '__main__':
    resumable.run_cli(main)
