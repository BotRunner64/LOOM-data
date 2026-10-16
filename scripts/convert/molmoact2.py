#!/usr/bin/env python3
"""Convert a local MolmoAct2-BimanualYAM LeRobot v3 dataset or episode subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow.parquet as pq

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import (
    copy_h264_video, seconds_to_ns, task_instructions, write_state, write_video_index,
)
from scripts.robot.yam import YamFK
from scripts.convert.lerobot_v3 import LeRobotV3Reader, extract_video_segment

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = {'observation.images.top': 'top',
           'observation.images.left': 'left_wrist',
           'observation.images.right': 'right_wrist'}
STATE_NAMES = [name for side in ('left', 'right') for name in
               [*(f'{side}_joint_{i}.pos' for i in range(6)), f'{side}_gripper.pos']]
COLUMNS = ('observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index')


def validate_source(info):
    if info.get('robot_type') != 'bi_yam_follower' or info.get('codebase_version') != 'v3.0':
        raise ValueError('Expected MolmoAct2 bi_yam_follower LeRobot v3.0')
    features = info.get('features', {})
    state = features.get('observation.state', {})
    if state.get('shape') != [14] or state.get('names') != STATE_NAMES:
        raise ValueError('Unsupported MolmoAct2 observation.state layout')
    if not all(features.get(key, {}).get('dtype') == 'video' for key in CAMERAS):
        raise ValueError('Expected all three MolmoAct2 video features')


def load_tasks(directory):
    tasks, annotated = {}, {}
    for row in pq.read_table(directory / 'tasks.parquet',
                             columns=['task_index', '__index_level_0__']).to_pylist():
        index, text = row['task_index'], row['__index_level_0__']
        if (type(index) is not int or index < 0 or index in tasks
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid/duplicate task_index or empty task text')
        tasks[index] = text
    path = directory / 'tasks_annotated.parquet'
    if path.exists():
        seen = set()
        for row in pq.read_table(path, columns=['episode_index', 'task']).to_pylist():
            index, text = row['episode_index'], row['task']
            if type(index) is not int or index < 0 or index in seen:
                raise ValueError('Invalid/duplicate annotation episode_index')
            seen.add(index)
            # The dataset card explicitly permits fallback for invalid annotations.
            if isinstance(text, str) and text.strip():
                annotated[index] = text
    return tasks, annotated


def read_episode(source, tasks, annotated, fk):
    metadata = json.loads((source / 'episode.json').read_text(encoding='utf-8'))
    rows = pq.read_table(source / 'data.parquet', columns=list(COLUMNS)).to_pydict()
    index = metadata.get('episode_index')
    if type(index) is not int or source.name != f'episode_{index:06d}':
        raise ValueError(f'{source}: inconsistent episode identity')
    return read_rows(source, metadata, rows, tasks, annotated, fk)


def read_rows(source, metadata, rows, tasks, annotated, fk):
    count = len(rows['timestamp'])
    index = metadata.get('episode_index')
    start, stop = metadata.get('dataset_from_index'), metadata.get('dataset_to_index')
    if (type(index) is not int or index < 0
            or type(start) is not int or type(stop) is not int or start < 0
            or count == 0 or metadata.get('length') != count or stop - start != count
            or any(i != index for i in rows['episode_index'])):
        raise ValueError(f'{source}: inconsistent episode identity/length')
    if rows['frame_index'] != list(range(count)) or rows['index'] != list(range(start, stop)):
        raise ValueError(f'{source}: noncontiguous frame/index; cannot map video frames')
    if any(t is None for t in rows['timestamp']):
        raise ValueError(f'{source}: null timestamp')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if times[0] < 0 or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{source}: negative or decreasing timestamps')
    state = np.asarray(rows['observation.state'], dtype=np.float64)
    if state.shape != (count, 14) or not np.isfinite(state).all():
        raise ValueError(f'{source}: expected finite N x 14 observation.state')
    fallback = task_instructions(rows['task_index'], tasks, times)
    if set(metadata.get('tasks', [])) != {tasks[i] for i in rows['task_index']}:
        raise ValueError(f'{source}: episode tasks disagree with task_index')
    annotation = ([{'start_ns': 0, 'end_ns': times[-1] - times[0], 'text': annotated[index]}]
                  if index in annotated else fallback)
    if set(metadata.get('videos', {})) != set(CAMERAS):
        raise ValueError(f'{source}: expected all three camera views')
    streams = {}
    for side, offset in (('left', 0), ('right', 7)):
        streams[f'{side}_eef'] = [fk.pose(side, q) for q in state[:, offset:offset + 6]]
        streams[f'{side}_gripper'] = np.clip(state[:, offset + 6], 0, 1)
    return times, streams, annotation


def convert_episode(source, destination, tasks, annotated, fk):
    times, streams, annotation = read_episode(source, tasks, annotated, fk)
    return write_episode(source.name, destination, times, streams, annotation,
                         lambda key, target: copy_h264_video(
                             source / 'videos' / f'{key}.mp4', target, len(times)))


def write_episode(episode_id, destination, times, streams, annotation, video):
    # This source supplies one shared timestamp per recorded observation row.
    # The segmented videos preserve that row order; container PTS are irrelevant.
    for name, values in streams.items():
        write_state(destination / 'state' / f'{name}.parquet', times, values, times[0],
                    pose=name.endswith('_eef'))
    for key, camera in CAMERAS.items():
        print(f'  {camera}', flush=True)
        target = destination / 'rgb' / camera
        video(key, target.with_suffix('.mp4'))
        write_video_index(target.with_suffix('.parquet'), times, times[0])
    return {'episode_id': episode_id, 'cameras': sorted(CAMERAS.values()),
            'instructions': annotation}


class DatasetReader(LeRobotV3Reader):
    def __init__(self, source):
        # Validate decoded segment lengths per episode: a bad source range must
        # not prevent preparing or converting the rest of the full dataset.
        super().__init__(source, CAMERAS, COLUMNS, validate_source, check_video_length=False)
        self.tasks, self.annotated = load_tasks(source / 'meta')

    def convert_episode(self, index, destination, fk):
        row = self.episodes[index]
        for key, video in row['videos'].items():
            frames = (video['to_timestamp'] - video['from_timestamp']) * self.fps
            if abs(frames - row['length']) > .01:
                raise ValueError(f'{self.source}: episode {index} {key} video range implies '
                                 f'{frames:g} frames; expected {row["length"]}')
        rows = self.read_rows(index)
        times, streams, annotation = read_rows(self.source, row, rows, self.tasks, self.annotated, fk)
        return write_episode(f'episode_{index:06d}', destination, times, streams, annotation,
                             lambda key, target: extract_video_segment(
                                 row['videos'][key], target, len(times), self.fps))


def convert(source, output, model_dir=ROOT / 'assets/robot_models/yam', *, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    if not (source / 'source').is_dir():
        return convert_dataset(source, output, model_dir, limit)
    meta = source / 'source/meta'
    validate_source(json.loads((meta / 'info.json').read_text(encoding='utf-8')))
    tasks, annotated = load_tasks(meta)
    episodes = sorted(p for p in (source / 'episodes').iterdir() if p.is_dir())
    if not episodes or any(not re.fullmatch(r'episode_\d+', p.name) for p in episodes):
        raise ValueError(f'No episodes or invalid episode directory names under {source}')
    episodes = episodes[:limit]
    # Use each fixed arm base, without assuming the ABC dual-arm mounting is
    # MolmoAct2's calibration. The contract permits independent fixed references.
    fk = YamFK(model_dir, arm_local=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
            for i, episode in enumerate(episodes, 1):
                print(f'[{i}/{len(episodes)}] {episode.name}', flush=True)
                record = convert_episode(episode, staged / 'episodes' / episode.name,
                                         tasks, annotated, fk)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {len(episodes)} episodes to {output}', flush=True)


def convert_dataset(source, output, model_dir, limit):
    reader = DatasetReader(source)
    indices = sorted(reader.episodes)[:limit]
    fk = YamFK(model_dir, arm_local=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
            for number, index in enumerate(indices, 1):
                episode_id = f'episode_{index:06d}'
                print(f'[{number}/{len(indices)}] {episode_id}', flush=True)
                record = reader.convert_episode(index, staged / 'episodes' / episode_id, fk)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {len(indices)} episodes to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/raw/MolmoAct2-BimanualYAM')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/processed/MolmoAct2-BimanualYAM')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'assets/robot_models/yam')
    parser.add_argument('--limit', type=int, help='convert only the first N sorted episodes')
    args = parser.parse_args()
    import av
    try:
        convert(args.input_dir, args.output_dir, args.model_dir, limit=args.limit)
    except (ValueError, OSError, av.FFmpegError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
