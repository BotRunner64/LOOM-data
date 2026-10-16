"""Detect anomalies in an exported dataset without resampling state streams."""

import argparse
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class Config:
    position_tolerance_m: float = .01
    orientation_tolerance_deg: float = 5.
    openness_tolerance: float = .05
    max_boundary_idle_s: float = 3.
    position_jump_min_m: float = .02
    orientation_jump_min_deg: float = 10.
    openness_jump_min: float = .25
    jump_max_duration_s: float = .05
    jump_context_s: float = .1
    max_internal_idle_s: float = 10.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{field.name} must be finite and nonnegative')
        if self.jump_max_duration_s < 1e-9:
            raise ValueError('jump_max_duration_s must be at least 1 ns')
        if self.jump_context_s < 2 * self.jump_max_duration_s:
            raise ValueError('jump_context_s must be >= 2 * jump_max_duration_s')


def rotation_quaternions(poses):
    """Contract column ordered 3x3 rotation matrices to unit wxyz quaternions."""
    r = poses[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1).copy()
    if (not np.isfinite(r).all()
            or not np.allclose(r @ np.swapaxes(r, 1, 2), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid rotation matrix')
    r00, r11, r22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    squared = np.column_stack((1+r00+r11+r22, 1+r00-r11-r22,
                               1-r00+r11-r22, 1-r00-r11+r22))
    candidates = np.stack((
        np.column_stack((squared[:, 0], r[:, 2, 1]-r[:, 1, 2],
                         r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1])),
        np.column_stack((r[:, 2, 1]-r[:, 1, 2], squared[:, 1],
                         r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0])),
        np.column_stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1],
                         squared[:, 2], r[:, 2, 1]+r[:, 1, 2])),
        np.column_stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0],
                         r[:, 2, 1]+r[:, 1, 2], squared[:, 3])),
    ), axis=1)
    q = candidates[np.arange(len(poses)), np.argmax(squared, axis=1)]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rotation_distance_deg(a, b):
    b = np.where(np.sum(a*b, axis=-1, keepdims=True) < 0, -b, b)
    return np.degrees(4 * np.arctan2(np.linalg.norm(a-b, axis=-1),
                                    np.linalg.norm(a+b, axis=-1)))


@dataclass
class StateStream:
    times: np.ndarray
    values: np.ndarray
    rotations: np.ndarray | None = None

    def changed(self, left, right, config):
        if self.rotations is None:
            return np.abs(self.values[left] - self.values[right]) > config.openness_tolerance
        return ((np.linalg.norm(self.values[left, :3] - self.values[right, :3], axis=-1)
                 > config.position_tolerance_m)
                | (rotation_distance_deg(self.rotations[left], self.rotations[right])
                   > config.orientation_tolerance_deg))


def read_times(table, path):
    column = table['timestamp_ns']
    if column.type != pa.int64() or column.null_count:
        raise ValueError(f'{path}: timestamps must be non-null int64')
    times = column.to_numpy()
    if not len(times) or times[0] < 0 or np.any(times[1:] < times[:-1]):
        raise ValueError(f'{path}: empty, negative or unordered timestamps')
    return times


def episode_paths(root, record):
    """Validate manifest identifiers and list only the inputs detection reads."""
    episode_id = record['episode_id']
    if (not isinstance(episode_id, str) or episode_id in ('', '.', '..')
            or '/' in episode_id or '\\' in episode_id):
        raise ValueError(f'Invalid episode_id: {episode_id!r}')
    base = root / 'episodes' / episode_id
    cameras = record['cameras']
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(c, str) or not re.fullmatch(r'[a-z0-9_]+', c) for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError(f'{episode_id}: invalid cameras')
    return ([base / 'state' / f'{name}.parquet' for name in STATE_NAMES]
            + [base / 'rgb' / f'{camera}.parquet' for camera in cameras])


def read_episode(root, record):
    paths = episode_paths(root, record)
    streams = {}
    for name, path in zip(STATE_NAMES, paths):
        pose = name.endswith('_eef')
        field = 'pose' if pose else 'openness'
        table = pq.read_table(path, columns=['timestamp_ns', field])
        times = read_times(table, path)
        values = np.asarray(table[field].to_pylist(), dtype=np.float64)
        if (table[field].null_count or not np.isfinite(values).all()
                or values.shape != ((len(times), 12) if pose else (len(times),))):
            raise ValueError(f'{path}: invalid {field}')
        if not pose and np.any((values < 0) | (values > 1)):
            raise ValueError(f'{path}: openness outside [0,1]')
        streams[name] = StateStream(times, values, rotation_quaternions(values) if pose else None)
    end_ns = max(int(stream.times[-1]) for stream in streams.values())
    for path in paths[len(STATE_NAMES):]:
        times = read_times(pq.read_table(path, columns=['timestamp_ns']), path)
        end_ns = max(end_ns, int(times[-1]))
    return streams, end_ns


def state_distances(stream, left, right):
    """Separate physical channels; never mix metres and degrees."""
    if stream.rotations is None:
        return np.abs(stream.values[left] - stream.values[right])[..., None]
    return np.stack((np.linalg.norm(stream.values[left, :3] - stream.values[right, :3], axis=-1),
                     rotation_distance_deg(stream.rotations[left], stream.rotations[right])), axis=-1)


def state_jump_intervals(stream, config):
    """Only isolated one-sample excursions between stable local contexts."""
    keep = np.r_[np.diff(stream.times) > 0, True]
    stream = StateStream(stream.times[keep], stream.values[keep],
                         None if stream.rotations is None else stream.rotations[keep])
    times = stream.times
    if len(times) < 7:
        return []
    horizon = round(config.jump_max_duration_s * NS_PER_SECOND)
    window = round(config.jump_context_s * NS_PER_SECOND)
    pose = stream.rotations is not None
    minimum = np.array([config.position_jump_min_m, config.orientation_jump_min_deg]
                       if pose else [config.openness_jump_min])
    tolerance = np.array([config.position_tolerance_m, config.orientation_tolerance_deg]
                         if pose else [config.openness_tolerance])
    samples = np.arange(1, len(times)-1)
    outward = state_distances(stream, samples-1, samples)
    returning = state_distances(stream, samples, samples+1)
    bridge = state_distances(stream, samples-1, samples+1)
    # Subtract the neighbours' drift from BOTH excursion magnitudes. This
    # bounds departure from any intermediate state without fitting a trajectory.
    possible = (np.minimum(outward, returning)-bridge > minimum) & (bridge <= tolerance)
    candidates = samples[np.any(possible, axis=1)]
    gaps = np.diff(times)
    events = []
    for sample in candidates:
        before, after = sample-1, sample+1
        if times[before]-window < times[0] or times[after]+window > times[-1]:
            continue
        left = int(np.searchsorted(times, times[before]-window))
        right = int(np.searchsorted(times, times[after]+window, side='right'))
        # Complete contexts, at least three distinct native samples per side,
        # and no sampling gaps anywhere across the context or excursion.
        if before-left+1 < 3 or right-after < 3 or np.any(gaps[left:right-1] > horizon):
            continue
        stable_before = np.all(state_distances(stream, np.arange(left, before+1), before)
                               <= tolerance, axis=0)
        stable_after = np.all(state_distances(stream, np.arange(after, right), after)
                              <= tolerance, axis=0)
        if not np.any(possible[sample-1] & stable_before & stable_after):
            continue
        start, end = int(times[sample]), int(times[sample+1])
        if events and start-events[-1][1] <= horizon:
            events[-1] = (events[-1][0], end)
        else:
            events.append((start, end))
    return events


def state_change_times(stream, config):
    """Advance an anchor whenever a sample leaves its tolerance band.

    A fixed anchor lets slow motion accumulate while bounded noise does not.
    """
    visible = np.flatnonzero(np.r_[np.diff(stream.times) > 0, True])
    changes, start = [], 0
    while start + 1 < len(visible):
        reach = 1
        while True:
            window = visible[start+1:start+1+reach]
            moved = np.flatnonzero(stream.changed(window, visible[start], config))
            if len(moved) or len(window) < reach:
                break
            reach *= 2
        if not len(moved):
            break
        start += 1 + int(moved[0])
        changes.append(stream.times[visible[start]])
    return np.asarray(changes, dtype=np.int64)


def detect_episode(index, streams, end_ns, config=Config()):
    """Return contract records in rule order; timestamps remain native int64 ns."""
    records = []

    def add(name, start, end, reason):
        records.append(dict(index=index, stream=name, start_ns=int(start),
                            end_ns=int(end), reason=reason))

    starts, tails = [], []
    changes = []
    for name in STATE_NAMES:
        stream = streams[name]
        first = np.flatnonzero(stream.changed(slice(None), 0, config))
        last = np.flatnonzero(stream.changed(slice(None), -1, config))
        starts.append(int(stream.times[first[0]]) if len(first) else end_ns)
        tails.append(end_ns - int(stream.times[last[-1]]) if len(last) else end_ns)
        changes.append(state_change_times(stream, config))
    first, last = int(np.argmin(starts)), int(np.argmin(tails))
    if starts[first] / NS_PER_SECOND > config.max_boundary_idle_s:
        add(STATE_NAMES[first], 0, starts[first], 'boundary_idle')
    if tails[last] / NS_PER_SECOND > config.max_boundary_idle_s:
        add(STATE_NAMES[last], end_ns - tails[last], end_ns, 'boundary_idle')

    for name in STATE_NAMES:
        for start, end in state_jump_intervals(streams[name], config):
            add(name, start, end, 'state_jump')

    changes = np.unique(np.concatenate(changes))
    for left, right in zip(changes[:-1], changes[1:]):
        if (int(right) - int(left)) / NS_PER_SECOND > config.max_internal_idle_s:
            for name in STATE_NAMES:
                add(name, left, right, 'internal_idle')
    return records


def report_path(root, output=None):
    """Resolve the report destination and protect detection inputs."""
    output = Path(output).resolve() if output is not None else root / 'anomalies.jsonl'
    manifest = root / 'episodes.jsonl'
    if (output == manifest or output.suffix != '.jsonl'
            or output.is_relative_to(root / 'episodes')):
        raise ValueError('Output must be a .jsonl report outside episodes/ and cannot replace episodes.jsonl')
    return output


def detect_dataset(root, output=None, config=Config()):
    """Process one episode at a time and atomically replace the JSONL report."""
    root = Path(root).resolve()
    output = report_path(root, output)
    manifest = root / 'episodes.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    episodes, anomalies = 0, 0
    seen = set()
    try:
        with manifest.open(encoding='utf-8') as source, tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=output.parent,
                prefix=f'.{output.name}.', suffix='.tmp', delete=False) as report:
            temporary = Path(report.name)
            for index, line in enumerate(source):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    streams, end_ns = read_episode(root, record)
                    if record['episode_id'] in seen:
                        raise ValueError(f'Duplicate episode_id: {record["episode_id"]}')
                    seen.add(record['episode_id'])
                    results = detect_episode(index, streams, end_ns, config)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    raise ValueError(f'{manifest}: line {index + 1}: {exc}') from exc
                for result in results:
                    report.write(json.dumps(result, separators=(',', ':')) + '\n')
                episodes += 1
                anomalies += len(results)
            if not episodes:
                raise ValueError('episodes.jsonl is empty')
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return episodes, anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Exported dataset root')
    parser.add_argument('--output', type=Path, help='JSONL report (default: DATASET/anomalies.jsonl)')
    for field in fields(Config):
        parser.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                            type=field.type, help=f'Default: {field.default}')
    args = parser.parse_args()
    try:
        config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
        episodes, anomalies = detect_dataset(args.dataset, args.output, config)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Checked {episodes} episodes; wrote {anomalies} anomalies to '
          f'{args.output or args.dataset / "anomalies.jsonl"}')


if __name__ == '__main__':
    main()
