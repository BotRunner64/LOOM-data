"""Anomaly rules for exported episodes: boundary idling, internal idling and EEF spikes.

docs/contract/anomaly-contract.md is the specification. Rules read the exported
30 fps state frames; `loom_data.check.job` runs them over a dataset.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import numpy as np

from loom_data.contract import FPS, STATE_NAMES, STATE_SLICES, is_pose

FIELDS = ('episode_index', 'reason', 'stream', 'start_frame', 'end_frame')
REASONS = ('boundary_idle', 'internal_idle', 'eef_spike')
SPIKE_MAX_FRAMES = 4       # a native one-sample spike spreads over at most 4 frames at 30 fps
SPIKE_CONTEXT_FRAMES = 3   # frames on each side that must agree with the reference frame


@dataclass(frozen=True)
class Config:
    position_tolerance_m: float = .01
    orientation_tolerance_deg: float = 5.
    openness_tolerance: float = .05
    max_boundary_idle_s: float = 3.
    max_internal_idle_s: float = 10.
    position_spike_min_m: float = .02
    orientation_spike_min_deg: float = 10.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{field.name} must be finite and nonnegative')


def rotation_quaternions(poses):
    """Contract column ordered 3x3 rotation matrices to unit wxyz quaternions."""
    r = poses[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1).copy()
    if (not np.isfinite(r).all()
            or not np.allclose(r @ np.swapaxes(r, 1, 2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
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
    """One state stream, one row per frame: N x 12 poses with their quaternions, or N openness values."""
    values: np.ndarray
    rotations: np.ndarray | None = None

    def exceeds(self, left, right, position, orientation, openness=None):
        """Whether any quantity differs by more than its limit between frames `left` and `right`."""
        if self.rotations is None:
            return np.abs(self.values[left] - self.values[right]) > openness
        return ((np.linalg.norm(self.values[left, :3] - self.values[right, :3], axis=-1) > position)
                | (rotation_distance_deg(self.rotations[left], self.rotations[right]) > orientation))

    def changed(self, left, right, config):
        return self.exceeds(left, right, config.position_tolerance_m, config.orientation_tolerance_deg,
                            config.openness_tolerance)


def read_episode(dataset, episode):
    """The four state streams of one episode of a `contract.Dataset`."""
    state, _ = dataset.frames(episode)
    streams = {}
    for name in STATE_NAMES:
        values = state[:, STATE_SLICES[name]]
        streams[name] = (StateStream(values, rotation_quaternions(values)) if is_pose(name)
                         else StateStream(values[:, 0]))
    return streams


def moved(streams, frames, reference, config):
    """Per frame: whether any stream changed relative to frame `reference`."""
    result = False
    for stream in streams.values():
        result = result | stream.changed(frames, reference, config)
    return result


def idle_segments(streams, count, config):
    """Split the episode into [start, end) segments whose frames have not moved since `start`."""
    starts = [0]
    while starts[-1] + 1 < count:
        start, reach = starts[-1], 8
        while True:
            frames = np.arange(start + 1, min(count, start + 1 + reach))
            hits = np.flatnonzero(moved(streams, frames, start, config))
            if len(hits) or frames[-1] == count - 1:
                break
            reach *= 4
        if not len(hits):
            break
        starts.append(int(frames[hits[0]]))
    return list(zip(starts, starts[1:] + [count]))


def spike_intervals(stream, config):
    """[start, end) frame runs that leave reference frame `start - 1` for 1 to
    SPIKE_MAX_FRAMES frames and return to it, with stable frames on both sides."""
    count = len(stream.values)
    frames = np.arange(count)
    changed, large, exists = {}, {}, {}
    for offset in range(-SPIKE_CONTEXT_FRAMES, SPIKE_MAX_FRAMES + SPIKE_CONTEXT_FRAMES + 1):
        other = frames + offset
        exists[offset] = (other >= 0) & (other < count)
        other = np.clip(other, 0, count - 1)
        changed[offset] = stream.changed(frames, other, config)
        large[offset] = stream.exceeds(frames, other, config.position_spike_min_m,
                                       config.orientation_spike_min_deg)

    def stable(offsets):
        result = np.ones(count, dtype=bool)
        for offset in offsets:
            result &= exists[offset] & ~changed[offset]
        return result

    before = stable(range(-SPIKE_CONTEXT_FRAMES, 0))
    intervals = []
    for length in range(1, SPIKE_MAX_FRAMES + 1):
        away = before.copy()
        peak = np.zeros(count, dtype=bool)
        for offset in range(1, length + 1):
            away &= exists[offset] & changed[offset]
            peak |= large[offset]
        after = stable(range(length + 1, length + 1 + SPIKE_CONTEXT_FRAMES))
        intervals += [(int(a) + 1, int(a) + 1 + length) for a in np.flatnonzero(away & peak & after)]
    return sorted(intervals)


def detect_episode(index, streams, config=Config()):
    """Report records of one episode, ordered by start frame."""
    count = len(streams[STATE_NAMES[0]].values)
    records = []

    def add(reason, start, end, stream=None):
        records.append(dict(episode_index=index, reason=reason, stream=stream,
                            start_frame=int(start), end_frame=int(end)))

    frames = np.arange(count)
    since_first = np.flatnonzero(moved(streams, frames, 0, config))
    until_last = np.flatnonzero(moved(streams, frames, count - 1, config))
    head = int(since_first[0]) if len(since_first) else count
    tail = int(until_last[-1]) + 1 if len(until_last) else 0
    if head / FPS > config.max_boundary_idle_s:
        add('boundary_idle', 0, head)
    if head < count and (count - tail) / FPS > config.max_boundary_idle_s:
        add('boundary_idle', tail, count)

    for start, end in idle_segments(streams, count, config):
        if start > 0 and end < count and (end - start) / FPS > config.max_internal_idle_s:
            add('internal_idle', start, end)

    for name in STATE_NAMES:
        if is_pose(name):
            for start, end in spike_intervals(streams[name], config):
                add('eef_spike', start, end, name)
    return sorted(records, key=lambda record: record['start_frame'])


def report_path(root, output=None):
    """Resolve the report destination and protect detection inputs."""
    output = Path(output).resolve() if output is not None else root / 'anomalies.jsonl'
    if output.suffix != '.jsonl' or any(output.is_relative_to(root / name) for name in ('meta', 'data', 'videos')):
        raise ValueError('Output must be a .jsonl report outside the dataset\'s meta/, data/ and videos/')
    return output
