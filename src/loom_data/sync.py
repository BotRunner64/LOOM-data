"""Resampling onto the contract's fixed 30 fps frame sequence (export contract, section 3)."""
import numpy as np

from loom_data.sources.policy import SourceDataError

FPS = 30
NS_PER_SECOND = 1_000_000_000
STATE_GAP_NS = 500_000_000   # widest pair of state samples a frame may be interpolated across
FRAME_GAP_NS = 250_000_000   # farthest RGB frame a frame time may take; nearer gaps repeat frames


def frame_ns(index):
    """Time of frame `index` on the episode clock, rounded to whole nanoseconds."""
    return (2 * NS_PER_SECOND * index + FPS) // (2 * FPS)


def ordered(times, label='sample'):
    times = np.asarray(times, dtype=np.int64)
    if times.ndim != 1 or not len(times) or np.any(times[1:] < times[:-1]):
        raise ValueError(f'Empty or unordered {label} timestamps')
    return times


def frame_grid(streams):
    """Frame times covering the interval every stream has data for; never extrapolates."""
    streams = [ordered(times) for times in streams]
    start, end = max(int(t[0]) for t in streams), min(int(t[-1]) for t in streams)
    count = 0
    if end >= start:
        # Frames 0..last with frame_ns(last) <= end - start. Frame times are rounded to
        # whole ns, so a stream ending exactly on a rounded-down frame time keeps that frame.
        last = (end - start) * FPS // NS_PER_SECOND
        last += frame_ns(last + 1) <= end - start
        count = last + 1
    if count < 2:
        raise SourceDataError('no_common_interval', 'Streams share less than two 30 fps frames')
    return start + np.array([frame_ns(k) for k in range(count)], dtype=np.int64)


def table_grid(times, fps=None):
    """Grid for a row-aligned frame table; None when its rows already are the 30 fps frames.

    `fps` is the source's declared rate. Without one, the median row interval decides.
    """
    times = ordered(times)
    if len(times) < 2 or times[-1] <= times[0]:
        raise SourceDataError('no_common_interval', 'Empty or zero-duration episode')
    if fps is None:
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        fps = NS_PER_SECOND / float(np.median(intervals))
    if abs(fps - FPS) <= .01 * FPS:
        return None
    return frame_grid([times])


def brackets(times, grid):
    """Indices of the samples around each frame time and the weight of the later one."""
    times = ordered(times)
    if len(times) == 1:
        zeros = np.zeros(len(grid), dtype=np.int64)
        return zeros, zeros, np.zeros(len(grid)), np.zeros(len(grid), dtype=np.int64)
    high = np.clip(np.searchsorted(times, grid, side='right'), 1, len(times) - 1)
    low = high - 1
    span = times[high] - times[low]
    weight = np.where(span > 0, (grid - times[low]) / np.maximum(span, 1), 0.)
    if np.any((weight < 0) | (weight > 1)):
        raise ValueError('Frame times lie outside the sampled interval')
    return low, high, weight, span


def check_gap(weight, span, label):
    between = (weight > 0) & (weight < 1)
    if np.any(span[between] > STATE_GAP_NS):
        raise SourceDataError('sampling_gap', f'{label}: state samples are more than '
                              f'{STATE_GAP_NS // 1_000_000} ms apart')


def resample_scalar(times, values, grid, label='state'):
    low, high, weight, span = brackets(times, grid)
    check_gap(weight, span, label)
    values = np.asarray(values, dtype=np.float64)
    return values[low] * (1 - weight) + values[high] * weight


def quaternions(rotations):
    """Rotation matrices -> unit wxyz quaternions."""
    r = np.asarray(rotations, dtype=np.float64)
    r00, r11, r22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    squared = np.column_stack((1+r00+r11+r22, 1+r00-r11-r22, 1-r00+r11-r22, 1-r00-r11+r22))
    candidates = np.stack((
        np.column_stack((squared[:, 0], r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1])),
        np.column_stack((r[:, 2, 1]-r[:, 1, 2], squared[:, 1], r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0])),
        np.column_stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1], squared[:, 2], r[:, 2, 1]+r[:, 1, 2])),
        np.column_stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0], r[:, 2, 1]+r[:, 1, 2], squared[:, 3])),
    ), axis=1)
    q = candidates[np.arange(len(r)), np.argmax(squared, axis=1)]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rotation_matrices(q):
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack((
        np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), axis=1),
        np.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), axis=1),
        np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1)), axis=1)


def slerp(a, b, weight):
    b = np.where(np.sum(a * b, axis=1, keepdims=True) < 0, -b, b)
    angle = np.arccos(np.clip(np.sum(a * b, axis=1), -1., 1.))
    sine = np.sin(angle)
    small = sine < 1e-9
    safe = np.where(small, 1., sine)
    first = np.where(small, 1 - weight, np.sin((1 - weight) * angle) / safe)
    second = np.where(small, weight, np.sin(weight * angle) / safe)
    return first[:, None] * a + second[:, None] * b


def resample_poses(times, poses, grid, label='pose'):
    """N x 12 contract poses: positions interpolate linearly, rotations by slerp."""
    low, high, weight, span = brackets(times, grid)
    check_gap(weight, span, label)
    poses = np.asarray(poses, dtype=np.float64)
    position = poses[low, :3] * (1 - weight[:, None]) + poses[high, :3] * weight[:, None]
    q = quaternions(poses[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1))
    rotation = rotation_matrices(slerp(q[low], q[high], weight))
    return np.concatenate((position, rotation.transpose(0, 2, 1).reshape(-1, 9)), axis=1)


def nearest(times, grid):
    """Index of the source frame acquired closest to each frame time."""
    times = ordered(times, 'video')
    low, high, _, _ = brackets(times, grid)
    # A frame time midway between two source frames takes the earlier one, so a 15 fps
    # source repeats each frame instead of showing an image from the future. The
    # microsecond margin keeps that choice stable against timestamp rounding.
    index = np.where(grid - times[low] <= times[high] - grid + 1000, low, high)
    if np.any(np.abs(times[index] - grid) > FRAME_GAP_NS):
        raise SourceDataError('sampling_gap', f'No RGB frame within {FRAME_GAP_NS // 1_000_000} ms of a frame time')
    return index
