"""Helpers shared by the conversion tests."""
from fractions import Fraction
from pathlib import Path
import json
import shutil
import tempfile

import av
import numpy as np

from loom_data import contract, paths
from loom_data.video import encode_frames
from loom_data.check import anomalies, job as check_job
from loom_data.convert import convert, job

MODELS = paths.MODELS

IDENTITY_POSE = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]


def datasets(output):
    """The layout datasets of a published conversion, by layout name."""
    return {path.name: contract.Dataset(path) for path in sorted(Path(output).iterdir())}


def read_export(output):
    """Every exported episode of a conversion, across layouts, sorted by episode_id.

    Each is a dict: episode_id, layout, cameras, length, state (N x 26 float64),
    one N x 12 / N array per contract.STATE_NAMES entry, tasks (per-frame text),
    instructions ([{start_ns, end_ns, text}] runs on the 30 fps clock) and
    videos (camera -> (path, first frame index)).
    """
    episodes = []
    for layout, dataset in datasets(output).items():
        for episode in dataset.episodes:
            state, tasks = dataset.frames(episode)
            item = {'episode_id': episode['episode_id'], 'layout': layout, 'cameras': dataset.cameras,
                    'length': episode['length'], 'state': state, 'tasks': tasks,
                    'instructions': contract.instruction_intervals(tasks),
                    'videos': {camera: dataset.video(episode, camera) for camera in dataset.cameras}}
            for name in contract.STATE_NAMES:
                values = state[:, contract.STATE_SLICES[name]]
                item[name] = values if contract.is_pose(name) else values[:, 0]
            episodes.append(item)
    return sorted(episodes, key=lambda item: item['episode_id'])


def manifest(output):
    """Episode ids of a published conversion, in dataset order within each layout."""
    return [{'episode_id': episode['episode_id']} for dataset in datasets(output).values()
            for episode in dataset.episodes]


def write_video(path, values, size=(64, 48), rate=30, codec='libx264', options=None):
    """A fixture video with one uniform gray frame per entry of `values` (0-255). Default 4:3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as output:
        stream = output.add_stream(codec, rate=rate)
        stream.width, stream.height = size
        stream.pix_fmt = 'yuv420p'
        if options:
            stream.options = options
        for index, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), value, np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, rate)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def video_means(path, first=0, count=None):
    """Mean gray level of each decoded frame, optionally of a frame range."""
    with av.open(str(path)) as reader:
        means = [frame.to_ndarray(format='rgb24').mean() for frame in reader.decode(video=0)]
    return means[first:None if count is None else first + count]


def make_dataset(root, episodes, cameras=('top',), size=(64, 48)):
    """Build an exported dataset at `root` straight from arrays, without a source.

    `episodes` maps episode_id -> dict with optional keys: `poses` (N x 12, both
    arms) or `left_eef` / `right_eef`, `left_gripper` / `right_gripper` (N), and
    `tasks` (one text, or one per frame). Rows are 30 fps frames. Returns the Dataset.
    """
    root = Path(root)
    with tempfile.TemporaryDirectory(dir=root.parent) as staging:
        staged = []
        for episode_id, values in episodes.items():
            length = next(len(values[k]) for k in ('poses', 'left_eef', 'left_gripper', 'right_eef',
                                                      'right_gripper') if k in values)
            times = contract.frame_times(length)
            streams = {}
            for name in contract.STATE_NAMES:
                default = np.tile(IDENTITY_POSE, (length, 1)) if contract.is_pose(name) else np.zeros(length)
                streams[name] = (times, np.asarray(values.get(
                    name, values.get('poses', default) if contract.is_pose(name) else default), dtype=float))
            tasks = values.get('tasks', 'task')
            tasks = [tasks] * length if isinstance(tasks, str) else list(tasks)
            starts = [0] + [k for k in range(1, length) if tasks[k] != tasks[k - 1]]
            instructions = [{'start_ns': int(times[a]), 'end_ns': int(times[b] if b < length else times[-1] + 1),
                             'text': tasks[a]} for a, b in zip(starts, starts[1:] + [length])]
            directory = Path(staging) / episode_id
            contract.write_frames(directory, streams, instructions, 0)
            videos = {camera: encode_frames(directory / f'{camera}.mp4', _gray_frames(length, size), times)
                      for camera in cameras}
            record = contract.episode_record(episode_id, length, videos)
            contract.validate_episode(directory, record)
            staged.append((record, directory))
        contract.build_dataset(root, 'test', staged)
    return contract.Dataset(root)


def _gray_frames(count, size):
    for index in range(count):
        yield av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), index % 256, np.uint8), format='rgb24')


def episode_errors(output):
    """Text of every recorded episode failure of a conversion."""
    return ''.join(p.read_text() for p in sorted(job.work_directory(output).glob('errors/*.err')))


def assert_conversion_fails(case, pattern, dataset, source, output, **options):
    """A fresh conversion must fail without publishing.

    A nonempty `pattern` must match the raised error or a recorded episode failure.
    """
    shutil.rmtree(job.work_directory(output), ignore_errors=True)
    try:
        convert(dataset, source, output, **options)
    except (ValueError, OSError, RuntimeError) as exc:
        if pattern:
            case.assertRegex(f'{type(exc).__name__}: {exc}\n{episode_errors(output)}', pattern)
    else:
        case.fail('conversion unexpectedly succeeded')
    case.assertFalse(Path(output).exists(), 'a failed conversion must not publish')


def check(root, output=None, config=None):
    """Run anomaly detection from a clean state; return (episodes, anomalies)."""
    root = Path(root).resolve()
    shutil.rmtree(check_job.work_directory(anomalies.report_path(root, output)), ignore_errors=True)
    return check_job.run_serial(root, output, config)


def expected_report(root, config=anomalies.Config()):
    """The report the job must write, computed directly from the rules."""
    dataset = contract.Dataset(Path(root))
    rows = []
    for episode in dataset.episodes:
        rows += anomalies.detect_episode(episode['episode_index'], anomalies.read_episode(dataset, episode), config)
    return ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
