"""Pack the videos of exported datasets into large shard files, per docs/contract/packed-contract.md.

The packets are copied byte for byte from the export MP4s; nothing is decoded or re-encoded.
A dataset is packed shard by shard: a shard is committed when its index fragment is written,
so an interrupted run resumes by running the same command again. Several machines may run
the command on the same dataset at once; they divide the shards through claims on the shared
filesystem, and once every shard is committed one of them, holding the publish claim, writes the
index and info.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from loom_data import contract, paths, resumable
from loom_data.check.clean import packets
from loom_data.convert.claims import POLL_S, Claims

FORMAT = 'loom-packed'
VERSION = 1
SHARD_BYTES = 4 * 2**30
WORK = '.work'              # inside the packed directory until it is published
INFO = 'info.json'
EPISODES = 'episodes.parquet'
FRAMES = 'frames.parquet'


def shard_path(output, shard):
    return output / 'shards' / f'{shard:06d}.bin'


def video_keys(dataset):
    """Camera feature keys in packing order: top, left_wrist, right_wrist."""
    return [contract.video_key(camera) for camera in contract.LAYOUT_ORDER if camera in dataset.cameras]


def output_for(dataset_root, output_root=None):
    """<packed root>/<export name>/<source>/<layout> for a dataset at <export root>/<export name>/<source>/<layout>."""
    if output_root is not None:
        return Path(output_root).resolve() / dataset_root.parent.name / dataset_root.name
    export = paths.EXPORT.resolve()
    if not dataset_root.is_relative_to(export) or len(dataset_root.relative_to(export).parts) != 3:
        raise ValueError(f'{dataset_root}: not at <export root>/<export>/<source>/<layout>; pass --output')
    return paths.PACKED / dataset_root.relative_to(export)


def identity(dataset):
    """What the packed data must agree with: episode_index, episode_id, length of every episode."""
    return [[row['episode_index'], row['episode_id'], row['length']] for row in dataset.episodes]


def plan(dataset):
    """Shards as lists of episode indexes, each about SHARD_BYTES of video."""
    def size(episode):
        return sum(dataset.video(episode, camera)[0].stat().st_size for camera in dataset.cameras)

    with ThreadPoolExecutor(32) as pool:
        sizes = list(pool.map(size, dataset.episodes))
    shards, current, total = [], [], 0
    for episode, episode_size in zip(dataset.episodes, sizes, strict=True):
        current.append(episode['episode_index'])
        total += episode_size
        if total >= SHARD_BYTES:
            shards.append(current)
            current, total = [], 0
    if current:
        shards.append(current)
    return shards


def read_episode(dataset, episode, keys):
    """[(packet bytes, keyframe flags)] per camera, one packet per frame, and the extradata per camera."""
    import av
    streams, extradata = [], []
    for key in keys:
        path, first = dataset.video(episode, key.removeprefix('observation.images.'))
        if first != 0:
            raise ValueError(f'{path}: video does not start at the episode')
        with av.open(str(path)) as reader:
            frames = [(bytes(packet), packet.is_keyframe) for packet in packets(reader, path)]
            extradata.append(bytes(reader.streams.video[0].codec_context.extradata or b''))
        flags = [key_frame for _, key_frame in frames]
        if len(frames) != episode['length']:
            raise ValueError(f'{path}: {len(frames)} frames, episode has {episode["length"]}')
        if not flags[0] or any(not a and not b for a, b in zip(flags, flags[1:])):
            raise ValueError(f'{path}: keyframes are more than {contract.VIDEO_GOP} frames apart')
        streams.append(frames)
    return streams, extradata


def pack_shard(dataset, keys, episodes, shard, output, work):
    """Write one shard and commit it with its index fragment."""
    target = shard_path(output, shard)
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    digest, offset = hashlib.sha256(), 0
    frames = {'episode_index': [], 'frame_index': [], 'offset': [],
              **{f'{key}.size': [] for key in keys}, **{f'{key}.keyframe': [] for key in keys}}
    rows = []
    try:
        with open(temporary, 'wb', buffering=16 * 2**20) as file:
            for index in episodes:
                episode = dataset.episodes[index]
                streams, extradata = read_episode(dataset, episode, keys)
                rows.append({'episode_index': index, 'episode_id': episode['episode_id'],
                             'length': episode['length'], 'shard': shard,
                             **{f'{key}.extradata': data for key, data in zip(keys, extradata, strict=True)}})
                for frame in range(episode['length']):
                    frames['episode_index'].append(index)
                    frames['frame_index'].append(frame)
                    frames['offset'].append(offset)
                    for key, stream in zip(keys, streams, strict=True):
                        data, key_frame = stream[frame]
                        file.write(data)
                        digest.update(data)
                        offset += len(data)
                        frames[f'{key}.size'].append(len(data))
                        frames[f'{key}.keyframe'].append(key_frame)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    pq.write_table(frame_table(frames, keys), work / 'fragments' / f'{shard:06d}.frames.parquet')
    pq.write_table(episode_table(rows, keys), work / 'fragments' / f'{shard:06d}.episodes.parquet')
    resumable.atomic_json(work / 'fragments' / f'{shard:06d}.json', {'bytes': offset, 'sha256': digest.hexdigest()})


def frame_table(columns, keys):
    fields = [pa.field('episode_index', pa.int64()), pa.field('frame_index', pa.int64()), pa.field('offset', pa.int64()),
              *(pa.field(f'{key}.size', pa.int32()) for key in keys),
              *(pa.field(f'{key}.keyframe', pa.bool_()) for key in keys)]
    return pa.table({field.name: pa.array(columns[field.name], field.type) for field in fields},
                    schema=pa.schema(fields))


def episode_table(rows, keys):
    schema = pa.schema([pa.field('episode_index', pa.int64()), pa.field('episode_id', pa.string()),
                        pa.field('length', pa.int64()), pa.field('shard', pa.int64()),
                        *(pa.field(f'{key}.extradata', pa.binary()) for key in keys)])
    return pa.Table.from_pylist(rows, schema=schema)


def committed(output, work, shard):
    """The shard's record, if the shard and its index fragment are both in place."""
    record = work / 'fragments' / f'{shard:06d}.json'
    if not record.exists():
        return None
    record = resumable.read_json(record)
    path = shard_path(output, shard)
    return record if path.exists() and path.stat().st_size == record['bytes'] else None


def prepare(dataset, output):
    """Create the work directory and its plan, or check an existing one against the export."""
    work = output / WORK
    (work / 'fragments').mkdir(parents=True, exist_ok=True)
    (work / 'claims').mkdir(exist_ok=True)
    (output / 'shards').mkdir(exist_ok=True)
    plan_path = work / 'plan.json'
    if not plan_path.exists():
        resumable.atomic_json(plan_path, {'identity': identity(dataset), 'shards': plan(dataset)})
    saved = resumable.read_json(plan_path)
    if saved['identity'] != identity(dataset):
        raise ValueError(f'{dataset.root}: the export changed since packing started; delete {output} and pack again')
    return saved['shards']


def claim(claims, output, task, take_over=True):
    """Claim a task; False also when another machine has published and removed the work directory."""
    try:
        return claims.acquire(task) or (take_over and claims.take_over(task))
    except FileNotFoundError:
        if (output / INFO).exists():
            return False
        raise


def work(dataset_root, output, shards):
    """Pack unclaimed shards until every shard is committed or another machine has published."""
    resumable.configure_worker()
    dataset = contract.Dataset(dataset_root)
    keys, work_dir = video_keys(dataset), output / WORK
    claims = Claims(work_dir / 'claims')
    try:
        first_pass = True
        while not (output / INFO).exists():
            pending = [s for s in range(len(shards)) if committed(output, work_dir, s) is None]
            if not pending:
                return
            for shard in pending:
                if claim(claims, output, shard, take_over=not first_pass):
                    with claims.held(shard):
                        if committed(output, work_dir, shard) is None:
                            pack_shard(dataset, keys, shards[shard], shard, output, work_dir)
            if any(committed(output, work_dir, s) is None for s in pending):
                time.sleep(POLL_S)  # other workers hold the rest
            first_pass = False
    finally:
        claims.close()


def publish(dataset, output, shards):
    """Merge the index fragments and write info.json last."""
    work_dir, keys = output / WORK, video_keys(dataset)
    records = [committed(output, work_dir, shard) for shard in range(len(shards))]
    if any(record is None for record in records):
        raise ValueError(f'{output}: not every shard is packed')
    frames = pa.concat_tables(pq.read_table(work_dir / 'fragments' / f'{s:06d}.frames.parquet') for s in range(len(shards)))
    episodes = pa.concat_tables(pq.read_table(work_dir / 'fragments' / f'{s:06d}.episodes.parquet')
                                for s in range(len(shards)))
    if ([[i, e, n] for i, e, n in zip(*(episodes[c].to_pylist() for c in ('episode_index', 'episode_id', 'length')))]
            != identity(dataset) or frames.num_rows != sum(row['length'] for row in dataset.episodes)):
        raise ValueError(f'{output}: packed index does not match the export')
    for name, table in ((FRAMES, frames), (EPISODES, episodes)):
        temporary = output / f'.{name}.{uuid.uuid4().hex}.tmp'
        pq.write_table(table, temporary)
        temporary.replace(output / name)
    resumable.atomic_json(output / INFO, {
        'format': FORMAT, 'version': VERSION, 'codec': 'h264', 'fps': contract.FPS, 'cameras': keys,
        'episodes': episodes.num_rows, 'frames': frames.num_rows,
        'shards': [{'bytes': record['bytes'], 'sha256': record['sha256']} for record in records]})
    shutil.rmtree(work_dir, ignore_errors=True)
    for leftover in (output / 'shards').glob('.*.tmp'):  # from interrupted workers
        leftover.unlink(missing_ok=True)


def is_current(dataset, output):
    """True if `output` holds a published packing of this export."""
    if not (output / INFO).exists():
        return False
    table = pq.read_table(output / EPISODES, columns=['episode_index', 'episode_id', 'length'])
    if [list(row.values()) for row in table.to_pylist()] != identity(dataset):
        raise ValueError(f'{output}: packed from a different version of {dataset.root}; delete it and pack again')
    return True


def pack(dataset_root, output, workers):
    dataset = contract.Dataset(dataset_root)
    if is_current(dataset, output):
        print(f'{output}: already packed', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    shards = prepare(dataset, output)
    print(f'{dataset_root} -> {output}: {len(dataset.episodes)} episodes in {len(shards)} shards', flush=True)
    if workers == 1:
        work(dataset_root, output, shards)
    else:
        context = multiprocessing.get_context('spawn')
        processes = [context.Process(target=work, args=(dataset_root, output, shards)) for _ in range(workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        if any(process.exitcode for process in processes):
            raise RuntimeError(f'{output}: a worker failed; run the command again to resume')
    # One machine publishes; the others wait for its info.json.
    claims = Claims(output / WORK / 'claims')
    try:
        while not (output / INFO).exists():
            if claim(claims, output, 'publish'):
                with claims.held('publish'):
                    if not (output / INFO).exists():
                        publish(dataset, output, shards)
            else:
                time.sleep(POLL_S)
    finally:
        claims.close()
    shutil.rmtree(output / WORK, ignore_errors=True)
    print(f'{output}: packed', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='loom-data pack', description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='+', type=Path, help='a layout dataset, a source folder or an export folder')
    parser.add_argument('--output', type=Path,
                        help='packed root; each layout goes to <output>/<source>/<layout> '
                             '(default: dataset/packed/<export> for datasets under dataset/export)')
    parser.add_argument('--workers', type=int, default=resumable.available_cpus(),
                        help='worker processes on this machine (default: available CPUs)')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    for root in contract.discover_datasets(args.paths):
        pack(root, output_for(root, args.output), args.workers)
    return 0
