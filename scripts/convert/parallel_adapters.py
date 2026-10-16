"""Source discovery and conversion adapters for the fixed parallel plan."""
from __future__ import annotations

import importlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import tempfile

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def _no_progress(label, total=None, *, unit='item'):
    return tqdm(desc=label, total=total, unit=unit, disable=True)


def module(dataset):
    return importlib.import_module(f'scripts.convert.{dataset}')


def make_config(dataset, source, *, model_dir=None, limit=None, limit_episodes=None,
                robot_type=None, tcp_offset=None, r1lite_tcp_offset=None, r1pro_tcp_offset=None,
                gripper_open_rad=None, gripper_closed_rad=None, tool_rpy=None,
                left_tool_rpy=None, right_tool_rpy=None, gripper_open=None, gripper_closed=None,
                instruction=None):
    source = Path(source).resolve()
    if not source.exists() or (dataset != 'agibotworld2026' and not source.is_dir()):
        raise ValueError(f'Input does not exist or has the wrong type: {source}')
    for value in (limit, limit_episodes):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Limits must be positive integers')
    if dataset not in ('abc130k', 'molmoact2', 'hifi_umi', 'galaxea', 'agibotworld2026', 'realomin', 'agibotworld_beta'):
        raise ValueError(f'Unknown dataset: {dataset}')
    if dataset != 'galaxea' and any(x is not None for x in (robot_type, r1lite_tcp_offset, r1pro_tcp_offset)):
        raise ValueError('Robot type and Galaxea TCP overrides require --dataset galaxea')
    if tcp_offset is not None and dataset not in ('agibotworld2026', 'realomin', 'agibotworld_beta'):
        raise ValueError('--tcp-offset requires --dataset agibotworld2026, realomin or agibotworld_beta')
    if dataset != 'hifi_umi' and any(x is not None for x in (gripper_open_rad, gripper_closed_rad)):
        raise ValueError('Gripper angle overrides require --dataset hifi_umi')
    if dataset != 'realomin' and any(x is not None for x in (tool_rpy, instruction)):
        raise ValueError('--tool-rpy and --instruction require --dataset realomin')
    if dataset != 'agibotworld_beta' and any(x is not None for x in (left_tool_rpy, right_tool_rpy)):
        raise ValueError('Left/right tool RPY overrides require --dataset agibotworld_beta')
    if dataset not in ('realomin', 'agibotworld_beta') and any(x is not None for x in (gripper_open, gripper_closed)):
        raise ValueError('Source gripper endpoints require --dataset realomin or agibotworld_beta')
    source_module = module(dataset)
    params = {}
    if dataset in ('realomin', 'agibotworld_beta'):
        params['data_policy_version'] = 1
        if model_dir is not None:
            raise ValueError(f'{dataset} uses recorded poses, not --model-dir')
        params['tcp_offset'] = list(tcp_offset if tcp_offset is not None else source_module.DEFAULT_TCP_OFFSET)
        params['gripper_open'] = source_module.DEFAULT_GRIPPER_OPEN if gripper_open is None else gripper_open
        params['gripper_closed'] = source_module.DEFAULT_GRIPPER_CLOSED if gripper_closed is None else gripper_closed
        source_module.normalized_gripper(0., params['gripper_closed'], params['gripper_open'])
        if dataset == 'realomin':
            params['tool_rpy'] = list(tool_rpy if tool_rpy is not None else source_module.DEFAULT_TOOL_RPY)
            params['instruction'] = instruction
            if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
                raise ValueError('Instruction override must be nonempty')
            source_module.tool_transform(params['tcp_offset'], params['tool_rpy'])
        else:
            for side, override in (('left', left_tool_rpy), ('right', right_tool_rpy)):
                params[f'{side}_tool_rpy'] = list(override if override is not None else getattr(source_module, f'DEFAULT_{side.upper()}_TOOL_RPY'))
                source_module.tool_transform(params['tcp_offset'], params[f'{side}_tool_rpy'])
    elif dataset == 'hifi_umi':
        if model_dir is not None:
            raise ValueError('HiFi-UMI does not use a robot model')
        opened = source_module.DEFAULT_GRIPPER_OPEN_RAD if gripper_open_rad is None else gripper_open_rad
        closed = 0. if gripper_closed_rad is None else gripper_closed_rad
        source_module.validate_gripper_range(closed, opened)
        params.update(gripper_open_rad=opened, gripper_closed_rad=closed)
    else:
        default = {'abc130k': 'yam', 'molmoact2': 'yam', 'agibotworld2026': 'agibot_g2', 'galaxea': ''}[dataset]
        model_dir = str(Path(model_dir or ROOT / 'assets/robot_models' / default).resolve())
    if dataset == 'agibotworld2026':
        params['tcp_offset'] = list(tcp_offset if tcp_offset is not None else source_module.DEFAULT_TCP_OFFSET)
    if dataset == 'galaxea':
        if robot_type is not None and robot_type not in source_module.DEFAULT_TCP_OFFSETS:
            raise ValueError(f'Unsupported robot_type: {robot_type}')
        params['robot_type'] = robot_type
        params['tcp_offsets'] = {kind: list(offset if offset is not None else source_module.DEFAULT_TCP_OFFSETS[kind])
                                 for kind, offset in [('r1lite', r1lite_tcp_offset), ('r1pro', r1pro_tcp_offset)]}
    offsets = ([params['tcp_offset']] if 'tcp_offset' in params else []) + list(params.get('tcp_offsets', {}).values())
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in offsets):
        raise ValueError('TCP offsets require three finite metres')
    return {'dataset': dataset, 'source': str(source), 'model_dir': model_dir,
            'limit': limit, 'limit_episodes': limit_episodes, 'params': params}


def configuration(args):
    names = ('model_dir', 'limit', 'limit_episodes', 'robot_type', 'tcp_offset',
             'r1lite_tcp_offset', 'r1pro_tcp_offset', 'gripper_open_rad', 'gripper_closed_rad',
             'tool_rpy', 'left_tool_rpy', 'right_tool_rpy', 'gripper_open', 'gripper_closed', 'instruction')
    return make_config(args.dataset, args.input_dir, **{k: getattr(args, k, None) for k in names})


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def discover(config, progress=_no_progress):
    dataset, source = config['dataset'], Path(config['source'])
    m = module(dataset)
    tasks, shared = [], []
    task_limit = min((v for v in (config['limit'], config['limit_episodes']) if v is not None), default=None)
    if dataset in ('realomin', 'agibotworld_beta'):
        found = m.iter_discover(source, config['params']['instruction']) if dataset == 'realomin' else m.iter_discover(source)
        if (source / 'subset_manifest.json').exists():
            shared.append(source / 'subset_manifest.json')
        with progress('Discovering native-pose episodes', unit='episode') as bar:
            for episode_id, item in islice(found, task_limit):
                if dataset == 'realomin':
                    tasks.append({'source': str(item['path']), 'instruction': item['instruction'],
                                  'episode_ids': [episode_id], 'inputs': [str(item['path'])]})
                else:
                    task_id = int(item['state'].parent.parent.name)
                    shared.append(source / f'task_info/task_{task_id}.json')
                    tasks.append({'source': str(item['state']),
                                  'videos': {camera: str(path) for camera, path in item['videos'].items()},
                                  'annotation_path': str(item['annotation_path']),
                                  'episode_index': item['episode_index'], 'episode_ids': [episode_id],
                                  'inputs': [str(item['state']), *(str(p) for p in item['videos'].values())]})
                if 'drop' in item:
                    tasks[-1]['drop'] = item['drop']
                bar.update()
    elif dataset == 'abc130k':
        found = []
        with progress('Scanning for episode.mcap files', unit='episode') as bar:
            for path in source.rglob('episode.mcap'):
                found.append(path)
                bar.update()
        for path in sorted(found)[:task_limit]:
            tasks.append({'source': str(path.parent), 'episode_ids': [path.parent.name],
                          'inputs': [str(path), str(path.parent / 'annotation.mcap')]})
    elif dataset == 'molmoact2' and not (source / 'source').is_dir():
        reader = m.DatasetReader(source)
        shared = [source / 'meta' / name for name in ('info.json', 'tasks.parquet', 'tasks_annotated.parquet')]
        shared.extend(reader.metadata_files)
        indices = sorted(reader.episodes)[:task_limit]
        with progress('Discovering MolmoAct2 episodes', len(indices), unit='episode') as bar:
            for index in indices:
                row = reader.episodes[index]
                inputs = {row['data_path'], *(video['path'] for video in row['videos'].values())}
                tasks.append({'source': str(source), 'layout': 'lerobot_v3', 'index': index,
                              'episode_ids': [f'episode_{index:06d}'], 'inputs': sorted(inputs)})
                bar.update()
    elif dataset == 'hifi_umi' and not (source / 'source').is_dir():
        shards = m.discover_shards(source)
        remaining = task_limit
        with progress('Indexing HiFi-UMI shards', len(shards), unit='shard') as bar:
            for shard in shards:
                reader = m.ShardReader(shard)
                indices = sorted(reader.episodes)[:remaining]
                if indices:
                    shared.extend(shard / 'meta' / name for name in ('info.json', 'modality.json', 'tasks.parquet'))
                    tasks.append({'source': str(shard), 'layout': 'lerobot_v3_shard',
                                  'episode_ids': [m.shard_episode_id(shard, i) for i in indices],
                                  'inputs': [str(p) for p in reader.inputs(indices)]})
                bar.update()
                if remaining is not None:
                    remaining -= len(indices)
                    if remaining == 0:
                        break
    elif dataset in ('molmoact2', 'hifi_umi'):
        meta = source / ('source/meta' if dataset == 'molmoact2' else 'source')
        info = load_json(meta / 'info.json')
        shared = [meta / 'info.json', meta / 'tasks.parquet']
        if dataset == 'molmoact2':
            m.validate_source(info)
            m.load_tasks(meta)
            shared.append(meta / 'tasks_annotated.parquet')
        else:
            m.validate_source(info, load_json(meta / 'modality.json'))
            m.load_tasks(meta / 'tasks.parquet')
            shared.append(meta / 'modality.json')
        episodes = sorted(p for p in (source / 'episodes').iterdir() if p.is_dir())
        selected = episodes if task_limit is None else episodes[:task_limit]
        with progress('Discovering episodes', len(selected), unit='episode') as bar:
            for path in selected:
                if not re.fullmatch(r'episode_\d+', path.name):
                    raise ValueError(f'Invalid episode directory: {path}')
                tasks.append({'source': str(path), 'episode_ids': [path.name],
                              'inputs': [str(path / 'data.parquet'), str(path / 'episode.json'),
                                         *(str(path / 'videos' / f'{key}.mp4') for key in m.CAMERAS)]})
                bar.update()
    elif dataset == 'galaxea':
        directories = m.discover_tasks(source)
        with progress('Discovering tasks', len(directories), unit='task') as bar:
            for directory in directories:
                if task_limit is not None and len(tasks) >= task_limit:
                    break
                bar.update()
                info = load_json(directory / 'meta/info.json')
                if config['params']['robot_type'] is not None and info.get('robot_type') != config['params']['robot_type']:
                    continue
                _, cameras = m.validate_source(info)
                metadata = m.read_jsonl(directory / 'meta/episodes.jsonl', 'episode_index')
                m.read_jsonl(directory / 'meta/tasks.jsonl', 'task_index')
                chunks = info['chunks_size']
                if len(metadata) != info['total_episodes'] or type(chunks) is not int or chunks < 1:
                    raise ValueError(f'{directory}: invalid episode metadata/chunks_size')
                shared.extend(directory / 'meta' / name for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'))
                remaining = None if task_limit is None else task_limit - len(tasks)
                for index in sorted(metadata)[:remaining]:
                    episode_id = f'{directory.name}_episode_{index:06d}'
                    paths = [m.source_path(directory, info['data_path'], index, chunks)]
                    paths += [m.source_path(directory, info['video_path'], index, chunks, key) for key in cameras]
                    tasks.append({'source': str(directory), 'index': index, 'episode_ids': [episode_id],
                                  'inputs': [str(p) for p in paths]})
    elif dataset == 'agibotworld2026':
        if source.is_dir():
            archives = []
            with progress('Scanning for archives', unit='archive') as bar:
                for path in source.rglob('*.tar.gz'):
                    archives.append(path)
                    bar.update()
            paths = sorted(archives)
        else:
            paths = [source]
        count = 0
        selected = paths if config['limit'] is None else paths[:config['limit']]
        with progress('Indexing archives', len(selected), unit='archive') as bar:
            for path in selected:
                _, _, entries = m.archive_index(path)
                tasks.append({'source': str(path), 'episode_ids': list(entries.values()), 'inputs': [str(path)]})
                count += len(entries)
                bar.update()
                if config['limit_episodes'] is not None and count >= config['limit_episodes']:
                    break
    remaining = config['limit_episodes']
    if remaining is not None:
        selected = []
        for task in tasks:
            task['episode_ids'] = task['episode_ids'][:remaining]
            selected.append(task)
            remaining -= len(task['episode_ids'])
            if remaining == 0:
                break
        tasks = selected
    if not tasks:
        raise ValueError(f'No tasks found for {dataset}: {source}')
    return tasks, sorted(set(shared))


class Adapter:
    def __init__(self, config):
        self.config = config
        self.m = module(config['dataset'])
        self.cache = {}
        self.annotation_cache = None

    def context(self, key, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def episode(self, task, destination):
        m, config = self.m, self.config
        dataset, source, params = config['dataset'], Path(task['source']), config['params']
        model = Path(config['model_dir']) if config['model_dir'] else None
        if dataset == 'realomin':
            transform = self.context('tool', lambda: m.tool_transform(params['tcp_offset'], params['tool_rpy']))
            return m.convert_episode({'path': source, 'instruction': task['instruction']}, destination,
                                     transform, params['gripper_closed'], params['gripper_open'])
        if dataset == 'agibotworld_beta':
            transforms = self.context('tools', lambda: tuple(
                m.tool_transform(params['tcp_offset'], params[f'{side}_tool_rpy']) for side in ('left', 'right')))
            path = Path(task['annotation_path'])
            if self.annotation_cache is None or self.annotation_cache[0] != path:
                self.annotation_cache = (path, m.load_annotations(path.parents[1], int(source.parent.parent.name)))
            annotation = self.annotation_cache[1][task['episode_index']]
            return m.convert_episode({'state': source, 'videos': {c: Path(p) for c, p in task['videos'].items()},
                                      **annotation}, destination, transforms,
                                     params['gripper_closed'], params['gripper_open'])
        if dataset in ('abc130k', 'molmoact2'):
            fk = self.context('fk', lambda: m.YamFK(model, **({'arm_local': True} if dataset == 'molmoact2' else {})))
            if dataset == 'abc130k':
                return m.convert_episode(source, destination, fk)
            if task.get('layout') == 'lerobot_v3':
                reader = self.context('lerobot_v3', lambda: m.DatasetReader(Path(config['source'])))
                return reader.convert_episode(task['index'], destination, fk)
            tasks, annotated = self.context('tasks', lambda: m.load_tasks(Path(config['source']) / 'source/meta'))
            return m.convert_episode(source, destination, tasks, annotated, fk)
        if dataset == 'hifi_umi':
            tasks = self.context('tasks', lambda: m.load_tasks(Path(config['source']) / 'source/tasks.parquet'))
            return m.convert_episode(source, destination, tasks, params['gripper_closed_rad'], params['gripper_open_rad'])
        info, metadata, tasks = self.context(str(source), lambda: (
            load_json(source / 'meta/info.json'), m.read_jsonl(source / 'meta/episodes.jsonl', 'episode_index'),
            {i: r['task'] for i, r in m.read_jsonl(source / 'meta/tasks.jsonl', 'task_index').items()}))
        kind, _ = m.validate_source(info)
        fk = self.context(kind, lambda: m.GalaxeaKinematics(model / f'galaxea_{kind}', kind, params['tcp_offsets'][kind]))
        index = task['index']
        return m.convert_episode(source, index, metadata[index], tasks, fk, info, destination)

    def process(self, task, pending, staging, store, failed):
        if self.config['dataset'] == 'agibotworld2026':
            return self.archive(task, pending, staging, store, failed)
        if self.config['dataset'] == 'hifi_umi' and task.get('layout') == 'lerobot_v3_shard':
            return self.shard(task, pending, staging, store, failed)
        episode_id = pending[0]
        destination = staging / episode_id
        from scripts.convert.source_policy import SourceDataError
        native_policy = self.config['params'].get('data_policy_version') == 1
        if native_policy and 'drop' in task:
            drop = task['drop']
            store.skip(episode_id, drop['reason'], detail=drop['detail'], reasons=drop.get('reasons'))
            return
        try:
            record = self.episode(task, destination)
        except SourceDataError as exc:
            if not native_policy:
                raise
            store.skip(episode_id, exc.reason, detail=str(exc), reasons=exc.reasons)
            return
        if record is None:
            store.skip(episode_id, 'Galaxea chassis commands do not meet the stationary selection rule')
        else:
            store.commit(episode_id, destination, record)

    def shard(self, task, pending, staging, store, failed):
        source = Path(task['source'])
        reader = self.m.ShardReader(source)
        indices = {self.m.shard_episode_id(source, i): i for i in reader.episodes}
        params = self.config['params']
        for episode_id in pending:
            with tempfile.TemporaryDirectory(prefix='episode-', dir=staging) as temporary:
                destination = Path(temporary) / episode_id
                try:
                    record = reader.convert_episode(indices[episode_id], destination,
                                                    params['gripper_closed_rad'], params['gripper_open_rad'])
                    store.commit(episode_id, destination, record)
                except Exception as exc:
                    failed(episode_id, exc)

    def archive(self, task, pending, staging, store, failed):
        fk = self.context('fk', lambda: self.m.G2FK(Path(self.config['model_dir']),
                                                  self.config['params']['tcp_offset']))
        self.m.convert_archive(Path(task['source']), staging, fk, episode_ids=pending,
                               on_episode=lambda record, directory: store.commit(record['episode_id'], directory, record),
                               on_error=failed)
