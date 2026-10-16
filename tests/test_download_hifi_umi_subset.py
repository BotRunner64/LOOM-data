import json
from pathlib import Path
import tempfile
import unittest

from huggingface_hub.hf_api import RepoFile

from scripts.download import hifi_umi_subset as script


SHA = 'a' * 40
SHARD = 'chunk-0000/part-0000'


def file(path, size=100):
    return RepoFile(path=path, size=size, oid='c' * 40)


def episode(index, length=100):
    videos = {
        key: {'path': f'{SHARD}/videos/{key}/chunk-000/file-000.mp4', 'from_timestamp': 0.0,
              'to_timestamp': length / 25.0}
        for key in script.VIDEO_KEYS
    }
    return {
        'episode_index': index, 'length': length, 'tasks': ['task'],
        'dataset_from_index': index * length, 'dataset_to_index': (index + 1) * length,
        'videos': videos,
    }


def manifest(**overrides):
    value = {
        'schema_version': 1, 'repo_id': script.REPO_ID, 'revision': SHA, 'shard': SHARD,
        'files': [{'role': 'info', 'path': f'{SHARD}/meta/info.json', 'size': 3}],
        'total_bytes': 3, 'episodes': [episode(0)],
    }
    value.update(overrides)
    return value


class ShardTests(unittest.TestCase):
    def test_shard_name_only_matches_shard_paths(self):
        self.assertEqual(script.shard_name(f'{SHARD}/data/chunk-000/file-000.parquet'), SHARD)
        self.assertIsNone(script.shard_name('README.md'))
        self.assertIsNone(script.shard_name('meta/info.json'))

    def test_collect_shards_orders_by_size(self):
        shards = script.collect_shards([
            file('chunk-0001/part-0000/meta/tasks.parquet', 5),
            file('chunk-0000/part-0000/meta/tasks.parquet', 3),
            file('chunk-0001/part-0000/meta/info.json', 4),
            file('README.md', 1),
        ])
        self.assertEqual([item['shard'] for item in shards], ['chunk-0000/part-0000', 'chunk-0001/part-0000'])
        self.assertEqual(shards[1]['total_bytes'], 9)

    def test_pick_shard_defaults_to_smallest_and_validates_name(self):
        shards = [{'shard': 'chunk-0000/part-0000', 'total_bytes': 3, 'file_count': 1}]
        self.assertEqual(script.pick_shard(shards, None), shards[0])
        self.assertEqual(script.pick_shard(shards, 'chunk-0000/part-0000'), shards[0])
        with self.assertRaises(ValueError):
            script.pick_shard(shards, 'chunk-9999/part-0000')


class ClassifyTests(unittest.TestCase):
    def test_classify_requires_every_role_and_camera(self):
        siblings = [
            file(f'{SHARD}/meta/info.json'), file(f'{SHARD}/meta/modality.json'),
            file(f'{SHARD}/meta/stats.json'), file(f'{SHARD}/meta/tasks.parquet'),
            file(f'{SHARD}/meta/episodes/chunk-000/file-000.parquet'),
            file(f'{SHARD}/data/chunk-000/file-000.parquet'),
        ] + [file(f'{SHARD}/videos/{key}/chunk-000/file-000.mp4') for key in script.VIDEO_KEYS]
        classified = script.classify_shard_files(siblings, SHARD)
        self.assertEqual(classified['data']['path'], f'{SHARD}/data/chunk-000/file-000.parquet')
        self.assertEqual(sorted(classified['videos']), sorted(script.VIDEO_KEYS))

    def test_classify_rejects_missing_camera(self):
        siblings = [
            file(f'{SHARD}/meta/info.json'), file(f'{SHARD}/meta/modality.json'),
            file(f'{SHARD}/meta/stats.json'), file(f'{SHARD}/meta/tasks.parquet'),
            file(f'{SHARD}/meta/episodes/chunk-000/file-000.parquet'),
            file(f'{SHARD}/data/chunk-000/file-000.parquet'),
        ] + [file(f'{SHARD}/videos/{key}/chunk-000/file-000.mp4') for key in script.VIDEO_KEYS[:-1]]
        with self.assertRaises(ValueError):
            script.classify_shard_files(siblings, SHARD)


class SamplingTests(unittest.TestCase):
    def test_smallest_orders_by_length_then_index(self):
        episodes = [episode(2, 50), episode(0, 10), episode(1, 50)]
        selected = script.sample_episodes(episodes, 2, 42, 'smallest')
        self.assertEqual([item['episode_index'] for item in selected], [0, 1])

    def test_random_is_deterministic_and_seed_sensitive(self):
        episodes = [episode(index, 10) for index in range(50)]
        first = script.sample_episodes(episodes, 5, 42, 'random')
        self.assertEqual(first, script.sample_episodes(list(reversed(episodes)), 5, 42, 'random'))
        self.assertEqual(len({item['episode_index'] for item in first}), 5)
        self.assertNotEqual(first, script.sample_episodes(episodes, 5, 43, 'random'))


class FrameTests(unittest.TestCase):
    def test_frame_in_segment_is_half_open(self):
        self.assertTrue(script.frame_in_segment(10.0, 10.0, 20.0))
        self.assertTrue(script.frame_in_segment(19.96, 10.0, 20.0))
        self.assertFalse(script.frame_in_segment(20.0, 10.0, 20.0))
        self.assertFalse(script.frame_in_segment(9.96, 10.0, 20.0))
        self.assertFalse(script.frame_in_segment(None, 10.0, 20.0))


class ManifestTests(unittest.TestCase):
    def test_manifest_round_trip(self):
        value = manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            script.write_json(path, value)
            self.assertEqual(script.load_manifest(path), value)

    def test_manifest_rejects_bad_paths_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for invalid in ('../outside', '/absolute', f'{SHARD}/../../outside'):
                value = manifest(files=[{'role': 'info', 'path': invalid, 'size': 3}])
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    script.load_manifest(path)
            path.write_text(json.dumps(manifest(revision='main')))
            with self.assertRaisesRegex(ValueError, 'commit SHA'):
                script.load_manifest(path)

    def test_manifest_rejects_inconsistent_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            broken = episode(0)
            broken['dataset_to_index'] = broken['dataset_from_index'] + 1
            path.write_text(json.dumps(manifest(episodes=[broken])))
            with self.assertRaises(ValueError):
                script.load_manifest(path)
            missing_camera = episode(0)
            del missing_camera['videos'][script.VIDEO_KEYS[0]]
            path.write_text(json.dumps(manifest(episodes=[missing_camera])))
            with self.assertRaises(ValueError):
                script.load_manifest(path)

    def test_manifest_total_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(manifest(total_bytes=999)))
            with self.assertRaises(ValueError):
                script.load_manifest(path)


if __name__ == '__main__':
    unittest.main()
