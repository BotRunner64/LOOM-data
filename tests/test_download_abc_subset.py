import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from huggingface_hub.hf_api import RepoFile, RepoFolder

from scripts.download import abc_subset as script


SHA = 'a' * 40
TASK = 'data/train/fold_towels'


def folder(path):
    return RepoFolder(path=path, oid='b' * 40)


def file(path, size=100):
    return RepoFile(path=path, size=size, oid='c' * 40)


class SelectionTests(unittest.TestCase):
    def test_sampling_is_independent_of_api_order_and_other_tasks(self):
        episodes = [f'episode_{index}' for index in range(100)]
        selected = script.sample_episodes(episodes, 30, 42, 'fold')
        script.sample_episodes(episodes, 30, 42, 'unrelated_task')
        self.assertEqual(selected, script.sample_episodes(list(reversed(episodes)), 30, 42, 'fold'))
        self.assertEqual(len(set(selected)), 30)
        self.assertNotEqual(selected, script.sample_episodes(episodes, 30, 43, 'fold'))

    def test_small_task_keeps_all_and_annotation_is_optional(self):
        episode_a, episode_b = f'{TASK}/episode_a', f'{TASK}/episode_b'
        api = Mock()
        api.list_repo_tree.return_value = [
            folder(episode_b), file(f'{episode_b}/episode.mcap'),
            folder(episode_a), file(f'{episode_a}/episode.mcap'),
            file(f'{episode_a}/annotation.mcap', 10),
            file(f'{TASK}/unrelated.txt'),
        ]
        selected, files = script.select_task(api, SHA, TASK, 30, 42, 0)
        self.assertEqual(selected['selected_episodes'], [episode_a, episode_b])
        self.assertEqual(selected['available_episodes'], 2)
        self.assertEqual(len(files), 3)
        self.assertEqual(sum(entry['size'] for entry in files), 210)
        self.assertEqual(api.list_repo_tree.call_args.kwargs['revision'], SHA)

    def test_unselected_episodes_are_not_in_file_manifest(self):
        api = Mock()
        api.list_repo_tree.return_value = [
            entry for index in range(10)
            for entry in (folder(f'{TASK}/episode_{index}'), file(f'{TASK}/episode_{index}/episode.mcap'))
        ]
        selected, files = script.select_task(api, SHA, TASK, 2, 42, 0)
        self.assertEqual(len(files), 2)
        self.assertEqual(
            {str(Path(entry['path']).parent) for entry in files},
            set(selected['selected_episodes']),
        )

    def test_missing_episode_payload_is_reported(self):
        api = Mock()
        api.list_repo_tree.return_value = [folder(f'{TASK}/episode_a')]
        with self.assertRaisesRegex(ValueError, 'missing episode.mcap'):
            script.select_task(api, SHA, TASK, 30, 42, 0)


class DownloadTests(unittest.TestCase):
    @patch('huggingface_hub.hf_hub_download')
    def test_download_uses_pinned_revision_and_repairs_truncated_file(self, download):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'episode.mcap'
            destination.write_bytes(b'x')

            def fetch(**kwargs):
                if kwargs.get('force_download'):
                    destination.write_bytes(b'xxx')
                return str(destination)

            download.side_effect = fetch
            script.download_file({'path': f'{TASK}/episode_a/episode.mcap', 'size': 3}, SHA, Path(directory), 0)
            self.assertEqual(download.call_count, 2)
            for call in download.call_args_list:
                self.assertEqual(call.kwargs['revision'], SHA)
                self.assertEqual(call.kwargs['endpoint'], 'https://huggingface.co')
            self.assertTrue(download.call_args.kwargs['force_download'])

    @patch('huggingface_hub.hf_hub_download')
    def test_valid_download_does_not_force_redownload(self, download):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'episode.mcap'
            destination.write_bytes(b'xxx')
            download.return_value = str(destination)
            script.download_file({'path': 'episode.mcap', 'size': 3}, SHA, Path(directory), 0)
            self.assertEqual(download.call_count, 1)
            self.assertNotIn('force_download', download.call_args.kwargs)

    def test_manifest_requires_safe_paths_and_pinned_version(self):
        manifest = {
            'schema_version': 1, 'repo_id': script.REPO_ID, 'revision': SHA,
            'files': [{'path': 'data/train/task/episode_a/episode.mcap', 'size': 3}],
            'total_bytes': 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            script.write_json(path, manifest)
            self.assertEqual(script.load_manifest(path), manifest)
            for invalid in ('../outside', '/absolute', 'data/../../outside'):
                manifest['files'][0]['path'] = invalid
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    script.load_manifest(path)
            manifest['revision'] = 'main'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'commit SHA'):
                script.load_manifest(path)


class RetryTests(unittest.TestCase):
    def error(self, status):
        response = httpx.Response(status, request=httpx.Request('GET', 'https://huggingface.co'))
        return httpx.HTTPStatusError('error', request=response.request, response=response)

    @patch.object(script.time, 'sleep')
    def test_transient_failure_is_retried(self, sleep):
        operation = Mock(side_effect=[self.error(503), 'ok'])
        self.assertEqual(script.retry(operation, 2), 'ok')
        self.assertEqual(operation.call_count, 2)

    @patch.object(script.time, 'sleep')
    def test_authentication_failure_is_not_retried(self, sleep):
        operation = Mock(side_effect=self.error(401))
        with self.assertRaises(httpx.HTTPStatusError):
            script.retry(operation, 3)
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
