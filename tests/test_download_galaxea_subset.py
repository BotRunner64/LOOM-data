import io
import json
import tarfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from scripts.download import galaxea_subset as script


SHA = 'a' * 40


class ParseTests(unittest.TestCase):
    def test_archive_with_collection_suffix(self):
        parsed = script.parse_archive('lerobot/Turn_On_Off_The_Light_20250619_001.tar.gz', 42)
        self.assertEqual(parsed['task'], 'Turn_On_Off_The_Light_20250619_001')
        self.assertEqual(parsed['date'], '20250619')
        self.assertEqual(parsed['version'], '001')
        self.assertEqual(parsed['size'], 42)

    def test_archive_with_missing_underscore_and_without_suffix(self):
        self.assertEqual(
            script.parse_archive('lerobot/Egg_Placement20250703_002.tar.gz', 1)['task'],
            'Egg_Placement20250703_002',
        )
        bare = script.parse_archive('lerobot/Dispose_Of_Garbage_In_The_Trash_Can.tar.gz', 1)
        self.assertIsNone(bare['date'])
        self.assertIsNone(bare['version'])

    def test_non_archive_is_ignored(self):
        self.assertIsNone(script.parse_archive('lerobot/README.md', 1))


class SamplingTests(unittest.TestCase):
    def tasks(self):
        return [
            {'path': 'lerobot/b.tar.gz', 'size': 2},
            {'path': 'lerobot/a.tar.gz', 'size': 1},
            {'path': 'lerobot/c.tar.gz', 'size': 3},
        ]

    def test_smallest_orders_by_size(self):
        selected = script.sample_tasks(self.tasks(), 2, 42, 'smallest')
        self.assertEqual([item['path'] for item in selected], ['lerobot/a.tar.gz', 'lerobot/b.tar.gz'])

    def test_random_is_deterministic_and_seed_sensitive(self):
        first = script.sample_tasks(self.tasks(), 2, 42, 'random')
        self.assertEqual(first, script.sample_tasks(list(reversed(self.tasks())), 2, 42, 'random'))
        self.assertNotEqual(first, script.sample_tasks(self.tasks(), 2, 43, 'random'))


class ManifestTests(unittest.TestCase):
    def test_manifest_requires_safe_paths_and_pinned_version(self):
        manifest = {
            'schema_version': 1, 'repo_id': script.REPO_ID, 'revision': SHA,
            'files': [{'path': 'lerobot/a.tar.gz', 'size': 3}],
            'total_bytes': 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            script.write_json(path, manifest)
            self.assertEqual(script.load_manifest(path), manifest)
            for invalid in ('../outside', '/absolute', 'lerobot/../../outside'):
                manifest['files'][0]['path'] = invalid
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    script.load_manifest(path)
            manifest['revision'] = 'main'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'commit SHA'):
                script.load_manifest(path)


class DownloadTests(unittest.TestCase):
    @patch('huggingface_hub.hf_hub_download')
    def test_truncated_file_is_redownloaded(self, download):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'a.tar.gz'
            destination.write_bytes(b'x')

            def fetch(**kwargs):
                if kwargs.get('force_download'):
                    destination.write_bytes(b'xxx')
                return str(destination)

            download.side_effect = fetch
            script.download_file({'path': 'lerobot/a.tar.gz', 'size': 3}, SHA, Path(directory), 0)
            self.assertEqual(download.call_count, 2)
            for call in download.call_args_list:
                self.assertEqual(call.kwargs['revision'], SHA)
                self.assertEqual(call.kwargs['endpoint'], 'https://huggingface.co')
            self.assertTrue(download.call_args.kwargs['force_download'])


class ExtractTests(unittest.TestCase):
    def make_archive(self, directory, members):
        archive = Path(directory) / 'lerobot' / 'a.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'w:gz') as tar:
            for name, payload in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        return archive

    def test_extract_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_archive(directory, [('task/file.txt', b'hello')])
            script.extract_archive({'path': 'lerobot/a.tar.gz'}, Path(directory))
            self.assertEqual((Path(directory) / 'task' / 'file.txt').read_bytes(), b'hello')

    def test_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_archive(directory, [('../escape.txt', b'x')])
            with self.assertRaises((tarfile.TarError, ValueError)):
                script.extract_archive({'path': 'lerobot/a.tar.gz'}, Path(directory))


class RetryTests(unittest.TestCase):
    def error(self, status):
        response = httpx.Response(status, request=httpx.Request('GET', 'https://huggingface.co'))
        return httpx.HTTPStatusError('error', request=response.request, response=response)

    @patch.object(script.time, 'sleep')
    def test_authentication_failure_is_not_retried(self, sleep):
        operation = unittest.mock.Mock(side_effect=self.error(401))
        with self.assertRaises(httpx.HTTPStatusError):
            script.retry(operation, 3)
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
