from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download.subset_common import download_files, load_manifest, write_json
from scripts.download.realomin_subset import REPO_ID


class SharedSubsetTests(unittest.TestCase):
    def test_download_pins_revision_and_repairs_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "episode.mcap"
            destination.write_bytes(b"x")

            def fetch(**kwargs):
                if kwargs["force_download"]:
                    destination.write_bytes(b"abc")
                return str(destination)

            manifest = {"repo_id": REPO_ID, "revision": "a" * 40,
                        "files": [{"path": "episode.mcap", "size": 3}]}
            with patch("huggingface_hub.hf_hub_download", side_effect=fetch) as download:
                download_files(manifest, Path(directory), 1, 0)
            self.assertEqual(download.call_count, 2)
            for call in download.call_args_list:
                self.assertEqual(call.kwargs["revision"], "a" * 40)
                self.assertEqual(call.kwargs["endpoint"], "https://huggingface.co")
            self.assertEqual(destination.read_bytes(), b"abc")

    def test_manifest_rejects_unpinned_revisions_and_unsafe_paths(self):
        manifest = {"schema_version": 1, "repo_id": REPO_ID, "revision": "a" * 40,
                    "files": [{"path": "episode.mcap", "size": 3}], "total_bytes": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.json"
            write_json(path, manifest)
            self.assertEqual(load_manifest(path, REPO_ID), manifest)
            manifest["revision"] = "main"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                load_manifest(path, REPO_ID)
            manifest["revision"] = "a" * 40
            for name in ("../outside", "/absolute", "x/../outside", "a\\b"):
                manifest["files"][0]["path"] = name
                write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, "Invalid relative path"):
                    load_manifest(path, REPO_ID)


if __name__ == "__main__":
    unittest.main()
