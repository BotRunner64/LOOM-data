import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.download import agibotworld_beta_subset as script


def make_tar(path, members):
    with tarfile.open(path, "w:") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


class BetaSubsetTests(unittest.TestCase):
    def test_sparse_archive_ranges_are_matched_by_actual_episode_and_task(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.tar"
            video = Path(directory) / "video.tar"
            make_tar(state, [("676/879812/proprio_stats.h5", b"state"),
                             ("694/879910/proprio_stats.h5", b"other task")])
            make_tar(video, [("879791/videos/head_color.mp4", b"first"),
                             ("879910/videos/head_color.mp4", b"last")])
            # Both video IDs bound the state ID, but neither is an exact match.
            self.assertFalse(set(script.state_members(state)) & set(script.video_members(video, 676)))

    def test_only_selected_rgb_and_state_are_extracted_and_truncation_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "video.tar"
            output = Path(directory) / "raw"
            make_tar(archive, [("123/videos/head_color.mp4", b"video"),
                               ("123/depth/depth.png", b"depth"),
                               ("124/videos/head_color.mp4", b"other")])
            members = script.video_members(archive, 676)[(676, 123)]
            script.extract_members(archive, members, output)
            destination = output / "observations/676/123/videos/head_color.mp4"
            self.assertEqual(destination.read_bytes(), b"video")
            self.assertEqual(len(list(output.rglob("*mp4"))), 1)
            self.assertFalse(list(output.rglob("*png")))
            destination.write_bytes(b"v")
            script.extract_members(archive, members, output)
            self.assertEqual(destination.read_bytes(), b"video")

    def test_member_links_and_traversal_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "video.tar"
            with tarfile.open(archive, "w:") as tar:
                info = tarfile.TarInfo("123/videos/head.mp4")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                tar.addfile(info)
            member = {"member": info.name, "size": 0, "destination": "head.mp4"}
            with self.assertRaisesRegex(ValueError, "Unexpected archive member"):
                script.extract_members(archive, [member], Path(directory) / "raw")
            make_tar(archive, [("123/videos/head.mp4", b"v")])
            member.update(size=1, destination="../outside")
            with self.assertRaisesRegex(ValueError, "Invalid relative path"):
                script.extract_members(archive, [member], Path(directory) / "raw")


if __name__ == "__main__":
    unittest.main()
