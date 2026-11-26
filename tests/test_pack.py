import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow.parquet as pq

from support import make_dataset
from loom_data import contract, pack


def decode(output, episode_index, frame, key):
    """One frame decoded as docs/contract/packed-contract.md section 4 describes, as rgb24."""
    info = json.loads((output / 'info.json').read_text())
    episode = pq.read_table(output / 'episodes.parquet').to_pylist()[episode_index]
    rows = [row for row in pq.read_table(output / 'frames.parquet').to_pylist() if row['episode_index'] == episode_index]
    first = frame
    while not rows[first][f'{key}.keyframe']:
        first -= 1
    data = pack.shard_path(output, episode['shard']).read_bytes()
    before = info['cameras'][:info['cameras'].index(key)]
    codec = av.CodecContext.create('h264', 'r')
    codec.extradata = episode[f'{key}.extradata']
    decoded = []
    for row in rows[first:frame + 1]:
        start = row['offset'] + sum(row[f'{other}.size'] for other in before)
        decoded += codec.decode(av.Packet(data[start:start + row[f'{key}.size']]))
    decoded += codec.decode(None)
    return decoded[-1].to_ndarray(format='rgb24')


def mp4_frames(dataset, episode, camera):
    path, _ = dataset.video(episode, camera)
    with av.open(str(path)) as reader:
        return [frame.to_ndarray(format='rgb24') for frame in reader.decode(video=0)]


class PackTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.export = self.directory / 'export/run/source/top+left_wrist_64x48'
        self.export.parent.mkdir(parents=True)
        self.dataset = make_dataset(self.export, {f'ep{i}': {'left_gripper': np.zeros(n)} for i, n in
                                                  enumerate((7, 12, 5))}, cameras=('left_wrist', 'top'))
        self.output = self.directory / 'packed/run/source/top+left_wrist_64x48'

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_every_frame_decodes_as_in_the_export(self):
        pack.pack(self.export, self.output, workers=1)
        info = json.loads((self.output / 'info.json').read_text())
        self.assertEqual(info['cameras'], ['observation.images.top', 'observation.images.left_wrist'])
        self.assertEqual((info['episodes'], info['frames']), (3, 24))
        self.assertFalse((self.output / pack.WORK).exists())
        for episode in self.dataset.episodes:
            for camera in self.dataset.cameras:
                expected = mp4_frames(self.dataset, episode, camera)
                for frame in range(episode['length']):
                    np.testing.assert_array_equal(
                        decode(self.output, episode['episode_index'], frame, contract.video_key(camera)), expected[frame])

    def test_shards_hold_whole_episodes_and_carry_checksums(self):
        with patch.object(pack, 'SHARD_BYTES', 1):
            pack.pack(self.export, self.output, workers=2)
        info = json.loads((self.output / 'info.json').read_text())
        episodes = pq.read_table(self.output / 'episodes.parquet').to_pylist()
        self.assertEqual([row['shard'] for row in episodes], [0, 1, 2])
        for shard, record in enumerate(info['shards']):
            data = pack.shard_path(self.output, shard).read_bytes()
            self.assertEqual(record, {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        frames = pq.read_table(self.output / 'frames.parquet').to_pylist()
        self.assertEqual([row['offset'] for row in frames if row['frame_index'] == 0], [0, 0, 0])
        np.testing.assert_array_equal(decode(self.output, 2, 4, 'observation.images.top'),
                                      mp4_frames(self.dataset, self.dataset.episodes[2], 'top')[4])

    def test_an_interrupted_run_resumes_without_repacking_committed_shards(self):
        original, packed = pack.pack_shard, []

        def fail_on_second(dataset, keys, episodes, shard, output, work):
            if shard == 1 and not packed.count(1):
                packed.append(1)
                raise RuntimeError('interrupted')
            packed.append(shard)
            original(dataset, keys, episodes, shard, output, work)

        with patch.object(pack, 'SHARD_BYTES', 1), patch.object(pack, 'pack_shard', fail_on_second):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                pack.pack(self.export, self.output, workers=1)
            self.assertFalse((self.output / 'info.json').exists())
            pack.pack(self.export, self.output, workers=1)
        self.assertEqual(packed, [0, 1, 1, 2])
        np.testing.assert_array_equal(decode(self.output, 1, 11, 'observation.images.left_wrist'),
                                      mp4_frames(self.dataset, self.dataset.episodes[1], 'left_wrist')[11])

    def test_a_changed_export_is_refused(self):
        pack.pack(self.export, self.output, workers=1)
        pack.pack(self.export, self.output, workers=1)  # up to date: nothing to do
        other = self.directory / 'other/run/source/top+left_wrist_64x48'
        other.parent.mkdir(parents=True)
        make_dataset(other, {'ep0': {'left_gripper': np.zeros(9)}}, cameras=('left_wrist', 'top'))
        with self.assertRaisesRegex(ValueError, 'different version'):
            pack.pack(other, self.output, workers=1)

    def test_workers_stop_once_another_machine_has_published(self):
        pack.pack(self.export, self.output, workers=1)
        self.assertFalse((self.output / pack.WORK).exists())
        pack.work(self.export, self.output, [[0], [1], [2]])  # its work directory is already gone

    def test_output_mirrors_the_export_tree(self):
        with patch.object(pack.paths, 'EXPORT', self.directory / 'export'), patch.object(pack.paths, 'PACKED', self.directory / 'packed'):
            self.assertEqual(pack.output_for(self.export.resolve()), self.output)
        self.assertEqual(pack.output_for(self.export.resolve(), self.directory / 'elsewhere'),
                         self.directory.resolve() / 'elsewhere/source/top+left_wrist_64x48')
        with self.assertRaisesRegex(ValueError, '--output'):
            pack.output_for(self.directory.resolve() / 'loose/top+left_wrist_64x48')


if __name__ == '__main__':
    unittest.main()
