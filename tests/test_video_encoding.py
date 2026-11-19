"""The training export must support short random seeks without losing synchronized rows."""
from fractions import Fraction
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from loom_data import sync, video


class VideoEncodingTests(unittest.TestCase):
    def test_short_gop_and_nearest_frame_alignment(self):
        # Repeating sources on a 30 Hz grid exercises offline queue backpressure too.
        grid = np.array([sync.frame_ns(i) for i in range(33)])
        times = grid[::2]
        selected = sync.nearest(times, grid)
        frames = (av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 12, np.uint8), format='rgb24')
                  for i in range(len(times)))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOOM_DATA_SERIAL_CODECS': '1'}):
            path = Path(tmp) / 'camera.mp4'
            result = video.encode_frames(path, frames, times, grid)
            self.assertEqual(result['size'], [640, 480])
            with av.open(str(path)) as reader:
                self.assertEqual(len(reader.streams), 1)
                stream = reader.streams.video[0]
                self.assertEqual((stream.codec_context.name, stream.format.name, stream.average_rate),
                                 ('h264', 'yuv420p', 30))
                self.assertEqual(stream.frames, len(grid))
                keys, values = [], []
                for index, frame in enumerate(reader.decode(stream)):
                    self.assertEqual(frame.pts * stream.time_base, Fraction(index, 30))
                    if frame.key_frame:
                        keys.append(index)
                    values.append(frame.to_ndarray(format='rgb24')[0, 0, 0])
            self.assertEqual(keys[0], 0)
            self.assertLessEqual(max(np.diff([*keys, len(grid)])), 2)
            np.testing.assert_allclose(values, selected * 12, atol=3, rtol=0)

    def test_partial_source_does_not_publish_a_truncated_video(self):
        frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format='rgb24')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'camera.mp4'
            with self.assertRaisesRegex(ValueError, 'Decoded 1 frames'):
                video.encode_frames(path, [frame], [0, sync.frame_ns(1)])
            self.assertEqual(list(path.parent.iterdir()), [])
