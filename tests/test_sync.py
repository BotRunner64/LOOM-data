import unittest

import numpy as np

from loom_data import sync
from loom_data.sources.policy import SourceDataError

NS = 1_000_000_000


def rotation_z(degrees):
    a = np.deg2rad(degrees)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return r.T.reshape(-1)  # contract column-major


class GridTests(unittest.TestCase):
    def test_frame_times_are_k_over_30_without_drift(self):
        self.assertEqual([sync.frame_ns(k) for k in (0, 1, 2, 3, 30, 90000)],
                         [0, 33_333_333, 66_666_667, 100_000_000, NS, 3000 * NS])

    def test_grid_covers_only_the_interval_every_stream_has(self):
        grid = sync.frame_grid([[0, NS], [NS // 10, 2 * NS], [0, NS // 2]])
        self.assertEqual(grid[0], NS // 10)
        self.assertLessEqual(grid[-1], NS // 2)
        self.assertGreater(grid[-1] + NS // 30, NS // 2)
        np.testing.assert_array_equal(grid - grid[0], [sync.frame_ns(k) for k in range(len(grid))])

    def test_streams_sampled_exactly_on_frame_times_keep_every_frame(self):
        for count in range(2, 40):
            times = [sync.frame_ns(k) + 7 for k in range(count)]
            np.testing.assert_array_equal(sync.frame_grid([times]), times)
            if count > 2:
                self.assertEqual(len(sync.frame_grid([times[:-1] + [times[-1] - 1]])), count - 1)

    def test_streams_without_two_shared_frames_are_dropped(self):
        for streams in ([[0, NS], [2 * NS, 3 * NS]], [[0, NS], [NS, NS + 1]]):
            with self.assertRaises(SourceDataError) as caught:
                sync.frame_grid(streams)
            self.assertEqual(caught.exception.reason, 'no_common_interval')
        for bad in ([], [3, 2]):
            with self.assertRaisesRegex(ValueError, 'Empty or unordered'):
                sync.frame_grid([bad])

    def test_frame_table_passes_through_only_at_a_nominal_30_fps(self):
        rows = [0, 33_300_000, 70_000_000, 103_400_000, 130_000_000]  # jittery, median interval 1/30 s
        self.assertIsNone(sync.table_grid(rows))
        self.assertIsNone(sync.table_grid(rows, 30))
        self.assertIsNone(sync.table_grid([0, 2 * NS, 5 * NS], 30))  # declared rate wins over row times
        grid = sync.table_grid([k * NS // 15 for k in range(4)])
        self.assertEqual(len(grid), 7)
        self.assertEqual(len(sync.table_grid(rows, 15)), 4)
        self.assertEqual(len(sync.table_grid([0, 40_000_000, 80_000_000, 120_000_000])), 4)  # 25 fps
        with self.assertRaises(SourceDataError):
            sync.table_grid([5, 5])


class ResampleTests(unittest.TestCase):
    def test_scalars_interpolate_linearly_and_keep_exact_samples(self):
        times = [0, 50_000_000, 100_000_000]
        grid = np.array([0, 25_000_000, 50_000_000, 100_000_000])
        np.testing.assert_allclose(sync.resample_scalar(times, [0., 1., 0.], grid), [0, .5, 1, 0])

    def test_poses_interpolate_position_linearly_and_rotation_by_slerp(self):
        poses = np.array([[0, 0, 0, *rotation_z(0)], [1, 2, 4, *rotation_z(90)]], dtype=float)
        result = sync.resample_poses([0, 90_000_000], poses, np.array([0, 30_000_000, 90_000_000]))
        np.testing.assert_allclose(result[0], poses[0], atol=1e-12)
        np.testing.assert_allclose(result[2], poses[1], atol=1e-12)
        np.testing.assert_allclose(result[1], [1 / 3, 2 / 3, 4 / 3, *rotation_z(30)], atol=1e-12)
        rotation = result[1, 3:].reshape(3, 3).T
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_slerp_takes_the_short_way_round(self):
        poses = np.array([[0, 0, 0, *rotation_z(170)], [0, 0, 0, *rotation_z(-170)]], dtype=float)
        result = sync.resample_poses([0, 80_000_000], poses, np.array([40_000_000]))
        np.testing.assert_allclose(result[0, 3:], rotation_z(180), atol=1e-9)

    def test_a_state_gap_over_the_limit_is_dropped_unless_a_sample_sits_on_the_frame(self):
        wide = sync.STATE_GAP_NS + 50_000_000
        times, values = [0, wide], [0., 1.]
        still = np.tile([0, 0, 0, *rotation_z(0)], (2, 1))
        np.testing.assert_allclose(sync.resample_scalar(times, values, np.array([0, wide])), [0, 1])
        for function, data in ((sync.resample_scalar, values), (sync.resample_poses, still)):
            with self.assertRaises(SourceDataError) as caught:
                function(times, data, np.array([wide // 2]))
            self.assertEqual(caught.exception.reason, 'sampling_gap')
            self.assertIn(f'{sync.STATE_GAP_NS // 1_000_000} ms', str(caught.exception))
            # Samples exactly the limit apart are still interpolated across.
            function([0, sync.STATE_GAP_NS], data, np.array([sync.STATE_GAP_NS // 2]))
        with self.assertRaisesRegex(ValueError, 'outside the sampled interval'):
            sync.resample_scalar(times, values, np.array([wide + 50_000_000]))

    def test_a_125_ms_state_gap_is_interpolated_across(self):
        times = [0, 125_000_000]                            # a dropout seen in real recordings
        grid = np.array([sync.frame_ns(k) for k in range(4)])
        np.testing.assert_allclose(sync.resample_scalar(times, [0., 1.], grid), grid / 125_000_000, atol=1e-12)
        poses = np.array([[0, 0, 0, *rotation_z(0)], [1, 2, 4, *rotation_z(90)]], dtype=float)
        result = sync.resample_poses(times, poses, grid)
        for row, weight in zip(result, grid / 125_000_000):
            np.testing.assert_allclose(row, [weight, 2 * weight, 4 * weight, *rotation_z(90 * weight)], atol=1e-12)

    def test_rgb_takes_the_nearest_frame_repeating_and_dropping(self):
        slow = [k * NS // 15 for k in range(3)]             # 15 fps: every frame used twice
        grid = np.array([sync.frame_ns(k) for k in range(5)])
        np.testing.assert_array_equal(sync.nearest(slow, grid), [0, 0, 1, 1, 2])
        fast = [k * NS // 60 for k in range(9)]             # 60 fps: every other frame dropped
        np.testing.assert_array_equal(sync.nearest(fast, grid), [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(sync.nearest([0, 10, 20], np.array([5])), [0])  # ties take the earlier
        # A frame time exactly the limit from both neighbours still takes one; any farther is a gap.
        limit = sync.FRAME_GAP_NS
        np.testing.assert_array_equal(sync.nearest([0, 2 * limit], np.array([limit])), [0])
        with self.assertRaises(SourceDataError) as caught:
            sync.nearest([0, 2 * limit + 4_000_000], np.array([limit + 2_000_000]))
        self.assertEqual(caught.exception.reason, 'sampling_gap')
        self.assertIn(f'{limit // 1_000_000} ms', str(caught.exception))

    def test_a_short_camera_dropout_repeats_the_nearest_frames(self):
        # A 30 fps camera loses frames 4..8 (167 ms): source index 3 is frame 3, index 4 is frame 9.
        source = [sync.frame_ns(k) for k in (0, 1, 2, 3, 9, 10, 11, 12)]
        grid = np.array([sync.frame_ns(k) for k in range(13)])
        # The frame before the dropout covers its first half and the midpoint, the frame after the rest.
        np.testing.assert_array_equal(sync.nearest(source, grid), [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 7])
        # The same dropout seen from frame times offset by a third of a frame interval.
        np.testing.assert_array_equal(sync.nearest(source, grid[:-1] + 11_111_111),
                                      [0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
