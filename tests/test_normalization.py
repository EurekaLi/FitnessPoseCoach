"""检查归一化的几何不变性、因果性与缺失/分段约束。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitness_pose.normalization import make_windows, normalize_pose


def skeletons(total=100):
    points = np.zeros((total, 17, 2), dtype=float)
    points[:] = [200, 200]
    points[:, 5], points[:, 6] = [170, 100], [230, 100]
    points[:, 11], points[:, 12] = [180, 200], [220, 200]
    return points, np.ones((total, 17)), np.ones((total, 17), dtype=bool), np.arange(total) / 20, np.zeros(total, dtype=int)


class NormalizationContracts(unittest.TestCase):
    def test_translation_and_uniform_scale_do_not_change_local_pose(self):
        points, scores, mask, times, segments = skeletons()
        first = normalize_pose(points, scores, mask, times, segments)
        second = normalize_pose(points * 2 + [500, 300], scores, mask, times, segments)
        np.testing.assert_allclose(first["features"], second["features"], atol=1e-6)

    def test_future_change_does_not_rewrite_past_features(self):
        points, scores, mask, times, segments = skeletons()
        first = normalize_pose(points, scores, mask, times, segments)
        points[60:, :, 1] *= 3
        second = normalize_pose(points, scores, mask, times, segments)
        np.testing.assert_array_equal(first["features"][:60], second["features"][:60])

    def test_missing_joint_keeps_mask_and_zero_confidence(self):
        points, scores, mask, times, segments = skeletons()
        points[2, 9] = np.nan
        result = normalize_pose(points, scores, mask, times, segments)
        self.assertFalse(result["valid"][2, 9])
        np.testing.assert_array_equal(result["features"][2, 9], [0, 0, 0])
        self.assertTrue(np.isfinite(result["features"]).all())

    def test_windows_never_mix_tracks_or_read_a_future_sample(self):
        points, scores, mask, times, segments = skeletons(160)
        segments[80:] = 1
        times = times + np.where(np.arange(160) % 3 == 0, 0.01, 0)
        normalized = normalize_pose(points, scores, mask, times, segments)
        windows = make_windows(normalized, times, segments)
        self.assertGreater(len(windows["x"]), 0)
        for index in windows["source_index"]:
            self.assertEqual(len(np.unique(segments[index])), 1)
        self.assertTrue(np.all(times[windows["source_index"]] <= windows["time_grid_s"]))
        np.testing.assert_array_equal(windows["x"].shape[1:], [3, 48, 17])


if __name__ == "__main__":
    unittest.main()
