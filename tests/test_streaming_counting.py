"""一次覆盖在线时间边界、训练窗口一致性和计数完整周期，不依赖GPU。"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitness_pose.counting import CountConfig, CycleCounter, ExerciseCounter, StableSideSelector, HeightRange, hip_height_signal
from fitness_pose.normalization import CausalPoseNormalizer, make_windows, normalize_pose
from fitness_pose.streaming import ActionFilter, OnlineWindowBuffer


def pose_sequence(total):
    points = np.full((total, 17, 2), 200.0)
    points[:, 5], points[:, 6] = [170, 100], [230, 100]
    points[:, 11], points[:, 12] = [180, 200], [220, 200]
    return points, np.ones((total, 17)), np.ones((total, 17), dtype=bool)


class StreamingCountingContracts(unittest.TestCase):
    def test_stable_side_requires_sustained_advantage(self):
        selector = StableSideSelector(CountConfig(stable_side=True))
        self.assertEqual(selector.select({"left": (.8, 170), "right": (.7, 160)}, 0), ("left", False))
        self.assertEqual(selector.select({"left": (.8, 170), "right": (.82, 160)}, .1), ("left", False))
        selector.select({"left": (.5, 170), "right": (.9, 160)}, .2)
        self.assertEqual(selector.select({"left": (.5, 170), "right": (.9, 160)}, .4), ("left", False))
        self.assertEqual(selector.select({"left": (.5, 170), "right": (.9, 160)}, .5), ("right", True))

    def test_height_signal_scale_translation_and_static_guard(self):
        points, scores, valid = pose_sequence(1)
        points[0, [15, 16]] = [[180, 400], [220, 400]]
        first = hip_height_signal(points[0], scores[0], valid[0])
        second = hip_height_signal(points[0] * 2 + [20, 30], scores[0], valid[0])
        np.testing.assert_allclose(first, second)
        tracker = HeightRange(CountConfig())
        for i in range(100):
            zone, _ = tracker.zone(2.0 + .001 * np.sin(i), i / 20)
            self.assertIsNone(zone)

    def test_height_calibrates_during_warmup_but_does_not_count_the_past(self):
        config = replace(CountConfig(), squat_signal="hip_height")
        counter = ExerciseCounter(config)
        for i in range(140):
            now = i / 20
            points, scores, valid = pose_sequence(1)
            hip_y = 200 + 50 * (1 - np.cos(np.pi * now))
            points[0, [11, 12], 1] = hip_y
            points[0, [5, 6], 1] = hip_y - 100
            points[0, [15, 16]] = [[180, 400], [220, 400]]
            action = None if i < 80 else "squat"
            counter.step(action, points[0], scores[0], valid[0], now, 0)
            if i <= 80:
                self.assertEqual(counter.counts["squat"], 0)
        self.assertEqual(counter.counts["squat"], 1)

    def test_new_track_cannot_reuse_short_gap_window(self):
        points, scores, valid = pose_sequence(220)
        times, segments = np.arange(220) / 20, np.zeros(220, dtype=int)
        segments[80], segments[81:] = -1, 1
        normalized = normalize_pose(points, scores, valid, times, segments)
        recovered = make_windows(normalized, times, segments, short_gap_s=.15)
        for indexes in recovered["source_index"]:
            self.assertEqual(len(set(segments[indexes])), 1)
            self.assertNotIn(-1, segments[indexes])

    def test_short_gap_uses_only_past_valid_sources_and_matches_batch(self):
        total = 220
        points, scores, valid = pose_sequence(total)
        times, segments = np.arange(total) / 20, np.zeros(total, dtype=int)
        normalized = normalize_pose(points, scores, valid, times, segments)
        normalized["geometry_valid"][80] = False
        normalized["valid"][80] = False
        normalized["features"][80] = 0
        segments[80] = -1
        strict = make_windows(normalized, times, segments)
        recovered = make_windows(normalized, times, segments, short_gap_s=.15)
        self.assertGreater(len(recovered["x"]), len(strict["x"]))
        self.assertNotIn(80, recovered["source_index"])
        self.assertTrue(np.all(times[recovered["source_index"]] <= recovered["time_grid_s"]))
        self.assertLessEqual(np.max(recovered["time_grid_s"] - times[recovered["source_index"]]), .15 + 1e-9)
        buffer, emitted = OnlineWindowBuffer(short_gap_s=.15), []
        for i, now in enumerate(times):
            frame = {k: normalized[k][i] for k in ("features", "valid", "geometry_valid")}
            emitted.extend(buffer.step(frame, float(now), int(segments[i]), i))
        np.testing.assert_array_equal([r["source_index"] for r in emitted], recovered["source_index"])
        for source in recovered["source_index"]:
            self.assertEqual(len(set(segments[source])), 1)
        normalized["geometry_valid"][80:100] = False
        long_gap = make_windows(normalized, times, segments, short_gap_s=.15)
        for source in long_gap["source_index"]:
            self.assertFalse(np.any(source < 80) and np.any(source >= 100))

    def test_online_windows_match_training_and_never_read_future(self):
        total = 260
        points, scores, valid = pose_sequence(total)
        times = np.arange(total) / 20 + np.where(np.arange(total) % 3 == 0, .01, 0)
        segments = np.zeros(total, dtype=int)
        segments[125] = -1
        segments[126:] = 1
        normalized = normalize_pose(points, scores, valid, times, segments)
        offline = make_windows(normalized, times, segments)
        normalizer, buffer, online = CausalPoseNormalizer(), OnlineWindowBuffer(), []
        for i, now in enumerate(times):
            frame = normalizer.step(points[i], scores[i], valid[i], float(now), int(segments[i]))
            emitted = buffer.step(frame, float(now), int(segments[i]), i)
            for item in emitted:
                self.assertTrue(np.all(times[item["source_index"]] <= item["time_grid_s"]))
                self.assertLessEqual(times[item["source_index"][-1]], now)
                self.assertEqual(len(set(segments[item["source_index"]])), 1)
            online.extend(emitted)
        self.assertEqual(len(online), len(offline["x"]))
        np.testing.assert_array_equal(np.array([r["source_index"] for r in online]), offline["source_index"])
        np.testing.assert_array_equal(np.array([r["x"] for r in online]), offline["x"])

    def test_unknown_filter_requires_confirmation_and_expires(self):
        action = ActionFilter()
        action.update([.9, .05, .05], 0)
        self.assertIsNone(action.action)
        action.update([.9, .05, .05], .5)
        self.assertEqual(action.action, 0)
        action.expire(1.3)
        self.assertIsNone(action.action)
        self.assertEqual(action.status, "stale")
        action.update([.34, .33, .33], 1.4)
        self.assertEqual(action.status, "uncertain")

    def test_cycles_require_both_endpoints_and_reject_jitter(self):
        config = CountConfig()
        cycle = CycleCounter(config)
        for i, zone in enumerate(["home"] * 20 + ["away"] * 20 + ["home"] * 20):
            cycle.step(zone, i / 20)
        self.assertEqual(cycle.count, 1)
        incomplete = CycleCounter(config)
        for i, zone in enumerate(["home"] * 20 + ["away", "middle"] * 20 + ["home"] * 20):
            incomplete.step(zone, i / 20)
        self.assertEqual(incomplete.count, 0)
        half = CycleCounter(config)
        for i, zone in enumerate(["home"] * 20 + ["away"] * 20):
            half.step(zone, i / 20)
        self.assertEqual(half.count, 0)
        half.reset()
        for i in range(20):
            half.step("home", 5 + i / 20)
        self.assertEqual(half.count, 0)

    def test_three_exercise_signals_complete_one_cycle(self):
        for action in ("squat", "push_up", "jump_jack"):
            counter = ExerciseCounter()
            for i, home in enumerate([True] * 20 + [False] * 20 + [True] * 20):
                points = np.zeros((17, 2), dtype=float)
                if action == "jump_jack":
                    points[[5, 6]] = [[180, 100], [220, 100]]
                    points[[11, 12]] = [[180, 200], [220, 200]]
                    points[[9, 10]] = [[175, 250], [225, 250]] if home else [[120, 30], [280, 30]]
                    points[[15, 16]] = [[180, 400], [220, 400]] if home else [[100, 400], [300, 400]]
                else:
                    a, b, c = (11, 13, 15) if action == "squat" else (5, 7, 9)
                    radians = np.deg2rad(170 if home else 80)
                    points[a], points[b], points[c] = [0, 0], [0, 100], [100 * np.sin(radians), 100 - 100 * np.cos(radians)]
                counter.step(action, points, np.ones(17), np.ones(17, dtype=bool), i / 20, 0)
            self.assertEqual(counter.counts[action], 1, action)

    def test_tracking_change_does_not_join_a_half_cycle(self):
        counter = ExerciseCounter(replace(CountConfig(), signal_tau_s=.01))
        for i, (home, segment) in enumerate([(True, 0)] * 20 + [(False, 0)] * 20 + [(True, 1)] * 20):
            points = np.zeros((17, 2), dtype=float)
            radians = np.deg2rad(170 if home else 80)
            points[11], points[13], points[15] = [0, 0], [0, 100], [100 * np.sin(radians), 100 - 100 * np.cos(radians)]
            counter.step("squat", points, np.ones(17), np.ones(17, dtype=bool), i / 20, segment)
        self.assertEqual(counter.counts["squat"], 0)


if __name__ == "__main__":
    unittest.main()
