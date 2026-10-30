"""只检查在线重置、最新帧及界面数据边界，不打开真实相机。"""

from dataclasses import asdict
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from online_pipeline import OnlineActionPipeline
from frame_source import LatestFrameCapture, VideoFrames
from live_camera import draw_panel
from fitness_pose.normalization import GeometryConfig
from fitness_pose.streaming import ActionFilterConfig
from fitness_pose.counting import CountConfig


def config():
    return {"preprocessing": {"geometry": asdict(GeometryConfig()), "windows": {
                "window_size": 48, "stride": 10, "fps": 20., "max_age_s": .15,
                "min_valid_ratio": .7, "min_joints_per_frame": 6, "short_gap_s": .15}},
            "filter": asdict(ActionFilterConfig()), "counter": asdict(CountConfig())}


class FakeCapture:
    def __init__(self, video=False):
        self.video, self.index, self.released = video, 0, False

    def isOpened(self):
        return True

    def set(self, key, value):
        return True

    def get(self, key):
        if key == cv2.CAP_PROP_FPS:
            return 30 if self.video else float("nan")
        if key == cv2.CAP_PROP_FRAME_COUNT:
            return 3
        return 0

    def getBackendName(self):
        return "FAKE"

    def read(self):
        if self.video and self.index >= 3:
            return False, None
        time.sleep(.001)
        self.index += 1
        return True, np.full((2, 3, 3), self.index % 255, dtype=np.uint8)

    def release(self):
        self.released = True


class OnlineCameraContracts(unittest.TestCase):
    def test_reset_missing_and_new_track_do_not_reuse_old_window(self):
        pipeline = OnlineActionPipeline(config(), lambda x: np.array([[9., 0., 0.]], dtype=np.float32))
        points = np.full((17, 2), 200.)
        points[[5, 6]], points[[11, 12]] = [[170, 100], [230, 100]], [[180, 200], [220, 200]]
        scores, valid = np.ones(17), np.ones(17, dtype=bool)
        for index in range(100):
            item = pipeline.step(points, scores, valid, index / 20, 0, index)
        self.assertEqual(item["timeline"]["action"], "squat")
        missing = pipeline.step(np.full((17, 2), np.nan), np.zeros(17), np.zeros(17, dtype=bool), 5., -1, 100)
        self.assertIsNone(missing["timeline"]["action"])
        self.assertEqual(missing["timeline"]["count_status"], "paused")
        resumed = pipeline.step(points, scores, valid, 5.05, 1, 101)
        self.assertIsNone(resumed["timeline"]["action"])
        self.assertEqual(resumed["predictions"], [])
        pipeline.counter.counters["squat"].count = 4
        pipeline.reset()
        self.assertEqual(sum(pipeline.counter.counts.values()), 0)
        self.assertIsNone(pipeline.buffer.anchor)
        self.assertIsNone(pipeline.filter.last_prediction)

    def test_latest_frame_capture_skips_backlog_and_releases(self):
        fake = FakeCapture()
        with patch.object(cv2, "VideoCapture", return_value=fake):
            capture = LatestFrameCapture(0, "DSHOW", 640, 480, 30).start()
            try:
                first = capture.get_after(-1, timeout=1)
                time.sleep(.03)
                latest = capture.get_after(first.index, timeout=1)
                self.assertGreater(latest.index, first.index + 1)
                self.assertGreater(latest.timestamp_s, first.timestamp_s)
                self.assertIsNone(capture.metadata["reported_fps"])
            finally:
                self.assertTrue(capture.close())
            self.assertTrue(fake.released)

    def test_video_timestamp_fallback_is_monotonic(self):
        fake = FakeCapture(video=True)
        with patch.object(cv2, "VideoCapture", return_value=fake):
            source = VideoFrames(Path("synthetic.mp4"))
            try:
                frames = [source.get_after(-1) for _ in range(3)]
                np.testing.assert_allclose([f.timestamp_s for f in frames], [0., 1 / 30, 2 / 30])
                self.assertIsNone(source.get_after(2))
                self.assertTrue(source.finished)
            finally:
                source.close()

    def test_panel_preserves_image_and_accepts_missing_pose(self):
        image = np.full((480, 640, 3), 100, dtype=np.uint8)
        rendered = draw_panel(image, None, None, False, 0, 0, None)
        self.assertEqual(rendered.shape, (612, 640, 3))
        np.testing.assert_array_equal(rendered[132:], image)
        np.testing.assert_array_equal(image, np.full_like(image, 100))


if __name__ == "__main__":
    unittest.main()
