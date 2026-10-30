"""用临时文件和合成骨架检查测试入口，不读取真实预留测试视频。"""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import independent_test as heldout
import replay_action
from fitness_pose.counting import CountConfig
from fitness_pose.normalization import GeometryConfig
from fitness_pose.streaming import ActionFilterConfig


class ArrayTensor:
    """只替代本测试用到的张量API；真实模型推理仍由现有PyTorch入口负责。"""

    def __init__(self, data):
        self.data = np.asarray(data)

    def unsqueeze(self, axis):
        return ArrayTensor(np.expand_dims(self.data, axis))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        return ArrayTensor(self.data[key])


class IndependentTestContracts(unittest.TestCase):
    def test_source_split_not_official_folder_and_unknown_count(self):
        original = {"sample_id": "train/example", "official_split": "train", "action": "squat",
                    "class_id": 0, "count": None}
        assigned = {**original, "training_split": "test", "source_group": "sourceA"}
        selected = heldout.select_test_records([assigned], [original], {"squat": 1})
        self.assertEqual(selected[0]["sample_id"], "train/example")
        self.assertIsNone(selected[0]["count"])
        other = {**assigned, "sample_id": "valid/other", "training_split": "valid"}
        other_original = {**original, "sample_id": "valid/other"}
        with self.assertRaisesRegex(ValueError, "来源跨集合"):
            heldout.select_test_records([assigned, other], [original, other_original])
        with self.assertRaisesRegex(ValueError, "数量"):
            heldout.select_test_records([assigned], [original], heldout.EXPECTED_COUNTS)

    def test_release_rejects_changed_snapshot_or_workspace(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs") as directory:
            root = Path(directory)
            release = root / "release"
            (release / "snapshot").mkdir(parents=True)
            original = root / "algorithm.py"
            snapshot = release / "snapshot/algorithm.py"
            original.write_text("frozen", encoding="utf-8")
            snapshot.write_text("frozen", encoding="utf-8")
            heldout.write_json(release / "release.json", {"files": {"algorithm.py": heldout.file_hash(original)}})
            (release / "release.sha256").write_text(heldout.file_hash(release / "release.json"), encoding="utf-8")
            with patch.object(heldout, "PROJECT_ROOT", root):
                heldout.verify_release(release, check_environment=False)
                original.write_text("modified", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "不一致"):
                    heldout.verify_release(release, check_environment=False)
                original.write_text("frozen", encoding="utf-8")
                snapshot.write_text("modified", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "不一致"):
                    heldout.verify_release(release, check_environment=False)

    def test_cache_reuse_ignores_preview_but_not_inference_settings(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs") as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"synthetic-file-not-decoded")
            record, config = {"sample_id": "train/example"}, {"preview": False, "sample_fps": 20}
            signature = {"source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
                         "record": record, "config": {**config, "preview": True}}
            summary = {"status": "ok", "label": record, "signature": signature}
            self.assertTrue(heldout.compatible_pose(summary, record, config, source))
            self.assertFalse(heldout.compatible_pose(summary, record, {**config, "sample_fps": 10}, source))

    def test_no_window_videos_and_unknown_counts_remain_visible(self):
        base = {"sample_id": "test/a", "label": 0, "reference_action": "squat", "samples": 100,
                "ready_samples": 0, "correct_ready_samples_by_clip_label": 0, "windows": 0,
                "raw_prediction_counts": {}, "video_prediction": None,
                "absolute_count_error": 8, "actual_geometry_missing_samples": 100}
        second = {**base, "sample_id": "test/b", "windows": 1, "raw_prediction_counts": {"0": 1},
                  "video_prediction": 0, "absolute_count_error": None}
        report = heldout.aggregate([base, second])
        self.assertEqual(report["count_mae"], 8)
        self.assertEqual(report["count_evaluated_videos"], 1)
        self.assertEqual(report["videos_without_windows"], ["test/a"])
        self.assertEqual(report["video_accuracy_all_processed_videos"], .5)
        self.assertIsNone(report["ready_clip_label_agreement"])

    def test_injected_input_matches_legacy_replay(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "outputs") as directory:
            root = Path(directory)
            sample = "valid/synthetic"
            config = {"geometry": asdict(GeometryConfig()), "windows": {
                "window_size": 48, "stride": 10, "fps": 20.0, "max_age_s": .15,
                "min_valid_ratio": .7, "min_joints_per_frame": 6}}
            pose_root = root / "data/processed/repcount_a/pose_v1"
            cache = pose_root / sample
            cache.mkdir(parents=True)
            heldout.write_json(cache / "summary.json", {"status": "ok", "label": {"sample_id": sample}})
            normalized = root / "data/processed/repcount_a/normalized_v1" / sample
            normalized.mkdir(parents=True)
            heldout.write_json(normalized / "summary.json", {"signature": {"config": config}})
            points = np.full((100, 17, 2), 200.0)
            points[:, 5], points[:, 6] = [170, 100], [230, 100]
            points[:, 11], points[:, 12] = [180, 200], [220, 200]
            np.savez_compressed(cache / "pose.npz", keypoints=points, scores=np.ones((100, 17)),
                                valid=np.ones((100, 17), dtype=bool), timestamp_s=np.arange(100) / 20,
                                frame_index=np.arange(100), segment_id=np.zeros(100, dtype=int))
            tensors = []

            def model(tensor):
                tensors.append(tensor.data.copy())
                return ArrayTensor([[5., 0., 0.]])

            def softmax(tensor, dim):
                values = np.exp(tensor.data - tensor.data.max(axis=dim, keepdims=True))
                return ArrayTensor(values / values.sum(axis=dim, keepdims=True))

            torch = SimpleNamespace(from_numpy=ArrayTensor, softmax=softmax, inference_mode=nullcontext)
            args = SimpleNamespace(count=True, render=False, device="cpu", short_gap_s=.15)
            record = {"sample_id": sample, "source_group": "synthetic", "training_split": "valid",
                      "class_id": 0, "count": None}
            with patch.object(replay_action, "PROJECT_ROOT", root):
                first = replay_action.replay_one(record, args, model, torch, ActionFilterConfig(), CountConfig(), root / "legacy")
                # 即使没有测试normalized缓存，新入口也必须给出相同的模型输入和在线行为。
                (normalized / "summary.json").unlink()
                second = replay_action.replay_one(record, args, model, torch, ActionFilterConfig(), CountConfig(), root / "injected",
                                                  pose_root=pose_root, preprocessing_config=config)
            for key in ("windows", "counts", "ready_samples", "state_samples", "input_pose_sha256"):
                self.assertEqual(first[key], second[key])
            self.assertGreater(first["windows"], 0)
            np.testing.assert_array_equal(tensors[:first["windows"]], tensors[first["windows"]:])
            for name in ("timeline.jsonl", "predictions.jsonl", "count_events.jsonl"):
                self.assertEqual((root / "legacy" / sample / name).read_bytes(), (root / "injected" / sample / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
