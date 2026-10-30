"""来源隔离和训练输入的必要检查；GPU 前反向检查由 --smoke 统一完成。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_source_split import assign_sources
from fitness_pose.action_data import WindowDataset, classification_metrics, load_index, summarize_predictions


class ActionDataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_source_priority_and_unknown_exclusion(self):
        manifest = [{"sample_id": f"{s}/{n}", "official_split": s, "action": "squat"}
                    for s, n in (("train", "a"), ("valid", "b"), ("test", "c"), ("train", "unknown"), ("valid", "bad"))]
        mapping = {n: {"source_video_id": "123456789ab", "action": "squat"} for n in ("a", "b", "c")}
        mapping["bad"] = {"source_video_id": "#NAME?", "action": "squat"}
        assigned, excluded = assign_sources(manifest, mapping)
        self.assertEqual({r["training_split"] for r in assigned}, {"test"})
        self.assertEqual({r["reason"] for r in excluded}, {"no_mapping", "invalid_source_id"})
        assigned, _ = assign_sources(manifest[:2], mapping)
        self.assertEqual({r["training_split"] for r in assigned}, {"valid"})

    def test_loader_rejects_source_leakage(self):
        rows = [{"window_id": str(i), "sample_id": str(i), "source_group": "same",
                 "training_split": s, "class_id": 0, "action": "squat"}
                for i, s in enumerate(("train", "valid"))]
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "跨集合"):
                load_index(path)

    def test_dataset_reads_selected_window_and_balances_videos(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "windows.npz", x=np.ones((3, 3, 48, 17), dtype=np.float32))
            rows = [{"sample_id": name, "class_id": 0, "cache_path": "windows.npz", "window_index": i}
                    for i, name in enumerate(("short", "long", "long"))]
            data = WindowDataset(rows, root)
            self.assertEqual(data[2][0].shape, (3, 48, 17))
            weights = data.sampling_weights()
            self.assertAlmostEqual(weights[0], weights[1] + weights[2])

    def test_metrics_and_video_aggregation(self):
        rows = [{"sample_id": "a", "source_group": "src1", "class_id": 0},
                {"sample_id": "a", "source_group": "src1", "class_id": 0},
                {"sample_id": "b", "source_group": "src2", "class_id": 1},
                {"sample_id": "c", "source_group": "src3", "class_id": 2}]
        result = summarize_predictions(rows, np.array([[.9, .05, .05], [.4, .5, .1], [.05, .9, .05], [.05, .05, .9]]))
        self.assertEqual(result["video"]["macro_f1"], 1)
        self.assertEqual(result["window"]["accuracy"], .75)
        self.assertEqual(classification_metrics([0, 1, 2], [0, 0, 0])["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
