"""转换检查的最小边界测试，不加载训练模型或真实数据。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from action_onnx import compare_logits, load_pose, softmax


class ActionOnnxContracts(unittest.TestCase):
    def test_close_scores_and_probabilities_pass(self):
        reference = np.array([[2., 1., -1.], [-2., 4., .5]], dtype=np.float32)
        comparison = compare_logits(reference, reference + 1e-6)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["top1_agreement"], 1)
        np.testing.assert_allclose(softmax(reference + 1000), softmax(reference), atol=1e-12)

    def test_near_tie_changed_class_fails_even_when_numbers_are_close(self):
        comparison = compare_logits(np.array([[0., 1e-6, 0.]]), np.array([[1e-6, 0., 0.]]))
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["top1_mismatch_indices"], [0])

    def test_invalid_outputs_and_test_split_are_rejected(self):
        for actual in (np.array([[np.nan, 0, 0]]), np.zeros((0, 3)), np.zeros((1, 4))):
            with self.assertRaises(ValueError):
                compare_logits(np.zeros((1, 3)), actual)
        with self.assertRaisesRegex(ValueError, "拒绝测试"):
            load_pose({"training_split": "test"}, {})


if __name__ == "__main__":
    unittest.main()
