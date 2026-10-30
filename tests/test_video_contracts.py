"""只检查会破坏训练输入的标签别名、帧顺序和跟踪缺失语义。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_repcount import ALIASES, cycle_annotations, parse_count
from pose_video import select_person


class VideoContracts(unittest.TestCase):
    def test_actual_aliases_from_downloaded_csv(self):
        self.assertEqual(ALIASES["squant"], ALIASES["squat"])
        self.assertEqual(ALIASES["pushups"], ALIASES["push_up"])
        self.assertEqual(ALIASES["jumpjacks"], ALIASES["jump_jack"])

    def test_cycle_fields_use_numeric_order_and_preserve_shared_endpoints(self):
        row = {"name": "sample.mp4", "count": "5"}
        for number in reversed(range(1, 11)):
            row[f"L{number}"] = str((number // 2) * 10)
        pairs, complete = cycle_annotations(row)
        self.assertTrue(complete)
        self.assertEqual(pairs[-1], [40, 50])

    def test_incomplete_cycle_annotations_are_not_fabricated(self):
        pairs, complete = cycle_annotations({"name": "sample.mp4", "count": "2", "L1": "0", "L2": "10"})
        self.assertEqual(pairs, [[0, 10]])
        self.assertFalse(complete)

    def test_missing_count_is_unknown_instead_of_zero(self):
        self.assertIsNone(parse_count(""))
        self.assertEqual(parse_count("0"), 0)
        self.assertEqual(cycle_annotations({"name": "sample.mp4", "count": "", "L1": "", "L2": ""}), ([], False))

    def test_short_gap_does_not_jump_to_an_unrelated_person(self):
        previous = [0, 0, 100, 100]
        unrelated = [300, 0, 500, 200]
        box, restarted = select_person([unrelated], previous, 1.0, 1.1)
        self.assertIsNone(box)
        self.assertFalse(restarted)
        box, restarted = select_person([unrelated], previous, 1.0, 1.6)
        self.assertEqual(box, unrelated)
        self.assertTrue(restarted)
        self.assertEqual(select_person([], previous, 1.0, 1.1), (None, False))


if __name__ == "__main__":
    unittest.main()
