"""复用已产生的逐时刻动作判断，只比较计数策略；不重跑分类模型或姿态模型。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fitness_pose.counting import CountConfig, ExerciseCounter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/count_valid_v2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CountConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    original = json.loads((args.input / "report.json").read_text(encoding="utf-8"))
    if original["split"] != "valid" or original["test_evaluated"]:
        raise ValueError("本次对照仅接受验证结果")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("不覆盖旧结果，请指定新的输出目录")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for old in original["results"]:
        sample = old["sample_id"]
        with np.load(ROOT / "data/processed/repcount_a/pose_v1" / sample / "pose.npz", allow_pickle=False) as raw:
            cache = {k: raw[k] for k in raw.files}
        timeline = [json.loads(s) for s in (args.input / sample / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(timeline) != len(cache["timestamp_s"]):
            raise ValueError("骨架与动作日志长度不匹配")
        counter, events = ExerciseCounter(config), []
        for i, row in enumerate(timeline):
            if abs(row["timestamp_s"] - float(cache["timestamp_s"][i])) > 1e-8:
                raise ValueError("骨架与动作日志时间戳不匹配")
            item = counter.step(row["action"], cache["keypoints"][i], cache["scores"][i], cache["valid"][i],
                                row["timestamp_s"], int(cache["segment_id"][i]))
            if item["event"]:
                events.append({"timestamp_s": row["timestamp_s"], "action": item["event"]})
        # 标签仅在预测全部结束后用于评价，前面的计数规则不接触次数或类别真值。
        prediction = counter.counts[old["reference_action"]]
        truth = old["reference_count"]
        results.append({"sample_id": sample, "reference_action": old["reference_action"], "reference_count": truth,
                        "prediction": prediction, "absolute_count_error": abs(prediction - truth) if truth is not None else None,
                        "counts": counter.counts, "side_changes": counter.side_changes, "events": events})
    errors = [r["absolute_count_error"] for r in results if r["absolute_count_error"] is not None]
    report = {"split": "valid", "test_evaluated": False, "videos": len(results), "config": asdict(config),
              "input_report_sha256": hashlib.sha256((args.input / "report.json").read_bytes()).hexdigest(),
              "counting_code_sha256": hashlib.sha256((ROOT / "src/fitness_pose/counting.py").read_bytes()).hexdigest(),
              "count_mae": sum(errors) / len(errors) if errors else None,
              "within_one_ratio": sum(e <= 1 for e in errors) / len(errors) if errors else None, "results": results}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("videos", "count_mae", "within_one_ratio")}, ensure_ascii=False))
    print("RECOUNT_COMPARE_OK")


if __name__ == "__main__":
    main()
