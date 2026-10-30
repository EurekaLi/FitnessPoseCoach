"""只读已有验证回放与骨架，分离真实缺失、恢复预热和计数信号问题。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fitness_pose.counting import pose_signal
from fitness_pose.normalization import normalize_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/count_valid_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/replay_diagnostics_v3")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("请用新的诊断输出目录，不覆盖旧结果")
    report = json.loads((args.input / "report.json").read_text(encoding="utf-8"))
    if report["split"] != "valid" or report["test_evaluated"]:
        raise ValueError("本次诊断只接受验证集")
    results, aggregate = [], Counter()
    for old in report["results"]:
        sample = old["sample_id"]
        with np.load(ROOT / "data/processed/repcount_a/pose_v1" / sample / "pose.npz", allow_pickle=False) as cache:
            raw = {key: cache[key] for key in cache.files}
        norm = normalize_pose(raw["keypoints"], raw["scores"], raw["valid"], raw["timestamp_s"], raw["segment_id"])
        timeline = [json.loads(line) for line in (args.input / sample / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(timeline) != len(raw["timestamp_s"]):
            raise ValueError(f"日志与缓存长度不匹配：{sample}")
        counts, angles, sides, runs = Counter(), [], [], []
        start, previous_side, previous_segment = None, None, -1
        for i, now in enumerate(raw["timestamp_s"]):
            geometry = bool(norm["geometry_valid"][i])
            segment = int(raw["segment_id"][i])
            counts["samples"] += 1
            counts["actual_geometry_missing"] += int(not geometry)
            counts["legacy_missing_but_geometry_recovered"] += int(geometry and timeline[i]["status"] == "missing")
            continuous = (i > 0 and geometry and norm["geometry_valid"][i - 1]
                          and segment == raw["segment_id"][i - 1]
                          and now - raw["timestamp_s"][i - 1] <= .15)
            if start is not None and not continuous:
                runs.append(float(raw["timestamp_s"][i - 1] - raw["timestamp_s"][start]))
                start = None
            if geometry and start is None:
                start = i
            if not geometry:
                if segment < 0:
                    counts["tracking_missing"] += 1
                elif not raw["valid"][i, [5, 6, 11, 12]].all():
                    counts["anchor_joints_missing"] += 1
                else:
                    counts["anchor_scale_invalid"] += 1
            signal, side = pose_signal(old["reference_action"], raw["keypoints"][i], raw["scores"][i], raw["valid"][i])
            if signal is not None:
                angles.append(float(signal[0]))
                if segment == previous_segment and previous_side is not None and side != previous_side:
                    counts["instant_side_changes"] += 1
                previous_side = side
                sides.append(side)
            else:
                previous_side = None
            previous_segment = segment
        if start is not None:
            runs.append(float(raw["timestamp_s"][-1] - raw["timestamp_s"][start]))
        aggregate.update(counts)
        results.append({"sample_id": sample, "reference_action": old["reference_action"], "counts": dict(counts),
                        "continuous_runs": len(runs), "runs_ge_2_35s": sum(d >= 2.35 - 1e-7 for d in runs),
                        "longest_run_s": max(runs, default=0), "legacy_state_samples": old["state_samples"],
                        "ready_coverage": old["ready_coverage"], "count_error": old["absolute_count_error"],
                        "signal_p10_p50_p90": np.quantile(angles, [.1, .5, .9]).tolist() if angles else None})
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"split": "valid", "test_evaluated": False, "videos": len(results), "totals": dict(aggregate),
              "results": results, "note": "只定位已有验证结果；未用真实标签控制在线预测或计数"}
    (args.output / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(results), "totals": dict(aggregate)}, ensure_ascii=False, indent=2))
    print("REPLAY_DIAGNOSTICS_OK")


if __name__ == "__main__":
    main()
