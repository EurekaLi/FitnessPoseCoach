"""Windows无PyTorch在线流程与已保存的三段验证时间线对照，不打开摄像头。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from action_onnx import (PROJECT_ROOT, REPLAY_SAMPLES, ORTActionClassifier, validation_records,
                         load_pose, read_jsonl, file_hash)
from online_pipeline import load_candidate, OnlineActionPipeline
from replay_action import write_json, write_jsonl
from check_devices import prepare_dlls


def verify(release: Path, bundle: Path, output: Path, device: str) -> int:
    frozen, exported = load_candidate(release, bundle)
    if output.exists():
        raise FileExistsError(f"不覆盖旧验证结果，请指定新的--output：{output}")
    handles = prepare_dlls() if device == "cuda" else []
    try:
        classifier = ORTActionClassifier(bundle / "action_stgcnpp.onnx", device)
        output.mkdir(parents=True, exist_ok=False)
        report = {"status": "incomplete", "device": device, "torch_imported": False,
                  "release_sha256": exported["release_sha256"], "onnx_sha256": exported["onnx_sha256"],
                  "pipeline_code_sha256": file_hash(Path(__file__).with_name("online_pipeline.py")),
                  "test_dataset_used": False, "results": []}
        records = {row["sample_id"]: row for row in validation_records(release)}
        keys = ("timestamp_s", "frame_index", "segment_id", "action", "status", "counts", "count_status",
                "geometry_valid", "geometry_carried", "valid_joints", "window_reset_reason", "signal_type", "side", "side_changed")
        for sample in REPLAY_SAMPLES:
            raw, provenance = load_pose(records[sample], frozen)
            pipeline, timeline, predictions, events = OnlineActionPipeline(frozen, classifier), [], [], []
            for index, timestamp in enumerate(raw["timestamp_s"]):
                item = pipeline.step(raw["keypoints"][index], raw["scores"][index], raw["valid"][index],
                                     float(timestamp), int(raw["segment_id"][index]), int(raw["frame_index"][index]))
                timeline.append(item["timeline"])
                predictions.extend(item["predictions"])
                if item["event"]:
                    events.append(item["event"])
            reference_dir = bundle / "replay_torch" / sample
            reference = read_jsonl(reference_dir / "timeline.jsonl")
            reference_predictions = read_jsonl(reference_dir / "predictions.jsonl")
            mismatch = [i for i, (a, b) in enumerate(zip(timeline, reference)) if any(a[key] != b[key] for key in keys)]
            predictions_equal = (len(predictions) == len(reference_predictions) and all(
                a["window_end_s"] == b["window_end_s"] and a["raw_prediction"] == b["raw_prediction"]
                and np.allclose(a["raw_probabilities"], b["raw_probabilities"], atol=1e-5, rtol=0)
                for a, b in zip(predictions, reference_predictions)))
            events_equal = events == read_jsonl(reference_dir / "count_events.jsonl")
            passed = len(timeline) == len(reference) and not mismatch and predictions_equal and events_equal
            row = {**provenance, "passed": passed, "samples": len(timeline), "windows": len(predictions),
                   "state_mismatch_indices": mismatch, "predictions_match": predictions_equal,
                   "count_events_identical": events_equal, "counts": pipeline.counter.counts}
            directory = output / sample
            directory.mkdir(parents=True)
            write_jsonl(directory / "timeline.jsonl", timeline)
            write_jsonl(directory / "predictions.jsonl", predictions)
            write_jsonl(directory / "count_events.jsonl", events)
            write_json(directory / "summary.json", row)
            report["results"].append(row)
            report["status"] = "ok" if len(report["results"]) == 3 and all(r["passed"] for r in report["results"]) else "incomplete"
            write_json(output / "report.json", report)
            print(f"ONLINE_REPLAY {'OK' if passed else 'FAILED'} {sample} windows={len(predictions)} counts={pipeline.counter.counts}", flush=True)
        import sys
        report["torch_imported"] = "torch" in sys.modules
        if report["torch_imported"]:
            report["status"] = "failed"
        write_json(output / "report.json", report)
        success = report["status"] == "ok"
        print(f"WINDOWS_ONLINE_{'OK' if success else 'FAILED'} videos=3 report={output / 'report.json'}", flush=True)
        return 0 if success else 1
    finally:
        for handle in handles:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=PROJECT_ROOT / "releases/v6_test_v1")
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "outputs/action_onnx_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/windows_online_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    return verify(args.release.resolve(), args.bundle.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    raise SystemExit(main())
