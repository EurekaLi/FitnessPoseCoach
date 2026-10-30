"""按时间顺序回放缓存骨架，运行真实动作模型，可选周期计数与原视频叠加。

这不是检测/姿态/采集全链路测速：姿态来自已有缓存，默认按顺序尽快执行。
真实动作与次数只在每段预测完成后评价，绝不用于门控分类和计数。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def render_video(source: Path, output: Path, raw: dict, timeline: list[dict], max_age: float) -> dict:
    import cv2
    import numpy as np
    capture, writer = cv2.VideoCapture(str(source)), None
    frame_index, pointer, last_clock = 0, -1, -math.inf
    edges = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (11, 5), (12, 6),
             (5, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0)]
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"无法打开原视频或帧率无效：{source}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            now = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(now) or now < 0 or now <= last_clock:
                now = max(frame_index / fps, last_clock + 1 / fps)
            last_clock = now
            while pointer + 1 < len(timeline) and raw["frame_index"][pointer + 1] <= frame_index:
                pointer += 1
            lines = [f"cached-pose causal replay | t={now:.2f}s", "warming_up"]
            if pointer >= 0:
                row = timeline[pointer]
                age = now - row["timestamp_s"]
                fresh = age <= max_age and age >= -1e-6
                if fresh and row["segment_id"] >= 0:
                    points, mask = raw["keypoints"][pointer], raw["valid"][pointer]
                    for a, b in edges:
                        if mask[a] and mask[b] and np.isfinite(points[[a, b]]).all():
                            cv2.line(frame, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (80, 230, 80), 2)
                    for point, valid in zip(points, mask):
                        if valid and np.isfinite(point).all():
                            cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, (30, 180, 255), -1)
                action = row["action"] if fresh else None
                lines = [f"cached-pose causal replay | t={now:.2f}s",
                         f"action={action or 'unknown'}  status={row['status'] if fresh else 'stale_pose'}",
                         f"p={row['confidence']:.2f}  segment={row['segment_id']}"]
                if row["counts"] is not None:
                    lines.extend(["counts: " + "  ".join(f"{k}={v}" for k, v in row["counts"].items()),
                                  f"phase={row['count_status']}"])
            height, width = frame.shape[:2]
            scale = max(0.30, min(0.5, width / 1300))
            line_height = max(15, round(20 * scale / 0.5))
            panel_height = line_height * (6 if timeline[0]["counts"] is not None else 4)
            # 独立顶栏不遮住肩髋等关键关节；保持原图完整尺寸，只增加输出画布高度。
            rendered = np.full((height + panel_height, width, 3), 25, dtype=np.uint8)
            rendered[panel_height:] = frame
            for i, text in enumerate(lines, 1):
                cv2.putText(rendered, text, (8, i * line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + panel_height))
                if not writer.isOpened():
                    raise RuntimeError("MP4编码器不可用；可去掉 --render 先生成数值日志")
            writer.write(rendered)
            frame_index += 1
        if frame_index == 0 or expected > 0 and frame_index < expected - 1:
            raise RuntimeError(f"视频解码不完整：{frame_index}/{expected}")
        if frame_index <= int(raw["frame_index"][-1]):
            raise RuntimeError("原视频长度与骨架缓存帧号不一致")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return {"frames": frame_index, "fps": fps, "source_size_wh": [width, height],
            "output_size_wh": [width, height + panel_height], "audio_preserved": False}


def replay_one(record: dict, args, model, torch, filter_config, count_config, output: Path,
               *, pose_root: Path | None = None, preprocessing_config: dict | None = None) -> dict:
    import numpy as np
    from fitness_pose.action_data import CLASSES, classification_metrics
    from fitness_pose.counting import ExerciseCounter
    from fitness_pose.normalization import CausalPoseNormalizer
    from fitness_pose.streaming import ActionFilter, OnlineWindowBuffer

    sample_id = record["sample_id"]
    cache_dir = (pose_root or PROJECT_ROOT / "data/processed/repcount_a/pose_v1") / sample_id
    summary_path = cache_dir / "summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary["status"] != "ok" or source_summary["label"]["sample_id"] != sample_id:
        raise ValueError(f"骨架缓存未完成或不匹配：{sample_id}")
    with np.load(cache_dir / "pose.npz", allow_pickle=False) as cache:
        raw = {name: cache[name] for name in ("keypoints", "scores", "valid", "timestamp_s", "frame_index", "segment_id")}
    timestamps = raw["timestamp_s"]
    total = len(timestamps)
    if (not total or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
            or np.any(np.diff(raw["frame_index"]) <= 0)):
        raise ValueError(f"缓存时间/帧号异常：{sample_id}")
    # 训练前处理的网格与几何参数固定；遇到不同配置不悄悄使用不匹配的在线输入。
    if preprocessing_config is None:
        normalized_summary = PROJECT_ROOT / "data/processed/repcount_a/normalized_v1" / sample_id / "summary.json"
        prepared = json.loads(normalized_summary.read_text(encoding="utf-8"))
        config = prepared["signature"]["config"]
    else:
        # 独立测试沿用冻结的验证前处理，不为测试视频重新选择配置或覆盖训练索引。
        config = preprocessing_config
    from fitness_pose.normalization import GeometryConfig
    if config["geometry"] != asdict(GeometryConfig()):
        raise ValueError("训练几何配置与回放默认配置不同")
    window_config = {**config["windows"], "short_gap_s": args.short_gap_s}
    window = OnlineWindowBuffer(**window_config)
    if window.window_size != 48 or window.fps != 20 or window.stride != 10:
        raise ValueError("当前最佳模型回放要求48点/20Hz/步长10")
    normalizer, action_filter = CausalPoseNormalizer(), ActionFilter(filter_config)
    counter = ExerciseCounter(count_config) if args.count else None
    destination = output / sample_id
    destination.mkdir(parents=True, exist_ok=False)
    predictions, timeline, events, model_times = [], [], [], []
    reset_counts = Counter()
    missing_geometry_samples = 0
    started = time.perf_counter()
    for index, now_value in enumerate(timestamps):
        now, segment = float(now_value), int(raw["segment_id"][index])
        frame = normalizer.step(raw["keypoints"][index], raw["scores"][index], raw["valid"][index], now, segment)
        emitted = window.step(frame, now, segment, index)
        if window.last_reset_reason:
            action_filter.reset()
            reset_counts[window.last_reset_reason] += 1
        action_filter.expire(now)
        for item in emitted:
            tensor = torch.from_numpy(item["x"]).unsqueeze(0).to(args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            infer_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            if args.device == "cuda":
                torch.cuda.synchronize()
            model_times.append((time.perf_counter() - infer_started) * 1000)
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("分类概率包含NaN/Inf")
            action_filter.update(probabilities, now)
            predictions.append({"window_start_s": float(item["time_grid_s"][0]),
                                "window_end_s": float(item["time_grid_s"][-1]), "emitted_at_s": now,
                                "latest_source_timestamp_s": float(timestamps[item["source_index"][-1]]),
                                "max_source_age_s": item["max_source_age_s"],
                                "segment_id": segment, "raw_prediction": int(probabilities.argmax()),
                                "raw_probabilities": probabilities.tolist(),
                                "smoothed_probabilities": action_filter.probabilities.tolist(),
                                "action": CLASSES[action_filter.action] if action_filter.action is not None else None,
                                "status": action_filter.status})
        # 显示状态与分类历史分开：当前缺失必须暂停，短暂恢复不应永远沿用missing标签。
        pose_usable = bool(frame["geometry_valid"] and segment >= 0)
        missing_geometry_samples += int(not pose_usable)
        status = action_filter.status if pose_usable else "missing"
        if pose_usable and action_filter.last_prediction is None and status not in ("stale", "confirming"):
            status = "warming_up"
        action = CLASSES[action_filter.action] if pose_usable and action_filter.action is not None else None
        count_result = counter.step(action, raw["keypoints"][index], raw["scores"][index], raw["valid"][index], now, segment) if counter else None
        if count_result and count_result["event"]:
            events.append({"timestamp_s": now, "frame_index": int(raw["frame_index"][index]), "segment_id": segment,
                           "action": count_result["event"], "counts": count_result["counts"]})
        timeline.append({"timestamp_s": now, "frame_index": int(raw["frame_index"][index]), "segment_id": segment,
                         "action": action, "status": status,
                         "geometry_valid": pose_usable, "valid_joints": int(frame["valid"].sum()),
                         "geometry_carried": bool(frame["geometry_carried"]),
                         "window_reset_reason": window.last_reset_reason,
                         "confidence": float(action_filter.probabilities.max()) if action_filter.probabilities is not None else 0.0,
                         "counts": count_result["counts"] if count_result else None,
                         "count_status": count_result["status"] if count_result else None,
                         "signal": count_result.get("signal") if count_result else None,
                         "signal_type": count_result.get("signal_type") if count_result else None,
                         "side": count_result.get("side") if count_result else None,
                         "side_changed": count_result.get("side_changed", False) if count_result else False,
                         "thresholds": count_result.get("thresholds") if count_result else None})
    # 此处才读取真实动作/次数，预测与计数在前面的循环中完全不接触它们。
    label = record["class_id"]
    ready = [r for r in timeline if r["action"] is not None]
    correct = sum(r["action"] == CLASSES[label] for r in ready)
    truth_count = record["count"]
    predicted_count = counter.counts[CLASSES[label]] if counter else None
    count_error = abs(predicted_count - truth_count) if counter and truth_count is not None else None
    result = {"status": "ok", "sample_id": sample_id, "source_group": record["source_group"],
              "training_split": record["training_split"], "label": label, "reference_action": CLASSES[label],
              "samples": total, "windows": len(predictions), "duration_s": float(timestamps[-1] - timestamps[0]),
              "ready_samples": len(ready), "correct_ready_samples_by_clip_label": correct,
              "ready_coverage": len(ready) / total,
              "ready_clip_label_agreement": correct / len(ready) if ready else None,
              "all_samples_clip_label_agreement": correct / total,
              "raw_window_metrics_by_clip_label": classification_metrics([label] * len(predictions), [r["raw_prediction"] for r in predictions]),
              "raw_prediction_counts": dict(Counter(r["raw_prediction"] for r in predictions)),
              "state_samples": dict(Counter(r["status"] for r in timeline)),
              "actual_geometry_missing_samples": missing_geometry_samples,
              "window_reset_counts": dict(reset_counts), "rejected_windows": window.rejected_windows,
              "short_gap_s": args.short_gap_s,
              "side_changes": counter.side_changes if counter else None,
              "counts": counter.counts if counter else None, "reference_count": truth_count,
              "predicted_count_for_reference_action": predicted_count, "absolute_count_error": count_error,
              "count_events": len(events), "processing_s_without_render": time.perf_counter() - started,
              "mean_action_inference_ms": float(np.mean(model_times)) if model_times else None,
              "input_pose_sha256": file_hash(cache_dir / "pose.npz"),
              "notes": ["整段标签一致率不是有逐帧标注的真实逐帧准确率", "计数不补算分类预热期间、未知或丢失期间的周期",
                        "姿态来自缓存，推理耗时不含检测/姿态估计，不代表端到端摄像头速度"]}
    write_jsonl(destination / "predictions.jsonl", predictions)
    write_jsonl(destination / "timeline.jsonl", timeline)
    if counter:
        write_jsonl(destination / "count_events.jsonl", events)
    if args.render:
        result["render"] = render_video(PROJECT_ROOT / source_summary["label"]["video_path"], destination / "replay.mp4", raw, timeline, window.max_age_s)
    write_json(destination / "summary.json", result)
    print(f"REPLAY_VIDEO_OK {sample_id} windows={len(predictions)} coverage={result['ready_coverage']:.3f} "
          f"count={predicted_count} reference={truth_count} error={count_error}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/action_stgcnpp_v1/best.pt")
    parser.add_argument("--assignments", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/source_split_v1/video_assignments.jsonl")
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--actions", nargs="+", choices=("squat", "push_up", "jump_jack"), default=["squat", "push_up", "jump_jack"])
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--sample-id", help="只回放指定sample_id，但仍必须属于所选training_split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--render", action="store_true", help="解码原视频，输出无音频MP4；默认只写数值日志")
    parser.add_argument("--count", action="store_true", help="启用三类几何周期计数")
    parser.add_argument("--count-config", type=Path, help="CountConfig字段的JSON覆盖；记录实际全部参数")
    parser.add_argument("--filter-config", type=Path, help="ActionFilterConfig字段的JSON覆盖")
    parser.add_argument("--short-gap-s", type=float, default=0.0,
                        help="可选短缺失恢复时长，最大0.15秒；0保留旧窗口策略，不改变48点输入")
    args = parser.parse_args()
    if args.per_class < 1 or not math.isfinite(args.short_gap_s) or not 0 <= args.short_gap_s <= .15:
        parser.error("per-class必须为正数，short-gap-s必须位于[0,0.15]")
    import torch
    from fitness_pose.action_data import CLASSES
    from fitness_pose.counting import CountConfig
    from fitness_pose.streaming import ActionFilterConfig
    from fitness_pose.stgcnpp import STGCNPP
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请确认激活fitness-pose；不自动退回CPU")
    # 只接受本项目生成、用户可信的检查点；weights_only限制反序列化对象类型。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training = checkpoint["config"]
    if training["smoke_only"] or training["classes"] != list(CLASSES) or training["input_shape"] != ["B", 3, 48, 17]:
        raise ValueError("检查点是冒烟模型或输入/类别不匹配")
    model_path = PROJECT_ROOT / "src/fitness_pose/stgcnpp.py"
    if training["code_sha256"]["src/fitness_pose/stgcnpp.py"] != file_hash(model_path):
        raise ValueError("模型实现与训练时指纹不一致，不能直接声称兼容")
    index = args.assignments.parent / "windows_index.jsonl"
    if training["index_sha256"] != file_hash(index):
        raise ValueError("当前来源划分与训练时不同，停止，避免误用验证/测试来源")
    records = [json.loads(line) for line in args.assignments.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["training_split"] == args.split]
    if args.sample_id:
        selected = [r for r in records if r["sample_id"] == args.sample_id]
    else:
        selected = [r for action in args.actions for r in sorted((r for r in records if r["action"] == action), key=lambda r: r["sample_id"])[:args.per_class]]
    if not selected:
        raise ValueError("没有匹配所选来源集合的视频")
    filter_config = ActionFilterConfig(**(json.loads(args.filter_config.read_text(encoding="utf-8")) if args.filter_config else {}))
    count_config = CountConfig(**(json.loads(args.count_config.read_text(encoding="utf-8")) if args.count_config else {}))
    output = args.output or PROJECT_ROOT / "outputs/replay_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出已有文件，请使用新的--output：{output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
               "checkpoint_epoch": checkpoint["epoch"], "assignments_sha256": file_hash(args.assignments),
               "window_index_sha256": file_hash(index), "device": args.device, "torch": str(torch.__version__),
               "model": training["architecture"], "filter": asdict(filter_config), "counter": asdict(count_config) if args.count else None,
               "split": args.split, "samples": [r["sample_id"] for r in selected], "render": args.render,
               "short_gap_s": args.short_gap_s,
               "code_sha256": {name: file_hash(PROJECT_ROOT / name) for name in
                    ("scripts/replay_action.py", "src/fitness_pose/normalization.py", "src/fitness_pose/streaming.py", "src/fitness_pose/counting.py")}})
    model = STGCNPP(dropout=training["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros(1, 3, 48, 17, device=args.device))
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"REPLAY_START videos={len(selected)} counter={args.count} checkpoint_epoch={checkpoint['epoch']} output={output}", flush=True)
    results = [replay_one(record, args, model, torch, filter_config, count_config, output) for record in selected]
    from fitness_pose.action_data import classification_metrics
    labels, predictions = [], []
    for row in results:
        for pred, amount in row["raw_prediction_counts"].items():
            labels.extend([row["label"]] * amount)
            predictions.extend([int(pred)] * amount)
    total = sum(r["samples"] for r in results)
    ready = sum(r["ready_samples"] for r in results)
    correct = sum(r["correct_ready_samples_by_clip_label"] for r in results)
    errors = [r["absolute_count_error"] for r in results if r["absolute_count_error"] is not None]
    inventory = {"status": "ok", "split": args.split, "videos": len(results),
                 "raw_windows": len(labels), "raw_window_metrics_by_clip_label": classification_metrics(labels, predictions),
                 "ready_coverage": ready / total, "ready_clip_label_agreement": correct / ready if ready else None,
                 "all_samples_clip_label_agreement": correct / total,
                 "counting_enabled": args.count, "count_evaluated_videos": len(errors),
                 "actual_geometry_missing_samples": sum(r["actual_geometry_missing_samples"] for r in results),
                 "state_samples": dict(sum((Counter(r["state_samples"]) for r in results), Counter())),
                 "side_changes": sum(r["side_changes"] or 0 for r in results),
                 "count_mae": sum(errors) / len(errors) if errors else None,
                 "count_within_one_ratio": sum(e <= 1 for e in errors) / len(errors) if errors else None,
                 "count_by_action": {a: {"videos": sum(r["reference_action"] == a and r["absolute_count_error"] is not None for r in results),
                     "mae": (sum(r["absolute_count_error"] for r in results if r["reference_action"] == a and r["absolute_count_error"] is not None) /
                             sum(r["reference_action"] == a and r["absolute_count_error"] is not None for r in results))
                     if any(r["reference_action"] == a and r["absolute_count_error"] is not None for r in results) else None} for a in CLASSES},
                 "test_evaluated": False, "results": results,
                 "notes": ["验证只按来源分组选择；test入口尚未开放", "整段标签一致率不能当作逐帧精确标注评估",
                           "默认规则原型，计数有效性以实际误差为准；缓存回放不是摄像头全链路测速"]}
    write_json(output / "report.json", inventory)
    print(f"REPLAY_BATCH_OK videos={len(results)} ready_coverage={inventory['ready_coverage']:.3f} count_mae={inventory['count_mae']}", flush=True)
    print(f"report: {output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
