"""从 RepCount 清单提取连续 COCO17 关键点缓存，支持小批量和断点续跑。

所有输出保留原视频帧号和时间戳；缺失关节写 NaN 并带掩码，跟踪重启产生
新的 segment_id。后续训练不能跨跟踪段拼窗口，也不能把缺失当作零坐标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from importlib import metadata
from pathlib import Path

from pose_image import (COCO17_NAMES, DETECTOR_URL, POSE_URL, PROJECT_ROOT, inspect_session)


def box_iou(first, second) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def select_person(boxes, previous, last_seen: float, now: float, max_gap: float = 0.5):
    """短时匹配失败先留空；超时重选人体时显式开启新跟踪段。"""
    if len(boxes) == 0:
        return None, False
    if previous is None or now - last_seen > max_gap:
        index = max(range(len(boxes)), key=lambda i: max(0, boxes[i][2] - boxes[i][0])
                    * max(0, boxes[i][3] - boxes[i][1]))
        return boxes[index], True
    overlaps = [box_iou(previous, box) for box in boxes]
    index = max(range(len(boxes)), key=overlaps.__getitem__)
    # 0.1 是初始几何跟踪阈值，不代表经过身份识别或验证集调参。
    return (boxes[index], False) if overlaps[index] >= 0.1 else (None, False)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_one(record, args, detector, pose_model, model_info, config) -> dict:
    import cv2
    import numpy as np
    from rtmlib import draw_bbox, draw_skeleton

    source = Path(record["video_path"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    stat = source.stat()
    signature = {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                 "record": record, "config": config}
    split, video_id = record["sample_id"].split("/", 1)
    if split not in ("train", "valid", "test") or Path(video_id).name != video_id:
        raise ValueError("manifest 中存在非法 sample_id")
    destination = args.output / split / video_id
    summary_path = destination / "summary.json"
    cache_path = destination / "pose.npz"
    if args.resume and summary_path.is_file() and cache_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("signature") == signature and previous_summary.get("status") == "ok":
            print(f"CACHE_REUSED {record['sample_id']}", flush=True)
            return previous_summary
        raise ValueError(f"已有缓存与本次配置不同，请使用新的 --output：{destination}")
    if destination.exists() and any(destination.iterdir()):
        partial_files = {path.name for path in destination.iterdir()}
        if args.resume and partial_files.issubset({"pose.npz", "preview.mp4"}):
            # 中断任务没有成功摘要；仅允许重做本入口的已知生成文件，不动其他文件。
            print(f"RESTART_PARTIAL {record['sample_id']}", flush=True)
        else:
            raise FileExistsError(f"输出目录非空；完整缓存请加 --resume，否则使用新目录：{destination}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{source}")
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频未提供有效 FPS，不猜测时间尺度")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_fps = min(fps, args.sample_fps)
        destination.mkdir(parents=True, exist_ok=True)
        frames, timestamps, coordinates, confidence, masks, boxes_out, segments = [], [], [], [], [], [], []
        previous_box, last_seen, segment_id = None, -math.inf, -1
        next_due, last_clock, fallback_clocks, decoded = 0.0, -math.inf, 0, 0
        started = time.perf_counter()
        print(f"START {record['sample_id']} action={record['action']} fps={fps:.3f}", flush=True)
        while True:
            success, image = capture.read()
            if not success:
                break
            frame_index = decoded
            decoded += 1
            raw_clock = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(raw_clock) or raw_clock < 0 or raw_clock <= last_clock:
                raw_clock = max(frame_index / fps, last_clock + 1 / fps)
                fallback_clocks += 1
            last_clock = raw_clock
            if raw_clock + 1e-8 < next_due:
                continue
            # 在线帧到达后再采样，不读取未来帧插值；保存实际时刻而非伪造均匀时间戳。
            next_due = (math.floor((raw_clock + 1e-8) * sample_fps) + 1) / sample_fps
            boxes = np.asarray(detector(image), dtype=np.float32).reshape(-1, 4)
            box, restarted = select_person(boxes, previous_box, last_seen, raw_clock)
            points = np.full((17, 2), np.nan, dtype=np.float32)
            scores = np.zeros(17, dtype=np.float32)
            visible = np.zeros(17, dtype=bool)
            output_box = np.full(4, np.nan, dtype=np.float32)
            current_segment = -1
            if box is not None:
                keypoints, keypoint_scores = pose_model(image, bboxes=np.asarray([box]))
                if keypoints.shape != (1, 17, 2) or keypoint_scores.shape != (1, 17):
                    raise ValueError("RTMPose 输出不是 COCO17 格式")
                scores = np.nan_to_num(keypoint_scores[0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                visible = np.isfinite(keypoints[0]).all(axis=1) & (scores >= args.keypoint_threshold)
                points[visible] = keypoints[0][visible]
                previous_box, last_seen = box.copy(), raw_clock
                if restarted:
                    segment_id += 1
                current_segment = segment_id
                output_box = box.astype(np.float32)
            frames.append(frame_index)
            timestamps.append(raw_clock)
            coordinates.append(points)
            confidence.append(scores)
            masks.append(visible)
            boxes_out.append(output_box)
            segments.append(current_segment)

            if args.preview:
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(str(destination / "preview.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("预览视频编码器不可用，可不加 --preview 先提取缓存")
                if box is not None:
                    # 无效点以零分屏蔽绘制，缓存中的 NaN 保持缺失语义。
                    rendered = draw_bbox(image.copy(), [box])
                    rendered = draw_skeleton(rendered, np.nan_to_num(points)[None],
                                             (scores * visible)[None], kpt_thr=args.keypoint_threshold)
                else:
                    rendered = image
                cv2.putText(rendered, f"{record['action']}  frame={frame_index}  segment={current_segment}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                writer.write(rendered)
            if len(frames) % 100 == 0:
                print(f"  sampled={len(frames)} source_frame={frame_index}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not frames:
        raise RuntimeError("没有解码到可用视频帧")
    if expected_frames > 0 and decoded < expected_frames - 1:
        raise RuntimeError(f"视频疑似提前解码终止：实际 {decoded}，元数据 {expected_frames}；不标记完整缓存")
    valid_mask = np.asarray(masks, dtype=bool)
    if not valid_mask.any():
        raise RuntimeError("整段视频没有有效关键点；检查骨架预览或模型输出")
    np.savez_compressed(cache_path,
        keypoints=np.asarray(coordinates, dtype=np.float32), scores=np.asarray(confidence, dtype=np.float32),
        valid=valid_mask, frame_index=np.asarray(frames, dtype=np.int64),
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        bbox_xyxy=np.asarray(boxes_out, dtype=np.float32), segment_id=np.asarray(segments, dtype=np.int32))
    summary = {
        "status": "ok", "sample_id": record["sample_id"], "label": record,
        "source": str(source), "source_fps": fps, "source_decoded_frames": decoded,
        "sample_fps_requested": args.sample_fps, "sample_fps_limit": sample_fps,
        "sampled_frames": len(frames), "timestamp_fallback_frames": fallback_clocks,
        "valid_joint_ratio": float(valid_mask.mean()),
        "frames_with_any_valid_joint_ratio": float(valid_mask.any(axis=1).mean()),
        "tracking_segments": segment_id + 1, "processing_seconds": elapsed,
        "keypoint_format": "coco17", "keypoint_names": COCO17_NAMES,
        "coordinate_system": "original_image_pixels_xy_top_left_origin",
        "cache_schema": 1, "signature": signature, "models": model_info,
        "notes": ["NaN+valid 表示缺失；segment_id=-1 表示本帧没有匹配人物",
                  "只做几何跟踪，无人物身份识别保证；训练窗口不得跨 segment",
                  "保留实际时间戳，预览采用固定帧率，仅供视觉确认",
                  "count 为真实标签，本阶段没有预测动作次数或质量错误"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"POSE_VIDEO_OK {record['sample_id']} samples={len(frames)} valid_joint_ratio={valid_mask.mean():.3f}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/pose_v1")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--actions", nargs="+", choices=("squat", "push_up", "jump_jack"), default=["squat"])
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--sample-fps", type=float, default=20.0)
    parser.add_argument("--keypoint-threshold", type=float, default=0.3)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.per_class < 1 or not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error("per-class 和 sample-fps 必须为正数")
    if not 0 < args.keypoint_threshold <= 1:
        parser.error("keypoint-threshold 必须在 (0,1] 内")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for action in args.actions:
        candidates = sorted((row for row in records if row["official_split"] == args.split and row["action"] == action),
                            key=lambda row: row["sample_id"])
        if not candidates:
            raise ValueError(f"清单没有 {args.split}/{action}")
        selected.extend(candidates[:args.per_class])

    import torch  # 先加载已在本机验证可供 ORT 复用的 CUDA/cuDNN 动态库。
    import onnxruntime as ort
    from rtmlib import RTMPose, YOLOX
    from rtmlib.tools.file import download_checkpoint
    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("本入口要求可用的 CUDA 环境")
    model_root = PROJECT_ROOT / "models/pretrained"
    detector_path = Path(download_checkpoint(DETECTOR_URL, dst_dir=str(model_root / "yolox_tiny")))
    pose_path = Path(download_checkpoint(POSE_URL, dst_dir=str(model_root / "rtmpose_m")))
    detector = YOLOX(str(detector_path), model_input_size=(416, 416), score_thr=0.5, device="cuda", backend="onnxruntime")
    pose_model = RTMPose(str(pose_path), model_input_size=(192, 256), to_openpose=False, device="cuda", backend="onnxruntime")
    model_info = {"detector": inspect_session(detector, (416, 416), "cuda"),
                  "pose": inspect_session(pose_model, (256, 192), "cuda")}
    config = {"cache_schema": 1, "sample_fps": args.sample_fps,
              "keypoint_threshold": args.keypoint_threshold, "preview": args.preview,
              "detector_score_threshold": 0.5, "tracking_iou": 0.1, "tracking_max_gap_s": 0.5,
              "detector_sha256": file_hash(detector_path), "pose_sha256": file_hash(pose_path),
              "rtmlib": metadata.version("rtmlib"), "onnxruntime-gpu": metadata.version("onnxruntime-gpu")}
    print(f"selected videos: {len(selected)}; CUDA sessions: {model_info}", flush=True)
    for record in selected:
        extract_one(record, args, detector, pose_model, model_info, config)
    print("POSE_CACHE_BATCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
