"""将已有姿态缓存归一化并整理为固定时长训练样本，不重新运行视觉模型。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_video(summary_path: Path, output: Path, config: dict, resume: bool) -> dict:
    import numpy as np
    from fitness_pose.normalization import GeometryConfig, make_windows, normalize_pose

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary["status"] != "ok":
        raise ValueError(f"姿态缓存未完成：{summary_path}")
    source = summary_path.parent / "pose.npz"
    label = source_summary["label"]
    split, video_id = label["sample_id"].split("/", 1)
    if split not in ("train", "valid", "test") or Path(video_id).name != video_id:
        raise ValueError("非法 sample_id")
    destination = output / split / video_id
    result_path = destination / "summary.json"
    signature = {"pose_sha256": file_hash(source), "summary_sha256": file_hash(summary_path), "config": config}
    if resume and result_path.is_file():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if (old.get("signature") == signature and (destination / "normalized.npz").is_file()
                and (destination / "windows.npz").is_file()):
            print(f"NORMALIZED_REUSED {label['sample_id']}")
            return old
        raise ValueError(f"现有结果与本次配置不一致，请用新的 --output：{destination}")
    if destination.exists() and any(destination.iterdir()):
        if not resume or not {p.name for p in destination.iterdir()}.issubset({"normalized.npz", "windows.npz"}):
            raise FileExistsError(f"输出目录非空，请加 --resume 或使用新目录：{destination}")
    with np.load(source, allow_pickle=False) as cache:
        raw = {name: cache[name] for name in ("keypoints", "scores", "valid", "timestamp_s", "segment_id", "frame_index")}
    normalized = normalize_pose(raw["keypoints"], raw["scores"], raw["valid"], raw["timestamp_s"], raw["segment_id"],
                                GeometryConfig(**config["geometry"]))
    windows = make_windows(normalized, raw["timestamp_s"], raw["segment_id"], **config["windows"])
    windows["source_frame_index"] = raw["frame_index"][windows["source_index"]]
    normalized.update({"timestamp_s": raw["timestamp_s"], "segment_id": raw["segment_id"], "frame_index": raw["frame_index"]})
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "normalized.npz", **normalized)
    np.savez_compressed(destination / "windows.npz", **windows)
    cache_path = (destination / "windows.npz").resolve()
    try:
        cache_path_text = cache_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        cache_path_text = str(cache_path)
    entries = [{"window_id": f"{label['sample_id']}/{i:05d}", "sample_id": label["sample_id"],
                "official_split": label["official_split"], "action": label["action"], "class_id": label["class_id"],
                "cache_path": cache_path_text, "window_index": i, "segment_id": int(windows["segment_id"][i]),
                "time_start_s": float(grid[0]), "time_end_s": float(grid[-1])}
               for i, grid in enumerate(windows["time_grid_s"])]
    example = None
    candidates = np.flatnonzero(normalized["valid"][:, 9] & ~normalized["geometry_carried"])
    if len(candidates):
        i = int(candidates[0])
        example = {"joint": "left_wrist", "source_frame": int(raw["frame_index"][i]),
                   "raw_xy_px": raw["keypoints"][i, 9].tolist(),
                   "hip_center_xy_px": normalized["center_xy_px"][i].tolist(),
                   "torso_scale_px": float(normalized["torso_scale_px"][i]),
                   "normalized_xy": normalized["features"][i, 9, :2].tolist()}
    result = {"status": "ok", "sample_id": label["sample_id"], "label": label,
              "frames": len(raw["timestamp_s"]), "window_count": len(entries),
              "normalized_valid_joint_ratio": float(normalized["valid"].mean()),
              "geometry_unavailable_frames": int((~normalized["geometry_valid"]).sum()),
              "geometry_carried_frames": int(normalized["geometry_carried"].sum()),
              "window_shape": list(windows["x"].shape), "signature": signature,
              "numpy_version_used": np.__version__, "example": example, "windows": entries}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"NORMALIZED_OK {label['sample_id']} windows={len(entries)} shape={windows['x'].shape}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/pose_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/normalized_v1")
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train"])
    parser.add_argument("--window-size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--short-gap-s", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.short_gap_s <= .15:
        parser.error("short-gap-s必须位于[0,0.15]")
    from fitness_pose.normalization import GeometryConfig
    config = {"schema": 1, "geometry": asdict(GeometryConfig()),
              "windows": {"window_size": args.window_size, "stride": args.stride, "fps": args.fps,
                          "max_age_s": 0.15, "min_valid_ratio": 0.7, "min_joints_per_frame": 6},
              "channels": ["hip_relative_x", "hip_relative_y", "confidence"],
              "split_policy": "retain_official_video_split; no random window split"}
    if args.short_gap_s:
        config["windows"]["short_gap_s"] = args.short_gap_s
    summaries = sorted(path for split in args.splits for path in (args.input / split).glob("*/summary.json"))
    missing_splits = [split for split in args.splits if not any(path.parent.parent.name == split for path in summaries)]
    if missing_splits:
        raise FileNotFoundError(f"这些集合没有已完成的视频缓存：{missing_splits}；先运行 pose_video.py")
    results = [process_video(path, args.output, config, args.resume) for path in summaries]
    entries = [entry for result in results for entry in result["windows"]]
    if not entries:
        raise ValueError("没有合格窗口；查看各视频 summary 的缺失与跟踪情况")
    counts = Counter((entry["official_split"], entry["action"]) for entry in entries)
    inventory = {"status": "ok", "videos": len(results), "windows": len(entries), "config": config,
                 "window_counts": {split: {action: counts[split, action] for action in ("squat", "push_up", "jump_jack")}
                                   for split in args.splits},
                 "videos_without_windows": [result["sample_id"] for result in results if not result["window_count"]],
                 "training_status": "not_started; validation/source grouping still required"}
    # 不同集合组合使用不同清单名，避免一次 train-only 运行覆盖已有 train+valid 索引。
    scope = "_".join(sorted(set(args.splits)))
    index_path = args.output / f"windows_index_{scope}.jsonl"
    index_path.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
    (args.output / f"dataset_summary_{scope}.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(f"index: {index_path}")
    print("ACTION_WINDOWS_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
