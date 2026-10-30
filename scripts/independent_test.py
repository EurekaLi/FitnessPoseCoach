"""冻结v6候选、按来源清单补骨架、显式启动独立测试；不训练也不搜索阈值。

三个子命令分开，freeze不解码测试视频。测试入口复用replay_one的在线循环，
只改变数据定位和事后汇总，不用真实动作/次数驱动分类或计数。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from replay_action import file_hash, replay_one, write_json

CLASSES = ("squat", "push_up", "jump_jack")
EXPECTED_COUNTS = {"squat": 18, "push_up": 19, "jump_jack": 24}
PACKAGES = ("torch", "numpy", "opencv-python", "onnxruntime-gpu", "rtmlib")
ASSIGNMENTS = "data/processed/repcount_a/source_split_v1/video_assignments.jsonl"
INDEX = "data/processed/repcount_a/source_split_v1/windows_index.jsonl"
MANIFEST = "data/processed/repcount_a/manifest.jsonl"
CHECKPOINT = "outputs/action_stgcnpp_v1/best.pt"
V6_CONFIG = "outputs/count_repaired_v6/config.json"
POSE_REFERENCE = "data/processed/repcount_a/pose_v1/train/stu10_30/summary.json"
NORMALIZATION_REFERENCE = "data/processed/repcount_a/normalized_v1/valid/stu10_67/summary.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def runtime_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def select_test_records(assignments: list[dict], manifest: list[dict], expected_counts=None) -> list[dict]:
    """按training_split选视频，保留原sample_id；来源隔离不等同于人物身份隔离。"""
    originals = {r["sample_id"]: r for r in manifest}
    if len(originals) != len(manifest):
        raise ValueError("原始清单包含重复sample_id")
    seen, source_splits, selected = set(), {}, []
    for row in assignments:
        sample, split, source = row["sample_id"], row["training_split"], row["source_group"]
        parts = sample.split("/")
        if (len(parts) != 2 or parts[0] not in ("train", "valid", "test")
                or not parts[1] or parts[1] in (".", "..") or "\\" in sample):
            raise ValueError(f"非法sample_id：{sample}")
        if sample in seen or split not in ("train", "valid", "test") or not source:
            raise ValueError(f"重复视频或非法分组：{sample}")
        seen.add(sample)
        if source in source_splits and source_splits[source] != split:
            raise ValueError(f"原视频来源跨集合：{source}")
        source_splits[source] = split
        original = originals.get(sample)
        if original is None or any(row.get(key) != value for key, value in original.items()):
            raise ValueError(f"来源分组记录与原始清单不一致：{sample}")
        if row["class_id"] not in range(3) or CLASSES[row["class_id"]] != row["action"]:
            raise ValueError(f"类别编号不匹配：{sample}")
        if split == "test":
            selected.append(row)
    if expected_counts is not None and dict(Counter(r["action"] for r in selected)) != expected_counts:
        raise ValueError("测试视频数量与预期18/19/24不同；停止，不自动重分数据")
    return sorted(selected, key=lambda r: (r["class_id"], r["sample_id"]))


def verify_release(release: Path, *, check_environment: bool = True) -> dict:
    path = release / "release.json"
    if file_hash(path) != (release / "release.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("冻结清单已改变，请勿直接编辑release.json")
    frozen = read_json(path)
    for name, expected in frozen["files"].items():
        if file_hash(release / "snapshot" / name) != expected or file_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f"冻结文件与当前工作区不一致：{name}；不要测试后悄悄换版本")
    if check_environment and frozen["environment"] != runtime_versions():
        raise ValueError("Python或关键推理包版本与冻结时不同，请使用原fitness-pose环境")
    return frozen


def freeze(release: Path) -> int:
    from fitness_pose.counting import CountConfig
    from fitness_pose.normalization import GeometryConfig
    from fitness_pose.streaming import ActionFilterConfig

    if release.exists():
        verify_release(release)
        print(f"CANDIDATE_FROZEN_REUSED {release}")
        return 0
    v6 = read_json(PROJECT_ROOT / V6_CONFIG)
    training = read_json(PROJECT_ROOT / "outputs/action_stgcnpp_v1/config.json")
    for name, expected in ((CHECKPOINT, v6["checkpoint_sha256"]),
                           (ASSIGNMENTS, v6["assignments_sha256"]), (INDEX, v6["window_index_sha256"])):
        if file_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f"与v6候选不一致：{name}")
    # replay_action只增加了缓存根目录/冻结配置注入；算法模块仍须与实测v6逐字节一致。
    for name, expected in v6["code_sha256"].items():
        if name != "scripts/replay_action.py" and file_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f"v6在线算法实现已改变：{name}")
    if (training["smoke_only"] or training["classes"] != list(CLASSES)
            or training["input_shape"] != ["B", 3, 48, 17]
            or training["index_sha256"] != v6["window_index_sha256"]
            or training["code_sha256"]["src/fitness_pose/stgcnpp.py"]
            != file_hash(PROJECT_ROOT / "src/fitness_pose/stgcnpp.py")):
        raise ValueError("训练模型、输入格式或来源索引不匹配")
    selected = select_test_records(read_jsonl(PROJECT_ROOT / ASSIGNMENTS),
                                   read_jsonl(PROJECT_ROOT / MANIFEST), EXPECTED_COUNTS)
    preprocessing = read_json(PROJECT_ROOT / NORMALIZATION_REFERENCE)["signature"]["config"]
    if preprocessing["geometry"] != asdict(GeometryConfig()) or v6["short_gap_s"] != .15:
        raise ValueError("当前前处理与v6约定不同")
    preprocessing["windows"]["short_gap_s"] = v6["short_gap_s"]
    if any(preprocessing["windows"][key] != expected for key, expected in
           (("window_size", 48), ("stride", 10), ("fps", 20.0), ("max_age_s", .15),
            ("min_valid_ratio", .7), ("min_joints_per_frame", 6))):
        raise ValueError("窗口规则与当前模型/v6约定不同")
    # 把默认值也展开保存，避免只冻结JSON覆盖项却遗漏实际生效参数。
    counter = asdict(CountConfig(**v6["counter"]))
    action_filter = asdict(ActionFilterConfig(**v6["filter"]))
    pose_config = read_json(PROJECT_ROOT / POSE_REFERENCE)["signature"]["config"]
    pose_config["preview"] = False
    environment = runtime_versions()
    if any(environment["packages"][name] is None for name in PACKAGES):
        raise ValueError("关键依赖不完整，请在现有fitness-pose环境冻结")
    if any(environment["packages"][name] != pose_config[name] for name in ("rtmlib", "onnxruntime-gpu")):
        raise ValueError("姿态推理包与已有缓存不同，不自动混用两个版本")
    if any(pose_config[key] != value for key, value in
           (("sample_fps", 20.0), ("keypoint_threshold", .3), ("detector_score_threshold", .5),
            ("tracking_iou", .1), ("tracking_max_gap_s", .5))):
        raise ValueError("姿态提取配置与当前v6约定不同")
    model_paths = {}
    for key, folder in (("detector", "yolox_tiny"), ("pose", "rtmpose_m")):
        matches = [p for p in (PROJECT_ROOT / "models/pretrained" / folder).glob("*.onnx")
                   if file_hash(p) == pose_config[f"{key}_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"没有找到唯一的冻结姿态模型：{key}")
        model_paths[key] = matches[0].relative_to(PROJECT_ROOT).as_posix()
    names = {p.relative_to(PROJECT_ROOT).as_posix() for folder in ("scripts", "src")
             for p in (PROJECT_ROOT / folder).rglob("*.py")}
    names.update((CHECKPOINT, ASSIGNMENTS, INDEX, MANIFEST, V6_CONFIG, POSE_REFERENCE,
                  NORMALIZATION_REFERENCE, "configs/counting_repaired_v6.json",
                  "outputs/action_stgcnpp_v1/config.json", *model_paths.values()))
    fingerprints = {name: file_hash(PROJECT_ROOT / name) for name in sorted(names)}
    release.mkdir(parents=True, exist_ok=False)
    for name in fingerprints:
        target = release / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, target)
    frozen = {"schema": 1, "candidate": "counting_repaired_v6", "created_utc": datetime.now(timezone.utc).isoformat(),
              "files": fingerprints, "environment": environment, "checkpoint": CHECKPOINT,
              "checkpoint_epoch": v6["checkpoint_epoch"], "classes": list(CLASSES),
              "preprocessing": preprocessing, "filter": action_filter, "counter": counter,
              "short_gap_s": v6["short_gap_s"], "pose_config": pose_config, "pose_models": model_paths,
              "samples": selected, "expected_counts": EXPECTED_COUNTS,
              "test_evaluated": False,
              "notes": ["仅冻结候选与已有标注清单，未解码/预测测试视频", "按原视频来源隔离，不宣称人物身份隔离",
                        "v6验证计数MAE约6.78次，冻结不代表已经达到实用精度",
                        "回放入口仅增加缓存/配置注入，原train/valid默认调用保持不变"]}
    write_json(release / "release.json", frozen)
    (release / "release.sha256").write_text(file_hash(release / "release.json") + "\n", encoding="utf-8")
    verify_release(release)
    print(f"CANDIDATE_FROZEN videos={len(selected)} counts={EXPECTED_COUNTS} release={release}")
    return 0


def compatible_pose(summary: dict, record: dict, config: dict, source: Path) -> bool:
    """preview只影响绘图，不影响骨架；其余推理参数和视频元数据必须匹配。"""
    signature = summary.get("signature", {})
    actual = dict(signature.get("config", {}))
    expected = dict(config)
    actual.pop("preview", None)
    expected.pop("preview", None)
    stat = source.stat()
    return (summary.get("status") == "ok" and summary.get("label") == record
            and signature.get("record") == record and actual == expected
            and signature.get("source_size") == stat.st_size
            and signature.get("source_mtime_ns") == stat.st_mtime_ns)


def bind_work(work: Path, release: Path) -> str:
    digest = file_hash(release / "release.json")
    path = work / "candidate.json"
    if path.exists() and read_json(path) != {"release_sha256": digest}:
        raise ValueError("工作目录属于其他候选，请使用新的--work目录")
    work.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, {"release_sha256": digest})
    return digest


def prepare(release: Path, work: Path) -> int:
    import pose_video

    frozen = verify_release(release)
    digest = bind_work(work, release)
    originals = {r["sample_id"]: r for r in read_jsonl(release / "snapshot" / MANIFEST)}
    config = frozen["pose_config"]
    args = SimpleNamespace(output=work / "pose", resume=True, preview=False,
                           sample_fps=config["sample_fps"], keypoint_threshold=config["keypoint_threshold"])
    detector = pose_model = model_info = None
    results = []
    for assigned in frozen["samples"]:
        sample = assigned["sample_id"]
        record = originals[sample]
        source = PROJECT_ROOT / record["video_path"]
        destination = args.output / sample
        try:
            summary_path = destination / "summary.json"
            old_cache = PROJECT_ROOT / "data/processed/repcount_a/pose_v1" / sample
            if summary_path.is_file():
                if not compatible_pose(read_json(summary_path), record, config, source) or not (destination / "pose.npz").is_file():
                    raise ValueError("测试缓存不完整或与冻结配置不符，请保留现场并换工作目录")
                method = "prepared_reused"
            elif (old_cache / "summary.json").is_file() and (old_cache / "pose.npz").is_file() and compatible_pose(
                    read_json(old_cache / "summary.json"), record, config, source):
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError("目标有未完成文件，不自动覆盖")
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_cache / "pose.npz", destination / "pose.npz")
                shutil.copy2(old_cache / "summary.json", summary_path)
                method = "existing_pose_reused"
            else:
                if detector is None:
                    # 本环境须先加载torch提供CUDA/cuDNN库；不下载模型，也不静默改用CPU。
                    import torch
                    import onnxruntime as ort
                    from rtmlib import RTMPose, YOLOX
                    from pose_image import inspect_session
                    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
                        raise RuntimeError("需要已配置的fitness-pose CUDA环境")
                    new_detector = YOLOX(str(release / "snapshot" / frozen["pose_models"]["detector"]),
                                         model_input_size=(416, 416), score_thr=config["detector_score_threshold"],
                                         device="cuda", backend="onnxruntime")
                    new_pose = RTMPose(str(release / "snapshot" / frozen["pose_models"]["pose"]),
                                      model_input_size=(192, 256), to_openpose=False, device="cuda", backend="onnxruntime")
                    model_info = {"detector": inspect_session(new_detector, (416, 416), "cuda"),
                                  "pose": inspect_session(new_pose, (256, 192), "cuda")}
                    detector, pose_model = new_detector, new_pose
                pose_video.extract_one(record, args, detector, pose_model, model_info, config)
                method = "new_pose_extracted"
            results.append({"sample_id": sample, "status": "ok", "method": method,
                            "pose_sha256": file_hash(destination / "pose.npz"),
                            "summary_sha256": file_hash(summary_path)})
            print(f"TEST_POSE_READY {sample} method={method}", flush=True)
        except Exception as error:
            results.append({"sample_id": sample, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"TEST_POSE_FAILED {sample}: {error}", flush=True)
    failures = [r for r in results if r["status"] != "ok"]
    write_json(work / "prepare_report.json", {"status": "incomplete" if failures else "ok",
               "release_sha256": digest, "videos": len(results), "failures": failures, "results": results,
               "action_test_evaluated": False})
    print(f"{'INDEPENDENT_TEST_POSE_INCOMPLETE' if failures else 'INDEPENDENT_TEST_POSE_READY'} videos={len(results)} failures={len(failures)}")
    return 1 if failures else 0


def aggregate(results: list[dict]) -> dict:
    from fitness_pose.action_data import classification_metrics

    labels, predictions, video_labels, video_predictions = [], [], [], []
    for row in results:
        for pred, amount in row["raw_prediction_counts"].items():
            labels.extend([row["label"]] * amount)
            predictions.extend([int(pred)] * amount)
        if row["video_prediction"] is not None:
            video_labels.append(row["label"])
            video_predictions.append(row["video_prediction"])
    total = sum(r["samples"] for r in results)
    ready = sum(r["ready_samples"] for r in results)
    correct = sum(r["correct_ready_samples_by_clip_label"] for r in results)
    errors = [r["absolute_count_error"] for r in results if r["absolute_count_error"] is not None]
    return {"raw_windows": len(labels), "raw_window_metrics_by_clip_label": classification_metrics(labels, predictions),
            "video_metrics_on_predictable_videos": classification_metrics(video_labels, video_predictions),
            "video_prediction_coverage": len(video_labels) / len(results) if results else None,
            "video_accuracy_all_processed_videos": sum(a == b for a, b in zip(video_labels, video_predictions)) / len(results) if results else None,
            "videos_without_windows": [r["sample_id"] for r in results if not r["windows"]],
            "ready_coverage": ready / total if total else None,
            "ready_clip_label_agreement": correct / ready if ready else None,
            "all_samples_clip_label_agreement": correct / total if total else None,
            "actual_geometry_missing_samples": sum(r["actual_geometry_missing_samples"] for r in results),
            "count_evaluated_videos": len(errors), "count_mae": sum(errors) / len(errors) if errors else None,
            "count_within_one_ratio": sum(e <= 1 for e in errors) / len(errors) if errors else None,
            "count_by_action": {action: count_summary([r for r in results if r["reference_action"] == action]) for action in CLASSES}}


def count_summary(rows: list[dict]) -> dict:
    errors = [r["absolute_count_error"] for r in rows if r["absolute_count_error"] is not None]
    return {"videos": len(rows), "count_evaluated_videos": len(errors),
            "mae": sum(errors) / len(errors) if errors else None,
            "within_one_ratio": sum(e <= 1 for e in errors) / len(errors) if errors else None}


def evaluate(release: Path, work: Path, output: Path, confirmed: bool) -> int:
    if not confirmed:
        raise ValueError("这将正式查看预留测试成绩；请显式添加--confirm-held-out-test")
    import torch
    import numpy as np
    from fitness_pose.counting import CountConfig
    from fitness_pose.streaming import ActionFilterConfig
    from fitness_pose.stgcnpp import STGCNPP

    frozen = verify_release(release)
    digest = bind_work(work, release)
    prepared = read_json(work / "prepare_report.json")
    expected_ids = [r["sample_id"] for r in frozen["samples"]]
    if (prepared["release_sha256"] != digest or prepared["status"] != "ok"
            or [r["sample_id"] for r in prepared["results"]] != expected_ids):
        raise ValueError("61段测试骨架尚未全部准备成功，先查看prepare_report.json；不评价挑选后的子集")
    for row in prepared["results"]:
        folder = work / "pose" / row["sample_id"]
        if (file_hash(folder / "pose.npz") != row["pose_sha256"]
                or file_hash(folder / "summary.json") != row["summary_sha256"]):
            raise ValueError(f"准备完成后缓存发生变化：{row['sample_id']}")
    if output.exists():
        raise FileExistsError(f"不覆盖已有测试结果，请保留并指定新的--output：{output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；请激活fitness-pose")
    checkpoint = torch.load(release / "snapshot" / frozen["checkpoint"], map_location="cpu", weights_only=True)
    training = checkpoint["config"]
    if (training["smoke_only"] or training["classes"] != list(CLASSES)
            or training["input_shape"] != ["B", 3, 48, 17]
            or training["index_sha256"] != frozen["files"][INDEX]
            or training["code_sha256"]["src/fitness_pose/stgcnpp.py"] != frozen["files"]["src/fitness_pose/stgcnpp.py"]):
        raise ValueError("冻结检查点内部配置不匹配")
    model = STGCNPP(dropout=training["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to("cuda").eval()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros(1, 3, 48, 17, device="cuda"))
    torch.cuda.synchronize()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"release_sha256": digest, "environment": runtime_versions(),
               "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "preprocessing": frozen["preprocessing"], "filter": frozen["filter"], "counter": frozen["counter"],
               "short_gap_s": frozen["short_gap_s"], "samples": expected_ids})
    args = SimpleNamespace(device="cuda", count=True, render=False, short_gap_s=frozen["short_gap_s"])
    results, failures = [], []
    for record in frozen["samples"]:
        try:
            result = replay_one(record, args, model, torch, ActionFilterConfig(**frozen["filter"]),
                                CountConfig(**frozen["counter"]), output, pose_root=work / "pose",
                                preprocessing_config=frozen["preprocessing"])
            predictions = read_jsonl(output / record["sample_id"] / "predictions.jsonl")
            # 整段概率平均仅用于事后视频级指标，不回流到在线动作门控或次数预测。
            result["video_prediction"] = int(np.mean([r["raw_probabilities"] for r in predictions], axis=0).argmax()) if predictions else None
            write_json(output / record["sample_id"] / "summary.json", result)
            results.append(result)
        except Exception as error:
            failure = {"sample_id": record["sample_id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            print(f"TEST_REPLAY_FAILED {record['sample_id']}: {error}", flush=True)
        # 中断或单视频失败不应留下一个看似完整、实则只包含容易样本的总成绩。
        report = {"status": "incomplete" if failures or len(results) != len(expected_ids) else "ok",
                  "release_sha256": digest, "split": "test", "videos_expected": len(expected_ids),
                  "videos_processed": len(results), "test_evaluated": True,
                  "metrics": aggregate(results) if len(results) == len(expected_ids) and not failures else None,
                  "failures": failures, "results": results,
                  "notes": ["按来源隔离的61段子集，非原官方test60段，也非已验证的人物隔离",
                            "无窗口视频仍纳入计数与全视频准确率分母；无次数标注不填0",
                            "整段标签一致率不是逐帧精确标注准确率；缓存回放不代表摄像头端到端速度",
                            "测试后根据本报告调参，须披露这批数据已参与开发"]}
        write_json(output / "report.json", report)
    success = not failures and len(results) == len(expected_ids)
    if success:
        metrics = report["metrics"]
        print(f"TEST_METRICS video_accuracy_all={metrics['video_accuracy_all_processed_videos']:.4f} "
              f"ready_coverage={metrics['ready_coverage']:.4f} count_mae={metrics['count_mae']} "
              f"within_one={metrics['count_within_one_ratio']}", flush=True)
        for action, scores in metrics["count_by_action"].items():
            print(f"TEST_COUNT {action} videos={scores['videos']} mae={scores['mae']} "
                  f"within_one={scores['within_one_ratio']}", flush=True)
    print(f"{'INDEPENDENT_TEST_OK' if success else 'INDEPENDENT_TEST_INCOMPLETE'} videos={len(results)}/{len(expected_ids)} report={output / 'report.json'}")
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "evaluate"))
    parser.add_argument("--release", type=Path, default=PROJECT_ROOT / "releases/v6_test_v1")
    parser.add_argument("--work", type=Path, default=PROJECT_ROOT / "outputs/independent_test_v6")
    parser.add_argument("--output", type=Path, help="评价输出目录；默认work/evaluation，不覆盖已有结果")
    parser.add_argument("--confirm-held-out-test", action="store_true", help="明确正式查看测试成绩，不用于选参数")
    args = parser.parse_args()
    release, work = args.release.resolve(), args.work.resolve()
    if args.command == "freeze":
        return freeze(release)
    if args.command == "prepare":
        return prepare(release, work)
    return evaluate(release, work, (args.output or work / "evaluation").resolve(), args.confirm_held_out_test)


if __name__ == "__main__":
    raise SystemExit(main())
