"""真实ST-GCN++导出和跨平台FP32对齐，仅使用冻结来源划分的验证数据。

export在fitness-export执行；verify在Windows部署环境执行且不依赖PyTorch。
不修改冻结源码/模型/计数规则；整段回放复用原replay_one，不另写一套计数算法。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from independent_test import ASSIGNMENTS, INDEX, read_json, read_jsonl, verify_release
from replay_action import file_hash, replay_one, write_json

INPUT_SHAPE = (1, 3, 48, 17)
CLASSES = ("squat", "push_up", "jump_jack")
REPLAY_SAMPLES = ("valid/stu10_67", "valid/stu10_42", "valid/stu1_26")
# FP32转换的工程验收门槛，不是动作准确率；失败时保留结果，不自动放宽。
ATOL, RTOL, PROBABILITY_ATOL = 1e-4, 1e-3, 1e-5


def package_versions() -> dict:
    names = ("torch", "onnx", "onnxscript", "onnx_ir", "onnxruntime-gpu", "numpy")
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": result}


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    exponent = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def compare_logits(reference: np.ndarray, actual: np.ndarray) -> dict:
    """同时检查数值、概率和类别；相对误差近零时会夸大，验收使用混合容差。"""
    reference, actual = np.asarray(reference), np.asarray(actual)
    if (reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 3
            or len(reference) == 0 or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise ValueError("对照输出必须是同形状、非空且有限的(N,3)分数")
    difference = np.abs(actual - reference)
    top1_equal = reference.argmax(1) == actual.argmax(1)
    probability_error = float(np.abs(softmax(reference) - softmax(actual)).max())
    return {"passed": bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)
                           and top1_equal.all() and probability_error <= PROBABILITY_ATOL),
            "windows": len(reference), "max_absolute_logit_error": float(difference.max()),
            "mean_absolute_logit_error": float(difference.mean()),
            "max_absolute_probability_error": probability_error,
            "top1_agreement": float(top1_equal.mean()), "top1_mismatch_indices": np.flatnonzero(~top1_equal).tolist(),
            "criteria": {"logit_atol": ATOL, "logit_rtol": RTOL, "probability_atol": PROBABILITY_ATOL,
                         "top1_agreement_required": 1.0}}


def validation_records(release: Path) -> list[dict]:
    rows = read_jsonl(release / "snapshot" / ASSIGNMENTS)
    return sorted((r for r in rows if r["training_split"] == "valid"), key=lambda r: r["sample_id"])


def load_pose(record: dict, frozen: dict) -> tuple[dict, dict]:
    if record["training_split"] != "valid":
        raise ValueError("转换检查只能读来源验证集，拒绝测试视频")
    directory = PROJECT_ROOT / "data/processed/repcount_a/pose_v1" / record["sample_id"]
    summary = read_json(directory / "summary.json")
    config = dict(summary["signature"]["config"])
    expected = dict(frozen["pose_config"])
    config.pop("preview", None)
    expected.pop("preview", None)
    if (summary["status"] != "ok" or summary["label"]["sample_id"] != record["sample_id"] or config != expected):
        raise ValueError(f"验证骨架与冻结姿态配置不符：{record['sample_id']}")
    with np.load(directory / "pose.npz", allow_pickle=False) as cache:
        raw = {key: cache[key] for key in ("keypoints", "scores", "valid", "timestamp_s", "frame_index", "segment_id")}
    times = raw["timestamp_s"]
    if (len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or raw["keypoints"].shape != (len(times), 17, 2)
            or raw["scores"].shape != (len(times), 17) or raw["valid"].shape != (len(times), 17)):
        raise ValueError(f"验证骨架形状/时间戳异常：{record['sample_id']}")
    provenance = {"sample_id": record["sample_id"], "training_split": "valid", "action": record["action"],
                  "pose_sha256": file_hash(directory / "pose.npz"),
                  "summary_sha256": file_hash(directory / "summary.json")}
    return raw, provenance


def online_windows(raw: dict, frozen: dict) -> list[dict]:
    from fitness_pose.normalization import CausalPoseNormalizer, GeometryConfig
    from fitness_pose.streaming import OnlineWindowBuffer

    normalizer = CausalPoseNormalizer(GeometryConfig(**frozen["preprocessing"]["geometry"]))
    buffer = OnlineWindowBuffer(**frozen["preprocessing"]["windows"])
    result = []
    for index, timestamp in enumerate(raw["timestamp_s"]):
        now, segment = float(timestamp), int(raw["segment_id"][index])
        frame = normalizer.step(raw["keypoints"][index], raw["scores"][index], raw["valid"][index], now, segment)
        result.extend(buffer.step(frame, now, segment, index))
    return result


def collect_inputs(records: list[dict], frozen: dict) -> tuple[np.ndarray, list[dict], list[dict]]:
    """每类按ID取前三段、每段均匀取至多四窗；不根据分类/计数成绩挑选样本。"""
    selected = [r for action in CLASSES for r in [row for row in records if row["action"] == action][:3]]
    # 三段整段对照的全部窗口也进入便携包，让Windows覆盖对应每次动作推理。
    selected = list({r["sample_id"]: r for r in selected + [r for r in records if r["sample_id"] in REPLAY_SAMPLES]}.values())
    arrays, windows_metadata, videos = [], [], []
    for record in selected:
        raw, provenance = load_pose(record, frozen)
        windows = online_windows(raw, frozen)
        if record["sample_id"] in REPLAY_SAMPLES:
            indexes = np.arange(len(windows), dtype=int)
        else:
            indexes = np.unique(np.linspace(0, len(windows) - 1, min(4, len(windows))).astype(int)) if windows else []
        videos.append({**provenance, "available_windows": len(windows), "selected_windows": len(indexes)})
        for index in indexes:
            window = windows[int(index)]
            arrays.append(window["x"])
            windows_metadata.append({"sample_id": record["sample_id"], "class_id": record["class_id"],
                                     "online_window_index": int(index), "window_end_s": float(window["time_grid_s"][-1]),
                                     "emitted_at_s": window["emitted_at_s"], "max_source_age_s": window["max_source_age_s"]})
    if {row["class_id"] for row in windows_metadata} != {0, 1, 2}:
        raise ValueError("验证对齐输入未覆盖三类动作；不自动改用测试数据")
    x = np.ascontiguousarray(np.stack(arrays), dtype=np.float32)
    if x.shape[1:] != INPUT_SHAPE[1:] or not np.isfinite(x).all():
        raise ValueError("导出对齐输入不符合(N,3,48,17)有限浮点格式")
    return x, windows_metadata, videos


class ORTActionClassifier:
    """固定单窗口接口；CUDA图允许必要CPU算子，但不能静默整图退回CPU。"""

    def __init__(self, path: Path, device: str, profile_prefix: Path | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        providers = ["CPUExecutionProvider"] if device == "cpu" else [
            ("CUDAExecutionProvider", {"use_tf32": 0, "cudnn_conv_algo_search": "HEURISTIC",
                                       "gpu_mem_limit": 1024 * 1024 * 1024}), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.session.disable_fallback()
        expected = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
        if self.session.get_providers()[0] != expected:
            raise RuntimeError(f"没有启用请求的{expected}：{self.session.get_providers()}")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "skeleton" or inputs[0].shape != list(INPUT_SHAPE)
                or inputs[0].type != "tensor(float)" or len(outputs) != 1 or outputs[0].name != "logits"
                or outputs[0].shape != [1, 3] or outputs[0].type != "tensor(float)"):
            raise ValueError("ONNX模型接口不是固定FP32 skeleton(1,3,48,17)→logits(1,3)")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape != INPUT_SHAPE or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("动作模型每次只接受有限FP32数组(1,3,48,17)")
        result = self.session.run(["logits"], {"skeleton": np.ascontiguousarray(x)})[0]
        if result.shape != (1, 3) or not np.isfinite(result).all():
            raise ValueError("ONNX输出分数无效")
        return result

    def end_profile(self) -> dict:
        path = Path(self.session.end_profiling())
        profile = read_json(path)
        counts = Counter(row.get("args", {}).get("provider") for row in profile
                         if row.get("cat") == "Node" and row.get("args", {}).get("provider"))
        return {"path": str(path), "node_execution_counts_by_provider": dict(counts),
                "cuda_nodes_executed": counts["CUDAExecutionProvider"] > 0,
                "note": "重复执行次数，不是唯一节点数；存在CUDA节点不等于整图都在GPU"}


def run_logits(predictor, x: np.ndarray) -> np.ndarray:
    return np.concatenate([predictor(row[None]) for row in x], axis=0)


def compare_replays(first: Path, second: Path) -> dict:
    keys = ("timestamp_s", "frame_index", "segment_id", "action", "status", "counts", "count_status",
            "geometry_valid", "signal_type", "side", "side_changed", "window_reset_reason")
    a, b = read_jsonl(first / "timeline.jsonl"), read_jsonl(second / "timeline.jsonl")
    if len(a) != len(b):
        return {"passed": False, "reason": "timeline_length", "samples": [len(a), len(b)]}
    different = [i for i, (left, right) in enumerate(zip(a, b)) if any(left[k] != right[k] for k in keys)]
    events_equal = read_jsonl(first / "count_events.jsonl") == read_jsonl(second / "count_events.jsonl")
    return {"passed": not different and events_equal, "samples": len(a),
            "state_mismatch_indices": different, "count_events_identical": events_equal,
            "compared_fields": list(keys)}


def export_action(release: Path, output: Path) -> int:
    import torch
    import onnx
    from fitness_pose.stgcnpp import STGCNPP
    from fitness_pose.counting import CountConfig
    from fitness_pose.streaming import ActionFilterConfig

    frozen = verify_release(release, check_environment=False)
    if output.exists():
        raise FileExistsError(f"不覆盖已有导出，请指定新的--output：{output}")
    if frozen["classes"] != list(CLASSES):
        raise ValueError("冻结类别顺序不匹配")
    checkpoint = torch.load(release / "snapshot" / frozen["checkpoint"], map_location="cpu", weights_only=True)
    training = checkpoint["config"]
    if (training["smoke_only"] or training["classes"] != list(CLASSES)
            or training["input_shape"] != ["B", 3, 48, 17] or training["index_sha256"] != frozen["files"][INDEX]
            or training["code_sha256"]["src/fitness_pose/stgcnpp.py"] != frozen["files"]["src/fitness_pose/stgcnpp.py"]):
        raise ValueError("检查点内部接口/来源/代码指纹不匹配")
    records = validation_records(release)
    if not set(REPLAY_SAMPLES).issubset({r["sample_id"] for r in records}):
        raise ValueError("预定的三段回放不在来源验证集")
    torch.set_num_threads(2)
    model = STGCNPP(dropout=training["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    def reference(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    x, windows_metadata, videos = collect_inputs(records, frozen)
    expected_logits = run_logits(reference, x)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "action_stgcnpp.onnx"
    report = {"status": "incomplete", "release_sha256": file_hash(release / "release.json"),
              "checkpoint_sha256": frozen["files"][frozen["checkpoint"]], "epoch": checkpoint["epoch"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "environment": package_versions(),
              "exporter_code_sha256": file_hash(Path(__file__)), "classes": list(CLASSES),
              "input": {"name": "skeleton", "shape": list(INPUT_SHAPE), "dtype": "float32",
                        "channels": frozen["preprocessing"]["channels"]},
              "output": {"name": "logits", "shape": [1, 3], "dtype": "float32", "softmax_included": False},
              "opset": 18, "dynamo": True, "preprocessing": frozen["preprocessing"],
              "filter": frozen["filter"], "counter": frozen["counter"],
              "validation_videos": videos, "validation_windows": windows_metadata, "test_dataset_used": False,
              "note": "只转换动作分类网络；归一化、跟踪、分类过滤和计数仍属于应用代码"}
    try:
        torch.onnx.export(model, (torch.from_numpy(x[:1]),), str(path),
                          input_names=["skeleton"], output_names=["logits"], opset_version=18,
                          dynamo=True, external_data=False)
        exported = onnx.load(str(path))
        onnx.helper.set_model_props(exported, {"classes": json.dumps(list(CLASSES)),
            "input_format": "N,C,T,V; hip_relative_x,hip_relative_y,confidence; 20Hz; past48; stride10",
            "release_sha256": report["release_sha256"], "checkpoint_sha256": report["checkpoint_sha256"]})
        onnx.checker.check_model(exported)
        onnx.save(exported, str(path))
        classifier = ORTActionClassifier(path, "cpu")
        actual = run_logits(classifier, x)
        report["cpu_parity"] = compare_logits(expected_logits, actual)
        np.savez_compressed(output / "validation_parity.npz", x=x, torch_logits=expected_logits)
        report["onnx_sha256"] = file_hash(path)
        report["parity_npz_sha256"] = file_hash(output / "validation_parity.npz")
        report["online_replays"] = []
        args = SimpleNamespace(device="cpu", count=True, render=False, short_gap_s=frozen["short_gap_s"])

        def onnx_torch_adapter(tensor):
            # 只适配replay_one要求的张量接口；概率计算/状态/计数完全复用冻结实现。
            return torch.from_numpy(classifier(tensor.detach().cpu().numpy()))

        for sample in REPLAY_SAMPLES:
            record = next(r for r in records if r["sample_id"] == sample)
            common = (record, args)
            left = replay_one(*common, model, torch, ActionFilterConfig(**frozen["filter"]),
                              CountConfig(**frozen["counter"]), output / "replay_torch",
                              preprocessing_config=frozen["preprocessing"])
            right = replay_one(*common, onnx_torch_adapter, torch, ActionFilterConfig(**frozen["filter"]),
                               CountConfig(**frozen["counter"]), output / "replay_onnx_cpu",
                               preprocessing_config=frozen["preprocessing"])
            comparison = compare_replays(output / "replay_torch" / sample, output / "replay_onnx_cpu" / sample)
            report["online_replays"].append({"sample_id": sample, "windows": left["windows"],
                "torch_counts": left["counts"], "onnx_counts": right["counts"], **comparison})
        report["status"] = "ok" if report["cpu_parity"]["passed"] and all(r["passed"] for r in report["online_replays"]) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "export_report.json", report)
    print(f"ACTION_ONNX_{'EXPORT_OK' if report['status'] == 'ok' else 'EXPORT_FAILED'} windows={len(x)} "
          f"max_logit_error={report['cpu_parity']['max_absolute_logit_error']} "
          f"top1={report['cpu_parity']['top1_agreement']} replay_matches={sum(r['passed'] for r in report['online_replays'])}/3 output={output}", flush=True)
    return 0 if report["status"] == "ok" else 1


def verify_action(bundle: Path, device: str, output: Path) -> int:
    manifest = read_json(bundle / "export_report.json")
    if manifest["status"] != "ok" or manifest["test_dataset_used"] or manifest["classes"] != list(CLASSES):
        raise ValueError("只接受已通过CPU及在线回放检查的三分类导出包")
    if (file_hash(bundle / "action_stgcnpp.onnx") != manifest["onnx_sha256"]
            or file_hash(bundle / "validation_parity.npz") != manifest["parity_npz_sha256"]):
        raise ValueError("导出后的ONNX或固定验证输入发生变化")
    if output.exists():
        raise FileExistsError(f"不覆盖已有检查报告，请指定新的--report：{output}")
    handles = []
    try:
        if device == "cuda":
            from check_devices import prepare_dlls
            handles = prepare_dlls()
        output.parent.mkdir(parents=True, exist_ok=True)
        predictor = ORTActionClassifier(bundle / "action_stgcnpp.onnx", device,
                                        output.parent / f"{output.stem}_profile" if device == "cuda" else None)
        with np.load(bundle / "validation_parity.npz", allow_pickle=False) as data:
            x, reference = data["x"], data["torch_logits"]
        actual = run_logits(predictor, x)
        comparison = compare_logits(reference, actual)
        profile = predictor.end_profile() if device == "cuda" else None
        passed = comparison["passed"] and (device != "cuda" or profile["cuda_nodes_executed"])
        report = {"status": "ok" if passed else "failed", "device": device,
                  "environment": package_versions(), "onnx_sha256": manifest["onnx_sha256"],
                  "export_report_sha256": file_hash(bundle / "export_report.json"),
                  "parity": comparison, "profile": profile, "test_dataset_used": False,
                  "note": "对齐/执行后端检查，不是摄像头全链路速度或新一轮动作测试成绩"}
        write_json(output, report)
        print(f"ACTION_ONNX_{'VERIFY_OK' if passed else 'VERIFY_FAILED'} device={device} windows={len(x)} "
              f"max_logit_error={comparison['max_absolute_logit_error']} top1={comparison['top1_agreement']} report={output}", flush=True)
        return 0 if passed else 1
    finally:
        for handle in handles:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("--release", type=Path, default=PROJECT_ROOT / "releases/v6_test_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/action_onnx_v1", help="导出包目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="verify使用的ORT后端")
    parser.add_argument("--report", type=Path, help="verify报告；默认导出目录verify_DEVICE.json")
    args = parser.parse_args()
    if args.command == "export":
        return export_action(args.release.resolve(), args.output.resolve())
    return verify_action(args.output.resolve(), args.device,
                         (args.report or args.output / f"verify_{args.device}.json").resolve())


if __name__ == "__main__":
    raise SystemExit(main())
