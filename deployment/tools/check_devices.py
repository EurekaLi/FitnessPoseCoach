"""部署前检查：可选摄像头取帧、ORT真实GPU运算及原生TensorRT微型引擎。

不读取独立测试视频，不导出/修改训练模型，不保存相机图像。文件放deployment
而不是冻结的scripts/src中，避免为设备配置改变已选定候选的指纹。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_dlls():
    """进程内寻找已有CUDA及venv运行库，不修改系统PATH/注册表或全局安装。"""
    handles = []
    if os.name == "nt":
        cuda_root = os.environ.get("CUDA_PATH")
        candidates = [Path(sys.prefix) / "Lib/site-packages/nvidia/cublas/bin"]
        if cuda_root:
            candidates.extend((Path(cuda_root) / "bin/x64", Path(cuda_root) / "bin"))
        # 保留句柄到检查结束，否则Windows可能提前移除DLL搜索目录。
        for path in candidates:
            if path.is_dir():
                handles.append(os.add_dll_directory(str(path)))
    import onnxruntime as ort
    ort.preload_dlls(directory="")
    return handles


def check_camera(index: int, frames: int) -> dict:
    import cv2
    import numpy as np

    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_V4L2,)
    attempts = []
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                attempts.append({"backend_id": backend, "error": "open_failed"})
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            started, successful, shape = time.perf_counter(), 0, None
            for _ in range(frames):
                success, frame = capture.read()
                if not success or frame is None or not frame.size or not np.isfinite(frame).all():
                    break
                successful += 1
                shape = list(frame.shape)
            if successful == frames:
                return {"status": "ok", "index": index, "backend": capture.getBackendName(),
                        "frames_read": successful, "shape_hwc": shape,
                        "reported_fps": capture.get(cv2.CAP_PROP_FPS),
                        "read_seconds": time.perf_counter() - started,
                        "images_saved": False,
                        "note": "短时实际取帧通过；不等于长期稳定性或整条流水线帧率验收"}
            attempts.append({"backend_id": backend, "error": "incomplete_read", "frames": successful})
        finally:
            capture.release()
    raise RuntimeError(f"摄像头{index}取帧失败：{attempts}；请确认已接入且没有被其他应用独占")


def check_ort_gpu() -> dict:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort

    # 用MatMul而非Identity，避免被优化成直接返回输入，误把CPU通过当成GPU可用。
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "device_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    output_root = PROJECT_ROOT / "outputs/deployment_environment"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ort_check_", dir=output_root) as directory:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "gpu_profile")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": 0, "gpu_mem_limit": 64 * 1024 * 1024,
                                                  "cudnn_conv_algo_search": "HEURISTIC"})])
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA会话没有成功启用：{session.get_providers()}")
        result = session.run(None, {"x": np.ones((1, 4), dtype=np.float32)})[0]
        np.testing.assert_allclose(result, np.full((1, 4), 4., dtype=np.float32), rtol=0, atol=1e-6)
        profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        del session
        executed = [row.get("args", {}) for row in profile if row.get("cat") == "Node"]
        if not any(row.get("provider") == "CUDAExecutionProvider" and row.get("op_name") == "MatMul" for row in executed):
            raise RuntimeError("性能记录没有CUDA MatMul执行证据，不报告GPU检查通过")
    return {"status": "ok", "version": ort.__version__, "operation": "MatMul", "output": result.tolist(),
            "executed_provider": "CUDAExecutionProvider", "cpu_fallback_disabled": True}


def check_tensorrt() -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    # TensorRT 11使用强类型网络，直接指定半精度张量，不沿用已移除的全局FP16开关。
    with trt.Builder(logger) as builder, builder.create_network(0) as network, builder.create_builder_config() as config:
        x = network.add_input("x", trt.float16, (1, 4))
        layer = network.add_elementwise(x, x, trt.ElementWiseOperation.SUM)
        layer.get_output(0).name = "y"
        network.mark_output(layer.get_output(0))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * 1024 * 1024)
        config.builder_optimization_level = 0
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT微型半精度引擎构建失败")
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized)
            if engine is None:
                raise RuntimeError("TensorRT引擎反序列化失败")
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("TensorRT GPU执行上下文创建失败")
            del context, engine
    return {"status": "ok", "version": trt.__version__, "native_api": True,
            "fp16_toy_engine_built": True, "engine_deserialized": True, "execution_context_created": True,
            "actual_inference_executed": False,
            "note": "仅验证原生TensorRT构建/加载，不声称ORT TensorRT EP兼容、真实模型FP16运行或加速效果"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--gpu", action="store_true", help="运行微型ONNX GPU算子，不跑姿态或动作模型")
    parser.add_argument("--tensorrt", action="store_true", help="构建微型半精度引擎，不构建真实模型引擎")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/deployment_environment/windows_runtime.json")
    args = parser.parse_args()
    if args.frames < 1 or args.camera_index < 0:
        parser.error("frames必须为正数，camera-index不能为负")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(), "checks": {},
              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
              "test_dataset_used": False, "camera_images_saved": False}
    dll_handles = []
    checks = {}
    if args.camera:
        checks["camera"] = lambda: check_camera(args.camera_index, args.frames)
    if args.gpu:
        checks["onnxruntime_cuda"] = check_ort_gpu
    if args.tensorrt:
        checks["native_tensorrt"] = check_tensorrt
    try:
        if args.gpu or args.tensorrt:
            dll_handles = prepare_dlls()
        for name, check in checks.items():
            try:
                report["checks"][name] = check()
                print(f"DEVICE_CHECK_OK {name} {json.dumps(report['checks'][name], ensure_ascii=False)}", flush=True)
            except Exception as error:
                report["checks"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                print(f"DEVICE_CHECK_FAILED {name}: {error}", flush=True)
    except Exception as error:
        report["initialization_error"] = f"{type(error).__name__}: {error}"
    finally:
        for handle in dll_handles:
            handle.close()
    report["status"] = "ok" if checks and not report.get("initialization_error") and all(
        row["status"] == "ok" for row in report["checks"].values()) and len(report["checks"]) == len(checks) else "incomplete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"DEPLOYMENT_ENV_{report['status'].upper()} report={args.output}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
