"""使用官方示例图验证 YOLOX-tiny + RTMPose-M 的人体姿态推理。

模型与示例缓存到项目目录；默认要求 CUDA 会话可用，不将 CPU 回退报告为
GPU 成功。本入口只产出单图骨架与原图像素坐标，不进行动作分类或计数。
模型来源：https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DETECTOR_URL = MODEL_BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
POSE_URL = MODEL_BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
SAMPLE_URL = "https://raw.githubusercontent.com/Tau-J/rtmlib/main/demo.jpg"
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="本地人体图片；省略时使用官方示例图")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "pose_image")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--download-only", action="store_true", help="只获取模型与示例，不执行推理")
    return parser.parse_args()


def prepare_assets(source: Path | None) -> tuple[Path, Path, Path]:
    """缓存两种模型和输入图；各模型使用独立目录，避免解压文件互相覆盖。"""
    from rtmlib.tools.file import download_checkpoint, download_url_to_file

    model_root = PROJECT_ROOT / "models" / "pretrained"
    detector_path = Path(download_checkpoint(
        DETECTOR_URL, dst_dir=str(model_root / "yolox_tiny"),
    ))
    pose_path = Path(download_checkpoint(
        POSE_URL, dst_dir=str(model_root / "rtmpose_m"),
    ))
    if source is None:
        source = PROJECT_ROOT / "data" / "samples" / "rtmlib_demo.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            download_url_to_file(SAMPLE_URL, str(source))
    else:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"输入图片不存在：{source}")
    return detector_path, pose_path, source


def inspect_session(model, expected_hw: tuple[int, int], device: str) -> dict:
    """确认模型尺寸和实际注册的执行后端，阻止完全退回 CPU 后误报成功。"""
    session = model.session
    session.disable_fallback()
    providers = session.get_providers()
    if device == "cuda" and (not providers or providers[0] != "CUDAExecutionProvider"):
        raise RuntimeError(f"CUDA 会话初始化失败，实际后端：{providers}")
    input_meta = session.get_inputs()[0]
    if tuple(input_meta.shape[-2:]) != expected_hw:
        raise ValueError(f"模型尺寸与预处理不匹配：{input_meta.shape}，期望 H/W={expected_hw}")
    return {"providers": providers, "input_shape": input_meta.shape}


def run_image(args: argparse.Namespace, detector_path: Path, pose_path: Path, source: Path) -> None:
    import cv2
    import numpy as np

    # 双目项目已验证：先导入 torch，让 ORT 复用当前环境的 CUDA/cuDNN 动态库。
    import torch
    import onnxruntime as ort
    from rtmlib import RTMPose, YOLOX, draw_bbox, draw_skeleton

    if args.device == "cuda":
        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("当前环境没有可用的 PyTorch CUDA 或 ORT CUDA 后端")
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"图片无法解码：{source}")

    detector = YOLOX(
        str(detector_path), model_input_size=(416, 416), score_thr=0.5,
        backend="onnxruntime", device=args.device,
    )
    pose_model = RTMPose(
        str(pose_path), model_input_size=(192, 256), to_openpose=False,
        backend="onnxruntime", device=args.device,
    )
    # ORT 输入采用 H/W；RTMPose 预处理接口则接收 W/H，不能混用。
    sessions = {
        "detector": inspect_session(detector, (416, 416), args.device),
        "pose": inspect_session(pose_model, (256, 192), args.device),
    }
    for name, session in sessions.items():
        print(f"{name} providers: {session['providers']}")

    started = time.perf_counter()
    boxes = np.asarray(detector(image), dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        # 上游姿态接口会把空框列表当作整图人体框；这里拒绝伪造一个检测结果。
        raise RuntimeError("未检测到人体；请确认输入是全身可见的人体图片")
    keypoints, scores = pose_model(image, bboxes=boxes)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if keypoints.shape != (len(boxes), 17, 2) or scores.shape != (len(boxes), 17):
        raise ValueError(f"关键点形状异常：{keypoints.shape} / {scores.shape}")
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise ValueError("关键点或置信度包含非有限值")

    report = {
        "status": "ok",
        "source": str(source),
        "sample_url": SAMPLE_URL if args.source is None else None,
        "device_requested": args.device,
        "image_size_hw": list(image.shape[:2]),
        "coordinate_system": "original_image_pixels_xy_top_left_origin",
        "keypoint_format": "coco17",
        "keypoint_names": COCO17_NAMES,
        "person_count": len(boxes),
        "models": {
            "detector": {"name": "YOLOX-tiny", "path": str(detector_path), "url": DETECTOR_URL},
            "pose": {"name": "RTMPose-M", "path": str(pose_path), "url": POSE_URL},
        },
        "sessions": sessions,
        "environment": {name: metadata.version(name) for name in (
            "rtmlib", "torch", "onnxruntime-gpu", "opencv-python", "numpy",
        )},
        "first_image_pipeline_ms": elapsed_ms,
        "timing_note": "单图首次运行，含前后处理；未预热，不作为实时速度基准",
        "gpu_check_scope": "CUDA 会话已注册并完成模型调用；不声称每个算子都在 GPU 上执行",
        "people": [
            {"bbox_xyxy": box.tolist(), "keypoints_xy": points.tolist(), "scores": confidence.tolist()}
            for box, points, confidence in zip(boxes, keypoints, scores)
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "skeleton.jpg"
    report_path = args.output / "keypoints.json"
    annotated = draw_bbox(image.copy(), boxes)
    annotated = draw_skeleton(annotated, keypoints, scores, openpose_skeleton=False, kpt_thr=0.3)
    if not cv2.imwrite(str(image_path), annotated):
        raise OSError(f"无法保存骨架图：{image_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"persons: {len(boxes)}; keypoints: {keypoints.shape}")
    print(f"image: {image_path}")
    print(f"report: {report_path}")
    print("POSE_IMAGE_OK")


def main() -> int:
    args = parse_args()
    try:
        detector_path, pose_path, source = prepare_assets(args.source)
        if args.download_only:
            print(f"detector: {detector_path}\npose: {pose_path}\nimage: {source}")
            print("POSE_ASSETS_READY")
            return 0
        run_image(args, detector_path, pose_path, source)
        return 0
    except Exception as exc:
        print(f"POSE_IMAGE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
