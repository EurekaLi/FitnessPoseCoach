"""Windows单人摄像头ORT基线；Q退出，R清空会话，默认不保存影像。

--video可用公开验证视频检查整条模型链，按视频时间顺序处理；--headless
不打开窗口。离线视频吞吐率不能用于宣称摄像头实时速度。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from action_onnx import PROJECT_ROOT, ORTActionClassifier, file_hash, read_json
from check_devices import prepare_dlls
from frame_source import LatestFrameCapture, VideoFrames
from online_pipeline import load_candidate, OnlineActionPipeline, PersonPose
from replay_action import write_json

WINDOW_TITLE = "FitnessPoseCoach | Q exit | R reset"
EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (11, 5), (12, 6),
         (5, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0))


def percentiles(values) -> dict:
    return {"observations_in_last_window": len(values),
            "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None}


def draw_panel(image, row, pose, fresh: bool, process_rate: float, action_rate: float, latency: float | None):
    import cv2

    rendered = image.copy()
    height, width = rendered.shape[:2]
    if pose is not None and fresh:
        points, valid = pose["keypoints"], pose["valid"]
        visible = valid & np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        for a, b in EDGES:
            if visible[a] and visible[b]:
                cv2.line(rendered, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (80, 230, 80), 2)
        for point in points[visible]:
            cv2.circle(rendered, tuple(np.rint(point).astype(int)), 3, (30, 180, 255), -1)
        if pose["box"] is not None:
            x1, y1, x2, y2 = np.rint(pose["box"]).astype(int)
            cv2.rectangle(rendered, (int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1))),
                          (int(np.clip(x2, 0, width - 1)), int(np.clip(y2, 0, height - 1))), (255, 180, 40), 1)
    action = row["action"] if row and fresh else None
    status = row["status"] if row and fresh else "stale_pose" if row else "warming_up"
    counts = row["counts"] if row else {"squat": 0, "push_up": 0, "jump_jack": 0}
    lines = [f"action={action or 'unknown'}  status={status}  p={row['confidence'] if row and fresh else 0.:.2f}",
             "counts: " + "  ".join(f"{key}={value}" for key, value in counts.items()),
             f"phase={row['count_status'] if row and fresh else 'paused'}  signal={row.get('signal_type') if row else '-'}",
             f"pose={process_rate:.1f} Hz  action={action_rate:.1f} Hz  latency_p95={latency:.0f} ms" if latency is not None
             else f"pose={process_rate:.1f} Hz  action={action_rate:.1f} Hz",
             "Q: exit   R: reset history and counts   |   no recording",
             "EXPERIMENTAL COUNTER - not validated exercise correction"]
    panel = np.full((132, width, 3), 25, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (8, 20 + index * 21), cv2.FONT_HERSHEY_SIMPLEX,
                    min(.47, width / 1400), (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack((panel, rendered))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=PROJECT_ROOT / "releases/v6_test_v1")
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "outputs/action_onnx_v1")
    parser.add_argument("--device-config", type=Path, default=PROJECT_ROOT / "deployment/windows/device_config.json")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--backend", choices=("DSHOW", "MSMF"))
    parser.add_argument("--video", type=Path, help="可选现成视频；不使用相机，不进行实时速度评价")
    parser.add_argument("--headless", action="store_true", help="不打开显示窗口")
    parser.add_argument("--max-frames", type=int, help="处理的骨架采样数上限，用于简短联调")
    parser.add_argument("--max-seconds", type=float, help="应用运行墙钟时长上限，不含加载预热")
    parser.add_argument("--output", type=Path, help="仅保存配置/统计摘要；默认outputs/camera_runs/时间戳")
    args = parser.parse_args()
    if (args.camera_index is not None and args.camera_index < 0 or args.max_frames is not None and args.max_frames < 1
            or args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)):
        parser.error("相机索引不能为负；frames和seconds必须为正")
    output = (args.output or PROJECT_ROOT / "outputs/camera_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise FileExistsError(f"不覆盖已有输出目录：{output}")
    if args.video and not args.video.is_file():
        raise FileNotFoundError(args.video)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "initializing", "mode": "offline_video" if args.video else "live_camera",
               "images_saved": False, "video_saved": False, "test_dataset_evaluated": False,
               "output": str(output), "processed_samples": 0, "classification_windows": 0, "resets": 0,
               "notes": ["只保存配置和不含骨架/影像的汇总统计", "计数原型，不能作为动作质量或医疗建议",
                         "软件延迟不含相机内部曝光/缓冲，初始化预热另计", "离线视频吞吐率不等于摄像头帧率"]}
    handles, source, pipeline = [], None, None
    timing = defaultdict(lambda: deque(maxlen=2000))
    recent_pose, recent_action = deque(maxlen=256), deque(maxlen=64)
    state_counts = Counter()
    initialized_at = time.perf_counter()
    started, initialization_s = None, None
    shown, window_created = False, False
    try:
        import cv2

        frozen, exported = load_candidate(args.release.resolve(), args.bundle.resolve())
        config = read_json(args.device_config)
        camera = {**config["camera"]}
        if args.camera_index is not None:
            camera["index"] = args.camera_index
        if args.backend:
            camera["backend"] = args.backend
        summary.update({"release_sha256": exported["release_sha256"], "onnx_sha256": exported["onnx_sha256"],
                        "pipeline_sha256": file_hash(Path(__file__).with_name("online_pipeline.py")),
                        "camera_config": camera, "environment_python": sys.executable,
                        "preprocessing": frozen["preprocessing"], "filter": frozen["filter"], "counter": frozen["counter"]})
        print("CAMERA_APP_LOADING: loading CUDA models, camera not opened yet...", flush=True)
        handles = prepare_dlls()
        classifier = ORTActionClassifier(args.bundle.resolve() / "action_stgcnpp.onnx", "cuda")
        person = PersonPose(frozen, args.release.resolve())
        classifier(np.zeros((1, 3, 48, 17), dtype=np.float32))
        person.warmup()
        pipeline = OnlineActionPipeline(frozen, classifier)
        summary["sessions"] = person.sessions
        summary["sessions"]["action"] = {"providers": classifier.session.get_providers()}
        source = VideoFrames(args.video.resolve()) if args.video else LatestFrameCapture(
            camera["index"], camera["backend"], camera["requested_width"], camera["requested_height"], camera["requested_fps"])
        if not args.video:
            source.start()
        started = time.perf_counter()
        initialization_s = started - initialized_at
        summary["status"] = "running"
        write_json(output / "config.json", summary)
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            window_created = True
        print(f"CAMERA_APP_READY mode={summary['mode']} | Q exit | R reset | no recording | report={output}", flush=True)
        last_index, next_due, barrier, last_received_at = -1, 0., -math.inf, started
        last_row, last_pose, last_pose_acquired = None, None, -math.inf
        last_console = started
        while True:
            now_wall = time.perf_counter()
            if args.max_seconds is not None and now_wall - started >= args.max_seconds:
                summary["stop_reason"] = "max_seconds"
                break
            packet = source.get_after(last_index)
            key = -1
            if packet is not None:
                last_index, last_received_at = packet.index, time.perf_counter()
                if packet.acquired_at < barrier:
                    continue
                if packet.timestamp_s + 1e-8 >= next_due:
                    next_due = (math.floor((packet.timestamp_s + 1e-8) * frozen["pose_config"]["sample_fps"]) + 1) / frozen["pose_config"]["sample_fps"]
                    process_started = time.perf_counter()
                    pose = person.step(packet.image, packet.timestamp_s)
                    item = pipeline.step(pose["keypoints"], pose["scores"], pose["valid"], packet.timestamp_s,
                                         pose["segment_id"], packet.index)
                    last_row, last_pose, last_pose_acquired = item["timeline"], pose, packet.acquired_at
                    summary["processed_samples"] += 1
                    summary["classification_windows"] += len(item["predictions"])
                    state_counts[last_row["status"]] += 1
                    timing["detector"].append(pose["detector_ms"])
                    if pose["pose_ms"] is not None:
                        timing["pose"].append(pose["pose_ms"])
                    timing["action"].extend(item["action_inference_ms"])
                    timing["sample_processing"].append((time.perf_counter() - process_started) * 1000)
                    recent_pose.append(time.perf_counter())
                    recent_action.extend([time.perf_counter()] * len(item["predictions"]))
                fresh = (packet.timestamp_s - last_row["timestamp_s"] <= frozen["preprocessing"]["windows"]["max_age_s"] + 1e-9
                         if args.video and last_row else time.perf_counter() - last_pose_acquired <= frozen["preprocessing"]["windows"]["max_age_s"])
                if not args.headless:
                    wall = time.perf_counter()
                    for recent in (recent_pose, recent_action):
                        while recent and wall - recent[0] > 5:
                            recent.popleft()
                    span = max(.1, min(5., wall - started))
                    latency = percentiles(timing["display_latency"])["p95_ms"]
                    cv2.imshow(WINDOW_TITLE, draw_panel(packet.image, last_row, last_pose, fresh,
                                                       len(recent_pose) / span, len(recent_action) / span, latency))
                    shown = True
                    key = cv2.waitKey(1) & 0xFF
                    timing["display_latency"].append((time.perf_counter() - packet.acquired_at) * 1000)
                    try:
                        window_visible = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
                    except cv2.error:
                        window_visible = False
                    if not window_visible:
                        summary["stop_reason"] = "window_closed"
                        break
                if args.max_frames is not None and summary["processed_samples"] >= args.max_frames:
                    summary["stop_reason"] = "max_frames"
                    break
            elif source.finished:
                summary["stop_reason"] = "end_of_video" if args.video else "capture_ended"
                break
            elif time.perf_counter() - last_received_at > 3:
                raise TimeoutError("摄像头超过3秒没有新画面，请检查连接或设备占用")
            elif shown:
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                summary["stop_reason"] = "user_exit"
                break
            if key in (ord("r"), ord("R")):
                pipeline.reset()
                person.reset()
                # 排除按键前已采集但尚未处理的旧画面，避免重置后复用旧半周期。
                barrier, next_due = time.perf_counter(), 0.
                last_row, last_pose, last_pose_acquired = None, None, -math.inf
                summary["resets"] += 1
                print("SESSION_RESET: history and counts cleared", flush=True)
            if time.perf_counter() - last_console >= 5:
                print(f"CAMERA_STATE samples={summary['processed_samples']} windows={summary['classification_windows']} "
                      f"status={last_row['status'] if last_row else 'warming_up'} counts={pipeline.counter.counts}", flush=True)
                last_console = time.perf_counter()
        summary["status"] = "ok"
    except KeyboardInterrupt:
        summary["status"], summary["stop_reason"] = "ok", "keyboard_interrupt"
    except Exception as error:
        summary["status"], summary["error"] = "failed", f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if source is not None:
            summary["capture_released"] = source.close()
            summary["capture"] = dict(source.metadata)
            summary["frames_read"] = source.frames_read
            if not summary["capture_released"]:
                summary["status"] = "failed"
                summary["cleanup_warning"] = "相机读取线程未及时结束；本次进程退出后请确认设备可重新打开"
        if window_created:
            import cv2
            cv2.destroyAllWindows()
        for handle in handles:
            handle.close()
        duration = time.perf_counter() - started if started is not None else 0.
        summary.update({"initialization_s": initialization_s, "running_wall_s": duration,
                        "final_counts": pipeline.counter.counts if pipeline else None,
                        "sample_states": dict(state_counts), "timing_last_at_most_2000_observations": {
                            key: percentiles(values) for key, values in timing.items()},
                        "pose_samples_per_wall_s": summary["processed_samples"] / duration if duration else None,
                        "action_windows_per_wall_s": summary["classification_windows"] / duration if duration else None,
                        "torch_imported": "torch" in sys.modules})
        write_json(output / "summary.json", summary)
        print(f"CAMERA_APP_{'OK' if summary['status'] == 'ok' else 'FAILED'} samples={summary['processed_samples']} "
              f"counts={summary['final_counts']} report={output / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
