"""无PyTorch的在线连接层，复用冻结的几何、分类过滤与周期计数实现。

step只接受当前骨架和采集时刻，不接收真实动作标签。缓存重放与摄像头共用
此接口；reset清空整个会话，不能仅把显示次数清零却沿用上一次动作的历史。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from action_onnx import CLASSES, PROJECT_ROOT, ORTActionClassifier, file_hash, read_json, softmax, verify_release
from fitness_pose.normalization import CausalPoseNormalizer, GeometryConfig
from fitness_pose.streaming import ActionFilter, ActionFilterConfig, OnlineWindowBuffer
from fitness_pose.counting import CountConfig, ExerciseCounter
from pose_video import select_person


def load_candidate(release: Path, bundle: Path) -> tuple[dict, dict]:
    frozen = verify_release(release, check_environment=False)
    exported = read_json(bundle / "export_report.json")
    if (exported["status"] != "ok" or exported["classes"] != list(CLASSES)
            or exported["release_sha256"] != file_hash(release / "release.json")
            or exported["onnx_sha256"] != file_hash(bundle / "action_stgcnpp.onnx")
            or any(exported[key] != frozen[key] for key in ("preprocessing", "filter", "counter"))):
        raise ValueError("ONNX导出包与冻结候选不一致，不自动混用模型或规则")
    return frozen, exported


class OnlineActionPipeline:
    def __init__(self, frozen: dict, predictor):
        self.config, self.predictor = frozen, predictor
        self.reset()

    def reset(self):
        config = self.config
        self.normalizer = CausalPoseNormalizer(GeometryConfig(**config["preprocessing"]["geometry"]))
        self.buffer = OnlineWindowBuffer(**config["preprocessing"]["windows"])
        self.filter = ActionFilter(ActionFilterConfig(**config["filter"]))
        self.counter = ExerciseCounter(CountConfig(**config["counter"]))

    def step(self, points, scores, valid, now: float, segment: int, frame_index: int) -> dict:
        frame = self.normalizer.step(points, scores, valid, now, segment)
        windows = self.buffer.step(frame, now, segment, frame_index)
        if self.buffer.last_reset_reason:
            self.filter.reset()
        self.filter.expire(now)
        predictions, inference_ms = [], []
        for item in windows:
            started = time.perf_counter()
            logits = self.predictor(item["x"][None])
            inference_ms.append((time.perf_counter() - started) * 1000)
            if np.asarray(logits).shape != (1, 3) or not np.isfinite(logits).all():
                raise ValueError("动作执行后端必须输出有限分数(1,3)")
            probabilities = softmax(logits)[0]
            self.filter.update(probabilities, now)
            predictions.append({"window_end_s": float(item["time_grid_s"][-1]), "emitted_at_s": now,
                                "raw_prediction": int(probabilities.argmax()),
                                "raw_probabilities": probabilities.tolist(), "max_source_age_s": item["max_source_age_s"]})
        usable = bool(frame["geometry_valid"] and segment >= 0)
        status = self.filter.status if usable else "missing"
        if usable and self.filter.last_prediction is None and status not in ("stale", "confirming"):
            status = "warming_up"
        action = CLASSES[self.filter.action] if usable and self.filter.action is not None else None
        counted = self.counter.step(action, points, scores, valid, now, segment)
        event = {"timestamp_s": now, "frame_index": frame_index, "segment_id": segment,
                 "action": counted["event"], "counts": dict(counted["counts"])} if counted["event"] else None
        row = {"timestamp_s": now, "frame_index": frame_index, "segment_id": segment,
               "action": action, "status": status, "geometry_valid": usable,
               "geometry_carried": bool(frame["geometry_carried"]), "valid_joints": int(frame["valid"].sum()),
               "window_reset_reason": self.buffer.last_reset_reason,
               "confidence": float(self.filter.probabilities.max()) if self.filter.probabilities is not None else 0.,
               "counts": dict(counted["counts"]), "count_status": counted["status"],
               "signal": counted.get("signal"), "signal_type": counted.get("signal_type"),
               "side": counted.get("side"), "side_changed": counted.get("side_changed", False),
               "thresholds": counted.get("thresholds")}
        return {"timeline": row, "predictions": predictions, "event": event, "action_inference_ms": inference_ms}


class PersonPose:
    """沿用pose_video的单目标几何跟踪；不声称识别人物身份。"""

    def __init__(self, frozen: dict, release: Path):
        from rtmlib import YOLOX, RTMPose
        from pose_image import inspect_session

        self.config = frozen["pose_config"]
        self.detector = YOLOX(str(release / "snapshot" / frozen["pose_models"]["detector"]),
                              model_input_size=(416, 416), score_thr=self.config["detector_score_threshold"],
                              device="cuda", backend="onnxruntime")
        self.pose = RTMPose(str(release / "snapshot" / frozen["pose_models"]["pose"]),
                            model_input_size=(192, 256), to_openpose=False, device="cuda", backend="onnxruntime")
        self.sessions = {"detector": inspect_session(self.detector, (416, 416), "cuda"),
                         "pose": inspect_session(self.pose, (256, 192), "cuda")}
        self.reset()

    def reset(self):
        self.previous_box, self.last_seen, self.segment_id = None, -np.inf, -1

    def warmup(self):
        # 在开始采集前完成CUDA初始化，避免把首次模型加载耗时当成摄像头积压。
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector(image)
        self.pose(image, bboxes=np.array([[0, 0, 640, 480]], dtype=np.float32))

    def step(self, image, now: float) -> dict:
        started = time.perf_counter()
        boxes = np.asarray(self.detector(image), dtype=np.float32).reshape(-1, 4)
        detection_ms = (time.perf_counter() - started) * 1000
        box, restarted = select_person(boxes, self.previous_box, self.last_seen, now,
                                        max_gap=self.config["tracking_max_gap_s"])
        points = np.full((17, 2), np.nan, dtype=np.float32)
        scores, valid = np.zeros(17, dtype=np.float32), np.zeros(17, dtype=bool)
        segment, pose_ms = -1, None
        if box is not None:
            started = time.perf_counter()
            keypoints, confidence = self.pose(image, bboxes=np.asarray([box]))
            pose_ms = (time.perf_counter() - started) * 1000
            if keypoints.shape != (1, 17, 2) or confidence.shape != (1, 17):
                raise ValueError("姿态输出不是COCO17格式")
            scores = np.nan_to_num(confidence[0], nan=0., posinf=0., neginf=0.).astype(np.float32)
            valid = np.isfinite(keypoints[0]).all(1) & (scores >= self.config["keypoint_threshold"])
            points[valid] = keypoints[0][valid]
            self.previous_box, self.last_seen = box.copy(), now
            if restarted:
                self.segment_id += 1
            segment = self.segment_id
        return {"keypoints": points, "scores": scores, "valid": valid, "segment_id": segment,
                "box": box, "detector_ms": detection_ms, "pose_ms": pose_ms}
