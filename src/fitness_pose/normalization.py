"""将原图骨架转换成因果、带缺失标记的归一化序列和固定时间窗口。

原点取左右髋中点，单位取经过历史平滑的肩髋距离。只使用当前和过去帧，
不旋转人体，不跨跟踪段补点；保留中心与尺度以供后续分析整体运动。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """秒/像素为单位的初始工程参数；短躯干阈值用于防止除小数放大噪声。"""

    ema_tau_s: float = 0.15
    carry_s: float = 0.25
    min_torso_px: float = 5.0


class CausalPoseNormalizer:
    """训练预处理与回放共享的逐帧状态；step 只接收当前一组 COCO17 骨架。"""

    def __init__(self, config: GeometryConfig = GeometryConfig()):
        if config.ema_tau_s <= 0 or config.carry_s < 0 or config.min_torso_px <= 0:
            raise ValueError("几何处理参数必须使用有效的正数/非负数")
        self.config = config
        self.previous_segment = -1
        self.last_anchor = -math.inf
        self.last_time = -math.inf
        self.center = None
        self.scale = None

    def step(self, keypoints, scores, valid, timestamp: float, segment: int) -> dict:
        points = np.asarray(keypoints, dtype=np.float64)
        score = np.asarray(scores, dtype=np.float64)
        mask = np.asarray(valid, dtype=bool).copy()
        if points.shape != (17, 2) or score.shape != (17,) or mask.shape != (17,):
            raise ValueError("逐帧骨架形状异常")
        if not math.isfinite(timestamp) or timestamp <= self.last_time:
            raise ValueError("时间戳必须严格递增且有限")
        self.last_time = timestamp
        mask &= np.isfinite(points).all(axis=-1) & np.isfinite(score)
        features = np.zeros((17, 3), dtype=np.float32)
        result = {"features": features, "valid": np.zeros(17, dtype=bool),
                  "center_xy_px": np.full(2, np.nan, dtype=np.float32), "torso_scale_px": np.float32(np.nan),
                  "geometry_valid": False, "geometry_carried": False}
        if segment < 0 or segment != self.previous_segment:
            self.center, self.scale, self.last_anchor = None, None, -math.inf
        self.previous_segment = segment
        if segment < 0:
            return result
        if mask[[5, 6, 11, 12]].all():
            center = points[[11, 12]].mean(axis=0)
            length = float(np.linalg.norm(points[[5, 6]].mean(axis=0) - center))
            if length >= self.config.min_torso_px:
                if self.scale is None or timestamp - self.last_anchor > self.config.carry_s:
                    self.scale = length
                else:
                    alpha = -math.expm1(-(timestamp - self.last_anchor) / self.config.ema_tau_s)
                    self.scale = (1 - alpha) * self.scale + alpha * length
                self.center, self.last_anchor = center, timestamp
        if self.center is None or timestamp - self.last_anchor > self.config.carry_s:
            return result
        features[mask, :2] = ((points[mask] - self.center) / self.scale).astype(np.float32)
        features[mask, 2] = np.clip(score[mask], 0, 1).astype(np.float32)
        result.update(valid=mask, center_xy_px=self.center.astype(np.float32),
                      torso_scale_px=np.float32(self.scale), geometry_valid=True,
                      geometry_carried=timestamp > self.last_anchor)
        return result


def normalize_pose(keypoints, scores, valid, timestamps, segment_id,
                   config: GeometryConfig = GeometryConfig()) -> dict[str, np.ndarray]:
    """输入 T×17×2 像素坐标，输出 T×17×3 的 x/y/置信度和独立有效掩码。

无效网络输入填零且其置信度也置零，valid 仍显式保留，不能将零填充解释为
真实关节。几何参照最多沿用过去 carry_s 秒，丢失跟踪或更换目标时立即重置。
"""
    points = np.asarray(keypoints, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).copy()
    times = np.asarray(timestamps, dtype=np.float64)
    segments = np.asarray(segment_id)
    total = len(times)
    if points.shape != (total, 17, 2) or score.shape != (total, 17) or mask.shape != (total, 17):
        raise ValueError("原始骨架形状不是 (T,17,2)/(T,17)")
    if segments.shape != (total,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("跟踪段形状异常，或时间戳不是严格递增的有限数值")
    features = np.zeros((total, 17, 3), dtype=np.float32)
    output_mask = np.zeros((total, 17), dtype=bool)
    centers = np.full((total, 2), np.nan, dtype=np.float32)
    scales = np.full(total, np.nan, dtype=np.float32)
    geometry_valid = np.zeros(total, dtype=bool)
    carried = np.zeros(total, dtype=bool)
    normalizer = CausalPoseNormalizer(config)
    for index, now in enumerate(times):
        frame = normalizer.step(points[index], score[index], mask[index], float(now), int(segments[index]))
        features[index], output_mask[index] = frame["features"], frame["valid"]
        centers[index], scales[index] = frame["center_xy_px"], frame["torso_scale_px"]
        geometry_valid[index], carried[index] = frame["geometry_valid"], frame["geometry_carried"]
    return {"features": features, "valid": output_mask, "center_xy_px": centers,
            "torso_scale_px": scales, "geometry_valid": geometry_valid, "geometry_carried": carried}


def make_windows(normalized: dict, timestamps, segment_id, *, window_size: int = 48,
                 stride: int = 10, fps: float = 20.0, max_age_s: float = 0.15,
                 min_valid_ratio: float = 0.7, min_joints_per_frame: int = 6, short_gap_s: float = 0.0) -> dict[str, np.ndarray]:
    """在连续跟踪段内生成 C×T×V 窗口，按目标时刻取最近的过去样本。

不做使用未来帧的线性插值。时间超过 max_age_s 的旧姿态、跨跟踪段或有效
关节不足的窗口不进入训练；不把同段视频的窗口随机划成训练和验证集合。
"""
    if window_size < 2 or stride < 1 or not math.isfinite(fps) or fps <= 0 or max_age_s <= 0:
        raise ValueError("无效的窗口或采样参数")
    if not 0 < min_valid_ratio <= 1 or not 1 <= min_joints_per_frame <= 17:
        raise ValueError("无效的窗口质量阈值")
    times = np.asarray(timestamps, dtype=np.float64)
    segments = np.asarray(segment_id)
    if short_gap_s:
        # 可选恢复策略与在线代码共用实现；默认0保留旧训练窗口，避免悄悄改写缓存。
        from .streaming import OnlineWindowBuffer
        buffer = OnlineWindowBuffer(window_size, stride, fps, max_age_s, min_valid_ratio, min_joints_per_frame, short_gap_s)
        windows = []
        for i, now in enumerate(times):
            frame = {name: normalized[name][i] for name in ("features", "valid", "geometry_valid")}
            windows.extend(buffer.step(frame, float(now), int(segments[i]), i))
        source = np.array([w["source_index"] for w in windows], dtype=np.int64).reshape(-1, window_size)
        return {"x": np.ascontiguousarray(normalized["features"][source].transpose(0, 3, 1, 2)),
                "valid": normalized["valid"][source], "source_index": source,
                "time_grid_s": np.array([w["time_grid_s"] for w in windows]).reshape(-1, window_size),
                "segment_id": np.array([w["segment_id"] for w in windows], dtype=np.int32)}
    usable = normalized["geometry_valid"] & (segments >= 0)
    runs, begin = [], None
    for index in range(len(times)):
        same_run = (index > 0 and usable[index - 1] and segments[index] == segments[index - 1]
                    and times[index] - times[index - 1] <= max_age_s)
        if begin is not None and (not usable[index] or not same_run):
            runs.append((begin, index))
            begin = None
        if usable[index] and begin is None:
            begin = index
    if begin is not None:
        runs.append((begin, len(times)))

    indices, grids, ids = [], [], []
    duration, step = (window_size - 1) / fps, stride / fps
    for begin, end in runs:
        run_times = times[begin:end]
        number = max(0, math.floor((run_times[-1] - run_times[0] - duration) / step + 1e-7) + 1)
        for offset in range(number):
            grid = run_times[0] + offset * step + np.arange(window_size) / fps
            # side=right 保证取到时间不晚于当前格点的最后一个样本。
            local = np.searchsorted(run_times, grid, side="right") - 1
            if np.any(local < 0):
                continue
            source = local + begin
            if np.any(grid - times[source] > max_age_s):
                continue
            mask = normalized["valid"][source]
            if mask.mean() < min_valid_ratio or np.any(mask.sum(axis=1) < min_joints_per_frame):
                continue
            indices.append(source)
            grids.append(grid)
            ids.append(int(segments[begin]))
    source_index = np.asarray(indices, dtype=np.int64).reshape(-1, window_size)
    features = normalized["features"][source_index].transpose(0, 3, 1, 2)
    return {"x": np.ascontiguousarray(features), "valid": normalized["valid"][source_index],
            "source_index": source_index, "time_grid_s": np.asarray(grids, dtype=np.float64).reshape(-1, window_size),
            "segment_id": np.asarray(ids, dtype=np.int32)}
