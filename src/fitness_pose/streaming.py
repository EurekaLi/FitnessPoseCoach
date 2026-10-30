"""在线窗口与历史概率过滤；接口只接收已经到达的帧，不接受整段标签。"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


class OnlineWindowBuffer:
    """保持训练 make_windows 的时间网格与质量门槛，丢失/换人后重新预热。"""

    def __init__(self, window_size=48, stride=10, fps=20.0, max_age_s=0.15,
                 min_valid_ratio=0.7, min_joints_per_frame=6, short_gap_s=0.0):
        if not 0 <= short_gap_s <= max_age_s:
            raise ValueError("短缺失恢复不能超过已有过去样本最大年龄")
        self.window_size, self.stride, self.fps = window_size, stride, fps
        self.max_age_s, self.min_valid_ratio = max_age_s, min_valid_ratio
        self.min_joints_per_frame = min_joints_per_frame
        self.duration, self.interval = (window_size - 1) / fps, stride / fps
        self.samples = deque()
        self.segment = -1
        self.last_time = -math.inf
        self.anchor = None
        self.offset = 0
        self.short_gap_s = short_gap_s
        self.last_valid_time = -math.inf
        self.last_reset_reason = None
        self.rejected_windows = 0

    def step(self, frame: dict, timestamp: float, segment: int, source_index: int) -> list[dict]:
        if not math.isfinite(timestamp) or timestamp <= self.last_time:
            raise ValueError("窗口输入时间戳必须严格递增")
        usable = segment >= 0 and frame["geometry_valid"] and (self.short_gap_s == 0 or frame["valid"].sum() >= self.min_joints_per_frame)
        short_interruption = (self.short_gap_s > 0 and self.anchor is not None
                              and timestamp - self.last_valid_time <= self.short_gap_s + 1e-9)
        reason = ("tracking_missing" if segment < 0 and not short_interruption else
                  "segment_changed" if segment >= 0 and segment != self.segment else
                  "timestamp_gap" if timestamp - self.last_time > self.max_age_s + 1e-9 else None)
        allowed_gap = self.short_gap_s if self.short_gap_s else self.max_age_s
        if reason is None and (not usable or self.anchor is not None and timestamp - self.last_valid_time > allowed_gap + 1e-9):
            if self.short_gap_s == 0 or timestamp - self.last_valid_time > self.short_gap_s + 1e-9:
                reason = "geometry_or_joint_gap"
        self.last_time = timestamp
        # -1只是本帧未匹配；只有短时返回相同非负ID才可恢复。新ID、超时均强制清空。
        # 原始-1帧不入窗口，当前显示/计数仍暂停，不将缺失伪装为有效观测。
        if segment >= 0 or not short_interruption:
            self.segment = segment
        self.last_reset_reason = reason
        if reason:
            self.samples.clear()
            self.anchor, self.offset = None, 0
        if segment < 0 or not usable:
            return []
        if self.anchor is None:
            self.anchor = timestamp
        self.samples.append((timestamp, frame["features"].copy(), frame["valid"].copy(), source_index))
        self.last_valid_time = timestamp
        available = max(0, math.floor((timestamp - self.anchor - self.duration) / self.interval + 1e-7) + 1)
        emitted = []
        if self.offset < available:
            times = np.array([r[0] for r in self.samples])
            features = np.array([r[1] for r in self.samples])
            masks = np.array([r[2] for r in self.samples])
            indexes = np.array([r[3] for r in self.samples], dtype=np.int64)
            for offset in range(self.offset, available):
                grid = self.anchor + offset * self.interval + np.arange(self.window_size) / self.fps
                past = np.searchsorted(times, grid, side="right") - 1
                if np.any(past < 0) or np.any(grid - times[past] > self.max_age_s):
                    self.rejected_windows += 1
                    continue
                mask = masks[past]
                if mask.mean() < self.min_valid_ratio or np.any(mask.sum(1) < self.min_joints_per_frame):
                    self.rejected_windows += 1
                    continue
                # 短缺失时不制造关节：仍用已有 searchsorted 过去取样规则，严格检查年龄。
                # source_index 会显示重复采样来源，原始 missing 帧和计数暂停状态保持不变。
                emitted.append({"x": np.ascontiguousarray(features[past].transpose(2, 0, 1)),
                                "source_index": indexes[past], "time_grid_s": grid,
                                "segment_id": segment, "emitted_at_s": timestamp,
                                "max_source_age_s": float(np.max(grid - times[past]))})
            self.offset = available
        # 留足下一窗口起点之前的旧样本；仅保留一小段历史，不随视频长度增长。
        earliest = timestamp - self.duration - self.interval - self.max_age_s
        while len(self.samples) > 1 and self.samples[1][0] < earliest:
            self.samples.popleft()
        return emitted


@dataclass(frozen=True)
class ActionFilterConfig:
    """初始工程值：秒为单位的历史平滑/过期时间，非经标注校准的拒识器。"""

    tau_s: float = 0.4
    min_probability: float = 0.6
    min_margin: float = 0.15
    confirmation_windows: int = 2
    stale_s: float = 0.75


class ActionFilter:
    def __init__(self, config: ActionFilterConfig = ActionFilterConfig()):
        if (config.tau_s <= 0 or config.stale_s <= 0 or not 0 <= config.min_probability <= 1
                or not 0 <= config.min_margin <= 1 or config.confirmation_windows < 1):
            raise ValueError("动作过滤参数无效")
        self.config = config
        self.reset()

    def reset(self):
        self.probabilities = None
        self.last_prediction = None
        self.candidate, self.hits, self.action = None, 0, None
        self.status = "warming_up"

    def update(self, probabilities, now: float) -> None:
        self.expire(now)
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1, atol=1e-5):
            raise ValueError("动作概率必须是三个有限非负且和为1的数")
        if self.probabilities is None:
            self.probabilities = values.copy()
        else:
            alpha = -math.expm1(-(now - self.last_prediction) / self.config.tau_s)
            self.probabilities += alpha * (values - self.probabilities)
        self.last_prediction = now
        order = np.argsort(self.probabilities)
        candidate = int(order[-1])
        if (self.probabilities[candidate] < self.config.min_probability
                or self.probabilities[order[-1]] - self.probabilities[order[-2]] < self.config.min_margin):
            self.action, self.candidate, self.hits, self.status = None, None, 0, "uncertain"
            return
        self.hits = self.hits + 1 if candidate == self.candidate else 1
        self.candidate = candidate
        if self.hits >= self.config.confirmation_windows:
            self.action, self.status = candidate, "ready"
        elif self.action != candidate:
            # 切换确认期间不沿用旧动作来凑计数周期。
            self.action, self.status = None, "confirming"

    def expire(self, now: float) -> None:
        if self.last_prediction is not None and now - self.last_prediction > self.config.stale_s:
            self.reset()
            self.status = "stale"
