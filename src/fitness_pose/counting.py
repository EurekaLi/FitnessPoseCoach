"""根据当前动作与可见关节完成几何周期计数，不读取真实次数/动作标签。

阈值是待验证的工程初值，仅用于周期检测，不等于动作规范性标准。
状态按“起始端→另一端→起始端”推进；不补算预热前或缺失期间的周期。
"""

from dataclasses import dataclass
from collections import deque
import math

import numpy as np

from .action_data import CLASSES


@dataclass(frozen=True)
class CountConfig:
    squat_home_deg: float = 155.0
    squat_away_deg: float = 110.0
    push_home_deg: float = 155.0
    push_away_deg: float = 100.0
    jump_closed_leg_ratio: float = 0.55
    jump_open_leg_ratio: float = 0.90
    jump_closed_arm_ratio: float = -0.10
    jump_open_arm_ratio: float = 0.15
    signal_tau_s: float = 0.10
    dwell_s: float = 0.12
    min_cycle_s: float = 0.40
    max_cycle_s: float = 10.0
    max_gap_s: float = 0.20
    stable_side: bool = False
    stable_side_actions: tuple[str, ...] = ("squat", "push_up")
    side_switch_margin: float = 0.10
    side_confirm_s: float = 0.25
    squat_signal: str = "knee_angle"
    height_history_s: float = 6.0
    height_calibration_s: float = 1.0
    height_min_amplitude: float = 0.15

    def __post_init__(self):
        if not (0 < self.squat_away_deg < self.squat_home_deg <= 180
                and 0 < self.push_away_deg < self.push_home_deg <= 180
                and 0 <= self.jump_closed_leg_ratio < self.jump_open_leg_ratio
                and self.jump_closed_arm_ratio < self.jump_open_arm_ratio
                and self.signal_tau_s > 0 and self.dwell_s > 0 and self.max_gap_s > 0
                and 0 < self.min_cycle_s < self.max_cycle_s):
            raise ValueError("计数配置的阈值顺序或时间参数无效")
        if (self.squat_signal not in ("knee_angle", "hip_height", "auto_view") or self.side_switch_margin < 0
                or self.side_confirm_s <= 0 or not 0 < self.height_calibration_s < self.height_history_s
                or self.height_min_amplitude <= 0):
            raise ValueError("计数信号或稳定策略配置无效")
        if any(a not in ("squat", "push_up") for a in self.stable_side_actions):
            raise ValueError("稳定侧选择只适用于角度类动作")


def joint_angle(points, a: int, b: int, c: int) -> float | None:
    left, right = points[a] - points[b], points[c] - points[b]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-6:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1, 1))
    return math.degrees(math.acos(cosine))


def angle_candidates(action, points, scores, mask) -> dict:
    sides = ((11, 13, 15), (12, 14, 16)) if action == "squat" else ((5, 7, 9), (6, 8, 10))
    result = {}
    for name, joints in zip(("left", "right"), sides):
        if mask[list(joints)].all():
            angle = joint_angle(points, *joints)
            if angle is not None:
                result[name] = (float(scores[list(joints)].min()), angle)
    return result


class StableSideSelector:
    """保留可靠当前侧；候选持续胜出才切换，避免瞬时置信度波动产生角度跳变。"""

    def __init__(self, config):
        self.config = config
        self.side = self.candidate = self.since = None

    def select(self, choices: dict, now: float) -> tuple[str | None, bool]:
        if not choices:
            self.candidate = self.since = None
            return None, False
        best = max(choices, key=lambda name: choices[name][0])
        if self.side is None:
            self.side = best
            return best, False
        better = (self.side not in choices or
                  best != self.side and choices[best][0] >= choices[self.side][0] + self.config.side_switch_margin)
        if better:
            if self.candidate != best:
                self.candidate, self.since = best, now
            if now - self.since + 1e-9 >= self.config.side_confirm_s:
                self.side, self.candidate, self.since = best, None, None
                return best, True
        else:
            self.candidate = self.since = None
        return (self.side if self.side in choices else None), False


def hip_height_measurement(keypoints, scores, valid) -> tuple[float, float] | None:
    """返回脚踝-髋部竖直像素差与当前躯干像素长度，不使用未来或真实阶段标签。"""
    points = np.asarray(keypoints, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(points).all(1) & np.isfinite(scores)
    if not mask[[5, 6, 11, 12, 15, 16]].all():
        return None
    hip, shoulder = points[[11, 12]].mean(0), points[[5, 6]].mean(0)
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 5:
        return None
    return float(points[[15, 16], 1].mean() - hip[1]), torso


def hip_height_signal(keypoints, scores, valid) -> np.ndarray | None:
    """无状态几何量，用于诊断和尺度不变性检查；在线计数使用历史中位尺度。"""
    measurement = hip_height_measurement(keypoints, scores, valid)
    return np.array([measurement[0] / measurement[1]]) if measurement else None


class HeightRange:
    """阈值仅来自过去6秒的可靠高度，不用整段最大最小值偷看未来。

    10/90分位减少尖峰影响；振幅不足时不输出周期端点，避免静止抖动计数。
    """

    def __init__(self, config):
        self.config, self.history = config, deque()

    def zone(self, value, now):
        self.history.append((now, value))
        while self.history and now - self.history[0][0] > self.config.height_history_s:
            self.history.popleft()
        if now - self.history[0][0] < self.config.height_calibration_s:
            return None, None
        low, high = np.quantile([v for _, v in self.history], [.1, .9])
        if high - low < self.config.height_min_amplitude:
            return None, [float(low), float(high)]
        phase = (value - low) / (high - low)
        return ("home" if phase >= .75 else "away" if phase <= .25 else "middle"), [float(low), float(high)]


class HeightObserver:
    """可靠骨架到达就校准几何量，但不计数；分类预热不需要再额外丢一轮标定。

    真正缺少所需关节、换人或长间隔仍清空。所有尺度、振幅与视角代理仅来自历史。
    """

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.scales, self.widths = deque(), deque()
        self.range = HeightRange(self.config)
        self.filtered, self.last_time, self.segment = None, None, -1

    def step(self, points, scores, valid, now, segment):
        if (segment != self.segment or segment < 0
                or self.last_time is not None and now - self.last_time > self.config.max_gap_s):
            self.reset()
        measurement = hip_height_measurement(points, scores, valid)
        if measurement is None or segment < 0:
            self.reset()
            return None
        height_px, torso_px = measurement
        array = np.asarray(points)
        self.scales.append((now, torso_px))
        self.widths.append((now, float(np.linalg.norm(array[5] - array[6]) / torso_px)))
        while self.scales and now - self.scales[0][0] > self.config.height_history_s:
            self.scales.popleft()
        while self.widths and now - self.widths[0][0] > 1.0:
            self.widths.popleft()
        scale = float(np.median([v for _, v in self.scales]))
        signal = height_px / scale
        if self.filtered is None:
            self.filtered = signal
        else:
            alpha = -math.expm1(-(now - self.last_time) / self.config.signal_tau_s)
            self.filtered += alpha * (signal - self.filtered)
        self.last_time, self.segment = now, segment
        zone, thresholds = self.range.zone(self.filtered, now)
        return {"signal": self.filtered, "zone": zone, "thresholds": thresholds,
                "view_ratio": float(np.median([v for _, v in self.widths]))}


def pose_signal(action: str, keypoints, scores, valid) -> tuple[np.ndarray | None, str]:
    """角度单位为度；开合跳返回腿宽/抬臂两个躯干长度比值。"""
    points, scores = np.asarray(keypoints, dtype=float), np.asarray(scores, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(points).all(1) & np.isfinite(scores)
    if action in ("squat", "push_up"):
        choices = angle_candidates(action, points, scores, mask)
        if not choices:
            return None, "required_joints_missing"
        side = max(choices, key=lambda name: choices[name][0])
        return np.array([choices[side][1]]), side
    if action != "jump_jack":
        raise ValueError(f"未知动作：{action}")
    required = [5, 6, 9, 10, 11, 12, 15, 16]
    if not mask[required].all():
        return None, "required_joints_missing"
    shoulder, hip = points[[5, 6]].mean(0), points[[11, 12]].mean(0)
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 5:
        return None, "torso_too_small"
    leg = float(np.linalg.norm(points[15] - points[16]) / torso)
    # 图像y轴向下；正值表示手腕高于肩。只适用于相机大致竖直、全身可见的场景。
    arm = float((shoulder[1] - points[[9, 10], 1].mean()) / torso)
    return np.array([leg, arm]), "both_sides"


class CycleCounter:
    def __init__(self, config: CountConfig):
        self.config, self.count = config, 0
        self.reset()

    def reset(self):
        """只清未完成周期，累计次数不受跟踪丢失和类别切换影响。"""
        self.state = "need_home"
        self.zone, self.zone_since = None, None
        self.started_at = None
        self.last_time = None

    def step(self, zone: str, now: float) -> bool:
        if self.last_time is not None and now - self.last_time > self.config.max_gap_s:
            self.reset()
        if self.started_at is not None and now - self.started_at > self.config.max_cycle_s:
            self.reset()
        self.last_time = now
        if zone != self.zone:
            self.zone, self.zone_since = zone, now
        if zone == "middle" or now - self.zone_since + 1e-9 < self.config.dwell_s:
            return False
        if self.state == "need_home" and zone == "home":
            self.state = "need_away"
        elif self.state == "need_away":
            if zone == "home":
                # 长时间站立不应占满最大周期：离开起始姿态后才开始计时。
                self.started_at = now
            elif zone == "away":
                self.started_at = now if self.started_at is None else self.started_at
                self.state = "need_return"
        elif self.state == "need_return" and zone == "home":
            if now - self.started_at + 1e-9 >= self.config.min_cycle_s:
                self.count += 1
                self.state, self.started_at = "need_away", now
                return True
        return False


class ExerciseCounter:
    def __init__(self, config: CountConfig = CountConfig()):
        self.config = config
        self.counters = {action: CycleCounter(config) for action in CLASSES}
        self.action, self.segment, self.last_time = None, -1, None
        self.filtered, self.last_signal_time = None, None
        self.side_selector = StableSideSelector(config)
        self.height_observer = HeightObserver(config)
        self.squat_mode = None
        self.side_changes = 0

    @property
    def counts(self) -> dict[str, int]:
        return {name: counter.count for name, counter in self.counters.items()}

    def reset_partial(self, reset_selection=True):
        for counter in self.counters.values():
            counter.reset()
        self.filtered, self.last_signal_time = None, None
        self.squat_mode = None
        if reset_selection:
            self.side_selector = StableSideSelector(self.config)

    def step(self, action: str | None, points, scores, valid, now: float, segment: int) -> dict:
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("计数时间戳必须递增")
        height = self.height_observer.step(points, scores, valid, now, segment) if self.config.squat_signal != "knee_angle" else None
        if (action != self.action or segment != self.segment or segment < 0
                or self.last_time is not None and now - self.last_time > self.config.max_gap_s):
            self.reset_partial()
        self.action, self.segment, self.last_time = action, segment, now
        if action is None or segment < 0:
            return {"status": "paused", "counts": self.counts, "event": None, "signal": None}
        switched = False
        mode = self.config.squat_signal
        if action == "squat" and mode == "auto_view":
            if self.squat_mode is None:
                # 0.45是待验证的投影宽度代理阈值，不是精确视角估计；一段计数内锁定信号。
                self.squat_mode = "hip_height" if height and height["view_ratio"] >= .45 else "knee_angle"
            mode = self.squat_mode
        signal_type = "hip_height" if action == "squat" and mode == "hip_height" else "joint_angle" if action in ("squat", "push_up") else "jump_openness"
        if signal_type == "hip_height":
            signal = np.array([height["signal"]]) if height else None
            side = "both_sides" if height else "height_joints_missing"
        elif self.config.stable_side and action in self.config.stable_side_actions:
            array = np.asarray(points, dtype=float)
            mask = np.asarray(valid, dtype=bool) & np.isfinite(array).all(1) & np.isfinite(scores)
            choices = angle_candidates(action, array, np.asarray(scores), mask)
            side, switched = self.side_selector.select(choices, now)
            signal = np.array([choices[side][1]]) if side is not None else None
            side = side or "side_confirmation"
            if switched:
                self.side_changes += 1
                self.reset_partial(reset_selection=False)
        else:
            signal, side = pose_signal(action, points, scores, valid)
        if signal is None:
            # 候选侧确认期间清周期，但保留确认计时，否则永远无法完成换侧。
            self.reset_partial(reset_selection=False)
            return {"status": side, "counts": self.counts, "event": None, "signal": None}
        if self.filtered is None or signal_type == "hip_height":
            self.filtered = signal.copy()
        else:
            alpha = -math.expm1(-(now - self.last_signal_time) / self.config.signal_tau_s)
            self.filtered += alpha * (signal - self.filtered)
        self.last_signal_time = now
        value, config = self.filtered, self.config
        thresholds = None
        if signal_type == "hip_height":
            zone, thresholds = height["zone"], height["thresholds"]
            if zone is None:
                self.counters[action].reset()
                return {"status": "height_calibrating", "counts": self.counts, "event": None,
                        "signal": value.tolist(), "signal_type": signal_type, "thresholds": thresholds, "side": side}
        elif action == "squat":
            zone = "home" if value[0] >= config.squat_home_deg else "away" if value[0] <= config.squat_away_deg else "middle"
        elif action == "push_up":
            zone = "home" if value[0] >= config.push_home_deg else "away" if value[0] <= config.push_away_deg else "middle"
        else:
            zone = ("home" if value[0] <= config.jump_closed_leg_ratio and value[1] <= config.jump_closed_arm_ratio else
                    "away" if value[0] >= config.jump_open_leg_ratio and value[1] >= config.jump_open_arm_ratio else "middle")
        completed = self.counters[action].step(zone, now)
        return {"status": self.counters[action].state, "counts": self.counts, "event": action if completed else None,
                "signal": value.tolist(), "side": side, "side_changed": switched, "zone": zone,
                "signal_type": signal_type, "thresholds": thresholds}
