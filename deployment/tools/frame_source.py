"""采集线程只保留最新帧；相机时间取实际读帧完成时刻，不伪造固定FPS。

离线视频按文件顺序读取，不丢帧，并使用视频时间；不能把离线处理吞吐率当
作摄像头速度。两个源都只在内存中保留图像，不写入录像。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
import math
import time


@dataclass
class CapturedFrame:
    index: int
    timestamp_s: float
    acquired_at: float
    image: object


class LatestFrameCapture:
    def __init__(self, index: int, backend: str, width: int, height: int, fps: float):
        self.index, self.backend = index, backend
        self.width, self.height, self.fps = width, height, fps
        self.condition, self.stop, self.ready = Condition(), Event(), Event()
        self.latest, self.error, self.metadata = None, None, {}
        self.frames_read, self.finished = 0, False
        self.thread = Thread(target=self._run, name="fitness-camera", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(10):
            self.close()
            raise TimeoutError("摄像头打开超时，请关闭独占相机的应用后重试")
        if self.error:
            self.close()
            raise RuntimeError(self.error)
        return self

    def _run(self):
        import cv2

        capture = None
        try:
            backend = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}[self.backend]
            # 打开、读取和释放均由同一线程拥有，主线程不与阻塞read争用VideoCapture。
            capture = cv2.VideoCapture(self.index, backend)
            if not capture.isOpened():
                raise RuntimeError(f"摄像头{self.index}/{self.backend}打开失败，可能未连接或被独占")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.metadata = {"index": self.index, "backend": capture.getBackendName(),
                             "reported_fps": reported_fps if math.isfinite(reported_fps) else None}
            self.ready.set()
            origin = time.perf_counter()
            while not self.stop.is_set():
                success, image = capture.read()
                acquired = time.perf_counter()
                if not success or image is None or image.size == 0:
                    raise RuntimeError("摄像头读取失败或已断开")
                with self.condition:
                    self.latest = CapturedFrame(self.frames_read, acquired - origin, acquired, image)
                    self.frames_read += 1
                    self.metadata["actual_shape_hwc"] = list(image.shape)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = f"{type(error).__name__}: {error}"
                self.condition.notify_all()
        finally:
            if capture is not None:
                capture.release()
            self.finished = True
            self.ready.set()
            with self.condition:
                self.condition.notify_all()

    def get_after(self, index: int, timeout: float = .1):
        with self.condition:
            self.condition.wait_for(lambda: self.error or self.finished or self.latest is not None and self.latest.index > index,
                                    timeout=timeout)
            if self.error:
                raise RuntimeError(self.error)
            if self.latest is None or self.latest.index <= index:
                return None
            item = self.latest
            return CapturedFrame(item.index, item.timestamp_s, item.acquired_at, item.image.copy())

    def close(self) -> bool:
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
        return not self.thread.is_alive()


class VideoFrames:
    def __init__(self, path: Path):
        import cv2

        self.cv2, self.capture = cv2, cv2.VideoCapture(str(path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not self.capture.isOpened() or not math.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise RuntimeError(f"视频不可读或FPS无效：{path}")
        self.frames_read, self.finished, self.last_timestamp = 0, False, -math.inf
        self.metadata = {"path": str(path), "fps": self.fps, "mode": "sequential_offline_video"}

    def get_after(self, index: int, timeout: float = .1):
        success, image = self.capture.read()
        acquired = time.perf_counter()
        if not success:
            expected = int(self.capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
            if expected > 0 and self.frames_read < expected - 1:
                raise RuntimeError(f"视频提前解码终止：{self.frames_read}/{expected}")
            self.finished = True
            return None
        timestamp = float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC)) / 1000
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= self.last_timestamp:
            timestamp = max(self.frames_read / self.fps, self.last_timestamp + 1 / self.fps)
        self.last_timestamp = timestamp
        item = CapturedFrame(self.frames_read, timestamp, acquired, image)
        self.frames_read += 1
        self.metadata["actual_shape_hwc"] = list(image.shape)
        return item

    def close(self) -> bool:
        self.capture.release()
        return True
