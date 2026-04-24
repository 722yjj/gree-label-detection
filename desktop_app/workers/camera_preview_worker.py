"""Background worker for live camera preview and still capture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from desktop_app.devices.camera.base import CameraAdapter


@dataclass(frozen=True)
class CameraFrameBrightness:
    mean: float
    p95: float
    dark_ratio: float
    level: Literal["too_dark", "dim", "ok"]

    @property
    def is_too_dark(self) -> bool:
        return self.level == "too_dark"


class CameraPreviewWorker(QObject):
    """Keep one camera session open, emit preview frames, and save still frames."""

    started_status = Signal(str)
    frame_ready = Signal(object)
    frame_brightness_changed = Signal(object)
    photo_saved = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        camera_adapter: CameraAdapter,
        *,
        interval_ms: int = 100,
        brightness_interval_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.camera_adapter = camera_adapter
        self.interval_ms = interval_ms
        self.brightness_interval_s = brightness_interval_s
        self._session = None
        self._timer: QTimer | None = None
        self._latest_frame = None
        self._stopped = False
        self._last_brightness_level: str | None = None
        self._last_brightness_emit_at = 0.0

    @Slot()
    def start(self) -> None:
        self.started_status.emit(f"检查相机连接: {self.camera_adapter.name}")
        try:
            self._session = self.camera_adapter.start_preview()
        except Exception as exc:
            self.failed.emit(f"未检测到相机: {exc}")
            self.stop_preview()
            return

        display_name = getattr(self._session, "display_name", self.camera_adapter.name)
        self.started_status.emit(f"相机预览中: {display_name}")
        self._timer = QTimer(self)
        self._timer.setInterval(self.interval_ms)
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start()
        self._poll_frame()

    @Slot()
    def stop_preview(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None

        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None

        self.stopped.emit()

    @Slot(object)
    def save_current_frame(self, preferred_code: object = None) -> None:
        if self._session is None:
            self.failed.emit("未检测到相机: 预览未启动")
            return
        if self._latest_frame is None:
            self.failed.emit("尚无可保存的相机画面，请等待预览画面出现后再拍照")
            return

        code = str(preferred_code or "").strip() or None
        try:
            if hasattr(self._session, "save_frame_bgr"):
                saved_path = self._session.save_frame_bgr(self._latest_frame.copy(), code)
            else:
                saved_path = self.camera_adapter.capture(preferred_code=code)
        except Exception as exc:
            self.failed.emit(f"相机拍照保存失败: {exc}")
            return

        if not saved_path:
            self.failed.emit("相机拍照保存失败: 未获取到可用目标图片")
            return

        self.photo_saved.emit(Path(saved_path))

    @Slot()
    def _poll_frame(self) -> None:
        if self._session is None:
            return

        try:
            frame = self._session.read_frame_bgr()
        except Exception as exc:
            self.failed.emit(f"相机预览失败: {exc}")
            self.stop_preview()
            return

        self._latest_frame = frame
        self._emit_brightness_if_needed(frame)
        self.frame_ready.emit(_bgr_frame_to_qimage(frame))

    def _emit_brightness_if_needed(self, frame) -> None:
        brightness = measure_frame_brightness(frame)
        now = time.monotonic()
        should_emit = (
            brightness.level != self._last_brightness_level
            or now - self._last_brightness_emit_at >= self.brightness_interval_s
        )
        if not should_emit:
            return

        self._last_brightness_level = brightness.level
        self._last_brightness_emit_at = now
        self.frame_brightness_changed.emit(brightness)


def measure_frame_brightness(frame) -> CameraFrameBrightness:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    hist = np.bincount(gray.ravel(), minlength=256)
    p95_index = int(np.searchsorted(hist.cumsum(), gray.size * 0.95))
    p95 = float(min(p95_index, 255))
    dark_ratio = float(np.mean(gray < 50))

    if mean < 85.0 or p95 < 115.0:
        level: Literal["too_dark", "dim", "ok"] = "too_dark"
    elif mean < 115.0 or p95 < 145.0:
        level = "dim"
    else:
        level = "ok"

    return CameraFrameBrightness(
        mean=mean,
        p95=p95,
        dark_ratio=dark_ratio,
        level=level,
    )


def _bgr_frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


__all__ = [
    "CameraFrameBrightness",
    "CameraPreviewWorker",
    "measure_frame_brightness",
]
