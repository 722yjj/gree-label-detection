"""Background worker for live camera preview and still capture."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from desktop_app.devices.camera.base import CameraAdapter


class CameraPreviewWorker(QObject):
    """Keep one camera session open, emit preview frames, and save still frames."""

    started_status = Signal(str)
    frame_ready = Signal(object)
    photo_saved = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        camera_adapter: CameraAdapter,
        *,
        interval_ms: int = 100,
    ) -> None:
        super().__init__()
        self.camera_adapter = camera_adapter
        self.interval_ms = interval_ms
        self._session = None
        self._timer: QTimer | None = None
        self._latest_frame = None
        self._stopped = False

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
        self.frame_ready.emit(_bgr_frame_to_qimage(frame))


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
