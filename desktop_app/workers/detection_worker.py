"""Background worker for unified detection runs."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from desktop_app.models import DetectionJobRequest, DetectionJobResult
from desktop_app.services.detection_service import DetectionService


class DetectionWorker(QObject):
    """Run a detection job outside the main UI thread."""

    started_status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        detection_service: DetectionService,
        request: DetectionJobRequest,
    ) -> None:
        super().__init__()
        self.detection_service = detection_service
        self.request = request

    @Slot()
    def run(self) -> None:
        self.started_status.emit("检测任务启动中...")
        try:
            result = self.detection_service.run(self.request)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        if result.success:
            self.finished.emit(result)
            return

        self.failed.emit(result.error or "检测失败")

