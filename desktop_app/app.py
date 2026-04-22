"""Desktop app bootstrap helpers."""

from __future__ import annotations

from label_detection.core.config import PROJECT_ROOT

from desktop_app.controllers.app_controller import AppController
from desktop_app.devices.camera.mock_camera import MockCameraAdapter
from desktop_app.devices.scanner.base import ScannerAdapter
from desktop_app.repositories.history_repository import HistoryRepository
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.services.preview_service import PreviewService
from desktop_app.ui.main_window import MainWindow


def build_main_window(scanner_adapter: ScannerAdapter | None = None) -> MainWindow:
    """Create the main window and wire the first controller stack."""

    window = MainWindow()
    controller = AppController(
        view=window,
        template_repository=TemplateRepository(),
        detection_service=DetectionService(),
        camera_adapter=MockCameraAdapter(),
        scanner_adapter=scanner_adapter,
        preview_service=PreviewService(),
        history_repository=HistoryRepository(
            PROJECT_ROOT / "results" / "desktop_app" / "history.json"
        ),
    )
    window._controller = controller
    return window
