"""Desktop app bootstrap helpers."""

from __future__ import annotations

from label_detection.core.config import RESULTS_ROOT
from label_detection.license import LicenseStatus

from desktop_app.controllers.app_controller import AppController
from desktop_app.devices.camera.factory import build_camera_adapter
from desktop_app.devices.scanner.base import ScannerAdapter
from desktop_app.devices.scanner.keyboard_wedge_adapter import KeyboardWedgeScannerAdapter
from desktop_app.repositories.annotation_repository import AnnotationRepository
from desktop_app.repositories.history_repository import HistoryRepository
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.services.preview_service import PreviewService
from desktop_app.ui.main_window import MainWindow


def build_main_window(
    scanner_adapter: ScannerAdapter | None = None,
    license_status: LicenseStatus | None = None,
) -> MainWindow:
    """Create the main window and wire the first controller stack."""

    window = MainWindow()
    scanner_adapter = scanner_adapter or KeyboardWedgeScannerAdapter(window)
    controller = None
    if license_status is None or license_status.ok:
        controller = AppController(
            view=window,
            template_repository=TemplateRepository(),
            detection_service=DetectionService(),
            camera_adapter=build_camera_adapter(),
            scanner_adapter=scanner_adapter,
            preview_service=PreviewService(),
            history_repository=HistoryRepository(
                RESULTS_ROOT / "desktop_app" / "history.json"
            ),
            annotation_repository=AnnotationRepository(),
        )
    window._controller = controller
    if license_status is not None:
        window.set_license_status(license_status)
    return window
