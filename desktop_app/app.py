"""Desktop app bootstrap helpers."""

from __future__ import annotations

from desktop_app.controllers.app_controller import AppController
from desktop_app.devices.camera.mock_camera import MockCameraAdapter
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.ui.main_window import MainWindow


def build_main_window() -> MainWindow:
    """Create the main window and wire the first controller stack."""

    window = MainWindow()
    controller = AppController(
        view=window,
        template_repository=TemplateRepository(),
        detection_service=DetectionService(),
        camera_adapter=MockCameraAdapter(),
    )
    window._controller = controller
    return window

