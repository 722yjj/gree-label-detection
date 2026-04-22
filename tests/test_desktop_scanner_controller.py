import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from desktop_app.controllers.app_controller import AppController
from desktop_app.devices.scanner.mock import MockScannerAdapter
from desktop_app.models import TemplateRecord
from desktop_app.ui.main_window import MainWindow


class FakeTemplateRepository:
    def __init__(self, templates_by_code: dict[str, list[TemplateRecord]]) -> None:
        self.templates_by_code = templates_by_code
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1

    def find_by_code(self, code: str) -> list[TemplateRecord]:
        return list(self.templates_by_code.get(code, []))


class FakeCameraAdapter:
    name = "fake-camera"

    def capture(self, preferred_code: str | None = None) -> Path | None:
        return None


class FakeDetectionService:
    def run(self, request):  # pragma: no cover - should never be called in these tests
        raise AssertionError("detection should not run in scanner controller tests")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def make_template(tmp_path: Path, code: str, variant: str | None = None) -> TemplateRecord:
    source_path = tmp_path / f"{code}.pdf"
    source_path.write_text("pdf", encoding="utf-8")
    display_name = source_path.stem if variant is None else f"{code}-{variant}"
    return TemplateRecord(
        code=code,
        variant=variant,
        display_name=display_name,
        source_type="pdf",
        source_path=source_path,
    )


def build_controller(
    tmp_path: Path,
    qapp: QApplication,
    *,
    scanner_adapter=None,
) -> tuple[MainWindow, AppController, FakeTemplateRepository]:
    code = "600004075219"
    window = MainWindow()
    window.show()
    qapp.processEvents()

    repository = FakeTemplateRepository({code: [make_template(tmp_path, code)]})
    controller = AppController(
        view=window,
        template_repository=repository,
        detection_service=FakeDetectionService(),
        camera_adapter=FakeCameraAdapter(),
        scanner_adapter=scanner_adapter,
    )
    controller._resolve_template_preview_path = lambda template: None
    return window, controller, repository


def test_handle_scanned_code_updates_view_and_queries_templates(tmp_path, qapp):
    window, controller, repository = build_controller(tmp_path, qapp)

    try:
        controller.handle_scanned_code(" 600004075219\r\n")

        assert window.code_text() == "600004075219"
        assert repository.refresh_calls == 1
        assert window.selected_template() is not None
        assert "已自动选中模板" in window.status_value.text()
    finally:
        window.close()


def test_handle_scanned_code_is_ignored_while_detection_is_running(tmp_path, qapp):
    window, controller, _repository = build_controller(tmp_path, qapp)

    try:
        controller._thread = object()
        controller.handle_scanned_code("600004075219")

        assert window.code_text() == ""
        assert window.status_value.text() == "检测进行中，已忽略扫码输入"
    finally:
        window.close()


def test_mock_scanner_adapter_signal_drives_controller(tmp_path, qapp):
    adapter = MockScannerAdapter()
    window, _controller, repository = build_controller(
        tmp_path,
        qapp,
        scanner_adapter=adapter,
    )

    try:
        adapter.emit_code("600004075219\n")
        qapp.processEvents()

        assert repository.refresh_calls == 1
        assert window.code_text() == "600004075219"
        assert window.selected_template() is not None
    finally:
        window.close()


def test_set_busy_disables_code_input(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    try:
        window.set_busy(True)
        assert window.code_input.isEnabled() is False

        window.set_busy(False)
        assert window.code_input.isEnabled() is True
    finally:
        window.close()


def test_run_detection_uses_output_mode_toggle(tmp_path, qapp):
    window, controller, _repository = build_controller(tmp_path, qapp)
    captured_requests = []
    target_path = tmp_path / "target.jpg"
    target_path.write_bytes(b"not-an-image")

    try:
        controller.handle_scanned_code("600004075219")
        window.set_target_image_path(target_path)
        controller._start_worker = captured_requests.append

        controller.run_detection()
        assert captured_requests[-1].output_mode == "final"

        window.detailed_output_checkbox.setChecked(True)
        controller.run_detection()
        assert captured_requests[-1].output_mode == "debug"
    finally:
        window.close()
