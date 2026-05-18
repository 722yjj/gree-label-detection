import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import QApplication, QScrollArea

from desktop_app.controllers.app_controller import AppController
from desktop_app.controllers import app_controller as app_controller_module
from desktop_app.devices.scanner.keyboard_wedge_adapter import KeyboardWedgeScannerAdapter
from desktop_app.devices.scanner.mock import MockScannerAdapter
from desktop_app.models import DetectionJobResult, TemplateRecord
from desktop_app.ui.main_window import MainWindow, ScaledImageLabel
from desktop_app.workers.camera_preview_worker import (
    enhance_frame_for_detection,
    measure_frame_brightness,
)


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


class FakePreviewSession:
    display_name = "fake-preview-camera"

    def __init__(self, output_dir: Path, frame=None) -> None:
        self.output_dir = output_dir
        if frame is None:
            frame = np.full((48, 64, 3), 180, dtype=np.uint8)
        self.frame = frame
        self.closed = False
        self.saved_code = None

    def read_frame_bgr(self):
        return self.frame.copy()

    def save_frame_bgr(self, frame, preferred_code: str | None = None) -> Path:
        self.saved_code = preferred_code
        output_path = self.output_dir / f"{preferred_code or 'uncoded'}.jpg"
        assert cv2.imwrite(str(output_path), frame)
        return output_path

    def close(self) -> None:
        self.closed = True


class FakePreviewCameraAdapter:
    name = "fake-preview-camera"

    def __init__(self, output_dir: Path, frame=None) -> None:
        self.session = FakePreviewSession(output_dir, frame=frame)

    def start_preview(self):
        return self.session

    def capture(self, preferred_code: str | None = None) -> Path | None:
        raise AssertionError("single capture should not run during preview workflow")


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


def wait_until(qapp: QApplication, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    qapp.processEvents()
    assert predicate()


class DirectKeyboardWedgeScannerAdapter(KeyboardWedgeScannerAdapter):
    """Exercise keyboard-wedge event handling without a process-global filter."""

    def start(self) -> None:
        self._application = QApplication.instance()
        self._started = True
        self._available = True
        self.availability_changed.emit(True, "键盘扫码模式已启用")

    def stop(self) -> None:
        self._application = None
        self._buffer.clear()
        self._started = False
        self._available = False
        self.availability_changed.emit(False, "键盘扫码模式已停止")


def send_keyboard_scan(
    adapter: KeyboardWedgeScannerAdapter,
    qapp: QApplication,
    target,
    value: str,
    *,
    enter_count: int = 1,
) -> None:
    for character in value:
        if character == "\t":
            event = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Tab,
                Qt.KeyboardModifier.NoModifier,
                "\t",
            )
        else:
            event = QKeyEvent(
                QEvent.Type.KeyPress,
                0,
                Qt.KeyboardModifier.NoModifier,
                character,
            )
        adapter.eventFilter(target, event)
        qapp.processEvents()

    for _ in range(enter_count):
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.NoModifier,
            "\r",
        )
        adapter.eventFilter(target, event)
        qapp.processEvents()


def build_controller(
    tmp_path: Path,
    qapp: QApplication,
    *,
    camera_adapter=None,
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
        camera_adapter=camera_adapter or FakeCameraAdapter(),
        scanner_adapter=scanner_adapter,
        auto_start_camera_preview=False,
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
        adapter.stop()
        window.close()


def test_keyboard_wedge_adapter_captures_scan_without_code_input_focus(tmp_path, qapp):
    adapter = DirectKeyboardWedgeScannerAdapter()
    window, _controller, repository = build_controller(
        tmp_path,
        qapp,
        scanner_adapter=adapter,
    )
    adapter._active_window = window

    try:
        window.template_list.setFocus()
        window.activateWindow()
        qapp.processEvents()

        send_keyboard_scan(adapter, qapp, window.template_list, "600004075219")

        assert repository.refresh_calls == 1
        assert window.code_text() == "600004075219"
        assert window.selected_template() is not None
        assert window.code_input.hasSelectedText() is True
    finally:
        adapter.stop()
        window.close()


def test_keyboard_wedge_adapter_replaces_existing_code_instead_of_appending(
    tmp_path,
    qapp,
):
    adapter = DirectKeyboardWedgeScannerAdapter()
    window, _controller, repository = build_controller(
        tmp_path,
        qapp,
        scanner_adapter=adapter,
    )
    adapter._active_window = window

    try:
        window.set_code_text("OLD-CODE")
        window.template_list.setFocus()
        window.activateWindow()
        qapp.processEvents()

        send_keyboard_scan(adapter, qapp, window.template_list, "600004075219")

        assert repository.refresh_calls == 1
        assert window.code_text() == "600004075219"
        assert "OLD-CODE" not in window.code_text()
    finally:
        adapter.stop()
        window.close()


def test_keyboard_wedge_adapter_normalizes_scan_affixes(tmp_path, qapp):
    adapter = DirectKeyboardWedgeScannerAdapter()
    window, _controller, repository = build_controller(
        tmp_path,
        qapp,
        scanner_adapter=adapter,
    )
    adapter._active_window = window

    try:
        window.template_list.setFocus()
        window.activateWindow()
        qapp.processEvents()

        send_keyboard_scan(
            adapter,
            qapp,
            window.template_list,
            "\t 600004075219 \n",
            enter_count=2,
        )

        assert repository.refresh_calls == 1
        assert window.code_text() == "600004075219"
        assert window.selected_template() is not None
    finally:
        adapter.stop()
        window.close()


def test_keyboard_wedge_scan_is_ignored_while_detection_is_running(tmp_path, qapp):
    adapter = DirectKeyboardWedgeScannerAdapter()
    window, controller, repository = build_controller(
        tmp_path,
        qapp,
        scanner_adapter=adapter,
    )
    adapter._active_window = window

    try:
        window.set_code_text("600004075219")
        controller._thread = object()
        window.template_list.setFocus()
        window.activateWindow()
        qapp.processEvents()

        send_keyboard_scan(adapter, qapp, window.template_list, "999999999999")

        assert repository.refresh_calls == 0
        assert window.code_text() == "600004075219"
        assert window.status_value.text() == "检测进行中，已忽略扫码输入"
    finally:
        controller._thread = None
        adapter.stop()
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


def test_main_window_displays_camera_backend_name(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    try:
        window.set_camera_backend_name("hikrobot-mvs")

        assert window.camera_backend_value.text() == "相机：hikrobot-mvs"
    finally:
        window.close()


def test_main_window_displays_camera_quality(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    try:
        window.set_camera_quality("偏暗 53", "warning")

        assert window.camera_quality_value.text() == "画面：偏暗 53"
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


def test_camera_preview_reports_no_connected_camera(tmp_path, qapp):
    class DisconnectedCameraAdapter:
        name = "hikrobot-mvs"

        def start_preview(self):
            raise RuntimeError("当前无可见设备")

        def capture(self, preferred_code: str | None = None) -> Path | None:
            raise AssertionError("capture should not run when camera is disconnected")

    window, controller, _repository = build_controller(
        tmp_path,
        qapp,
        camera_adapter=DisconnectedCameraAdapter(),
    )

    try:
        controller.capture_camera_image()
        wait_until(qapp, lambda: controller._camera_preview_thread is None)

        assert window.status_value.text() == "未检测到相机"
        assert window.capture_button.text() == "重连相机"
        assert window.target_image_path() == ""
    finally:
        window.close()


def test_camera_preview_photo_save_locks_target_then_next_resumes_preview(tmp_path, qapp):
    camera_adapter = FakePreviewCameraAdapter(tmp_path)
    window, controller, _repository = build_controller(
        tmp_path,
        qapp,
        camera_adapter=camera_adapter,
    )

    try:
        controller.handle_scanned_code("600004075219")
        controller.start_camera_preview()
        wait_until(qapp, lambda: controller._camera_workflow_state == "PREVIEWING")

        assert window.capture_button.text() == "拍照保存"
        assert window.run_button.isEnabled() is False

        controller.capture_camera_image()
        wait_until(qapp, lambda: bool(window.target_image_path()))

        saved_path = Path(window.target_image_path())
        assert saved_path.exists()
        assert saved_path.name == "600004075219.jpg"
        assert camera_adapter.session.saved_code == "600004075219"
        assert window.capture_button.text() == "重拍"
        assert window.run_button.isEnabled() is True

        controller.prepare_next_target()
        wait_until(qapp, lambda: window.target_image_path() == "")

        assert window.capture_button.text() == "拍照保存"
        assert window.next_button.isEnabled() is False
        assert window.run_button.isEnabled() is False
    finally:
        controller._stop_camera_preview()
        wait_until(qapp, lambda: controller._camera_preview_thread is None)
        window.close()


def test_camera_preview_enhances_dark_frame_before_preview_and_save(tmp_path, qapp):
    dark_frame = np.full((48, 64, 3), 45, dtype=np.uint8)
    camera_adapter = FakePreviewCameraAdapter(tmp_path, frame=dark_frame)
    window, controller, _repository = build_controller(
        tmp_path,
        qapp,
        camera_adapter=camera_adapter,
    )

    try:
        controller.start_camera_preview()
        wait_until(qapp, lambda: "已增亮" in window.camera_quality_value.text())

        assert "偏暗 已增亮" in window.camera_quality_value.text()
        assert window.status_value.text() == "原始画面偏暗，预览和保存已自动增亮"

        controller.capture_camera_image()
        wait_until(qapp, lambda: bool(window.target_image_path()))

        saved = cv2.imread(window.target_image_path())
        assert saved is not None
        assert float(saved.mean()) > 100.0
        assert window.status_value.text() == "已拍照保存，原始画面偏暗，已使用增亮图检测"
    finally:
        controller._stop_camera_preview()
        wait_until(qapp, lambda: controller._camera_preview_thread is None)
        window.close()


def test_measure_frame_brightness_classifies_dark_and_normal_frames():
    dark = measure_frame_brightness(np.full((12, 16, 3), 45, dtype=np.uint8))
    normal = measure_frame_brightness(np.full((12, 16, 3), 180, dtype=np.uint8))

    assert dark.level == "too_dark"
    assert dark.is_too_dark is True
    assert normal.level == "ok"
    assert normal.is_too_dark is False


def test_enhance_frame_for_detection_scales_dark_frame_to_readable_range():
    frame = np.full((12, 16, 3), 55, dtype=np.uint8)
    enhanced, brightness = enhance_frame_for_detection(frame)

    assert brightness.is_enhanced is True
    assert brightness.enhancement_factor > 1.0
    assert float(enhanced.mean()) > 120.0


def test_detection_result_keeps_locked_image_until_next_target(tmp_path, qapp):
    camera_adapter = FakePreviewCameraAdapter(tmp_path)
    window, controller, _repository = build_controller(
        tmp_path,
        qapp,
        camera_adapter=camera_adapter,
    )

    try:
        controller.handle_scanned_code("600004075219")
        controller.start_camera_preview()
        wait_until(qapp, lambda: controller._camera_workflow_state == "PREVIEWING")
        controller.capture_camera_image()
        wait_until(qapp, lambda: bool(window.target_image_path()))
        locked_path = Path(window.target_image_path())

        controller._handle_detection_finished(
            DetectionJobResult(
                success=True,
                verdict="通过",
                output_dir=tmp_path,
                summary_text="ok",
                target_image_path=locked_path,
                template_path=window.selected_template().source_path,
                code="600004075219",
                template_display_name="600004075219",
            )
        )

        assert window.target_image_path() == str(locked_path)
        assert window.capture_button.text() == "重拍"
        assert window.next_button.isEnabled() is True

        controller.prepare_next_target()

        assert window.target_image_path() == ""
        assert window.capture_button.text() == "拍照保存"
        assert window.next_button.isEnabled() is False
    finally:
        controller._stop_camera_preview()
        wait_until(qapp, lambda: controller._camera_preview_thread is None)
        window.close()


def test_scaled_image_label_rescales_pixmap_on_label_resize(tmp_path, qapp):
    image_path = tmp_path / "preview.jpg"
    image = QImage(1600, 900, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(image_path))

    label = ScaledImageLabel("placeholder")
    label.resize(500, 300)
    label.show()
    qapp.processEvents()

    try:
        label.set_preview_pixmap(QPixmap(str(image_path)))
        qapp.processEvents()
        first_pixmap = label.pixmap()

        label.resize(240, 120)
        qapp.processEvents()
        second_pixmap = label.pixmap()

        assert first_pixmap is not None
        assert second_pixmap is not None
        assert second_pixmap.width() <= label.width()
        assert second_pixmap.height() <= label.height()
        assert second_pixmap.width() < first_pixmap.width()
    finally:
        label.close()


def test_scaled_image_label_keeps_stable_size_hints_after_pixmap_set(tmp_path, qapp):
    image_path = tmp_path / "preview.jpg"
    image = QImage(1600, 900, QImage.Format.Format_RGB32)
    image.fill(QColor("blue"))
    assert image.save(str(image_path))

    label = ScaledImageLabel("placeholder")
    label.show()
    qapp.processEvents()

    try:
        before_minimum = label.minimumSizeHint()
        before_preferred = label.sizeHint()

        label.set_preview_pixmap(QPixmap(str(image_path)))
        qapp.processEvents()

        after_minimum = label.minimumSizeHint()
        after_preferred = label.sizeHint()

        assert after_minimum == before_minimum
        assert after_preferred == before_preferred
    finally:
        label.close()


def test_main_window_enables_standard_window_buttons(qapp):
    from PySide6.QtCore import Qt

    window = MainWindow()
    window.show()
    qapp.processEvents()

    try:
        flags = window.windowFlags()
        assert flags & Qt.WindowType.Window
        assert flags & Qt.WindowType.WindowSystemMenuHint
        assert flags & Qt.WindowType.WindowMinimizeButtonHint
        assert flags & Qt.WindowType.WindowMaximizeButtonHint
        assert flags & Qt.WindowType.WindowCloseButtonHint
    finally:
        window.close()


def test_main_window_uses_scroll_area_for_overflow(qapp):
    window = MainWindow()
    window.resize(1180, 720)
    window.show()
    qapp.processEvents()

    try:
        central = window.centralWidget()
        assert isinstance(central, QScrollArea) is False

        scroll_area = window.right_lower_scroll_area
        assert isinstance(scroll_area, QScrollArea)
        assert scroll_area.widgetResizable() is True

        window.resize(1180, 640)
        qapp.processEvents()

        assert scroll_area.verticalScrollBar().maximum() > 0
    finally:
        window.close()


def test_main_window_uses_internal_scroll_areas_for_result_and_history(qapp):
    window = MainWindow()
    window.resize(1180, 720)
    window.show()
    qapp.processEvents()

    try:
        assert isinstance(window.result_scroll_area, QScrollArea)
        assert window.result_scroll_area.widgetResizable() is True
        assert window.result_scroll_area.verticalScrollBar().maximum() > 0

        assert isinstance(window.history_scroll_area, QScrollArea)
        assert window.history_scroll_area.widgetResizable() is True
        assert window.history_scroll_area.verticalScrollBar().maximum() > 0
    finally:
        window.close()


def test_main_window_displays_detection_duration_states(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    try:
        window.show_pending_result()
        assert window.detection_duration_value.text() == "-"

        window.show_running_result()
        assert window.detection_duration_value.text() == "计时中"

        window.set_detection_duration_seconds(1.234)
        assert window.detection_duration_value.text() == "1.23 秒"
    finally:
        window.close()


def test_detection_finished_displays_elapsed_duration(tmp_path, qapp, monkeypatch):
    window, controller, _repository = build_controller(tmp_path, qapp)
    target_path = tmp_path / "target.jpg"
    target_path.write_bytes(b"target")

    try:
        controller.handle_scanned_code("600004075219")
        window.set_target_image_path(target_path)
        controller._detection_started_at = 10.0
        monkeypatch.setattr(app_controller_module.time, "monotonic", lambda: 12.345)

        controller._handle_detection_finished(
            DetectionJobResult(
                success=True,
                verdict="通过",
                output_dir=tmp_path,
                summary_text="ok",
                target_image_path=target_path,
                template_path=window.selected_template().source_path,
                code="600004075219",
                template_display_name="600004075219",
            )
        )

        assert window.detection_duration_value.text() == "2.35 秒"
    finally:
        window.close()


def test_detection_failed_displays_elapsed_duration(tmp_path, qapp, monkeypatch):
    window, controller, _repository = build_controller(tmp_path, qapp)

    try:
        controller._detection_started_at = 20.0
        monkeypatch.setattr(app_controller_module.time, "monotonic", lambda: 21.5)
        monkeypatch.setattr(window, "show_error", lambda _message: None)

        controller._handle_detection_failed("服务不可用")

        assert window.detection_duration_value.text() == "1.50 秒"
    finally:
        window.close()
