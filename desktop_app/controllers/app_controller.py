"""Main application controller for the desktop UI."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, Signal

from desktop_app.devices.camera.base import CameraAdapter
from desktop_app.devices.scanner.base import ScannerAdapter
from desktop_app.devices.scanner_input import KeyboardWedgeScannerInput
from desktop_app.models import DetectionJobRequest, DetectionJobResult, HistoryRecord
from desktop_app.repositories.annotation_repository import AnnotationRepository
from desktop_app.repositories.history_repository import HistoryRepository
from desktop_app.repositories.result_repository import DetectionResultRepository
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.services.preview_service import PreviewService
from desktop_app.ui.main_window import MainWindow
from desktop_app.workers.camera_preview_worker import (
    CAMERA_ROTATION_DEGREES,
    CameraFrameBrightness,
    CameraPreviewWorker,
    normalize_camera_rotation_degrees,
)
from desktop_app.workers.detection_worker import DetectionWorker


CAMERA_NO_CAMERA = "NO_CAMERA"
CAMERA_STARTING = "STARTING"
CAMERA_PREVIEWING = "PREVIEWING"
CAMERA_CAPTURING = "CAPTURING"
CAMERA_CAPTURED = "CAPTURED"
CAMERA_DETECTING = "DETECTING"
CAMERA_RESULT_READY = "RESULT_READY"
CAMERA_ROTATION_SETTINGS_KEY = "camera/rotation_degrees"
SETTINGS_APPLICATION = "gree-label-detection"
SETTINGS_ORGANIZATION = "gree"


def _camera_brightness_label(
    brightness: CameraFrameBrightness,
) -> tuple[str, str]:
    mean = brightness.mean
    if brightness.is_enhanced:
        suffix = f"已增亮 {brightness.enhancement_factor:.1f}x"
        if brightness.level == "too_dark":
            return f"偏暗 {suffix}", "warning"
        if brightness.level == "dim":
            return f"略暗 {suffix}", "neutral"

    if brightness.level == "too_dark":
        return f"偏暗 {mean:.0f}", "warning"
    if brightness.level == "dim":
        return f"略暗 {mean:.0f}", "neutral"
    return f"正常 {mean:.0f}", "ok"


class AppController(QObject):
    """Own the high-level interaction flow for the first desktop version."""

    _save_preview_frame_requested = Signal(object)
    _stop_preview_requested = Signal()
    _camera_rotation_changed = Signal(int)

    def __init__(
        self,
        view: MainWindow,
        template_repository: TemplateRepository,
        detection_service: DetectionService,
        camera_adapter: CameraAdapter,
        scanner_adapter: ScannerAdapter | None = None,
        preview_service: PreviewService | None = None,
        history_repository: HistoryRepository | None = None,
        annotation_repository: AnnotationRepository | None = None,
        result_repository: DetectionResultRepository | None = None,
        auto_start_camera_preview: bool = True,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(view)
        self.view = view
        self.template_repository = template_repository
        self.detection_service = detection_service
        self.camera_adapter = camera_adapter
        self.scanner_adapter = scanner_adapter
        self.preview_service = preview_service
        self.history_repository = history_repository
        self.annotation_repository = annotation_repository
        self.result_repository = result_repository or DetectionResultRepository()
        self._thread: QThread | None = None
        self._worker: DetectionWorker | None = None
        self._camera_preview_thread: QThread | None = None
        self._camera_preview_worker: CameraPreviewWorker | None = None
        self._camera_workflow_state = CAMERA_NO_CAMERA
        self._latest_camera_brightness: CameraFrameBrightness | None = None
        self._last_camera_brightness_level: str | None = None
        self._detection_started_at: float | None = None
        self._auto_start_camera_preview = auto_start_camera_preview
        self.settings = settings or QSettings(
            SETTINGS_ORGANIZATION,
            SETTINGS_APPLICATION,
        )
        self._camera_rotation_degrees = self._load_camera_rotation_degrees()
        self._connect_signals()
        self._connect_scanner_signals()
        self._load_history_records()
        self.view.set_camera_backend_name(self.camera_adapter.name)
        self.view.set_camera_rotation_degrees(self._camera_rotation_degrees)
        self.view.show_pending_result()
        self._sync_camera_actions()
        self._refresh_detection_ready_state()
        self._start_scanner_adapter()
        if self._auto_start_camera_preview:
            self.start_camera_preview()

    def _connect_signals(self) -> None:
        self.view.manual_query_requested.connect(self.handle_manual_query)
        self.view.simulate_scan_requested.connect(self.handle_scan_from_input)
        self.view.browse_target_requested.connect(self.browse_target_image)
        self.view.capture_camera_requested.connect(self.capture_camera_image)
        self.view.camera_rotation_requested.connect(self.rotate_camera_clockwise)
        self.view.next_target_requested.connect(self.prepare_next_target)
        self.view.run_detection_requested.connect(self.run_detection)
        self.view.save_annotation_requested.connect(self.save_current_annotation)
        self.view.history_selection_changed.connect(self.load_history_result)
        self.view.template_selection_changed.connect(self.refresh_template_preview)
        self.view.destroyed.connect(self._stop_camera_preview)

    def _connect_scanner_signals(self) -> None:
        if self.scanner_adapter is None:
            return

        self.scanner_adapter.code_scanned.connect(self.handle_scanned_code)
        if hasattr(self.scanner_adapter, "scan_buffer_changed"):
            self.scanner_adapter.scan_buffer_changed.connect(
                self._handle_scanner_buffer_changed
            )
        self.scanner_adapter.availability_changed.connect(
            self._handle_scanner_availability_changed
        )
        self.view.destroyed.connect(self._stop_scanner_adapter)

    def _start_scanner_adapter(self) -> None:
        if self.scanner_adapter is None:
            return

        try:
            self.scanner_adapter.start()
        except Exception as exc:
            self.view.set_status(f"扫码器启动失败: {exc}")

    def _stop_scanner_adapter(self, *_args) -> None:
        if self.scanner_adapter is None:
            return

        try:
            self.scanner_adapter.code_scanned.disconnect(self.handle_scanned_code)
        except (RuntimeError, TypeError):
            pass

        if hasattr(self.scanner_adapter, "scan_buffer_changed"):
            try:
                self.scanner_adapter.scan_buffer_changed.disconnect(
                    self._handle_scanner_buffer_changed
                )
            except (RuntimeError, TypeError):
                pass

        try:
            self.scanner_adapter.availability_changed.disconnect(
                self._handle_scanner_availability_changed
            )
        except (RuntimeError, TypeError):
            pass

        try:
            self.scanner_adapter.stop()
        except Exception:
            return

    def query_templates(self) -> None:
        self.handle_manual_query()

    def handle_manual_query(self) -> None:
        self._handle_code_lookup(self.view.code_text(), source="manual")

    def handle_scan_from_input(self) -> None:
        self.handle_scanned_code(self.view.code_text(), source="input")

    def handle_scanned_code(self, code: str, source: str = "scanner") -> None:
        self._handle_code_lookup(code, source=source)

    def _handle_scanner_buffer_changed(self, raw_code: str) -> None:
        if self._is_detection_running():
            return

        self.view.set_code_text(KeyboardWedgeScannerInput.normalize(raw_code))

    def _handle_code_lookup(self, raw_code: str, source: str) -> None:
        if self._is_detection_running():
            if source == "manual":
                self.view.set_status("检测正在运行，请等待")
            else:
                self.view.set_status("检测进行中，已忽略扫码输入")
            return

        code = KeyboardWedgeScannerInput.normalize(raw_code)
        self.view.set_code_text(code)
        self._query_templates_for_code(code)
        self._restore_code_focus(select_all=True)

    def _query_templates_for_code(self, code: str) -> None:
        if not code:
            self.view.set_templates([])
            self.view.clear_template_image()
            self.view.set_status("请输入编码")
            self.view.show_pending_result()
            self._refresh_detection_ready_state()
            return

        self.template_repository.refresh()
        templates = self.template_repository.find_by_code(code)
        self.view.set_templates(templates)
        if not templates:
            self.view.clear_template_image()
            self.view.set_status("未找到对应模板")
            self.view.show_pending_result()
            self._refresh_detection_ready_state()
            return

        if len(templates) == 1:
            self.view.set_status("找到 1 个模板候选，已自动选中模板")
            self._refresh_detection_ready_state()
            return

        self.view.set_status(f"找到 {len(templates)} 个模板候选，请确认模板版本")
        self._refresh_detection_ready_state()

    def _handle_scanner_availability_changed(self, available: bool, message: str) -> None:
        if not message:
            return

        prefix = "扫码器已连接" if available else "扫码器不可用"
        if message == prefix:
            self.view.set_status(message)
            return

        self.view.set_status(f"{prefix}: {message}")

    def browse_target_image(self) -> None:
        file_path = self.view.choose_image_file()
        if not file_path:
            return
        self.view.set_target_image_path(file_path)
        self.view.set_camera_quality("手动图片", "neutral")
        self.view.show_pending_result()
        self.view.set_status("已选择目标图片")
        self._camera_workflow_state = CAMERA_CAPTURED
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def capture_camera_image(self) -> None:
        if self._is_detection_running():
            self.view.set_status("检测正在运行，请等待")
            return

        if self._camera_workflow_state == CAMERA_NO_CAMERA:
            self.start_camera_preview(restart=True)
            return

        if self._camera_workflow_state in {CAMERA_STARTING, CAMERA_CAPTURING}:
            self.view.set_status("相机正在准备，请等待")
            return

        if self._camera_workflow_state in {CAMERA_CAPTURED, CAMERA_RESULT_READY}:
            self.retake_target_image()
            return

        if self._camera_workflow_state != CAMERA_PREVIEWING:
            self.view.set_status("相机预览尚未就绪")
            return

        code = KeyboardWedgeScannerInput.normalize(self.view.code_text())
        self._camera_workflow_state = CAMERA_CAPTURING
        self.view.set_status("拍照保存中")
        self._sync_camera_actions()
        self._save_preview_frame_requested.emit(code or None)

    def rotate_camera_clockwise(self) -> None:
        active_degrees = self._camera_rotation_degrees
        current_index = CAMERA_ROTATION_DEGREES.index(active_degrees)
        next_degrees = CAMERA_ROTATION_DEGREES[
            (current_index + 1) % len(CAMERA_ROTATION_DEGREES)
        ]
        self._set_camera_rotation_degrees(next_degrees)

    def _handle_camera_photo_saved(self, captured: object) -> None:
        captured_path = Path(captured)
        self.view.set_target_image_path(captured_path)
        self.view.show_pending_result()
        if self._latest_camera_brightness and self._latest_camera_brightness.is_enhanced:
            self.view.set_status("已拍照保存，原始画面偏暗，已使用增亮图检测")
        elif self._latest_camera_brightness and self._latest_camera_brightness.is_too_dark:
            self.view.set_status(
                "已拍照保存，但画面偏暗，建议补光或提高曝光后重拍"
            )
        else:
            self.view.set_status("已拍照保存，目标图已锁定")
        self._camera_workflow_state = CAMERA_CAPTURED
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def _handle_camera_preview_failed(self, message: str) -> None:
        if message.startswith("未检测到相机"):
            self.view.set_status("未检测到相机")
            self.view.clear_target_image("未检测到相机")
            self.view.set_camera_quality("未检测到相机", "error")
            self._camera_workflow_state = CAMERA_NO_CAMERA
        elif self._camera_workflow_state == CAMERA_CAPTURING:
            self.view.set_status("拍照保存失败")
            self._camera_workflow_state = (
                CAMERA_PREVIEWING if self._is_camera_preview_running() else CAMERA_NO_CAMERA
            )
        elif self._camera_workflow_state in {
            CAMERA_CAPTURED,
            CAMERA_DETECTING,
            CAMERA_RESULT_READY,
        }:
            self.view.set_status("相机预览已断开，当前目标图已保留")
        else:
            self.view.set_status("相机预览失败")
            self.view.clear_target_image("相机预览失败")
            self.view.set_camera_quality("预览失败", "error")
            self._camera_workflow_state = CAMERA_NO_CAMERA
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def start_camera_preview(self, *, restart: bool = False) -> None:
        if self._is_detection_running():
            return
        if restart and self._is_camera_preview_running():
            self._stop_camera_preview()
            return
        if self._is_camera_preview_running():
            self._camera_workflow_state = CAMERA_PREVIEWING
            self._sync_camera_actions()
            return

        self.view.clear_target_image("正在检查相机...")
        self.view.set_camera_quality("检查中", "neutral")
        self._latest_camera_brightness = None
        self._last_camera_brightness_level = None
        self._camera_workflow_state = CAMERA_STARTING
        self._sync_camera_actions()
        self.view.set_status("正在检查相机")

        thread = QThread(self.view)
        worker = CameraPreviewWorker(
            self.camera_adapter,
            rotation_degrees=self._camera_rotation_degrees,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.start)
        worker.started_status.connect(self.view.set_status)
        worker.frame_ready.connect(self._handle_camera_preview_frame)
        worker.frame_brightness_changed.connect(self._handle_camera_frame_brightness)
        worker.photo_saved.connect(self._handle_camera_photo_saved)
        worker.failed.connect(self._handle_camera_preview_failed)
        worker.stopped.connect(thread.quit)
        worker.stopped.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_active_camera_preview_worker)
        self._save_preview_frame_requested.connect(worker.save_current_frame)
        self._camera_rotation_changed.connect(worker.set_rotation_degrees)
        self._stop_preview_requested.connect(worker.stop_preview)

        self._camera_preview_thread = thread
        self._camera_preview_worker = worker
        thread.start()

    def _handle_camera_preview_frame(self, image: object) -> None:
        started_previewing = self._camera_workflow_state in {
            CAMERA_STARTING,
            CAMERA_NO_CAMERA,
        }
        if self._camera_workflow_state in {CAMERA_STARTING, CAMERA_NO_CAMERA}:
            self._camera_workflow_state = CAMERA_PREVIEWING
            self._sync_camera_actions()
            self._refresh_detection_ready_state()
        if self._camera_workflow_state == CAMERA_PREVIEWING:
            self.view.set_target_preview_image(image)
            if started_previewing and self._latest_camera_brightness is not None:
                self._sync_camera_brightness_status(self._latest_camera_brightness)

    def _handle_camera_frame_brightness(self, brightness: object) -> None:
        if not isinstance(brightness, CameraFrameBrightness):
            return

        self._latest_camera_brightness = brightness
        if self._camera_workflow_state in {
            CAMERA_CAPTURED,
            CAMERA_DETECTING,
            CAMERA_RESULT_READY,
        }:
            return

        label, severity = _camera_brightness_label(brightness)
        self.view.set_camera_quality(label, severity)
        self._sync_camera_brightness_status(brightness)

    def _sync_camera_brightness_status(
        self,
        brightness: CameraFrameBrightness,
    ) -> None:
        if self._camera_workflow_state != CAMERA_PREVIEWING:
            return

        if brightness.is_enhanced:
            if self._last_camera_brightness_level != "enhanced":
                self.view.set_status("原始画面偏暗，预览和保存已自动增亮")
                self._last_camera_brightness_level = "enhanced"
            return

        if brightness.is_too_dark and self._last_camera_brightness_level != "too_dark":
            self.view.set_status("画面偏暗，建议补光或提高曝光后再拍照")
        elif (
            not brightness.is_too_dark
            and self._last_camera_brightness_level in {"too_dark", "enhanced"}
        ):
            self.view.set_status("相机预览中")

        self._last_camera_brightness_level = brightness.level

    def retake_target_image(self) -> None:
        self.view.clear_target_image("等待相机预览")
        self.view.show_pending_result()
        if self._is_camera_preview_running():
            self._camera_workflow_state = CAMERA_PREVIEWING
            self.view.set_status("相机预览中，请重新拍照")
        else:
            self.start_camera_preview(restart=True)
            return
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def prepare_next_target(self) -> None:
        if self._is_detection_running():
            self.view.set_status("检测正在运行，请等待")
            return
        self.view.clear_target_image("等待相机预览")
        self.view.show_pending_result()
        if self._is_camera_preview_running():
            self._camera_workflow_state = CAMERA_PREVIEWING
            self.view.set_status("请放入下一张标签，确认画面后拍照保存")
        else:
            self.start_camera_preview(restart=True)
            return
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def refresh_template_preview(self) -> None:
        template = self.view.selected_template()
        if template is None:
            self.view.clear_template_image()
            self.view.show_pending_result()
            self._refresh_detection_ready_state()
            return

        preview_path = self._resolve_template_preview_path(template)
        if preview_path is None:
            self.view.clear_template_image()
        else:
            self.view.set_template_image_path(preview_path)

        self.view.show_pending_result()
        self._refresh_detection_ready_state()

    def run_detection(self) -> None:
        if self._is_detection_running():
            self.view.set_status("检测正在运行，请等待")
            return
        if self._camera_workflow_state == CAMERA_CAPTURING:
            self.view.set_status("拍照保存中，请等待")
            return

        template = self.view.selected_template()
        if template is None:
            self.view.show_error("请先查询并选择一个模板")
            return
        if not template.source_path.exists():
            self.view.show_error("当前模板路径不存在，请重新查询模板")
            self._refresh_detection_ready_state()
            return

        target_image_path = self.view.target_image_path()
        if not target_image_path:
            self.view.show_error("请先拍照保存目标图片，或选择一张目标图片")
            return
        target_path = Path(target_image_path)
        if not target_path.exists():
            self.view.show_error("目标图片不存在，请重新选择图片")
            self._refresh_detection_ready_state()
            return

        request = DetectionJobRequest(
            template=template,
            target_image_path=target_path,
            output_mode="debug" if self.view.detailed_output_enabled() else "final",
        )
        self._start_worker(request)

    def _start_worker(self, request: DetectionJobRequest) -> None:
        thread = QThread(self.view)
        worker = DetectionWorker(self.detection_service, request)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.started_status.connect(self.view.set_status)
        worker.finished.connect(self._handle_detection_finished)
        worker.failed.connect(self._handle_detection_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_active_worker)

        self._thread = thread
        self._worker = worker
        self._detection_started_at = time.monotonic()
        self._camera_workflow_state = CAMERA_DETECTING
        self._sync_camera_actions()
        self.view.set_busy(True)
        self.view.set_status("检测中")
        self.view.show_running_result()
        thread.start()

    def _handle_detection_finished(self, result: DetectionJobResult) -> None:
        duration_seconds = self._finish_detection_timer()
        self.view.set_busy(False)
        self.view.set_status("检测完成")
        self.view.show_detection_result(result)
        if self.annotation_repository is not None:
            annotation = self.annotation_repository.build_from_detection_result(result)
            annotation["detection_duration_seconds"] = duration_seconds
            self.view.show_annotation(annotation)
        self.view.set_detection_duration_seconds(duration_seconds)
        record = HistoryRecord(
            created_at=datetime.now().isoformat(timespec="seconds"),
            code=result.code or self.view.code_text(),
            template_name=result.template_display_name or Path(
                result.template_path or ""
            ).stem,
            verdict=result.verdict or "-",
            output_dir=result.output_dir,
            target_image_path=result.target_image_path or Path(self.view.target_image_path()),
            visualization_path=result.visualization_path,
            summary_text=result.summary_text,
        )
        if self.history_repository is not None:
            try:
                self.history_repository.append(record)
            except OSError as exc:
                print(f"[AppController] failed to persist history: {exc}")
        self.view.prepend_history_record(record)
        self._camera_workflow_state = CAMERA_RESULT_READY
        self._sync_camera_actions()
        self._restore_code_focus(select_all=True)
        self._refresh_detection_ready_state()

    def save_current_annotation(self) -> None:
        if self.annotation_repository is None:
            self.view.set_status("标注存储未启用")
            return

        annotation = self.view.current_annotation_payload()
        if annotation is None:
            self.view.set_status("没有可保存的标注")
            return

        try:
            path = self.annotation_repository.save(annotation)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.view.show_error(f"保存标注失败: {exc}")
            return

        self.view.set_annotation_saved_path(path)
        self.view.set_status("人工标注已保存")

    def load_history_result(self, record: object) -> None:
        if not isinstance(record, HistoryRecord):
            self.view.clear_annotation()
            return
        if not record.output_dir.exists():
            self.view.set_status(f"历史结果目录不存在: {record.output_dir}")
            self.view.clear_annotation()
            return

        restored = self.result_repository.build_from_history_record(record)
        if restored is None:
            self.view.set_status(f"历史结果文件不存在: {record.output_dir}")
            self.view.clear_annotation()
            return

        result, duration_seconds = restored
        self.view.set_code_text(record.code)
        if record.target_image_path.exists():
            self.view.set_target_image_path(record.target_image_path)
        else:
            self.view.target_path_input.setText(str(record.target_image_path))
        self.view.show_detection_result(result)
        self.view.set_detection_duration_seconds(duration_seconds)
        if self.annotation_repository is not None:
            annotation = self.annotation_repository.build_from_history_record(record)
            if duration_seconds is not None:
                annotation["detection_duration_seconds"] = duration_seconds
            self.view.show_annotation(annotation)
        self.view.set_status("已加载历史结果")
        self._camera_workflow_state = CAMERA_RESULT_READY
        self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def load_history_annotation(self, record: object) -> None:
        self.load_history_result(record)

    def _handle_detection_failed(self, message: str) -> None:
        duration_seconds = self._finish_detection_timer()
        self.view.set_busy(False)
        self.view.set_status("检测失败")
        self.view.show_failure_result(message)
        self.view.set_detection_duration_seconds(duration_seconds)
        self.view.show_error(message)
        self._camera_workflow_state = CAMERA_RESULT_READY
        self._sync_camera_actions()
        self._restore_code_focus(select_all=True)
        self._refresh_detection_ready_state()

    def _clear_active_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._refresh_detection_ready_state()

    def _is_detection_running(self) -> bool:
        return self._thread is not None

    def _finish_detection_timer(self) -> float | None:
        if self._detection_started_at is None:
            return None

        duration_seconds = time.monotonic() - self._detection_started_at
        self._detection_started_at = None
        return duration_seconds

    def _clear_active_camera_preview_worker(self) -> None:
        self._camera_preview_thread = None
        self._camera_preview_worker = None
        if self._camera_workflow_state in {CAMERA_STARTING, CAMERA_PREVIEWING}:
            self._camera_workflow_state = CAMERA_NO_CAMERA
            self._sync_camera_actions()
        self._refresh_detection_ready_state()

    def _is_camera_preview_running(self) -> bool:
        return self._camera_preview_thread is not None

    def _stop_camera_preview(self, *_args) -> None:
        if self._camera_preview_worker is None:
            return
        self._stop_preview_requested.emit()

    def _load_camera_rotation_degrees(self) -> int:
        return normalize_camera_rotation_degrees(
            self.settings.value(CAMERA_ROTATION_SETTINGS_KEY, 0)
        )

    def _set_camera_rotation_degrees(self, rotation_degrees: int) -> None:
        normalized_degrees = normalize_camera_rotation_degrees(rotation_degrees)
        self._camera_rotation_degrees = normalized_degrees
        self.settings.setValue(CAMERA_ROTATION_SETTINGS_KEY, normalized_degrees)
        self.settings.sync()
        self.view.set_camera_rotation_degrees(normalized_degrees)
        self._camera_rotation_changed.emit(normalized_degrees)

    def _sync_camera_actions(self) -> None:
        state = self._camera_workflow_state
        if state == CAMERA_NO_CAMERA:
            self.view.set_capture_action("重连相机", True)
            self.view.set_next_enabled(False)
        elif state == CAMERA_STARTING:
            self.view.set_capture_action("检查中", False)
            self.view.set_next_enabled(False)
        elif state == CAMERA_PREVIEWING:
            self.view.set_capture_action("拍照保存", True)
            self.view.set_next_enabled(False)
        elif state == CAMERA_CAPTURING:
            self.view.set_capture_action("保存中", False)
            self.view.set_next_enabled(False)
        elif state == CAMERA_DETECTING:
            self.view.set_capture_action("检测中", False)
            self.view.set_next_enabled(False)
        elif state == CAMERA_RESULT_READY:
            self.view.set_capture_action("重拍", True)
            self.view.set_next_enabled(True)
        else:
            self.view.set_capture_action("重拍", True)
            self.view.set_next_enabled(False)

    def _restore_code_focus(self, *, select_all: bool) -> None:
        self.view.focus_code_input(select_all=select_all)

    def _load_history_records(self) -> None:
        if self.history_repository is None:
            self.view.set_history_records([])
            return
        self.view.set_history_records(self.history_repository.list_recent(limit=20))

    def _refresh_detection_ready_state(self) -> None:
        target_path = self.view.target_image_path()
        ready = (
            self.view.selected_template() is not None
            and bool(target_path)
            and Path(target_path).exists()
            and not self._is_detection_running()
            and self._camera_workflow_state != CAMERA_CAPTURING
        )
        self.view.set_detection_enabled(bool(ready))

    def _resolve_template_preview_path(self, template) -> Path | None:
        if self.preview_service is None:
            if template.preview_path and template.preview_path.exists():
                return template.preview_path.resolve()
            return None
        return self.preview_service.resolve_preview_path(template)
