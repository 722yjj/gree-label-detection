"""Main application controller for the desktop UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread

from desktop_app.devices.camera.base import CameraAdapter
from desktop_app.devices.scanner.base import ScannerAdapter
from desktop_app.devices.scanner_input import KeyboardWedgeScannerInput
from desktop_app.models import DetectionJobRequest, DetectionJobResult, HistoryRecord
from desktop_app.repositories.history_repository import HistoryRepository
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.services.preview_service import PreviewService
from desktop_app.ui.main_window import MainWindow
from desktop_app.workers.detection_worker import DetectionWorker


class AppController(QObject):
    """Own the high-level interaction flow for the first desktop version."""

    def __init__(
        self,
        view: MainWindow,
        template_repository: TemplateRepository,
        detection_service: DetectionService,
        camera_adapter: CameraAdapter,
        scanner_adapter: ScannerAdapter | None = None,
        preview_service: PreviewService | None = None,
        history_repository: HistoryRepository | None = None,
    ) -> None:
        super().__init__(view)
        self.view = view
        self.template_repository = template_repository
        self.detection_service = detection_service
        self.camera_adapter = camera_adapter
        self.scanner_adapter = scanner_adapter
        self.preview_service = preview_service
        self.history_repository = history_repository
        self._thread: QThread | None = None
        self._worker: DetectionWorker | None = None
        self._connect_signals()
        self._connect_scanner_signals()
        self._load_history_records()
        self.view.show_pending_result()
        self._refresh_detection_ready_state()
        self._start_scanner_adapter()

    def _connect_signals(self) -> None:
        self.view.manual_query_requested.connect(self.handle_manual_query)
        self.view.simulate_scan_requested.connect(self.handle_scan_from_input)
        self.view.browse_target_requested.connect(self.browse_target_image)
        self.view.capture_mock_requested.connect(self.capture_mock_image)
        self.view.run_detection_requested.connect(self.run_detection)
        self.view.template_selection_changed.connect(self.refresh_template_preview)

    def _connect_scanner_signals(self) -> None:
        if self.scanner_adapter is None:
            return

        self.scanner_adapter.code_scanned.connect(self.handle_scanned_code)
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
        self.view.show_pending_result()
        self.view.set_status("已选择目标图片")
        self._refresh_detection_ready_state()

    def capture_mock_image(self) -> None:
        code = KeyboardWedgeScannerInput.normalize(self.view.code_text())
        captured = self.camera_adapter.capture(preferred_code=code or None)
        if not captured:
            self.view.show_error("未找到可用的 mock 相机样本图片")
            return

        self.view.set_target_image_path(captured)
        self.view.show_pending_result()
        self.view.set_status(f"已从 {self.camera_adapter.name} 获取样本图")
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
            self.view.show_error("请先选择目标图片或使用 Mock 相机取图")
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
        self.view.set_busy(True)
        self.view.set_status("检测中")
        self.view.show_running_result()
        thread.start()

    def _handle_detection_finished(self, result: DetectionJobResult) -> None:
        self.view.set_busy(False)
        self.view.set_status("检测完成")
        self.view.show_detection_result(result)
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
        self._restore_code_focus(select_all=True)
        self._refresh_detection_ready_state()

    def _handle_detection_failed(self, message: str) -> None:
        self.view.set_busy(False)
        self.view.set_status("检测失败")
        self.view.show_failure_result(message)
        self.view.show_error(message)
        self._restore_code_focus(select_all=True)
        self._refresh_detection_ready_state()

    def _clear_active_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._refresh_detection_ready_state()

    def _is_detection_running(self) -> bool:
        return self._thread is not None

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
        )
        self.view.set_detection_enabled(bool(ready))

    def _resolve_template_preview_path(self, template) -> Path | None:
        if self.preview_service is None:
            if template.preview_path and template.preview_path.exists():
                return template.preview_path.resolve()
            return None
        return self.preview_service.resolve_preview_path(template)
