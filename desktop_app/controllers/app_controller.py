"""Main application controller for the desktop UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread

from desktop_app.devices.camera.mock_camera import MockCameraAdapter
from desktop_app.devices.scanner_input import KeyboardWedgeScannerInput
from desktop_app.models import DetectionJobRequest, DetectionJobResult
from desktop_app.repositories.template_repository import TemplateRepository
from desktop_app.services.detection_service import DetectionService
from desktop_app.ui.main_window import MainWindow
from desktop_app.workers.detection_worker import DetectionWorker
from label_detection.core.config import PROJECT_ROOT
from label_detection.extraction.template_source import resolve_template_input


class AppController(QObject):
    """Own the high-level interaction flow for the first desktop version."""

    def __init__(
        self,
        view: MainWindow,
        template_repository: TemplateRepository,
        detection_service: DetectionService,
        camera_adapter: MockCameraAdapter,
    ) -> None:
        super().__init__(view)
        self.view = view
        self.template_repository = template_repository
        self.detection_service = detection_service
        self.camera_adapter = camera_adapter
        self._thread: QThread | None = None
        self._worker: DetectionWorker | None = None
        self._template_preview_cache: dict[Path, Path | None] = {}
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.view.query_code_requested.connect(self.query_templates)
        self.view.browse_target_requested.connect(self.browse_target_image)
        self.view.capture_mock_requested.connect(self.capture_mock_image)
        self.view.run_detection_requested.connect(self.run_detection)
        self.view.template_selection_changed.connect(self.refresh_template_preview)

    def query_templates(self) -> None:
        code = KeyboardWedgeScannerInput.normalize(self.view.code_text())
        self.view.set_code_text(code)
        if not code:
            self.view.set_templates([])
            self.view.set_status("请输入编码")
            return

        self.template_repository.refresh()
        templates = self.template_repository.find_by_code(code)
        self.view.set_templates(templates)
        if templates:
            self.view.set_status(f"找到 {len(templates)} 个模板候选")
            return

        self.view.clear_template_image()
        self.view.set_status("未找到对应模板")

    def browse_target_image(self) -> None:
        file_path = self.view.choose_image_file()
        if not file_path:
            return
        self.view.set_target_image_path(file_path)
        self.view.clear_result_image()
        self.view.set_status("已选择目标图片")

    def capture_mock_image(self) -> None:
        code = KeyboardWedgeScannerInput.normalize(self.view.code_text())
        captured = self.camera_adapter.capture(preferred_code=code or None)
        if not captured:
            self.view.show_error("未找到可用的 mock 相机样本图片")
            return

        self.view.set_target_image_path(captured)
        self.view.clear_result_image()
        self.view.set_status(f"已从 {self.camera_adapter.name} 获取样本图")

    def refresh_template_preview(self) -> None:
        template = self.view.selected_template()
        if template is None:
            self.view.clear_template_image()
            return

        preview_path = self._resolve_template_preview_path(template)
        if preview_path is None:
            self.view.clear_template_image()
            return

        self.view.set_template_image_path(preview_path)

    def run_detection(self) -> None:
        if self._thread is not None:
            self.view.set_status("检测正在运行，请等待")
            return

        template = self.view.selected_template()
        if template is None:
            self.view.show_error("请先查询并选择一个模板")
            return

        target_image_path = self.view.target_image_path()
        if not target_image_path:
            self.view.show_error("请先选择目标图片或使用 Mock 相机取图")
            return

        request = DetectionJobRequest(
            template=template,
            target_image_path=Path(target_image_path),
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
        self.view.set_status("准备启动检测...")
        thread.start()

    def _handle_detection_finished(self, result: DetectionJobResult) -> None:
        self.view.set_busy(False)
        self.view.set_status("检测完成")
        self.view.set_verdict(result.verdict or "-")
        self.view.set_output_dir(str(result.output_dir))
        self.view.set_summary_text(result.summary_text)
        self.view.set_result_image_path(result.visualization_path)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.view.append_history(f"{timestamp} | {result.verdict or '-'} | {result.output_dir}")

    def _handle_detection_failed(self, message: str) -> None:
        self.view.set_busy(False)
        self.view.set_status("检测失败")
        self.view.clear_result_image()
        self.view.show_error(message)

    def _clear_active_worker(self) -> None:
        self._thread = None
        self._worker = None

    def _resolve_template_preview_path(self, template) -> Path | None:
        source_path = template.source_path.resolve()
        cached = self._template_preview_cache.get(source_path)
        if cached is not None and cached.exists():
            return cached

        if template.preview_path and template.preview_path.exists():
            resolved = template.preview_path.resolve()
            self._template_preview_cache[source_path] = resolved
            return resolved

        cache_dir = (
            PROJECT_ROOT
            / "results"
            / "desktop_app"
            / "template_preview_cache"
            / source_path.stem
        )
        existing = sorted(cache_dir.glob("*.png"))
        if existing:
            resolved = existing[0].resolve()
            self._template_preview_cache[source_path] = resolved
            return resolved

        resolved_path, _source_type, err = resolve_template_input(
            str(source_path),
            output_dir=str(cache_dir),
            target_dpi=180,
        )
        if err or not resolved_path:
            self._template_preview_cache[source_path] = None
            return None

        resolved = Path(resolved_path).resolve()
        self._template_preview_cache[source_path] = resolved
        return resolved
