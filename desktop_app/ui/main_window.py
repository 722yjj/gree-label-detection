"""Main window for the PySide6 desktop app."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop_app.models import TemplateRecord


class MainWindow(QMainWindow):
    """Desktop UI shell for code lookup and detection runs."""

    query_code_requested = Signal()
    browse_target_requested = Signal()
    capture_mock_requested = Signal()
    run_detection_requested = Signal()
    template_selection_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("格力标签检测桌面端")
        self.resize(1280, 820)

        self.code_input = QLineEdit()
        self.query_button = QPushButton("查询模板")
        self.mock_scan_button = QPushButton("模拟扫码")
        self.target_path_input = QLineEdit()
        self.target_path_input.setReadOnly(True)
        self.browse_button = QPushButton("选择图片")
        self.capture_button = QPushButton("Mock 相机取图")
        self.run_button = QPushButton("开始检测")

        self.template_list = QListWidget()
        self.current_template_value = QLineEdit()
        self.current_template_value.setReadOnly(True)
        self.status_value = QLabel("就绪")
        self.verdict_value = QLabel("-")
        self.output_dir_value = QLineEdit()
        self.output_dir_value.setReadOnly(True)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.history_list = QListWidget()
        self.template_preview_label = self._build_image_label("选择模板后显示模板图")
        self.preview_label = self._build_image_label("暂无目标图片")
        self.result_preview_label = self._build_image_label("检测完成后显示画框结果图")

        self._template_preview_path: Path | None = None
        self._target_preview_path: Path | None = None
        self._result_preview_path: Path | None = None

        self._build_layout()
        self._connect_signals()
        self._sync_current_template_label()

    def _build_layout(self) -> None:
        top_group = QGroupBox("编码与图像输入")
        top_layout = QFormLayout()
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setVerticalSpacing(8)
        code_row = QHBoxLayout()
        code_row.addWidget(self.code_input)
        code_row.addWidget(self.query_button)
        code_row.addWidget(self.mock_scan_button)
        top_layout.addRow("编码", code_row)

        target_row = QHBoxLayout()
        target_row.addWidget(self.target_path_input)
        target_row.addWidget(self.browse_button)
        target_row.addWidget(self.capture_button)
        top_layout.addRow("目标图片", target_row)
        top_group.setLayout(top_layout)
        top_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        top_group.setMaximumHeight(top_group.sizeHint().height() + 8)

        template_group = QGroupBox("模板信息")
        template_layout = QVBoxLayout()
        template_layout.addWidget(self.template_list)
        template_layout.addWidget(QLabel("当前模板"))
        template_layout.addWidget(self.current_template_value)
        template_group.setLayout(template_layout)

        template_preview_group = QGroupBox("模板图预览")
        template_preview_layout = QVBoxLayout()
        template_preview_layout.addWidget(self.template_preview_label)
        template_preview_group.setLayout(template_preview_layout)

        preview_group = QGroupBox("目标图预览")
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)

        result_preview_group = QGroupBox("画框结果图")
        result_preview_layout = QVBoxLayout()
        result_preview_layout.addWidget(self.result_preview_label)
        result_preview_group.setLayout(result_preview_layout)

        result_group = QGroupBox("检测结果")
        result_layout = QFormLayout()
        result_layout.addRow("状态", self.status_value)
        result_layout.addRow("综合判定", self.verdict_value)
        result_layout.addRow("输出目录", self.output_dir_value)
        result_layout.addRow("摘要", self.summary_text)
        result_group.setLayout(result_layout)
        result_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        result_group.setMaximumHeight(220)

        history_group = QGroupBox("历史记录")
        history_layout = QVBoxLayout()
        history_layout.addWidget(self.history_list)
        history_group.setLayout(history_layout)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(template_group)
        left_splitter.addWidget(history_group)
        left_splitter.setStretchFactor(0, 4)
        left_splitter.setStretchFactor(1, 2)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_splitter.addWidget(template_preview_group)
        preview_splitter.addWidget(preview_group)
        preview_splitter.addWidget(result_preview_group)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setStretchFactor(2, 1)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(preview_splitter)
        right_splitter.addWidget(result_group)
        right_splitter.setStretchFactor(0, 8)
        right_splitter.setStretchFactor(1, 2)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(left_splitter)
        left_panel.setMaximumWidth(420)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(right_splitter)
        right_layout.addWidget(self.run_button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.addWidget(top_group, 0)
        root_layout.addWidget(splitter, 1)
        root_layout.setStretch(0, 0)
        root_layout.setStretch(1, 1)
        self.setCentralWidget(central)

        left_splitter.setSizes([460, 180])
        preview_splitter.setSizes([360, 360, 360])
        right_splitter.setSizes([620, 180])
        splitter.setSizes([320, 960])

    def _connect_signals(self) -> None:
        self.query_button.clicked.connect(self.query_code_requested)
        self.mock_scan_button.clicked.connect(self.query_code_requested)
        self.code_input.returnPressed.connect(self.query_code_requested)
        self.browse_button.clicked.connect(self.browse_target_requested)
        self.capture_button.clicked.connect(self.capture_mock_requested)
        self.run_button.clicked.connect(self.run_detection_requested)
        self.template_list.currentItemChanged.connect(self._handle_template_selection_changed)

    def code_text(self) -> str:
        return self.code_input.text().strip()

    def set_code_text(self, value: str) -> None:
        self.code_input.setText(value)

    def selected_template(self) -> TemplateRecord | None:
        item = self.template_list.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def set_templates(self, templates: list[TemplateRecord]) -> None:
        self.template_list.clear()
        for template in templates:
            variant = template.variant or "default"
            label = f"{template.display_name} [{template.source_type}] ({variant})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, template)
            self.template_list.addItem(item)

        if templates:
            self.template_list.setCurrentRow(0)
        else:
            self.current_template_value.setText("-")

    def set_target_image_path(self, path: str | Path) -> None:
        text = str(path)
        self.target_path_input.setText(text)
        self._target_preview_path = Path(text)
        self._render_image(self.preview_label, self._target_preview_path, "暂无目标图片")

    def set_template_image_path(self, path: str | Path | None) -> None:
        self._template_preview_path = Path(path) if path else None
        self._render_image(
            self.template_preview_label,
            self._template_preview_path,
            "选择模板后显示模板图",
        )

    def clear_template_image(self) -> None:
        self._template_preview_path = None
        self._render_image(
            self.template_preview_label,
            None,
            "选择模板后显示模板图",
        )

    def target_image_path(self) -> str:
        return self.target_path_input.text().strip()

    def set_status(self, value: str) -> None:
        self.status_value.setText(value)

    def set_verdict(self, value: str) -> None:
        self.verdict_value.setText(value or "-")

    def set_output_dir(self, value: str) -> None:
        self.output_dir_value.setText(value or "-")

    def set_summary_text(self, value: str) -> None:
        self.summary_text.setPlainText(value or "")

    def set_result_image_path(self, path: str | Path | None) -> None:
        self._result_preview_path = Path(path) if path else None
        self._render_image(
            self.result_preview_label,
            self._result_preview_path,
            "检测完成后显示画框结果图",
        )

    def clear_result_image(self) -> None:
        self._result_preview_path = None
        self._render_image(
            self.result_preview_label,
            None,
            "检测完成后显示画框结果图",
        )

    def append_history(self, value: str) -> None:
        self.history_list.insertItem(0, value)

    def set_busy(self, busy: bool) -> None:
        self.query_button.setDisabled(busy)
        self.mock_scan_button.setDisabled(busy)
        self.browse_button.setDisabled(busy)
        self.capture_button.setDisabled(busy)
        self.run_button.setDisabled(busy)

    def choose_image_file(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择目标图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        return file_path

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "错误", message)

    def _handle_template_selection_changed(self, *_args) -> None:
        self._sync_current_template_label()
        self.template_selection_changed.emit()

    def _sync_current_template_label(self, *_args) -> None:
        template = self.selected_template()
        if not template:
            self.current_template_value.setText("")
            return
        self.current_template_value.setText(str(template.source_path))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_image(
            self.template_preview_label,
            self._template_preview_path,
            "选择模板后显示模板图",
        )
        self._render_image(self.preview_label, self._target_preview_path, "暂无目标图片")
        self._render_image(
            self.result_preview_label,
            self._result_preview_path,
            "检测完成后显示画框结果图",
        )

    @staticmethod
    def _build_image_label(placeholder: str) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(300)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        label.setStyleSheet("border: 1px solid #bbb; background: #fafafa;")
        return label

    @staticmethod
    def _clear_image(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setPixmap(QPixmap())

    def _render_image(self, label: QLabel, path: Path | None, empty_text: str) -> None:
        if path is None:
            self._clear_image(label, empty_text)
            return

        pixmap = QPixmap(str(path))
        if not path.exists():
            self._clear_image(label, "图片不存在")
            return
        if pixmap.isNull():
            self._clear_image(label, "无法加载图片预览")
            return

        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)
