"""Main window for the PySide6 desktop app."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop_app.models import DetectionJobResult, HistoryRecord, TemplateRecord


class MainWindow(QMainWindow):
    """Desktop UI shell for code lookup and detection runs."""

    manual_query_requested = Signal()
    simulate_scan_requested = Signal()
    browse_target_requested = Signal()
    capture_mock_requested = Signal()
    run_detection_requested = Signal()
    template_selection_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("标签检测桌面端")
        self.resize(1440, 920)

        self.code_input = QLineEdit()
        self.query_button = QPushButton("查询模板")
        self.target_path_input = QLineEdit()
        self.target_path_input.setReadOnly(True)
        self.browse_button = QPushButton("选择图片")
        self.capture_button = QPushButton("Mock 相机取图")
        self.run_button = QPushButton("开始检测")
        self.open_template_button = QPushButton("打开模板")
        self.copy_template_button = QPushButton("复制路径")
        self.open_output_dir_button = QPushButton("打开目录")
        self.copy_output_dir_button = QPushButton("复制路径")
        self.open_history_dir_button = QPushButton("打开目录")
        self.copy_history_path_button = QPushButton("复制路径")

        self.template_list = QTreeWidget()
        self.current_template_value = QLineEdit()
        self.current_template_value.setReadOnly(True)
        self.status_value = QLabel("就绪")
        self.result_state_value = QLabel("待检测")
        self.verdict_value = QLabel("待检测")
        self.text_stats_value = QLabel("-")
        self.graphic_stats_value = QLabel("-")
        self.summary_value = QLabel("请先选择模板和目标图片")
        self.summary_value.setWordWrap(True)
        self.output_dir_value = QLineEdit()
        self.output_dir_value.setReadOnly(True)
        self.history_output_dir_value = QLineEdit()
        self.history_output_dir_value.setReadOnly(True)
        self.history_list = QTreeWidget()
        self.template_preview_label = self._build_image_label("选择模板后显示模板图")
        self.preview_label = self._build_image_label("请选择或采集目标图片")
        self.result_preview_label = self._build_image_label("检测完成后显示画框结果图")
        self.result_card_frame = QFrame()
        self.result_header_frame = QFrame()

        self._template_preview_path: Path | None = None
        self._target_preview_path: Path | None = None
        self._result_preview_path: Path | None = None
        self._did_auto_focus_code_input = False
        self._detection_enabled = False
        self._busy = False

        self._build_layout()
        self._apply_styles()
        self._connect_signals()
        self._sync_current_template_label()
        self._sync_history_buttons()
        self._sync_template_buttons()
        self._sync_output_buttons()
        self._sync_run_button()

    def _build_layout(self) -> None:
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 14, 16, 14)
        toolbar_layout.setSpacing(10)
        toolbar_layout.addWidget(self._build_section_title("目标图"))
        self.target_path_input.setPlaceholderText("尚未选择目标图片")
        toolbar_layout.addWidget(self.target_path_input, 1)
        toolbar_layout.addWidget(self.browse_button)
        toolbar_layout.addWidget(self.capture_button)
        toolbar_layout.addWidget(self.run_button)

        code_group = QGroupBox("Step 1  编码输入")
        code_layout = QFormLayout()
        code_layout.setContentsMargins(12, 12, 12, 12)
        code_layout.setVerticalSpacing(10)
        code_row = QHBoxLayout()
        code_row.addWidget(self.code_input)
        code_row.addWidget(self.query_button)
        code_layout.addRow("编码", code_row)
        code_group.setLayout(code_layout)

        template_group = QGroupBox("Step 2  模板确认")
        template_layout = QVBoxLayout()
        template_layout.setContentsMargins(12, 12, 12, 12)
        template_layout.setSpacing(10)
        template_layout.addWidget(QLabel("当前模板路径"))
        template_layout.addWidget(self.current_template_value)
        template_button_row = QHBoxLayout()
        template_button_row.addWidget(self.open_template_button)
        template_button_row.addWidget(self.copy_template_button)
        template_layout.addLayout(template_button_row)
        template_group.setLayout(template_layout)

        template_preview_group = QGroupBox("模板图预览")
        template_preview_layout = QVBoxLayout()
        template_preview_layout.setContentsMargins(12, 12, 12, 12)
        template_preview_layout.addWidget(self.template_preview_label)
        template_preview_group.setLayout(template_preview_layout)

        template_candidates_group = QGroupBox("模板候选")
        template_candidates_layout = QVBoxLayout()
        template_candidates_layout.setContentsMargins(12, 12, 12, 12)
        self.template_list.setRootIsDecorated(False)
        self.template_list.setItemsExpandable(False)
        self.template_list.setAlternatingRowColors(True)
        self.template_list.setColumnCount(3)
        self.template_list.setHeaderLabels(["模板", "来源", "版本"])
        header = self.template_list.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        template_candidates_layout.addWidget(self.template_list)
        template_candidates_group.setLayout(template_candidates_layout)

        preview_group = QGroupBox("目标图预览")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)

        result_preview_group = QGroupBox("画框结果图")
        result_preview_layout = QVBoxLayout()
        result_preview_layout.setContentsMargins(12, 12, 12, 12)
        result_preview_layout.addWidget(self.result_preview_label)
        result_preview_group.setLayout(result_preview_layout)

        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(12, 12, 12, 12)
        result_layout.setSpacing(12)
        self.result_header_frame.setObjectName("resultHeader")
        header_layout = QHBoxLayout(self.result_header_frame)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.addWidget(self.verdict_value)
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("状态"))
        header_layout.addWidget(self.status_value)

        self.result_card_frame.setObjectName("resultCard")
        card_layout = QVBoxLayout(self.result_card_frame)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)
        metrics_layout = QFormLayout()
        metrics_layout.setVerticalSpacing(8)
        metrics_layout.addRow("结果阶段", self.result_state_value)
        metrics_layout.addRow("文字匹配", self.text_stats_value)
        metrics_layout.addRow("图形比对", self.graphic_stats_value)
        metrics_layout.addRow("摘要说明", self.summary_value)
        metrics_layout.addRow("输出目录", self.output_dir_value)
        output_button_row = QHBoxLayout()
        output_button_row.addWidget(self.open_output_dir_button)
        output_button_row.addWidget(self.copy_output_dir_button)
        card_layout.addLayout(metrics_layout)
        card_layout.addLayout(output_button_row)

        result_layout.addWidget(self.result_header_frame)
        result_layout.addWidget(self.result_card_frame)
        result_group.setLayout(result_layout)

        history_group = QGroupBox("历史记录")
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(12, 12, 12, 12)
        history_layout.setSpacing(10)
        self.history_list.setRootIsDecorated(False)
        self.history_list.setItemsExpandable(False)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setColumnCount(4)
        self.history_list.setHeaderLabels(["时间", "编码", "模板", "结论"])
        history_header = self.history_list.header()
        history_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        history_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        history_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        history_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        history_layout.addWidget(self.history_list)
        history_layout.addWidget(QLabel("选中记录输出目录"))
        history_layout.addWidget(self.history_output_dir_value)
        history_button_row = QHBoxLayout()
        history_button_row.addWidget(self.open_history_dir_button)
        history_button_row.addWidget(self.copy_history_path_button)
        history_layout.addLayout(history_button_row)
        history_group.setLayout(history_layout)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(code_group)
        left_layout.addWidget(template_group)
        left_layout.addWidget(template_preview_group, 3)
        left_layout.addWidget(template_candidates_group, 2)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_splitter.addWidget(preview_group)
        preview_splitter.addWidget(result_preview_group)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(preview_splitter)
        right_splitter.addWidget(result_group)
        right_splitter.addWidget(history_group)
        right_splitter.setStretchFactor(0, 8)
        right_splitter.setStretchFactor(1, 4)
        right_splitter.setStretchFactor(2, 4)
        left_panel.setMaximumWidth(420)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_splitter)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        root_layout.addWidget(toolbar, 0)
        root_layout.addWidget(splitter, 1)
        root_layout.setStretch(0, 0)
        root_layout.setStretch(1, 1)
        self.setCentralWidget(central)

        preview_splitter.setSizes([500, 500])
        right_splitter.setSizes([420, 230, 230])
        splitter.setSizes([360, 1080])

    def _apply_styles(self) -> None:
        self.run_button.setObjectName("primaryAction")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f6f8;
                color: #1f2933;
                font-size: 13px;
            }
            QFrame {
                border-radius: 10px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7dee7;
                border-radius: 12px;
                margin-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLineEdit, QTreeWidget {
                background: #ffffff;
                border: 1px solid #ccd6e0;
                border-radius: 8px;
                padding: 6px 8px;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cdd6df;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #f7fafc;
            }
            QPushButton:disabled {
                color: #92a1b1;
                background: #eef2f6;
            }
            QPushButton#primaryAction {
                background: #f58220;
                color: white;
                border: none;
                font-weight: 700;
                padding: 10px 18px;
            }
            QPushButton#primaryAction:hover {
                background: #e87412;
            }
            QTreeWidget::item {
                height: 28px;
            }
            #resultHeader {
                background: #e7f4ee;
                border: 1px solid #b8dfca;
            }
            #resultCard {
                background: #ffffff;
                border: 1px solid #d7dee7;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.query_button.clicked.connect(self.manual_query_requested)
        self.code_input.returnPressed.connect(self.simulate_scan_requested)
        self.browse_button.clicked.connect(self.browse_target_requested)
        self.capture_button.clicked.connect(self.capture_mock_requested)
        self.run_button.clicked.connect(self.run_detection_requested)
        self.template_list.currentItemChanged.connect(
            self._handle_template_selection_changed
        )
        self.history_list.currentItemChanged.connect(self._handle_history_selection_changed)
        self.open_template_button.clicked.connect(self._open_current_template)
        self.copy_template_button.clicked.connect(self._copy_current_template_path)
        self.open_output_dir_button.clicked.connect(self._open_output_dir)
        self.copy_output_dir_button.clicked.connect(self._copy_output_dir)
        self.open_history_dir_button.clicked.connect(self._open_history_output_dir)
        self.copy_history_path_button.clicked.connect(self._copy_history_output_dir)

    def code_text(self) -> str:
        return self.code_input.text().strip()

    def set_code_text(self, value: str) -> None:
        self.code_input.setText(value)

    def focus_code_input(self, select_all: bool = False) -> None:
        if not self.code_input.isEnabled():
            return

        self.code_input.setFocus(Qt.FocusReason.OtherFocusReason)
        if select_all:
            self.code_input.selectAll()

    def set_code_input_enabled(self, enabled: bool) -> None:
        self.code_input.setEnabled(enabled)

    def selected_template(self) -> TemplateRecord | None:
        item = self.template_list.currentItem()
        if not item:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def set_templates(self, templates: list[TemplateRecord]) -> None:
        self.template_list.clear()
        for template in templates:
            variant = template.variant or "default"
            item = QTreeWidgetItem(
                [
                    template.display_name,
                    template.source_type,
                    variant,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, template)
            self.template_list.addTopLevelItem(item)

        if templates:
            self.template_list.setCurrentItem(self.template_list.topLevelItem(0))
        else:
            self.current_template_value.setText("-")
        self._sync_template_buttons()
        self._sync_run_button()

    def set_target_image_path(self, path: str | Path) -> None:
        text = str(path)
        self.target_path_input.setText(text)
        self._target_preview_path = Path(text)
        self._render_image(self.preview_label, self._target_preview_path, "请选择或采集目标图片")
        self._sync_run_button()

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
        if "失败" in value:
            self.status_value.setStyleSheet("color: #c0392b; font-weight: 700;")
        elif "检测中" in value:
            self.status_value.setStyleSheet("color: #d97706; font-weight: 700;")
        elif "完成" in value or "就绪" in value:
            self.status_value.setStyleSheet("color: #1f7a4f; font-weight: 700;")
        else:
            self.status_value.setStyleSheet("color: #334e68; font-weight: 700;")

    def set_verdict(self, value: str) -> None:
        verdict = value or "待检测"
        self.verdict_value.setText(verdict)
        if "失败" in verdict or "❌" in verdict:
            self.result_header_frame.setStyleSheet("background: #fdecec; border: 1px solid #f3b4b4;")
        elif "检测中" in verdict:
            self.result_header_frame.setStyleSheet("background: #fff4df; border: 1px solid #f5d28a;")
        elif "复核" in verdict or "⚠" in verdict:
            self.result_header_frame.setStyleSheet("background: #e7f4ee; border: 1px solid #b8dfca;")
        else:
            self.result_header_frame.setStyleSheet("background: #e7f4ee; border: 1px solid #b8dfca;")

    def set_output_dir(self, value: str) -> None:
        self.output_dir_value.setText(value or "-")
        self._sync_output_buttons()

    def set_summary_text(self, value: str) -> None:
        self.summary_value.setText(value or "暂无摘要")

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
        record = HistoryRecord(
            created_at="",
            code="",
            template_name=value,
            verdict="",
            output_dir=Path("."),
            target_image_path=Path("."),
            summary_text="",
        )
        self.prepend_history_record(record)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.set_code_input_enabled(not busy)
        self.query_button.setDisabled(busy)
        self.browse_button.setDisabled(busy)
        self.capture_button.setDisabled(busy)
        self.template_list.setDisabled(busy)
        self._sync_run_button()

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

    def set_detection_enabled(self, enabled: bool) -> None:
        self._detection_enabled = enabled
        self._sync_run_button()

    def show_pending_result(self) -> None:
        self.result_state_value.setText("待检测")
        self.set_verdict("待检测")
        self.set_summary_text("请先确认模板与目标图片，再开始检测。")
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()

    def show_running_result(self) -> None:
        self.result_state_value.setText("检测中")
        self.set_verdict("检测中")
        self.set_summary_text("检测任务正在后台执行，请等待结果返回。")
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()

    def show_failure_result(self, message: str) -> None:
        self.result_state_value.setText("检测失败")
        self.set_verdict("失败")
        self.set_summary_text(message)
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()

    def show_detection_result(self, result: DetectionJobResult) -> None:
        self.result_state_value.setText("检测完成")
        self.set_verdict(result.verdict or "-")
        self.set_summary_text(result.summary_text)
        self.text_stats_value.setText(
            f"{result.text_match_count}/{result.text_total_count}"
            if result.text_total_count
            else "-"
        )
        self.graphic_stats_value.setText(
            f"匹配 {result.graphic_match_count}  |  不一致 {result.graphic_mismatch_count}  |  待复核 {result.graphic_review_count}"
        )
        self.set_output_dir(str(result.output_dir))
        self.set_result_image_path(result.visualization_path)

    def set_history_records(self, records: list[HistoryRecord]) -> None:
        self.history_list.clear()
        for record in records:
            self._add_history_item(record, prepend=False)
        if self.history_list.topLevelItemCount():
            self.history_list.setCurrentItem(self.history_list.topLevelItem(0))
        else:
            self.history_output_dir_value.setText("")
        self._sync_history_buttons()

    def prepend_history_record(self, record: HistoryRecord) -> None:
        self._add_history_item(record, prepend=True)
        self.history_list.setCurrentItem(self.history_list.topLevelItem(0))
        self._sync_history_buttons()

    def selected_history_record(self) -> HistoryRecord | None:
        item = self.history_list.currentItem()
        if item is None:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _handle_template_selection_changed(self, *_args) -> None:
        self._sync_current_template_label()
        self._sync_template_buttons()
        self._sync_run_button()
        self.template_selection_changed.emit()

    def _sync_current_template_label(self, *_args) -> None:
        template = self.selected_template()
        if not template:
            self.current_template_value.setText("")
            return
        self.current_template_value.setText(str(template.source_path))

    def _handle_history_selection_changed(self, *_args) -> None:
        record = self.selected_history_record()
        self.history_output_dir_value.setText(str(record.output_dir) if record else "")
        self._sync_history_buttons()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_image(
            self.template_preview_label,
            self._template_preview_path,
            "选择模板后显示模板图",
        )
        self._render_image(
            self.preview_label,
            self._target_preview_path,
            "请选择或采集目标图片",
        )
        self._render_image(
            self.result_preview_label,
            self._result_preview_path,
            "检测完成后显示画框结果图",
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._did_auto_focus_code_input:
            return

        self._did_auto_focus_code_input = True
        self.focus_code_input(select_all=True)

    @staticmethod
    def _build_image_label(placeholder: str) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(300)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        label.setWordWrap(True)
        label.setStyleSheet(
            "border: 1px solid #d7dee7; background: #f8fafc; border-radius: 10px;"
        )
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

    @staticmethod
    def _build_section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700; color: #334e68;")
        return label

    def _sync_run_button(self) -> None:
        self.run_button.setDisabled(self._busy or not self._detection_enabled)

    def _sync_template_buttons(self) -> None:
        has_template = self.selected_template() is not None
        self.open_template_button.setDisabled(not has_template)
        self.copy_template_button.setDisabled(not has_template)

    def _sync_output_buttons(self) -> None:
        has_output_dir = bool(self.output_dir_value.text().strip()) and self.output_dir_value.text().strip() != "-"
        self.open_output_dir_button.setDisabled(not has_output_dir)
        self.copy_output_dir_button.setDisabled(not has_output_dir)

    def _sync_history_buttons(self) -> None:
        has_history = self.selected_history_record() is not None
        self.open_history_dir_button.setDisabled(not has_history)
        self.copy_history_path_button.setDisabled(not has_history)

    def _open_current_template(self) -> None:
        template = self.selected_template()
        if template is None:
            return
        self._open_path(template.source_path)

    def _copy_current_template_path(self) -> None:
        template = self.selected_template()
        if template is None:
            return
        self._copy_text(str(template.source_path))

    def _open_output_dir(self) -> None:
        output_dir = self.output_dir_value.text().strip()
        if output_dir and output_dir != "-":
            self._open_path(Path(output_dir))

    def _copy_output_dir(self) -> None:
        output_dir = self.output_dir_value.text().strip()
        if output_dir and output_dir != "-":
            self._copy_text(output_dir)

    def _open_history_output_dir(self) -> None:
        record = self.selected_history_record()
        if record is None:
            return
        self._open_path(record.output_dir)

    def _copy_history_output_dir(self) -> None:
        record = self.selected_history_record()
        if record is None:
            return
        self._copy_text(str(record.output_dir))

    def _copy_text(self, value: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(value)

    def _open_path(self, path: Path) -> None:
        resolved = Path(path)
        if not resolved.exists():
            self.show_error(f"路径不存在: {resolved}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved.resolve())))

    def _add_history_item(self, record: HistoryRecord, *, prepend: bool) -> None:
        timestamp = record.created_at.replace("T", " ")
        item = QTreeWidgetItem(
            [
                timestamp,
                record.code,
                record.template_name,
                record.verdict,
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, record)
        if prepend:
            self.history_list.insertTopLevelItem(0, item)
        else:
            self.history_list.addTopLevelItem(item)
