"""Main window for the PySide6 desktop app."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop_app.models import DetectionJobResult, HistoryRecord, TemplateRecord
from label_detection.license import LicenseStatus


class ScaledImageLabel(QLabel):
    """QLabel that keeps the source pixmap and rescales on its own resize."""

    _preferred_width = 420
    _preferred_height = 260
    _minimum_width = 120
    _minimum_height = 220

    def __init__(
        self,
        placeholder: str,
        *,
        preferred_width: int = 420,
        preferred_height: int = 260,
        minimum_width: int = 120,
        minimum_height: int = 220,
    ) -> None:
        super().__init__(placeholder)
        self._preferred_width = preferred_width
        self._preferred_height = preferred_height
        self._minimum_width = minimum_width
        self._minimum_height = minimum_height
        self._placeholder_text = placeholder
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(self._minimum_height)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWordWrap(True)
        self.setStyleSheet(
            "border: 1px solid #d7dee7; background: #f8fafc; border-radius: 10px;"
        )

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint().expandedTo(
            super().sizeHint()
        ).expandedTo(
            QSize(self._preferred_width, self._preferred_height)
        )

    def minimumSizeHint(self) -> QSize:
        return QSize(self._minimum_width, self._minimum_height)

    def clear_preview(self, text: str | None = None) -> None:
        if text is not None:
            self._placeholder_text = text
        self._source_pixmap = None
        self.setText(self._placeholder_text)
        super().setPixmap(QPixmap())

    def set_preview_pixmap(self, pixmap: QPixmap, text: str | None = None) -> None:
        if text is not None:
            self._placeholder_text = text
        self._source_pixmap = pixmap
        self._sync_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._source_pixmap is not None:
            self._sync_pixmap()

    def _sync_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self.clear_preview()
            return

        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        super().setPixmap(scaled)


class AnnotationImageLabel(ScaledImageLabel):
    """Interactive image label for reviewing predicted boxes and drawing misses."""

    box_selected = Signal(str, str)
    manual_box_created = Signal(list)

    def __init__(self, placeholder: str) -> None:
        super().__init__(
            placeholder,
            preferred_width=520,
            preferred_height=420,
            minimum_width=240,
            minimum_height=260,
        )
        self._predicted_boxes: list[dict] = []
        self._manual_boxes: list[dict] = []
        self._selected_kind: str | None = None
        self._selected_id: str | None = None
        self._draw_start_image_point: QPoint | None = None
        self._draw_current_image_point: QPoint | None = None
        self._dragging_manual_box = False
        self.setMouseTracking(True)

    def set_annotation_boxes(
        self,
        predicted_boxes: list[dict],
        manual_boxes: list[dict],
        *,
        selected_kind: str | None = None,
        selected_id: str | None = None,
    ) -> None:
        self._predicted_boxes = [dict(item) for item in predicted_boxes]
        self._manual_boxes = [dict(item) for item in manual_boxes]
        self._selected_kind = selected_kind
        self._selected_id = selected_id
        self._sync_pixmap()

    def clear_preview(self, text: str | None = None) -> None:
        self._predicted_boxes = []
        self._manual_boxes = []
        self._selected_kind = None
        self._selected_id = None
        self._draw_start_image_point = None
        self._draw_current_image_point = None
        self._dragging_manual_box = False
        super().clear_preview(text)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._source_pixmap is None:
            super().mousePressEvent(event)
            return
        image_point = self._widget_to_image_point(event.position().toPoint())
        if image_point is None:
            super().mousePressEvent(event)
            return
        hit = self._hit_test(image_point)
        if hit is not None:
            self._selected_kind, self._selected_id = hit
            self.box_selected.emit(self._selected_kind, self._selected_id)
            self._sync_pixmap()
            return
        self._dragging_manual_box = True
        self._draw_start_image_point = image_point
        self._draw_current_image_point = image_point
        self._sync_pixmap()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging_manual_box:
            super().mouseMoveEvent(event)
            return
        image_point = self._widget_to_image_point(event.position().toPoint())
        if image_point is None:
            return
        self._draw_current_image_point = image_point
        self._sync_pixmap()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging_manual_box:
            super().mouseReleaseEvent(event)
            return
        start = self._draw_start_image_point
        end = self._widget_to_image_point(event.position().toPoint())
        self._dragging_manual_box = False
        self._draw_start_image_point = None
        self._draw_current_image_point = None
        if start is None or end is None:
            self._sync_pixmap()
            return
        bbox = self._points_to_bbox(start, end)
        if bbox is not None:
            self.manual_box_created.emit(bbox)
        self._sync_pixmap()

    def _sync_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            super()._sync_pixmap()
            return

        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        composed = QPixmap(scaled)
        painter = QPainter(composed)
        scale_x = scaled.width() / self._source_pixmap.width()
        scale_y = scaled.height() / self._source_pixmap.height()
        for item in self._predicted_boxes:
            self._draw_box(painter, item, scale_x, scale_y, kind="predicted")
        for item in self._manual_boxes:
            self._draw_box(painter, item, scale_x, scale_y, kind="manual")
        if self._draw_start_image_point is not None and self._draw_current_image_point is not None:
            preview_box = self._points_to_bbox(
                self._draw_start_image_point,
                self._draw_current_image_point,
            )
            if preview_box is not None:
                self._draw_rect(
                    painter,
                    preview_box,
                    scale_x,
                    scale_y,
                    QColor("#2f80ed"),
                    selected=True,
                )
        painter.end()
        self.setText("")
        QLabel.setPixmap(self, composed)

    def _draw_box(
        self,
        painter: QPainter,
        item: dict,
        scale_x: float,
        scale_y: float,
        *,
        kind: str,
    ) -> None:
        bbox = self._normalized_bbox(item.get("bbox"))
        if bbox is None:
            return
        decision = str(item.get("decision") or "unreviewed")
        if kind == "manual":
            color = QColor("#2f80ed")
        elif decision == "true_positive":
            color = QColor("#1f9d55")
        elif decision == "false_positive":
            color = QColor("#d64545")
        else:
            color = QColor("#f59f00")
        selected = self._selected_kind == kind and self._selected_id == str(item.get("box_id"))
        self._draw_rect(painter, bbox, scale_x, scale_y, color, selected=selected)

    @staticmethod
    def _draw_rect(
        painter: QPainter,
        bbox: list[int],
        scale_x: float,
        scale_y: float,
        color: QColor,
        *,
        selected: bool,
    ) -> None:
        x1, y1, x2, y2 = bbox
        pen = QPen(color)
        pen.setWidth(2 if selected else 1)
        painter.setPen(pen)
        painter.drawRect(
            QRect(
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round((x2 - x1) * scale_x)),
                int(round((y2 - y1) * scale_y)),
            )
        )

    def _widget_to_image_point(self, point: QPoint) -> QPoint | None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        displayed = self.pixmap()
        if displayed is None or displayed.isNull():
            return None
        rect = self.contentsRect()
        offset_x = rect.x() + max(0, (rect.width() - displayed.width()) // 2)
        offset_y = rect.y() + max(0, (rect.height() - displayed.height()) // 2)
        local_x = point.x() - offset_x
        local_y = point.y() - offset_y
        if local_x < 0 or local_y < 0 or local_x > displayed.width() or local_y > displayed.height():
            return None
        image_x = int(round(local_x * self._source_pixmap.width() / displayed.width()))
        image_y = int(round(local_y * self._source_pixmap.height() / displayed.height()))
        image_x = max(0, min(self._source_pixmap.width(), image_x))
        image_y = max(0, min(self._source_pixmap.height(), image_y))
        return QPoint(image_x, image_y)

    def _hit_test(self, image_point: QPoint) -> tuple[str, str] | None:
        candidates: list[tuple[str, dict]] = [
            *[("manual", item) for item in self._manual_boxes],
            *[("predicted", item) for item in self._predicted_boxes],
        ]
        for kind, item in reversed(candidates):
            bbox = self._normalized_bbox(item.get("bbox"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x1 <= image_point.x() <= x2 and y1 <= image_point.y() <= y2:
                return kind, str(item.get("box_id"))
        return None

    @staticmethod
    def _points_to_bbox(start: QPoint, end: QPoint) -> list[int] | None:
        x1, x2 = sorted([start.x(), end.x()])
        y1, y2 = sorted([start.y(), end.y()])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _normalized_bbox(value: object) -> list[int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(item))) for item in value]
        except (TypeError, ValueError):
            return None
        return [x1, y1, x2, y2]

class MainWindow(QMainWindow):
    """Desktop UI shell for code lookup and detection runs."""

    manual_query_requested = Signal()
    simulate_scan_requested = Signal()
    browse_target_requested = Signal()
    capture_camera_requested = Signal()
    camera_rotation_requested = Signal()
    next_target_requested = Signal()
    run_detection_requested = Signal()
    save_annotation_requested = Signal()
    template_selection_changed = Signal()
    history_selection_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("标签检测桌面端")
        self._configure_window_chrome()
        self.setMinimumSize(1180, 720)
        self.resize(1440, 920)

        self.code_input = QLineEdit()
        self.query_button = QPushButton("查询模板")
        self.target_path_input = QLineEdit()
        self.target_path_input.setReadOnly(True)
        self.camera_backend_value = QLabel("相机：-")
        self.camera_backend_value.setObjectName("cameraBackend")
        self.camera_quality_value = QLabel("画面：-")
        self.camera_quality_value.setObjectName("cameraQuality")
        self.rotation_button = QPushButton("方向：0°")
        self.browse_button = QPushButton("选择图片")
        self.capture_button = QPushButton("拍照保存")
        self.next_button = QPushButton("下一张")
        self.run_button = QPushButton("开始检测")
        self.detailed_output_checkbox = QCheckBox("保存详细调试结果")
        self.open_template_button = QPushButton("打开模板")
        self.copy_template_button = QPushButton("复制路径")
        self.open_output_dir_button = QPushButton("打开目录")
        self.copy_output_dir_button = QPushButton("复制路径")
        self.open_history_dir_button = QPushButton("打开目录")
        self.copy_history_path_button = QPushButton("复制路径")
        self.annotation_box_list = QTreeWidget()
        self.annotation_selection_value = QLabel("未选中")
        self.annotation_mark_tp_button = QPushButton("正确")
        self.annotation_mark_fp_button = QPushButton("错误")
        self.annotation_delete_manual_box_button = QPushButton("删除漏检框")
        self.annotation_save_button = QPushButton("保存标注")
        self.annotation_path_value = QLineEdit()
        self.annotation_path_value.setReadOnly(True)
        self.annotation_notes_edit = QPlainTextEdit()
        self.annotation_notes_edit.setMaximumHeight(66)

        self.template_list = QTreeWidget()
        self.current_template_value = QLineEdit()
        self.current_template_value.setReadOnly(True)
        self.status_value = QLabel("就绪")
        self.result_state_value = QLabel("待检测")
        self.verdict_value = QLabel("待检测")
        self.detection_duration_value = QLabel("-")
        self.text_stats_value = QLabel("-")
        self.graphic_stats_value = QLabel("-")
        self.summary_value = QLabel("请先选择模板和目标图片")
        self.summary_value.setWordWrap(True)
        self.license_machine_value = QLineEdit()
        self.license_machine_value.setReadOnly(True)
        self.license_status_value = QLabel("-")
        self.license_expiry_value = QLabel("-")
        self.license_path_value = QLineEdit()
        self.license_path_value.setReadOnly(True)
        self.license_reason_value = QLabel("-")
        self.license_reason_value.setWordWrap(True)
        self.output_dir_value = QLineEdit()
        self.output_dir_value.setReadOnly(True)
        self.history_output_dir_value = QLineEdit()
        self.history_output_dir_value.setReadOnly(True)
        self.history_list = QTreeWidget()
        self.template_preview_label = self._build_image_label(
            "选择模板后显示模板图",
            preferred_height=280,
        )
        self.preview_label = self._build_image_label(
            "请选择或采集目标图片",
            preferred_height=300,
        )
        self.result_preview_label = AnnotationImageLabel("检测完成后显示画框结果图")
        self.right_lower_scroll_area = QScrollArea()
        self.result_scroll_area = QScrollArea()
        self.history_scroll_area = QScrollArea()
        self.result_card_frame = QFrame()
        self.result_header_frame = QFrame()

        self._template_preview_path: Path | None = None
        self._target_preview_path: Path | None = None
        self._result_preview_path: Path | None = None
        self._annotation_payload: dict | None = None
        self._selected_annotation_kind: str | None = None
        self._selected_annotation_id: str | None = None
        self._did_auto_focus_code_input = False
        self._detection_enabled = False
        self._busy = False
        self._camera_busy = False
        self._capture_action_enabled = True
        self._next_enabled = False
        self._license_allows_detection = True

        self._build_layout()
        self._apply_styles()
        self._connect_signals()
        self._sync_current_template_label()
        self._sync_history_buttons()
        self._sync_annotation_controls()
        self._sync_template_buttons()
        self._sync_output_buttons()
        self._sync_capture_controls()
        self._sync_run_button()

    def _build_layout(self) -> None:
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 14, 16, 14)
        toolbar_layout.setSpacing(10)
        toolbar_layout.addWidget(self._build_section_title("目标图"))
        toolbar_layout.addWidget(self.camera_backend_value)
        toolbar_layout.addWidget(self.camera_quality_value)
        toolbar_layout.addWidget(self.rotation_button)
        self.target_path_input.setPlaceholderText("尚未选择目标图片")
        toolbar_layout.addWidget(self.target_path_input, 1)
        toolbar_layout.addWidget(self.browse_button)
        toolbar_layout.addWidget(self.capture_button)
        toolbar_layout.addWidget(self.next_button)
        toolbar_layout.addWidget(self.detailed_output_checkbox)
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
        result_outer_layout = QVBoxLayout()
        result_outer_layout.setContentsMargins(12, 12, 12, 12)
        result_outer_layout.setSpacing(0)
        result_content = QWidget()
        result_layout = QVBoxLayout(result_content)
        result_layout.setContentsMargins(0, 0, 0, 0)
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
        metrics_layout.addRow("检测耗时", self.detection_duration_value)
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
        self.result_scroll_area.setWidgetResizable(True)
        self.result_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.result_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.result_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.result_scroll_area.setWidget(result_content)
        result_outer_layout.addWidget(self.result_scroll_area)
        result_group.setLayout(result_outer_layout)
        result_group.setMaximumHeight(260)
        result_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        license_group = QGroupBox("授权状态")
        license_layout = QFormLayout()
        license_layout.setContentsMargins(12, 12, 12, 12)
        license_layout.setVerticalSpacing(8)
        license_layout.addRow("机器码", self.license_machine_value)
        license_layout.addRow("授权状态", self.license_status_value)
        license_layout.addRow("到期时间", self.license_expiry_value)
        license_layout.addRow("授权文件", self.license_path_value)
        license_layout.addRow("失败原因", self.license_reason_value)
        license_group.setLayout(license_layout)
        license_group.setMaximumHeight(190)
        license_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        history_group = QGroupBox("历史记录")
        history_outer_layout = QVBoxLayout()
        history_outer_layout.setContentsMargins(12, 12, 12, 12)
        history_outer_layout.setSpacing(0)
        history_content = QWidget()
        history_layout = QVBoxLayout(history_content)
        history_layout.setContentsMargins(0, 0, 0, 0)
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
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.history_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.history_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.history_scroll_area.setWidget(history_content)
        history_outer_layout.addWidget(self.history_scroll_area)
        history_group.setLayout(history_outer_layout)
        history_group.setMaximumHeight(190)
        history_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        annotation_group = QGroupBox("人工复核标注")
        annotation_layout = QVBoxLayout()
        annotation_layout.setContentsMargins(12, 12, 12, 12)
        annotation_layout.setSpacing(8)
        self.annotation_box_list.setRootIsDecorated(False)
        self.annotation_box_list.setItemsExpandable(False)
        self.annotation_box_list.setAlternatingRowColors(True)
        self.annotation_box_list.setColumnCount(3)
        self.annotation_box_list.setHeaderLabels(["框", "判定", "位置"])
        annotation_header = self.annotation_box_list.header()
        annotation_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        annotation_layout.addWidget(self.annotation_box_list)
        self.annotation_selection_value.setWordWrap(True)
        annotation_layout.addWidget(self.annotation_selection_value)
        annotation_button_row = QHBoxLayout()
        annotation_button_row.addWidget(self.annotation_mark_tp_button)
        annotation_button_row.addWidget(self.annotation_mark_fp_button)
        annotation_button_row.addWidget(self.annotation_delete_manual_box_button)
        annotation_button_row.addWidget(self.annotation_save_button)
        annotation_layout.addLayout(annotation_button_row)
        self.annotation_notes_edit.setPlaceholderText("人工备注")
        annotation_layout.addWidget(self.annotation_notes_edit)
        annotation_layout.addWidget(QLabel("标注文件"))
        annotation_layout.addWidget(self.annotation_path_value)
        annotation_group.setLayout(annotation_layout)
        annotation_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )

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
        preview_splitter.setStretchFactor(1, 2)

        right_lower_content = QWidget()
        right_lower_layout = QVBoxLayout(right_lower_content)
        right_lower_layout.setContentsMargins(0, 0, 0, 0)
        right_lower_layout.setSpacing(12)
        right_lower_layout.addWidget(result_group)
        right_lower_layout.addWidget(annotation_group)
        right_lower_layout.addWidget(license_group)
        right_lower_layout.addWidget(history_group)
        right_lower_layout.addStretch(1)

        self.right_lower_scroll_area.setWidgetResizable(True)
        self.right_lower_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.right_lower_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.right_lower_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.right_lower_scroll_area.setWidget(right_lower_content)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(preview_splitter)
        right_splitter.addWidget(self.right_lower_scroll_area)
        right_splitter.setStretchFactor(0, 14)
        right_splitter.setStretchFactor(1, 5)
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

        preview_splitter.setSizes([420, 660])
        right_splitter.setSizes([620, 260])
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
            QLabel#cameraBackend {
                background: #eef2f6;
                border: 1px solid #d7dee7;
                border-radius: 8px;
                color: #334e68;
                font-weight: 700;
                padding: 6px 10px;
            }
            QLabel#cameraQuality {
                background: #eef2f6;
                border: 1px solid #d7dee7;
                border-radius: 8px;
                color: #334e68;
                font-weight: 700;
                padding: 6px 10px;
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

    def _configure_window_chrome(self) -> None:
        flags = self.windowFlags()
        flags |= (
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint
        self.setWindowFlags(flags)

    def _connect_signals(self) -> None:
        self.query_button.clicked.connect(
            lambda _checked=False: self.manual_query_requested.emit()
        )
        self.code_input.returnPressed.connect(self.simulate_scan_requested)
        self.browse_button.clicked.connect(
            lambda _checked=False: self.browse_target_requested.emit()
        )
        self.capture_button.clicked.connect(
            lambda _checked=False: self.capture_camera_requested.emit()
        )
        self.rotation_button.clicked.connect(
            lambda _checked=False: self.camera_rotation_requested.emit()
        )
        self.next_button.clicked.connect(
            lambda _checked=False: self.next_target_requested.emit()
        )
        self.run_button.clicked.connect(
            lambda _checked=False: self.run_detection_requested.emit()
        )
        self.template_list.currentItemChanged.connect(
            self._handle_template_selection_changed
        )
        self.history_list.currentItemChanged.connect(self._handle_history_selection_changed)
        self.annotation_box_list.currentItemChanged.connect(
            self._handle_annotation_list_selection_changed
        )
        self.result_preview_label.box_selected.connect(self._select_annotation_box)
        self.result_preview_label.manual_box_created.connect(self._add_manual_gt_box)
        self.open_template_button.clicked.connect(self._open_current_template)
        self.copy_template_button.clicked.connect(self._copy_current_template_path)
        self.open_output_dir_button.clicked.connect(self._open_output_dir)
        self.copy_output_dir_button.clicked.connect(self._copy_output_dir)
        self.open_history_dir_button.clicked.connect(self._open_history_output_dir)
        self.copy_history_path_button.clicked.connect(self._copy_history_output_dir)
        self.annotation_mark_tp_button.clicked.connect(
            lambda _checked=False: self._mark_selected_annotation_box("true_positive")
        )
        self.annotation_mark_fp_button.clicked.connect(
            lambda _checked=False: self._mark_selected_annotation_box("false_positive")
        )
        self.annotation_delete_manual_box_button.clicked.connect(
            lambda _checked=False: self._delete_selected_manual_box()
        )
        self.annotation_save_button.clicked.connect(
            lambda _checked=False: self.save_annotation_requested.emit()
        )

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

    def clear_target_image(self, placeholder: str = "等待相机预览") -> None:
        self.target_path_input.setText("")
        self._target_preview_path = None
        self.preview_label.clear_preview(placeholder)
        self._sync_run_button()

    def set_target_preview_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.preview_label.set_preview_pixmap(pixmap, "相机实时预览")

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

    def detailed_output_enabled(self) -> bool:
        return self.detailed_output_checkbox.isChecked()

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

    def set_camera_backend_name(self, name: str) -> None:
        self.camera_backend_value.setText(f"相机：{name or '-'}")

    def set_camera_quality(self, value: str, severity: str = "neutral") -> None:
        self.camera_quality_value.setText(f"画面：{value or '-'}")
        if severity == "warning":
            self.camera_quality_value.setStyleSheet(
                "background: #fff4df; border: 1px solid #f5d28a; "
                "border-radius: 8px; color: #9a5b00; font-weight: 700; "
                "padding: 6px 10px;"
            )
        elif severity == "error":
            self.camera_quality_value.setStyleSheet(
                "background: #fdecec; border: 1px solid #f3b4b4; "
                "border-radius: 8px; color: #b42318; font-weight: 700; "
                "padding: 6px 10px;"
            )
        elif severity == "ok":
            self.camera_quality_value.setStyleSheet(
                "background: #e7f4ee; border: 1px solid #b8dfca; "
                "border-radius: 8px; color: #1f7a4f; font-weight: 700; "
                "padding: 6px 10px;"
            )
        else:
            self.camera_quality_value.setStyleSheet(
                "background: #eef2f6; border: 1px solid #d7dee7; "
                "border-radius: 8px; color: #334e68; font-weight: 700; "
                "padding: 6px 10px;"
            )

    def set_camera_rotation_degrees(self, rotation_degrees: int) -> None:
        self.rotation_button.setText(f"方向：{rotation_degrees}°")

    def set_capture_action(self, text: str, enabled: bool) -> None:
        self.capture_button.setText(text)
        self._capture_action_enabled = enabled
        self._sync_capture_controls()

    def set_next_enabled(self, enabled: bool) -> None:
        self._next_enabled = enabled
        self._sync_capture_controls()

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

    def set_detection_duration_seconds(self, seconds: float | None) -> None:
        self.detection_duration_value.setText(self._format_detection_duration(seconds))

    def set_result_image_path(self, path: str | Path | None) -> None:
        self._result_preview_path = Path(path) if path else None
        self._render_image(
            self.result_preview_label,
            self._result_preview_path,
            "检测完成后显示画框结果图",
        )
        self._sync_annotation_image()

    def set_license_status(self, status: LicenseStatus) -> None:
        self._license_allows_detection = status.ok
        self.license_machine_value.setText(status.machine_fingerprint)
        self.license_status_value.setText("有效" if status.ok else "无效")
        self.license_expiry_value.setText(status.expires_at or "-")
        self.license_path_value.setText(status.license_path)
        self.license_reason_value.setText(status.reason or "-")
        if status.ok:
            self.license_status_value.setStyleSheet("color: #1f7a4f; font-weight: 700;")
            self.set_status("授权有效")
        else:
            self.license_status_value.setStyleSheet("color: #c0392b; font-weight: 700;")
            self.set_status("授权失败")
            self.set_summary_text(status.reason)
            self.set_verdict("授权失败")
            self.set_busy(True)
            self.code_input.setReadOnly(True)
            self.query_button.setDisabled(True)
            self.template_list.setDisabled(True)
            self.detailed_output_checkbox.setDisabled(True)
        self._sync_run_button()

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
        self._sync_capture_controls()
        self.detailed_output_checkbox.setDisabled(busy)
        self.template_list.setDisabled(busy)
        self._sync_run_button()

    def set_camera_busy(self, busy: bool) -> None:
        self._camera_busy = busy
        self._sync_capture_controls()
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
        self.set_detection_duration_seconds(None)
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()
        self.clear_annotation()

    def show_running_result(self) -> None:
        self.result_state_value.setText("检测中")
        self.set_verdict("检测中")
        self.set_summary_text("检测任务正在后台执行，请等待结果返回。")
        self.detection_duration_value.setText("计时中")
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()
        self.clear_annotation()

    def show_failure_result(self, message: str) -> None:
        self.result_state_value.setText("检测失败")
        self.set_verdict("失败")
        self.set_summary_text(message)
        self.set_detection_duration_seconds(None)
        self.text_stats_value.setText("-")
        self.graphic_stats_value.setText("-")
        self.set_output_dir("")
        self.clear_result_image()
        self.clear_annotation()

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

    def clear_annotation(self) -> None:
        self._annotation_payload = None
        self._selected_annotation_kind = None
        self._selected_annotation_id = None
        self.annotation_box_list.clear()
        self.result_preview_label.set_annotation_boxes([], [])
        self.annotation_selection_value.setText("未选中")
        self.annotation_notes_edit.setPlainText("")
        self.annotation_path_value.setText("")
        self._sync_annotation_controls()

    def show_annotation(self, annotation: dict) -> None:
        self._annotation_payload = dict(annotation)
        self._selected_annotation_kind = None
        self._selected_annotation_id = None
        self.annotation_box_list.clear()
        self.annotation_notes_edit.setPlainText(str(annotation.get("notes") or ""))
        image_path = (annotation.get("image") or {}).get("path") if isinstance(annotation.get("image"), dict) else None
        if image_path:
            self.set_result_image_path(image_path)
        run_dir = annotation.get("run_dir")
        self.annotation_path_value.setText(
            str(Path(str(run_dir)) / "manual_annotation.json") if run_dir else ""
        )
        predicted_boxes = []
        for item in annotation.get("predicted_boxes") or []:
            if not isinstance(item, dict):
                continue
            predicted_boxes.append(dict(item))
            bbox = item.get("bbox") or []
            bbox_text = ",".join(str(value) for value in bbox)
            tree_item = QTreeWidgetItem(
                [
                    str(item.get("box_id") or ""),
                    self._annotation_decision_label(str(item.get("decision") or "unreviewed")),
                    bbox_text,
                ]
            )
            tree_payload = dict(item)
            tree_payload["annotation_kind"] = "predicted"
            tree_item.setData(0, Qt.ItemDataRole.UserRole, tree_payload)
            self.annotation_box_list.addTopLevelItem(tree_item)
        manual_boxes = self._manual_boxes_with_ids(annotation.get("manual_gt_boxes") or [])
        self._annotation_payload["manual_gt_boxes"] = manual_boxes
        for item in manual_boxes:
            bbox = item.get("bbox") or []
            bbox_text = ",".join(str(value) for value in bbox)
            tree_item = QTreeWidgetItem(
                [
                    str(item.get("box_id") or ""),
                    "漏检框",
                    bbox_text,
                ]
            )
            tree_payload = dict(item)
            tree_payload["annotation_kind"] = "manual"
            tree_item.setData(0, Qt.ItemDataRole.UserRole, tree_payload)
            self.annotation_box_list.addTopLevelItem(tree_item)
        self._sync_annotation_image()
        if self.annotation_box_list.topLevelItemCount():
            self.annotation_box_list.setCurrentItem(self.annotation_box_list.topLevelItem(0))
        else:
            self.annotation_selection_value.setText("未选中")
        self._sync_annotation_controls()

    def current_annotation_payload(self) -> dict | None:
        if self._annotation_payload is None:
            return None
        payload = dict(self._annotation_payload)
        payload["notes"] = self.annotation_notes_edit.toPlainText().strip()
        boxes = []
        manual_boxes = []
        for index in range(self.annotation_box_list.topLevelItemCount()):
            tree_item = self.annotation_box_list.topLevelItem(index)
            item = tree_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(item, dict):
                payload_item = dict(item)
                kind = payload_item.pop("annotation_kind", "predicted")
                if kind == "manual":
                    manual_boxes.append(payload_item)
                else:
                    boxes.append(payload_item)
        payload["predicted_boxes"] = boxes
        payload["manual_gt_boxes"] = manual_boxes
        return payload

    def set_annotation_saved_path(self, path: str | Path) -> None:
        self.annotation_path_value.setText(str(path))

    @staticmethod
    def _annotation_decision_label(value: str) -> str:
        return {
            "true_positive": "正确",
            "false_positive": "错误",
            "unreviewed": "未复核",
        }.get(value, value or "未复核")

    def _mark_selected_annotation_box(self, decision: str) -> None:
        item = self.annotation_box_list.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("annotation_kind") == "manual":
            return
        payload["decision"] = decision
        item.setData(0, Qt.ItemDataRole.UserRole, payload)
        item.setText(1, self._annotation_decision_label(decision))
        self._selected_annotation_kind = "predicted"
        self._selected_annotation_id = str(payload.get("box_id"))
        self._sync_annotation_image()
        self._sync_annotation_controls()

    def _add_manual_gt_box(self, bbox: list[int]) -> None:
        if self._annotation_payload is None:
            return
        manual_boxes = list(self._annotation_payload.get("manual_gt_boxes") or [])
        box_id = self._next_manual_box_id(manual_boxes)
        manual_box = {
            "box_id": box_id,
            "bbox": bbox,
            "kind": "manual_missing",
            "source": "manual_draw",
            "note": "",
        }
        manual_boxes.append(manual_box)
        self._annotation_payload["manual_gt_boxes"] = manual_boxes
        tree_item = QTreeWidgetItem(
            [
                box_id,
                "漏检框",
                ",".join(str(value) for value in bbox),
            ]
        )
        tree_payload = dict(manual_box)
        tree_payload["annotation_kind"] = "manual"
        tree_item.setData(0, Qt.ItemDataRole.UserRole, tree_payload)
        self.annotation_box_list.addTopLevelItem(tree_item)
        self.annotation_box_list.setCurrentItem(tree_item)
        self._selected_annotation_kind = "manual"
        self._selected_annotation_id = box_id
        self.set_status(f"已添加漏检框 {len(manual_boxes)} 个，保存后生效")
        self._sync_annotation_image()
        self._sync_annotation_controls()

    def _delete_selected_manual_box(self) -> None:
        item = self.annotation_box_list.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict) or payload.get("annotation_kind") != "manual":
            return
        box_id = str(payload.get("box_id"))
        index = self.annotation_box_list.indexOfTopLevelItem(item)
        if index >= 0:
            self.annotation_box_list.takeTopLevelItem(index)
        if self._annotation_payload is not None:
            manual_boxes = [
                dict(box)
                for box in self._annotation_payload.get("manual_gt_boxes") or []
                if str(box.get("box_id")) != box_id
            ]
            self._annotation_payload["manual_gt_boxes"] = manual_boxes
        self._selected_annotation_kind = None
        self._selected_annotation_id = None
        self._sync_annotation_image()
        self._sync_annotation_controls()

    def _handle_annotation_list_selection_changed(self, current: QTreeWidgetItem | None, *_args) -> None:
        if current is None:
            self._selected_annotation_kind = None
            self._selected_annotation_id = None
            self.annotation_selection_value.setText("未选中")
            self._sync_annotation_image()
            self._sync_annotation_controls()
            return
        payload = current.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        self._selected_annotation_kind = str(payload.get("annotation_kind") or "predicted")
        self._selected_annotation_id = str(payload.get("box_id") or "")
        self._update_annotation_selection_text(payload)
        self._sync_annotation_image()
        self._sync_annotation_controls()

    def _select_annotation_box(self, kind: str, box_id: str) -> None:
        for index in range(self.annotation_box_list.topLevelItemCount()):
            item = self.annotation_box_list.topLevelItem(index)
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(payload, dict):
                continue
            if str(payload.get("annotation_kind")) == kind and str(payload.get("box_id")) == box_id:
                self.annotation_box_list.setCurrentItem(item)
                return
        self._selected_annotation_kind = kind
        self._selected_annotation_id = box_id
        self._sync_annotation_controls()

    def _sync_annotation_image(self) -> None:
        if self._annotation_payload is None:
            self.result_preview_label.set_annotation_boxes([], [])
            return
        predicted_boxes = []
        manual_boxes = []
        for index in range(self.annotation_box_list.topLevelItemCount()):
            tree_item = self.annotation_box_list.topLevelItem(index)
            item = tree_item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            kind = payload.pop("annotation_kind", "predicted")
            if kind == "manual":
                manual_boxes.append(payload)
            else:
                predicted_boxes.append(payload)
        self.result_preview_label.set_annotation_boxes(
            predicted_boxes,
            manual_boxes,
            selected_kind=self._selected_annotation_kind,
            selected_id=self._selected_annotation_id,
        )

    def _update_annotation_selection_text(self, payload: dict) -> None:
        kind = str(payload.get("annotation_kind") or "predicted")
        bbox = payload.get("bbox") or []
        bbox_text = ",".join(str(value) for value in bbox)
        if kind == "manual":
            self.annotation_selection_value.setText(f"当前：漏检框 {payload.get('box_id')}  {bbox_text}")
            return
        decision = self._annotation_decision_label(str(payload.get("decision") or "unreviewed"))
        self.annotation_selection_value.setText(
            f"当前：检测框 {payload.get('box_id')}  {decision}  {bbox_text}"
        )

    @staticmethod
    def _manual_boxes_with_ids(values: object) -> list[dict]:
        boxes = []
        for index, item in enumerate(values if isinstance(values, list) else []):
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload.setdefault("box_id", f"miss_{index}")
            boxes.append(payload)
        return boxes

    @staticmethod
    def _next_manual_box_id(manual_boxes: list[dict]) -> str:
        existing = {str(item.get("box_id")) for item in manual_boxes if isinstance(item, dict)}
        index = 0
        while f"miss_{index}" in existing:
            index += 1
        return f"miss_{index}"

    @staticmethod
    def _format_detection_duration(seconds: float | None) -> str:
        if seconds is None:
            return "-"
        return f"{max(0.0, seconds):.2f} 秒"

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
        self.history_selection_changed.emit(record)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._did_auto_focus_code_input:
            return

        self._did_auto_focus_code_input = True
        self.focus_code_input(select_all=True)

    @staticmethod
    def _build_image_label(
        placeholder: str,
        *,
        preferred_width: int = 420,
        preferred_height: int = 260,
        minimum_width: int = 120,
        minimum_height: int = 220,
    ) -> ScaledImageLabel:
        return ScaledImageLabel(
            placeholder,
            preferred_width=preferred_width,
            preferred_height=preferred_height,
            minimum_width=minimum_width,
            minimum_height=minimum_height,
        )

    def _render_image(
        self, label: ScaledImageLabel, path: Path | None, empty_text: str
    ) -> None:
        if path is None:
            label.clear_preview(empty_text)
            return

        if not path.exists():
            label.clear_preview("图片不存在")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.clear_preview("无法加载图片预览")
            return

        label.set_preview_pixmap(pixmap, empty_text)

    @staticmethod
    def _build_section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700; color: #334e68;")
        return label

    def _sync_run_button(self) -> None:
        self.run_button.setDisabled(
            self._busy
            or self._camera_busy
            or not self._detection_enabled
            or not self._license_allows_detection
        )

    def _sync_capture_controls(self) -> None:
        disabled = self._busy or self._camera_busy
        self.browse_button.setDisabled(disabled)
        self.capture_button.setDisabled(disabled or not self._capture_action_enabled)
        self.next_button.setDisabled(disabled or not self._next_enabled)

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

    def _sync_annotation_controls(self) -> None:
        has_annotation = self._annotation_payload is not None
        has_box = self.annotation_box_list.currentItem() is not None
        self.annotation_notes_edit.setDisabled(not has_annotation)
        current = self.annotation_box_list.currentItem()
        current_payload = (
            current.data(0, Qt.ItemDataRole.UserRole)
            if current is not None
            else None
        )
        selected_predicted = (
            isinstance(current_payload, dict)
            and current_payload.get("annotation_kind") != "manual"
        )
        selected_manual = (
            isinstance(current_payload, dict)
            and current_payload.get("annotation_kind") == "manual"
        )
        self.annotation_mark_tp_button.setDisabled(not has_box or not selected_predicted)
        self.annotation_mark_fp_button.setDisabled(not has_box or not selected_predicted)
        self.annotation_delete_manual_box_button.setDisabled(not has_box or not selected_manual)
        self.annotation_save_button.setDisabled(not has_annotation)

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
