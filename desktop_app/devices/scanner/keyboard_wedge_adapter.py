"""Qt event-filter adapter for USB HID keyboard-wedge scanners."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QWidget

from desktop_app.devices.scanner.base import ScannerAdapter


class KeyboardWedgeScannerAdapter(ScannerAdapter):
    """Capture keyboard-wedge scans while the desktop window is active."""

    scan_buffer_changed = Signal(str)

    def __init__(self, active_window: QWidget | None = None) -> None:
        super().__init__(active_window)
        self._active_window = active_window
        self._application: QApplication | None = None
        self._buffer: list[str] = []
        self._started = False
        self._available = False

    def start(self) -> None:
        if self._started:
            return

        application = QApplication.instance()
        if application is None:
            raise RuntimeError("QApplication 尚未初始化")

        application.installEventFilter(self)
        self._application = application
        self._started = True
        self._available = True
        self.availability_changed.emit(True, "键盘扫码模式已启用")

    def stop(self) -> None:
        if not self._started:
            return

        if self._application is not None:
            self._application.removeEventFilter(self)

        self._application = None
        self._buffer.clear()
        self._started = False
        self._available = False
        self.availability_changed.emit(False, "键盘扫码模式已停止")

    def is_available(self) -> bool:
        return self._available

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            not getattr(self, "_started", False)
            or event.type() != QEvent.Type.KeyPress
        ):
            return False
        if not self._should_capture_active_window():
            return False

        key = event.key()
        text = event.text()
        modifiers = event.modifiers()
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self._commit_buffer()

        if key == Qt.Key.Key_Escape:
            self._clear_buffer()
            return False

        if text and text not in ("\r", "\n"):
            self._append_text(text)
            return True

        return False

    def _should_capture_active_window(self) -> bool:
        application = self._application or QApplication.instance()
        if application is None:
            return False

        active_window = application.activeWindow()
        if active_window is None:
            return False

        if self._active_window is None:
            return True

        configured_window = self._active_window.window()
        return active_window is configured_window or active_window == configured_window

    def _append_text(self, text: str) -> None:
        self._buffer.append(text)
        self.scan_buffer_changed.emit("".join(self._buffer))

    def _commit_buffer(self) -> bool:
        if not self._buffer:
            return False

        raw_code = "".join(self._buffer)
        self._buffer.clear()
        self.code_scanned.emit(raw_code)
        return True

    def _clear_buffer(self) -> None:
        if not self._buffer:
            return
        self._buffer.clear()
        self.scan_buffer_changed.emit("")
