"""Mock scanner adapter for development and tests."""

from __future__ import annotations

from desktop_app.devices.scanner.base import ScannerAdapter


class MockScannerAdapter(ScannerAdapter):
    """A small signal emitter that simulates a connected scanner."""

    def __init__(self) -> None:
        super().__init__()
        self._started = False
        self._available = False

    def start(self) -> None:
        self._started = True
        self._available = True
        self.availability_changed.emit(True, "mock scanner ready")

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._available = False
        self.availability_changed.emit(False, "mock scanner stopped")

    def is_available(self) -> bool:
        return self._available

    def emit_code(self, code: str) -> None:
        if not self._available:
            return
        self.code_scanned.emit(code)
