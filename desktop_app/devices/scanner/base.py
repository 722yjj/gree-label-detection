"""Base abstractions for desktop scanner adapters."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class ScannerAdapter(QObject):
    """Base signal-driven scanner adapter interface."""

    code_scanned = Signal(str)
    availability_changed = Signal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def start(self) -> None:
        """Start reading scanner input."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop reading scanner input."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Return whether the adapter is ready to emit scan results."""
        raise NotImplementedError
