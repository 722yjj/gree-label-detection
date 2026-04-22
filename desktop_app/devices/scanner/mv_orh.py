"""Placeholder adapter for MV-ORH scanners."""

from __future__ import annotations

from desktop_app.devices.scanner.base import ScannerAdapter


class MVORHScannerAdapter(ScannerAdapter):
    """Reserved adapter for MV-ORH devices once the protocol is confirmed."""

    def __init__(self) -> None:
        super().__init__()
        self._available = False

    def start(self) -> None:
        self._available = False
        self.availability_changed.emit(False, "MV-ORH 协议未实现")

    def stop(self) -> None:
        self._available = False

    def is_available(self) -> bool:
        return self._available
