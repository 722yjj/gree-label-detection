"""Scanner input helpers and optional device adapters."""

from desktop_app.devices.scanner.keyboard_wedge import KeyboardWedgeScannerInput

__all__ = ["KeyboardWedgeScannerInput"]

try:
    from desktop_app.devices.scanner.base import ScannerAdapter
    from desktop_app.devices.scanner.mock import MockScannerAdapter
    from desktop_app.devices.scanner.mv_orh import MVORHScannerAdapter
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
else:
    __all__.extend(
        [
            "ScannerAdapter",
            "MockScannerAdapter",
            "MVORHScannerAdapter",
        ]
    )
