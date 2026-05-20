"""Executable entry point for the PySide6 desktop app."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Start the desktop app when PySide6 is available."""

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print("PySide6 未安装。请先执行: uv sync --extra desktop")
        return 1

    from desktop_app.app import build_main_window
    from label_detection.license import verify_license

    app = QApplication(argv or sys.argv)
    license_status = verify_license()
    window = build_main_window(license_status=license_status)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
