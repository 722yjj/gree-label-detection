import os

import pytest

from desktop_app.devices.scanner_input import KeyboardWedgeScannerInput


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_normalize_removes_control_chars_and_known_affixes():
    assert (
        KeyboardWedgeScannerInput.normalize(
            "\tPREFIX:600004075219\r\n",
            strip_prefixes=("PREFIX:",),
        )
        == "600004075219"
    )


def test_keyboard_wedge_adapter_start_installs_global_filter_and_stop_removes_it():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QLineEdit

    from desktop_app.devices.scanner.keyboard_wedge_adapter import (
        KeyboardWedgeScannerAdapter,
    )

    app = QApplication.instance() or QApplication([])
    target = QLineEdit()
    target.show()
    target.activateWindow()
    target.setFocus()
    app.processEvents()

    adapter = KeyboardWedgeScannerAdapter(target)
    scanned_codes: list[str] = []
    adapter.code_scanned.connect(scanned_codes.append)

    try:
        adapter.start()
        for character in "600004075219":
            event = QKeyEvent(
                QEvent.Type.KeyPress,
                0,
                Qt.KeyboardModifier.NoModifier,
                character,
            )
            QApplication.sendEvent(target, event)
        QApplication.sendEvent(
            target,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
                "\r",
            ),
        )
        app.processEvents()

        assert scanned_codes == ["600004075219"]

        adapter.stop()
        QApplication.sendEvent(
            target,
            QKeyEvent(
                QEvent.Type.KeyPress,
                0,
                Qt.KeyboardModifier.NoModifier,
                "8",
            ),
        )
        QApplication.sendEvent(
            target,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
                "\r",
            ),
        )
        app.processEvents()

        assert scanned_codes == ["600004075219"]
    finally:
        adapter.stop()
        target.close()


def test_normalize_can_extract_main_code_when_requested():
    assert (
        KeyboardWedgeScannerInput.normalize(
            "barcode=600004075219-01",
            extract_main_code=True,
        )
        == "600004075219"
    )
