from desktop_app.devices.scanner_input import KeyboardWedgeScannerInput


def test_normalize_removes_control_chars_and_known_affixes():
    assert (
        KeyboardWedgeScannerInput.normalize(
            "\tPREFIX:600004075219\r\n",
            strip_prefixes=("PREFIX:",),
        )
        == "600004075219"
    )


def test_normalize_can_extract_main_code_when_requested():
    assert (
        KeyboardWedgeScannerInput.normalize(
            "barcode=600004075219-01",
            extract_main_code=True,
        )
        == "600004075219"
    )
