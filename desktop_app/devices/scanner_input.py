"""Scanner input helpers."""


class KeyboardWedgeScannerInput:
    """Normalize keyboard-wedge scanner input for the UI."""

    @staticmethod
    def normalize(raw_value: str) -> str:
        return str(raw_value or "").strip()

