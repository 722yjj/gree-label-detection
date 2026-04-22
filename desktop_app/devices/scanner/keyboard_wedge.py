"""Keyboard-wedge scanner input normalization helpers."""

from __future__ import annotations

from typing import Iterable

from label_detection.batch_samples import extract_label_code


class KeyboardWedgeScannerInput:
    """Normalize keyboard-wedge scanner input for the UI."""

    _CONTROL_CHAR_MAP = str.maketrans(
        {
            "\r": "",
            "\n": "",
            "\t": "",
            "\x00": "",
        }
    )

    @classmethod
    def normalize(
        cls,
        raw_value: str,
        *,
        strip_prefixes: Iterable[str] | None = None,
        strip_suffixes: Iterable[str] | None = None,
        extract_main_code: bool = False,
    ) -> str:
        value = str(raw_value or "").translate(cls._CONTROL_CHAR_MAP).strip()
        value = cls._strip_known_prefixes(value, strip_prefixes)
        value = cls._strip_known_suffixes(value, strip_suffixes)

        if extract_main_code and value:
            extracted = extract_label_code(value)
            if extracted:
                return extracted

        return value

    @staticmethod
    def _strip_known_prefixes(value: str, prefixes: Iterable[str] | None) -> str:
        cleaned = value
        for prefix in prefixes or ():
            if prefix and cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned

    @staticmethod
    def _strip_known_suffixes(value: str, suffixes: Iterable[str] | None) -> str:
        cleaned = value
        for suffix in suffixes or ():
            if suffix and cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        return cleaned
