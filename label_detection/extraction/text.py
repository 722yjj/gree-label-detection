"""OCR-text based structured extraction helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict


def _clean_match(value: str | None) -> str | None:
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value or None


def _search_group(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _clean_match(match.group(1))


def extract_compact_spec_from_text(ocr_text: object) -> Dict[str, str | None]:
    """
    Extract fields for compact labels directly from OCR text.

    This label family is highly regular, so a lightweight regex pass is more
    stable than forcing it through the standard indoor-unit schema prompt.
    """
    text = unicodedata.normalize("NFKC", str(ocr_text or ""))
    compacted = re.sub(r"[ \t]+", " ", text)
    upper = compacted.upper()

    model_candidates = re.findall(r"[A-Z0-9][A-Z0-9\-\/\(\)]{8,}", upper)
    barcode_candidates = re.findall(r"\d{10,13}", compacted)

    return {
        "model_number": max(model_candidates, key=len) if model_candidates else None,
        "net_weight": _search_group(
            r"N\s*\.?\s*W\s*\.?\s*:?\s*([0-9]+(?:\.[0-9]+)?\s*KG)",
            compacted,
        ),
        "gross_weight": _search_group(
            r"G\s*\.?\s*W\s*\.?\s*:?\s*([0-9]+(?:\.[0-9]+)?\s*KG)",
            compacted,
        ),
        "color": _search_group(r"COLOR\s*:?\s*([A-Z]+)", compacted),
        "connection_pipes": _search_group(
            r'CONNECTION\s*PIPES?\s*:?\s*([0-9\/\.\s"\']+)',
            compacted,
        ),
        "refrigerant": _search_group(r"REFRIGERANT\s*:?\s*([A-Z0-9]+)", compacted),
        "barcode": max(barcode_candidates, key=len) if barcode_candidates else None,
    }


def count_populated_fields(data: Dict[str, object]) -> int:
    return sum(1 for value in data.values() if value not in (None, "", "None"))
