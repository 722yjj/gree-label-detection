"""OCR-text based structured extraction helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable


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


def extract_standard_spec_from_text(ocr_text: object) -> Dict[str, str | None]:
    """Extract high-confidence standard label anchors directly from OCR text."""

    text = unicodedata.normalize("NFKC", str(ocr_text or ""))
    compacted = re.sub(r"[ \t]+", " ", text)

    model_candidates = re.findall(r"GWH[0-9A-Z][A-Z0-9\-/]{6,}", compacted, re.IGNORECASE)
    frequency = _search_group(r"\b([0-9]{2}\s*H+\s*z\.?)\b", compacted)
    heating_capacity = _search_group(
        r"Heating\s+Capacity(?:\s+[A-Za-z]){0,2}\s+([0-9]+(?:\.[0-9]+)?\s*kW)",
        compacted,
    )

    return {
        "model_number": max(model_candidates, key=len).upper() if model_candidates else None,
        "frequency": re.sub(r"\s+", "", frequency) if frequency else None,
        "heating_capacity": re.sub(r"\s+", "", heating_capacity) if heating_capacity else None,
    }


def count_populated_fields(data: Dict[str, object]) -> int:
    return sum(1 for value in data.values() if value not in (None, "", "None"))


COMMON_PIPE_SIZES = {
    "1/4",
    "3/8",
    "1/2",
    "5/8",
    "3/4",
    "7/8",
    "1",
    "1-1/8",
    "1-3/8",
}

COMMON_REFRIGERANTS = {
    "R22",
    "R32",
    "R134A",
    "R290",
    "R410A",
    "R407C",
    "R417A",
    "R454B",
    "R600A",
}


def _normalize_compact_value(field_name: str, value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\s+", "", text)
    if field_name in {"net_weight", "gross_weight"}:
        return text.lower()
    return text.upper()


def is_suspicious_compact_field(field_name: str, value: object) -> bool:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return True

    normalized = _normalize_compact_value(field_name, text)
    if field_name == "model_number":
        return len(normalized) < 10
    if field_name in {"net_weight", "gross_weight"}:
        return re.fullmatch(r"\d+(?:\.\d+)?kg", normalized) is None
    if field_name == "color":
        return re.fullmatch(r"[A-Za-z]+", text) is None
    if field_name == "barcode":
        digits = re.sub(r"\D+", "", text)
        return len(digits) < 10
    if field_name == "refrigerant":
        return normalized not in COMMON_REFRIGERANTS
    if field_name == "connection_pipes":
        parts = re.findall(r"\d+\s*/\s*\d+", text)
        if len(parts) < 2:
            return True
        normalized_parts = {part.replace(" ", "") for part in parts}
        return any(part not in COMMON_PIPE_SIZES for part in normalized_parts)
    return False


def find_missing_fields(
    data: Dict[str, object],
    field_names: Iterable[str],
) -> list[str]:
    return [
        field_name
        for field_name in field_names
        if data.get(field_name) in (None, "", "None")
    ]


def find_suspicious_fields(
    data: Dict[str, object],
    field_names: Iterable[str],
) -> list[str]:
    return [
        field_name
        for field_name in field_names
        if data.get(field_name) not in (None, "", "None")
        and is_suspicious_compact_field(field_name, data.get(field_name))
    ]


def needs_compact_llm(
    extracted: Dict[str, object],
    field_names: Iterable[str],
) -> bool:
    return bool(
        find_missing_fields(extracted, field_names)
        or find_suspicious_fields(extracted, field_names)
    )


def merge_compact_sources(
    rule_data: Dict[str, object],
    llm_data: Dict[str, object],
    field_names: Iterable[str],
) -> Dict[str, object]:
    """
    Merge OCR-rule extraction with image-based LLM extraction.

    Anchors like `model_number` and `barcode` prefer OCR/regex when present.
    For visually short fields, prefer LLM when it provides a plausible value.
    """
    merged: Dict[str, object] = {}
    anchor_fields = {"model_number", "barcode"}

    for field_name in field_names:
        rule_val = rule_data.get(field_name)
        llm_val = llm_data.get(field_name)
        rule_ok = rule_val not in (None, "", "None") and not is_suspicious_compact_field(
            field_name, rule_val
        )
        llm_ok = llm_val not in (None, "", "None") and not is_suspicious_compact_field(
            field_name, llm_val
        )

        if field_name in anchor_fields:
            merged[field_name] = rule_val if rule_ok or llm_val in (None, "", "None") else llm_val
            if merged[field_name] in (None, "", "None"):
                merged[field_name] = rule_val or llm_val
            continue

        if llm_ok:
            merged[field_name] = llm_val
        elif rule_ok:
            merged[field_name] = rule_val
        else:
            merged[field_name] = llm_val or rule_val

    return merged


STANDARD_OCR_ANCHOR_FIELDS = {"model_number", "frequency", "heating_capacity"}


def merge_standard_sources(
    rule_data: Dict[str, object],
    llm_data: Dict[str, object],
    field_names: Iterable[str],
) -> Dict[str, object]:
    """
    Merge standard-label OCR anchors with LLM extraction.

    The LLM is useful for layout understanding, but it can autocorrect actual
    printed defects such as "50HHz" into the expected value "50Hz". For a small
    set of high-confidence OCR anchors, preserve the OCR text when present.
    """

    merged: Dict[str, object] = {}
    for field_name in field_names:
        rule_val = rule_data.get(field_name)
        llm_val = llm_data.get(field_name)
        if field_name in STANDARD_OCR_ANCHOR_FIELDS and rule_val not in (
            None,
            "",
            "None",
        ):
            merged[field_name] = rule_val
        else:
            merged[field_name] = llm_val
    return merged
