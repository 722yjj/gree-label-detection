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


def _search_group(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return _clean_match(match.group(1))


def _strip_inner_spaces(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", "", value)


def _compact_value(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", "", text)


def _llm_value_completes_ocr_value(rule_val: object, llm_val: object) -> bool:
    """
    Return whether the LLM value appears to be the OCR value plus a short suffix.

    This preserves OCR anchors for real printed defects, but avoids dropping
    trailing symbols such as the `~` in `220-240V~` when OCR clipped them.
    """
    rule_text = _compact_value(rule_val)
    llm_text = _compact_value(llm_val)
    if not rule_text or not llm_text:
        return False
    if len(llm_text) <= len(rule_text):
        return False
    if not llm_text.casefold().startswith(rule_text.casefold()):
        return False

    suffix = llm_text[len(rule_text) :]
    return 0 < len(suffix) <= 4


def _search_label_value(label_pattern: str, value_pattern: str, text: str) -> str | None:
    return _search_group(
        rf"{label_pattern}\s*[:：]?\s*[\r\n ]*({value_pattern})",
        text,
        re.IGNORECASE | re.DOTALL,
    )


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
    """Extract high-confidence standard label fields directly from OCR text."""

    text = unicodedata.normalize("NFKC", str(ocr_text or ""))
    compacted = re.sub(r"[ \t]+", " ", text)
    one_line = re.sub(r"\s+", " ", compacted)

    model_candidates = re.findall(r"GWH[0-9A-Z][A-Z0-9\-/]{6,}", compacted, re.IGNORECASE)
    voltage = _search_label_value(
        r"(?:Rated\s+)?Voltage",
        r"[0-9OIl]{2,3}\s*[-–]\s*[0-9OIl]{2,3}\s*V\s*~?",
        compacted,
    ) or _search_group(
        r"\b([0-9OIl]{2,3}\s*[-–]\s*[0-9OIl]{2,3}\s*V\s*~?)\b",
        compacted,
    )
    frequency = _search_label_value(
        r"(?:Rated\s+)?Frequency",
        r"[0-9OIl]{2,3}\s*H+\s*z\.?",
        compacted,
    ) or _search_group(r"\b([0-9OIl]{2,3}\s*H+\s*z\.?)\b", compacted)
    capacity_pattern = r"[0-9OIl]+(?:[.,][0-9OIl]+)?\s*k?\s*W"
    heating_capacity = _search_label_value(
        r"Heating\s+Capacity(?:\s+[A-Za-z]){0,2}",
        capacity_pattern,
        compacted,
    )
    cooling_capacity = _search_label_value(
        r"Cooling\s+Capacity(?:\s+[A-Za-z]){0,2}",
        capacity_pattern,
        compacted,
    )
    air_volume = _search_label_value(
        r"Air\s*(?:Flow\s*)?Volume(?:\s+m\s*(?:3|\^3)?\s*/\s*h)?",
        r"[0-9OIl]+(?:[.,][0-9OIl]+)?\s*m\s*(?:3|\^3)?\s*/\s*h",
        compacted,
    )
    weight = _search_label_value(
        r"(?<!Net\s)(?<!Gross\s)Weight(?:\s+kg)?",
        r"[0-9OIl]+(?:[.,-][0-9OIl]+)?\s*kg",
        compacted,
    )
    noise = _search_label_value(
        r"(?:Sound\s+Pressure\s+Level\s*\(?H\)?|Noise(?:\s+Level)?)(?:\s+dB\s*\(?A\)?)?",
        r"[0-9OIl]+(?:[.,][0-9OIl]+)?\s*dB\s*\(?A\)?",
        compacted,
    )
    mfg_date = _search_label_value(
        r"(?:Manufactured|Mfg\.?)\s+Date",
        r"[A-Z0-9]{4}(?:[.\-/]?[A-Z0-9]{1,4})?",
        compacted,
    )
    product_type = _search_group(
        r"\b(SPLIT\s+AIR\s+CONDITIONER\s+INDOOR\s+UNIT)\b",
        one_line,
    )
    manufacturer = _search_group(
        r"\b(GREE\s+ELECTRIC\s+APPLIANCES\s*,?\s*INC\.?\s*OF\s+ZHUHAI)\b",
        one_line,
    )
    address = _search_group(
        r"\b(Add\s*:?\s*West\s+Jinji\s+Rd\s*,?\s*Qianshan\s*,?\s*Zhuhai\s*,?\s*Guangdong\s*,?\s*China\s*,?\s*519070)\b",
        one_line,
    )
    barcode_candidates = re.findall(r"\b\d{10,13}\b", compacted)

    return {
        "brand": "GREE" if re.search(r"\bGREE\b", one_line, re.IGNORECASE) else None,
        "product_type": product_type,
        "model_number": max(model_candidates, key=len).upper() if model_candidates else None,
        "voltage": _strip_inner_spaces(voltage),
        "frequency": _strip_inner_spaces(frequency),
        "heating_capacity": _strip_inner_spaces(heating_capacity),
        "cooling_capacity": _strip_inner_spaces(cooling_capacity),
        "air_volume": _strip_inner_spaces(air_volume),
        "weight": _strip_inner_spaces(weight),
        "noise": _strip_inner_spaces(noise),
        "mfg_date": _strip_inner_spaces(mfg_date),
        "manufacturer": manufacturer,
        "address": address,
        "barcode": max(barcode_candidates, key=len) if barcode_candidates else None,
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


STANDARD_OCR_ANCHOR_FIELDS = {
    "model_number",
    "voltage",
    "frequency",
    "heating_capacity",
    "cooling_capacity",
    "air_volume",
    "weight",
    "noise",
    "mfg_date",
    "barcode",
}


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
        if (
            field_name in STANDARD_OCR_ANCHOR_FIELDS
            and _llm_value_completes_ocr_value(rule_val, llm_val)
        ):
            merged[field_name] = llm_val
        elif field_name in STANDARD_OCR_ANCHOR_FIELDS and rule_val not in (None, "", "None"):
            merged[field_name] = rule_val
        elif llm_val in (None, "", "None") and rule_val not in (None, "", "None"):
            merged[field_name] = rule_val
        else:
            merged[field_name] = llm_val
    return merged
