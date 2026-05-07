"""
OCR text box matching helpers.

Keep this module dependency-light so it can be unit tested without loading
the OCR engine or LLM stack.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple


OCRBox = Tuple[object, str, float]


FIELD_LABEL_ALIASES: Dict[str, Sequence[str]] = {
    "net_weight": ("nw", "nw", "netweight"),
    "gross_weight": ("gw", "gw", "grossweight"),
    "connection_pipes": ("connectionpipes", "connectionpipe", "pipes"),
    "refrigerant": ("refrigerant",),
    "color": ("color",),
    "barcode": ("serialno", "serialnumber", "barcode"),
    "model_number": ("model",),
    "weight": ("weight",),
    "frequency": ("ratedfrequency", "frequency"),
    "voltage": ("ratedvoltage", "voltage"),
    "air_volume": ("airflowvolume", "airvolume"),
    "cooling_capacity": ("coolingcapacity",),
    "heating_capacity": ("heatingcapacity",),
    "noise": ("soundpressurelevelh", "soundpressurelevel", "noiselevel", "noise"),
    "mfg_date": ("manufactureddate", "mfgdate", "date"),
    "address": ("address", "add"),
}

LABEL_FIELD_PREFIX = "label:"

FIELD_LABEL_PATTERNS: Dict[str, str] = {
    "model_number": r"(?<![A-Za-z])(Model)(?![A-Za-z])",
    "net_weight": r"(?<![A-Za-z])(N\s*\.?\s*W\s*\.?|Net\s*Weight)(?![A-Za-z])",
    "gross_weight": r"(?<![A-Za-z])(G\s*\.?\s*W\s*\.?|Gross\s*Weight)(?![A-Za-z])",
    "color": r"(?<![A-Za-z])(Color)(?![A-Za-z])",
    "connection_pipes": r"(?<![A-Za-z])(Connection\s*Pipes?)(?![A-Za-z])",
    "refrigerant": r"(?<![A-Za-z])(Refrigerant)(?![A-Za-z])",
    "barcode": r"(?<![A-Za-z])(Serial\s*No\.*|Serial\s*Number|Barcode)(?![A-Za-z])",
    "weight": r"(?<![A-Za-z])(Weight)(?![A-Za-z])",
    "frequency": r"(?<![A-Za-z])(Rated\s*Frequency|Frequency)(?![A-Za-z])",
    "voltage": r"(?<![A-Za-z])(Rated\s*Voltage|Voltage)(?![A-Za-z])",
    "air_volume": r"(?<![A-Za-z])(Air\s*Flow\s*Volume|Air\s*Volume)(?![A-Za-z])",
    "cooling_capacity": r"(?<![A-Za-z])(Cooling\s*Capacity)(?![A-Za-z])",
    "heating_capacity": r"(?<![A-Za-z])(Heating\s*Capacity)(?![A-Za-z])",
    "noise": r"(?<![A-Za-z])(Sound\s*Pressure\s*Level(?:\s*\(H\))?|Noise(?:\s*Level)?)(?![A-Za-z])",
    "mfg_date": r"(?<![A-Za-z])(Manufactured\s*Date|MFG\s*Date|Date)(?![A-Za-z])",
    "address": r"(?<![A-Za-z])(Add(?:ress)?\.?)(?![A-Za-z])",
}

FIELD_LABEL_REGEXES: Dict[str, re.Pattern[str]] = {
    field_name: re.compile(pattern, re.IGNORECASE)
    for field_name, pattern in FIELD_LABEL_PATTERNS.items()
}

FIELD_LABEL_FUZZY_MIN_SCORE = 0.88
FIELD_LABEL_ALIAS_COMPARE_MIN_SCORE = 0.94
MAX_LABEL_WORD_CANDIDATE_LENGTH = 4


def make_label_field_name(field_name: str) -> str:
    return f"{LABEL_FIELD_PREFIX}{field_name}"


def is_label_field_name(field_name: object) -> bool:
    return str(field_name or "").startswith(LABEL_FIELD_PREFIX)


def get_base_field_name(field_name: object) -> str:
    normalized = str(field_name or "")
    if is_label_field_name(normalized):
        return normalized[len(LABEL_FIELD_PREFIX) :]
    return normalized


def normalize_text_for_match(text: object) -> str:
    """Normalize OCR / extracted text before matching."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("³", "3")
    return re.sub(r"[^0-9a-z]+", "", normalized)


def normalize_text_for_compare(text: object) -> str:
    """
    Normalize extracted field values for conservative equality checks.

    This is intentionally less aggressive than `normalize_text_for_match`:
    punctuation such as decimal points and hyphens remains significant, while
    whitespace-only formatting noise is ignored.
    """
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if not normalized or normalized.lower() == "none":
        return ""

    normalized = normalized.replace("³", "3")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([,.:;/~()\-])\s*", r"\1", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[a-zA-Z])", "", normalized)
    normalized = re.sub(r"(?<=[a-zA-Z])\s+(?=\d)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def field_values_match(expected: object, actual: object) -> bool:
    """Return whether two extracted field values are equivalent."""
    return normalize_text_for_compare(expected) == normalize_text_for_compare(actual)


def normalize_label_text_for_compare(text: object) -> str:
    """Normalize label captions while keeping case changes significant."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if not normalized or normalized.lower() == "none":
        return ""

    normalized = normalized.replace("³", "3")
    return re.sub(r"[^0-9A-Za-z]+", "", normalized)


def _normalize_barcode_label_text_for_compare(text: object) -> str:
    """Normalize barcode captions while preserving printed punctuation."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if not normalized or normalized.lower() == "none":
        return ""

    normalized = normalized.replace("³", "3")
    return re.sub(r"\s+", "", normalized)


def _label_tokens_for_compare(text: object) -> List[str]:
    """Tokenize label captions and repair common OCR single-letter splits."""
    if text is None:
        return []

    normalized = unicodedata.normalize("NFKC", str(text)).replace("³", "3")
    raw_tokens = re.findall(r"[A-Za-z]+|\d+", normalized)
    tokens: List[str] = []
    for token in raw_tokens:
        if (
            token.isalpha()
            and len(token) == 1
            and tokens
            and tokens[-1].isalpha()
        ):
            tokens[-1] = f"{tokens[-1]}{token}"
        else:
            tokens.append(token)
    return [token.casefold() for token in tokens if token]


def label_values_match(expected: object, actual: object) -> bool:
    return normalize_label_text_for_compare(expected) == normalize_label_text_for_compare(actual)


def label_field_values_match(field_name: object, expected: object, actual: object) -> bool:
    base_field_name = get_base_field_name(field_name)
    if base_field_name == "barcode":
        expected_strict_norm = _normalize_barcode_label_text_for_compare(expected)
        actual_strict_norm = _normalize_barcode_label_text_for_compare(actual)
        if expected_strict_norm == actual_strict_norm:
            return True
        if expected_strict_norm.casefold() == actual_strict_norm.casefold():
            return False

    expected_norm = normalize_label_text_for_compare(expected)
    actual_norm = normalize_label_text_for_compare(actual)
    if expected_norm == actual_norm:
        if base_field_name == "barcode":
            return False
        return True

    # Keep OCR label comparisons case-sensitive when the text is otherwise the
    # same: "Weight" and "weiGht" should still surface as a possible print/OCR
    # issue. Alias matching below is for genuinely shorter but valid captions
    # such as "Date" vs "Manufactured Date".
    if expected_norm.casefold() == actual_norm.casefold():
        return False

    alias_norms = {
        normalize_text_for_match(alias)
        for alias in FIELD_LABEL_ALIASES.get(base_field_name, ())
    }
    if not alias_norms:
        return False

    expected_match_norm = normalize_text_for_match(expected)
    actual_match_norm = normalize_text_for_match(actual)
    for alias_norm in alias_norms:
        if expected_match_norm != alias_norm:
            continue

        score = SequenceMatcher(None, actual_match_norm, alias_norm).ratio()
        length_gap = abs(len(actual_match_norm) - len(alias_norm))
        if (
            score >= FIELD_LABEL_ALIAS_COMPARE_MIN_SCORE
            and length_gap <= max(1, int(len(alias_norm) * 0.12))
        ):
            return True

    return (
        expected_match_norm in alias_norms
        and actual_match_norm in alias_norms
    ) or _label_token_sets_match(expected, actual)


def _label_token_sets_match(expected: object, actual: object) -> bool:
    expected_tokens = _label_tokens_for_compare(expected)
    actual_tokens = _label_tokens_for_compare(actual)
    if len(expected_tokens) < 2 or len(actual_tokens) < 2:
        return False

    expected_set = set(expected_tokens)
    actual_set = set(actual_tokens)
    return expected_set == actual_set


def text_field_values_match(field_name: object, expected: object, actual: object) -> bool:
    if is_label_field_name(field_name):
        return label_field_values_match(field_name, expected, actual)
    return field_values_match(expected, actual)


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_box_rect(points: object) -> tuple[float, float, float, float] | None:
    if points is None:
        return None

    try:
        point_list = list(points)
    except TypeError:
        return None

    if len(point_list) == 4:
        scalars = [_coerce_float(value) for value in point_list]
        if all(value is not None for value in scalars):
            x1, y1, x2, y2 = scalars
            return x1, y1, x2, y2

    xy_points = []
    for point in point_list:
        try:
            coords = list(point)
        except TypeError:
            continue
        if len(coords) < 2:
            continue
        x = _coerce_float(coords[0])
        y = _coerce_float(coords[1])
        if x is None or y is None:
            continue
        xy_points.append((x, y))

    if not xy_points:
        return None

    xs = [point[0] for point in xy_points]
    ys = [point[1] for point in xy_points]
    return min(xs), min(ys), max(xs), max(ys)


def _boxes_share_text_line(left: Dict[str, object], right: Dict[str, object]) -> bool:
    vertical_overlap = max(
        0.0,
        min(float(left["y2"]), float(right["y2"]))
        - max(float(left["y1"]), float(right["y1"])),
    )
    min_height = min(float(left["height"]), float(right["height"]))
    if vertical_overlap < 0.45 * min_height:
        return False

    center_delta = abs(float(left["center_y"]) - float(right["center_y"]))
    max_height = max(float(left["height"]), float(right["height"]))
    return center_delta <= 0.55 * max_height


def _boxes_can_chain_as_label_words(left: Dict[str, object], right: Dict[str, object]) -> bool:
    if not _boxes_share_text_line(left, right):
        return False

    max_height = max(float(left["height"]), float(right["height"]))
    horizontal_gap = float(right["x1"]) - float(left["x2"])
    if horizontal_gap < -0.60 * max_height:
        return False

    return horizontal_gap <= max(48.0, 1.25 * max_height)


def _iter_label_candidates(ocr_boxes: Sequence[OCRBox]) -> List[Dict[str, object]]:
    box_metas: List[Dict[str, object]] = []
    for idx, box_info in enumerate(ocr_boxes):
        box_text = box_info[1] if len(box_info) > 1 else ""
        raw_text = unicodedata.normalize("NFKC", str(box_text or "")).strip()
        if not raw_text:
            continue

        rect = _extract_box_rect(box_info[0] if box_info else None)
        if rect is None:
            continue

        x1, y1, x2, y2 = rect
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        confidence = (
            float(box_info[2])
            if len(box_info) > 2 and isinstance(box_info[2], (int, float))
            else 0.0
        )
        box_metas.append(
            {
                "text": raw_text,
                "box_indices": [idx],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "center_y": (y1 + y2) / 2.0,
                "confidence": confidence,
            }
        )

    candidates: List[Dict[str, object]] = []
    seen_candidates: set[tuple[int, ...]] = set()

    def add_candidate(items: Sequence[Dict[str, object]]) -> None:
        if not items:
            return

        box_indices = [
            box_idx
            for item in items
            for box_idx in list(item.get("box_indices") or [])
        ]
        candidate_key = tuple(int(box_idx) for box_idx in box_indices)
        if candidate_key in seen_candidates:
            return
        seen_candidates.add(candidate_key)

        candidates.append(
            {
                "text": " ".join(str(item["text"]).strip() for item in items).strip(),
                "box_indices": box_indices,
                "confidence": min(float(item["confidence"]) for item in items),
            }
        )

    for meta in box_metas:
        add_candidate([meta])

    x_ordered_metas = sorted(box_metas, key=lambda item: (item["x1"], item["center_y"]))
    for start_idx, start in enumerate(x_ordered_metas):
        sequence = [start]
        previous = start
        for candidate in x_ordered_metas[start_idx + 1 :]:
            if not _boxes_share_text_line(start, candidate):
                continue

            if not _boxes_can_chain_as_label_words(previous, candidate):
                if float(candidate["x1"]) >= float(previous["x2"]):
                    break
                continue

            sequence.append(candidate)
            add_candidate(sequence)
            previous = candidate

            if len(sequence) >= MAX_LABEL_WORD_CANDIDATE_LENGTH:
                break

    return candidates


def _extract_label_match_from_candidate(
    field_name: str,
    raw_text: str,
) -> tuple[str, int, float] | None:
    pattern = FIELD_LABEL_REGEXES.get(field_name)
    if pattern is not None:
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip(), 1, 1.0

    alpha_count = len(re.findall(r"[A-Za-z]", raw_text))
    digit_count = len(re.findall(r"\d", raw_text))
    if alpha_count < 3 or digit_count > 1:
        return None

    raw_norm = normalize_text_for_match(raw_text)
    if len(raw_norm) < 3:
        return None

    best_alias = None
    best_score = 0.0
    for alias in FIELD_LABEL_ALIASES.get(field_name, ()):
        alias_norm = normalize_text_for_match(alias)
        if len(alias_norm) < 2:
            continue
        score = SequenceMatcher(None, raw_norm, alias_norm).ratio()
        if score > best_score:
            best_score = score
            best_alias = alias_norm

    if best_alias is None or best_score < FIELD_LABEL_FUZZY_MIN_SCORE:
        return None

    if abs(len(raw_norm) - len(best_alias)) > max(2, int(len(best_alias) * 0.35)):
        return None

    return raw_text.strip(), 0, best_score


def extract_field_labels_from_ocr_boxes(
    ocr_boxes: Sequence[OCRBox],
    field_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    """
    Extract observed field-label captions from OCR boxes.

    Only returns fields with a confident label-like OCR hit.
    """
    observed: Dict[str, Dict[str, object]] = {}
    candidates = _iter_label_candidates(ocr_boxes)

    for field_name in field_names:
        if field_name not in FIELD_LABEL_REGEXES and field_name not in FIELD_LABEL_ALIASES:
            continue

        best_candidate = None
        for candidate in candidates:
            raw_text = str(candidate.get("text") or "").strip()
            if not raw_text:
                continue

            match_result = _extract_label_match_from_candidate(field_name, raw_text)
            if match_result is None:
                continue

            label_text, exact_rank, match_score = match_result
            label_norm = normalize_label_text_for_compare(label_text)
            box_norm = normalize_label_text_for_compare(raw_text)
            if not label_norm or not box_norm:
                continue

            extra_chars = max(len(box_norm) - len(label_norm), 0)
            confidence = float(candidate.get("confidence") or 0.0)
            score = (exact_rank, match_score, len(label_norm), -extra_chars, confidence)

            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "text": label_text,
                    "box_indices": list(candidate.get("box_indices") or []),
                    "score": score,
                }

        if best_candidate is not None:
            observed[field_name] = {
                "text": best_candidate["text"],
                "box_indices": best_candidate["box_indices"],
            }

    return observed


def _normalize_raw_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", "", normalized)


def _digit_signature(text: str) -> str:
    return re.sub(r"\D+", "", text)


def _air_volume_parts(text: object) -> Tuple[str, bool]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.replace("³", "3")
    compact = re.sub(r"\s+", "", normalized)
    if "m" not in compact:
        return "", False

    unit_like = "/" in compact or "h" in compact or "3" in compact or "$" in compact
    if not unit_like:
        return "", False

    match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*m", compact)
    if not match:
        return "", False
    number = re.sub(r"\D+", "", match.group(1))
    return number, bool(number)


def _air_volume_match_score(target_value: object, ocr_text: object) -> float:
    target_number, target_ok = _air_volume_parts(target_value)
    ocr_number, ocr_ok = _air_volume_parts(ocr_text)
    if not target_ok or not ocr_ok or target_number != ocr_number:
        return 0.0

    target_norm = normalize_text_for_match(target_value)
    ocr_norm = normalize_text_for_match(ocr_text)
    if not target_norm or not ocr_norm:
        return 0.0

    # OCR often drops the tiny superscript in m³/h, or reads the trailing h as
    # another symbol. The numeric value and unit marker are the reliable anchors.
    text_ratio = SequenceMatcher(None, target_norm, ocr_norm).ratio()
    return max(0.90, text_ratio)


def _has_labeled_value(
    target_value: object,
    ocr_text: object,
    field_name: str | None = None,
) -> bool:
    """
    Match values embedded inside a single OCR box with their field label.

    Examples:
    - `N.W.:14kg`
    - `Refrigerant:R32`
    - `Rated Frequency 50Hz`

    This stays conservative by requiring label evidence and by rejecting
    multi-value concatenations like `220-240V~50Hz` when no field label exists.
    """
    target_raw = _normalize_raw_text(target_value)
    ocr_raw = _normalize_raw_text(ocr_text)
    if len(target_raw) < 2 or len(ocr_raw) < 2:
        return False

    idx = ocr_raw.find(target_raw)
    if idx < 0:
        return False

    prefix = ocr_raw[:idx]
    suffix = ocr_raw[idx + len(target_raw) :]
    label_text = re.sub(r"[^a-z]+", "", prefix)
    suffix_text = re.sub(r"[^a-z0-9]+", "", suffix)
    prefix_digits = len(re.findall(r"\d", prefix))

    aliases = FIELD_LABEL_ALIASES.get(field_name or "", ())
    if aliases and any(alias in label_text for alias in aliases):
        return suffix_text == ""

    if suffix_text != "":
        return False

    if idx == 0:
        return False

    # Generic fallback: allow clear label prefixes with no significant numeric noise.
    if len(label_text) >= 2 and prefix_digits <= 1:
        return True

    return False


def _match_score(target_value: object, ocr_text: object) -> float:
    """
    Return a conservative match score.

    Rules:
    1. Exact normalized match is the only guaranteed match.
    2. Fuzzy fallback is only allowed when the overall similarity is high.
    3. Numeric-heavy fields must keep highly similar digit signatures.
    """
    target_norm = normalize_text_for_match(target_value)
    ocr_norm = normalize_text_for_match(ocr_text)

    if len(target_norm) < 2 or len(ocr_norm) < 2:
        return 0.0

    if target_norm == ocr_norm:
        return 1.0

    text_ratio = SequenceMatcher(None, target_norm, ocr_norm).ratio()
    if text_ratio < 0.88:
        return 0.0

    target_digits = _digit_signature(target_norm)
    ocr_digits = _digit_signature(ocr_norm)
    if target_digits and ocr_digits:
        digit_ratio = SequenceMatcher(None, target_digits, ocr_digits).ratio()
        if digit_ratio < 0.9:
            return 0.0

    shorter_len = min(len(target_norm), len(ocr_norm))
    longer_len = max(len(target_norm), len(ocr_norm))
    coverage = shorter_len / longer_len
    if coverage < 0.75:
        return 0.0

    return text_ratio


def find_matching_ocr_boxes(
    target_value: object,
    ocr_boxes: Sequence[OCRBox],
    min_score: float = 0.88,
    field_name: str | None = None,
) -> List[int]:
    """Find OCR boxes that best correspond to a structured field value."""
    target_norm = normalize_text_for_match(target_value)
    if len(target_norm) < 2:
        return []

    exact_matches: List[int] = []
    scored_matches: List[Tuple[int, float]] = []

    for idx, box_info in enumerate(ocr_boxes):
        ocr_text = box_info[1] if len(box_info) > 1 else ""
        ocr_norm = normalize_text_for_match(ocr_text)
        if len(ocr_norm) < 2:
            continue

        if ocr_norm == target_norm:
            exact_matches.append(idx)
            continue

        score = _match_score(target_value, ocr_text)
        if field_name == "air_volume" and score < min_score:
            score = _air_volume_match_score(target_value, ocr_text)
        if score < min_score and _has_labeled_value(target_value, ocr_text, field_name):
            score = 0.9
        if score >= min_score:
            scored_matches.append((idx, score))

    if exact_matches:
        return exact_matches

    if not scored_matches:
        return []

    scored_matches.sort(key=lambda item: item[1], reverse=True)
    best_score = scored_matches[0][1]
    return [
        idx
        for idx, score in scored_matches
        if score >= max(min_score, best_score - 0.01)
    ]
