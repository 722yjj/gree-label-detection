"""
OCR text box matching helpers.

Keep this module dependency-light so it can be unit tested without loading
the OCR engine or LLM stack.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Sequence, Tuple


OCRBox = Tuple[object, str, float]


def normalize_text_for_match(text: object) -> str:
    """Normalize OCR / extracted text before matching."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("³", "3")
    return re.sub(r"[^0-9a-z]+", "", normalized)


def _digit_signature(text: str) -> str:
    return re.sub(r"\D+", "", text)


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
