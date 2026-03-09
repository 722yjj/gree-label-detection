import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.matching.ocr import (
    find_matching_ocr_boxes,
    normalize_text_for_match,
)


class TestNormalizeTextForMatch:
    def test_normalizes_symbols_and_case(self):
        assert normalize_text_for_match("590m³/h") == "590m3h"
        assert normalize_text_for_match(" 2025.12 ") == "202512"
        assert normalize_text_for_match("50HZ") == "50hz"


class TestFindMatchingOcrBoxes:
    def _make_box(self, text):
        return ([[0, 0], [10, 0], [10, 10], [0, 10]], text, 0.99)

    def test_prefers_exact_value_over_partial_fragments(self):
        boxes = [
            self._make_box("50"),
            self._make_box("Hz"),
            self._make_box("220-240V~"),
            self._make_box("50Hz"),
        ]

        assert find_matching_ocr_boxes("50Hz", boxes) == [3]

    def test_prefers_exact_value_over_longer_combined_text(self):
        boxes = [
            self._make_box("37dB(A)2025.12"),
            self._make_box("2025.12"),
        ]

        assert find_matching_ocr_boxes("2025.12", boxes) == [1]

    def test_accepts_normalized_equivalent_text(self):
        boxes = [
            self._make_box("590m3/h"),
            self._make_box("590"),
        ]

        assert find_matching_ocr_boxes("590m³/h", boxes) == [0]

    def test_rejects_unrelated_high_overlap_noise(self):
        boxes = [
            self._make_box("220240v50hz"),
            self._make_box("3400W"),
        ]

        assert find_matching_ocr_boxes("50Hz", boxes) == []

    def test_matches_labeled_weight_value_in_same_box(self):
        boxes = [
            self._make_box("N.W.:14kg"),
            self._make_box("G.W.:16.5kg"),
        ]

        assert find_matching_ocr_boxes("14kg", boxes, field_name="net_weight") == [0]
        assert find_matching_ocr_boxes("16.5kg", boxes, field_name="gross_weight") == [1]

    def test_matches_labeled_refrigerant_value_in_same_box(self):
        boxes = [
            self._make_box("Refrigerant:R32"),
            self._make_box("Color:White"),
        ]

        assert find_matching_ocr_boxes("R32", boxes, field_name="refrigerant") == [0]

    def test_still_rejects_frequency_embedded_in_other_value_without_label(self):
        boxes = [
            self._make_box("220-240V~50Hz"),
            self._make_box("Rated Voltage 220-240V~"),
        ]

        assert find_matching_ocr_boxes("50Hz", boxes, field_name="frequency") == []
