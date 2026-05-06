import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.matching.ocr import (
    extract_field_labels_from_ocr_boxes,
    field_values_match,
    find_matching_ocr_boxes,
    normalize_label_text_for_compare,
    normalize_text_for_compare,
    normalize_text_for_match,
    text_field_values_match,
)


class TestNormalizeTextForMatch:
    def test_normalizes_symbols_and_case(self):
        assert normalize_text_for_match("590m³/h") == "590m3h"
        assert normalize_text_for_match(" 2025.12 ") == "202512"
        assert normalize_text_for_match("50HZ") == "50hz"


class TestNormalizeTextForCompare:
    def test_ignores_spacing_noise_but_keeps_significant_punctuation(self):
        assert (
            normalize_text_for_compare("GREE ELECTRIC APPLIANCES, INC. OF ZHUHAI")
            == "gree electric appliances,inc.of zhuhai"
        )
        assert normalize_text_for_compare("13.5 kg") == "13.5kg"


class TestNormalizeLabelTextForCompare:
    def test_ignores_spacing_and_punctuation_but_keeps_case(self):
        assert normalize_label_text_for_compare("Serial No.") == "SerialNo"
        assert normalize_label_text_for_compare("Air Flow Volume") == "AirFlowVolume"


class TestFieldValuesMatch:
    def test_treats_punctuation_spacing_difference_as_match(self):
        assert field_values_match(
            "GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI",
            "GREE ELECTRIC APPLIANCES, INC. OF ZHUHAI",
        )

    def test_keeps_decimal_difference_significant(self):
        assert not field_values_match("13.5kg", "135kg")


class TestTextFieldValuesMatch:
    def test_treats_label_case_change_as_difference(self):
        assert not text_field_values_match("label:weight", "Weight", "weiGht")

    def test_treats_short_ocr_label_aliases_as_match(self):
        assert text_field_values_match(
            "label:voltage",
            "Rated Voltage",
            "Voltage",
        )
        assert text_field_values_match(
            "label:frequency",
            "Rated Frequency",
            "Frequency",
        )
        assert text_field_values_match(
            "label:mfg_date",
            "Manufactured Date",
            "Date",
        )

    def test_treats_split_and_reordered_label_tokens_as_match(self):
        assert text_field_values_match(
            "label:heating_capacity",
            "Heatin g Capacity",
            "Capacity Heating",
        )
        assert text_field_values_match(
            "label:air_volume",
            "Air Flow Volum e",
            "Volume Air Flow",
        )

    def test_treats_single_letter_label_ocr_join_as_match(self):
        assert text_field_values_match(
            "label:noise",
            "Sound Pressure Level(H)",
            "Sound PressureL Level(H)",
        )

    def test_rejects_noisy_label_alias_prefix(self):
        assert not text_field_values_match(
            "label:voltage",
            "Rated Voltage",
            "dVoltage",
        )


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


class TestExtractFieldLabelsFromOcrBoxes:
    def _make_box(self, text, score=0.99, x1=0, y1=0, x2=10, y2=10):
        return ([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, score)

    def test_prefers_label_only_box_and_extracts_combined_label_prefix(self):
        boxes = [
            self._make_box("Weight"),
            self._make_box("Weight 13.5kg", score=0.95),
            self._make_box("Rated Frequency 50Hz"),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["weight", "frequency"])

        assert labels == {
            "weight": {"text": "Weight", "box_indices": [0]},
            "frequency": {"text": "Rated Frequency", "box_indices": [2]},
        }

    def test_accepts_fuzzy_weight_label_with_repeated_leading_character(self):
        boxes = [
            self._make_box("WWeight"),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["weight"])

        assert labels == {
            "weight": {"text": "WWeight", "box_indices": [0]},
        }

    def test_merges_adjacent_boxes_for_manufactured_date_label(self):
        boxes = [
            self._make_box("Manufactured", x1=0, y1=100, x2=120, y2=130),
            self._make_box("Date", x1=124, y1=100, x2=170, y2=130),
            self._make_box("2026.01", x1=250, y1=100, x2=320, y2=130),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["mfg_date"])

        assert labels == {
            "mfg_date": {"text": "Manufactured Date", "box_indices": [0, 1]},
        }

    def test_merges_label_words_by_x_when_value_sorts_between_them_by_y(self):
        boxes = [
            self._make_box("Heating", x1=841, y1=280, x2=1057, y2=375),
            self._make_box("3400W", x1=1414, y1=290, x2=1604, y2=370),
            self._make_box("Capacity", x1=1051, y1=286, x2=1289, y2=379),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["heating_capacity"])

        assert labels == {
            "heating_capacity": {"text": "Heating Capacity", "box_indices": [0, 2]},
        }

    def test_merges_three_word_air_volume_label(self):
        boxes = [
            self._make_box("50Hz", x1=683, y1=376, x2=823, y2=449),
            self._make_box("Air", x1=843, y1=377, x2=939, y2=449),
            self._make_box("Flow", x1=933, y1=374, x2=1076, y2=452),
            self._make_box("Volume", x1=1072, y1=377, x2=1268, y2=451),
            self._make_box("590m/h", x1=1390, y1=364, x2=1606, y2=452),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["air_volume"])

        assert labels == {
            "air_volume": {"text": "Air Flow Volume", "box_indices": [1, 2, 3]},
        }

    def test_merges_three_word_noise_label(self):
        boxes = [
            self._make_box("Sound", x1=26, y1=575, x2=174, y2=643),
            self._make_box("Pressure", x1=184, y1=576, x2=366, y2=646),
            self._make_box("Level(H)", x1=376, y1=569, x2=548, y2=657),
            self._make_box("37dB(A)", x1=700, y1=575, x2=900, y2=646),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["noise"])

        assert labels == {
            "noise": {"text": "Sound Pressure Level(H)", "box_indices": [0, 1, 2]},
        }

    def test_merges_noise_label_with_overlapping_ocr_tail(self):
        boxes = [
            self._make_box("Sound", x1=65, y1=730, x2=260, y2=813),
            self._make_box("PressureL", x1=260, y1=728, x2=549, y2=819),
            self._make_box("Level(H)", x1=499, y1=728, x2=756, y2=829),
            self._make_box("37dB(A)", x1=925, y1=722, x2=1120, y2=837),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["noise"])

        assert labels == {
            "noise": {"text": "Sound PressureL Level(H)", "box_indices": [0, 1, 2]},
        }

    def test_supports_numpy_coordinate_arrays_from_real_ocr(self):
        boxes = [
            (np.array([[0, 0], [50, 0], [50, 20], [0, 20]], dtype=np.int16), "WWeight", 0.99),
        ]

        labels = extract_field_labels_from_ocr_boxes(boxes, ["weight"])

        assert labels == {
            "weight": {"text": "WWeight", "box_indices": [0]},
        }
