import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.extraction.text import (
    count_populated_fields,
    extract_compact_spec_from_text,
    find_missing_fields,
    find_suspicious_fields,
    merge_compact_sources,
    needs_compact_llm,
)
from label_detection.schema import (
    LABEL_KIND_COMPACT,
    LABEL_KIND_STANDARD,
    CompactSpecLabel,
    get_label_model,
    infer_label_kind,
)


class TestInferLabelKind:
    def test_detects_compact_spec_label(self):
        text = """
        GWH24AGD-K6DNA1C/I(WIFI)
        N.W.:14kg
        G.W.:16.5kg
        Color:White
        Connection Pipes:1/4"/1/2"
        Refrigerant:R32
        600001076226
        """
        assert infer_label_kind(text) == LABEL_KIND_COMPACT

    def test_falls_back_to_standard_label(self):
        text = """
        SPLIT AIR CONDITIONER INDOOR UNIT
        Rated Voltage 220-240V~
        Rated Frequency 50Hz
        Heating Capacity 5.20kW
        """
        assert infer_label_kind(text) == LABEL_KIND_STANDARD

    def test_returns_matching_model(self):
        assert get_label_model(LABEL_KIND_COMPACT) is CompactSpecLabel


class TestExtractCompactSpecFromText:
    def test_extracts_expected_fields(self):
        text = """
        GWH24AGD-K6DNA1C/I(WIFI)
        N.W.:14kg   G.W.:16.5kg   Color:White
        Connection Pipes:1/4"/1/2"
        Refrigerant:R32
        600001076226
        """

        extracted = extract_compact_spec_from_text(text)

        assert extracted["model_number"] == "GWH24AGD-K6DNA1C/I(WIFI)"
        assert extracted["net_weight"] == "14kg"
        assert extracted["gross_weight"] == "16.5kg"
        assert extracted["color"] == "White"
        assert extracted["connection_pipes"] == '1/4"/1/2"'
        assert extracted["refrigerant"] == "R32"
        assert extracted["barcode"] == "600001076226"
        assert count_populated_fields(extracted) == 7

    def test_missing_fields_trigger_llm_for_compact_label(self):
        extracted = {
            "model_number": "GWH24AGD-K6DNA1C/I(WIFI)",
            "net_weight": None,
            "gross_weight": "16kg",
            "color": "White",
            "connection_pipes": '1/5"/1/2"',
            "refrigerant": "R31",
            "barcode": "600001076226",
        }
        fields = CompactSpecLabel.model_fields.keys()

        assert find_missing_fields(extracted, fields) == ["net_weight"]
        assert find_suspicious_fields(extracted, fields) == [
            "connection_pipes",
            "refrigerant",
        ]
        assert needs_compact_llm(extracted, fields) is True

    def test_merge_prefers_llm_for_visual_short_fields(self):
        rule_data = {
            "model_number": "GWH24AGD-K6DNA1C/I(WIFI)",
            "net_weight": None,
            "gross_weight": "16kg",
            "color": "White",
            "connection_pipes": '1/5"/1/2"',
            "refrigerant": "R31",
            "barcode": "600001076226",
        }
        llm_data = {
            "model_number": "GWH24AGD-K6DNA1C/I(WIFI)",
            "net_weight": "14kg",
            "gross_weight": "16.5kg",
            "color": "White",
            "connection_pipes": '1/4"/1/2"',
            "refrigerant": "R32",
            "barcode": "600001076226",
        }

        merged = merge_compact_sources(rule_data, llm_data, CompactSpecLabel.model_fields.keys())

        assert merged["model_number"] == "GWH24AGD-K6DNA1C/I(WIFI)"
        assert merged["barcode"] == "600001076226"
        assert merged["net_weight"] == "14kg"
        assert merged["gross_weight"] == "16.5kg"
        assert merged["connection_pipes"] == '1/4"/1/2"'
        assert merged["refrigerant"] == "R32"
