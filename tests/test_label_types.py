import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.extraction.text import (
    count_populated_fields,
    extract_compact_spec_from_text,
    extract_standard_spec_from_text,
    find_missing_fields,
    find_suspicious_fields,
    merge_compact_sources,
    merge_standard_sources,
    needs_compact_llm,
)
from label_detection.schema import (
    AirConditionerLabel,
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


class TestExtractStandardSpecFromText:
    def test_extracts_standard_fields_for_empty_llm_fallback(self):
        text = """
        GREE
        SPLIT AIR CONDITIONER INDOOR UNIT
        Model GWH12ACBXB-K3NNA1B/I
        Rated Voltage
        220-240V~
        Rated Frequency
        50Hz
        Heating Capacity W
        3400W
        Cooling Capacity W
        3250W
        Air Flow Volume m3/h
        590m3/h
        Weight kg
        8.5kg
        Sound Pressure Level(H) dB(A)
        37dB(A)
        Manufactured Date
        YYYY.MM
        GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI
        Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070
        600004078454
        """

        extracted = extract_standard_spec_from_text(text)

        assert extracted["brand"] == "GREE"
        assert extracted["product_type"] == "SPLIT AIR CONDITIONER INDOOR UNIT"
        assert extracted["model_number"] == "GWH12ACBXB-K3NNA1B/I"
        assert extracted["voltage"] == "220-240V~"
        assert extracted["frequency"] == "50Hz"
        assert extracted["heating_capacity"] == "3400W"
        assert extracted["cooling_capacity"] == "3250W"
        assert extracted["air_volume"] == "590m3/h"
        assert extracted["weight"] == "8.5kg"
        assert extracted["noise"] == "37dB(A)"
        assert extracted["mfg_date"] == "YYYY.MM"
        assert extracted["manufacturer"] == "GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI"
        assert extracted["address"] == "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070"
        assert extracted["barcode"] == "600004078454"

    def test_extracts_standard_anchors_without_correcting_defects(self):
        text = """
        SPLIT AIR CONDITIONER INDOOR UNIT
        Model GWH18AAD-K6DNA2E/I
        Rated Frequency
        50HHz
        Heating Capacity
        520kW
        """

        extracted = extract_standard_spec_from_text(text)

        assert extracted["model_number"] == "GWH18AAD-K6DNA2E/I"
        assert extracted["frequency"] == "50HHz"
        assert extracted["heating_capacity"] == "520kW"

    def test_extracts_incomplete_air_volume_unit_as_ocr_anchor(self):
        text = """
        Air
        Flow
        Volume
        850m/
        """

        extracted = extract_standard_spec_from_text(text)

        assert extracted["air_volume"] == "850m/"

    def test_merge_preserves_ocr_anchor_when_llm_autocorrects(self):
        rule_data = {
            "model_number": "GWH118AAD-K6DNA2E/I",
            "frequency": "50HHz",
            "heating_capacity": "520kW",
        }
        llm_data = {
            "brand": "GREE",
            "model_number": "GWH18AAD-K6DNA2E/I",
            "frequency": "50Hz",
            "heating_capacity": "5.20kW",
            "weight": "13.5kg",
        }

        merged = merge_standard_sources(
            rule_data,
            llm_data,
            AirConditionerLabel.model_fields.keys(),
        )

        assert merged["brand"] == "GREE"
        assert merged["model_number"] == "GWH118AAD-K6DNA2E/I"
        assert merged["frequency"] == "50HHz"
        assert merged["heating_capacity"] == "520kW"
        assert merged["weight"] == "13.5kg"

    def test_merge_prefers_llm_when_it_completes_truncated_ocr_anchor(self):
        rule_data = {"voltage": "220-240V"}
        llm_data = {"voltage": "220-240V~"}

        merged = merge_standard_sources(
            rule_data,
            llm_data,
            AirConditionerLabel.model_fields.keys(),
        )

        assert merged["voltage"] == "220-240V~"

    def test_merge_preserves_incomplete_air_volume_anchor(self):
        rule_data = {"air_volume": "850m/"}
        llm_data = {"air_volume": "850m³/h"}

        merged = merge_standard_sources(
            rule_data,
            llm_data,
            AirConditionerLabel.model_fields.keys(),
        )

        assert merged["air_volume"] == "850m/"

    def test_merge_uses_ocr_when_llm_returns_nulls(self):
        rule_data = {
            "brand": "GREE",
            "product_type": "SPLIT AIR CONDITIONER INDOOR UNIT",
            "model_number": "GWH12ACBXB-K3NNA1B/I",
            "voltage": "220-240V~",
            "frequency": "50Hz",
            "heating_capacity": "3400W",
            "cooling_capacity": "3250W",
            "air_volume": "590m3/h",
            "weight": "8.5kg",
            "noise": "37dB(A)",
            "mfg_date": "YYYY.MM",
            "manufacturer": "GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI",
            "address": "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070",
            "barcode": "600004078454",
        }
        llm_data = {field_name: None for field_name in AirConditionerLabel.model_fields}

        merged = merge_standard_sources(
            rule_data,
            llm_data,
            AirConditionerLabel.model_fields.keys(),
        )

        assert merged == rule_data
