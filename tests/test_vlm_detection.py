"""VLM object detection parsing tests."""

import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.services.vlm_detection import VLMObjectDetector


class TestVLMObjectDetectionParsing:
    @pytest.fixture
    def detector(self):
        return object.__new__(VLMObjectDetector)

    def test_parse_bbox_1000_response(self, detector):
        response = (
            '{"objects":[{"label":"barcode","confidence":0.92,'
            '"bbox_1000":[100,200,600,800]}],"summary":"found barcode"}'
        )
        result = detector._parse_response(response, (1000, 2000), "barcode")

        assert result["parse_error"] is False
        assert result["summary"] == "found barcode"
        assert len(result["objects"]) == 1
        assert result["objects"][0]["bbox_px"] == [200, 200, 1200, 800]
        assert result["objects"][0]["coord_mode"] == "norm_1000"

    def test_parse_codeblock_with_pixel_box(self, detector):
        response = """```json
{"detections":[{"name":"logo","score":0.81,"box":{"x1":50,"y1":60,"x2":150,"y2":200}}],"summary":"ok"}
```"""
        result = detector._parse_response(response, (400, 300), "logo")

        assert result["parse_error"] is False
        assert len(result["objects"]) == 1
        assert result["objects"][0]["label"] == "logo"
        assert result["objects"][0]["bbox_px"] == [50, 60, 150, 200]
        assert result["objects"][0]["coord_mode"] == "pixel"

    def test_parse_ratio_box(self, detector):
        response = (
            '{"objects":[{"label":"qr","confidence":0.77,'
            '"bbox":[0.1,0.2,0.4,0.5]}],"summary":"ratio"}'
        )
        result = detector._parse_response(response, (1000, 500), "qr")

        assert result["parse_error"] is False
        assert result["objects"][0]["bbox_px"] == [50, 200, 200, 500]
        assert result["objects"][0]["coord_mode"] == "ratio"

    def test_parse_regions_key(self, detector):
        response = (
            '{"regions":[{"label":"image","confidence":0.66,'
            '"bbox_1000":[200,200,400,400]}],"summary":"regions"}'
        )
        result = detector._parse_response(response, (500, 500), "image")

        assert result["parse_error"] is False
        assert len(result["objects"]) == 1
        assert result["objects"][0]["bbox_px"] == [100, 100, 200, 200]

    def test_parse_failure(self, detector):
        result = detector._parse_response("not a json response", (100, 100), "barcode")

        assert result["parse_error"] is True
        assert result["objects"] == []
