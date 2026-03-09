import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.preprocessing.border import crop_to_border, find_template_crop_rect


def _draw_barcode(image, x_start, y1, y2, count=18, spacing=10):
    for idx in range(count):
        x = x_start + idx * spacing
        thickness = 2 if idx % 3 else 4
        cv2.line(image, (x, y1), (x, y2), (0, 0, 0), thickness)


def _make_borderless_template():
    image = np.full((320, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (980, 110), (20, 20, 20), -1)
    cv2.rectangle(image, (40, 140), (260, 182), (0, 0, 0), -1)
    cv2.rectangle(image, (320, 140), (560, 182), (0, 0, 0), -1)
    cv2.rectangle(image, (620, 140), (860, 182), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 205), (610, 238), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 260), (420, 294), (0, 0, 0), -1)
    _draw_barcode(image, 730, 200, 288)
    return image


def _make_bordered_template():
    image = np.full((320, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (976, 296), (0, 0, 0), 8)
    cv2.rectangle(image, (55, 42), (945, 96), (0, 0, 0), -1)
    cv2.rectangle(image, (65, 130), (285, 168), (0, 0, 0), -1)
    cv2.rectangle(image, (350, 130), (595, 168), (0, 0, 0), -1)
    cv2.rectangle(image, (640, 130), (900, 168), (0, 0, 0), -1)
    cv2.rectangle(image, (65, 195), (640, 228), (0, 0, 0), -1)
    cv2.rectangle(image, (65, 250), (450, 282), (0, 0, 0), -1)
    _draw_barcode(image, 730, 190, 276)
    return image


class TestFindTemplateCropRect:
    def test_prefers_content_strategy_for_borderless_label(self):
        image = _make_borderless_template()

        candidate = find_template_crop_rect(image)

        assert candidate is not None
        assert candidate["strategy"] == "content"
        x, y, w, h = candidate["rect"]
        assert y <= 30
        assert h >= 240
        assert x <= 40
        assert x + w >= 940

        cropped = crop_to_border(image, candidate["rect"], padding=-6)
        assert cropped.shape[0] >= 250

    def test_prefers_border_strategy_when_outer_frame_exists(self):
        image = _make_bordered_template()

        candidate = find_template_crop_rect(image)

        assert candidate is not None
        assert candidate["strategy"] == "border"
        x, y, w, h = candidate["rect"]
        assert x <= 30
        assert y <= 30
        assert x + w >= 970
        assert y + h >= 290

        cropped = crop_to_border(image, candidate["rect"], padding=3)
        assert cropped.shape[0] >= 250
