import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.extraction.pdf import _select_page_crop_rect


def _make_text_block(bbox, text):
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"spans": [{"text": text}]}],
    }


class FakePage:
    def __init__(self, blocks, drawings, rect=(0, 0, 1000, 800)):
        self._blocks = blocks
        self._drawings = drawings
        self.rect = rect

    def get_text(self, mode, clip=None):
        assert mode == "dict"
        assert clip is None
        return {"blocks": self._blocks}

    def get_drawings(self):
        return list(self._drawings)


def test_page_crop_prefers_image_container_with_text_context():
    page = FakePage(
        blocks=[
            {"type": 1, "bbox": (660, 420, 820, 500)},
            _make_text_block((340, 300, 520, 330), "Model"),
            _make_text_block((340, 340, 520, 370), "Voltage"),
            _make_text_block((340, 380, 520, 410), "Date"),
        ],
        drawings=[
            {"rect": (0, 0, 1000, 800), "color": (0, 0, 0), "dashes": None},
            {"rect": (650, 410, 830, 510), "color": (0, 0, 0), "dashes": None},
            {"rect": (300, 260, 880, 560), "color": (0, 0, 0), "dashes": None},
        ],
    )

    rect, strategy = _select_page_crop_rect(page)

    assert strategy == "image_container"
    assert rect == (300.0, 260.0, 880.0, 560.0)


def test_page_crop_falls_back_to_red_dashed_union():
    page = FakePage(
        blocks=[
            {"type": 1, "bbox": (250, 210, 320, 260)},
        ],
        drawings=[
            {"rect": (100, 100, 500, 110), "color": (1.0, 0.0, 0.0), "dashes": "[3 2] 0"},
            {"rect": (100, 390, 500, 400), "color": (1.0, 0.0, 0.0), "dashes": "[3 2] 0"},
            {"rect": (100, 100, 110, 400), "color": (1.0, 0.0, 0.0), "dashes": "[3 2] 0"},
            {"rect": (490, 100, 500, 400), "color": (1.0, 0.0, 0.0), "dashes": "[3 2] 0"},
        ],
    )

    rect, strategy = _select_page_crop_rect(page)

    assert strategy == "red_dashed_union"
    assert rect == (100.0, 100.0, 500.0, 400.0)
