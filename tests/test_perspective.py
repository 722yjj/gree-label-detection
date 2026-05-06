import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.preprocessing.perspective import _score_quad


def test_outline_candidate_allows_low_fill_ratio():
    image_shape = (1200, 1800, 3)
    template_ratio = 1017 / 662
    quad = np.array(
        [
            [240, 180],
            [1540, 180],
            [1540, 1025],
            [240, 1025],
        ],
        dtype=np.float32,
    )
    contour_area = cv2.contourArea(quad) * 0.12

    assert _score_quad(
        quad,
        contour_area,
        image_shape,
        template_ratio,
        source_kind="content",
    ) is None

    candidate = _score_quad(
        quad,
        contour_area,
        image_shape,
        template_ratio,
        source_kind="outline",
    )

    assert candidate is not None
    assert candidate["fill_ratio"] == pytest.approx(0.12)
    assert candidate["aspect_ratio"] == pytest.approx(template_ratio, rel=0.01)


def test_touching_content_blob_is_rejected_when_aspect_drifts_from_template():
    image_shape = (2048, 3072, 3)
    template_ratio = 1017 / 662
    touching_neighbor_blob = np.array(
        [
            [0, 158],
            [2612.1, 160.2],
            [2610.9, 1487.3],
            [0, 1485],
        ],
        dtype=np.float32,
    )
    true_label_outline = np.array(
        [
            [568.7, 171.0],
            [2623.3, 178.1],
            [2618.7, 1493.8],
            [564.2, 1486.8],
        ],
        dtype=np.float32,
    )

    assert _score_quad(
        touching_neighbor_blob,
        3_061_047,
        image_shape,
        template_ratio,
        source_kind="content",
    ) is None

    candidate = _score_quad(
        true_label_outline,
        321_624,
        image_shape,
        template_ratio,
        source_kind="outline",
    )

    assert candidate is not None
    assert candidate["touch_count"] == 0
    assert candidate["aspect_ratio"] == pytest.approx(template_ratio, rel=0.03)
