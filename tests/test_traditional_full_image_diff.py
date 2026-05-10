import numpy as np

from label_detection.workflows.traditional_full_image_diff import (
    merge_standard_and_micro_candidates_for_vlm,
    refine_display_boxes,
    scale_boxes,
)


def test_standard_candidate_absorbs_same_line_micro_candidate():
    mask = np.zeros((120, 320), dtype=np.uint8)
    mask[45:70, 80:180] = 255

    standard_boxes = [
        {
            "box": [100, 50, 140, 70],
            "review_box": [80, 35, 190, 90],
            "area": 80,
            "density": 0.1,
            "centroid": [120.0, 60.0],
            "child_count": 1,
        }
    ]
    micro_boxes = [
        {
            "box": [145, 55, 151, 65],
            "review_box": [90, 35, 190, 90],
            "area": 20,
            "density": 0.6,
            "centroid": [148.0, 60.0],
            "source": "micro_text_candidate_batch",
            "sub_candidates": [
                {
                    "box": [145, 55, 151, 65],
                    "area": 20,
                    "density": 0.6,
                    "centroid": [148.0, 60.0],
                    "source": "micro_text_candidate",
                    "candidate_id": 1,
                }
            ],
        },
        {
            "box": [260, 55, 266, 65],
            "review_box": [230, 35, 300, 90],
            "area": 18,
            "density": 0.6,
            "centroid": [263.0, 60.0],
            "source": "micro_text_candidate",
        },
    ]

    merged = merge_standard_and_micro_candidates_for_vlm(
        standard_boxes,
        micro_boxes,
        mask,
        mask.shape,
        padding=16,
        min_size=64,
        merge_gap=18,
    )

    assert len(merged) == 2
    absorbed = merged[0]
    assert absorbed["vlm_group_source"] == "standard_with_micro"
    assert absorbed["absorbed_micro_candidate_count"] == 1
    assert absorbed["box"] == [100, 50, 151, 70]


def test_refine_display_boxes_expands_and_merges_word_fragment_boxes():
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[42:62, 50:122] = 255

    refined = refine_display_boxes(
        [
            {
                "box": [70, 46, 76, 56],
                "area": 20,
                "density": 0.5,
                "centroid": [73.0, 51.0],
                "child_count": 2,
            },
            {
                "box": [95, 47, 101, 57],
                "area": 20,
                "density": 0.5,
                "centroid": [98.0, 52.0],
                "child_count": 2,
            },
        ],
        mask,
        mask.shape,
    )

    assert len(refined) == 1
    assert refined[0]["display_box"][0] <= 52
    assert refined[0]["display_box"][2] >= 121
    assert refined[0]["merged_final_box_count"] == 2


def test_scale_boxes_preserves_display_box():
    scaled = scale_boxes(
        [{"box": [10, 20, 30, 40], "display_box": [8, 18, 34, 44]}],
        scale=0.5,
        shape=(100, 100),
        box_keys=("box", "display_box"),
    )

    assert scaled[0]["box"] == [20, 40, 60, 80]
    assert scaled[0]["display_box"] == [16, 36, 68, 88]
