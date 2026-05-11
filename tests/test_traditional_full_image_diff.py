import numpy as np

import label_detection.workflows.traditional_full_image_diff as traditional_diff
from label_detection.workflows.traditional_full_image_diff import (
    apply_vlm_filter,
    build_vlm_batch_filter_prompt,
    build_vlm_filter_prompt,
    merge_standard_and_micro_candidates_for_vlm,
    refine_display_box,
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


def test_refine_display_box_leaves_plain_graphic_box_unchanged():
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[45:75, 40:180] = 255

    refined = refine_display_box(
        {
            "box": [80, 50, 110, 70],
            "area": 120,
            "density": 0.2,
            "centroid": [95.0, 60.0],
            "child_count": 1,
        },
        mask,
        mask.shape,
    )

    assert refined == [80, 50, 110, 70]


def test_refine_display_box_expands_standard_text_line_review_candidate():
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[45:65, 50:150] = 255

    refined = refine_display_box(
        {
            "box": [95, 48, 108, 62],
            "area": 120,
            "density": 0.6,
            "centroid": [101.5, 55.0],
            "child_count": 1,
            "review_box_source": "text_line",
        },
        mask,
        mask.shape,
    )

    assert refined[0] <= 52
    assert refined[2] >= 149


def test_refine_display_box_uses_pdf_text_region_gate_for_graphics():
    mask = np.zeros((140, 260), dtype=np.uint8)
    mask[45:65, 40:180] = 255
    text_regions = [{"text": "Model", "box": [10, 10, 90, 30]}]

    refined = refine_display_box(
        {
            "box": [80, 50, 110, 70],
            "area": 120,
            "density": 0.2,
            "centroid": [95.0, 60.0],
            "child_count": 2,
            "review_box_source": "text_line",
        },
        mask,
        mask.shape,
        text_regions=text_regions,
    )

    assert refined == [80, 50, 110, 70]


def test_refine_display_box_snaps_to_matching_pdf_text_line():
    mask = np.zeros((140, 260), dtype=np.uint8)
    mask[45:65, 40:180] = 255
    text_regions = [{"text": "Rated Voltage", "box": [38, 42, 182, 68]}]

    refined = refine_display_box(
        {
            "box": [94, 49, 108, 61],
            "area": 120,
            "density": 0.6,
            "centroid": [101.0, 55.0],
            "child_count": 1,
            "review_box_source": "text_line",
        },
        mask,
        mask.shape,
        text_regions=text_regions,
    )

    assert refined[0] <= 42
    assert refined[2] >= 178


def test_scale_boxes_preserves_display_box():
    scaled = scale_boxes(
        [{"box": [10, 20, 30, 40], "display_box": [8, 18, 34, 44]}],
        scale=0.5,
        shape=(100, 100),
        box_keys=("box", "display_box"),
    )

    assert scaled[0]["box"] == [20, 40, 60, 80]
    assert scaled[0]["display_box"] == [16, 36, 68, 88]


def test_vlm_prompts_use_two_unannotated_images():
    single_prompt = build_vlm_filter_prompt(
        [10, 20, 30, 40],
        [0, 0, 80, 80],
        0.1,
        0.2,
        1,
    )
    batch_prompt = build_vlm_batch_filter_prompt(
        [0, 0, 80, 80],
        [{"box": [10, 20, 30, 40], "candidate_id": 1}],
        0.1,
        0.2,
    )

    assert "two unannotated cropped images" in single_prompt
    assert "two unannotated cropped images" in batch_prompt
    assert "red numbered boxes" not in batch_prompt
    assert "Image 3" not in single_prompt
    assert "Image 3" not in batch_prompt


def test_apply_vlm_filter_sends_only_unannotated_template_and_target(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        content = '{"decision":"discard","confidence":0.9,"reason":"same content"}'

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, messages, **kwargs):
            calls.append(messages[0])
            return FakeResponse()

    monkeypatch.setattr(traditional_diff, "OpenAICompatibleHTTPClient", FakeClient)
    image = np.full((80, 120, 3), 255, dtype=np.uint8)

    kept, decisions = apply_vlm_filter(
        [
            {
                "box": [20, 20, 35, 35],
                "review_box": [0, 0, 80, 80],
                "area": 20,
                "density": 0.5,
                "source": "micro_text_candidate_batch",
                "sub_candidates": [
                    {
                        "box": [20, 20, 35, 35],
                        "candidate_id": 1,
                        "area": 20,
                        "density": 0.5,
                    }
                ],
            }
        ],
        image,
        image,
        model="fake-model",
        api_base="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        timeout=1,
        max_tokens=32,
        crop_padding=8,
        review_min_size=32,
        keep_unknown=False,
        skip_model_check=True,
        debug_dir=tmp_path,
    )

    assert kept == []
    assert decisions[0]["vlm_input_mode"] == "two_image_unmarked_region"
    assert len(calls) == 1
    assert len(calls[0]["images"]) == 2
    assert (tmp_path / "candidate_01_template.jpg").exists()
    assert (tmp_path / "candidate_01_target.jpg").exists()
    assert not (tmp_path / "candidate_01_target_marked.jpg").exists()
