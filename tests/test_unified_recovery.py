import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.workflows import unified


class TestUnresolvedRecoveryResults:
    def test_unmatched_target_with_failed_inference_creates_review_result(self, monkeypatch):
        template_regions = [
            {"coordinate": [10, 10, 40, 40], "label": "image", "score": 0.9},
        ]
        target_regions = [
            {"coordinate": [60, 60, 100, 100], "label": "image", "score": 0.9},
        ]
        template_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        target_image = np.full((160, 160, 3), 255, dtype=np.uint8)

        monkeypatch.setattr(unified, "infer_corresponding_region", lambda *args, **kwargs: None)

        payload = unified._recover_unmatched_regions(
            template_regions=template_regions,
            target_regions=target_regions,
            matched_pairs=[],
            unmatched1=[],
            unmatched2=[0],
            template_image=template_image,
            target_image=target_image,
            output_dir=".",
            use_vlm=False,
        )

        assert payload["resolved_unmatched2"] == []
        assert len(payload["recovery_results"]) == 1

        result = payload["recovery_results"][0]
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True
        assert result["unresolved_unmatched"] is True
        assert result["error_type"] == "recovery_inference_failed"
        assert result["target_idx"] == 0
        assert result["template_idx"] is None
        assert result["inferred_template_box"] is None

    def test_recovered_uncovered_target_with_failed_inference_becomes_mismatch(self, monkeypatch):
        template_regions = [
            {"coordinate": [10, 10, 40, 40], "label": "image", "score": 0.9},
        ]
        target_regions = [
            {
                "coordinate": [60, 60, 100, 100],
                "label": "image",
                "score": 0.0,
                "recovered_uncovered_graphic": True,
            },
        ]
        template_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        target_image = np.full((160, 160, 3), 255, dtype=np.uint8)

        monkeypatch.setattr(unified, "infer_corresponding_region", lambda *args, **kwargs: None)

        payload = unified._recover_unmatched_regions(
            template_regions=template_regions,
            target_regions=target_regions,
            matched_pairs=[],
            unmatched1=[],
            unmatched2=[0],
            template_image=template_image,
            target_image=target_image,
            output_dir=".",
            use_vlm=False,
        )

        assert payload["resolved_unmatched2"] == [0]
        assert len(payload["recovery_results"]) == 1

        result = payload["recovery_results"][0]
        assert result["decision"] == "mismatch"
        assert result["needs_review"] is False
        assert result["unresolved_unmatched"] is False
        assert result["error_type"] == "recovery_extra_target_graphic"
        assert result["judgment_source"] == "recovery_uncovered_target_graphic"
        assert result["target_idx"] == 0
        assert result["template_idx"] is None
        assert result["inferred_template_box"] is None

    def test_unmatched_template_with_low_foreground_becomes_mismatch(self, monkeypatch):
        template_regions = [
            {"coordinate": [15, 15, 55, 55], "label": "image", "score": 0.9},
        ]
        target_regions = []
        template_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        target_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        inferred_target = {
            "coordinate": [70, 70, 110, 110],
            "label": "image",
            "score": 0.0,
            "inferred": True,
            "inferred_from": "template",
        }

        monkeypatch.setattr(unified, "infer_corresponding_region", lambda *args, **kwargs: inferred_target)

        payload = unified._recover_unmatched_regions(
            template_regions=template_regions,
            target_regions=target_regions,
            matched_pairs=[],
            unmatched1=[0],
            unmatched2=[],
            template_image=template_image,
            target_image=target_image,
            output_dir=".",
            use_vlm=False,
        )

        assert payload["resolved_unmatched1"] == [0]
        assert len(payload["recovery_results"]) == 1

        result = payload["recovery_results"][0]
        assert result["decision"] == "mismatch"
        assert result["needs_review"] is False
        assert result["unresolved_unmatched"] is False
        assert result["error_type"] == "recovery_missing_target_graphic"
        assert result["template_idx"] == 0
        assert result["target_idx"] is None
        assert result["inferred_target_box"] == [70, 70, 110, 110]

    def test_unmatched_target_with_low_foreground_becomes_extra_graphic_mismatch(self, monkeypatch):
        template_regions = []
        target_regions = [
            {"coordinate": [60, 60, 100, 100], "label": "image", "score": 0.9},
        ]
        template_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        target_image = np.full((160, 160, 3), 255, dtype=np.uint8)
        inferred_template = {
            "coordinate": [20, 20, 50, 50],
            "label": "image",
            "score": 0.0,
            "inferred": True,
            "inferred_from": "target",
        }

        monkeypatch.setattr(unified, "infer_corresponding_region", lambda *args, **kwargs: inferred_template)

        payload = unified._recover_unmatched_regions(
            template_regions=template_regions,
            target_regions=target_regions,
            matched_pairs=[],
            unmatched1=[],
            unmatched2=[0],
            template_image=template_image,
            target_image=target_image,
            output_dir=".",
            use_vlm=False,
        )

        assert payload["resolved_unmatched2"] == [0]
        result = payload["recovery_results"][0]
        assert result["decision"] == "mismatch"
        assert result["needs_review"] is False
        assert result["unresolved_unmatched"] is False
        assert result["error_type"] == "recovery_extra_target_graphic"
        assert result["template_idx"] is None
        assert result["target_idx"] == 0
        assert result["inferred_template_box"] == [20, 20, 50, 50]


class TestFinalVerdict:
    def test_text_difference_count_uses_actual_field_delta(self):
        verdict = unified.build_final_verdict(
            match_count=13,
            total_fields=14,
            graphic_pass=True,
            review_count=0,
            mismatch_count=0,
            unresolved_graphics=0,
        )

        assert verdict == "⚠️ 文字存在差异 (1 处)，图形一致"

    def test_text_difference_with_graphic_review_mentions_review(self):
        verdict = unified.build_final_verdict(
            match_count=13,
            total_fields=14,
            graphic_pass=True,
            review_count=1,
            mismatch_count=0,
            unresolved_graphics=0,
        )

        assert verdict == "⚠️ 文字存在差异 (1 处)，另有 1 处图形需人工复核"


class TestTextVisualizationGuards:
    def test_label_text_match_allows_ocr_word_reordering(self):
        assert unified._ocr_text_matches_expected(
            "label:noise",
            "Sound Pressure Level(H)",
            "Pressure Level(H) 46dB(A) Sound",
        )

    def test_label_text_match_keeps_keyword_typo_as_difference(self):
        assert not unified._ocr_text_matches_expected(
            "label:air_volume",
            "Air Flow Volume",
            "Air Klow Volume",
        )

    def test_aligned_address_visual_scan_skips_when_structured_value_matches(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()

        def fail_if_address_scan_runs(*args, **kwargs):
            raise AssertionError("address visual scan should be skipped")

        monkeypatch.setattr(unified, "find_matching_ocr_boxes", fail_if_address_scan_runs)

        count = unified._add_aligned_value_visual_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[],
            structured_fields=["address"],
            template_data={
                "address": "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070",
            },
            target_data={
                "address": "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070",
            },
        )

        assert count == 0

    def test_value_anchor_address_scan_skips_when_structured_value_matches(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()

        def fail_if_address_scan_runs(*args, **kwargs):
            raise AssertionError("address value-anchor scan should be skipped")

        monkeypatch.setattr(unified, "find_matching_ocr_boxes", fail_if_address_scan_runs)

        count = unified._add_value_anchor_text_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[],
            target_boxes=[],
            structured_fields=["address"],
            template_data={
                "address": "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070",
            },
            target_data={
                "address": "Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070",
            },
        )

        assert count == 0

    def test_value_anchor_scan_skips_other_matching_structured_fields(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()

        def fail_if_weight_scan_runs(*args, **kwargs):
            raise AssertionError("matching structured field should be skipped")

        monkeypatch.setattr(unified, "find_matching_ocr_boxes", fail_if_weight_scan_runs)

        count = unified._add_value_anchor_text_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[],
            target_boxes=[],
            structured_fields=["weight"],
            template_data={"weight": "55kg"},
            target_data={"weight": "55kg"},
        )

        assert count == 0

    def test_aligned_visual_scan_skips_other_matching_structured_fields(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()

        def fail_if_weight_scan_runs(*args, **kwargs):
            raise AssertionError("matching structured field should be skipped")

        monkeypatch.setattr(unified, "find_matching_ocr_boxes", fail_if_weight_scan_runs)

        count = unified._add_aligned_value_visual_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[],
            structured_fields=["weight"],
            template_data={"weight": "55kg"},
            target_data={"weight": "55kg"},
        )

        assert count == 0

    def test_aligned_visual_scan_draws_matching_value_with_strong_local_diff(self):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()
        cv2.putText(
            template_image,
            "5.20kW",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            target_image,
            "5.20kWW",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        visualization_annotations = []

        count = unified._add_aligned_value_visual_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=visualization_annotations,
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[
                (
                    np.array(
                        [[18, 20], [120, 20], [120, 55], [18, 55]],
                        dtype=np.float32,
                    ),
                    "5.20kW",
                    0.98,
                )
            ],
            structured_fields=["heating_capacity"],
            template_data={"heating_capacity": "5.20kW"},
            target_data={"heating_capacity": "5.20kW"},
        )

        assert count == 1
        assert visualization_annotations[0]["field"] == "heating_capacity"
        assert visualization_annotations[0]["source"] == "aligned_matching_value_visual"

    def test_missing_label_anchor_fallback_skips_low_local_diff(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()
        template_boxes = [
            (
                np.array([[20, 20], [120, 20], [120, 45], [20, 45]], dtype=np.float32),
                "Sound Pressure Level(H)",
                0.98,
            )
        ]
        suppressed_annotations = []

        monkeypatch.delenv("ENABLE_MISSING_LABEL_MAPPED_FALLBACK", raising=False)

        count = unified._add_label_anchor_text_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=suppressed_annotations,
            template_image=template_image,
            target_image=target_image,
            template_boxes=template_boxes,
            target_boxes=[],
            structured_fields=["noise"],
            template_label_hits={
                "noise": {"text": "Sound Pressure Level(H)", "box_indices": [0]},
            },
            target_label_hits={},
        )

        assert count == 0
        assert suppressed_annotations == []

    def test_missing_label_anchor_skips_matching_overlap_text(self):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        target_image = template_image.copy()
        box = np.array([[20, 20], [120, 20], [120, 45], [20, 45]], dtype=np.float32)

        count = unified._add_label_anchor_text_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=[],
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=[(box, "Sound Pressure Level(H)", 0.98)],
            target_boxes=[(box, "Pressure Level(H) 46dB(A) Sound", 0.98)],
            structured_fields=["noise"],
            template_label_hits={
                "noise": {"text": "Sound Pressure Level(H)", "box_indices": [0]},
            },
            target_label_hits={},
        )

        assert count == 0

    def test_missing_label_anchor_fallback_draws_when_local_diff_is_visible(self, monkeypatch):
        template_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        template_image[20:45, 20:120] = 0
        target_image = np.full((120, 240, 3), 255, dtype=np.uint8)
        visualization_annotations = []
        template_boxes = [
            (
                np.array([[20, 20], [120, 20], [120, 45], [20, 45]], dtype=np.float32),
                "Sound Pressure Level(H)",
                0.98,
            )
        ]

        monkeypatch.delenv("ENABLE_MISSING_LABEL_MAPPED_FALLBACK", raising=False)

        count = unified._add_label_anchor_text_diffs(
            vis_image=target_image.copy(),
            visualization_annotations=visualization_annotations,
            suppressed_annotations=[],
            template_image=template_image,
            target_image=target_image,
            template_boxes=template_boxes,
            target_boxes=[],
            structured_fields=["noise"],
            template_label_hits={
                "noise": {"text": "Sound Pressure Level(H)", "box_indices": [0]},
            },
            target_label_hits={},
        )

        assert count == 1
        assert visualization_annotations[0]["source"] == "label_anchor_mapped"
