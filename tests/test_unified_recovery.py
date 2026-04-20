import os
import sys

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
