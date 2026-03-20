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

    def test_unmatched_template_with_low_foreground_keeps_inferred_target_box(self, monkeypatch):
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

        assert payload["resolved_unmatched1"] == []
        assert len(payload["recovery_results"]) == 1

        result = payload["recovery_results"][0]
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True
        assert result["unresolved_unmatched"] is True
        assert result["error_type"] == "recovery_low_foreground"
        assert result["template_idx"] == 0
        assert result["target_idx"] is None
        assert result["inferred_target_box"] == [70, 70, 110, 110]
