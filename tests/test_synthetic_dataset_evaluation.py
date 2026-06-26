import json
from pathlib import Path

import cv2
import numpy as np

from scripts import evaluate_synthetic_dataset as evaluator


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_resolve_template_path_prefers_cropped_pdf_shape(tmp_path):
    dataset_root = tmp_path / "dataset"
    template_dir = dataset_root / "template"
    template_dir.mkdir(parents=True)
    pdf_path = template_dir / "template.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    _write_image(template_dir / "template.png", np.full((8, 8, 3), 255, dtype=np.uint8))

    assert evaluator.resolve_template_path(dataset_root, {}) == pdf_path


def test_resolve_template_preview_for_pdf_uses_same_folder_png(tmp_path):
    dataset_root = tmp_path / "dataset"
    template_dir = dataset_root / "template"
    template_dir.mkdir(parents=True)
    pdf_path = template_dir / "template.pdf"
    preview_path = template_dir / "template.png"
    pdf_path.write_bytes(b"%PDF-1.4")
    _write_image(preview_path, np.full((8, 8, 3), 255, dtype=np.uint8))

    assert evaluator.resolve_template_preview_path(dataset_root, pdf_path, {}) == preview_path


def test_load_cases_uses_sample_specific_template_paths(tmp_path):
    dataset_root = tmp_path / "dataset"
    template_root = dataset_root / "template"
    template_root.mkdir(parents=True)
    template_a = template_root / "template_a.pdf"
    template_b = template_root / "template_b.pdf"
    template_a.write_bytes(b"%PDF-1.4")
    template_b.write_bytes(b"%PDF-1.4")
    _write_image(template_root / "template_a.png", np.full((8, 8, 3), 255, dtype=np.uint8))
    _write_image(template_root / "template_b.png", np.full((8, 8, 3), 255, dtype=np.uint8))
    _write_image(dataset_root / "images" / "sample_a.png", np.full((8, 8, 3), 255, dtype=np.uint8))
    _write_image(dataset_root / "images" / "sample_b.png", np.full((8, 8, 3), 255, dtype=np.uint8))
    (dataset_root / "visual_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "fixture",
                "samples": [
                    {
                        "sample_id": "sample_a",
                        "image_filename": "sample_a.png",
                        "template_filename": "template_a.pdf",
                        "change_regions": [],
                    },
                    {
                        "sample_id": "sample_b",
                        "image_filename": "sample_b.png",
                        "template_filename": "template_b.pdf",
                        "change_regions": [],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    default_template, cases, _ = evaluator.load_cases(dataset_root)

    assert default_template == template_a
    assert [case.template_path for case in cases] == [template_a, template_b]


def test_annotate_case_stability_filters_blank_graphic_gt(tmp_path):
    template = np.full((40, 40, 3), 255, dtype=np.uint8)
    target = template.copy()
    target[20:30, 20:30] = 0
    template_path = tmp_path / "template.png"
    target_path = tmp_path / "target.png"
    blank_target_path = tmp_path / "blank_target.png"
    _write_image(template_path, template)
    _write_image(target_path, target)
    _write_image(blank_target_path, template)

    visible_case = evaluator.SyntheticCase(
        sample_id="visible",
        defect_source="svg_only",
        image_path=target_path,
        compare_path=None,
        manifest_sample={
            "change_regions": [
                {"kind": "svg", "pixel_bbox": [20, 20, 30, 30]},
            ]
        },
        ground_truth_path=tmp_path / "visible_preview.png",
    )
    blank_case = evaluator.SyntheticCase(
        sample_id="blank",
        defect_source="svg_only",
        image_path=blank_target_path,
        compare_path=None,
        manifest_sample={
            "change_regions": [
                {"kind": "svg", "pixel_bbox": [20, 20, 30, 30]},
            ]
        },
    )

    annotated = evaluator.annotate_case_stability(
        [visible_case, blank_case],
        template_path,
        template_preview_path=None,
        diff_threshold=12,
        min_diff_ratio=0.0005,
    )

    assert annotated[0].stable is True
    assert annotated[1].stable is False
    assert annotated[0].ground_truth_path == visible_case.ground_truth_path
    assert annotated[1].stability_issues == ("graphic_gt_0_no_visible_diff",)
    assert evaluator.filter_cases(annotated, None, None, None) == [annotated[0]]


def test_summarize_includes_pass_rate_duration_and_verdict_distribution():
    summary = evaluator.summarize(
        [
            {
                "sample_id": "s1",
                "defect_source": "text_only",
                "passed": True,
                "issues": [],
                "warnings": [],
                "duration_seconds": 1.0,
                "detection_duration_seconds": 0.8,
                "actual": {"verdict": "不一致"},
                "box_evaluation": {
                    "expected_count": 1,
                    "predicted_count": 1,
                    "matched_count": 1,
                    "missing_count": 0,
                    "extra_count": 0,
                },
            },
            {
                "sample_id": "s2",
                "defect_source": "mixed",
                "passed": False,
                "issues": ["missing_box"],
                "warnings": ["text_content_miss"],
                "duration_seconds": 3.0,
                "detection_duration_seconds": 2.6,
                "actual": {"verdict": "通过"},
                "box_evaluation": {
                    "expected_count": 1,
                    "predicted_count": 0,
                    "matched_count": 0,
                    "missing_count": 1,
                    "extra_count": 0,
                },
            },
        ]
    )

    assert summary["stats"]["pass_rate"] == 0.5
    assert summary["stats"]["duration_seconds"] == {"average": 2.0, "median": 2.0}
    assert summary["stats"]["detection_duration_seconds"] == {
        "average": 1.7,
        "median": 1.7,
    }
    assert summary["stats"]["verdict_distribution"] == {"不一致": 1, "通过": 1}
    assert summary["stats"]["box_stats"]["missing"] == 1


def test_run_case_uses_desktop_service_result(tmp_path, monkeypatch):
    template_path = tmp_path / "template.png"
    target_path = tmp_path / "target.png"
    _write_image(template_path, np.full((20, 20, 3), 255, dtype=np.uint8))
    _write_image(target_path, np.full((20, 20, 3), 255, dtype=np.uint8))

    case = evaluator.SyntheticCase(
        sample_id="sample_0001",
        defect_source="text_only",
        image_path=target_path,
        compare_path=None,
        manifest_sample={
            "code": "600004075219",
            "change_regions": [
                {"kind": "text", "pixel_bbox": [2, 2, 8, 8]},
            ],
        },
        template_path=template_path,
    )

    class FakeResult:
        success = True
        verdict = "不一致"
        summary_text = "summary"
        text_match_count = 0
        text_total_count = 1
        graphic_match_count = 0
        graphic_mismatch_count = 0
        graphic_review_count = 0
        error = None

        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.visualization_path = output_dir / "visualization_diff.jpg"
            self.raw_result = {}

    def fake_run_desktop_detection(case, template_path, output_dir, output_mode, desktop_pipeline):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "visualization_diff.jpg").write_bytes(b"vis")
        payload = {
            "success": True,
            "verdict": "不一致",
            "visualization_annotations": [
                {
                    "kind": "text",
                    "status": "diff",
                    "decision": "mismatch",
                    "box": [2, 2, 8, 8],
                }
            ],
            "text_detection": {
                "fields": ["model_number"],
                "template_data": {"model_number": "A"},
                "target_data": {"model_number": "B"},
            },
            "graphic_comparison": {"comparison_results": []},
        }
        (output_dir / "final_result.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return FakeResult(output_dir)

    monkeypatch.setattr(evaluator, "run_desktop_detection", fake_run_desktop_detection)

    result = evaluator.run_case(
        case,
        template_path=template_path,
        output_root=tmp_path / "eval",
        output_mode="final",
        desktop_pipeline="unified",
        box_iou_threshold=0.1,
        box_coverage_threshold=0.5,
        max_box_area_ratio=6.0,
    )

    assert result["passed"] is True
    assert result["desktop_pipeline"] == "unified"
    assert result["desktop_result"]["verdict"] == "不一致"
    assert result["detection_duration_seconds"] >= 0
    assert result["paths"]["desktop_output_dir"].endswith("sample_0001")
    assert result["paths"]["ground_truth_boxes"].endswith("ground_truth_boxes.png")
    assert (tmp_path / "eval" / "cases" / "sample_0001" / "ground_truth_boxes.png").exists()
    assert (tmp_path / "eval" / "cases" / "sample_0001" / "evaluation_result.json").exists()


def test_evaluate_result_uses_traditional_final_boxes_display_box():
    case = evaluator.SyntheticCase(
        sample_id="sample_traditional",
        defect_source="svg_only",
        image_path=Path("/tmp/target.png"),
        compare_path=None,
        manifest_sample={
            "code": "600001076226",
            "change_regions": [
                {"kind": "svg", "pixel_bbox": [265, 95, 355, 138]},
            ],
        },
        template_path=Path("/tmp/template.png"),
    )

    raw_result = {
        "pipeline": "traditional_full_image_diff",
        "verdict": "不一致",
        "final_boxes": [
            {
                "box": [27, 29, 1989, 362],
                "display_box": [51, 69, 1922, 239],
                "vlm_decision": "keep",
                "vlm_reason": "missing printed content",
            }
        ],
        "graphic_comparison": {"comparison_results": []},
        "text_detection": {"fields": [], "template_data": {}, "target_data": {}},
    }

    evaluation = evaluator.evaluate_result(
        case,
        raw_result,
        box_iou_threshold=0.1,
        box_coverage_threshold=0.5,
        max_box_area_ratio=6.0,
    )

    assert evaluation["evaluation_schema"] == "box_v1"
    assert evaluation["passed"] is True
    assert evaluation["issues"] == []
    assert evaluation["warnings"] == []
    assert evaluation["box_evaluation"]["matched_count"] == 1
    assert evaluation["box_evaluation"]["missing_count"] == 0
    assert evaluation["actual"]["box_count"] == 1
    assert evaluation["actual"]["graphic_diff"] is True


def test_evaluate_result_matches_text_span_gt_with_text_line_display_box():
    case = evaluator.SyntheticCase(
        sample_id="sample_text_line",
        defect_source="text_only",
        image_path=Path("/tmp/target.png"),
        compare_path=None,
        manifest_sample={
            "code": "600004075219",
            "text_mutation_count": 1,
            "change_regions": [
                {
                    "kind": "text",
                    "pixel_bbox": [120, 40, 145, 52],
                    "original_text": "5.20kW",
                    "mutated_text": "5.2OkW",
                },
            ],
        },
        template_path=Path("/tmp/template.pdf"),
    )

    raw_result = {
        "pipeline": "traditional_full_image_diff",
        "verdict": "不一致",
        "final_boxes": [
            {
                "box": [130, 44, 136, 50],
                "display_box": [80, 34, 240, 62],
                "vlm_decision": "keep",
                "matched_template_text_line": {
                    "text": "Heating Capacity 5.20kW",
                    "box": [80, 34, 240, 62],
                },
            }
        ],
        "text_detection": {},
        "graphic_comparison": {},
    }

    evaluation = evaluator.evaluate_result(
        case,
        raw_result,
        box_iou_threshold=0.1,
        box_coverage_threshold=0.5,
        max_box_area_ratio=6.0,
    )

    assert evaluation["passed"] is True
    assert evaluation["issues"] == []
    assert evaluation["box_evaluation"]["policy"]["match_mode"] == "closed_loop"
    assert evaluation["box_evaluation"]["policy"]["text_match_mode"] == "gt_span_coverage"
    assert evaluation["box_evaluation"]["text_expected_count"] == 1
    assert evaluation["box_evaluation"]["text_matched_count"] == 1
    assert evaluation["box_evaluation"]["matches"][0]["expected_coverage"] == 1.0


def test_run_case_replaces_existing_output_dir(tmp_path, monkeypatch):
    template_path = tmp_path / "template.png"
    target_path = tmp_path / "target.png"
    _write_image(template_path, np.full((20, 20, 3), 255, dtype=np.uint8))
    _write_image(target_path, np.full((20, 20, 3), 255, dtype=np.uint8))

    case = evaluator.SyntheticCase(
        sample_id="sample_0002",
        defect_source="text_only",
        image_path=target_path,
        compare_path=None,
        manifest_sample={
            "code": "600004075219",
            "change_regions": [
                {"kind": "text", "pixel_bbox": [2, 2, 8, 8]},
            ],
        },
        template_path=template_path,
    )

    stale_dir = tmp_path / "eval" / "cases" / "sample_0002"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

    def fake_run_desktop_detection(case, template_path, output_dir, output_mode, desktop_pipeline):
        assert not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "visualization_diff.jpg").write_bytes(b"vis")
        (output_dir / "final_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "verdict": "不一致",
                    "visualization_annotations": [],
                    "text_detection": {},
                    "graphic_comparison": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class FakeResult:
            success = True
            verdict = "不一致"
            summary_text = "summary"
            text_match_count = 0
            text_total_count = 0
            graphic_match_count = 0
            graphic_mismatch_count = 0
            graphic_review_count = 0
            error = None

            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
                self.visualization_path = output_dir / "visualization_diff.jpg"
                self.raw_result = {}

        return FakeResult(output_dir)

    monkeypatch.setattr(evaluator, "run_desktop_detection", fake_run_desktop_detection)

    result = evaluator.run_case(
        case,
        template_path=template_path,
        output_root=tmp_path / "eval",
        output_mode="final",
        desktop_pipeline="traditional_full_image_diff",
        box_iou_threshold=0.1,
        box_coverage_threshold=0.5,
        max_box_area_ratio=6.0,
    )

    assert result["passed"] is False
    assert result["issues"] == ["missing_box"]
    assert not (stale_dir / "stale.txt").exists()
    assert (tmp_path / "eval" / "cases" / "sample_0002" / "evaluation_result.json").exists()


def test_main_dry_run_filters_unstable_cases(tmp_path, capsys):
    dataset_root = tmp_path / "dataset"
    template = np.full((40, 40, 3), 255, dtype=np.uint8)
    target_visible = template.copy()
    target_visible[10:20, 10:20] = 0
    target_blank = template.copy()
    _write_image(dataset_root / "template" / "template.png", template)
    _write_image(dataset_root / "images" / "sample_visible.png", target_visible)
    _write_image(dataset_root / "images" / "sample_blank.png", target_blank)
    (dataset_root / "visual_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "fixture",
                "samples": [
                    {
                        "sample_id": "sample_visible",
                        "image_filename": "sample_visible.png",
                        "defect_source": "svg_only",
                        "change_regions": [
                            {"kind": "svg", "pixel_bbox": [10, 10, 20, 20]},
                        ],
                    },
                    {
                        "sample_id": "sample_blank",
                        "image_filename": "sample_blank.png",
                        "defect_source": "svg_only",
                        "change_regions": [
                            {"kind": "svg", "pixel_bbox": [10, 10, 20, 20]},
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = evaluator.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(tmp_path / "eval"),
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "sample_visible" in output
    assert "sample_blank" not in output
    assert "Filtered unstable GT cases: 1" in output


def test_build_representative_failure_artifact_creates_comparison_image(tmp_path):
    output_root = tmp_path / "eval"
    case_dir = output_root / "cases" / "sample_fail"
    case_dir.mkdir(parents=True, exist_ok=True)
    template = np.full((32, 40, 3), 255, dtype=np.uint8)
    target = np.full((32, 40, 3), 200, dtype=np.uint8)
    diff = np.full((32, 40, 3), 50, dtype=np.uint8)
    template_path = case_dir / "template.png"
    target_path = case_dir / "target.png"
    diff_path = case_dir / "visualization_diff.jpg"
    _write_image(template_path, template)
    _write_image(target_path, target)
    _write_image(diff_path, diff)

    artifact = evaluator.build_representative_failure_artifact(
        output_root,
        [
            {
                "sample_id": "sample_ok",
                "passed": True,
                "paths": {},
            },
            {
                "sample_id": "sample_fail",
                "defect_source": "text_only",
                "passed": False,
                "issues": ["missing_box"],
                "warnings": ["text_content_miss"],
                "duration_seconds": 1.23,
                "box_evaluation": {"expected_count": 1, "predicted_count": 0},
                "paths": {
                    "template": str(template_path),
                    "target": str(target_path),
                    "visualization_diff": str(diff_path),
                },
            },
        ],
    )

    assert artifact is not None
    comparison_path = output_root / evaluator.REPRESENTATIVE_FAILURE_DIR_NAME / evaluator.REPRESENTATIVE_FAILURE_IMAGE_NAME
    metadata_path = output_root / evaluator.REPRESENTATIVE_FAILURE_DIR_NAME / "metadata.json"
    assert comparison_path.exists()
    assert metadata_path.exists()
    assert artifact["sample_id"] == "sample_fail"
    assert artifact["paths"]["comparison_image"] == str(comparison_path)
