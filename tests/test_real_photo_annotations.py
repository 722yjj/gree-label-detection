import json
from pathlib import Path

from desktop_app.models import DetectionJobResult
from desktop_app.repositories.annotation_repository import AnnotationRepository
from scripts.evaluate_real_photo_annotations import evaluate_manifest
from scripts.export_real_photo_annotations import export_annotations


def test_annotation_repository_builds_and_saves_from_detection_result(tmp_path):
    run_dir = tmp_path / "desktop_app" / "600004075219" / "run_default"
    run_dir.mkdir(parents=True)
    visualization = run_dir / "visualization_diff.jpg"
    annotation_base = run_dir / "annotation_base.jpg"
    visualization.write_bytes(b"jpg")
    annotation_base.write_bytes(b"base")
    target = tmp_path / "target.jpg"
    template = tmp_path / "template.pdf"
    target.write_bytes(b"target")
    template.write_bytes(b"template")
    raw_result = {
        "final_boxes": [
            {
                "box": [10, 20, 30, 40],
                "review_box_source": "pdf_text_line",
                "matched_template_text_line": {"text": "Rated Voltage 220-240V"},
            }
        ],
        "visualization_diff": str(visualization),
        "annotation_image": str(annotation_base),
        "annotation_coordinate_space": "aligned_label_image",
    }
    result = DetectionJobResult(
        success=True,
        verdict="不一致",
        output_dir=run_dir,
        summary_text="summary",
        visualization_path=visualization,
        raw_result=raw_result,
        target_image_path=target,
        template_path=template,
        code="600004075219",
        template_display_name="600004075219",
    )

    repository = AnnotationRepository()
    annotation = repository.build_from_detection_result(result)

    assert annotation["run_dir"] == str(run_dir)
    assert annotation["image"]["path"] == str(annotation_base)
    assert annotation["coordinate_space"] == "aligned_label_image"
    assert "case_label" not in annotation
    assert "capture_quality" not in annotation
    assert annotation["predicted_boxes"][0]["bbox"] == [10, 20, 30, 40]
    assert annotation["predicted_boxes"][0]["decision"] == "unreviewed"

    annotation["predicted_boxes"][0]["decision"] = "true_positive"
    path = repository.save(annotation)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["predicted_boxes"][0]["decision"] == "true_positive"


def test_export_and_evaluate_real_photo_annotations(tmp_path):
    source_root = tmp_path / "desktop_app"
    run_dir = source_root / "600004075219" / "run_default"
    run_dir.mkdir(parents=True)
    (run_dir / "final_result.json").write_text(
        json.dumps({"duration_seconds": 2.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    annotation = {
        "version": "real_photo_annotation_v1",
        "run_dir": str(run_dir),
        "code": "600004075219",
        "template": str(tmp_path / "template.pdf"),
        "target_image": str(tmp_path / "target.jpg"),
        "image": {"path": str(run_dir / "visualization_diff.jpg")},
        "detection_duration_seconds": 1.25,
        "predicted_boxes": [
            {"box_id": "pred_0", "bbox": [1, 2, 3, 4], "decision": "true_positive"},
            {"box_id": "pred_1", "bbox": [5, 6, 7, 8], "decision": "false_positive"},
        ],
        "manual_gt_boxes": [
            {"bbox": [9, 10, 11, 12], "source": "manual_missing"},
        ],
    }
    (run_dir / "manual_annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )

    output_dir = tmp_path / "real_photo_dataset"
    exported = export_annotations(source_root, output_dir, include_unreviewed=False)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = evaluate_manifest(manifest)

    assert exported["sample_count"] == 1
    assert manifest["samples"][0]["detection_duration_seconds"] == 1.25
    assert summary["sample_count"] == 1
    assert summary["reviewed_count"] == 1
    assert summary["box_decision_counts"] == {
        "false_positive": 1,
        "true_positive": 1,
    }
    assert summary["manual_missing_box_count"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall_proxy"] == 0.5
    assert summary["detection_duration_seconds"]["average"] == 1.25


def test_export_skips_unreviewed_annotation_by_default(tmp_path):
    source_root = tmp_path / "desktop_app"
    run_dir = source_root / "600004075219" / "run_default"
    run_dir.mkdir(parents=True)
    annotation = {
        "version": "real_photo_annotation_v1",
        "run_dir": str(run_dir),
        "code": "600004075219",
        "image": {"path": str(run_dir / "visualization_diff.jpg")},
        "predicted_boxes": [
            {"box_id": "pred_0", "bbox": [1, 2, 3, 4], "decision": "unreviewed"},
        ],
        "manual_gt_boxes": [],
    }
    (run_dir / "manual_annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )

    output_dir = tmp_path / "real_photo_dataset"
    exported = export_annotations(source_root, output_dir, include_unreviewed=False)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert exported["sample_count"] == 0
    assert exported["skipped_unreviewed"] == 1
    assert manifest["samples"] == []
