import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from label_detection.workflows import unified


class DummyLabelModel(BaseModel):
    model: str | None = None


def _write_image(path: Path, value: int = 255) -> None:
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _install_workflow_stubs(monkeypatch, tmp_path: Path):
    compare_text_calls = []

    def fake_resolve_template_input(template_path, output_dir=None, target_dpi=300):
        assert output_dir is not None
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        extracted_path = output_dir_path / "template_page1.png"
        _write_image(extracted_path, value=240)
        return str(extracted_path), "pdf", None

    def fake_preprocess_template_image(template_path, output_dir):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(template_path))
        output_path = output_dir_path / "template_preprocessed.jpg"
        assert cv2.imwrite(str(output_path), image)
        return image, str(output_path), None

    def fake_preprocess_target(target_path, output_dir="results/preprocessed", template_image=None):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(target_path))
        return image, None, None

    def fake_compare_text_results(data1, data2, output_path=None):
        compare_text_calls.append(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"xlsx")
        return str(output_path)

    monkeypatch.setattr(unified, "resolve_template_input", fake_resolve_template_input)
    monkeypatch.setattr(unified, "preprocess_template_image", fake_preprocess_template_image)
    monkeypatch.setattr(unified, "preprocess_target", fake_preprocess_target)
    monkeypatch.setattr(unified, "get_ocr_with_boxes", lambda image_path: ("MODEL-A", []))
    monkeypatch.setattr(unified, "infer_label_kind", lambda text: "dummy")
    monkeypatch.setattr(unified, "get_label_model", lambda label_kind: DummyLabelModel)
    monkeypatch.setattr(
        unified,
        "run_llm_extraction",
        lambda image_path, ocr_text, model_cls, label_kind, max_retries=None: DummyLabelModel(model="A"),
    )
    monkeypatch.setattr(unified, "compare_text_results", fake_compare_text_results)
    monkeypatch.setattr(unified, "detect_layout_regions", lambda image_path: [])
    monkeypatch.setattr(unified, "extract_regions_by_type", lambda regions, region_type: list(regions))
    monkeypatch.setattr(unified, "split_barcode_regions", lambda regions, image, boxes: (list(regions), []))
    monkeypatch.setattr(
        unified,
        "split_composite_image_regions",
        lambda regions, image, boxes: (list(regions), []),
    )
    monkeypatch.setattr(unified, "match_regions", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(unified, "draw_regions", lambda image, regions, color=(0, 255, 0): image.copy())
    return compare_text_calls


def test_run_unified_detection_final_mode_keeps_only_final_outputs(tmp_path, monkeypatch):
    template_path = tmp_path / "template.pdf"
    target_path = tmp_path / "target.jpg"
    output_dir = tmp_path / "results-final"
    template_path.write_bytes(b"%PDF-1.4")
    _write_image(target_path, value=220)
    compare_text_calls = _install_workflow_stubs(monkeypatch, tmp_path)

    result = unified.run_unified_detection(
        str(template_path),
        str(target_path),
        output_dir=str(output_dir),
        output_mode="final",
    )

    assert result["success"] is True
    assert compare_text_calls == []
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "final_result.json",
        "visualization_diff.jpg",
    ]

    payload = json.loads((output_dir / "final_result.json").read_text(encoding="utf-8"))
    assert payload["output_mode"] == "final"
    assert payload["text_detection"]["excel_path"] is None
    assert payload["template_input"]["resolved_image_path"] is None
    assert not (output_dir / "debug").exists()


def test_run_unified_detection_debug_mode_keeps_debug_artifacts(tmp_path, monkeypatch):
    template_path = tmp_path / "template.pdf"
    target_path = tmp_path / "target.jpg"
    output_dir = tmp_path / "results-debug"
    template_path.write_bytes(b"%PDF-1.4")
    _write_image(target_path, value=220)
    compare_text_calls = _install_workflow_stubs(monkeypatch, tmp_path)

    result = unified.run_unified_detection(
        str(template_path),
        str(target_path),
        output_dir=str(output_dir),
        output_mode="debug",
    )

    assert result["success"] is True
    assert len(compare_text_calls) == 1
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "debug",
        "final_result.json",
        "visualization_diff.jpg",
    ]

    debug_dir = output_dir / "debug"
    assert (debug_dir / "preprocess" / "template_preprocessed.jpg").exists()
    assert (debug_dir / "preprocess" / "target_preprocessed.jpg").exists()
    assert (debug_dir / "text_comparison.xlsx").exists()
    assert (debug_dir / "graphic_comparison" / "template_regions_detected.jpg").exists()
    assert (debug_dir / "graphic_comparison" / "target_regions_detected.jpg").exists()
    assert (debug_dir / "template_assets" / "template_page1.png").exists()

    payload = json.loads((output_dir / "final_result.json").read_text(encoding="utf-8"))
    assert payload["output_mode"] == "debug"
    assert payload["text_detection"]["excel_path"].endswith("debug/text_comparison.xlsx")
    assert payload["template_input"]["resolved_image_path"].endswith(
        "debug/template_assets/template_page1.png"
    )
