import json
from pathlib import Path

import cv2
import fitz
import numpy as np

from scripts.prepare_dataset_project_eval_dataset import convert_dataset_project_output


def _write_pdf(path: Path, text: str = "Model 123") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=100, height=50)
    page.insert_text((10, 20), text)
    doc.save(path)
    doc.close()


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (10, 10), (0, 0, 255), 2)
    assert cv2.imwrite(str(path), image)


def test_convert_dataset_project_output_writes_visual_manifest(tmp_path):
    source_root = tmp_path / "dataset_source"
    dataset_dir = source_root / "600000000001"
    sample_pdf = dataset_dir / "pdfs" / "sample_0001.pdf"
    sample_preview = dataset_dir / "pdfs" / "sample_0001.png"
    source_pdf = tmp_path / "source.pdf"
    extracted_pdf = tmp_path / "extracted.pdf"
    _write_pdf(source_pdf, "Template")
    _write_pdf(extracted_pdf, "Template")
    _write_pdf(sample_pdf, "Model 124")
    _write_image(sample_preview)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "600000000001",
                "source_pdf": str(source_pdf),
                "extracted_pdf": str(extracted_pdf),
                "samples": [
                    {
                        "sample_id": "sample_0001",
                        "filename": "sample_0001.pdf",
                        "output_pdf": str(sample_pdf),
                        "label_binary": "defect",
                        "text_mutation_count": 1,
                        "text_change_summary": "confusable:3->4",
                        "seed": 123,
                        "preview_image": str(sample_preview),
                        "text_debug": {
                            "mutations": [
                                {
                                    "strategy": "confusable",
                                    "original_text": "3",
                                    "mutated_text": "4",
                                    "bbox": [10, 10, 20, 20],
                                    "font_size": 10,
                                }
                            ]
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "converted"
    summary = convert_dataset_project_output(
        source_root,
        output_dir,
        target_dpi=144,
        overwrite=False,
    )

    assert summary["template_count"] == 1
    assert summary["sample_count"] == 1
    assert (output_dir / "images" / "600000000001_sample_0001.png").exists()
    assert (output_dir / "pdfs" / "600000000001_sample_0001.pdf").exists()
    assert (output_dir / "ground_truth" / "600000000001_sample_0001.png").exists()
    manifest = json.loads((output_dir / "visual_manifest.json").read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    assert sample["sample_id"] == "600000000001_sample_0001"
    assert sample["defect_source"] == "text_only"
    assert sample["ground_truth_filename"] == "600000000001_sample_0001.png"
    assert sample["change_regions"][0]["kind"] == "text"
    assert sample["change_regions"][0]["pixel_bbox"] == [20, 20, 40, 40]
