"""Convert the external dataset project output into an evaluable visual_manifest dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import PROJECT_ROOT as REPO_ROOT


DEFAULT_SOURCE_ROOT = Path("/home/jnu/projects/dataset/outputs/datasets_rendered_label_only")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dataset_project_eval_dataset"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert /home/jnu/projects/dataset outputs into the visual_manifest "
            "format consumed by scripts/evaluate_synthetic_dataset.py."
        )
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Dataset project output root. May contain per-template subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Converted evaluation dataset output directory.",
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=300,
        help="DPI used to render sample PDFs into unmarked target PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing converted output directory first.",
    )
    return parser


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def discover_manifests(source_root: Path) -> list[Path]:
    if (source_root / "manifest.json").exists():
        return [source_root / "manifest.json"]
    manifests = sorted(path for path in source_root.glob("*/manifest.json") if path.is_file())
    if not manifests:
        raise FileNotFoundError(f"No dataset project manifest.json files found under {source_root}")
    return manifests


def render_pdf_page(
    pdf_path: Path,
    output_path: Path,
    *,
    dpi: int,
) -> tuple[int, int, float]:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyMuPDF is required to render dataset project PDFs.") from exc

    scale = dpi / 72.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if doc.page_count < 1:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pix.save(output_path)
        return int(pix.width), int(pix.height), scale


def resolve_manifest_path(manifest_path: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path.resolve()
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate.resolve()
    return path


def scaled_bbox(bbox: Sequence[float], scale: float, image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) * scale for value in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]


def change_regions_from_text_debug(
    text_debug: Optional[Dict[str, Any]],
    *,
    scale: float,
    image_size: tuple[int, int],
) -> list[Dict[str, Any]]:
    regions: list[Dict[str, Any]] = []
    for index, mutation in enumerate((text_debug or {}).get("mutations") or []):
        bbox = mutation.get("bbox")
        if not bbox:
            continue
        regions.append(
            {
                "kind": "text",
                "label": "text_mutation",
                "pixel_bbox": scaled_bbox(bbox, scale, image_size),
                "strategy": mutation.get("strategy"),
                "original_text": mutation.get("original_text"),
                "mutated_text": mutation.get("mutated_text"),
                "mutation_index": index,
                "font_size": mutation.get("font_size"),
            }
        )
    return regions


def write_csv_manifest(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "template_id",
        "defect_source",
        "image_filename",
        "sample_pdf_filename",
        "text_mutation_count",
        "change_region_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.get("sample_id"),
                    "template_id": sample.get("template_id"),
                    "defect_source": sample.get("defect_source"),
                    "image_filename": sample.get("image_filename"),
                    "sample_pdf_filename": sample.get("sample_pdf_filename"),
                    "text_mutation_count": sample.get("text_mutation_count"),
                    "change_region_count": len(sample.get("change_regions") or []),
                }
            )


def convert_dataset_project_output(
    source_root: Path,
    output_dir: Path,
    *,
    target_dpi: int,
    overwrite: bool,
) -> Dict[str, Any]:
    if target_dpi <= 0:
        raise ValueError("--target-dpi must be positive")

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    manifests = discover_manifests(source_root)
    prepare_output_dir(output_dir, overwrite=overwrite)

    template_dir = output_dir / "template"
    source_template_dir = output_dir / "source_templates"
    image_dir = output_dir / "images"
    pdf_dir = output_dir / "pdfs"
    ground_truth_dir = output_dir / "ground_truth"
    source_manifest_dir = output_dir / "source_manifests"
    for path in (
        template_dir,
        source_template_dir,
        image_dir,
        pdf_dir,
        ground_truth_dir,
        source_manifest_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    samples: list[Dict[str, Any]] = []
    template_entries: list[Dict[str, Any]] = []
    copied_templates: dict[Path, str] = {}
    rendered_template_previews: dict[Path, str] = {}

    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        template_id = str(manifest.get("dataset_name") or manifest_path.parent.name)
        source_pdf = resolve_manifest_path(manifest_path, manifest.get("source_pdf"))
        if source_pdf is None or not source_pdf.exists():
            raise FileNotFoundError(f"source_pdf not found in {manifest_path}: {source_pdf}")
        extracted_pdf = resolve_manifest_path(manifest_path, manifest.get("extracted_pdf"))

        if source_pdf not in copied_templates:
            template_target = source_template_dir / source_pdf.name
            shutil.copy2(source_pdf, template_target)
            copied_templates[source_pdf] = template_target.name
        template_filename = copied_templates[source_pdf]

        preview_source = extracted_pdf if extracted_pdf and extracted_pdf.exists() else source_pdf
        if preview_source not in rendered_template_previews:
            preview_name = f"{template_id}.png"
            render_pdf_page(preview_source, template_dir / preview_name, dpi=target_dpi)
            rendered_template_previews[preview_source] = preview_name
        template_preview_filename = rendered_template_previews[preview_source]

        shutil.copy2(manifest_path, source_manifest_dir / f"{template_id}_manifest.json")

        source_samples = list(manifest.get("samples") or [])
        template_entries.append(
            {
                "template_id": template_id,
                "source_manifest": str(manifest_path),
                "source_pdf": str(source_pdf),
                "template_filename": template_filename,
                "template_preview_filename": template_preview_filename,
                "sample_count": len(source_samples),
            }
        )

        for sample in source_samples:
            source_sample_id = str(sample.get("sample_id") or Path(sample.get("filename", "")).stem)
            if not source_sample_id:
                continue
            sample_id = f"{template_id}_{source_sample_id}"
            output_pdf = resolve_manifest_path(
                manifest_path,
                sample.get("output_pdf") or sample.get("filename"),
            )
            if output_pdf is None or not output_pdf.exists():
                raise FileNotFoundError(f"sample output_pdf not found: {output_pdf}")

            sample_pdf_name = f"{sample_id}.pdf"
            sample_image_name = f"{sample_id}.png"
            ground_truth_image_name = f"{sample_id}.png"
            shutil.copy2(output_pdf, pdf_dir / sample_pdf_name)
            width, height, scale = render_pdf_page(
                output_pdf,
                image_dir / sample_image_name,
                dpi=target_dpi,
            )
            change_regions = change_regions_from_text_debug(
                sample.get("text_debug"),
                scale=scale,
                image_size=(width, height),
            )
            if not change_regions:
                raise ValueError(f"No text mutation bboxes found for {sample_id}")

            preview_source = resolve_manifest_path(manifest_path, sample.get("preview_image"))
            if preview_source and preview_source.exists():
                shutil.copy2(preview_source, ground_truth_dir / ground_truth_image_name)
            else:
                ground_truth_image_name = ""

            samples.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_sample_id,
                    "template_id": template_id,
                    "template_filename": template_filename,
                    "template_preview_filename": template_preview_filename,
                    "image_filename": sample_image_name,
                    "sample_pdf_filename": sample_pdf_name,
                    "ground_truth_filename": ground_truth_image_name,
                    "defect_source": "text_only",
                    "label_binary": sample.get("label_binary"),
                    "text_mutation_count": sample.get("text_mutation_count"),
                    "text_change_summary": sample.get("text_change_summary") or "",
                    "seed": sample.get("seed"),
                    "change_regions": change_regions,
                }
            )

    if not samples:
        raise RuntimeError("No samples were converted.")

    manifest = {
        "dataset_name": output_dir.name,
        "source_project": str(source_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_dpi": target_dpi,
        "template_filename": template_entries[0]["template_filename"],
        "template_preview_filename": template_entries[0]["template_preview_filename"],
        "templates": template_entries,
        "samples": samples,
    }
    write_json(output_dir / "visual_manifest.json", manifest)
    write_json(
        output_dir / "manifest.json",
        {
            "dataset_name": output_dir.name,
            "source_project": str(source_root),
            "generated_at": manifest["generated_at"],
            "target_dpi": target_dpi,
            "templates": template_entries,
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "template_id": sample["template_id"],
                    "image_path": repo_relative(image_dir / sample["image_filename"]),
                    "sample_pdf_path": repo_relative(pdf_dir / sample["sample_pdf_filename"]),
                    "ground_truth_path": (
                        repo_relative(ground_truth_dir / sample["ground_truth_filename"])
                        if sample.get("ground_truth_filename")
                        else None
                    ),
                    "text_mutation_count": sample["text_mutation_count"],
                    "change_region_count": len(sample["change_regions"]),
                }
                for sample in samples
            ],
        },
    )
    write_csv_manifest(output_dir / "manifest.csv", samples)
    return {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "template_count": len(template_entries),
        "sample_count": len(samples),
        "target_dpi": target_dpi,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = convert_dataset_project_output(
        Path(args.source_root),
        Path(args.output_dir),
        target_dpi=args.target_dpi,
        overwrite=args.overwrite,
    )
    print(
        "Dataset project samples converted: "
        f"templates={summary['template_count']} samples={summary['sample_count']} "
        f"target_dpi={summary['target_dpi']} output={summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
