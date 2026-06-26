"""Evaluate the detection workflow on a manifest-driven synthetic dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import PROJECT_ROOT as REPO_ROOT
from label_detection.extraction.template_source import SUPPORTED_TEMPLATE_IMAGE_SUFFIXES
from label_detection.matching.ocr import text_field_values_match


DEFAULT_DATASET_ROOT = Path("/home/data/数据集/outputs/datasets/label_aug_mixed_20260406_200")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "synthetic_dataset_eval"

RESULT_JSON_NAME = "evaluation_result.json"
SUMMARY_JSON_NAME = "summary.json"
EVALUATION_SCHEMA = "box_v4_text_span_coverage"
GROUND_TRUTH_BOXES_IMAGE_NAME = "ground_truth_boxes.png"
DATASET_PREVIEW_IMAGE_NAME = "dataset_preview.png"
REPRESENTATIVE_FAILURE_DIR_NAME = "representative_failure"
REPRESENTATIVE_FAILURE_IMAGE_NAME = "comparison.jpg"
REPRESENTATIVE_FAILURE_TEMPLATE_NAME = "template.png"
REPRESENTATIVE_FAILURE_TARGET_NAME = "target.png"
REPRESENTATIVE_FAILURE_DIFF_NAME = "visualization_diff.jpg"


@dataclass(frozen=True)
class SyntheticCase:
    sample_id: str
    defect_source: str
    image_path: Path
    compare_path: Optional[Path]
    manifest_sample: Dict[str, Any]
    template_path: Optional[Path] = None
    template_preview_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None
    stable: bool = True
    stability_issues: Sequence[str] = ()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic label samples using visual_manifest metadata."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Synthetic dataset root containing template/, images/, and visual_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used to store evaluation outputs.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("final", "debug"),
        default="final",
        help="Detection workflow output mode.",
    )
    parser.add_argument(
        "--desktop-pipeline",
        choices=("unified", "traditional_full_image_diff"),
        default="traditional_full_image_diff",
        help=(
            "Desktop DetectionService pipeline used for evaluation. "
            "Default matches the desktop app's own default pipeline."
        ),
    )
    parser.add_argument(
        "--include-unstable-gt",
        action="store_true",
        help=(
            "Do not filter samples whose manifest GT boxes are outside the image "
            "or have no visible local pixel difference."
        ),
    )
    parser.add_argument(
        "--gt-diff-threshold",
        type=int,
        default=12,
        help="Pixel absolute-difference threshold used by the stable-GT filter.",
    )
    parser.add_argument(
        "--gt-min-diff-ratio",
        type=float,
        default=0.0005,
        help="Minimum local changed-pixel ratio required for a GT region to be stable.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Run only this sample id. Repeat to include multiple samples.",
    )
    parser.add_argument(
        "--defect-source",
        action="append",
        choices=("svg_only", "text_only", "mixed"),
        help="Run only samples with this defect source. Repeat to include multiple groups.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Reuse an existing {RESULT_JSON_NAME} if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned samples without running detection.",
    )
    parser.add_argument(
        "--box-iou-threshold",
        type=float,
        default=0.1,
        help="Minimum IoU for a predicted box to match a GT change box.",
    )
    parser.add_argument(
        "--box-coverage-threshold",
        type=float,
        default=0.5,
        help=(
            "Fallback overlap threshold: both GT coverage and predicted-box "
            "coverage must reach this value when IoU is below threshold."
        ),
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=6.0,
        help="Maximum predicted/GT area ratio allowed for the coverage fallback.",
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
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_dataset_path(
    dataset_root: Path,
    windows_or_relative_path: Optional[str],
) -> Optional[Path]:
    if not windows_or_relative_path:
        return None
    candidate = Path(windows_or_relative_path)
    if candidate.exists():
        return candidate
    return dataset_root / candidate.name


def resolve_template_path(dataset_root: Path, manifest: Dict[str, Any]) -> Path:
    template_root = dataset_root / "template"
    manifest_candidates = [
        manifest.get("template_path"),
        manifest.get("template_filename"),
        manifest.get("template"),
    ]
    for value in manifest_candidates:
        path = resolve_dataset_path(template_root, str(value)) if value else None
        if path and path.exists():
            return path

    suffixes = [".pdf", *sorted(SUPPORTED_TEMPLATE_IMAGE_SUFFIXES)]
    for name in ("template", "cropped_template"):
        for suffix in suffixes:
            candidate = template_root / f"{name}{suffix}"
            if candidate.exists():
                return candidate

    discovered = sorted(
        path
        for path in template_root.glob("*")
        if path.is_file()
        and (
            path.suffix.lower() == ".pdf"
            or path.suffix.lower() in SUPPORTED_TEMPLATE_IMAGE_SUFFIXES
        )
    )
    if discovered:
        return discovered[0]

    raise FileNotFoundError(
        f"template file not found under {template_root}; expected template.pdf or template image"
    )


def resolve_template_preview_path(
    dataset_root: Path,
    template_path: Path,
    manifest: Dict[str, Any],
) -> Optional[Path]:
    if template_path.suffix.lower() in SUPPORTED_TEMPLATE_IMAGE_SUFFIXES:
        return template_path

    template_root = dataset_root / "template"
    manifest_candidates = [
        manifest.get("template_preview_path"),
        manifest.get("template_preview_filename"),
        manifest.get("preview_path"),
        manifest.get("preview_filename"),
    ]
    for value in manifest_candidates:
        if not value:
            continue
        template_candidate = resolve_dataset_path(template_root, str(value))
        if template_candidate and template_candidate.exists():
            return template_candidate
        dataset_candidate = resolve_dataset_path(dataset_root, str(value))
        if dataset_candidate and dataset_candidate.exists():
            return dataset_candidate

    for candidate in (
        template_path.with_suffix(".png"),
        template_root / "template.png",
        template_root / "cropped_template.png",
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_sample_template_path(
    dataset_root: Path,
    sample: Dict[str, Any],
    default_template_path: Path,
) -> Path:
    for key in ("template_path", "template_filename"):
        value = sample.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
        for root in (
            dataset_root / "source_templates",
            dataset_root / "template",
            dataset_root,
        ):
            rooted = root / candidate.name
            if rooted.exists():
                return rooted
    return default_template_path


def resolve_sample_template_preview_path(
    dataset_root: Path,
    sample: Dict[str, Any],
    template_path: Path,
    manifest: Dict[str, Any],
) -> Optional[Path]:
    for key in ("template_preview_path", "template_preview_filename", "preview_filename"):
        value = sample.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
        for root in (
            dataset_root / "template",
            dataset_root,
        ):
            rooted = root / candidate.name
            if rooted.exists():
                return rooted
    return resolve_template_preview_path(dataset_root, template_path, manifest)


def resolve_sample_ground_truth_path(
    dataset_root: Path,
    sample: Dict[str, Any],
) -> Optional[Path]:
    for key in (
        "ground_truth_path",
        "ground_truth_filename",
        "preview_image",
        "preview_filename",
    ):
        value = sample.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
        for root in (
            dataset_root / "ground_truth",
            dataset_root / "compare",
            dataset_root,
        ):
            rooted = root / candidate.name
            if rooted.exists():
                return rooted
    return None


def load_cases(dataset_root: Path) -> tuple[Path, List[SyntheticCase], Dict[str, Any]]:
    visual_manifest_path = dataset_root / "visual_manifest.json"
    if not visual_manifest_path.exists():
        raise FileNotFoundError(f"visual_manifest.json not found: {visual_manifest_path}")

    manifest = read_json(visual_manifest_path)
    template_path = resolve_template_path(dataset_root, manifest)

    cases: List[SyntheticCase] = []
    for sample in manifest.get("samples") or []:
        sample_id = str(sample.get("sample_id") or "").strip()
        image_filename = str(sample.get("image_filename") or "").strip()
        if not sample_id or not image_filename:
            continue
        image_path = dataset_root / "images" / image_filename
        compare_filename = str(sample.get("compare_filename") or "").strip()
        compare_path = dataset_root / "compare" / compare_filename if compare_filename else None
        case_template_path = resolve_sample_template_path(
            dataset_root,
            sample,
            template_path,
        )
        cases.append(
            SyntheticCase(
                sample_id=sample_id,
                defect_source=str(sample.get("defect_source") or ""),
                image_path=image_path,
                compare_path=compare_path if compare_path and compare_path.exists() else None,
                manifest_sample=sample,
                template_path=case_template_path,
                template_preview_path=resolve_sample_template_preview_path(
                    dataset_root,
                    sample,
                    case_template_path,
                    manifest,
                ),
                ground_truth_path=resolve_sample_ground_truth_path(dataset_root, sample),
            )
        )

    cases.sort(key=lambda item: item.sample_id)
    return template_path, cases, manifest


def filter_cases(
    cases: Sequence[SyntheticCase],
    sample_ids: Optional[Sequence[str]],
    defect_sources: Optional[Sequence[str]],
    limit: Optional[int],
    include_unstable_gt: bool = False,
) -> List[SyntheticCase]:
    selected = list(cases)
    if sample_ids:
        allowed = set(sample_ids)
        selected = [case for case in selected if case.sample_id in allowed]
    if defect_sources:
        allowed_sources = set(defect_sources)
        selected = [case for case in selected if case.defect_source in allowed_sources]
    if not include_unstable_gt:
        selected = [case for case in selected if case.stable]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def load_image_for_stability(path: Path):
    try:
        import cv2
    except ImportError:
        return None

    image = cv2.imread(str(path))
    return image


def clip_box_to_image(box: Sequence[float], image_shape: Sequence[int]) -> Optional[List[int]]:
    normalized = normalize_box(box)
    if normalized is None:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = normalized
    clipped = [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def graphic_gt_stability_issues(
    sample: Dict[str, Any],
    template_image_path: Path,
    target_image_path: Path,
    *,
    diff_threshold: int,
    min_diff_ratio: float,
) -> List[str]:
    graphic_regions = [
        region
        for region in sample.get("change_regions") or []
        if normalize_region_kind(region.get("kind")) == "graphic"
    ]
    if not graphic_regions:
        return []

    template_image = load_image_for_stability(template_image_path)
    target_image = load_image_for_stability(target_image_path)
    if template_image is None or target_image is None:
        return ["stability_image_unreadable"]
    if template_image.shape[:2] != target_image.shape[:2]:
        return []

    try:
        import cv2
    except ImportError:
        return []

    issues: List[str] = []
    for index, region in enumerate(graphic_regions):
        raw_box = region.get("pixel_bbox")
        clipped = clip_box_to_image(raw_box, target_image.shape[:2]) if raw_box else None
        if clipped is None:
            issues.append(f"graphic_gt_{index}_box_outside_image")
            continue
        x1, y1, x2, y2 = clipped
        template_crop = template_image[y1:y2, x1:x2]
        target_crop = target_image[y1:y2, x1:x2]
        if template_crop.size == 0 or target_crop.size == 0:
            issues.append(f"graphic_gt_{index}_empty_crop")
            continue
        diff = cv2.absdiff(template_crop, target_crop)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_ratio = float((gray_diff > diff_threshold).mean())
        if diff_ratio < min_diff_ratio:
            issues.append(f"graphic_gt_{index}_no_visible_diff")
    return issues


def annotate_case_stability(
    cases: Sequence[SyntheticCase],
    template_path: Path,
    template_preview_path: Optional[Path] = None,
    *,
    diff_threshold: int,
    min_diff_ratio: float,
) -> List[SyntheticCase]:
    stability_template_path = template_preview_path or template_path
    if stability_template_path.suffix.lower() not in SUPPORTED_TEMPLATE_IMAGE_SUFFIXES:
        return list(cases)

    annotated: List[SyntheticCase] = []
    for case in cases:
        case_template_path = case.template_path or template_path
        case_preview_path = case.template_preview_path or template_preview_path
        stability_template_path = case_preview_path or case_template_path
        issues = graphic_gt_stability_issues(
            case.manifest_sample,
            stability_template_path,
            case.image_path,
            diff_threshold=diff_threshold,
            min_diff_ratio=min_diff_ratio,
        )
        annotated.append(
            SyntheticCase(
                sample_id=case.sample_id,
                defect_source=case.defect_source,
                image_path=case.image_path,
                compare_path=case.compare_path,
                manifest_sample=case.manifest_sample,
                template_path=case.template_path,
                template_preview_path=case.template_preview_path,
                ground_truth_path=case.ground_truth_path,
                stable=not issues,
                stability_issues=tuple(issues),
            )
        )
    return annotated


def build_template_record(template_path: Path, case: SyntheticCase):
    from desktop_app.models import TemplateRecord

    code = str(
        case.manifest_sample.get("code")
        or case.manifest_sample.get("label_code")
        or template_path.stem
    )
    return TemplateRecord(
        code=code,
        variant=case.sample_id,
        display_name=template_path.stem,
        source_type="pdf" if template_path.suffix.lower() == ".pdf" else "image",
        source_path=template_path,
    )


def run_desktop_detection(
    case: SyntheticCase,
    template_path: Path,
    output_dir: Path,
    output_mode: str,
    desktop_pipeline: str,
):
    from desktop_app.models import DetectionJobRequest
    from desktop_app.services.detection_service import DetectionService

    previous_pipeline = os.environ.get("DESKTOP_DETECTION_PIPELINE")
    os.environ["DESKTOP_DETECTION_PIPELINE"] = desktop_pipeline
    try:
        service = DetectionService(output_root=output_dir.parent)
        service._build_output_dir = (  # type: ignore[method-assign]
            lambda _request: output_dir
        )
        request = DetectionJobRequest(
            template=build_template_record(template_path, case),
            target_image_path=case.image_path,
            output_mode=output_mode,
        )
        return service.run(request)
    finally:
        if previous_pipeline is None:
            os.environ.pop("DESKTOP_DETECTION_PIPELINE", None)
        else:
            os.environ["DESKTOP_DETECTION_PIPELINE"] = previous_pipeline


def expected_flags(sample: Dict[str, Any]) -> Dict[str, Any]:
    regions = list(sample.get("change_regions") or [])
    expected_text_regions = [region for region in regions if region.get("kind") == "text"]
    expected_graphic_regions = [region for region in regions if region.get("kind") == "svg"]

    defect_source = str(sample.get("defect_source") or "")
    return {
        "text_diff": bool(expected_text_regions)
        or defect_source in {"text_only", "mixed"}
        or int(sample.get("text_mutation_count") or 0) > 0,
        "graphic_diff": bool(expected_graphic_regions)
        or defect_source in {"svg_only", "mixed"}
        or bool(sample.get("svg_name")),
        "defect_source": defect_source,
        "text_change_summary": sample.get("text_change_summary") or "",
        "svg_name": sample.get("svg_name"),
        "text_regions": expected_text_regions,
        "graphic_regions": expected_graphic_regions,
    }


def text_different_fields(result: Dict[str, Any]) -> List[str]:
    text_detection = dict(result.get("text_detection") or {})
    template_data = dict(text_detection.get("template_data") or {})
    target_data = dict(text_detection.get("target_data") or {})
    fields = list(text_detection.get("fields") or template_data.keys() or target_data.keys())
    return [
        field_name
        for field_name in fields
        if not text_field_values_match(
            field_name,
            template_data.get(field_name),
            target_data.get(field_name),
        )
    ]


def detected_label_union_diffs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Diagnostic only: compare the union of extracted label names."""
    text_detection = dict(result.get("text_detection") or {})
    detected = dict(text_detection.get("detected_labels") or {})
    template_labels = dict(detected.get("template") or {})
    target_labels = dict(detected.get("target") or {})
    diffs: List[Dict[str, Any]] = []
    for field_name in sorted(set(template_labels) | set(target_labels)):
        template_value = template_labels.get(field_name)
        target_value = target_labels.get(field_name)
        if not text_field_values_match(
            f"label:{field_name}",
            template_value,
            target_value,
        ):
            diffs.append(
                {
                    "field": field_name,
                    "template": template_value,
                    "target": target_value,
                }
            )
    return diffs


def graphic_result_buckets(result: Dict[str, Any]) -> Dict[str, Any]:
    graphic = dict(result.get("graphic_comparison") or {})
    comparison_results = list(graphic.get("comparison_results") or [])
    if not comparison_results:
        fallback_boxes = predicted_visual_boxes(result)
        if fallback_boxes:
            comparison_results = [
                {
                    "decision": "mismatch",
                    "box": item.get("box"),
                    "summary": "traditional full-image diff",
                }
                for item in fallback_boxes
            ]
    mismatches = [
        item
        for item in comparison_results
        if item.get("decision") == "mismatch" and not item.get("needs_review", False)
    ]
    reviews = [
        item
        for item in comparison_results
        if item.get("decision") == "unknown" or item.get("needs_review", False)
    ]
    remaining_template = list(graphic.get("remaining_unmatched_template") or [])
    remaining_target = list(graphic.get("remaining_unmatched_target") or [])
    return {
        "mismatch_count": len(mismatches),
        "review_count": len(reviews),
        "remaining_unmatched_template_count": len(remaining_template),
        "remaining_unmatched_target_count": len(remaining_target),
        "mismatches": mismatches,
        "reviews": reviews,
        "remaining_unmatched_template": remaining_template,
        "remaining_unmatched_target": remaining_target,
    }


def normalize_region_kind(kind: Any) -> str:
    if kind == "svg":
        return "graphic"
    if kind in {"text", "graphic"}:
        return str(kind)
    return str(kind or "unknown")


def normalize_box(box: Any) -> Optional[List[float]]:
    if box is None:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compact_box(box: Sequence[float]) -> List[float | int]:
    compact: List[float | int] = []
    for value in box:
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-6:
            compact.append(int(rounded))
        else:
            compact.append(round(float(value), 3))
    return compact


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    intersection = box_intersection_area(box_a, box_b)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def box_match_metrics(
    expected_box: Sequence[float],
    predicted_box: Sequence[float],
) -> Dict[str, float]:
    intersection = box_intersection_area(expected_box, predicted_box)
    expected_area = box_area(expected_box)
    predicted_area = box_area(predicted_box)
    union = expected_area + predicted_area - intersection
    return {
        "iou": intersection / union if union > 0 else 0.0,
        "expected_coverage": intersection / expected_area if expected_area > 0 else 0.0,
        "predicted_coverage": intersection / predicted_area if predicted_area > 0 else 0.0,
        "area_ratio": predicted_area / expected_area if expected_area > 0 else 0.0,
    }


def is_box_match(
    metrics: Dict[str, float],
    iou_threshold: float,
    coverage_threshold: float,
    max_area_ratio: float,
) -> bool:
    if metrics["iou"] >= iou_threshold:
        return True
    return (
        metrics["expected_coverage"] >= coverage_threshold
        and metrics["predicted_coverage"] >= coverage_threshold
        and metrics["area_ratio"] <= max_area_ratio
    )


def expected_visual_boxes(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for index, region in enumerate(sample.get("change_regions") or []):
        box = normalize_box(region.get("pixel_bbox"))
        if box is None:
            continue
        boxes.append(
            {
                "id": f"gt_{index}",
                "kind": normalize_region_kind(region.get("kind")),
                "source_kind": region.get("kind"),
                "box": compact_box(box),
                "label": region.get("label"),
                "strategy": region.get("strategy"),
                "original_text": region.get("original_text"),
                "mutated_text": region.get("mutated_text"),
            }
        )
    return boxes


def predicted_visual_boxes(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    annotations = list(result.get("visualization_annotations") or [])
    if annotations:
        for index, annotation in enumerate(annotations):
            box = normalize_box(annotation.get("box"))
            if box is None:
                continue
            boxes.append(
                {
                    "id": f"pred_{index}",
                    "kind": normalize_region_kind(annotation.get("kind")),
                    "status": annotation.get("status"),
                    "decision": annotation.get("decision"),
                    "box": compact_box(box),
                    "field": annotation.get("field"),
                    "summary": annotation.get("summary"),
                    "ocr_text": annotation.get("ocr_text"),
                }
            )
        return boxes

    final_boxes = list(
        result.get("final_boxes")
        or result.get("candidate_boxes_final")
        or result.get("candidate_boxes_scaled")
        or []
    )
    for index, item in enumerate(final_boxes):
        annotation = dict(item or {})
        chosen_box = normalize_box(annotation.get("display_box") or annotation.get("box"))
        if chosen_box is None:
            continue
        source_box = normalize_box(annotation.get("box"))
        display_box = normalize_box(annotation.get("display_box"))
        boxes.append(
            {
                "id": f"pred_{index}",
                "kind": normalize_region_kind(annotation.get("kind") or "graphic"),
                "status": annotation.get("status") or annotation.get("vlm_decision"),
                "decision": annotation.get("decision") or annotation.get("vlm_decision"),
                "box": compact_box(chosen_box),
                "source_box": compact_box(source_box) if source_box else None,
                "display_box": compact_box(display_box) if display_box else None,
                "field": annotation.get("field"),
                "summary": annotation.get("summary") or annotation.get("vlm_reason"),
                "ocr_text": annotation.get("ocr_text"),
            }
        )
    return boxes


def evaluate_visual_boxes(
    expected_boxes: Sequence[Dict[str, Any]],
    predicted_boxes: Sequence[Dict[str, Any]],
    iou_threshold: float,
    coverage_threshold: float,
    max_area_ratio: float,
    *,
    match_mode: str = "strict",
) -> Dict[str, Any]:
    if match_mode == "coverage":
        matched_expected: set[int] = set()
        covering_predicted: set[int] = set()
        matches: List[Dict[str, Any]] = []

        for expected_index, expected_box in enumerate(expected_boxes):
            best_candidate: Optional[tuple[float, int, Dict[str, float]]] = None
            for predicted_index, predicted_box in enumerate(predicted_boxes):
                metrics = box_match_metrics(expected_box["box"], predicted_box["box"])
                if metrics["expected_coverage"] < coverage_threshold:
                    continue
                covering_predicted.add(predicted_index)
                score = metrics["expected_coverage"] + 0.01 * metrics["iou"]
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, predicted_index, metrics)

            if best_candidate is None:
                continue

            matched_expected.add(expected_index)
            _, predicted_index, metrics = best_candidate
            expected_box = expected_boxes[expected_index]
            predicted_box = predicted_boxes[predicted_index]
            matches.append(
                {
                    "expected_id": expected_box["id"],
                    "predicted_id": predicted_box["id"],
                    "expected_kind": expected_box.get("kind"),
                    "predicted_kind": predicted_box.get("kind"),
                    "kind_match": expected_box.get("kind") == predicted_box.get("kind"),
                    "expected_box": expected_box["box"],
                    "predicted_box": predicted_box["box"],
                    "iou": round(float(metrics["iou"]), 4),
                    "expected_coverage": round(float(metrics["expected_coverage"]), 4),
                    "predicted_coverage": round(float(metrics["predicted_coverage"]), 4),
                    "area_ratio": round(float(metrics["area_ratio"]), 4),
                }
            )

        missing_expected = [
            box for index, box in enumerate(expected_boxes) if index not in matched_expected
        ]
        extra_predicted = [
            box for index, box in enumerate(predicted_boxes) if index not in covering_predicted
        ]
        return {
            "policy": {
                "match_mode": match_mode,
                "iou_threshold": iou_threshold,
                "coverage_threshold": coverage_threshold,
                "max_area_ratio": max_area_ratio,
            },
            "expected_boxes": list(expected_boxes),
            "predicted_boxes": list(predicted_boxes),
            "matches": matches,
            "missing_expected": missing_expected,
            "extra_predicted": extra_predicted,
            "expected_count": len(expected_boxes),
            "predicted_count": len(predicted_boxes),
            "matched_count": len(matches),
            "missing_count": len(missing_expected),
            "extra_count": len(extra_predicted),
        }

    candidates: List[tuple[float, float, float, int, int, Dict[str, float]]] = []
    for expected_index, expected_box in enumerate(expected_boxes):
        for predicted_index, predicted_box in enumerate(predicted_boxes):
            metrics = box_match_metrics(expected_box["box"], predicted_box["box"])
            if not is_box_match(metrics, iou_threshold, coverage_threshold, max_area_ratio):
                continue
            score = metrics["iou"] + 0.01 * min(
                metrics["expected_coverage"],
                metrics["predicted_coverage"],
            )
            candidates.append(
                (
                    score,
                    metrics["expected_coverage"],
                    metrics["predicted_coverage"],
                    expected_index,
                    predicted_index,
                    metrics,
                )
            )

    candidates.sort(reverse=True)
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    matches: List[Dict[str, Any]] = []

    for _, _, _, expected_index, predicted_index, metrics in candidates:
        if expected_index in matched_expected or predicted_index in matched_predicted:
            continue
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
        expected_box = expected_boxes[expected_index]
        predicted_box = predicted_boxes[predicted_index]
        matches.append(
            {
                "expected_id": expected_box["id"],
                "predicted_id": predicted_box["id"],
                "expected_kind": expected_box.get("kind"),
                "predicted_kind": predicted_box.get("kind"),
                "kind_match": expected_box.get("kind") == predicted_box.get("kind"),
                "expected_box": expected_box["box"],
                "predicted_box": predicted_box["box"],
                "iou": round(float(metrics["iou"]), 4),
                "expected_coverage": round(float(metrics["expected_coverage"]), 4),
                "predicted_coverage": round(float(metrics["predicted_coverage"]), 4),
                "area_ratio": round(float(metrics["area_ratio"]), 4),
            }
        )

    missing_expected = [
        box for index, box in enumerate(expected_boxes) if index not in matched_expected
    ]
    extra_predicted = [
        box for index, box in enumerate(predicted_boxes) if index not in matched_predicted
    ]
    return {
        "policy": {
            "match_mode": match_mode,
            "iou_threshold": iou_threshold,
            "coverage_threshold": coverage_threshold,
            "max_area_ratio": max_area_ratio,
        },
        "expected_boxes": list(expected_boxes),
        "predicted_boxes": list(predicted_boxes),
        "matches": matches,
        "missing_expected": missing_expected,
        "extra_predicted": extra_predicted,
        "expected_count": len(expected_boxes),
        "predicted_count": len(predicted_boxes),
        "matched_count": len(matches),
        "missing_count": len(missing_expected),
        "extra_count": len(extra_predicted),
    }


def evaluate_closed_loop_boxes(
    expected_boxes: Sequence[Dict[str, Any]],
    predicted_boxes: Sequence[Dict[str, Any]],
    *,
    text_coverage_threshold: float,
    iou_threshold: float,
    coverage_threshold: float,
    max_area_ratio: float,
) -> Dict[str, Any]:
    """Match generated GT boxes against final user-visible boxes.

    Text mutations are span-level GT, while the detector intentionally displays
    mapped PDF text-line boxes. The closed-loop signal is whether the final
    user-visible box covers the generated GT region.
    """
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    covering_predicted: set[int] = set()
    matches: List[Dict[str, Any]] = []

    for expected_index, expected_box in enumerate(expected_boxes):
        expected_kind = normalize_region_kind(expected_box.get("kind"))
        best_candidate: Optional[tuple[float, int, Dict[str, float]]] = None
        for predicted_index, predicted_box in enumerate(predicted_boxes):
            metrics = box_match_metrics(expected_box["box"], predicted_box["box"])
            if expected_kind == "text":
                is_match = metrics["expected_coverage"] >= text_coverage_threshold
                score = metrics["expected_coverage"] + 0.01 * metrics["iou"]
            else:
                is_match = metrics["expected_coverage"] >= coverage_threshold
                score = metrics["expected_coverage"] + 0.01 * metrics["iou"]
            if not is_match:
                continue
            covering_predicted.add(predicted_index)
            if predicted_index not in matched_predicted and (
                best_candidate is None or score > best_candidate[0]
            ):
                best_candidate = (score, predicted_index, metrics)

        if best_candidate is None:
            continue

        matched_expected.add(expected_index)
        _, predicted_index, metrics = best_candidate
        matched_predicted.add(predicted_index)
        predicted_box = predicted_boxes[predicted_index]
        matches.append(
            {
                "expected_id": expected_box["id"],
                "predicted_id": predicted_box["id"],
                "expected_kind": expected_kind,
                "predicted_kind": predicted_box.get("kind"),
                "kind_match": expected_kind == normalize_region_kind(predicted_box.get("kind")),
                "expected_box": expected_box["box"],
                "predicted_box": predicted_box["box"],
                "iou": round(float(metrics["iou"]), 4),
                "expected_coverage": round(float(metrics["expected_coverage"]), 4),
                "predicted_coverage": round(float(metrics["predicted_coverage"]), 4),
                "area_ratio": round(float(metrics["area_ratio"]), 4),
            }
        )

    missing_expected = [
        box for index, box in enumerate(expected_boxes) if index not in matched_expected
    ]
    extra_predicted = [
        box for index, box in enumerate(predicted_boxes) if index not in covering_predicted
    ]
    return {
        "policy": {
            "match_mode": "closed_loop",
            "text_match_mode": "gt_span_coverage",
            "text_coverage_threshold": text_coverage_threshold,
            "graphic_match_mode": "gt_coverage",
            "graphic_iou_threshold": iou_threshold,
            "graphic_coverage_threshold": coverage_threshold,
            "graphic_max_area_ratio": max_area_ratio,
        },
        "expected_boxes": list(expected_boxes),
        "predicted_boxes": list(predicted_boxes),
        "matches": matches,
        "missing_expected": missing_expected,
        "extra_predicted": extra_predicted,
        "expected_count": len(expected_boxes),
        "predicted_count": len(predicted_boxes),
        "matched_count": len(matches),
        "missing_count": len(missing_expected),
        "extra_count": len(extra_predicted),
        "text_expected_count": sum(
            1 for item in expected_boxes if normalize_region_kind(item.get("kind")) == "text"
        ),
        "text_matched_count": sum(
            1 for item in matches if normalize_region_kind(item.get("expected_kind")) == "text"
        ),
        "graphic_expected_count": sum(
            1 for item in expected_boxes if normalize_region_kind(item.get("kind")) == "graphic"
        ),
        "graphic_matched_count": sum(
            1 for item in matches if normalize_region_kind(item.get("expected_kind")) == "graphic"
        ),
    }


def evaluate_result(
    case: SyntheticCase,
    raw_result: Dict[str, Any],
    box_iou_threshold: float,
    box_coverage_threshold: float,
    max_box_area_ratio: float,
) -> Dict[str, Any]:
    expected = expected_flags(case.manifest_sample)
    different_fields = text_different_fields(raw_result)
    label_union_diffs = detected_label_union_diffs(raw_result)
    graphic_buckets = graphic_result_buckets(raw_result)
    expected_boxes = expected_visual_boxes(case.manifest_sample)
    predicted_boxes = predicted_visual_boxes(raw_result)
    traditional_flow = bool(
        raw_result.get("pipeline") == "traditional_full_image_diff"
        or (
            not raw_result.get("visualization_annotations")
            and (raw_result.get("final_boxes") or raw_result.get("candidate_boxes_final"))
        )
    )
    if traditional_flow:
        box_evaluation = evaluate_closed_loop_boxes(
            expected_boxes,
            predicted_boxes,
            text_coverage_threshold=min(0.3, box_coverage_threshold),
            iou_threshold=box_iou_threshold,
            coverage_threshold=box_coverage_threshold,
            max_area_ratio=max_box_area_ratio,
        )
    else:
        box_evaluation = evaluate_visual_boxes(
            expected_boxes,
            predicted_boxes,
            iou_threshold=box_iou_threshold,
            coverage_threshold=box_coverage_threshold,
            max_area_ratio=max_box_area_ratio,
            match_mode="strict",
        )

    actual_text_diff = bool(different_fields)
    if traditional_flow:
        actual_graphic_diff = bool(predicted_boxes)
    else:
        actual_graphic_diff = bool(
            graphic_buckets["mismatch_count"]
            or graphic_buckets["review_count"]
            or graphic_buckets["remaining_unmatched_template_count"]
            or graphic_buckets["remaining_unmatched_target_count"]
        )

    issues: List[str] = []
    if box_evaluation["missing_count"]:
        issues.append("missing_box")

    warnings: List[str] = []
    if (
        not raw_result.get("visualization_annotations")
        and not raw_result.get("final_boxes")
        and not raw_result.get("candidate_boxes_final")
    ):
        warnings.append("no_visualization_annotations")
    if expected["text_diff"] and not actual_text_diff:
        warnings.append("text_content_miss")
    if not expected["text_diff"] and actual_text_diff:
        warnings.append("text_content_false_positive")
    if expected["graphic_diff"] and not actual_graphic_diff:
        warnings.append("graphic_content_miss")
    if not expected["graphic_diff"] and actual_graphic_diff:
        warnings.append("graphic_content_false_positive")
    if any(not match["kind_match"] for match in box_evaluation["matches"]):
        warnings.append("matched_box_kind_mismatch")

    return {
        "evaluation_schema": "box_v1",
        "sample_id": case.sample_id,
        "defect_source": case.defect_source,
        "expected": {
            **expected,
            "box_count": len(expected_boxes),
        },
        "actual": {
            "text_diff": actual_text_diff,
            "graphic_diff": actual_graphic_diff,
            "box_count": len(predicted_boxes),
            "text_different_fields": different_fields,
            "label_union_differences": label_union_diffs,
            "graphic_mismatch_count": graphic_buckets["mismatch_count"],
            "graphic_review_count": graphic_buckets["review_count"],
            "remaining_unmatched_template_count": graphic_buckets[
                "remaining_unmatched_template_count"
            ],
            "remaining_unmatched_target_count": graphic_buckets[
                "remaining_unmatched_target_count"
            ],
            "verdict": raw_result.get("verdict"),
        },
        "box_evaluation": box_evaluation,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def case_output_dir(output_root: Path, case: SyntheticCase) -> Path:
    return output_root / "cases" / case.sample_id


def load_existing_evaluation(output_root: Path, case: SyntheticCase) -> Optional[Dict[str, Any]]:
    result_path = case_output_dir(output_root, case) / RESULT_JSON_NAME
    if not result_path.exists():
        return None
    existing = read_json(result_path)
    if existing.get("evaluation_schema") != EVALUATION_SCHEMA:
        return None
    return existing


def copy_compare_image(case: SyntheticCase, output_dir: Path) -> Optional[Path]:
    if case.compare_path is None or not case.compare_path.exists():
        return None
    target = output_dir / case.compare_path.name
    shutil.copy2(case.compare_path, target)
    return target


def draw_ground_truth_boxes(
    target_image_path: Path,
    change_regions: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    image = cv2.imread(str(target_image_path))
    if image is None:
        return None

    for index, region in enumerate(change_regions):
        box = normalize_box(region.get("pixel_bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(image.shape[1] - 1, x1))
        y1 = max(0, min(image.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(image.shape[1], x2))
        y2 = max(y1 + 1, min(image.shape[0], y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            image,
            f"GT{index + 1}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        return None
    return output_path


def copy_ground_truth_images(
    case: SyntheticCase,
    output_dir: Path,
) -> Dict[str, Optional[Path]]:
    copied_preview = None
    if case.ground_truth_path and case.ground_truth_path.exists():
        copied_preview = output_dir / DATASET_PREVIEW_IMAGE_NAME
        shutil.copy2(case.ground_truth_path, copied_preview)

    generated = draw_ground_truth_boxes(
        case.image_path,
        list(case.manifest_sample.get("change_regions") or []),
        output_dir / GROUND_TRUTH_BOXES_IMAGE_NAME,
    )
    return {
        "ground_truth_boxes": generated,
        "dataset_preview": copied_preview,
    }


def ensure_case_ground_truth_artifacts(
    case: SyntheticCase,
    output_root: Path,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = case_output_dir(output_root, case)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(evaluation.get("paths") or {})
    needs_ground_truth = not paths.get("ground_truth_boxes") or not Path(
        str(paths.get("ground_truth_boxes"))
    ).exists()
    needs_dataset_preview = bool(case.ground_truth_path) and (
        not paths.get("dataset_preview")
        or not Path(str(paths.get("dataset_preview"))).exists()
    )
    if not needs_ground_truth and not needs_dataset_preview:
        return evaluation

    ground_truth_images = copy_ground_truth_images(case, output_dir)
    if ground_truth_images.get("ground_truth_boxes"):
        paths["ground_truth_boxes"] = str(ground_truth_images["ground_truth_boxes"])
    if ground_truth_images.get("dataset_preview"):
        paths["dataset_preview"] = str(ground_truth_images["dataset_preview"])
    evaluation = dict(evaluation)
    evaluation["paths"] = paths
    write_json(output_dir / RESULT_JSON_NAME, evaluation)
    return evaluation


def _read_display_image(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return None
    return image


def _placeholder_image(label: str, *, height: int = 540, width: int = 540) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    cv2.putText(
        image,
        "missing",
        (24, max(40, height // 2 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (24, max(76, height // 2 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )
    return image


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return _placeholder_image("invalid")
    if height == target_height:
        return image
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _build_failure_comparison_image(
    template_image: Optional[np.ndarray],
    target_image: Optional[np.ndarray],
    diff_image: Optional[np.ndarray],
    *,
    labels: Sequence[str] = ("template", "target", "diff"),
    target_height: int = 540,
    panel_padding: int = 12,
    label_height: int = 28,
    gap: int = 16,
) -> Optional[np.ndarray]:
    images = []
    for label, image in zip(labels, (template_image, target_image, diff_image), strict=True):
        candidate = image if image is not None else _placeholder_image(label, height=target_height)
        candidate = _resize_to_height(candidate, target_height)
        panel = np.full(
            (target_height + label_height + panel_padding * 2, candidate.shape[1] + panel_padding * 2, 3),
            255,
            dtype=np.uint8,
        )
        cv2.putText(
            panel,
            label,
            (panel_padding, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        y0 = label_height + panel_padding
        x0 = panel_padding
        panel[y0 : y0 + candidate.shape[0], x0 : x0 + candidate.shape[1]] = candidate
        cv2.rectangle(
            panel,
            (x0, y0),
            (x0 + candidate.shape[1] - 1, y0 + candidate.shape[0] - 1),
            (180, 180, 180),
            1,
        )
        images.append(panel)

    if not images:
        return None

    stitched: list[np.ndarray] = []
    for index, panel in enumerate(images):
        if index > 0:
            stitched.append(np.full((panel.shape[0], gap, 3), 255, dtype=np.uint8))
        stitched.append(panel)
    return np.hstack(stitched)


def build_representative_failure_artifact(
    output_root: Path,
    evaluations: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    failed = [item for item in evaluations if not item.get("passed")]
    artifact_dir = output_root / REPRESENTATIVE_FAILURE_DIR_NAME
    if not failed:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        return None

    chosen = failed[0]
    paths = dict(chosen.get("paths") or {})
    artifact_dir.mkdir(parents=True, exist_ok=True)

    template_source = next(
        (
            Path(str(value))
            for value in (paths.get("template_preview"), paths.get("template"))
            if value and Path(str(value)).exists()
        ),
        None,
    )
    target_source = Path(str(paths.get("target"))) if paths.get("target") else None
    if target_source is not None and not target_source.exists():
        target_source = None
    diff_source = Path(str(paths.get("visualization_diff"))) if paths.get("visualization_diff") else None
    if diff_source is not None and not diff_source.exists():
        diff_source = None

    copied_template = None
    if template_source is not None:
        copied_template = artifact_dir / REPRESENTATIVE_FAILURE_TEMPLATE_NAME
        shutil.copy2(template_source, copied_template)

    copied_target = None
    if target_source is not None:
        copied_target = artifact_dir / REPRESENTATIVE_FAILURE_TARGET_NAME
        shutil.copy2(target_source, copied_target)

    copied_diff = None
    if diff_source is not None:
        copied_diff = artifact_dir / REPRESENTATIVE_FAILURE_DIFF_NAME
        shutil.copy2(diff_source, copied_diff)

    comparison = _build_failure_comparison_image(
        _read_display_image(copied_template or template_source),
        _read_display_image(copied_target or target_source),
        _read_display_image(copied_diff or diff_source),
    )
    comparison_path = artifact_dir / REPRESENTATIVE_FAILURE_IMAGE_NAME
    if comparison is not None:
        cv2.imwrite(str(comparison_path), comparison)
    else:
        comparison_path = copied_diff or copied_target or copied_template or comparison_path

    artifact = {
        "sample_id": chosen.get("sample_id"),
        "defect_source": chosen.get("defect_source"),
        "passed": chosen.get("passed"),
        "issues": list(chosen.get("issues") or []),
        "warnings": list(chosen.get("warnings") or []),
        "duration_seconds": chosen.get("duration_seconds"),
        "box_counts": dict(chosen.get("box_evaluation") or {}),
        "paths": {
            "artifact_dir": str(artifact_dir),
            "template": str(copied_template) if copied_template else None,
            "target": str(copied_target) if copied_target else None,
            "visualization_diff": str(copied_diff) if copied_diff else None,
            "comparison_image": str(comparison_path) if comparison_path else None,
        },
    }
    write_json(artifact_dir / "metadata.json", artifact)
    return artifact


def run_case(
    case: SyntheticCase,
    template_path: Path,
    output_root: Path,
    output_mode: str,
    desktop_pipeline: str,
    box_iou_threshold: float,
    box_coverage_threshold: float,
    max_box_area_ratio: float,
) -> Dict[str, Any]:
    output_dir = case_output_dir(output_root, case)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    started = time.time()
    case_template_path = case.template_path or template_path
    detection_started = time.time()
    desktop_result = run_desktop_detection(
        case,
        template_path=case_template_path,
        output_dir=output_dir,
        output_mode=output_mode,
        desktop_pipeline=desktop_pipeline,
    )
    detection_duration_seconds = round(time.time() - detection_started, 3)
    final_result_path = Path(desktop_result.output_dir) / "final_result.json"
    if final_result_path.exists():
        raw_result = read_json(final_result_path)
    else:
        raw_result = dict(desktop_result.raw_result or {})

    evaluation = evaluate_result(
        case,
        raw_result,
        box_iou_threshold=box_iou_threshold,
        box_coverage_threshold=box_coverage_threshold,
        max_box_area_ratio=max_box_area_ratio,
    )
    compare_copy = copy_compare_image(case, output_dir)
    ground_truth_images = copy_ground_truth_images(case, output_dir)
    visualization = desktop_result.visualization_path or (
        Path(desktop_result.output_dir) / "visualization_diff.jpg"
    )
    duration_seconds = round(time.time() - started, 3)
    evaluation.update(
        {
            "evaluation_schema": EVALUATION_SCHEMA,
            "success": bool(desktop_result.success),
            "desktop_pipeline": desktop_pipeline,
            "duration_seconds": duration_seconds,
            "detection_duration_seconds": detection_duration_seconds,
            "stability": {
                "stable": case.stable,
                "issues": list(case.stability_issues),
            },
            "paths": {
                "template": str(case_template_path),
                "template_preview": (
                    str(case.template_preview_path)
                    if case.template_preview_path
                    else None
                ),
                "target": str(case.image_path),
                "compare": str(compare_copy) if compare_copy else None,
                "ground_truth_boxes": (
                    str(ground_truth_images["ground_truth_boxes"])
                    if ground_truth_images.get("ground_truth_boxes")
                    else None
                ),
                "dataset_preview": (
                    str(ground_truth_images["dataset_preview"])
                    if ground_truth_images.get("dataset_preview")
                    else None
                ),
                "raw_result": str(final_result_path) if final_result_path.exists() else None,
                "desktop_output_dir": str(desktop_result.output_dir),
                "visualization_diff": (
                    str(visualization)
                    if visualization and visualization.exists()
                    else None
                ),
            },
            "desktop_result": {
                "verdict": desktop_result.verdict,
                "summary_text": desktop_result.summary_text,
                "text_match_count": desktop_result.text_match_count,
                "text_total_count": desktop_result.text_total_count,
                "graphic_match_count": desktop_result.graphic_match_count,
                "graphic_mismatch_count": desktop_result.graphic_mismatch_count,
                "graphic_review_count": desktop_result.graphic_review_count,
                "error": desktop_result.error,
            },
        }
    )
    write_json(output_dir / RESULT_JSON_NAME, evaluation)
    return evaluation


def summarize(evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, int]] = {}
    issue_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    verdict_counts: Dict[str, int] = {}
    durations = [
        float(item.get("duration_seconds"))
        for item in evaluations
        if item.get("duration_seconds") is not None
    ]
    detection_durations = [
        float(item.get("detection_duration_seconds"))
        for item in evaluations
        if item.get("detection_duration_seconds") is not None
    ]
    box_stats = {
        "expected": 0,
        "predicted": 0,
        "matched": 0,
        "missing": 0,
        "extra": 0,
    }
    for item in evaluations:
        source = str(item.get("defect_source") or "unknown")
        bucket = by_source.setdefault(source, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if item.get("passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        for issue in item.get("issues") or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for warning in item.get("warnings") or []:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        verdict = str(
            item.get("actual", {}).get("verdict")
            or item.get("desktop_result", {}).get("verdict")
            or "unknown"
        )
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        box_evaluation = dict(item.get("box_evaluation") or {})
        box_stats["expected"] += int(box_evaluation.get("expected_count") or 0)
        box_stats["predicted"] += int(box_evaluation.get("predicted_count") or 0)
        box_stats["matched"] += int(box_evaluation.get("matched_count") or 0)
        box_stats["missing"] += int(box_evaluation.get("missing_count") or 0)
        box_stats["extra"] += int(box_evaluation.get("extra_count") or 0)

    passed = sum(1 for item in evaluations if item.get("passed"))
    total = len(evaluations)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "duration_seconds": {
                "average": round(mean(durations), 3) if durations else 0.0,
                "median": round(median(durations), 3) if durations else 0.0,
            },
            "detection_duration_seconds": {
                "average": (
                    round(mean(detection_durations), 3) if detection_durations else 0.0
                ),
                "median": (
                    round(median(detection_durations), 3) if detection_durations else 0.0
                ),
            },
            "verdict_distribution": dict(sorted(verdict_counts.items())),
            "by_defect_source": by_source,
            "issue_counts": dict(sorted(issue_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
            "box_stats": box_stats,
        },
        "cases": [
            {
                "sample_id": item.get("sample_id"),
                "defect_source": item.get("defect_source"),
                "passed": item.get("passed"),
                "issues": item.get("issues"),
                "warnings": item.get("warnings"),
                "expected": {
                    "text_diff": item.get("expected", {}).get("text_diff"),
                    "graphic_diff": item.get("expected", {}).get("graphic_diff"),
                    "box_count": item.get("expected", {}).get("box_count"),
                },
                "actual": {
                    "text_diff": item.get("actual", {}).get("text_diff"),
                    "graphic_diff": item.get("actual", {}).get("graphic_diff"),
                    "box_count": item.get("actual", {}).get("box_count"),
                    "text_different_fields": item.get("actual", {}).get("text_different_fields"),
                    "graphic_mismatch_count": item.get("actual", {}).get("graphic_mismatch_count"),
                    "graphic_review_count": item.get("actual", {}).get("graphic_review_count"),
                    "verdict": item.get("actual", {}).get("verdict"),
                },
                "box_counts": {
                    "expected": item.get("box_evaluation", {}).get("expected_count"),
                    "predicted": item.get("box_evaluation", {}).get("predicted_count"),
                    "matched": item.get("box_evaluation", {}).get("matched_count"),
                    "missing": item.get("box_evaluation", {}).get("missing_count"),
                    "extra": item.get("box_evaluation", {}).get("extra_count"),
                },
                "duration_seconds": item.get("duration_seconds"),
                "detection_duration_seconds": item.get("detection_duration_seconds"),
                "desktop_pipeline": item.get("desktop_pipeline"),
                "paths": item.get("paths"),
            }
            for item in evaluations
        ],
    }


def print_plan(template_path: Path, cases: Sequence[SyntheticCase], output_dir: Path) -> None:
    print("=" * 70)
    print("Synthetic dataset evaluation")
    print("=" * 70)
    print(f"Template: {template_path}")
    print(f"Output:   {output_dir}")
    print(f"Cases:    {len(cases)}")
    for index, case in enumerate(cases, start=1):
        print(
            f"[plan {index:03d}] {case.sample_id} "
            f"source={case.defect_source} target={case.image_path.name}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    template_path, all_cases, manifest = load_cases(dataset_root)
    template_preview_path = resolve_template_preview_path(dataset_root, template_path, manifest)
    all_cases = annotate_case_stability(
        all_cases,
        template_path,
        template_preview_path=template_preview_path,
        diff_threshold=args.gt_diff_threshold,
        min_diff_ratio=args.gt_min_diff_ratio,
    )
    cases = filter_cases(
        all_cases,
        sample_ids=args.sample_id,
        defect_sources=args.defect_source,
        limit=args.limit,
        include_unstable_gt=args.include_unstable_gt,
    )
    skipped_unstable_count = sum(1 for case in all_cases if not case.stable)
    print_plan(template_path, cases, output_dir)
    if skipped_unstable_count and not args.include_unstable_gt:
        print(f"Filtered unstable GT cases: {skipped_unstable_count}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if args.skip_existing:
            existing = load_existing_evaluation(output_dir, case)
            if existing is not None:
                print(f"[skip {index:03d}/{len(cases):03d}] {case.sample_id}")
                existing = ensure_case_ground_truth_artifacts(case, output_dir, existing)
                evaluations.append(existing)
                continue

        print(f"[run  {index:03d}/{len(cases):03d}] {case.sample_id} ({case.defect_source})")
        case_started = time.time()
        try:
            evaluation = run_case(
                case,
                template_path=template_path,
                output_root=output_dir,
                output_mode=args.output_mode,
                desktop_pipeline=args.desktop_pipeline,
                box_iou_threshold=args.box_iou_threshold,
                box_coverage_threshold=args.box_coverage_threshold,
                max_box_area_ratio=args.max_box_area_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            evaluation = {
                "sample_id": case.sample_id,
                "defect_source": case.defect_source,
                "success": False,
                "desktop_pipeline": args.desktop_pipeline,
                "passed": False,
                "issues": ["run_error"],
                "warnings": [],
                "error": str(exc),
                "duration_seconds": round(time.time() - case_started, 3),
                "detection_duration_seconds": round(time.time() - case_started, 3),
                "actual": {"verdict": None},
                "stability": {
                    "stable": case.stable,
                    "issues": list(case.stability_issues),
                },
                "paths": {
                    "template": str(case.template_path or template_path),
                    "template_preview": (
                        str(case.template_preview_path)
                        if case.template_preview_path
                        else None
                    ),
                    "target": str(case.image_path),
                },
            }
            case_dir = case_output_dir(output_dir, case)
            write_json(case_dir / RESULT_JSON_NAME, evaluation)
        evaluations.append(evaluation)
        status = "PASS" if evaluation.get("passed") else "FAIL"
        box_counts = dict(evaluation.get("box_evaluation") or {})
        print(
            f"       {status} issues={evaluation.get('issues') or []} "
            f"warnings={evaluation.get('warnings') or []} "
            f"boxes={box_counts.get('matched_count', 0)}/"
            f"{box_counts.get('expected_count', 0)} "
            f"pred={box_counts.get('predicted_count', 0)} "
            f"missing={box_counts.get('missing_count', 0)} "
            f"extra={box_counts.get('extra_count', 0)}"
        )

    summary = summarize(evaluations)
    representative_failure = build_representative_failure_artifact(output_dir, evaluations)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "dataset_name": manifest.get("dataset_name"),
            "template": str(template_path),
            "template_preview": str(template_preview_path) if template_preview_path else None,
            "output_dir": str(output_dir),
            "output_mode": args.output_mode,
            "desktop_pipeline": args.desktop_pipeline,
            "gt_stability_filter": {
                "enabled": not args.include_unstable_gt,
                "diff_threshold": args.gt_diff_threshold,
                "min_diff_ratio": args.gt_min_diff_ratio,
                "filtered_unstable_count": skipped_unstable_count
                if not args.include_unstable_gt
                else 0,
                "unstable_cases": [
                    {
                        "sample_id": case.sample_id,
                        "issues": list(case.stability_issues),
                    }
                    for case in all_cases
                    if not case.stable
                ],
            },
            "box_match_policy": {
                "iou_threshold": args.box_iou_threshold,
                "coverage_threshold": args.box_coverage_threshold,
                "max_area_ratio": args.max_box_area_ratio,
            },
            "representative_failure": representative_failure,
        }
    )
    summary_path = output_dir / SUMMARY_JSON_NAME
    write_json(summary_path, summary)

    print("=" * 70)
    print(f"Summary saved: {summary_path}")
    print(
        "Result: "
        f"passed={summary['stats']['passed']} "
        f"failed={summary['stats']['failed']} "
        f"total={summary['stats']['total']} "
        f"boxes={summary['stats']['box_stats']}"
    )
    if representative_failure:
        print(
            "Representative failure: "
            f"{representative_failure['sample_id']} -> "
            f"{representative_failure['paths']['comparison_image']}"
        )
    return 0 if summary["stats"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
