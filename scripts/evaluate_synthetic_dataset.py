"""Evaluate the detection workflow on a manifest-driven synthetic dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import PROJECT_ROOT as REPO_ROOT
from label_detection.matching.ocr import text_field_values_match


DEFAULT_DATASET_ROOT = Path("/home/data/数据集/outputs/datasets/label_aug_mixed_20260406_200")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "synthetic_dataset_eval"

RESULT_JSON_NAME = "evaluation_result.json"
SUMMARY_JSON_NAME = "summary.json"


@dataclass(frozen=True)
class SyntheticCase:
    sample_id: str
    defect_source: str
    image_path: Path
    compare_path: Optional[Path]
    manifest_sample: Dict[str, Any]


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


def resolve_dataset_path(dataset_root: Path, windows_or_relative_path: Optional[str]) -> Optional[Path]:
    if not windows_or_relative_path:
        return None
    candidate = Path(windows_or_relative_path)
    if candidate.exists():
        return candidate
    return dataset_root / candidate.name


def load_cases(dataset_root: Path) -> tuple[Path, List[SyntheticCase], Dict[str, Any]]:
    visual_manifest_path = dataset_root / "visual_manifest.json"
    if not visual_manifest_path.exists():
        raise FileNotFoundError(f"visual_manifest.json not found: {visual_manifest_path}")

    manifest = read_json(visual_manifest_path)
    template_path = dataset_root / "template" / "template.png"
    if not template_path.exists():
        raise FileNotFoundError(f"template image not found: {template_path}")

    cases: List[SyntheticCase] = []
    for sample in manifest.get("samples") or []:
        sample_id = str(sample.get("sample_id") or "").strip()
        image_filename = str(sample.get("image_filename") or "").strip()
        if not sample_id or not image_filename:
            continue
        image_path = dataset_root / "images" / image_filename
        compare_path = dataset_root / "compare" / str(sample.get("compare_filename") or "")
        cases.append(
            SyntheticCase(
                sample_id=sample_id,
                defect_source=str(sample.get("defect_source") or ""),
                image_path=image_path,
                compare_path=compare_path if compare_path.exists() else None,
                manifest_sample=sample,
            )
        )

    cases.sort(key=lambda item: item.sample_id)
    return template_path, cases, manifest


def filter_cases(
    cases: Sequence[SyntheticCase],
    sample_ids: Optional[Sequence[str]],
    defect_sources: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[SyntheticCase]:
    selected = list(cases)
    if sample_ids:
        allowed = set(sample_ids)
        selected = [case for case in selected if case.sample_id in allowed]
    if defect_sources:
        allowed_sources = set(defect_sources)
        selected = [case for case in selected if case.defect_source in allowed_sources]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def get_detection_runner():
    from label_detection.workflows.unified import run_unified_detection

    return run_unified_detection


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


def box_match_metrics(expected_box: Sequence[float], predicted_box: Sequence[float]) -> Dict[str, float]:
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


def evaluate_visual_boxes(
    expected_boxes: Sequence[Dict[str, Any]],
    predicted_boxes: Sequence[Dict[str, Any]],
    iou_threshold: float,
    coverage_threshold: float,
    max_area_ratio: float,
) -> Dict[str, Any]:
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
    box_evaluation = evaluate_visual_boxes(
        expected_boxes,
        predicted_boxes,
        iou_threshold=box_iou_threshold,
        coverage_threshold=box_coverage_threshold,
        max_area_ratio=max_box_area_ratio,
    )

    actual_text_diff = bool(different_fields)
    actual_graphic_diff = bool(
        graphic_buckets["mismatch_count"]
        or graphic_buckets["review_count"]
        or graphic_buckets["remaining_unmatched_template_count"]
        or graphic_buckets["remaining_unmatched_target_count"]
    )

    issues: List[str] = []
    if box_evaluation["missing_count"]:
        issues.append("missing_box")
    if box_evaluation["extra_count"]:
        issues.append("extra_box")

    warnings: List[str] = []
    if "visualization_annotations" not in raw_result:
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
    if existing.get("evaluation_schema") != "box_v1":
        return None
    return existing


def copy_compare_image(case: SyntheticCase, output_dir: Path) -> Optional[Path]:
    if case.compare_path is None or not case.compare_path.exists():
        return None
    target = output_dir / case.compare_path.name
    shutil.copy2(case.compare_path, target)
    return target


def run_case(
    case: SyntheticCase,
    template_path: Path,
    output_root: Path,
    output_mode: str,
    box_iou_threshold: float,
    box_coverage_threshold: float,
    max_box_area_ratio: float,
) -> Dict[str, Any]:
    output_dir = case_output_dir(output_root, case)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    runner = get_detection_runner()
    raw_result = runner(
        str(template_path),
        str(case.image_path),
        output_dir=str(output_dir),
        output_mode=output_mode,
    )
    final_result_path = output_dir / "final_result.json"
    if final_result_path.exists():
        raw_result = read_json(final_result_path)
    else:
        raw_result = dict(raw_result or {})

    evaluation = evaluate_result(
        case,
        raw_result,
        box_iou_threshold=box_iou_threshold,
        box_coverage_threshold=box_coverage_threshold,
        max_box_area_ratio=max_box_area_ratio,
    )
    compare_copy = copy_compare_image(case, output_dir)
    visualization = output_dir / "visualization_diff.jpg"
    evaluation.update(
        {
            "success": bool(raw_result.get("success", True)),
            "duration_seconds": round(time.time() - started, 3),
            "paths": {
                "template": str(template_path),
                "target": str(case.image_path),
                "compare": str(compare_copy) if compare_copy else None,
                "raw_result": str(final_result_path) if final_result_path.exists() else None,
                "visualization_diff": str(visualization) if visualization.exists() else None,
            },
        }
    )
    write_json(output_dir / RESULT_JSON_NAME, evaluation)
    return evaluation


def summarize(evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, int]] = {}
    issue_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
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
        box_evaluation = dict(item.get("box_evaluation") or {})
        box_stats["expected"] += int(box_evaluation.get("expected_count") or 0)
        box_stats["predicted"] += int(box_evaluation.get("predicted_count") or 0)
        box_stats["matched"] += int(box_evaluation.get("matched_count") or 0)
        box_stats["missing"] += int(box_evaluation.get("missing_count") or 0)
        box_stats["extra"] += int(box_evaluation.get("extra_count") or 0)

    passed = sum(1 for item in evaluations if item.get("passed"))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {
            "total": len(evaluations),
            "passed": passed,
            "failed": len(evaluations) - passed,
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
    cases = filter_cases(
        all_cases,
        sample_ids=args.sample_id,
        defect_sources=args.defect_source,
        limit=args.limit,
    )
    print_plan(template_path, cases, output_dir)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if args.skip_existing:
            existing = load_existing_evaluation(output_dir, case)
            if existing is not None:
                print(f"[skip {index:03d}/{len(cases):03d}] {case.sample_id}")
                evaluations.append(existing)
                continue

        print(f"[run  {index:03d}/{len(cases):03d}] {case.sample_id} ({case.defect_source})")
        try:
            evaluation = run_case(
                case,
                template_path=template_path,
                output_root=output_dir,
                output_mode=args.output_mode,
                box_iou_threshold=args.box_iou_threshold,
                box_coverage_threshold=args.box_coverage_threshold,
                max_box_area_ratio=args.max_box_area_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            evaluation = {
                "sample_id": case.sample_id,
                "defect_source": case.defect_source,
                "success": False,
                "passed": False,
                "issues": ["run_error"],
                "warnings": [],
                "error": str(exc),
                "paths": {
                    "template": str(template_path),
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
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "dataset_name": manifest.get("dataset_name"),
            "template": str(template_path),
            "output_dir": str(output_dir),
            "output_mode": args.output_mode,
            "box_match_policy": {
                "iou_threshold": args.box_iou_threshold,
                "coverage_threshold": args.box_coverage_threshold,
                "max_area_ratio": args.max_box_area_ratio,
            },
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
    return 0 if summary["stats"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
