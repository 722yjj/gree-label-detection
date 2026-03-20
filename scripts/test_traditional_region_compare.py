"""Direct-run experiment for deterministic graphic region comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from label_detection.core.config import PROJECT_ROOT as REPO_ROOT
from label_detection.extraction.pdf import extract_red_box_info
from label_detection.matching.layout import (
    align_images_sift,
    compare_contours_hu,
    detect_layout_regions,
    draw_regions,
    extract_main_contours,
    extract_regions_by_type,
    match_regions,
    split_barcode_regions,
)
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template
from label_detection.services.ocr_service import get_ocr_with_boxes


# ==================== 直接在这里改测试参数 ====================
TEST_NAME = "traditional_region_compare"

PDF_PATH = REPO_ROOT / "samples" / "pdfs" / "600004075219-01.pdf"
TARGET_IMAGE_PATH = REPO_ROOT / "samples" / "images" / "produce" / "type1" / "600004075219_1.jpg"

USE_PROJECT_PREPROCESSING = True
REGION_TYPE = "image"
LAYOUT_THRESHOLD = 0.30
MATCH_COST_THRESHOLD = 0.60
PAIR_INDEX = None  # None 表示跑所有匹配对；填 0/1/2... 只跑指定区域对

OUTPUT_DIR = REPO_ROOT / "results" / "traditional_region_compare"

# ===== 实验阈值（先保守，主要用于观察，不直接写回主流程） =====
MATCH_FOREGROUND_IOU_MIN = 0.82
MATCH_XOR_RATIO_MAX = 0.08
MATCH_HU_MIN = 0.60
MATCH_COMPONENT_DELTA_MAX = 1

MISMATCH_FOREGROUND_IOU_MAX = 0.45
MISMATCH_XOR_RATIO_MIN = 0.22
MISMATCH_HU_MAX = 0.25
MISMATCH_COMPONENT_DELTA_MIN = 2


def prepare_test_images(output_dir: Path) -> Dict[str, Dict[str, object]]:
    """Prepare template and target images using the same preprocessing path as the project."""
    output_dir.mkdir(parents=True, exist_ok=True)

    template_asset_dir = output_dir / "template_assets"
    template_asset_dir.mkdir(parents=True, exist_ok=True)

    template_raw_path = extract_red_box_info(
        str(PDF_PATH),
        target_dpi=300,
        output_dir=str(template_asset_dir),
    )
    if not template_raw_path:
        raise RuntimeError(f"无法从 PDF 提取模板图: {PDF_PATH}")

    template_image, _, template_err = preprocess_template(
        template_raw_path,
        output_dir=str(output_dir / "template_preprocess"),
    )
    if template_err or template_image is None:
        raise RuntimeError(template_err or "模板预处理失败")

    if USE_PROJECT_PREPROCESSING:
        target_image, _, target_err = preprocess_target(
            str(TARGET_IMAGE_PATH),
            output_dir=str(output_dir / "target_preprocess"),
            template_image=template_image,
        )
        if target_err or target_image is None:
            raise RuntimeError(target_err or "目标图预处理失败")
        target_source = "target_preprocess"
    else:
        target_image = cv2.imread(str(TARGET_IMAGE_PATH))
        if target_image is None:
            raise RuntimeError(f"无法读取目标图: {TARGET_IMAGE_PATH}")
        target_source = "target_raw_image"

    template_prepared_path = output_dir / "template_prepared.jpg"
    target_prepared_path = output_dir / "target_prepared.jpg"
    cv2.imwrite(str(template_prepared_path), template_image)
    cv2.imwrite(str(target_prepared_path), target_image)

    return {
        "template": {
            "image": template_image,
            "prepared_path": template_prepared_path,
            "source_path": template_raw_path,
        },
        "target": {
            "image": target_image,
            "prepared_path": target_prepared_path,
            "source_path": str(TARGET_IMAGE_PATH),
            "prepared_from": target_source,
        },
    }


def to_binary_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    fg1 = mask1 > 0
    fg2 = mask2 > 0
    union = np.logical_or(fg1, fg2).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(fg1, fg2).sum()
    return float(inter / union)


def xor_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    fg1 = mask1 > 0
    fg2 = mask2 > 0
    union = np.logical_or(fg1, fg2).sum()
    if union == 0:
        return 0.0
    xor = np.logical_xor(fg1, fg2).sum()
    return float(xor / union)


def component_summary(mask: np.ndarray) -> Dict[str, float]:
    contours = extract_main_contours(mask)
    mask_area = float(mask.shape[0] * mask.shape[1])
    largest_area_ratio = 0.0
    if contours:
        largest_area_ratio = float(cv2.contourArea(contours[0]) / max(1.0, mask_area))
    return {
        "component_count": len(contours),
        "largest_component_area_ratio": largest_area_ratio,
    }


def compute_hu_metrics(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
    contours1 = extract_main_contours(mask1)
    contours2 = extract_main_contours(mask2)

    hu_avg = 0.0
    hull_score = 0.0
    if contours1 and contours2:
        n = min(3, len(contours1), len(contours2))
        hu_scores = [compare_contours_hu(contours1[i], contours2[i]) for i in range(n)]
        hu_avg = float(np.mean(hu_scores))

        hull1 = cv2.convexHull(np.vstack(contours1))
        hull2 = cv2.convexHull(np.vstack(contours2))
        hull_score = float(compare_contours_hu(hull1, hull2))

    return {
        "hu_avg": hu_avg,
        "hull_score": hull_score,
        "contour_count_1": len(contours1),
        "contour_count_2": len(contours2),
    }


def compute_ssim_score(mask1: np.ndarray, mask2: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim

        score, _ = ssim(mask1, mask2, full=True)
        return float(score)
    except Exception:
        return None


def make_diff_overlay(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    overlay = np.ones((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8) * 255

    fg1 = mask1 > 0
    fg2 = mask2 > 0
    overlap = np.logical_and(fg1, fg2)
    only_template = np.logical_and(fg1, np.logical_not(fg2))
    only_target = np.logical_and(np.logical_not(fg1), fg2)

    overlay[overlap] = (0, 180, 0)
    overlay[only_template] = (0, 0, 255)
    overlay[only_target] = (255, 0, 0)
    return overlay


def suggest_decision(metrics: Dict[str, float]) -> str:
    if (
        metrics["foreground_iou"] >= MATCH_FOREGROUND_IOU_MIN
        and metrics["xor_ratio"] <= MATCH_XOR_RATIO_MAX
        and metrics["hu_avg"] >= MATCH_HU_MIN
        and metrics["component_count_diff"] <= MATCH_COMPONENT_DELTA_MAX
    ):
        return "match"

    if (
        metrics["foreground_iou"] <= MISMATCH_FOREGROUND_IOU_MAX
        or metrics["xor_ratio"] >= MISMATCH_XOR_RATIO_MIN
        or metrics["hu_avg"] <= MISMATCH_HU_MAX
        or metrics["component_count_diff"] >= MISMATCH_COMPONENT_DELTA_MIN
    ):
        return "mismatch"

    return "unknown"


def compare_one_pair(
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_box: Sequence[int],
    target_box: Sequence[int],
    pair_output_dir: Path,
) -> Dict[str, object]:
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    tx1, ty1, tx2, ty2 = [int(v) for v in template_box]
    sx1, sy1, sx2, sy2 = [int(v) for v in target_box]

    template_crop = template_image[ty1:ty2, tx1:tx2]
    target_crop = target_image[sy1:sy2, sx1:sx2]
    if template_crop.size == 0 or target_crop.size == 0:
        return {"error": "empty_crop"}

    aligned_target, align_success = align_images_sift(template_crop, target_crop)

    template_mask = to_binary_mask(template_crop)
    aligned_target_mask = to_binary_mask(aligned_target)

    comp1 = component_summary(template_mask)
    comp2 = component_summary(aligned_target_mask)
    hu_metrics = compute_hu_metrics(template_mask, aligned_target_mask)
    ssim_score = compute_ssim_score(template_mask, aligned_target_mask)

    metrics = {
        "align_success": align_success,
        "foreground_iou": mask_iou(template_mask, aligned_target_mask),
        "xor_ratio": xor_ratio(template_mask, aligned_target_mask),
        "component_count_1": comp1["component_count"],
        "component_count_2": comp2["component_count"],
        "component_count_diff": abs(comp1["component_count"] - comp2["component_count"]),
        "largest_component_area_ratio_1": comp1["largest_component_area_ratio"],
        "largest_component_area_ratio_2": comp2["largest_component_area_ratio"],
        "largest_component_area_ratio_diff": abs(
            comp1["largest_component_area_ratio"] - comp2["largest_component_area_ratio"]
        ),
        "hu_avg": hu_metrics["hu_avg"],
        "hull_score": hu_metrics["hull_score"],
        "ssim_score": ssim_score,
    }
    metrics["suggested_decision"] = suggest_decision(metrics)

    cv2.imwrite(str(pair_output_dir / "template_crop.jpg"), template_crop)
    cv2.imwrite(str(pair_output_dir / "target_crop.jpg"), target_crop)
    cv2.imwrite(str(pair_output_dir / "target_aligned.jpg"), aligned_target)
    cv2.imwrite(str(pair_output_dir / "template_mask.jpg"), template_mask)
    cv2.imwrite(str(pair_output_dir / "target_aligned_mask.jpg"), aligned_target_mask)
    cv2.imwrite(
        str(pair_output_dir / "difference_overlay.jpg"),
        make_diff_overlay(template_mask, aligned_target_mask),
    )

    return {
        "template_box": [int(v) for v in template_box],
        "target_box": [int(v) for v in target_box],
        "template_size": list(template_crop.shape[:2]),
        "target_size": list(target_crop.shape[:2]),
        "metrics": metrics,
    }


def main() -> int:
    output_dir = OUTPUT_DIR / TEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("传统区域比较实验")
    print("=" * 60)

    payload = prepare_test_images(output_dir)
    template_image = payload["template"]["image"]
    target_image = payload["target"]["image"]
    template_path = payload["template"]["prepared_path"]
    target_path = payload["target"]["prepared_path"]

    print(f"template prepared: {template_path}")
    print(f"target prepared:   {target_path}")

    template_all_regions = detect_layout_regions(str(template_path), threshold=LAYOUT_THRESHOLD)
    target_all_regions = detect_layout_regions(str(target_path), threshold=LAYOUT_THRESHOLD)
    template_regions = extract_regions_by_type(template_all_regions, REGION_TYPE)
    target_regions = extract_regions_by_type(target_all_regions, REGION_TYPE)

    _, template_boxes = get_ocr_with_boxes(str(template_path))
    _, target_boxes = get_ocr_with_boxes(str(target_path))
    template_regions, skipped_template_regions = split_barcode_regions(
        template_regions,
        template_image,
        template_boxes,
    )
    target_regions, skipped_target_regions = split_barcode_regions(
        target_regions,
        target_image,
        target_boxes,
    )

    template_vis = draw_regions(template_image, template_regions, color=(0, 255, 0))
    target_vis = draw_regions(target_image, target_regions, color=(0, 255, 0))
    if skipped_template_regions:
        template_vis = draw_regions(template_vis, skipped_template_regions, color=(0, 165, 255))
    if skipped_target_regions:
        target_vis = draw_regions(target_vis, skipped_target_regions, color=(0, 165, 255))
    cv2.imwrite(str(output_dir / "template_regions.jpg"), template_vis)
    cv2.imwrite(str(output_dir / "target_regions.jpg"), target_vis)

    matched_pairs, unmatched1, unmatched2 = match_regions(
        template_regions,
        target_regions,
        template_image.shape[:2],
        target_image.shape[:2],
        cost_threshold=MATCH_COST_THRESHOLD,
    )

    print(f"template comparable regions: {len(template_regions)}")
    print(f"target comparable regions:   {len(target_regions)}")
    print(f"matched pairs:               {len(matched_pairs)}")
    print(f"unmatched template:          {len(unmatched1)}")
    print(f"unmatched target:            {len(unmatched2)}")

    selected_pairs = matched_pairs
    if PAIR_INDEX is not None:
        if PAIR_INDEX < 0 or PAIR_INDEX >= len(matched_pairs):
            raise IndexError(f"PAIR_INDEX 越界: {PAIR_INDEX}, 总匹配对数 {len(matched_pairs)}")
        selected_pairs = [matched_pairs[PAIR_INDEX]]

    pair_results: List[Dict[str, object]] = []
    for idx, (template_idx, target_idx, cost) in enumerate(selected_pairs):
        pair_output_dir = output_dir / f"pair_{idx:02d}_t{template_idx}_s{target_idx}"
        print("\n" + "-" * 60)
        print(f"pair #{idx}: template[{template_idx}] vs target[{target_idx}]")
        print(f"match_cost = {cost:.4f}")

        result = compare_one_pair(
            template_image,
            target_image,
            template_regions[template_idx]["coordinate"],
            target_regions[target_idx]["coordinate"],
            pair_output_dir,
        )
        result["template_idx"] = int(template_idx)
        result["target_idx"] = int(target_idx)
        result["match_cost"] = float(cost)
        pair_results.append(result)

        metrics = result.get("metrics", {})
        if metrics:
            print(
                "decision="
                f"{metrics.get('suggested_decision')} | "
                f"iou={metrics.get('foreground_iou', 0.0):.4f} | "
                f"xor={metrics.get('xor_ratio', 0.0):.4f} | "
                f"hu={metrics.get('hu_avg', 0.0):.4f} | "
                f"comp_diff={metrics.get('component_count_diff', 0)}"
            )

    summary = {
        "test_name": TEST_NAME,
        "pdf_path": str(PDF_PATH),
        "target_image_path": str(TARGET_IMAGE_PATH),
        "prepared_template_path": str(template_path),
        "prepared_target_path": str(target_path),
        "layout_threshold": LAYOUT_THRESHOLD,
        "match_cost_threshold": MATCH_COST_THRESHOLD,
        "pair_index": PAIR_INDEX,
        "decision_thresholds": {
            "match_foreground_iou_min": MATCH_FOREGROUND_IOU_MIN,
            "match_xor_ratio_max": MATCH_XOR_RATIO_MAX,
            "match_hu_min": MATCH_HU_MIN,
            "match_component_delta_max": MATCH_COMPONENT_DELTA_MAX,
            "mismatch_foreground_iou_max": MISMATCH_FOREGROUND_IOU_MAX,
            "mismatch_xor_ratio_min": MISMATCH_XOR_RATIO_MIN,
            "mismatch_hu_max": MISMATCH_HU_MAX,
            "mismatch_component_delta_min": MISMATCH_COMPONENT_DELTA_MIN,
        },
        "template_regions": template_regions,
        "target_regions": target_regions,
        "skipped_template_regions": skipped_template_regions,
        "skipped_target_regions": skipped_target_regions,
        "matched_pairs": matched_pairs,
        "unmatched_template": unmatched1,
        "unmatched_target": unmatched2,
        "pair_results": pair_results,
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\nsummary json:")
    print(summary_path)
    print("建议先看每个 pair 目录下的 difference_overlay.jpg，再对照各项分数调阈值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
