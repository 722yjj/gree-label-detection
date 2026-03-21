"""Direct-run experiment for deterministic graphic region comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from label_detection.core.config import PROJECT_ROOT as REPO_ROOT
from label_detection.extraction.pdf import extract_red_box_info
from label_detection.matching.layout import (
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
MATCH_UNMATCHED_AREA_RATIO_MAX = 0.03

MISMATCH_FOREGROUND_IOU_MAX = 0.45
MISMATCH_XOR_RATIO_MIN = 0.22
MISMATCH_HU_MAX = 0.25
MISMATCH_COMPONENT_DELTA_MIN = 2
MISMATCH_UNMATCHED_AREA_RATIO_MIN = 0.06

COMPONENT_MIN_AREA_RATIO = 0.008
COMPONENT_MATCH_IOU_MIN = 0.18
COMPONENT_MATCH_CENTER_DIST_MAX = 0.18
NORMALIZE_MARGIN = 8

TEXTURE_ASPECT_MIN = 1.8
TEXTURE_TRANSITION_MIN = 0.08
TEXTURE_VERTICAL_BIAS_MIN = 1.6
TEXTURE_GRADIENT_BIAS_MIN = 1.8
TEXTURE_ROW_TRANSITION_MAX = 0.18


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


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def compute_vertical_texture_metrics(image: np.ndarray) -> Dict[str, float]:
    if image.size == 0:
        return {
            "aspect_ratio": 0.0,
            "dark_ratio": 0.0,
            "transition_density": 0.0,
            "row_transition_density": 0.0,
            "vertical_bias": 0.0,
            "gradient_bias": 0.0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = binary > 0

    if fg.shape[0] < 8 or fg.shape[1] < 8:
        return {
            "aspect_ratio": float(fg.shape[1] / max(1.0, fg.shape[0])),
            "dark_ratio": float(fg.mean()) if fg.size else 0.0,
            "transition_density": 0.0,
            "row_transition_density": 0.0,
            "vertical_bias": 0.0,
            "gradient_bias": 0.0,
        }

    column_dark = fg.mean(axis=0)
    row_dark = fg.mean(axis=1)
    dark_threshold = max(0.08, float(fg.mean()) * 0.65)
    dark_columns = column_dark > dark_threshold
    dark_rows = row_dark > dark_threshold

    column_transitions = np.abs(np.diff(dark_columns.astype(np.int8))).sum()
    row_transitions = np.abs(np.diff(dark_rows.astype(np.int8))).sum()
    transition_density = float(column_transitions / max(1, dark_columns.size - 1))
    row_transition_density = float(row_transitions / max(1, dark_rows.size - 1))
    vertical_bias = float(transition_density / max(0.01, row_transition_density))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_bias = float(np.mean(np.abs(grad_x)) / max(1e-6, np.mean(np.abs(grad_y))))

    return {
        "aspect_ratio": float(gray.shape[1] / max(1.0, gray.shape[0])),
        "dark_ratio": float(fg.mean()),
        "transition_density": transition_density,
        "row_transition_density": row_transition_density,
        "vertical_bias": vertical_bias,
        "gradient_bias": gradient_bias,
    }


def is_repetitive_vertical_texture(image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    metrics = compute_vertical_texture_metrics(image)
    is_texture = (
        metrics["aspect_ratio"] >= TEXTURE_ASPECT_MIN
        and 0.05 <= metrics["dark_ratio"] <= 0.90
        and metrics["transition_density"] >= TEXTURE_TRANSITION_MIN
        and metrics["row_transition_density"] <= TEXTURE_ROW_TRANSITION_MAX
        and (
            metrics["vertical_bias"] >= TEXTURE_VERTICAL_BIAS_MIN
            or metrics["gradient_bias"] >= TEXTURE_GRADIENT_BIAS_MIN
        )
    )
    return is_texture, metrics


def crop_region(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, image.shape[1]))
    x2 = max(0, min(x2, image.shape[1]))
    y1 = max(0, min(y1, image.shape[0]))
    y2 = max(0, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


def filter_repetitive_texture_regions(
    regions: Sequence[Dict[str, object]],
    image: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    comparable: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for region in regions:
        crop = crop_region(image, region["coordinate"])
        is_texture, metrics = is_repetitive_vertical_texture(crop)
        region_copy = dict(region)
        if is_texture:
            region_copy["skip_reason"] = "repetitive_vertical_texture"
            region_copy["texture_hint"] = metrics
            skipped.append(region_copy)
        else:
            comparable.append(region_copy)

    return comparable, skipped


def foreground_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, mask.shape[1], mask.shape[0]]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop_to_foreground(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    mask = to_binary_mask(image)
    x1, y1, x2, y2 = foreground_bbox(mask)
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], [x1, y1, x2, y2]


def resize_to_height(image: np.ndarray, target_height: int, is_mask: bool = False) -> np.ndarray:
    target_height = max(1, int(target_height))
    src_h, src_w = image.shape[:2]
    if src_h == target_height:
        return image.copy()
    scale = target_height / max(1.0, float(src_h))
    target_width = max(1, int(round(src_w * scale)))
    interpolation = cv2.INTER_NEAREST if is_mask else (cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA)
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def normalize_pair_crops(template_crop: np.ndarray, target_crop: np.ndarray) -> Dict[str, object]:
    template_fg, _, template_fg_box = crop_to_foreground(template_crop)
    target_fg, _, target_fg_box = crop_to_foreground(target_crop)

    common_height = max(template_fg.shape[0], target_fg.shape[0], 1)
    template_norm = resize_to_height(template_fg, common_height)
    target_norm = resize_to_height(target_fg, common_height)

    template_mask = to_binary_mask(template_norm)
    target_mask = to_binary_mask(target_norm)

    canvas_height = max(template_norm.shape[0], target_norm.shape[0]) + NORMALIZE_MARGIN * 2
    canvas_width = max(template_norm.shape[1], target_norm.shape[1]) + NORMALIZE_MARGIN * 2

    template_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    target_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    template_mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    target_mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    max_content_height = max(template_norm.shape[0], target_norm.shape[0])

    def paste_image(src: np.ndarray, dst: np.ndarray) -> List[int]:
        top = NORMALIZE_MARGIN + (max_content_height - src.shape[0])
        left = NORMALIZE_MARGIN
        bottom = top + src.shape[0]
        right = left + src.shape[1]
        dst[top:bottom, left:right] = src
        return [left, top, right, bottom]

    template_canvas_box = paste_image(template_norm, template_canvas)
    target_canvas_box = paste_image(target_norm, target_canvas)
    paste_image(template_mask, template_mask_canvas)
    paste_image(target_mask, target_mask_canvas)

    return {
        "template_image": template_canvas,
        "target_image": target_canvas,
        "template_mask": template_mask_canvas,
        "target_mask": target_mask_canvas,
        "meta": {
            "method": "foreground_crop_resize_pad_bottom_align",
            "template_foreground_box": template_fg_box,
            "target_foreground_box": target_fg_box,
            "template_canvas_box": template_canvas_box,
            "target_canvas_box": target_canvas_box,
        },
    }


def extract_component_boxes(mask: np.ndarray) -> List[List[int]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    min_area = mask.shape[0] * mask.shape[1] * COMPONENT_MIN_AREA_RATIO
    boxes: List[List[int]] = []

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area:
            continue
        if w < 4 or h < 4:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    boxes.sort(key=lambda item: (item[0], item[1]))
    return boxes


def box_area(box: Sequence[int]) -> float:
    return float(max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1])))


def box_iou(box1: Sequence[int], box2: Sequence[int]) -> float:
    inter_x1 = max(int(box1[0]), int(box2[0]))
    inter_y1 = max(int(box1[1]), int(box2[1]))
    inter_x2 = min(int(box1[2]), int(box2[2]))
    inter_y2 = min(int(box1[3]), int(box2[3]))

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0:
        return 0.0

    union = box_area(box1) + box_area(box2) - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def normalized_center_distance(box1: Sequence[int], box2: Sequence[int], canvas_shape: Sequence[int]) -> float:
    c1x = (float(box1[0]) + float(box1[2])) / 2.0
    c1y = (float(box1[1]) + float(box1[3])) / 2.0
    c2x = (float(box2[0]) + float(box2[2])) / 2.0
    c2y = (float(box2[1]) + float(box2[3])) / 2.0
    width = max(1.0, float(canvas_shape[1]))
    height = max(1.0, float(canvas_shape[0]))
    dx = (c1x - c2x) / width
    dy = (c1y - c2y) / height
    return float(np.hypot(dx, dy))


def match_component_boxes(
    template_boxes: Sequence[Sequence[int]],
    target_boxes: Sequence[Sequence[int]],
    canvas_shape: Sequence[int],
) -> Tuple[List[Dict[str, float]], List[List[int]], List[List[int]]]:
    matches: List[Dict[str, float]] = []
    used_target_indices: set[int] = set()
    unmatched_template: List[List[int]] = []

    for template_idx, template_box in enumerate(template_boxes):
        best_idx = -1
        best_score = -1.0
        best_iou = 0.0
        best_distance = 1.0

        template_area = box_area(template_box)
        for target_idx, target_box in enumerate(target_boxes):
            if target_idx in used_target_indices:
                continue

            iou = box_iou(template_box, target_box)
            center_dist = normalized_center_distance(template_box, target_box, canvas_shape)
            area_ratio = min(template_area, box_area(target_box)) / max(1.0, max(template_area, box_area(target_box)))

            valid_candidate = iou >= COMPONENT_MATCH_IOU_MIN or (
                center_dist <= COMPONENT_MATCH_CENTER_DIST_MAX and area_ratio >= 0.45
            )
            if not valid_candidate:
                continue

            score = iou * 0.55 + area_ratio * 0.25 + max(
                0.0,
                1.0 - center_dist / max(1e-6, COMPONENT_MATCH_CENTER_DIST_MAX),
            ) * 0.20
            if score > best_score:
                best_idx = target_idx
                best_score = score
                best_iou = iou
                best_distance = center_dist

        if best_idx >= 0:
            used_target_indices.add(best_idx)
            matches.append(
                {
                    "template_idx": float(template_idx),
                    "target_idx": float(best_idx),
                    "score": float(best_score),
                    "iou": float(best_iou),
                    "center_distance": float(best_distance),
                }
            )
        else:
            unmatched_template.append([int(v) for v in template_box])

    unmatched_target = [
        [int(v) for v in target_box]
        for target_idx, target_box in enumerate(target_boxes)
        if target_idx not in used_target_indices
    ]
    return matches, unmatched_template, unmatched_target


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], color: Tuple[int, int, int]) -> np.ndarray:
    canvas = ensure_bgr(image)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def unmatched_area_ratio(boxes: Sequence[Sequence[int]], mask: np.ndarray) -> float:
    fg_area = float((mask > 0).sum())
    if fg_area <= 0:
        return 0.0
    return float(sum(box_area(box) for box in boxes) / fg_area)


def suggest_decision(metrics: Dict[str, float]) -> str:
    unmatched_area_ratio_total = float(metrics.get("unmatched_area_ratio_total") or 0.0)
    if (
        float(metrics.get("foreground_iou") or 0.0) >= MATCH_FOREGROUND_IOU_MIN
        and float(metrics.get("xor_ratio") or 0.0) <= MATCH_XOR_RATIO_MAX
        and float(metrics.get("hu_avg") or 0.0) >= MATCH_HU_MIN
        and int(metrics.get("component_count_diff") or 0) <= MATCH_COMPONENT_DELTA_MAX
        and unmatched_area_ratio_total <= MATCH_UNMATCHED_AREA_RATIO_MAX
    ):
        return "match"

    if (
        float(metrics.get("foreground_iou") or 0.0) <= MISMATCH_FOREGROUND_IOU_MAX
        or float(metrics.get("xor_ratio") or 0.0) >= MISMATCH_XOR_RATIO_MIN
        or float(metrics.get("hu_avg") or 0.0) <= MISMATCH_HU_MAX
        or int(metrics.get("component_count_diff") or 0) >= MISMATCH_COMPONENT_DELTA_MIN
        or unmatched_area_ratio_total >= MISMATCH_UNMATCHED_AREA_RATIO_MIN
        or int(metrics.get("unmatched_template_component_count") or 0) >= 1
        or int(metrics.get("unmatched_target_component_count") or 0) >= 1
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

    template_crop = crop_region(template_image, template_box)
    target_crop = crop_region(target_image, target_box)
    if template_crop.size == 0 or target_crop.size == 0:
        return {"error": "empty_crop"}

    template_texture_skip, template_texture_metrics = is_repetitive_vertical_texture(template_crop)
    target_texture_skip, target_texture_metrics = is_repetitive_vertical_texture(target_crop)

    cv2.imwrite(str(pair_output_dir / "template_crop.jpg"), template_crop)
    cv2.imwrite(str(pair_output_dir / "target_crop.jpg"), target_crop)

    if template_texture_skip or target_texture_skip:
        return {
            "template_box": [int(v) for v in template_box],
            "target_box": [int(v) for v in target_box],
            "template_size": list(template_crop.shape[:2]),
            "target_size": list(target_crop.shape[:2]),
            "normalization": {"method": "skipped_repetitive_vertical_texture"},
            "metrics": {
                "suggested_decision": "skipped",
                "skip_reason": "repetitive_vertical_texture",
                "template_texture_metrics": template_texture_metrics,
                "target_texture_metrics": target_texture_metrics,
            },
        }

    normalized = normalize_pair_crops(template_crop, target_crop)
    template_normalized = normalized["template_image"]
    target_normalized = normalized["target_image"]
    template_mask = normalized["template_mask"]
    target_mask = normalized["target_mask"]

    comp1 = component_summary(template_mask)
    comp2 = component_summary(target_mask)
    hu_metrics = compute_hu_metrics(template_mask, target_mask)
    ssim_score = compute_ssim_score(template_mask, target_mask)

    template_component_boxes = extract_component_boxes(template_mask)
    target_component_boxes = extract_component_boxes(target_mask)
    component_matches, unmatched_template_boxes, unmatched_target_boxes = match_component_boxes(
        template_component_boxes,
        target_component_boxes,
        template_mask.shape,
    )

    template_component_area_ratio = unmatched_area_ratio(unmatched_template_boxes, template_mask)
    target_component_area_ratio = unmatched_area_ratio(unmatched_target_boxes, target_mask)

    metrics = {
        "align_success": False,
        "foreground_iou": mask_iou(template_mask, target_mask),
        "xor_ratio": xor_ratio(template_mask, target_mask),
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
        "template_component_box_count": len(template_component_boxes),
        "target_component_box_count": len(target_component_boxes),
        "matched_component_count": len(component_matches),
        "unmatched_template_component_count": len(unmatched_template_boxes),
        "unmatched_target_component_count": len(unmatched_target_boxes),
        "unmatched_template_area_ratio": template_component_area_ratio,
        "unmatched_target_area_ratio": target_component_area_ratio,
        "unmatched_area_ratio_total": template_component_area_ratio + target_component_area_ratio,
    }
    metrics["suggested_decision"] = suggest_decision(metrics)

    diff_overlay = make_diff_overlay(template_mask, target_mask)
    diff_boxes_overlay = draw_boxes(diff_overlay, unmatched_template_boxes, (0, 0, 255))
    diff_boxes_overlay = draw_boxes(diff_boxes_overlay, unmatched_target_boxes, (255, 0, 0))

    template_boxes_preview = draw_boxes(template_normalized, template_component_boxes, (0, 180, 0))
    template_boxes_preview = draw_boxes(template_boxes_preview, unmatched_template_boxes, (0, 0, 255))
    target_boxes_preview = draw_boxes(target_normalized, target_component_boxes, (0, 180, 0))
    target_boxes_preview = draw_boxes(target_boxes_preview, unmatched_target_boxes, (255, 0, 0))

    cv2.imwrite(str(pair_output_dir / "template_normalized.jpg"), template_normalized)
    cv2.imwrite(str(pair_output_dir / "target_normalized.jpg"), target_normalized)
    cv2.imwrite(str(pair_output_dir / "template_mask.jpg"), template_mask)
    cv2.imwrite(str(pair_output_dir / "target_normalized_mask.jpg"), target_mask)
    cv2.imwrite(str(pair_output_dir / "template_component_boxes.jpg"), template_boxes_preview)
    cv2.imwrite(str(pair_output_dir / "target_component_boxes.jpg"), target_boxes_preview)
    cv2.imwrite(str(pair_output_dir / "difference_overlay.jpg"), diff_overlay)
    cv2.imwrite(str(pair_output_dir / "difference_boxes_overlay.jpg"), diff_boxes_overlay)

    return {
        "template_box": [int(v) for v in template_box],
        "target_box": [int(v) for v in target_box],
        "template_size": list(template_crop.shape[:2]),
        "target_size": list(target_crop.shape[:2]),
        "normalization": normalized["meta"],
        "template_component_boxes": template_component_boxes,
        "target_component_boxes": target_component_boxes,
        "unmatched_template_boxes": unmatched_template_boxes,
        "unmatched_target_boxes": unmatched_target_boxes,
        "component_matches": component_matches,
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

    template_regions, texture_skipped_template_regions = filter_repetitive_texture_regions(
        template_regions,
        template_image,
    )
    target_regions, texture_skipped_target_regions = filter_repetitive_texture_regions(
        target_regions,
        target_image,
    )
    skipped_template_regions = skipped_template_regions + texture_skipped_template_regions
    skipped_target_regions = skipped_target_regions + texture_skipped_target_regions

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
    print(f"skipped template regions:    {len(skipped_template_regions)}")
    print(f"skipped target regions:      {len(skipped_target_regions)}")
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
                f"iou={float(metrics.get('foreground_iou', 0.0) or 0.0):.4f} | "
                f"xor={float(metrics.get('xor_ratio', 0.0) or 0.0):.4f} | "
                f"hu={float(metrics.get('hu_avg', 0.0) or 0.0):.4f} | "
                f"unmatched_t={int(metrics.get('unmatched_template_component_count', 0) or 0)} | "
                f"unmatched_s={int(metrics.get('unmatched_target_component_count', 0) or 0)}"
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
        "comparison_method": "foreground_crop_resize_pad_bottom_align + component_delta",
        "decision_thresholds": {
            "match_foreground_iou_min": MATCH_FOREGROUND_IOU_MIN,
            "match_xor_ratio_max": MATCH_XOR_RATIO_MAX,
            "match_hu_min": MATCH_HU_MIN,
            "match_component_delta_max": MATCH_COMPONENT_DELTA_MAX,
            "match_unmatched_area_ratio_max": MATCH_UNMATCHED_AREA_RATIO_MAX,
            "mismatch_foreground_iou_max": MISMATCH_FOREGROUND_IOU_MAX,
            "mismatch_xor_ratio_min": MISMATCH_XOR_RATIO_MIN,
            "mismatch_hu_max": MISMATCH_HU_MAX,
            "mismatch_component_delta_min": MISMATCH_COMPONENT_DELTA_MIN,
            "mismatch_unmatched_area_ratio_min": MISMATCH_UNMATCHED_AREA_RATIO_MIN,
        },
        "texture_skip_thresholds": {
            "aspect_min": TEXTURE_ASPECT_MIN,
            "transition_min": TEXTURE_TRANSITION_MIN,
            "vertical_bias_min": TEXTURE_VERTICAL_BIAS_MIN,
            "gradient_bias_min": TEXTURE_GRADIENT_BIAS_MIN,
            "row_transition_max": TEXTURE_ROW_TRANSITION_MAX,
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
    print("建议先看 difference_boxes_overlay.jpg，再看 comparison_summary.json 里的组件差异指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
