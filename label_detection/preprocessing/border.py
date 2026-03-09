"""
边框检测与裁剪模块

提供黑色边框检测和模板自动裁剪功能，用于模板图片和实拍图片的预处理。

STATUS: main
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]


def _build_dark_mask(gray: np.ndarray) -> np.ndarray:
    """Build a stable dark-foreground mask for both borders and text blocks."""
    _, thresh_fixed = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    _, thresh_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    thresh = cv2.bitwise_or(thresh_fixed, thresh_otsu)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


def _build_content_mask(thresh: np.ndarray) -> np.ndarray:
    """
    Merge header bar, text rows and barcode into one content-level candidate.

    This is used for templates without an explicit outer frame.
    """
    h, w = thresh.shape
    content = thresh.copy()
    content = cv2.morphologyEx(
        content,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 18), 5)),
    )
    content = cv2.morphologyEx(
        content,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(13, h // 8))),
    )
    content = cv2.dilate(
        content,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 200), max(3, h // 200))),
        iterations=1,
    )
    return content


def _rect_iou(rect1: Rect, rect2: Rect) -> float:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / float(union) if union > 0 else 0.0


def _extract_rect_candidates(
    mask: np.ndarray,
    min_area_ratio: float,
    max_candidates: int = 6,
) -> List[Rect]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    image_area = float(mask.shape[0] * mask.shape[1])
    rects: List[Rect] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[: max_candidates * 4]:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        area_ratio = (w * h) / image_area
        if area_ratio < min_area_ratio:
            continue

        rect = (x, y, w, h)
        if any(_rect_iou(rect, existing) > 0.92 for existing in rects):
            continue

        rects.append(rect)
        if len(rects) >= max_candidates:
            break

    return rects


def _count_bands(active_rows: np.ndarray, min_band_height: int) -> int:
    bands = 0
    run = 0
    for is_active in active_rows:
        if is_active:
            run += 1
        elif run:
            if run >= min_band_height:
                bands += 1
            run = 0
    if run >= min_band_height:
        bands += 1
    return bands


def _compute_candidate_metrics(rect: Rect, dark_mask: np.ndarray) -> Dict[str, float]:
    x, y, w, h = rect
    img_h, img_w = dark_mask.shape
    crop = dark_mask[y : y + h, x : x + w]

    total_foreground = cv2.countNonZero(dark_mask)
    crop_foreground = cv2.countNonZero(crop)
    if crop.size == 0 or total_foreground == 0:
        return {}

    band = max(2, min(w, h) // 18)
    top_ratio = cv2.countNonZero(crop[:band, :]) / float(max(band * w, 1))
    bottom_ratio = cv2.countNonZero(crop[h - band :, :]) / float(max(band * w, 1))
    left_ratio = cv2.countNonZero(crop[:, :band]) / float(max(band * h, 1))
    right_ratio = cv2.countNonZero(crop[:, w - band :]) / float(max(band * h, 1))

    inner_margin = min(max(3, band), max(1, min(w, h) // 3))
    inner = crop[
        inner_margin : max(inner_margin + 1, h - inner_margin),
        inner_margin : max(inner_margin + 1, w - inner_margin),
    ]
    inner_ratio = (
        cv2.countNonZero(inner) / float(inner.size)
        if inner.size
        else cv2.countNonZero(crop) / float(crop.size)
    )

    row_density = np.count_nonzero(crop, axis=1) / float(max(w, 1))
    active_rows = row_density > 0.05
    band_count = _count_bands(active_rows, min_band_height=max(2, h // 40))

    lower_half = crop[h // 2 :, :]
    lower_half_ratio = cv2.countNonZero(lower_half) / float(max(crop_foreground, 1))

    right_lower = crop[h // 2 :, w // 2 :]
    right_lower_ratio = cv2.countNonZero(right_lower) / float(max(crop_foreground, 1))

    return {
        "area_ratio": (w * h) / float(max(img_w * img_h, 1)),
        "height_ratio": h / float(max(img_h, 1)),
        "fg_recall": crop_foreground / float(max(total_foreground, 1)),
        "edge_mean": (top_ratio + bottom_ratio + left_ratio + right_ratio) / 4.0,
        "edge_min": min(top_ratio, bottom_ratio, left_ratio, right_ratio),
        "inner_ratio": inner_ratio,
        "frame_score": max(
            0.0,
            ((top_ratio + bottom_ratio + left_ratio + right_ratio) / 4.0) - inner_ratio,
        ),
        "band_count": float(band_count),
        "lower_half_ratio": lower_half_ratio,
        "right_lower_ratio": right_lower_ratio,
    }


def _score_template_candidate(metrics: Dict[str, float], strategy: str) -> float:
    if not metrics:
        return float("-inf")

    score = 0.0
    score += metrics["fg_recall"] * 3.2
    score += min(metrics["height_ratio"] / 0.45, 1.0) * 1.4
    score += min(metrics["area_ratio"] / 0.35, 1.0) * 0.8
    score += min(metrics["band_count"], 3.0) * 0.35
    score += min(metrics["lower_half_ratio"] / 0.18, 1.0) * 1.1

    if strategy == "border":
        score += metrics["frame_score"] * 2.4
        score += min(metrics["edge_min"] / 0.35, 1.0) * 0.8
        score -= max(metrics["inner_ratio"] - 0.60, 0.0) * 1.2
    else:
        score += min(metrics["right_lower_ratio"] / 0.12, 1.0) * 0.7
        score += min(metrics["edge_mean"] / 0.55, 1.0) * 0.2

    return score


def _is_plausible_template_crop(metrics: Dict[str, float], strategy: str) -> bool:
    if not metrics:
        return False

    if metrics["fg_recall"] < 0.58:
        return False
    if metrics["height_ratio"] < 0.20:
        return False

    # Reject the common failure mode where only the title/header strip is cropped.
    if metrics["band_count"] <= 1 and metrics["lower_half_ratio"] < 0.08:
        return False

    if strategy == "border":
        if metrics["frame_score"] < 0.03 and metrics["fg_recall"] < 0.82:
            return False
    else:
        if metrics["lower_half_ratio"] < 0.10:
            return False

    return True


def find_black_border(image, debug_dir=None, name="image"):
    """
    检测图片中的黑色边框，返回边框内区域的坐标。

    This keeps the original behavior for frame-heavy labels and for callers that
    explicitly want "largest dark frame" rather than automatic strategy selection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thresh = _build_dark_mask(gray)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_thresh.jpg"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    area_ratio = (cw * ch) / (w * h)
    if area_ratio < 0.3:
        print(f"  警告: 检测到的区域太小 ({area_ratio:.2%})")
        return None

    print(f"  检测到边框区域: x={x}, y={y}, w={cw}, h={ch} (占比: {area_ratio:.2%})")
    return (x, y, cw, ch)


def find_template_crop_rect(
    image: np.ndarray,
    debug_dir: Optional[str] = None,
    name: str = "template",
) -> Optional[Dict[str, object]]:
    """
    Automatically choose a template crop strategy.

    Strategy A: explicit border/frame candidate.
    Strategy B: content bounding box candidate for labels without outer borders.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = _build_dark_mask(gray)
    content_mask = _build_content_mask(thresh)

    candidates: List[Dict[str, object]] = []

    for rect in _extract_rect_candidates(thresh, min_area_ratio=0.03):
        metrics = _compute_candidate_metrics(rect, thresh)
        candidates.append(
            {
                "rect": rect,
                "strategy": "border",
                "metrics": metrics,
                "score": _score_template_candidate(metrics, "border"),
                "valid": _is_plausible_template_crop(metrics, "border"),
            }
        )

    for rect in _extract_rect_candidates(content_mask, min_area_ratio=0.08):
        metrics = _compute_candidate_metrics(rect, thresh)
        candidates.append(
            {
                "rect": rect,
                "strategy": "content",
                "metrics": metrics,
                "score": _score_template_candidate(metrics, "content"),
                "valid": _is_plausible_template_crop(metrics, "content"),
            }
        )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_thresh.jpg"), thresh)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_content_mask.jpg"), content_mask)
        debug_path = os.path.join(debug_dir, f"{name}_crop_candidates.txt")
        with open(debug_path, "w", encoding="utf-8") as fh:
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
                fh.write(
                    f"{candidate['strategy']}\tvalid={candidate['valid']}\t"
                    f"score={candidate['score']:.3f}\trect={candidate['rect']}\t"
                    f"metrics={candidate['metrics']}\n"
                )

    if not candidates:
        return None

    valid_candidates = [item for item in candidates if item["valid"]]
    if valid_candidates:
        return max(valid_candidates, key=lambda item: item["score"])

    best = max(candidates, key=lambda item: item["score"])
    if best["metrics"].get("fg_recall", 0.0) >= 0.80:
        return best

    return None


def crop_to_border(image, border_rect, padding=5):
    """
    裁剪图像到边框区域。

    Args:
        image: 原始图像
        border_rect: (x, y, w, h) 边框坐标
        padding: 正数表示向内裁掉边框线；负数表示向外补白边
    """
    x, y, w, h = border_rect
    ih, iw = image.shape[:2]

    x1 = max(min(x + padding, iw), 0)
    y1 = max(min(y + padding, ih), 0)
    x2 = max(min(x + w - padding, iw), 0)
    y2 = max(min(y + h - padding, ih), 0)

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2]
