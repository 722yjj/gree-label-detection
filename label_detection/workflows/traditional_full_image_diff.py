"""Traditional whole-image diff on two preprocessed label images."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import (  # noqa: E402
    OPENAI_COMPATIBLE_API_BASE,
    OPENAI_COMPATIBLE_API_KEY,
    OPENAI_COMPATIBLE_MODEL,
    PROJECT_ROOT as REPO_ROOT,
    VLM_TIMEOUT,
)
from label_detection.services.openai_compatible_client import OpenAICompatibleHTTPClient  # noqa: E402
from scripts.experiment_vlm_diff_localization import (  # noqa: E402
    align_target_to_template,
    extract_json_object,
    imwrite,
    prepare_output_dir,
    resolve_and_preprocess,
    scale_pair_for_canvas,
    write_json,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "traditional_full_image_diff"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traditional full-image difference experiment matching VLM inputs."
    )
    parser.add_argument("--template", required=True, help="模板 PDF 或图片路径")
    parser.add_argument("--target", required=True, help="实拍目标图片路径")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="输出根目录，默认 results/traditional_full_image_diff",
    )
    parser.add_argument("--run-name", help="输出子目录名称；默认使用时间戳")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="保存预处理图、mask、overlay、panel 等调试中间产物；默认只保存 final_result.jpg 和 result.json",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=1024,
        help="传统差异计算时的最大图像宽度",
    )
    parser.add_argument(
        "--skip-feature-align",
        action="store_true",
        help="跳过 SIFT/ORB 单应性对齐，仅使用预处理透视矫正和缩放",
    )
    parser.add_argument("--skip-ecc", action="store_true", help="跳过 ECC 微调对齐")
    parser.add_argument(
        "--ecc-motion",
        choices=("translation", "euclidean", "affine"),
        default="translation",
        help="ECC 微调运动模型",
    )
    parser.add_argument(
        "--diff-tolerance",
        type=int,
        default=3,
        help="像素容差半径，用于忽略轻微错位",
    )
    parser.add_argument(
        "--merge-radius",
        type=int,
        default=6,
        help="差异像素合并半径",
    )
    parser.add_argument("--min-area", type=int, default=80, help="最小差异连通域面积")
    parser.add_argument("--min-side", type=int, default=6, help="最小差异框边长")
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.02,
        help="差异框内最小差异像素密度",
    )
    parser.add_argument(
        "--ignore-edge-margin",
        type=int,
        default=8,
        help="忽略贴近图像边缘的差异框",
    )
    parser.add_argument("--max-boxes", type=int, default=50, help="最多保留差异框")
    parser.add_argument(
        "--disable-small-text-candidates",
        action="store_true",
        help="禁用数值/单位文字行附近的小面积高密度补充候选",
    )
    parser.add_argument(
        "--candidate-merge-gap",
        type=int,
        default=18,
        help="送 VLM 前合并相距不超过该像素的候选框",
    )
    parser.add_argument(
        "--vlm-filter",
        action="store_true",
        help="用 OpenAI-compatible VLM 对传统候选框二次判别，最终只画 VLM 保留的框",
    )
    parser.add_argument(
        "--model",
        default=OPENAI_COMPATIBLE_MODEL,
        help="OpenAI-compatible 模型名，默认 OPENAI_COMPATIBLE_MODEL/VLLM_MODEL",
    )
    parser.add_argument(
        "--api-base",
        default=OPENAI_COMPATIBLE_API_BASE,
        help="OpenAI-compatible API base，例如 http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=OPENAI_COMPATIBLE_API_KEY,
        help="OpenAI-compatible API key；vLLM 本地服务通常用 EMPTY",
    )
    parser.add_argument("--timeout", type=int, default=VLM_TIMEOUT, help="VLM 请求超时秒数")
    parser.add_argument("--max-tokens", type=int, default=256, help="VLM 最大输出 token")
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=32,
        help="送 VLM 判别时 review crop 的最小 padding",
    )
    parser.add_argument(
        "--review-min-size",
        type=int,
        default=128,
        help="送 VLM 判别时 review crop 的最小宽高",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="VLM 返回 unknown 时也保留该候选框；默认丢弃 unknown",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过 /models 连通和模型存在性检查",
    )
    return parser


def foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)


def tolerant_diff(
    template_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    tolerance: int,
    merge_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tolerance > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1),
        )
        template_dilated = cv2.dilate(template_mask, kernel)
        target_dilated = cv2.dilate(target_mask, kernel)
    else:
        template_dilated = template_mask
        target_dilated = target_mask

    missing = cv2.bitwise_and(template_mask, cv2.bitwise_not(target_dilated))
    extra = cv2.bitwise_and(target_mask, cv2.bitwise_not(template_dilated))
    diff = cv2.bitwise_or(missing, extra)

    if merge_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (merge_radius * 2 + 1, merge_radius * 2 + 1),
        )
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return missing, extra, cv2.morphologyEx(diff, cv2.MORPH_OPEN, clean)


def extract_boxes(
    diff_mask: np.ndarray,
    *,
    min_area: int,
    min_side: int,
    min_density: float,
    ignore_edge_margin: int,
    max_boxes: int,
) -> List[Dict[str, Any]]:
    height, width = diff_mask.shape[:2]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(diff_mask, 8)
    boxes: List[Dict[str, Any]] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if ignore_edge_margin > 0 and (
            x <= ignore_edge_margin
            or y <= ignore_edge_margin
            or x + w >= width - ignore_edge_margin
            or y + h >= height - ignore_edge_margin
        ):
            continue
        if area < min_area or w < min_side or h < min_side:
            continue
        density = area / max(1.0, float(w * h))
        if density < min_density:
            continue
        cx, cy = centroids[idx]
        boxes.append(
            {
                "box": [x, y, x + w, y + h],
                "area": area,
                "density": float(density),
                "centroid": [float(cx), float(cy)],
            }
        )
    boxes.sort(key=lambda item: int(item["area"]), reverse=True)
    return boxes[:max_boxes]


def _box_intersection_area(box_a: Sequence[int], box_b: Sequence[int]) -> int:
    ax1, ay1, ax2, ay2 = _box_values(box_a)
    bx1, by1, bx2, by2 = _box_values(box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def extract_small_text_candidates(
    diff_mask: np.ndarray,
    existing_boxes: Sequence[Dict[str, Any]],
    foreground_context_mask: np.ndarray,
    *,
    ignore_edge_margin: int,
    normal_min_area: int,
    max_candidates: int = 8,
) -> List[Dict[str, Any]]:
    """Find tiny dense diffs likely to be value/unit text changes."""
    height, width = diff_mask.shape[:2]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(diff_mask, 8)
    boxes: List[Dict[str, Any]] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if ignore_edge_margin > 0 and (
            x <= ignore_edge_margin
            or y <= ignore_edge_margin
            or x + w >= width - ignore_edge_margin
            or y + h >= height - ignore_edge_margin
        ):
            continue
        if area < 8 or area >= normal_min_area:
            continue
        if w < 2 or h < 2 or w > 24 or h > 24:
            continue
        density = area / max(1.0, float(w * h))
        if density < 0.45:
            continue
        cx, cy = centroids[idx]
        if cy < height * 0.25 or cy > height * 0.72:
            continue
        if cx < width * 0.35:
            continue

        box = [x, y, x + w, y + h]
        if any(_box_intersection_area(box, existing.get("box", [])) > 0 for existing in existing_boxes):
            continue
        review_box = expand_text_line_review_box(
            box,
            diff_mask.shape[:2],
            foreground_context_mask,
            padding=24,
            min_size=96,
            child_count=2,
        )
        if review_box is None:
            continue
        review_w = review_box[2] - review_box[0]
        review_h = review_box[3] - review_box[1]
        if review_w < 80 or review_h < 40:
            continue
        if any(
            review_box[0] <= float(existing.get("centroid", [0, 0])[0]) <= review_box[2]
            and review_box[1] <= float(existing.get("centroid", [0, 0])[1]) <= review_box[3]
            for existing in existing_boxes
        ):
            continue

        boxes.append(
            {
                "box": box,
                "area": area,
                "density": float(density),
                "centroid": [float(cx), float(cy)],
                "source": "micro_text_candidate",
                "force_text_line_review": True,
            }
        )

    boxes.sort(key=lambda item: (int(item["area"]), float(item["density"])), reverse=True)
    return boxes[:max_candidates]


def suppress_duplicate_micro_text_boxes(
    boxes: Sequence[Dict[str, Any]],
    standard_boxes: Sequence[Dict[str, Any]],
    foreground_context_mask: np.ndarray,
    shape: Sequence[int],
    *,
    padding: int,
    min_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    for item in boxes:
        if item.get("source") != "micro_text_candidate":
            kept.append(item)
            continue
        review_box = expand_text_line_review_box(
            item.get("box", [0, 0, 0, 0]),
            shape,
            foreground_context_mask,
            padding=padding,
            min_size=min_size,
            child_count=2,
        )
        duplicate = False
        if review_box is not None:
            duplicate = any(
                review_box[0] <= float(existing.get("centroid", [0, 0])[0]) <= review_box[2]
                and review_box[1] <= float(existing.get("centroid", [0, 0])[1]) <= review_box[3]
                for existing in standard_boxes
            )
        if duplicate:
            copied = dict(item)
            copied["suppressed_reason"] = "same text-line already has a standard candidate"
            suppressed.append(copied)
        else:
            kept.append(item)
    return kept, suppressed


def draw_boxes(image: np.ndarray, boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    output = image.copy()
    for index, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            output,
            f"D{index}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return output


def make_overlay(target: np.ndarray, missing: np.ndarray, extra: np.ndarray) -> np.ndarray:
    output = target.copy()
    red = np.zeros_like(output)
    red[:, :] = (0, 0, 255)
    blue = np.zeros_like(output)
    blue[:, :] = (255, 0, 0)
    output = np.where(
        missing[:, :, None] > 0,
        (0.55 * output + 0.45 * red).astype(np.uint8),
        output,
    )
    return np.where(
        extra[:, :, None] > 0,
        (0.55 * output + 0.45 * blue).astype(np.uint8),
        output,
    )


def make_panel(images: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
    max_height = max(image.shape[0] for _, image in images)
    rendered = []
    for label, image in images:
        scale = max_height / image.shape[0]
        item = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), max_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            item,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        rendered.append(item)
    separator = np.full((max_height, 12, 3), 255, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for index, image in enumerate(rendered):
        if index:
            parts.append(separator)
        parts.append(image)
    return np.hstack(parts)


def scale_boxes(
    boxes: Sequence[Dict[str, Any]],
    *,
    scale: float,
    shape: Sequence[int],
) -> List[Dict[str, Any]]:
    if scale <= 0:
        scale = 1.0
    height, width = int(shape[0]), int(shape[1])
    scaled = []
    for item in boxes:
        copied = dict(item)
        x1, y1, x2, y2 = [int(round(float(v) / scale)) for v in item["box"]]
        copied["box"] = [
            min(max(x1, 0), width - 1),
            min(max(y1, 0), height - 1),
            min(max(x2, 0), width - 1),
            min(max(y2, 0), height - 1),
        ]
        scaled.append(copied)
    return scaled


def _box_values(item: Dict[str, Any] | Sequence[int]) -> List[int]:
    if isinstance(item, dict):
        return [int(v) for v in item.get("box", [0, 0, 0, 0])[:4]]
    return [int(v) for v in item[:4]]


def _boxes_near(box_a: Sequence[int], box_b: Sequence[int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = _box_values(box_a)
    bx1, by1, bx2, by2 = _box_values(box_b)
    vertical_gap = max(4, int(round(gap * 0.5)))
    return not (
        ax2 + gap < bx1
        or bx2 + gap < ax1
        or ay2 + vertical_gap < by1
        or by2 + vertical_gap < ay1
    )


def _merge_box_values(boxes: Sequence[Sequence[int]]) -> List[int]:
    x1 = min(_box_values(box)[0] for box in boxes)
    y1 = min(_box_values(box)[1] for box in boxes)
    x2 = max(_box_values(box)[2] for box in boxes)
    y2 = max(_box_values(box)[3] for box in boxes)
    return [x1, y1, x2, y2]


def merge_candidate_boxes(
    boxes: Sequence[Dict[str, Any]],
    *,
    gap: int,
) -> List[Dict[str, Any]]:
    """Merge nearby raw diff boxes so VLM reviews one semantic candidate."""
    groups: List[List[Dict[str, Any]]] = []
    for item in boxes:
        item_box = _box_values(item)
        matching_indexes = [
            idx
            for idx, group in enumerate(groups)
            if any(_boxes_near(item_box, member.get("box", [0, 0, 0, 0]), gap) for member in group)
        ]
        if not matching_indexes:
            groups.append([item])
            continue

        first = matching_indexes[0]
        groups[first].append(item)
        for idx in reversed(matching_indexes[1:]):
            groups[first].extend(groups.pop(idx))

    merged: List[Dict[str, Any]] = []
    for group in groups:
        group_boxes = [item.get("box", [0, 0, 0, 0]) for item in group]
        box = _merge_box_values(group_boxes)
        x1, y1, x2, y2 = box
        area = int(sum(int(item.get("area", 0)) for item in group))
        density = area / max(1.0, float((x2 - x1) * (y2 - y1)))
        if area > 0:
            cx = sum(float(item.get("centroid", [0, 0])[0]) * int(item.get("area", 0)) for item in group) / area
            cy = sum(float(item.get("centroid", [0, 0])[1]) * int(item.get("area", 0)) for item in group) / area
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
        merged.append(
            {
                "box": box,
                "area": area,
                "density": float(density),
                "centroid": [float(cx), float(cy)],
                "child_count": len(group),
                "child_boxes": [dict(item) for item in group],
                "source": (
                    "micro_text_candidate"
                    if any(item.get("source") == "micro_text_candidate" for item in group)
                    else None
                ),
                "force_text_line_review": any(
                    bool(item.get("force_text_line_review")) for item in group
                ),
            }
        )

    merged.sort(key=lambda item: int(item["area"]), reverse=True)
    return merged


def image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("局部图片编码失败")
    return base64.b64encode(buffer).decode("ascii")


def expand_review_box(
    box: Sequence[int],
    shape: Sequence[int],
    *,
    padding: int,
    min_size: int,
) -> List[int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    aspect = box_w / float(box_h)
    if aspect >= 3.0:
        pad_x = max(int(padding), int(round(box_h * 0.6)))
        pad_y = max(int(padding), int(round(box_h * 0.5)))
    else:
        dynamic_padding = max(int(padding), int(round(max(box_w, box_h) * 0.75)))
        pad_x = dynamic_padding
        pad_y = dynamic_padding
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    review_w = max(box_w + pad_x * 2, int(min_size))
    review_h = max(box_h + pad_y * 2, int(min_size))
    rx1 = int(round(cx - review_w / 2.0))
    ry1 = int(round(cy - review_h / 2.0))
    rx2 = int(round(cx + review_w / 2.0))
    ry2 = int(round(cy + review_h / 2.0))

    if rx1 < 0:
        rx2 = min(width, rx2 - rx1)
        rx1 = 0
    if ry1 < 0:
        ry2 = min(height, ry2 - ry1)
        ry1 = 0
    if rx2 > width:
        rx1 = max(0, rx1 - (rx2 - width))
        rx2 = width
    if ry2 > height:
        ry1 = max(0, ry1 - (ry2 - height))
        ry2 = height
    return [rx1, ry1, rx2, ry2]


def _contiguous_true_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(values.astype(bool).tolist()):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(values.shape[0])))
    return runs


def _run_overlapping_box(
    runs: Sequence[Tuple[int, int]],
    start: int,
    end: int,
    center: int,
) -> Tuple[int, int] | None:
    overlapping = [run for run in runs if not (run[1] < start or end < run[0])]
    if overlapping:
        return min(run[0] for run in overlapping), max(run[1] for run in overlapping)
    for run in runs:
        if run[0] <= center <= run[1]:
            return run
    return None


def expand_text_line_review_box(
    box: Sequence[int],
    shape: Sequence[int],
    foreground_context_mask: np.ndarray,
    *,
    padding: int,
    min_size: int,
    child_count: int,
) -> List[int] | None:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    if child_count < 2 and box_w < 32:
        return None
    if box_w > max(180, int(round(box_h * 10.0))):
        return None
    if box_h > 45 or box_w > int(width * 0.55):
        return None

    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    y_margin = max(18, int(round(box_h * 2.0)))
    search_y1 = max(0, y1 - y_margin)
    search_y2 = min(height, y2 + y_margin)
    if search_y2 <= search_y1:
        return None

    search_band = foreground_context_mask[search_y1:search_y2, :]
    row_counts = np.count_nonzero(search_band, axis=1)
    row_threshold = max(5, int(round(width * 0.008)))
    row_runs = _contiguous_true_runs(row_counts >= row_threshold)
    local_cy = min(max(cy - search_y1, 0), search_band.shape[0] - 1)
    row_run = _run_overlapping_box(
        row_runs,
        max(0, y1 - search_y1),
        min(search_band.shape[0] - 1, y2 - search_y1),
        local_cy,
    )
    if row_run is None:
        line_y1, line_y2 = y1, y2
    else:
        line_y1, line_y2 = search_y1 + row_run[0], search_y1 + row_run[1]

    y_pad = max(int(padding), int(round(box_h * 0.8)))
    line_y1 = max(0, line_y1 - y_pad)
    line_y2 = min(height, line_y2 + y_pad)
    if line_y2 <= line_y1:
        return None

    line_band = foreground_context_mask[line_y1:line_y2, :]
    col_counts = np.count_nonzero(line_band, axis=0)
    col_threshold = max(1, int(round((line_y2 - line_y1) * 0.08)))
    col_active = (col_counts >= col_threshold).astype(np.uint8)
    close_width = max(18, min(64, int(round(box_h * 2.4))))
    kernel = np.ones((1, close_width), dtype=np.uint8)
    col_active = cv2.morphologyEx(col_active.reshape(1, -1), cv2.MORPH_CLOSE, kernel)[0] > 0
    col_runs = _contiguous_true_runs(col_active)
    col_run = _run_overlapping_box(col_runs, x1, x2, cx)
    if col_run is None:
        return None

    x_pad = max(int(padding), int(round(box_h * 0.8)))
    review_x1 = max(0, col_run[0] - x_pad)
    review_x2 = min(width, col_run[1] + x_pad)
    review_y1 = line_y1
    review_y2 = line_y2

    if (review_x2 - review_x1) < box_w or (review_y2 - review_y1) < box_h:
        return None
    if (review_x2 - review_x1) < int(min_size):
        extra = int(min_size) - (review_x2 - review_x1)
        review_x1 = max(0, review_x1 - extra // 2)
        review_x2 = min(width, review_x2 + extra - extra // 2)
    if (review_y2 - review_y1) < int(min_size):
        extra = int(min_size) - (review_y2 - review_y1)
        review_y1 = max(0, review_y1 - extra // 2)
        review_y2 = min(height, review_y2 + extra - extra // 2)

    return [int(review_x1), int(review_y1), int(review_x2), int(review_y2)]


def crop_box(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image[y1:y2, x1:x2].copy()


def draw_local_candidate_box(
    crop: np.ndarray,
    candidate_box: Sequence[int],
    review_box: Sequence[int],
    *,
    label: bool = True,
) -> np.ndarray:
    output = crop.copy()
    x1, y1, x2, y2 = _box_values(candidate_box)
    rx1, ry1, _, _ = _box_values(review_box)
    local_box = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]
    lx1, ly1, lx2, ly2 = [
        min(max(int(v), 0), output.shape[1 if idx % 2 == 0 else 0] - 1)
        for idx, v in enumerate(local_box)
    ]
    cv2.rectangle(output, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
    if label:
        cv2.putText(
            output,
            "candidate",
            (lx1, max(18, ly1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    return output


def build_vlm_filter_prompt(
    box: Sequence[int],
    review_box: Sequence[int],
    template_foreground: float,
    target_foreground: float,
    child_count: int,
    candidate_source: str | None = None,
) -> str:
    micro_text_note = ""
    if candidate_source == "micro_text_candidate":
        micro_text_note = """
This candidate came from a sensitive small text detector for numbers/units.
For this candidate, be conservative:
- Keep only if the surrounding word, number, unit, or superscript/subscript is actually changed.
- Discard isolated dots, tiny specks, dust, ink blobs, red annotation marks, compression noise, or local edge artifacts, even if visible in only one crop.
- Do not treat a tiny standalone mark as a real difference unless it changes a readable character, digit, or unit.
"""
    return f"""You are checking one candidate difference from a product label inspection.

You will see three cropped images:
- Image 1 is the template review crop.
- Image 2 is the target review crop.
- Image 3 is the target review crop with the candidate region marked by a red box.

The candidate box on the comparison image is {list(box)}.
The larger review crop box on the comparison image is {list(review_box)}.
This candidate merges {child_count} raw diff component(s).
Foreground ratios inside the review crops:
- template: {template_foreground:.4f}
- target: {target_foreground:.4f}
{micro_text_note}

Decide whether the red-box candidate region, using its surrounding context, is a real label-content difference.

Keep real differences:
- text character changes, including case changes or typos
- number, unit, model, date, barcode value changes
- icon/symbol/logo/barcode graphic added, removed, or meaningfully changed
- missing or extra printed content
- any printed icon, symbol, mark, or text that is visible in one crop but absent in the other, even if it is small or looks decorative

Discard false positives:
- slight alignment shift
- blur, exposure, shadow, compression, camera noise
- line thickness, ink darkness, antialiasing, small edge residue
- white/black border artifacts from alignment
- crop boundary artifacts

Important:
- If one review crop is blank or nearly blank around the red box and the other review crop contains a visible printed black mark, icon, symbol, or text at that location, output keep.
- Do not discard added or missing printed symbols just because they seem minor, decorative, or hard to interpret.
- Do not require the red box to contain the complete object. It may only cover the changed pixels; use nearby context inside the review crop.

Output exactly one JSON object and nothing else:
{{"decision":"keep|discard|unknown","confidence":0.0,"reason":"short reason"}}
"""


def normalize_vlm_filter_response(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if payload is None:
        return {
            "decision": "unknown",
            "confidence": 0.0,
            "reason": "模型输出无法解析为 JSON",
        }
    decision = str(payload.get("decision") or "unknown").strip().lower()
    if decision not in {"keep", "discard", "unknown"}:
        decision = "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "decision": decision,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(payload.get("reason") or ""),
    }


def foreground_ratio(image: np.ndarray) -> float:
    mask = foreground_mask(image)
    return float(np.count_nonzero(mask) / max(1, mask.size))


def apply_vlm_filter(
    boxes: Sequence[Dict[str, Any]],
    template_image: np.ndarray,
    target_image: np.ndarray,
    *,
    model: str,
    api_base: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    crop_padding: int,
    review_min_size: int,
    keep_unknown: bool,
    skip_model_check: bool,
    debug_dir: Path | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    client = OpenAICompatibleHTTPClient(
        model_name=model,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        num_predict=max_tokens,
        check_model=not skip_model_check,
    )
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    foreground_context_mask = cv2.bitwise_or(
        foreground_mask(template_image),
        foreground_mask(target_image),
    )

    for index, item in enumerate(boxes, start=1):
        box = item.get("box") or [0, 0, 0, 0]
        review_source = "local"
        review_box = expand_text_line_review_box(
            box,
            target_image.shape[:2],
            foreground_context_mask,
            padding=crop_padding,
            min_size=review_min_size,
            child_count=(
                2
                if item.get("force_text_line_review")
                else int(item.get("child_count", 1))
            ),
        )
        if review_box is not None:
            review_source = "text_line"
        else:
            review_box = expand_review_box(
                box,
                target_image.shape[:2],
                padding=crop_padding,
                min_size=review_min_size,
            )
        template_crop = crop_box(template_image, review_box)
        target_crop = crop_box(target_image, review_box)
        if item.get("source") == "micro_text_candidate":
            target_annotated = draw_local_candidate_box(
                target_crop,
                review_box,
                review_box,
                label=False,
            )
        else:
            target_annotated = draw_local_candidate_box(target_crop, box, review_box)
        template_foreground = foreground_ratio(template_crop)
        target_foreground = foreground_ratio(target_crop)

        foreground_presence_signal = (
            min(template_foreground, target_foreground) < 0.01
            and max(template_foreground, target_foreground) >= 0.03
        )

        prompt = build_vlm_filter_prompt(
            box,
            review_box,
            template_foreground,
            target_foreground,
            int(item.get("child_count", 1)),
            str(item.get("source") or ""),
        )
        response = client.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_to_base64(template_crop),
                        image_to_base64(target_crop),
                        image_to_base64(target_annotated),
                    ],
                }
            ],
            json_mode=True,
            timeout=timeout,
            num_predict=max_tokens,
        )
        parsed = normalize_vlm_filter_response(extract_json_object(response.content))
        record = {
            **dict(item),
            "review_box": review_box,
            "review_box_source": review_source,
            "vlm_decision": parsed["decision"],
            "vlm_confidence": parsed["confidence"],
            "vlm_reason": parsed["reason"],
            "filter_source": "vlm",
            "template_foreground_ratio": template_foreground,
            "target_foreground_ratio": target_foreground,
            "foreground_presence_signal": foreground_presence_signal,
            "raw_response": response.content,
        }
        decisions.append(record)
        if parsed["decision"] == "keep" or (keep_unknown and parsed["decision"] == "unknown"):
            kept.append(record)

        if debug_dir is not None:
            imwrite(debug_dir / f"candidate_{index:02d}_template.jpg", template_crop)
            imwrite(debug_dir / f"candidate_{index:02d}_target.jpg", target_crop)
            imwrite(debug_dir / f"candidate_{index:02d}_target_marked.jpg", target_annotated)
            (debug_dir / f"candidate_{index:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return kept, decisions


def _run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), args.run_name)

    preprocess_root: Path
    temp_context = None
    if args.save_debug:
        preprocess_root = output_dir
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="traditional_full_image_diff_")
        preprocess_root = Path(temp_context.name)

    try:
        template_img, target_img, input_summary = resolve_and_preprocess(
            args.template,
            args.target,
            preprocess_root,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()
            temp_context = None

    if args.save_debug:
        imwrite(output_dir / "template_preprocessed.jpg", template_img)
        imwrite(output_dir / "target_preprocessed.jpg", target_img)

    aligned_target, alignment = align_target_to_template(
        template_img,
        target_img,
        skip_feature_align=args.skip_feature_align,
        skip_ecc=args.skip_ecc,
        ecc_motion=args.ecc_motion,
    )
    if args.save_debug:
        imwrite(output_dir / "target_aligned.jpg", aligned_target)

    template_panel, target_panel, panel_scale = scale_pair_for_canvas(
        template_img,
        aligned_target,
        args.panel_width,
    )

    template_mask = foreground_mask(template_panel)
    target_mask = foreground_mask(target_panel)
    missing, extra, diff_mask = tolerant_diff(
        template_mask,
        target_mask,
        tolerance=args.diff_tolerance,
        merge_radius=args.merge_radius,
    )
    standard_raw_boxes = extract_boxes(
        diff_mask,
        min_area=args.min_area,
        min_side=args.min_side,
        min_density=args.min_density,
        ignore_edge_margin=args.ignore_edge_margin,
        max_boxes=args.max_boxes,
    )
    small_text_boxes: List[Dict[str, Any]] = []
    small_missing = np.zeros_like(diff_mask)
    small_extra = np.zeros_like(diff_mask)
    small_diff_mask = np.zeros_like(diff_mask)
    if not args.disable_small_text_candidates:
        small_missing, small_extra, small_diff_mask = tolerant_diff(
            template_mask,
            target_mask,
            tolerance=min(1, int(args.diff_tolerance)),
            merge_radius=min(3, int(args.merge_radius)),
        )
        small_text_boxes = extract_small_text_candidates(
            small_diff_mask,
            standard_raw_boxes,
            cv2.bitwise_or(template_mask, target_mask),
            ignore_edge_margin=args.ignore_edge_margin,
            normal_min_area=args.min_area,
        )
    raw_boxes = sorted(
        [*standard_raw_boxes, *small_text_boxes],
        key=lambda item: int(item.get("area", 0)),
        reverse=True,
    )[: args.max_boxes]
    merged_boxes = merge_candidate_boxes(
        raw_boxes,
        gap=args.candidate_merge_gap,
    )
    merged_boxes, suppressed_micro_text_boxes = suppress_duplicate_micro_text_boxes(
        merged_boxes,
        standard_raw_boxes,
        cv2.bitwise_or(template_mask, target_mask),
        target_panel.shape[:2],
        padding=args.crop_padding,
        min_size=args.review_min_size,
    )

    overlay = make_overlay(target_panel, missing, extra)
    boxed = draw_boxes(target_panel, merged_boxes if args.vlm_filter else raw_boxes)
    final_boxes_scaled = raw_boxes
    vlm_decisions: List[Dict[str, Any]] = []
    if args.vlm_filter and merged_boxes:
        debug_dir = output_dir / "vlm_filter_debug" if args.save_debug else None
        final_boxes_scaled, vlm_decisions = apply_vlm_filter(
            merged_boxes,
            template_panel,
            target_panel,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            crop_padding=args.crop_padding,
            review_min_size=args.review_min_size,
            keep_unknown=args.keep_unknown,
            skip_model_check=args.skip_model_check,
            debug_dir=debug_dir,
        )
    aligned_boxes = scale_boxes(final_boxes_scaled, scale=panel_scale, shape=aligned_target.shape[:2])
    raw_candidate_boxes_final = scale_boxes(raw_boxes, scale=panel_scale, shape=aligned_target.shape[:2])
    merged_candidate_boxes_final = scale_boxes(merged_boxes, scale=panel_scale, shape=aligned_target.shape[:2])
    final_result = draw_boxes(aligned_target, aligned_boxes)

    final_result_path = output_dir / "final_result.jpg"
    imwrite(final_result_path, final_result)

    artifacts: Dict[str, str] = {"final_result": str(final_result_path)}
    if args.save_debug:
        debug_paths = {
            "template_foreground_mask": output_dir / "template_foreground_mask.png",
            "target_foreground_mask": output_dir / "target_foreground_mask.png",
            "missing_from_target_mask": output_dir / "missing_from_target_mask.png",
            "extra_in_target_mask": output_dir / "extra_in_target_mask.png",
            "diff_mask": output_dir / "diff_mask.png",
            "small_text_diff_mask": output_dir / "small_text_diff_mask.png",
            "diff_overlay": output_dir / "diff_overlay.jpg",
            "target_diff_boxes_scaled": output_dir / "target_diff_boxes_scaled.jpg",
            "comparison_panel": output_dir / "comparison_panel.jpg",
        }
        imwrite(debug_paths["template_foreground_mask"], template_mask)
        imwrite(debug_paths["target_foreground_mask"], target_mask)
        imwrite(debug_paths["missing_from_target_mask"], missing)
        imwrite(debug_paths["extra_in_target_mask"], extra)
        imwrite(debug_paths["diff_mask"], diff_mask)
        imwrite(debug_paths["small_text_diff_mask"], small_diff_mask)
        imwrite(debug_paths["diff_overlay"], overlay)
        imwrite(debug_paths["target_diff_boxes_scaled"], boxed)
        imwrite(
            debug_paths["comparison_panel"],
            make_panel(
                [
                    ("template", template_panel),
                    ("target", target_panel),
                    ("diff overlay", overlay),
                    ("boxes", boxed),
                ]
            ),
        )
        artifacts.update({name: str(path) for name, path in debug_paths.items()})

    result = {
        "success": True,
        "verdict": "不一致" if final_boxes_scaled else "通过",
        "output_dir": str(output_dir),
        "template": args.template,
        "target": args.target,
        "summary": {
            "template_input": str(args.template),
            "template_source_type": input_summary.get("template_source_type"),
            "target_input": str(args.target),
        },
        "alignment": alignment,
        "image_scale": {
            "scale_from_aligned_images": panel_scale,
            "comparison_shape": list(template_panel.shape[:2]),
            "final_result_shape": list(final_result.shape[:2]),
        },
        "parameters": {
            "diff_tolerance": args.diff_tolerance,
            "merge_radius": args.merge_radius,
            "min_area": args.min_area,
            "min_side": args.min_side,
            "min_density": args.min_density,
            "ignore_edge_margin": args.ignore_edge_margin,
            "max_boxes": args.max_boxes,
            "small_text_candidates": not args.disable_small_text_candidates,
            "candidate_merge_gap": args.candidate_merge_gap,
            "skip_feature_align": args.skip_feature_align,
            "skip_ecc": args.skip_ecc,
            "ecc_motion": args.ecc_motion,
            "save_debug": args.save_debug,
            "vlm_filter": args.vlm_filter,
            "model": args.model if args.vlm_filter else None,
            "api_base": args.api_base if args.vlm_filter else None,
            "crop_padding": args.crop_padding,
            "review_min_size": args.review_min_size,
            "keep_unknown": args.keep_unknown,
        },
        "foreground_pixels": {
            "template": int(np.count_nonzero(template_mask)),
            "target": int(np.count_nonzero(target_mask)),
            "missing_from_target": int(np.count_nonzero(missing)),
            "extra_in_target": int(np.count_nonzero(extra)),
            "diff": int(np.count_nonzero(diff_mask)),
        },
        "standard_raw_candidate_box_count": len(standard_raw_boxes),
        "small_text_candidate_box_count": len(small_text_boxes),
        "suppressed_micro_text_candidate_box_count": len(suppressed_micro_text_boxes),
        "raw_candidate_box_count": len(raw_boxes),
        "merged_candidate_box_count": len(merged_boxes),
        "candidate_box_count": len(merged_boxes if args.vlm_filter else raw_boxes),
        "final_box_count": len(final_boxes_scaled),
        "diff_box_count": len(final_boxes_scaled),
        "raw_candidate_boxes_scaled": raw_boxes,
        "raw_candidate_boxes_final": raw_candidate_boxes_final,
        "merged_candidate_boxes_scaled": merged_boxes,
        "merged_candidate_boxes_final": merged_candidate_boxes_final,
        "candidate_boxes_scaled": merged_boxes if args.vlm_filter else raw_boxes,
        "candidate_boxes_final": merged_candidate_boxes_final if args.vlm_filter else raw_candidate_boxes_final,
        "vlm_filter_results": vlm_decisions,
        "final_boxes_scaled": final_boxes_scaled,
        "final_boxes": aligned_boxes,
        "artifacts": artifacts,
    }
    write_json(output_dir / "result.json", result)

    latest_dir = Path(args.output_dir).resolve() / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    latest_dir.symlink_to(output_dir, target_is_directory=True)

    print("=" * 72)
    print("Traditional full-image diff experiment complete")
    print("=" * 72)
    print(f"Raw candidate boxes: {len(raw_boxes)}")
    print(f"Merged candidate boxes: {len(merged_boxes)}")
    print(f"Final boxes: {len(final_boxes_scaled)}")
    print(f"Output: {output_dir}")
    print(f"Final result: {final_result_path}")
    print(f"Result: {output_dir / 'result.json'}")
    return result


def run_traditional_full_image_diff(
    template: str | Path,
    target: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    run_name: Optional[str] = None,
    output_mode: str = "final",
    save_debug: Optional[bool] = None,
    panel_width: int = 1024,
    skip_feature_align: bool = False,
    skip_ecc: bool = False,
    ecc_motion: str = "translation",
    diff_tolerance: int = 3,
    merge_radius: int = 6,
    min_area: int = 80,
    min_side: int = 6,
    min_density: float = 0.02,
    ignore_edge_margin: int = 8,
    max_boxes: int = 50,
    small_text_candidates: bool = True,
    candidate_merge_gap: int = 18,
    vlm_filter: bool = True,
    model: Optional[str] = None,
    api_base: str = OPENAI_COMPATIBLE_API_BASE,
    api_key: str = OPENAI_COMPATIBLE_API_KEY,
    timeout: int = VLM_TIMEOUT,
    max_tokens: int = 256,
    crop_padding: int = 32,
    review_min_size: int = 128,
    keep_unknown: bool = False,
    skip_model_check: bool = False,
) -> Dict[str, Any]:
    """Run the traditional full-image diff workflow and return the result payload."""
    normalized_mode = str(output_mode or "final").strip().lower()
    if normalized_mode not in {"final", "debug"}:
        raise ValueError(f"不支持的 output_mode: {output_mode}")

    args = argparse.Namespace(
        template=str(template),
        target=str(target),
        output_dir=str(output_dir),
        run_name=run_name,
        save_debug=(normalized_mode == "debug") if save_debug is None else bool(save_debug),
        panel_width=panel_width,
        skip_feature_align=skip_feature_align,
        skip_ecc=skip_ecc,
        ecc_motion=ecc_motion,
        diff_tolerance=diff_tolerance,
        merge_radius=merge_radius,
        min_area=min_area,
        min_side=min_side,
        min_density=min_density,
        ignore_edge_margin=ignore_edge_margin,
        max_boxes=max_boxes,
        disable_small_text_candidates=not small_text_candidates,
        candidate_merge_gap=candidate_merge_gap,
        vlm_filter=vlm_filter,
        model=model or OPENAI_COMPATIBLE_MODEL,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        crop_padding=crop_padding,
        review_min_size=review_min_size,
        keep_unknown=keep_unknown,
        skip_model_check=skip_model_check,
    )
    result = _run_from_args(args)
    result["output_mode"] = normalized_mode
    return result


def main() -> int:
    args = build_arg_parser().parse_args()
    _run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
