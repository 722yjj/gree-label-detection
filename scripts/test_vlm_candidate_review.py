"""Direct-run experiment: traditional candidate generation + VLM candidate review."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import requests

from label_detection.core.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    PROJECT_ROOT as REPO_ROOT,
    VLM_CANVAS_SIZE,
    VLM_MAX_RETRIES,
    VLM_NUM_PREDICT,
    VLM_TIMEOUT,
)
from label_detection.extraction.pdf import extract_red_box_info
from label_detection.matching.layout import (
    detect_barcode_region,
    detect_layout_regions,
    draw_regions,
    extract_regions_by_type,
    match_regions,
)
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template
from label_detection.services.ocr_service import get_ocr_with_boxes
from label_detection.services.vlm_service import get_vlm_comparator


# ==================== 直接在这里改测试参数 ====================
TEST_NAME = "vlm_candidate_review"

PDF_PATH = REPO_ROOT / "samples" / "pdfs" / "600004075219-01.pdf"
TARGET_IMAGE_PATH = REPO_ROOT / "samples" / "images" / "produce" / "type1" / "600004075219_1.jpg"

USE_PROJECT_PREPROCESSING = True
REGION_TYPE = "image"
LAYOUT_THRESHOLD = 0.30
MATCH_COST_THRESHOLD = 0.60
PAIR_INDEX = None  # None 表示跑所有匹配对；填 0/1/2... 只跑指定区域对

OUTPUT_DIR = REPO_ROOT / "results" / "vlm_candidate_review"
VLM_MODEL = OLLAMA_MODEL
VLM_API_BASE = OLLAMA_API_BASE
VLM_TIMEOUT_SECONDS = VLM_TIMEOUT

NORMALIZE_MARGIN = 8
TEXTURE_ASPECT_MIN = 1.8
TEXTURE_TRANSITION_MIN = 0.08
TEXTURE_VERTICAL_BIAS_MIN = 1.6
TEXTURE_GRADIENT_BIAS_MIN = 1.8
TEXTURE_ROW_TRANSITION_MAX = 0.18
LOCAL_BARCODE_CLOSE_WIDTH = 17
LOCAL_BARCODE_CLOSE_HEIGHT = 3
LOCAL_BARCODE_DILATE_WIDTH = 11
LOCAL_BARCODE_DILATE_HEIGHT = 5
LOCAL_BARCODE_BAND_WINDOW = 11
LOCAL_BARCODE_BAND_EDGE_MIN = 0.20
LOCAL_BARCODE_BAND_EDGE_FLOOR = 0.08
LOCAL_BARCODE_BAND_DARK_MIN = 0.08
LOCAL_BARCODE_BAND_DARK_MAX = 0.88
LOCAL_BARCODE_BAND_MIN_WIDTH_RATIO = 0.18
LOCAL_BARCODE_BAND_MIN_START_RATIO = 0.35
LOCAL_BARCODE_BAND_CLOSE_GAP = 9
LOCAL_BARCODE_BAND_ADAPTIVE_PERCENTILE = 72
LOCAL_BARCODE_BAND_ADAPTIVE_SCALE = 0.92
LOCAL_BARCODE_BAND_ADAPTIVE_RIGHT_START_RATIO = 0.28
LOCAL_BARCODE_BAND_ADAPTIVE_MIN_ACTIVE_COLS = 18
LOCAL_BARCODE_MIN_WIDTH_RATIO = 0.16
LOCAL_BARCODE_MIN_HEIGHT_RATIO = 0.16
LOCAL_BARCODE_MIN_AREA_RATIO = 0.008
LOCAL_BARCODE_BOX_PAD_X = 10
LOCAL_BARCODE_BOX_PAD_TOP = 4
LOCAL_BARCODE_BOX_PAD_BOTTOM_RATIO = 0.45
LOCAL_BARCODE_MERGE_GAP = 18
SALVAGE_COMPONENT_MIN_AREA_RATIO = 0.006
SALVAGE_COMPONENT_MIN_SIZE = 10
SALVAGE_COMPONENT_PAD_X = 6
SALVAGE_COMPONENT_PAD_Y = 6
SALVAGE_COMPONENT_MERGE_GAP = 14

DIFF_TOLERANCE_KERNEL = 9
CANDIDATE_CLOSE_KERNEL = 11
CANDIDATE_DILATE_KERNEL = 7
CANDIDATE_MIN_AREA_RATIO = 0.003
CANDIDATE_MERGE_GAP = 16
CANDIDATE_CONTEXT_PAD = 20
MAX_CANDIDATES_PER_PAIR = 5
REVIEW_MIN_DIFF_PIXELS = 130
REVIEW_MIN_SIDE_DIFF_PIXELS = 90
REVIEW_MIN_DOMINANCE_RATIO = 2.2
FINAL_SINGLE_SIDED_HINTS = {"target_extra", "template_extra"}
FINAL_BACKGROUND_REJECT_MAX_DIFF_PIXELS = 320
FINAL_BACKGROUND_REJECT_MAX_SIDE_PIXELS = 220
FINAL_SMALL_SINGLE_SIDED_RATIO = 0.25

CANDIDATE_REVIEW_PROMPT = """你是局部图形差异复核器。
你将看到一个 2x2 复核画布：
- 左上：Template 整体区域，红框标出候选位置
- 右上：Target 整体区域，红框标出候选位置
- 左下：Template 红框区域放大图
- 右下：Target 红框区域放大图

你的任务只有一个：判断红框内的深色前景图形结构是否存在真实差异。
只关注黑色/深色前景笔画、轮廓、连通结构是否多出、缺少或被替换。
不要描述图形像什么，不要猜测图标语义。

以下内容必须忽略，不要判为差异：
- 背景颜色不同
- 灰底、黄底、白底或亮度差异
- 纸张底色、阴影、污渍、反光
- 留白差异
- 轻微模糊
- 位置偏移
- 线条粗细不同
- 小范围轮廓边缘残差
- 小于15%的缩放差异

判为 mismatch 只允许基于前景结构差异：
1. 红框内 Target 多出前景图形/符号
2. 红框内 Target 缺少前景图形/符号
3. 红框内主体前景结构被替换成不同结构

如果主体前景图形一致，而只有背景灰白、明暗、底色不同，必须输出 "decision":"match"。

输出规则（必须严格遵守）：
1. 只输出一个 JSON 对象，不要输出任何其他文本。
2. JSON 必须且仅包含以下 4 个键：
   - "decision": "match" | "mismatch" | "unknown"
   - "confidence": 0.0 到 1.0 的数字
   - "differences": 字符串数组
   - "summary": 字符串
3. 如果只是背景差异、边缘厚度变化或轻微对齐误差，输出 "decision":"match"。
4. 如果无法可靠判断前景结构，输出 "decision":"unknown"。
5. 当 decision 为 "match" 时，differences 必须是空数组 []。
6. summary 只描述“是否存在前景结构差异”，不要描述背景差异，不要描述图形类别。

仅输出如下格式的 JSON：
{"decision":"match|mismatch|unknown","confidence":0.0,"differences":[],"summary":"..."}"""


def prepare_test_images(output_dir: Path) -> Dict[str, Dict[str, object]]:
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


def crop_region(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, image.shape[1]))
    x2 = max(0, min(x2, image.shape[1]))
    y1 = max(0, min(y1, image.shape[0]))
    y2 = max(0, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


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


def _map_local_box_to_global(parent_box: Sequence[int], local_box: Sequence[int]) -> List[int]:
    px1, py1, _, _ = [int(v) for v in parent_box]
    lx1, ly1, lx2, ly2 = [int(v) for v in local_box]
    return [px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2]


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def detect_barcode_band_box_with_debug(image: np.ndarray) -> Tuple[List[int] | None, Dict[str, object]]:
    debug: Dict[str, object] = {
        "dark_profile": [],
        "smooth_edge_profile": [],
        "candidate_segments": [],
        "candidate_mask": None,
        "fixed_edge_threshold": float(LOCAL_BARCODE_BAND_EDGE_MIN),
        "edge_threshold_mode": "fixed",
        "edge_threshold": float(LOCAL_BARCODE_BAND_EDGE_MIN),
        "adaptive_edge_threshold": None,
        "adaptive_pool_count": 0,
        "adaptive_right_start": 0,
        "adaptive_pool_stats": {},
    }
    if image.size == 0:
        return None, debug

    img_h, img_w = image.shape[:2]
    if img_h < 24 or img_w < 64:
        return None, debug

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    upper_h = max(18, int(round(img_h * 0.72)))
    upper = (binary[:upper_h] > 0).astype(np.uint8)
    if upper.size == 0:
        return None, debug

    dark_profile = upper.mean(axis=0)
    edge_profile = np.zeros(img_w, dtype=np.float32)
    if img_w > 1:
        edge_profile[1:] = np.abs(np.diff(upper.astype(np.float32), axis=1)).mean(axis=0)
    smooth_edge = _smooth_1d(edge_profile, LOCAL_BARCODE_BAND_WINDOW)
    debug["dark_profile"] = [float(v) for v in dark_profile.tolist()]
    debug["smooth_edge_profile"] = [float(v) for v in smooth_edge.tolist()]

    min_width = max(56, int(round(img_w * LOCAL_BARCODE_BAND_MIN_WIDTH_RATIO)))
    min_start = int(round(img_w * LOCAL_BARCODE_BAND_MIN_START_RATIO))
    dark_ok = (
        (dark_profile >= LOCAL_BARCODE_BAND_DARK_MIN)
        & (dark_profile <= LOCAL_BARCODE_BAND_DARK_MAX)
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LOCAL_BARCODE_BAND_CLOSE_GAP, 1))

    def build_segments(edge_threshold: float) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        candidate_cols = (smooth_edge >= edge_threshold) & dark_ok
        if not np.any(candidate_cols):
            return np.zeros((1, img_w), dtype=np.uint8), []

        candidate_img = (candidate_cols.astype(np.uint8) * 255)[None, :]
        candidate_img = cv2.morphologyEx(candidate_img, cv2.MORPH_CLOSE, close_kernel)
        candidate_cols = candidate_img[0] > 0

        segments: List[Tuple[int, int, float]] = []
        start = None
        for idx, is_on in enumerate(candidate_cols.tolist() + [False]):
            if is_on and start is None:
                start = idx
            elif not is_on and start is not None:
                end = idx
                width = end - start
                if width >= min_width and end >= min_start:
                    score = float(smooth_edge[start:end].mean() * width + end * 0.02)
                    segments.append((start, end, score))
                start = None
        return candidate_img, segments

    candidate_img, segments = build_segments(float(LOCAL_BARCODE_BAND_EDGE_MIN))
    debug["candidate_mask"] = candidate_img.copy()

    if not segments:
        right_start = min(img_w - 1, max(0, int(round(img_w * LOCAL_BARCODE_BAND_ADAPTIVE_RIGHT_START_RATIO))))
        adaptive_pool = smooth_edge[right_start:][dark_ok[right_start:]]
        adaptive_pool = adaptive_pool[adaptive_pool > 0]
        debug["adaptive_right_start"] = int(right_start)
        debug["adaptive_pool_count"] = int(adaptive_pool.size)
        if adaptive_pool.size >= LOCAL_BARCODE_BAND_ADAPTIVE_MIN_ACTIVE_COLS:
            adaptive_percentile = float(
                np.percentile(adaptive_pool, LOCAL_BARCODE_BAND_ADAPTIVE_PERCENTILE)
            )
            adaptive_mean = float(adaptive_pool.mean())
            adaptive_std = float(adaptive_pool.std())
            adaptive_threshold = max(
                LOCAL_BARCODE_BAND_EDGE_FLOOR,
                min(
                    LOCAL_BARCODE_BAND_EDGE_MIN,
                    max(
                        adaptive_percentile * LOCAL_BARCODE_BAND_ADAPTIVE_SCALE,
                        adaptive_mean + adaptive_std * 0.15,
                    ),
                ),
            )
            debug["adaptive_pool_stats"] = {
                "mean": adaptive_mean,
                "std": adaptive_std,
                "percentile": adaptive_percentile,
            }
            debug["adaptive_edge_threshold"] = float(adaptive_threshold)

            adaptive_candidate_img, adaptive_segments = build_segments(adaptive_threshold)
            debug["candidate_mask"] = adaptive_candidate_img.copy()
            debug["edge_threshold_mode"] = "adaptive_probe"
            debug["edge_threshold"] = float(adaptive_threshold)
            if adaptive_segments:
                candidate_img = adaptive_candidate_img
                segments = adaptive_segments
                debug["edge_threshold_mode"] = "adaptive"
                debug["edge_threshold"] = float(adaptive_threshold)

    debug["candidate_segments"] = [
        {"x1": int(seg_start), "x2": int(seg_end), "score": float(score)}
        for seg_start, seg_end, score in segments
    ]
    if not segments:
        return None, debug

    x1, x2, _ = max(segments, key=lambda item: item[2])
    band_pixels = binary[:, x1:x2] > 0
    if not np.any(band_pixels):
        return None, debug

    rows = np.where(np.any(band_pixels, axis=1))[0]
    if rows.size == 0:
        return None, debug

    local_box = clip_box(
        [
            int(x1 - LOCAL_BARCODE_BOX_PAD_X),
            int(rows.min() - LOCAL_BARCODE_BOX_PAD_TOP),
            int(x2 + LOCAL_BARCODE_BOX_PAD_X),
            int(rows.max() + 1 + max(10, int(round((rows.max() - rows.min() + 1) * LOCAL_BARCODE_BOX_PAD_BOTTOM_RATIO)))),
        ],
        image.shape[:2],
    )
    debug["proposed_box"] = [int(v) for v in local_box]

    region = {
        "label": "image",
        "score": 1.0,
        "coordinate": [int(v) for v in local_box],
    }
    is_barcode, _ = detect_barcode_region(region, image, ocr_boxes=None)
    if not is_barcode:
        upper_region = dict(region)
        upper_region["coordinate"] = [
            int(local_box[0]),
            int(local_box[1]),
            int(local_box[2]),
            int(min(image.shape[0], local_box[1] + max(18, int(round((local_box[3] - local_box[1]) * 0.72))))),
        ]
        is_barcode, _ = detect_barcode_region(upper_region, image, ocr_boxes=None)
        if not is_barcode:
            return None, debug

    return local_box, debug


def detect_barcode_band_box(image: np.ndarray) -> List[int] | None:
    box, _ = detect_barcode_band_box_with_debug(image)
    return box


def detect_local_barcode_boxes_with_debug(image: np.ndarray) -> Tuple[List[List[int]], Dict[str, object]]:
    debug: Dict[str, object] = {
        "binary_mask": None,
        "morph_mask": None,
        "component_candidate_boxes": [],
        "band_box": None,
        "band_debug": {},
        "merged_boxes": [],
    }
    if image.size == 0:
        return [], debug

    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug["binary_mask"] = binary.copy()

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (LOCAL_BARCODE_CLOSE_WIDTH, LOCAL_BARCODE_CLOSE_HEIGHT),
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (LOCAL_BARCODE_DILATE_WIDTH, LOCAL_BARCODE_DILATE_HEIGHT),
    )
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    merged = cv2.dilate(merged, dilate_kernel, iterations=1)
    debug["morph_mask"] = merged.copy()

    min_width = max(56, int(round(img_w * LOCAL_BARCODE_MIN_WIDTH_RATIO)))
    min_height = max(22, int(round(img_h * LOCAL_BARCODE_MIN_HEIGHT_RATIO)))
    min_area = img_h * img_w * LOCAL_BARCODE_MIN_AREA_RATIO

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), connectivity=8)
    raw_boxes: List[List[int]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area or w < min_width or h < min_height:
            continue

        region = {
            "label": "image",
            "score": 1.0,
            "coordinate": [int(x), int(y), int(x + w), int(y + h)],
        }
        is_barcode, _ = detect_barcode_region(region, image, ocr_boxes=None)
        if not is_barcode:
            upper_h = max(12, int(round(h * 0.72)))
            upper_region = {
                "label": "image",
                "score": 1.0,
                "coordinate": [int(x), int(y), int(x + w), int(y + upper_h)],
            }
            is_barcode, _ = detect_barcode_region(upper_region, image, ocr_boxes=None)
            if not is_barcode:
                continue

        pad_bottom = max(10, int(round(h * LOCAL_BARCODE_BOX_PAD_BOTTOM_RATIO)))
        padded = clip_box(
            [
                int(x - LOCAL_BARCODE_BOX_PAD_X),
                int(y - LOCAL_BARCODE_BOX_PAD_TOP),
                int(x + w + LOCAL_BARCODE_BOX_PAD_X),
                int(y + h + pad_bottom),
            ],
            image.shape[:2],
        )
        raw_boxes.append(padded)

    debug["component_candidate_boxes"] = [[int(v) for v in box] for box in raw_boxes]
    band_box, band_debug = detect_barcode_band_box_with_debug(image)
    debug["band_debug"] = band_debug
    if band_box is not None:
        raw_boxes.append(band_box)
        debug["band_box"] = [int(v) for v in band_box]

    if not raw_boxes:
        return [], debug

    merged_boxes = merge_boxes(raw_boxes, LOCAL_BARCODE_MERGE_GAP)
    debug["merged_boxes"] = [[int(v) for v in box] for box in merged_boxes]
    return merged_boxes, debug


def detect_local_barcode_boxes(image: np.ndarray) -> List[List[int]]:
    boxes, _ = detect_local_barcode_boxes_with_debug(image)
    return boxes


def salvage_non_barcode_subregions(
    region: Dict[str, object],
    image: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    crop = crop_region(image, region["coordinate"])
    local_barcode_boxes, local_debug = detect_local_barcode_boxes_with_debug(crop)
    debug_payload: Dict[str, object] = {
        "crop": crop,
        "local_barcode_boxes": [[int(v) for v in box] for box in local_barcode_boxes],
        "local_debug": local_debug,
        "masked_image": crop.copy(),
        "masked_mask": None,
        "salvage_boxes_local": [],
    }
    if not local_barcode_boxes:
        return [], [], debug_payload

    crop_mask = to_binary_mask(crop)
    crop_mask = apply_ignore_boxes_to_mask(crop_mask, local_barcode_boxes)
    crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    debug_payload["masked_image"] = apply_ignore_boxes_to_image(crop, local_barcode_boxes)
    debug_payload["masked_mask"] = crop_mask.copy()

    img_h, img_w = crop_mask.shape[:2]
    min_area = img_h * img_w * SALVAGE_COMPONENT_MIN_AREA_RATIO
    raw_boxes: List[List[int]] = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((crop_mask > 0).astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area or w < SALVAGE_COMPONENT_MIN_SIZE or h < SALVAGE_COMPONENT_MIN_SIZE:
            continue
        raw_boxes.append(
            clip_box(
                [
                    int(x - SALVAGE_COMPONENT_PAD_X),
                    int(y - SALVAGE_COMPONENT_PAD_Y),
                    int(x + w + SALVAGE_COMPONENT_PAD_X),
                    int(y + h + SALVAGE_COMPONENT_PAD_Y),
                ],
                crop.shape[:2],
            )
        )

    if not raw_boxes:
        return [], [], debug_payload

    child_regions: List[Dict[str, object]] = []
    merged_local_boxes = merge_boxes(raw_boxes, SALVAGE_COMPONENT_MERGE_GAP)
    debug_payload["salvage_boxes_local"] = [[int(v) for v in box] for box in merged_local_boxes]
    for child_idx, local_box in enumerate(merged_local_boxes):
        child_region = dict(region)
        child_region["coordinate"] = _map_local_box_to_global(region["coordinate"], local_box)
        child_region["salvaged_from_mixed_barcode"] = True
        child_region["salvage_child_idx"] = child_idx
        child_regions.append(child_region)

    skipped_regions: List[Dict[str, object]] = []
    for local_box in local_barcode_boxes:
        skipped_region = dict(region)
        skipped_region["coordinate"] = _map_local_box_to_global(region["coordinate"], local_box)
        skipped_region["skip_reason"] = "local_barcode_subregion"
        skipped_region["salvaged_from_mixed_barcode"] = True
        skipped_regions.append(skipped_region)

    return child_regions, skipped_regions, debug_payload


def filter_regions_with_barcode_salvage(
    regions: Sequence[Dict[str, object]],
    image: np.ndarray,
    ocr_boxes: Sequence[Tuple[object, str, float]] | None = None,
    debug_dir: Path | None = None,
    debug_prefix: str = "region",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    comparable: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for idx, region in enumerate(regions):
        region_copy = dict(region)
        region_copy.setdefault("original_idx", idx)
        crop = crop_region(image, region_copy["coordinate"])
        is_barcode, barcode_meta = detect_barcode_region(region_copy, image, ocr_boxes)
        is_texture, texture_meta = is_repetitive_vertical_texture(crop)

        if is_barcode or is_texture:
            salvaged_regions, salvaged_skipped, salvage_debug = salvage_non_barcode_subregions(region_copy, image)
            if debug_dir is not None:
                save_salvage_region_debug(
                    debug_dir / f"{debug_prefix}_region_{idx:02d}",
                    region_copy,
                    is_barcode,
                    barcode_meta,
                    is_texture,
                    texture_meta,
                    salvage_debug,
                    salvaged_regions,
                    salvaged_skipped,
                )
            if salvaged_regions:
                comparable.extend(salvaged_regions)
                skipped.extend(salvaged_skipped)
                continue

        if is_barcode:
            region_copy["skip_reason"] = "barcode"
            region_copy["barcode_hint"] = barcode_meta
            skipped.append(region_copy)
        elif is_texture:
            region_copy["skip_reason"] = "repetitive_vertical_texture"
            region_copy["texture_hint"] = texture_meta
            skipped.append(region_copy)
        else:
            comparable.append(region_copy)

    return comparable, skipped


def apply_ignore_boxes_to_mask(mask: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
    output = mask.copy()
    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, mask.shape[:2])
        output[y1:y2, x1:x2] = 0
    return output


def apply_ignore_boxes_to_image(image: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, image.shape[:2])
        output[y1:y2, x1:x2] = 255
    return output


def save_salvage_region_debug(
    region_dir: Path,
    region: Dict[str, object],
    is_barcode: bool,
    barcode_meta: Dict[str, object],
    is_texture: bool,
    texture_meta: Dict[str, float],
    salvage_debug: Dict[str, object],
    salvaged_regions: Sequence[Dict[str, object]],
    salvaged_skipped: Sequence[Dict[str, object]],
) -> None:
    region_dir.mkdir(parents=True, exist_ok=True)
    crop = np.asarray(salvage_debug.get("crop"))
    if crop.size == 0:
        return

    local_barcode_boxes = [[int(v) for v in box] for box in salvage_debug.get("local_barcode_boxes", [])]
    masked_image = np.asarray(salvage_debug.get("masked_image"))
    masked_mask = salvage_debug.get("masked_mask")
    local_debug = dict(salvage_debug.get("local_debug", {}))
    salvage_boxes_local = [[int(v) for v in box] for box in salvage_debug.get("salvage_boxes_local", [])]

    candidate_overlay = crop.copy()
    for box in local_debug.get("component_candidate_boxes", []):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(candidate_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    band_box = local_debug.get("band_box")
    if band_box:
        x1, y1, x2, y2 = [int(v) for v in band_box]
        cv2.rectangle(candidate_overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
    for box in local_debug.get("merged_boxes", []):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(candidate_overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

    salvage_overlay = masked_image.copy() if masked_image.size else crop.copy()
    for box in salvage_boxes_local:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(salvage_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in local_barcode_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(salvage_overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

    cv2.imwrite(str(region_dir / "crop.jpg"), crop)
    cv2.imwrite(str(region_dir / "barcode_candidates_overlay.jpg"), candidate_overlay)
    cv2.imwrite(str(region_dir / "masked_foreground.jpg"), masked_image if masked_image.size else crop)
    cv2.imwrite(str(region_dir / "salvage_boxes_overlay.jpg"), salvage_overlay)

    binary_mask = local_debug.get("binary_mask")
    if isinstance(binary_mask, np.ndarray):
        cv2.imwrite(str(region_dir / "binary_mask.jpg"), binary_mask)
    morph_mask = local_debug.get("morph_mask")
    if isinstance(morph_mask, np.ndarray):
        cv2.imwrite(str(region_dir / "morph_mask.jpg"), morph_mask)
    band_candidate_mask = local_debug.get("band_debug", {}).get("candidate_mask")
    if isinstance(band_candidate_mask, np.ndarray):
        cv2.imwrite(str(region_dir / "band_candidate_mask.jpg"), band_candidate_mask)
    if isinstance(masked_mask, np.ndarray):
        cv2.imwrite(str(region_dir / "masked_foreground_mask.jpg"), masked_mask)

    debug_json = {
        "region_coordinate": [int(v) for v in region.get("coordinate", [])],
        "is_barcode": bool(is_barcode),
        "barcode_meta": barcode_meta,
        "is_texture": bool(is_texture),
        "texture_meta": texture_meta,
        "local_barcode_boxes": local_barcode_boxes,
        "component_candidate_boxes": local_debug.get("component_candidate_boxes", []),
        "band_box": band_box,
        "band_edge_threshold_mode": local_debug.get("band_debug", {}).get("edge_threshold_mode"),
        "band_edge_threshold": local_debug.get("band_debug", {}).get("edge_threshold"),
        "band_fixed_edge_threshold": local_debug.get("band_debug", {}).get("fixed_edge_threshold"),
        "band_adaptive_edge_threshold": local_debug.get("band_debug", {}).get("adaptive_edge_threshold"),
        "band_adaptive_pool_count": local_debug.get("band_debug", {}).get("adaptive_pool_count"),
        "band_adaptive_right_start": local_debug.get("band_debug", {}).get("adaptive_right_start"),
        "band_adaptive_pool_stats": local_debug.get("band_debug", {}).get("adaptive_pool_stats", {}),
        "band_candidate_segments": local_debug.get("band_debug", {}).get("candidate_segments", []),
        "merged_local_barcode_boxes": local_debug.get("merged_boxes", []),
        "salvage_boxes_local": salvage_boxes_local,
        "salvaged_region_boxes": [
            [int(v) for v in item.get("coordinate", [])]
            for item in salvaged_regions
        ],
        "salvaged_skipped_boxes": [
            [int(v) for v in item.get("coordinate", [])]
            for item in salvaged_skipped
        ],
    }
    with open(region_dir / "debug.json", "w", encoding="utf-8") as file_obj:
        json.dump(debug_json, file_obj, ensure_ascii=False, indent=2)


def foreground_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, mask.shape[1], mask.shape[0]]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


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
    template_mask_raw = to_binary_mask(template_crop)
    target_mask_raw = to_binary_mask(target_crop)
    tx1, ty1, tx2, ty2 = foreground_bbox(template_mask_raw)
    sx1, sy1, sx2, sy2 = foreground_bbox(target_mask_raw)

    template_fg = template_crop[ty1:ty2, tx1:tx2]
    target_fg = target_crop[sy1:sy2, sx1:sx2]
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

    def paste(src: np.ndarray, dst: np.ndarray) -> List[int]:
        top = NORMALIZE_MARGIN + (max_content_height - src.shape[0])
        left = NORMALIZE_MARGIN
        bottom = top + src.shape[0]
        right = left + src.shape[1]
        dst[top:bottom, left:right] = src
        return [left, top, right, bottom]

    template_canvas_box = paste(template_norm, template_canvas)
    target_canvas_box = paste(target_norm, target_canvas)
    paste(template_mask, template_mask_canvas)
    paste(target_mask, target_mask_canvas)

    return {
        "template_image": template_canvas,
        "target_image": target_canvas,
        "template_mask": template_mask_canvas,
        "target_mask": target_mask_canvas,
        "meta": {
            "method": "foreground_crop_resize_pad_bottom_align",
            "template_canvas_box": template_canvas_box,
            "target_canvas_box": target_canvas_box,
        },
    }


def compute_tolerant_difference_masks(
    template_mask: np.ndarray,
    target_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DIFF_TOLERANCE_KERNEL, DIFF_TOLERANCE_KERNEL),
    )
    template_relaxed = cv2.dilate(template_mask, kernel, iterations=1)
    target_relaxed = cv2.dilate(target_mask, kernel, iterations=1)

    template_only = np.where((template_mask > 0) & (target_relaxed == 0), 255, 0).astype(np.uint8)
    target_only = np.where((target_mask > 0) & (template_relaxed == 0), 255, 0).astype(np.uint8)
    return template_only, target_only


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def merge_boxes(boxes: Sequence[Sequence[int]], gap: int) -> List[List[int]]:
    pending = [[int(v) for v in box] for box in boxes]
    merged = True
    while merged:
        merged = False
        result: List[List[int]] = []
        while pending:
            current = pending.pop(0)
            idx = 0
            while idx < len(pending):
                other = pending[idx]
                overlap_or_close = not (
                    current[2] + gap < other[0]
                    or other[2] + gap < current[0]
                    or current[3] + gap < other[1]
                    or other[3] + gap < current[1]
                )
                if overlap_or_close:
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    ]
                    pending.pop(idx)
                    merged = True
                else:
                    idx += 1
            result.append(current)
        pending = result
    pending.sort(key=lambda item: (item[0], item[1]))
    return pending


def clip_box(box: Sequence[int], image_shape: Sequence[int], pad: int = 0) -> List[int]:
    h, w = int(image_shape[0]), int(image_shape[1])
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(w, int(box[2]) + pad)
    y2 = min(h, int(box[3]) + pad)
    return [x1, y1, x2, y2]


def draw_single_box(image: np.ndarray, box: Sequence[int], color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def extract_candidate_boxes(
    template_only_mask: np.ndarray,
    target_only_mask: np.ndarray,
) -> List[Dict[str, object]]:
    diff_union = np.where((template_only_mask > 0) | (target_only_mask > 0), 255, 0).astype(np.uint8)
    close_kernel = np.ones((CANDIDATE_CLOSE_KERNEL, CANDIDATE_CLOSE_KERNEL), dtype=np.uint8)
    dilate_kernel = np.ones((CANDIDATE_DILATE_KERNEL, CANDIDATE_DILATE_KERNEL), dtype=np.uint8)
    merged = cv2.morphologyEx(diff_union, cv2.MORPH_CLOSE, close_kernel)
    merged = cv2.dilate(merged, dilate_kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), connectivity=8)
    min_area = diff_union.shape[0] * diff_union.shape[1] * CANDIDATE_MIN_AREA_RATIO
    raw_boxes: List[List[int]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area or w < 8 or h < 8:
            continue
        raw_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    merged_boxes = merge_boxes(raw_boxes, CANDIDATE_MERGE_GAP)
    candidates: List[Dict[str, object]] = []
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        template_pixels = int((template_only_mask[y1:y2, x1:x2] > 0).sum())
        target_pixels = int((target_only_mask[y1:y2, x1:x2] > 0).sum())
        diff_pixels = template_pixels + target_pixels
        if diff_pixels <= 0:
            continue

        if target_pixels >= template_pixels * 1.5 and target_pixels > 20:
            hint = "target_extra"
        elif template_pixels >= target_pixels * 1.5 and template_pixels > 20:
            hint = "template_extra"
        else:
            hint = "mixed"

        candidates.append(
            {
                "box": box,
                "template_diff_pixels": template_pixels,
                "target_diff_pixels": target_pixels,
                "diff_pixels": diff_pixels,
                "type_hint": hint,
            }
        )

    candidates.sort(key=lambda item: int(item["diff_pixels"]), reverse=True)
    return candidates[:MAX_CANDIDATES_PER_PAIR]


def should_review_candidate(candidate: Dict[str, object]) -> Tuple[bool, Dict[str, float | str]]:
    template_pixels = int(candidate["template_diff_pixels"])
    target_pixels = int(candidate["target_diff_pixels"])
    diff_pixels = int(candidate["diff_pixels"])
    max_side_pixels = max(template_pixels, target_pixels)
    min_side_pixels = min(template_pixels, target_pixels)
    dominance_ratio = float(max_side_pixels / max(1, min_side_pixels))

    keep = (
        diff_pixels >= REVIEW_MIN_DIFF_PIXELS
        or max_side_pixels >= REVIEW_MIN_SIDE_DIFF_PIXELS
        or dominance_ratio >= REVIEW_MIN_DOMINANCE_RATIO
    )
    reason = "review"
    if not keep:
        reason = "skip_small_balanced_residual"

    return keep, {
        "diff_pixels": float(diff_pixels),
        "max_side_pixels": float(max_side_pixels),
        "min_side_pixels": float(min_side_pixels),
        "dominance_ratio": dominance_ratio,
        "reason": reason,
    }


def build_marked_candidate_crop(
    image: np.ndarray,
    crop_box: Sequence[int],
    candidate_box: Sequence[int],
) -> np.ndarray:
    cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
    bx1, by1, bx2, by2 = [int(v) for v in candidate_box]
    crop = image[cy1:cy2, cx1:cx2].copy()
    rel_box = [bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1]
    return draw_single_box(crop, rel_box, (0, 0, 255))


def fit_image_to_panel(image: np.ndarray, panel_size: int) -> np.ndarray:
    panel = np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 255
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return panel

    scale = panel_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    x_offset = (panel_size - new_w) // 2
    y_offset = (panel_size - new_h) // 2
    panel[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return panel


def create_candidate_review_canvas(
    template_full_marked: np.ndarray,
    target_full_marked: np.ndarray,
    template_candidate_marked: np.ndarray,
    target_candidate_marked: np.ndarray,
) -> np.ndarray:
    panel_size = VLM_CANVAS_SIZE
    gap = 20
    header = 36
    footer = 8
    canvas_h = header * 2 + panel_size * 2 + gap + footer
    canvas_w = panel_size * 2 + gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    panels = [
        ("Template Region", fit_image_to_panel(template_full_marked, panel_size), 0, header),
        ("Target Region", fit_image_to_panel(target_full_marked, panel_size), panel_size + gap, header),
        ("Template Zoom", fit_image_to_panel(template_candidate_marked, panel_size), 0, header * 2 + panel_size + gap),
        ("Target Zoom", fit_image_to_panel(target_candidate_marked, panel_size), panel_size + gap, header * 2 + panel_size + gap),
    ]

    for title, panel, x, y in panels:
        cv2.putText(
            canvas,
            title,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        canvas[y:y + panel_size, x:x + panel_size] = panel

    cv2.line(
        canvas,
        (panel_size + gap // 2, 0),
        (panel_size + gap // 2, canvas_h),
        (180, 180, 180),
        2,
    )
    cv2.line(
        canvas,
        (0, header + panel_size + gap // 2),
        (canvas_w, header + panel_size + gap // 2),
        (180, 180, 180),
        2,
    )
    return canvas


def compare_canvas_with_vlm(
    comparator,
    canvas: np.ndarray,
    prompt: str,
) -> Dict[str, object]:
    canvas_b64 = comparator._image_to_base64(canvas)
    last_error = None

    for attempt in range(VLM_MAX_RETRIES):
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [canvas_b64],
                }
            ]
            response = requests.post(
                f"{comparator.api_base}/api/chat",
                json={
                    "model": comparator.model_name,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": VLM_NUM_PREDICT,
                    },
                },
                timeout=comparator.timeout,
            )
            response.raise_for_status()
            result = response.json()
            message = result.get("message", {})
            content = message.get("content", "") or message.get("thinking", "")
            if not content:
                if attempt < VLM_MAX_RETRIES - 1:
                    continue
                return comparator._make_unknown_result(
                    error_type="empty_response",
                    summary="API 返回空响应",
                )

            parsed = comparator._parse_response(content)
            if parsed.get("decision") == "unknown" and parsed.get("parse_error") and attempt < VLM_MAX_RETRIES - 1:
                continue
            return parsed
        except requests.exceptions.Timeout:
            last_error = "timeout"
            if attempt < VLM_MAX_RETRIES - 1:
                continue
            return comparator._make_unknown_result(
                error_type="timeout",
                summary="对比失败：请求超时",
            )
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            if attempt < VLM_MAX_RETRIES - 1:
                continue
            return comparator._make_unknown_result(
                error_type="request_error",
                summary=f"对比失败：{exc}",
            )

    return comparator._make_unknown_result(
        error_type="max_retries_exceeded",
        summary=f"对比失败：{last_error or '超过最大重试次数'}",
    )


def is_background_only_mismatch(candidate: Dict[str, object]) -> bool:
    vlm_result = candidate.get("vlm_result", {})
    summary = str(vlm_result.get("summary", "")).lower()
    differences = [str(item).lower() for item in vlm_result.get("differences", [])]
    text = " ".join([summary] + differences)

    background_keywords = [
        "背景",
        "background",
        "灰色",
        "白色",
        "gray",
        "grey",
        "white",
        "底色",
        "明暗",
        "亮度",
    ]
    same_structure_phrases = [
        "主体图形一致",
        "图形一致",
        "前景图形一致",
        "主体前景一致",
        "结构一致",
        "轮廓一致",
        "same graphic",
        "same foreground",
        "same structure",
    ]
    structural_difference_keywords = [
        "额外",
        "多出",
        "新增",
        "缺少",
        "缺失",
        "替换",
        "不同结构",
        "结构不同",
        "不同图形",
        "图形不同",
        "空白",
        "extra",
        "additional",
        "missing",
        "replace",
        "replaced",
        "different structure",
        "different graphic",
        "blank",
        "non-empty",
    ]

    has_background = any(keyword in text for keyword in background_keywords)
    if not has_background:
        return False

    has_structural_difference = any(keyword in text for keyword in structural_difference_keywords)
    has_same_structure = any(keyword in summary for keyword in same_structure_phrases)
    all_differences_are_background = bool(differences) and all(
        any(keyword in item for keyword in background_keywords) for item in differences
    )

    return (has_same_structure or all_differences_are_background) and not has_structural_difference


def split_final_candidates(
    candidate_results: Sequence[Dict[str, object]],
) -> Tuple[List[List[int]], List[List[int]], Dict[str, object]]:
    reviewed = [candidate for candidate in candidate_results if bool(candidate.get("sent_to_vlm"))]
    mismatch_candidates = [
        candidate
        for candidate in reviewed
        if str(candidate.get("vlm_result", {}).get("decision", "unknown")) == "mismatch"
    ]
    unknown_candidates = [
        candidate
        for candidate in reviewed
        if str(candidate.get("vlm_result", {}).get("decision", "unknown")) == "unknown"
    ]

    single_sided_mismatches = [
        candidate
        for candidate in mismatch_candidates
        if str(candidate.get("type_hint", "")) in FINAL_SINGLE_SIDED_HINTS
    ]

    suppressed_candidate_indices: List[int] = []
    if single_sided_mismatches:
        largest_single_sided_diff = max(int(candidate.get("diff_pixels", 0)) for candidate in single_sided_mismatches)
        filtered_single_sided: List[Dict[str, object]] = []
        for candidate in single_sided_mismatches:
            diff_pixels = int(candidate.get("diff_pixels", 0))
            template_pixels = int(candidate.get("template_diff_pixels", 0))
            target_pixels = int(candidate.get("target_diff_pixels", 0))
            max_side_pixels = max(template_pixels, target_pixels)
            relative_ratio = float(diff_pixels / max(1, largest_single_sided_diff))
            background_only = is_background_only_mismatch(candidate)
            should_suppress = (
                background_only
                and diff_pixels <= FINAL_BACKGROUND_REJECT_MAX_DIFF_PIXELS
                and max_side_pixels <= FINAL_BACKGROUND_REJECT_MAX_SIDE_PIXELS
                and relative_ratio <= FINAL_SMALL_SINGLE_SIDED_RATIO
            )
            if should_suppress:
                suppressed_candidate_indices.append(int(candidate["candidate_idx"]))
            else:
                filtered_single_sided.append(candidate)

        final_mismatch_candidates = filtered_single_sided
        suppressed_candidate_indices.extend(
            [
                int(candidate["candidate_idx"])
                for candidate in mismatch_candidates
                if str(candidate.get("type_hint", "")) not in FINAL_SINGLE_SIDED_HINTS
            ]
        )
        suppressed_candidate_indices = sorted(set(suppressed_candidate_indices))
    else:
        final_mismatch_candidates = mismatch_candidates

    if single_sided_mismatches and not final_mismatch_candidates:
        final_mismatch_candidates = [
            candidate
            for candidate in mismatch_candidates
            if str(candidate.get("type_hint", "")) not in FINAL_SINGLE_SIDED_HINTS
        ]
        suppressed_candidate_indices = sorted(
            set(suppressed_candidate_indices).union(
                int(candidate["candidate_idx"])
                for candidate in single_sided_mismatches
            )
        )

    mismatch_boxes = [[int(v) for v in candidate["box"]] for candidate in final_mismatch_candidates]
    unknown_boxes = [[int(v) for v in candidate["box"]] for candidate in unknown_candidates]
    selection_meta = {
        "selection_rule": "prefer_single_sided_mismatch_over_mixed",
        "single_sided_mismatch_count": len(single_sided_mismatches),
        "suppressed_candidate_indices": suppressed_candidate_indices,
        "background_only_small_reject_max_diff_pixels": FINAL_BACKGROUND_REJECT_MAX_DIFF_PIXELS,
        "background_only_small_reject_max_side_pixels": FINAL_BACKGROUND_REJECT_MAX_SIDE_PIXELS,
        "background_only_small_reject_max_relative_ratio": FINAL_SMALL_SINGLE_SIDED_RATIO,
    }
    return mismatch_boxes, unknown_boxes, selection_meta


def draw_final_diff_canvas(
    template_image: np.ndarray,
    target_image: np.ndarray,
    mismatch_boxes: Sequence[Sequence[int]],
    unknown_boxes: Sequence[Sequence[int]],
) -> np.ndarray:
    template_canvas = template_image.copy()
    target_canvas = target_image.copy()
    for box in mismatch_boxes:
        template_canvas = draw_single_box(template_canvas, box, (0, 0, 255))
        target_canvas = draw_single_box(target_canvas, box, (0, 0, 255))
    for box in unknown_boxes:
        template_canvas = draw_single_box(template_canvas, box, (0, 165, 255))
        target_canvas = draw_single_box(target_canvas, box, (0, 165, 255))

    canvas = np.ones((max(template_canvas.shape[0], target_canvas.shape[0]) + 40, template_canvas.shape[1] + target_canvas.shape[1] + 20, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Template", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Target", (template_canvas.shape[1] + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    canvas[40:40 + template_canvas.shape[0], 0:template_canvas.shape[1]] = template_canvas
    canvas[40:40 + target_canvas.shape[0], template_canvas.shape[1] + 20:template_canvas.shape[1] + 20 + target_canvas.shape[1]] = target_canvas
    return canvas


def draw_overview_diff_canvas(
    template_image: np.ndarray,
    target_image: np.ndarray,
    mismatch_template_boxes: Sequence[Sequence[int]],
    mismatch_target_boxes: Sequence[Sequence[int]],
    unknown_template_boxes: Sequence[Sequence[int]],
    unknown_target_boxes: Sequence[Sequence[int]],
) -> np.ndarray:
    template_canvas = template_image.copy()
    target_canvas = target_image.copy()
    for box in mismatch_template_boxes:
        template_canvas = draw_single_box(template_canvas, box, (0, 0, 255))
    for box in mismatch_target_boxes:
        target_canvas = draw_single_box(target_canvas, box, (0, 0, 255))
    for box in unknown_template_boxes:
        template_canvas = draw_single_box(template_canvas, box, (0, 165, 255))
    for box in unknown_target_boxes:
        target_canvas = draw_single_box(target_canvas, box, (0, 165, 255))

    canvas = np.ones(
        (max(template_canvas.shape[0], target_canvas.shape[0]) + 40, template_canvas.shape[1] + target_canvas.shape[1] + 20, 3),
        dtype=np.uint8,
    ) * 255
    cv2.putText(canvas, "Template", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Target", (template_canvas.shape[1] + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    canvas[40:40 + template_canvas.shape[0], 0:template_canvas.shape[1]] = template_canvas
    canvas[40:40 + target_canvas.shape[0], template_canvas.shape[1] + 20:template_canvas.shape[1] + 20 + target_canvas.shape[1]] = target_canvas
    return canvas


def summarize_pair_decision(candidate_results: Sequence[Dict[str, object]]) -> str:
    if not candidate_results:
        return "match"

    decisions = [str(item.get("vlm_result", {}).get("decision", "unknown")) for item in candidate_results]
    if any(decision == "mismatch" for decision in decisions):
        return "mismatch"
    if any(decision == "unknown" for decision in decisions):
        return "unknown"
    return "match"


def count_reviewed_candidates(candidate_results: Sequence[Dict[str, object]]) -> int:
    return sum(1 for item in candidate_results if bool(item.get("sent_to_vlm")))


def compare_one_pair(
    comparator,
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

    normalized = normalize_pair_crops(template_crop, target_crop)
    template_normalized = normalized["template_image"]
    target_normalized = normalized["target_image"]
    template_mask = normalized["template_mask"]
    target_mask = normalized["target_mask"]

    template_barcode_boxes = detect_local_barcode_boxes(template_normalized)
    target_barcode_boxes = detect_local_barcode_boxes(target_normalized)
    barcode_ignore_boxes = merge_boxes(
        template_barcode_boxes + target_barcode_boxes,
        LOCAL_BARCODE_MERGE_GAP,
    ) if (template_barcode_boxes or target_barcode_boxes) else []

    comparison_template_image = apply_ignore_boxes_to_image(template_normalized, barcode_ignore_boxes)
    comparison_target_image = apply_ignore_boxes_to_image(target_normalized, barcode_ignore_boxes)
    comparison_template_mask = apply_ignore_boxes_to_mask(template_mask, barcode_ignore_boxes)
    comparison_target_mask = apply_ignore_boxes_to_mask(target_mask, barcode_ignore_boxes)

    template_only_mask, target_only_mask = compute_tolerant_difference_masks(
        comparison_template_mask,
        comparison_target_mask,
    )
    candidates = extract_candidate_boxes(template_only_mask, target_only_mask)

    candidate_results: List[Dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        candidate_box = clip_box(candidate["box"], comparison_template_image.shape[:2], pad=CANDIDATE_CONTEXT_PAD)
        template_full_marked = draw_single_box(comparison_template_image, candidate["box"], (0, 0, 255))
        target_full_marked = draw_single_box(comparison_target_image, candidate["box"], (0, 0, 255))
        template_candidate_marked = build_marked_candidate_crop(
            comparison_template_image,
            candidate_box,
            candidate["box"],
        )
        target_candidate_marked = build_marked_candidate_crop(
            comparison_target_image,
            candidate_box,
            candidate["box"],
        )

        should_review, review_meta = should_review_candidate(candidate)

        if should_review:
            review_canvas = create_candidate_review_canvas(
                template_full_marked,
                target_full_marked,
                template_candidate_marked,
                target_candidate_marked,
            )
            vlm_result = compare_canvas_with_vlm(
                comparator,
                review_canvas,
                CANDIDATE_REVIEW_PROMPT,
            )
        else:
            vlm_result = {
                "decision": "match",
                "is_match": True,
                "confidence": 0.0,
                "needs_review": False,
                "error_type": None,
                "differences": [],
                "summary": "候选框被规则过滤为小面积且双边均衡的边缘残差，未送入 VLM 复核。",
                "raw_response": "",
            }

        candidate_payload = {
            "candidate_idx": idx,
            "box": [int(v) for v in candidate["box"]],
            "crop_box": candidate_box,
            "type_hint": candidate["type_hint"],
            "template_diff_pixels": int(candidate["template_diff_pixels"]),
            "target_diff_pixels": int(candidate["target_diff_pixels"]),
            "diff_pixels": int(candidate["diff_pixels"]),
            "review_gate": review_meta,
            "sent_to_vlm": bool(should_review),
            "vlm_result": vlm_result,
        }
        candidate_results.append(candidate_payload)

    pair_decision = summarize_pair_decision(candidate_results)
    mismatch_boxes, unknown_boxes, final_selection = split_final_candidates(candidate_results)
    final_diff_canvas = draw_final_diff_canvas(
        comparison_template_image,
        comparison_target_image,
        mismatch_boxes,
        unknown_boxes,
    )
    final_result = {
        "pair_decision": pair_decision,
        "barcode_ignore_template_boxes": template_barcode_boxes,
        "barcode_ignore_target_boxes": target_barcode_boxes,
        "barcode_ignore_boxes": barcode_ignore_boxes,
        "final_mismatch_boxes": mismatch_boxes,
        "final_unknown_boxes": unknown_boxes,
        "final_selection": final_selection,
        "candidate_results": candidate_results,
    }
    cv2.imwrite(str(pair_output_dir / "final_diff_canvas.jpg"), final_diff_canvas)
    with open(pair_output_dir / "final_result.json", "w", encoding="utf-8") as file_obj:
        json.dump(final_result, file_obj, ensure_ascii=False, indent=2)

    return {
        "template_box": [int(v) for v in template_box],
        "target_box": [int(v) for v in target_box],
        "template_size": list(template_crop.shape[:2]),
        "target_size": list(target_crop.shape[:2]),
        "normalization": normalized["meta"],
        "barcode_ignore_template_boxes": template_barcode_boxes,
        "barcode_ignore_target_boxes": target_barcode_boxes,
        "barcode_ignore_boxes": barcode_ignore_boxes,
        "candidate_count": len(candidate_results),
        "reviewed_candidate_count": count_reviewed_candidates(candidate_results),
        "pair_decision": pair_decision,
        "final_mismatch_boxes": mismatch_boxes,
        "final_unknown_boxes": unknown_boxes,
        "final_selection": final_selection,
        "candidates": candidate_results,
    }


def main() -> int:
    output_dir = OUTPUT_DIR / TEST_NAME
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("候选差异 + VLM 复核实验")
    print("=" * 60)

    payload = prepare_test_images(output_dir)
    template_image = payload["template"]["image"]
    target_image = payload["target"]["image"]
    template_path = payload["template"]["prepared_path"]
    target_path = payload["target"]["prepared_path"]

    print(f"template prepared: {template_path}")
    print(f"target prepared:   {target_path}")

    salvage_debug_dir = output_dir / "salvage_debug"
    template_all_regions = detect_layout_regions(str(template_path), threshold=LAYOUT_THRESHOLD)
    target_all_regions = detect_layout_regions(str(target_path), threshold=LAYOUT_THRESHOLD)
    template_regions = extract_regions_by_type(template_all_regions, REGION_TYPE)
    target_regions = extract_regions_by_type(target_all_regions, REGION_TYPE)

    _, template_boxes = get_ocr_with_boxes(str(template_path))
    _, target_boxes = get_ocr_with_boxes(str(target_path))
    template_regions, skipped_template_regions = filter_regions_with_barcode_salvage(
        template_regions,
        template_image,
        template_boxes,
        debug_dir=salvage_debug_dir,
        debug_prefix="template",
    )
    target_regions, skipped_target_regions = filter_regions_with_barcode_salvage(
        target_regions,
        target_image,
        target_boxes,
        debug_dir=salvage_debug_dir,
        debug_prefix="target",
    )

    template_vis = draw_regions(template_image, template_regions, color=(0, 255, 0))
    target_vis = draw_regions(target_image, target_regions, color=(0, 255, 0))
    if skipped_template_regions:
        template_vis = draw_regions(template_vis, skipped_template_regions, color=(0, 165, 255))
    if skipped_target_regions:
        target_vis = draw_regions(target_vis, skipped_target_regions, color=(0, 165, 255))
    cv2.imwrite(str(output_dir / "template_regions.jpg"), template_vis)
    cv2.imwrite(str(output_dir / "target_regions.jpg"), target_vis)

    matched_pairs, unmatched_template, unmatched_target = match_regions(
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
    print(f"unmatched template:          {len(unmatched_template)}")
    print(f"unmatched target:            {len(unmatched_target)}")

    selected_pairs = matched_pairs
    if PAIR_INDEX is not None:
        if PAIR_INDEX < 0 or PAIR_INDEX >= len(matched_pairs):
            raise IndexError(f"PAIR_INDEX 越界: {PAIR_INDEX}, 总匹配对数 {len(matched_pairs)}")
        selected_pairs = [matched_pairs[PAIR_INDEX]]

    comparator = get_vlm_comparator(model_name=VLM_MODEL, api_base=VLM_API_BASE)
    pair_results: List[Dict[str, object]] = []
    for idx, (template_idx, target_idx, cost) in enumerate(selected_pairs):
        pair_output_dir = output_dir / f"pair_{idx:02d}_t{template_idx}_s{target_idx}"
        print("\n" + "-" * 60)
        print(f"pair #{idx}: template[{template_idx}] vs target[{target_idx}]")
        print(f"match_cost = {cost:.4f}")

        result = compare_one_pair(
            comparator,
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

        print(
            "pair_decision="
            f"{result.get('pair_decision')} | "
            f"candidate_count={result.get('candidate_count', 0)} | "
            f"reviewed_candidate_count={result.get('reviewed_candidate_count', 0)} | "
            f"final_mismatch_boxes={len(result.get('final_mismatch_boxes', []))} | "
            f"final_unknown_boxes={len(result.get('final_unknown_boxes', []))}"
        )

    pair_decisions = [str(item.get("pair_decision", "unknown")) for item in pair_results]
    overview_mismatch_template_boxes = [
        [int(v) for v in item["template_box"]]
        for item in pair_results
        if str(item.get("pair_decision", "unknown")) == "mismatch"
    ] + [
        [int(v) for v in template_regions[idx]["coordinate"]]
        for idx in unmatched_template
    ]
    overview_mismatch_target_boxes = [
        [int(v) for v in item["target_box"]]
        for item in pair_results
        if str(item.get("pair_decision", "unknown")) == "mismatch"
    ] + [
        [int(v) for v in target_regions[idx]["coordinate"]]
        for idx in unmatched_target
    ]
    overview_unknown_template_boxes = [
        [int(v) for v in item["template_box"]]
        for item in pair_results
        if str(item.get("pair_decision", "unknown")) == "unknown"
    ]
    overview_unknown_target_boxes = [
        [int(v) for v in item["target_box"]]
        for item in pair_results
        if str(item.get("pair_decision", "unknown")) == "unknown"
    ]

    overall_decision = "match"
    if unmatched_template or unmatched_target or any(item == "mismatch" for item in pair_decisions):
        overall_decision = "mismatch"
    elif any(item == "unknown" for item in pair_decisions):
        overall_decision = "unknown"

    final_overview_canvas = draw_overview_diff_canvas(
        template_image,
        target_image,
        overview_mismatch_template_boxes,
        overview_mismatch_target_boxes,
        overview_unknown_template_boxes,
        overview_unknown_target_boxes,
    )
    cv2.imwrite(str(output_dir / "final_overview.jpg"), final_overview_canvas)

    summary = {
        "test_name": TEST_NAME,
        "pdf_path": str(PDF_PATH),
        "target_image_path": str(TARGET_IMAGE_PATH),
        "prepared_template_path": str(template_path),
        "prepared_target_path": str(target_path),
        "salvage_debug_dir": str(salvage_debug_dir),
        "layout_threshold": LAYOUT_THRESHOLD,
        "match_cost_threshold": MATCH_COST_THRESHOLD,
        "pair_index": PAIR_INDEX,
        "vlm_model": VLM_MODEL,
        "vlm_api_base": VLM_API_BASE,
        "vlm_timeout_seconds": VLM_TIMEOUT_SECONDS,
        "candidate_strategy": {
            "method": "internal_barcode_ignore -> tolerant_diff_union -> merged candidate boxes -> local VLM review",
            "tolerance_kernel": DIFF_TOLERANCE_KERNEL,
            "local_barcode_close_kernel": [LOCAL_BARCODE_CLOSE_WIDTH, LOCAL_BARCODE_CLOSE_HEIGHT],
            "local_barcode_dilate_kernel": [LOCAL_BARCODE_DILATE_WIDTH, LOCAL_BARCODE_DILATE_HEIGHT],
            "local_barcode_band_window": LOCAL_BARCODE_BAND_WINDOW,
            "local_barcode_band_edge_min": LOCAL_BARCODE_BAND_EDGE_MIN,
            "local_barcode_band_edge_floor": LOCAL_BARCODE_BAND_EDGE_FLOOR,
            "local_barcode_band_dark_range": [LOCAL_BARCODE_BAND_DARK_MIN, LOCAL_BARCODE_BAND_DARK_MAX],
            "local_barcode_band_min_width_ratio": LOCAL_BARCODE_BAND_MIN_WIDTH_RATIO,
            "local_barcode_band_min_start_ratio": LOCAL_BARCODE_BAND_MIN_START_RATIO,
            "local_barcode_band_close_gap": LOCAL_BARCODE_BAND_CLOSE_GAP,
            "local_barcode_band_adaptive_percentile": LOCAL_BARCODE_BAND_ADAPTIVE_PERCENTILE,
            "local_barcode_band_adaptive_scale": LOCAL_BARCODE_BAND_ADAPTIVE_SCALE,
            "local_barcode_band_adaptive_right_start_ratio": LOCAL_BARCODE_BAND_ADAPTIVE_RIGHT_START_RATIO,
            "local_barcode_band_adaptive_min_active_cols": LOCAL_BARCODE_BAND_ADAPTIVE_MIN_ACTIVE_COLS,
            "local_barcode_min_width_ratio": LOCAL_BARCODE_MIN_WIDTH_RATIO,
            "local_barcode_min_height_ratio": LOCAL_BARCODE_MIN_HEIGHT_RATIO,
            "local_barcode_min_area_ratio": LOCAL_BARCODE_MIN_AREA_RATIO,
            "local_barcode_merge_gap": LOCAL_BARCODE_MERGE_GAP,
            "salvage_component_min_area_ratio": SALVAGE_COMPONENT_MIN_AREA_RATIO,
            "salvage_component_min_size": SALVAGE_COMPONENT_MIN_SIZE,
            "salvage_component_merge_gap": SALVAGE_COMPONENT_MERGE_GAP,
            "candidate_close_kernel": CANDIDATE_CLOSE_KERNEL,
            "candidate_dilate_kernel": CANDIDATE_DILATE_KERNEL,
            "candidate_min_area_ratio": CANDIDATE_MIN_AREA_RATIO,
            "candidate_merge_gap": CANDIDATE_MERGE_GAP,
            "candidate_context_pad": CANDIDATE_CONTEXT_PAD,
            "max_candidates_per_pair": MAX_CANDIDATES_PER_PAIR,
            "review_min_diff_pixels": REVIEW_MIN_DIFF_PIXELS,
            "review_min_side_diff_pixels": REVIEW_MIN_SIDE_DIFF_PIXELS,
            "review_min_dominance_ratio": REVIEW_MIN_DOMINANCE_RATIO,
            "unmatched_region_rule": "unmatched comparable regions count as mismatch",
            "final_selection_rule": "prefer_single_sided_mismatch_over_mixed",
            "background_only_small_reject_max_diff_pixels": FINAL_BACKGROUND_REJECT_MAX_DIFF_PIXELS,
            "background_only_small_reject_max_side_pixels": FINAL_BACKGROUND_REJECT_MAX_SIDE_PIXELS,
            "background_only_small_reject_max_relative_ratio": FINAL_SMALL_SINGLE_SIDED_RATIO,
        },
        "template_regions": template_regions,
        "target_regions": target_regions,
        "skipped_template_regions": skipped_template_regions,
        "skipped_target_regions": skipped_target_regions,
        "matched_pairs": matched_pairs,
        "unmatched_template": unmatched_template,
        "unmatched_target": unmatched_target,
        "overall_decision": overall_decision,
        "overview_mismatch_template_boxes": overview_mismatch_template_boxes,
        "overview_mismatch_target_boxes": overview_mismatch_target_boxes,
        "overview_unknown_template_boxes": overview_unknown_template_boxes,
        "overview_unknown_target_boxes": overview_unknown_target_boxes,
        "pair_results": pair_results,
        "overview_output": str(output_dir / "final_overview.jpg"),
        "final_outputs": [
            {
                "pair_dir": str(output_dir / f"pair_{idx:02d}_t{item['template_idx']}_s{item['target_idx']}"),
                "final_diff_canvas": str(output_dir / f"pair_{idx:02d}_t{item['template_idx']}_s{item['target_idx']}" / "final_diff_canvas.jpg"),
                "final_result_json": str(output_dir / f"pair_{idx:02d}_t{item['template_idx']}_s{item['target_idx']}" / "final_result.json"),
            }
            for idx, item in enumerate(pair_results)
        ],
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\nsummary json:")
    print(summary_path)
    print("请直接查看每个 pair 目录下的 final_diff_canvas.jpg 和 final_result.json。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
