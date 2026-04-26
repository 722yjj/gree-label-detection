"""Unified text and graphic comparison workflow."""

import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np
import pandas as pd
from openpyxl.styles import Font
from pydantic import BaseModel

from label_detection.core import langchain_compat as _langchain_compat  # noqa: F401
from label_detection.core.config import (
    OLLAMA_API_BASE,
    TEXT_LLM_MODEL,
    TEXT_NUM_PREDICT,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDF_PATH,
    DEFAULT_TARGET_PATH,
    USE_VLM_FOR_GRAPHIC,
    ENABLE_IMAGE_REGION_SPLIT,
)
from label_detection.core.paddle_runtime import paddle_cache_cleanup_scope
from label_detection.extraction.template_source import resolve_template_input
from label_detection.extraction.text import (
    count_populated_fields,
    extract_compact_spec_from_text,
    extract_standard_spec_from_text,
    find_missing_fields,
    merge_compact_sources,
    merge_standard_sources,
    needs_compact_llm,
    find_suspicious_fields,
)
from label_detection.matching.layout import (
    detect_layout_regions,
    extract_regions_by_type,
    match_regions,
    compare_region_pair,
    draw_regions,
    split_barcode_regions,
    split_composite_image_regions,
    merge_fragmented_split_regions,
    filter_split_image_regions,
    recover_uncovered_graphic_regions,
    infer_corresponding_region,
    estimate_region_foreground_ratio,
)
from label_detection.matching.ocr import (
    extract_field_labels_from_ocr_boxes,
    field_values_match,
    find_matching_ocr_boxes,
    get_base_field_name,
    is_label_field_name,
    make_label_field_name,
    normalize_text_for_compare,
    normalize_text_for_match,
    text_field_values_match,
)
from label_detection.schema import (
    LABEL_KIND_COMPACT,
    LABEL_KIND_STANDARD,
    get_label_model,
    infer_label_kind,
)
from label_detection.preprocessing.border import crop_to_border, find_template_crop_rect
from label_detection.preprocessing.pipeline import preprocess_target
from label_detection.services.ollama_client import OllamaHTTPClient
from label_detection.services.ocr_service import get_ocr_with_boxes

_llm = None

FORCE_CONFIRMED_TEXT_FIELDS = {
    "model_number",
    "voltage",
    "frequency",
    "heating_capacity",
    "cooling_capacity",
    "air_volume",
    "weight",
    "noise",
    "mfg_date",
    "barcode",
}

ALIGNED_VALUE_VISUAL_SCAN_FIELDS = FORCE_CONFIRMED_TEXT_FIELDS | {"address"}


@dataclass(frozen=True)
class WorkflowOutputOptions:
    """Control which workflow artifacts should persist in the result directory."""

    mode: str = "debug"
    save_template_assets: bool = True
    save_preprocess_images: bool = True
    save_text_excel: bool = True
    save_graphic_debug: bool = True
    save_region_crops: bool = True
    save_vlm_debug: bool = True

    @classmethod
    def from_mode(cls, mode: Optional[str]) -> "WorkflowOutputOptions":
        normalized = str(mode or "debug").strip().lower()
        if normalized not in {"final", "debug"}:
            raise ValueError(f"不支持的 output_mode: {mode}")

        detailed = normalized == "debug"
        return cls(
            mode=normalized,
            save_template_assets=detailed,
            save_preprocess_images=detailed,
            save_text_excel=detailed,
            save_graphic_debug=detailed,
            save_region_crops=detailed,
            save_vlm_debug=detailed,
        )


def get_llm():
    """Lazily initialize the native Ollama client when the workflow runs."""
    global _llm
    if _llm is None:
        _llm = OllamaHTTPClient(
            model_name=TEXT_LLM_MODEL,
            api_base=OLLAMA_API_BASE,
            timeout=LLM_TIMEOUT,
            num_predict=TEXT_NUM_PREDICT,
        )
    return _llm


def encode_image(image_path):
    """将图片转换为 Base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None


def preprocess_template_image(template_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    预处理模板图片：检测黑框，去除黑框外白边，裁剪到黑框内部

    Args:
        template_path: 模板图片路径（从 PDF 提取的 PNG）
        output_dir: 输出目录

    Returns:
        cropped_image: 裁剪后的图像
        cropped_path: 保存路径
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n[预处理] 处理模板图片...")
    template = cv2.imread(template_path)

    if template is None:
        return None, None, f"无法读取图片: {template_path}"

    print(f"  原始尺寸: {template.shape}")

    crop_candidate = find_template_crop_rect(template, output_dir, "template")

    if crop_candidate is None:
        print("  未找到可靠的模板裁剪区域，使用原图")
        cropped = template
    else:
        padding = 3 if crop_candidate["strategy"] == "border" else -6
        cropped = crop_to_border(template, crop_candidate["rect"], padding=padding)
        print(
            "  使用模板裁剪策略: "
            f"{crop_candidate['strategy']} -> {crop_candidate['rect']}"
        )
        print(f"  裁剪后尺寸: {cropped.shape}")

    # 保存预处理结果
    output_path = os.path.join(output_dir, "template_preprocessed.png")
    cv2.imwrite(output_path, cropped)
    print(f"  已保存: {output_path}")

    return cropped, output_path, None


def _build_extraction_prompt(label_kind: str, ocr_text: str) -> str:
    if label_kind == LABEL_KIND_COMPACT:
        return f"""Task: extract structured text from a compact product label image.

Reference OCR text:
{ocr_text}

Fields:
- model_number: the Model value.
- net_weight: the N.W. / Net Weight value.
- gross_weight: the G.W. / Gross Weight value.
- color: the Color value.
- connection_pipes: the Connection Pipes value.
- refrigerant: the Refrigerant value.
- barcode: the digits printed below the barcode.

Extraction rules:
1. Read the image first. Use the OCR text only as a spelling/digit reference when the image is hard to read.
2. Preserve the exact printed value as much as possible. Do not translate, normalize units, calculate values, or correct apparent printing/OCR defects.
3. Output exactly these 7 keys. If a field is not visible or not present, set it to null.
4. Output one valid JSON object only. Do not output Markdown, code fences, explanations, analysis, or thinking text.

Example output:
{{"model_number":"GWH24AGD-K6DNA1C/I(WIFI)","net_weight":"14kg","gross_weight":"16.5kg","color":"White","connection_pipes":"1/4\\"/1/2\\"","refrigerant":"R32","barcode":"600001076226"}}"""

    return f"""Task: extract structured text from the air-conditioner product label image.

Reference OCR text:
The OCR text below may be unordered or noisy. Use it only as a reference for small characters, spelling, and digits:
{ocr_text}

Fields to output:
- brand
- product_type
- model_number
- voltage
- frequency
- heating_capacity
- cooling_capacity
- air_volume
- weight
- noise
- mfg_date
- manufacturer
- address
- barcode

Extraction rules:
1. Read the image first. Use OCR only when the image text is hard to read.
2. Preserve the exact printed text and value as much as possible, including unusual punctuation, duplicated letters, malformed dates, or apparent defects.
3. Do not translate, normalize units, calculate values, infer expected values, or correct spelling/printing defects.
4. If a field cannot be found in the image or OCR reference, set it to null.
5. Output exactly one valid JSON object and nothing else. Do not output Markdown, code fences, explanations, analysis, or thinking text.

Example output:
{{"brand":"GREE","product_type":"SPLIT AIR CONDITIONER INDOOR UNIT","model_number":"GWH18AAD-K6DNA2E/I","voltage":"220-240V~","frequency":"50Hz","heating_capacity":"5.20kW","cooling_capacity":"4.60kW","air_volume":"850m³/h","weight":"13.5kg","noise":"46dB(A)","mfg_date":"2026.01","manufacturer":"GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI","address":"Add: West Jinji Rd, Qianshan, Zhuhai, Guangdong, China, 519070","barcode":"600004075219"}}"""


def _extract_json_object(content: str) -> Optional[Dict[str, object]]:
    text = str(content or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _extract_structured_from_text(
    label_kind: str,
    model_cls: Type[BaseModel],
    ocr_text: str,
) -> Optional[Dict[str, object]]:
    if label_kind != LABEL_KIND_COMPACT:
        return None

    extracted = extract_compact_spec_from_text(ocr_text)
    if count_populated_fields(extracted) < 4:
        return None

    print(f"    [规则提取] 紧凑标签命中 {count_populated_fields(extracted)} 个字段")
    return extracted


def build_final_verdict(
    *,
    match_count: int,
    total_fields: int,
    graphic_pass: bool,
    review_count: int,
    mismatch_count: int,
    unresolved_graphics: int,
) -> str:
    text_diff_count = max(total_fields - match_count, 0)
    has_review = review_count > 0

    if text_diff_count == 0 and graphic_pass and not has_review:
        return "✅ 标签完全一致"
    if text_diff_count == 0 and graphic_pass and has_review:
        return f"⚠️ 标签基本一致，{review_count} 处图形需人工复核"
    if text_diff_count > 0 and graphic_pass and has_review:
        return f"⚠️ 文字存在差异 ({text_diff_count} 处)，另有 {review_count} 处图形需人工复核"
    if text_diff_count > 0 and graphic_pass:
        return f"⚠️ 文字存在差异 ({text_diff_count} 处)，图形一致"
    if text_diff_count == 0 and not graphic_pass:
        if unresolved_graphics > 0 and mismatch_count == 0:
            return f"⚠️ 文字一致，但图形有 {unresolved_graphics} 个区域未恢复"
        return (
            "⚠️ 文字一致，图形存在差异 "
            f"({mismatch_count} 处不匹配, {unresolved_graphics} 个未恢复)"
        )

    review_suffix = f", {review_count} 处待复核" if has_review else ""
    return (
        "❌ 标签差异较大 "
        f"(文字 {match_count}/{total_fields}，图形 {mismatch_count} 处不匹配, "
        f"{unresolved_graphics} 个未恢复{review_suffix})"
    )


def _clip_box_to_image(
    box: Sequence[float],
    image_shape: Tuple[int, int],
    *,
    min_size: int = 2,
) -> Optional[List[int]]:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return [x1, y1, x2, y2]


def _expand_box(
    box: Sequence[float],
    image_shape: Tuple[int, int],
    *,
    pad_ratio: float = 0.18,
    min_pad: int = 8,
    max_pad: int = 28,
) -> Optional[List[int]]:
    clipped = _clip_box_to_image(box, image_shape)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    pad = int(round(max(min_pad, min(max_pad, max(x2 - x1, y2 - y1) * pad_ratio))))
    return _clip_box_to_image([x1 - pad, y1 - pad, x2 + pad, y2 + pad], image_shape)


def _map_box_between_shapes(
    box: Sequence[float],
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Optional[List[int]]:
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    if source_w <= 0 or source_h <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    mapped = [
        x1 * target_w / source_w,
        y1 * target_h / source_h,
        x2 * target_w / source_w,
        y2 * target_h / source_h,
    ]
    return _clip_box_to_image(mapped, target_shape)


def _foreground_mask_for_local_diff(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype=bool)

    if crop.ndim == 2:
        gray = crop
        saturation = np.zeros_like(gray)
        value = gray
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    block_size = max(15, min(41, (min(gray.shape[:2]) // 2) * 2 + 1))
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )
    dark_or_colored = (
        (gray < 220)
        | ((saturation > 35) & (value < 248))
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(adaptive, dark_or_colored)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask > 0


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = mask.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return shifted > 0


def _largest_component_ratio(diff_mask: np.ndarray, reference_area: int) -> float:
    if reference_area <= 0 or diff_mask.size == 0:
        return 0.0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        diff_mask.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return 0.0
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest_area / max(1, reference_area))


def _local_graphic_diff_evidence(
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_box: Sequence[float] | None,
    target_box: Sequence[float] | None,
) -> Dict[str, Any]:
    if template_box is None or target_box is None:
        return {"valid": False, "reason": "missing_reference_box"}

    expanded_template_box = _expand_box(template_box, template_image.shape[:2])
    expanded_target_box = _expand_box(target_box, target_image.shape[:2])
    if expanded_template_box is None or expanded_target_box is None:
        return {"valid": False, "reason": "invalid_box"}

    tx1, ty1, tx2, ty2 = expanded_template_box
    sx1, sy1, sx2, sy2 = expanded_target_box
    template_crop = template_image[ty1:ty2, tx1:tx2]
    target_crop = target_image[sy1:sy2, sx1:sx2]
    if template_crop.size == 0 or target_crop.size == 0:
        return {"valid": False, "reason": "empty_crop"}

    target_h, target_w = target_crop.shape[:2]
    if template_crop.shape[:2] != target_crop.shape[:2]:
        template_crop = cv2.resize(
            template_crop,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    template_mask = _foreground_mask_for_local_diff(template_crop)
    target_mask = _foreground_mask_for_local_diff(target_crop)
    if template_mask.size == 0 or target_mask.size == 0:
        return {"valid": False, "reason": "empty_mask"}

    template_area = int(template_mask.sum())
    target_area = int(target_mask.sum())
    foreground_area = max(template_area, target_area)
    crop_area = max(1, int(template_mask.size))
    if foreground_area < max(20, int(crop_area * 0.002)):
        return {
            "valid": False,
            "reason": "low_foreground",
            "template_foreground_ratio": round(template_area / crop_area, 4),
            "target_foreground_ratio": round(target_area / crop_area, 4),
        }

    tolerance_radius = max(2, min(5, int(round(max(target_h, target_w) * 0.015))))
    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (tolerance_radius * 2 + 1, tolerance_radius * 2 + 1),
    )
    max_shift = max(2, min(8, int(round(max(target_h, target_w) * 0.035))))

    best: Dict[str, Any] | None = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted_template = _shift_mask(template_mask, dx, dy)
            shifted_template_dilated = (
                cv2.dilate(shifted_template.astype(np.uint8), tolerance_kernel) > 0
            )
            target_dilated = cv2.dilate(target_mask.astype(np.uint8), tolerance_kernel) > 0
            unmatched_template = shifted_template & ~target_dilated
            unmatched_target = target_mask & ~shifted_template_dilated
            diff_mask = cv2.morphologyEx(
                (unmatched_template | unmatched_target).astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8),
            ) > 0
            diff_area = int(diff_mask.sum())
            union_area = int((shifted_template | target_mask).sum())
            ratio = diff_area / max(1, union_area)
            candidate = {
                "diff_ratio": ratio,
                "unmatched_template_ratio": int(unmatched_template.sum()) / max(1, template_area),
                "unmatched_target_ratio": int(unmatched_target.sum()) / max(1, target_area),
                "largest_component_ratio": _largest_component_ratio(diff_mask, max(1, foreground_area)),
                "shift": [dx, dy],
                "diff_area": diff_area,
                "union_area": union_area,
            }
            if best is None or candidate["diff_ratio"] < best["diff_ratio"]:
                best = candidate

    assert best is not None
    return {
        "valid": True,
        "expanded_template_box": expanded_template_box,
        "expanded_target_box": expanded_target_box,
        "template_foreground_ratio": round(template_area / crop_area, 4),
        "target_foreground_ratio": round(target_area / crop_area, 4),
        "tolerance_radius": tolerance_radius,
        "max_shift": max_shift,
        "diff_ratio": round(float(best["diff_ratio"]), 4),
        "unmatched_template_ratio": round(float(best["unmatched_template_ratio"]), 4),
        "unmatched_target_ratio": round(float(best["unmatched_target_ratio"]), 4),
        "largest_component_ratio": round(float(best["largest_component_ratio"]), 4),
        "best_shift": best["shift"],
        "diff_area": int(best["diff_area"]),
        "union_area": int(best["union_area"]),
    }


def _should_suppress_graphic_visual_box(evidence: Dict[str, Any]) -> bool:
    if os.getenv("ENABLE_GRAPHIC_LOCAL_DIFF_FILTER", "1") == "0":
        return False
    if not evidence.get("valid"):
        return False
    return (
        float(evidence.get("diff_ratio", 1.0)) <= 0.08
        and float(evidence.get("unmatched_target_ratio", 1.0)) <= 0.08
        and float(evidence.get("unmatched_template_ratio", 1.0)) <= 0.10
        and float(evidence.get("largest_component_ratio", 1.0)) <= 0.04
    )


def _expand_small_graphic_visual_box(
    box: Sequence[float],
    image_shape: Tuple[int, int],
    evidence: Optional[Dict[str, Any]],
) -> Optional[List[int]]:
    clipped = _clip_box_to_image(box, image_shape)
    if clipped is None:
        return None
    if not evidence or not evidence.get("valid"):
        return clipped
    if float(evidence.get("unmatched_target_ratio", 0.0)) < 0.45:
        return clipped

    x1, y1, x2, y2 = clipped
    width = x2 - x1
    height = y2 - y1
    min_side = max(48, min(96, int(round(min(image_shape[:2]) * 0.20))))
    if width >= min_side and height >= min_side:
        return clipped

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_width = max(width, min_side)
    new_height = max(height, min_side)
    return _clip_box_to_image(
        [
            cx - new_width / 2.0,
            cy - new_height / 2.0,
            cx + new_width / 2.0,
            cy + new_height / 2.0,
        ],
        image_shape,
        min_size=8,
    )


def _ignored_regions_for_uncovered_recovery(
    regions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    barcode_skip_reasons = {
        "barcode",
        "barcode_cluster",
        "local_barcode_subregion",
    }
    return [
        dict(region)
        for region in regions
        if str(region.get("skip_reason") or "") not in barcode_skip_reasons
    ]


def _ocr_points_to_box(points: object) -> Optional[List[int]]:
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.size == 4 and arr.ndim == 1:
        x1, y1, x2, y2 = arr.tolist()
        return _clip_box_to_image([x1, y1, x2, y2], (10**9, 10**9))

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return None
        arr = arr.reshape((-1, 2))
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return None

    xs = arr[:, 0]
    ys = arr[:, 1]
    return [
        int(round(float(xs.min()))),
        int(round(float(ys.min()))),
        int(round(float(xs.max()))),
        int(round(float(ys.max()))),
    ]


def _ocr_points_to_poly_and_box(points: object) -> Tuple[np.ndarray, Optional[List[int]]]:
    box = _ocr_points_to_box(points)
    pts = np.array(points, dtype=np.int32)
    if pts.shape == (4,):
        x1, y1, x2, y2 = [int(v) for v in pts]
        poly = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        )
        box = [x1, y1, x2, y2]
    else:
        poly = pts.reshape((-1, 1, 2))
        if box is None:
            x, y, w, h = cv2.boundingRect(poly)
            box = [x, y, x + w, y + h]
    return poly, box


def _box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0,
        float(box1[3]) - float(box1[1]),
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0,
        float(box2[3]) - float(box2[1]),
    )
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_overlap_coverage(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(1.0, (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1])))
    return float(intersection / area1)


def _images_nearly_same_size(
    template_image: np.ndarray,
    target_image: np.ndarray,
    *,
    tolerance: int = 2,
) -> bool:
    th, tw = template_image.shape[:2]
    sh, sw = target_image.shape[:2]
    return abs(th - sh) <= tolerance and abs(tw - sw) <= tolerance


def _estimate_value_subspan_box(
    *,
    field_name: str,
    ocr_text: object,
    target_value: object,
    box: Sequence[float],
    image_shape: Tuple[int, int],
) -> Optional[List[int]]:
    if is_label_field_name(field_name):
        return None
    text = str(ocr_text or "").strip()
    value = str(target_value or "").strip()
    if not text or not value or len(text) <= len(value) + 4:
        return None

    start = text.casefold().find(value.casefold())
    end = start + len(value)
    if start < 0:
        compact_text = normalize_text_for_match(text)
        compact_value = normalize_text_for_match(value)
        if not compact_text or not compact_value:
            return None
        compact_start = compact_text.find(compact_value)
        if compact_start < 0:
            return None
        compact_to_original: List[int] = []
        for original_idx, char in enumerate(text):
            if normalize_text_for_match(char):
                compact_to_original.append(original_idx)
        if compact_start >= len(compact_to_original):
            return None
        start = compact_to_original[compact_start]
        compact_end = min(
            len(compact_to_original) - 1,
            compact_start + len(compact_value) - 1,
        )
        end = compact_to_original[compact_end] + 1

    clipped = _clip_box_to_image(box, image_shape)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    text_len = max(1, len(text))
    left = x1 + (x2 - x1) * max(0, start) / text_len
    right = x1 + (x2 - x1) * min(text_len, end) / text_len
    if right - left < 8:
        return None
    return _clip_box_to_image(
        [left - 6, y1 - 3, right + 6, y2 + 3],
        image_shape,
        min_size=6,
    )


def _local_text_component_box(
    *,
    template_image: np.ndarray,
    target_image: np.ndarray,
    target_box: Sequence[float],
    template_box: Sequence[float] | None = None,
) -> Optional[List[int]]:
    if not _images_nearly_same_size(template_image, target_image):
        return None

    clipped_target_box = _clip_box_to_image(target_box, target_image.shape[:2])
    if clipped_target_box is None:
        return None
    mapped_template_box = (
        _map_box_between_shapes(
            clipped_target_box,
            target_image.shape[:2],
            template_image.shape[:2],
        )
        if template_box is None
        else _clip_box_to_image(template_box, template_image.shape[:2])
    )
    if mapped_template_box is None:
        return None

    tx1, ty1, tx2, ty2 = mapped_template_box
    sx1, sy1, sx2, sy2 = clipped_target_box
    template_crop = template_image[ty1:ty2, tx1:tx2]
    target_crop = target_image[sy1:sy2, sx1:sx2]
    if template_crop.size == 0 or target_crop.size == 0:
        return None
    target_h, target_w = target_crop.shape[:2]
    if template_crop.shape[:2] != target_crop.shape[:2]:
        template_crop = cv2.resize(
            template_crop,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    template_gray = cv2.cvtColor(template_crop, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(template_gray, target_gray)
    foreground = (
        _foreground_mask_for_local_diff(template_crop)
        | _foreground_mask_for_local_diff(target_crop)
    )
    diff_mask = ((diff > 18) & foreground).astype(np.uint8)
    diff_mask = cv2.morphologyEx(
        diff_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        diff_mask,
        connectivity=8,
    )
    if component_count <= 1:
        return None

    components = []
    for idx in range(1, component_count):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 4 or w < 2 or h < 2:
            continue
        components.append((area, x, y, w, h))
    if not components:
        return None

    components.sort(reverse=True)
    area, x, y, w, h = components[0]
    box_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    if area / box_area > 0.18:
        return None
    if h > max(8, int(round((sy2 - sy1) * 0.72))):
        return None

    return _clip_box_to_image(
        [sx1 + x - 5, sy1 + y - 5, sx1 + x + w + 5, sy1 + y + h + 5],
        target_image.shape[:2],
        min_size=6,
    )


def _local_text_diff_evidence(
    template_image: np.ndarray,
    target_image: np.ndarray,
    target_box: Sequence[float],
    template_box: Sequence[float] | None = None,
) -> Dict[str, Any]:
    mapped_template_box = (
        _map_box_between_shapes(
            target_box,
            target_image.shape[:2],
            template_image.shape[:2],
        )
        if template_box is None
        else _clip_box_to_image(template_box, template_image.shape[:2])
    )
    evidence = _local_graphic_diff_evidence(
        template_image,
        target_image,
        mapped_template_box,
        target_box,
    )
    evidence["mapped_template_box"] = mapped_template_box
    return evidence


def _raw_text_box_diff_ratio(
    template_image: np.ndarray,
    target_image: np.ndarray,
    target_box: Sequence[float],
    template_box: Sequence[float] | None = None,
) -> float:
    clipped_target_box = _clip_box_to_image(target_box, target_image.shape[:2])
    if clipped_target_box is None:
        return 0.0
    mapped_template_box = (
        _map_box_between_shapes(
            clipped_target_box,
            target_image.shape[:2],
            template_image.shape[:2],
        )
        if template_box is None
        else _clip_box_to_image(template_box, template_image.shape[:2])
    )
    if mapped_template_box is None:
        return 0.0

    tx1, ty1, tx2, ty2 = mapped_template_box
    sx1, sy1, sx2, sy2 = clipped_target_box
    template_crop = template_image[ty1:ty2, tx1:tx2]
    target_crop = target_image[sy1:sy2, sx1:sx2]
    if template_crop.size == 0 or target_crop.size == 0:
        return 0.0
    if template_crop.shape[:2] != target_crop.shape[:2]:
        target_h, target_w = target_crop.shape[:2]
        template_crop = cv2.resize(
            template_crop,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    template_gray = cv2.cvtColor(template_crop, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
    return float((cv2.absdiff(template_gray, target_gray) > 12).mean())


def _should_suppress_text_visual_box(
    evidence: Dict[str, Any],
    annotation: Optional[Dict[str, Any]] = None,
) -> bool:
    if os.getenv("ENABLE_TEXT_LOCAL_DIFF_FILTER", "1") == "0":
        return False
    if not evidence.get("valid"):
        return False
    field_name = str((annotation or {}).get("field") or "")
    if field_name == make_label_field_name("air_volume") and max(
        float(evidence.get("diff_ratio", 0.0)),
        float(evidence.get("unmatched_template_ratio", 0.0)),
        float(evidence.get("unmatched_target_ratio", 0.0)),
    ) >= 0.009:
        return False
    return (
        float(evidence.get("diff_ratio", 1.0)) <= 0.05
        and float(evidence.get("unmatched_target_ratio", 1.0)) <= 0.05
        and float(evidence.get("unmatched_template_ratio", 1.0)) <= 0.07
        and float(evidence.get("largest_component_ratio", 1.0)) <= 0.045
    )


def _should_force_confirmed_text_field(field_name: str) -> bool:
    return (
        not is_label_field_name(field_name)
        and field_name in FORCE_CONFIRMED_TEXT_FIELDS
    )


def _add_text_visualization_annotation(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    box: Sequence[float],
    poly: Optional[np.ndarray],
    annotation: Dict[str, Any],
    template_box: Sequence[float] | None = None,
    force: bool = False,
) -> bool:
    clipped_box = _clip_box_to_image(box, target_image.shape[:2])
    if clipped_box is None:
        return False
    if _has_existing_text_annotation(clipped_box, visualization_annotations):
        return False

    evidence = _local_text_diff_evidence(
        template_image,
        target_image,
        clipped_box,
        template_box=template_box,
    )
    annotation = {
        **annotation,
        "box": [int(v) for v in clipped_box],
        "local_diff_evidence": evidence,
    }
    localized_text_component = False
    if str(annotation.get("field") or "") == "air_volume":
        component_box = _local_text_component_box(
            template_image=template_image,
            target_image=target_image,
            target_box=clipped_box,
            template_box=template_box,
        )
        if component_box is not None:
            clipped_box = component_box
            annotation = {
                **annotation,
                "box": [int(v) for v in clipped_box],
                "localized_component_box": True,
            }
            poly = None
            localized_text_component = True

    if (
        not localized_text_component
        and not force
        and _should_suppress_text_visual_box(evidence, annotation)
    ):
        suppressed_annotations.append(
            {
                **annotation,
                "kind": "text",
                "status": "suppressed",
                "reason": "local_diff_low",
            }
        )
        print(
            "    - 文字框过滤: "
            f"{annotation.get('field', annotation.get('source', 'text'))} "
            f"(local_diff={evidence.get('diff_ratio')}, "
            f"component={evidence.get('largest_component_ratio')})"
        )
        return False

    if _has_existing_text_annotation(clipped_box, visualization_annotations):
        return False

    if poly is not None:
        cv2.polylines(
            vis_image,
            [poly],
            isClosed=True,
            color=(0, 0, 255),
            thickness=3,
        )
    else:
        x1, y1, x2, y2 = [int(v) for v in clipped_box]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    visualization_annotations.append(annotation)
    return True

def _ocr_box_and_poly_for_index(
    ocr_boxes: Sequence[tuple],
    box_idx: int,
    image_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[List[int]], str]:
    if box_idx < 0 or box_idx >= len(ocr_boxes):
        return None, None, ""

    box_info = ocr_boxes[box_idx]
    text = str(box_info[1] if len(box_info) > 1 else "").strip()
    poly, box = _ocr_points_to_poly_and_box(box_info[0] if box_info else None)
    clipped_box = _clip_box_to_image(box or [], image_shape) if box is not None else None
    return poly, clipped_box, text


def _union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[int]]:
    valid_boxes = [box for box in boxes if box is not None and len(box) == 4]
    if not valid_boxes:
        return None

    return [
        int(round(min(float(box[0]) for box in valid_boxes))),
        int(round(min(float(box[1]) for box in valid_boxes))),
        int(round(max(float(box[2]) for box in valid_boxes))),
        int(round(max(float(box[3]) for box in valid_boxes))),
    ]


def _combined_ocr_text(
    ocr_boxes: Sequence[tuple],
    box_indices: Sequence[int],
) -> str:
    texts = []
    for box_idx in box_indices:
        if box_idx < 0 or box_idx >= len(ocr_boxes):
            continue
        text = str(ocr_boxes[box_idx][1] if len(ocr_boxes[box_idx]) > 1 else "").strip()
        if text:
            texts.append(text)
    return " ".join(texts).strip()


def _overlapping_ocr_indices(
    ocr_boxes: Sequence[tuple],
    image_shape: Tuple[int, int],
    anchor_box: Sequence[float],
) -> List[int]:
    clipped_anchor = _clip_box_to_image(anchor_box, image_shape)
    if clipped_anchor is None:
        return []

    ax1, ay1, ax2, ay2 = [float(v) for v in clipped_anchor]
    expanded_anchor = [
        max(0.0, ax1 - 10.0),
        max(0.0, ay1 - 8.0),
        ax2 + 10.0,
        ay2 + 8.0,
    ]
    matches: List[Tuple[float, int]] = []
    for box_idx, box_info in enumerate(ocr_boxes):
        box = _ocr_points_to_box(box_info[0] if box_info else None)
        box = _clip_box_to_image(box or [], image_shape) if box is not None else None
        if box is None:
            continue

        bx1, by1, bx2, by2 = [float(v) for v in box]
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        center_inside = (
            expanded_anchor[0] <= cx <= expanded_anchor[2]
            and expanded_anchor[1] <= cy <= expanded_anchor[3]
        )
        ocr_coverage = _box_overlap_coverage(box, clipped_anchor)
        anchor_coverage = _box_overlap_coverage(clipped_anchor, box)
        if not center_inside and ocr_coverage < 0.35 and anchor_coverage < 0.20:
            continue

        vertical_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        min_height = max(1.0, min(ay2 - ay1, by2 - by1))
        if vertical_overlap / min_height < 0.35:
            continue

        distance = abs(cx - ((ax1 + ax2) / 2.0)) + abs(cy - ((ay1 + ay2) / 2.0)) * 2.0
        score = max(ocr_coverage, anchor_coverage) - distance / 10000.0
        matches.append((score, box_idx))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [box_idx for _, box_idx in matches]


def _draw_ocr_indices_as_text_diff(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    target_boxes: Sequence[tuple],
    field_name: str,
    target_box_indices: Sequence[int],
    target_value: object,
    template_value: object = None,
    source: Optional[str] = None,
    template_box: Sequence[float] | None = None,
    base_field: Optional[str] = None,
    force: bool = False,
) -> int:
    added_count = 0
    for b_idx in target_box_indices:
        poly, text_box, ocr_text = _ocr_box_and_poly_for_index(
            target_boxes,
            int(b_idx),
            target_image.shape[:2],
        )
        if text_box is None:
            continue
        draw_box = text_box
        draw_poly = poly
        refined_box = _estimate_value_subspan_box(
            field_name=field_name,
            ocr_text=ocr_text,
            target_value=target_value,
            box=text_box,
            image_shape=target_image.shape[:2],
        )
        if refined_box is not None:
            draw_box = refined_box
            draw_poly = None

        annotation = {
            "kind": "text",
            "status": "diff",
            "field": field_name,
            "ocr_text": ocr_text,
            "target_value": target_value,
        }
        if source:
            annotation["source"] = source
        if template_value is not None:
            annotation["template_value"] = template_value
        if base_field:
            annotation["base_field"] = base_field

        added = _add_text_visualization_annotation(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_annotations,
            template_image=template_image,
            target_image=target_image,
            box=draw_box,
            poly=draw_poly,
            annotation=annotation,
            template_box=template_box,
            force=force,
        )
        if added:
            added_count += 1
    return added_count


def _draw_mapped_template_text_diff(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    field_name: str,
    template_box: Sequence[float],
    template_value: object,
    target_value: object = None,
    source: str,
    base_field: Optional[str] = None,
    force: bool = False,
) -> bool:
    target_box = _map_box_between_shapes(
        template_box,
        template_image.shape[:2],
        target_image.shape[:2],
    )
    if target_box is None:
        return False

    annotation = {
        "kind": "text",
        "status": "diff",
        "field": field_name,
        "source": source,
        "template_value": template_value,
        "target_value": target_value,
        "ocr_text": target_value,
    }
    if base_field:
        annotation["base_field"] = base_field

    return _add_text_visualization_annotation(
        vis_image=vis_image,
        visualization_annotations=visualization_annotations,
        suppressed_annotations=suppressed_annotations,
        template_image=template_image,
        target_image=target_image,
        box=target_box,
        poly=None,
        annotation=annotation,
        template_box=template_box,
        force=force,
    )


def _ocr_text_matches_expected(
    field_name: str,
    expected_value: object,
    observed_text: object,
) -> bool:
    if text_field_values_match(field_name, expected_value, observed_text):
        return True

    if (
        not is_label_field_name(field_name)
        and field_name in {"product_type", "brand"}
        and normalize_text_for_match(expected_value)
        == normalize_text_for_match(observed_text)
    ):
        return True

    expected_norm = normalize_text_for_compare(expected_value)
    observed_norm = normalize_text_for_compare(observed_text)
    if not expected_norm or not observed_norm:
        return False
    return expected_norm in observed_norm


def _add_label_anchor_text_diffs(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_boxes: Sequence[tuple],
    target_boxes: Sequence[tuple],
    structured_fields: Sequence[str],
    template_label_hits: Dict[str, Dict[str, object]],
    target_label_hits: Dict[str, Dict[str, object]],
) -> int:
    if os.getenv("ENABLE_TEXT_FIELD_ANCHOR_FALLBACK", "1") == "0":
        return 0

    added_count = 0
    for field_name in structured_fields:
        if field_name == "address":
            continue
        template_hit = template_label_hits.get(field_name)
        if not template_hit:
            continue

        template_label = template_hit.get("text")
        if not template_label:
            continue

        label_field_name = make_label_field_name(field_name)
        target_hit = target_label_hits.get(field_name)
        target_label = (target_hit or {}).get("text")
        if text_field_values_match(label_field_name, template_label, target_label):
            continue

        template_indices = [
            int(box_idx) for box_idx in (template_hit.get("box_indices") or [])
        ]
        template_label_boxes = []
        for b_idx in template_indices:
            _, box, _ = _ocr_box_and_poly_for_index(
                template_boxes,
                b_idx,
                template_image.shape[:2],
            )
            if box is not None:
                template_label_boxes.append(box)
        template_label_box = _union_boxes(template_label_boxes)

        target_indices = [
            int(box_idx) for box_idx in ((target_hit or {}).get("box_indices") or [])
        ]
        if target_indices:
            added_count += _draw_ocr_indices_as_text_diff(
                vis_image=vis_image,
                visualization_annotations=visualization_annotations,
                suppressed_annotations=suppressed_annotations,
                template_image=template_image,
                target_image=target_image,
                target_boxes=target_boxes,
                field_name=label_field_name,
                target_box_indices=target_indices,
                target_value=target_label,
                template_value=template_label,
                source="label_anchor",
                template_box=template_label_box,
                base_field=field_name,
                force=True,
            )
            continue

        if template_label_box is None:
            continue

        mapped_box = _map_box_between_shapes(
            template_label_box,
            template_image.shape[:2],
            target_image.shape[:2],
        )
        target_overlap_indices = (
            _overlapping_ocr_indices(target_boxes, target_image.shape[:2], mapped_box)
            if mapped_box is not None
            else []
        )
        target_overlap_text = _combined_ocr_text(target_boxes, target_overlap_indices)
        if target_overlap_text and not _ocr_text_matches_expected(
            label_field_name,
            template_label,
            target_overlap_text,
        ):
            added_count += _draw_ocr_indices_as_text_diff(
                vis_image=vis_image,
                visualization_annotations=visualization_annotations,
                suppressed_annotations=suppressed_annotations,
                template_image=template_image,
                target_image=target_image,
                target_boxes=target_boxes,
                field_name=label_field_name,
                target_box_indices=target_overlap_indices[:1],
                target_value=target_overlap_text,
                template_value=template_label,
                source="label_anchor_position",
                template_box=template_label_box,
                base_field=field_name,
                force=True,
            )
            continue

        added = _draw_mapped_template_text_diff(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_annotations,
            template_image=template_image,
            target_image=target_image,
            field_name=label_field_name,
            template_box=template_label_box,
            template_value=template_label,
            target_value=target_overlap_text or target_label,
            source="label_anchor_mapped",
            base_field=field_name,
            force=True,
        )
        if added:
            added_count += 1

    return added_count


def _add_value_anchor_text_diffs(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_boxes: Sequence[tuple],
    target_boxes: Sequence[tuple],
    structured_fields: Sequence[str],
    template_data: Dict[str, object],
) -> int:
    if os.getenv("ENABLE_TEXT_FIELD_ANCHOR_FALLBACK", "1") == "0":
        return 0

    added_count = 0
    for field_name in structured_fields:
        if _has_existing_text_field_annotation(field_name, visualization_annotations):
            continue
        template_value = template_data.get(field_name)
        if template_value is None or str(template_value).strip() in {"", "None"}:
            continue
        if field_name in {"air_volume", "barcode", "manufacturer"}:
            continue

        template_match_indices = find_matching_ocr_boxes(
            template_value,
            template_boxes,
            field_name=field_name,
        )
        if not template_match_indices:
            continue

        for template_box_idx in template_match_indices[:2]:
            _, template_box, template_ocr_text = _ocr_box_and_poly_for_index(
                template_boxes,
                int(template_box_idx),
                template_image.shape[:2],
            )
            if template_box is None:
                continue

            mapped_box = _map_box_between_shapes(
                template_box,
                template_image.shape[:2],
                target_image.shape[:2],
            )
            if mapped_box is None:
                continue

            target_overlap_indices = _overlapping_ocr_indices(
                target_boxes,
                target_image.shape[:2],
                mapped_box,
            )
            target_overlap_text = _combined_ocr_text(
                target_boxes,
                target_overlap_indices,
            )
            if target_overlap_text and _ocr_text_matches_expected(
                field_name,
                template_value,
                target_overlap_text,
            ):
                continue

            if target_overlap_indices:
                added = _draw_ocr_indices_as_text_diff(
                    vis_image=vis_image,
                    visualization_annotations=visualization_annotations,
                    suppressed_annotations=suppressed_annotations,
                    template_image=template_image,
                    target_image=target_image,
                    target_boxes=target_boxes,
                    field_name=field_name,
                    target_box_indices=target_overlap_indices,
                    target_value=target_overlap_text,
                    template_value=template_ocr_text or template_value,
                    source="value_anchor_position",
                    template_box=template_box,
                    force=_should_force_confirmed_text_field(field_name),
                )
                added_count += added
                continue

            added = _draw_mapped_template_text_diff(
                vis_image=vis_image,
                visualization_annotations=visualization_annotations,
                suppressed_annotations=suppressed_annotations,
                template_image=template_image,
                target_image=target_image,
                field_name=field_name,
                template_box=template_box,
                template_value=template_ocr_text or template_value,
                target_value=None,
                source="value_anchor_mapped",
            )
            if added:
                added_count += 1

    return added_count


def _add_aligned_value_visual_diffs(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_boxes: Sequence[tuple],
    structured_fields: Sequence[str],
    template_data: Dict[str, object],
) -> int:
    if os.getenv("ENABLE_ALIGNED_TEXT_VISUAL_SCAN", "1") == "0":
        return 0
    if not _images_nearly_same_size(template_image, target_image):
        return 0

    added_count = 0
    for field_name in structured_fields:
        if field_name not in ALIGNED_VALUE_VISUAL_SCAN_FIELDS:
            continue
        if field_name in {"barcode"}:
            continue
        if _has_existing_text_field_annotation(field_name, visualization_annotations):
            continue
        template_value = template_data.get(field_name)
        if template_value is None or str(template_value).strip() in {"", "None"}:
            continue

        template_match_indices = find_matching_ocr_boxes(
            template_value,
            template_boxes,
            field_name=field_name,
        )
        if not template_match_indices and field_name == "address":
            template_match_indices = _fallback_address_template_indices(
                template_boxes,
                template_image.shape[:2],
            )
        for template_box_idx in template_match_indices[:2]:
            _, template_box, template_ocr_text = _ocr_box_and_poly_for_index(
                template_boxes,
                int(template_box_idx),
                template_image.shape[:2],
            )
            if template_box is None:
                continue
            target_box = _map_box_between_shapes(
                template_box,
                template_image.shape[:2],
                target_image.shape[:2],
            )
            if target_box is None:
                continue
            evidence = _local_text_diff_evidence(
                template_image,
                target_image,
                target_box,
                template_box=template_box,
            )
            if not evidence.get("valid"):
                continue
            if field_name == "address":
                raw_diff_ratio = _raw_text_box_diff_ratio(
                    template_image,
                    target_image,
                    target_box,
                    template_box=template_box,
                )
                if raw_diff_ratio < 0.08:
                    continue
                added = _draw_mapped_template_text_diff(
                    vis_image=vis_image,
                    visualization_annotations=visualization_annotations,
                    suppressed_annotations=suppressed_annotations,
                    template_image=template_image,
                    target_image=target_image,
                    field_name=field_name,
                    template_box=template_box,
                    template_value=template_ocr_text or template_value,
                    target_value=None,
                    source="aligned_address_visual",
                    force=True,
                )
                if added:
                    added_count += 1
                    break
                continue
            if field_name == "air_volume":
                component_box = _local_text_component_box(
                    template_image=template_image,
                    target_image=target_image,
                    target_box=target_box,
                    template_box=template_box,
                )
                if component_box is not None:
                    added = _add_text_visualization_annotation(
                        vis_image=vis_image,
                        visualization_annotations=visualization_annotations,
                        suppressed_annotations=suppressed_annotations,
                        template_image=template_image,
                        target_image=target_image,
                        box=component_box,
                        poly=None,
                        annotation={
                            "kind": "text",
                            "status": "diff",
                            "field": field_name,
                            "source": "aligned_value_visual_component",
                            "template_value": template_ocr_text or template_value,
                            "target_value": None,
                            "ocr_text": None,
                            "localized_component_box": True,
                        },
                        template_box=template_box,
                        force=True,
                    )
                    if added:
                        added_count += 1
                        break
            diff_ratio = float(evidence.get("diff_ratio", 0.0))
            largest_component_ratio = float(
                evidence.get("largest_component_ratio", 0.0)
            )
            max_unmatched_ratio = max(
                float(evidence.get("unmatched_template_ratio", 0.0)),
                float(evidence.get("unmatched_target_ratio", 0.0)),
            )
            if (
                diff_ratio < 0.10
                or largest_component_ratio < 0.04
                or max_unmatched_ratio < 0.08
            ):
                continue
            added = _draw_mapped_template_text_diff(
                vis_image=vis_image,
                visualization_annotations=visualization_annotations,
                suppressed_annotations=suppressed_annotations,
                template_image=template_image,
                target_image=target_image,
                field_name=field_name,
                template_box=template_box,
                template_value=template_ocr_text or template_value,
                target_value=None,
                source="aligned_value_visual",
                force=True,
            )
            if added:
                added_count += 1
                break

    return added_count


def _ocr_items_for_position_matching(
    ocr_boxes: Sequence[tuple],
    image_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    img_h, img_w = image_shape[:2]
    items: List[Dict[str, Any]] = []
    for idx, box_info in enumerate(ocr_boxes):
        text = str(box_info[1] if len(box_info) > 1 else "").strip()
        if not text:
            continue
        box = _ocr_points_to_box(box_info[0] if box_info else None)
        box = _clip_box_to_image(box or [], image_shape) if box is not None else None
        if box is None:
            continue

        x1, y1, x2, y2 = box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        alpha_count = len(re.findall(r"[A-Za-z]", text))
        digit_count = len(re.findall(r"\d", text))
        normalized = normalize_text_for_compare(text)
        if not normalized:
            continue
        if digit_count >= 10 and alpha_count == 0:
            continue

        items.append(
            {
                "idx": idx,
                "text": text,
                "normalized": normalized,
                "box": box,
                "cx": ((x1 + x2) / 2.0) / max(1.0, img_w),
                "cy": ((y1 + y2) / 2.0) / max(1.0, img_h),
                "w": width / max(1.0, img_w),
                "h": height / max(1.0, img_h),
                "area": width * height,
            }
        )
    return items


def _match_ocr_items_by_position(
    template_items: Sequence[Dict[str, Any]],
    target_items: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for t_idx, template_item in enumerate(template_items):
        for s_idx, target_item in enumerate(target_items):
            dy = abs(float(template_item["cy"]) - float(target_item["cy"]))
            dx = abs(float(template_item["cx"]) - float(target_item["cx"]))
            max_h = max(float(template_item["h"]), float(target_item["h"]))
            max_w = max(float(template_item["w"]), float(target_item["w"]))
            if dy > max(0.035, max_h * 0.95):
                continue
            if dx > max(0.10, max_w * 0.90):
                continue

            width_delta = abs(float(template_item["w"]) - float(target_item["w"]))
            height_delta = abs(float(template_item["h"]) - float(target_item["h"]))
            cost = dy * 1.8 + dx + width_delta * 0.35 + height_delta * 0.25
            if cost <= 0.16:
                candidates.append((cost, t_idx, s_idx))

    candidates.sort(key=lambda item: item[0])
    used_template: set[int] = set()
    used_target: set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for cost, t_idx, s_idx in candidates:
        if t_idx in used_template or s_idx in used_target:
            continue
        used_template.add(t_idx)
        used_target.add(s_idx)
        pairs.append((t_idx, s_idx, cost))

    unmatched_template = [
        idx for idx in range(len(template_items)) if idx not in used_template
    ]
    unmatched_target = [
        idx for idx in range(len(target_items)) if idx not in used_target
    ]
    return pairs, unmatched_template, unmatched_target


def _has_existing_text_annotation(
    box: Sequence[float],
    visualization_annotations: Sequence[Dict[str, Any]],
) -> bool:
    for annotation in visualization_annotations:
        if annotation.get("kind") != "text":
            continue
        existing_box = annotation.get("box")
        if existing_box is None:
            continue
        if _box_iou(box, existing_box) >= 0.15:
            return True
        if _box_overlap_coverage(box, existing_box) >= 0.50:
            return True
    return False


def _has_existing_text_field_annotation(
    field_name: str,
    visualization_annotations: Sequence[Dict[str, Any]],
) -> bool:
    return any(
        annotation.get("kind") == "text"
        and annotation.get("field") == field_name
        for annotation in visualization_annotations
    )


def _fallback_address_template_indices(
    template_boxes: Sequence[tuple],
    image_shape: Tuple[int, int],
) -> List[int]:
    img_h, img_w = image_shape[:2]
    candidates: List[Tuple[float, int]] = []
    for idx, box_info in enumerate(template_boxes):
        text = str(box_info[1] if len(box_info) > 1 else "").strip()
        box = _ocr_points_to_box(box_info[0] if box_info else None)
        box = _clip_box_to_image(box or [], image_shape) if box is not None else None
        if box is None:
            continue

        x1, y1, x2, y2 = box
        width_ratio = (x2 - x1) / max(1.0, img_w)
        bottom_ratio = y2 / max(1.0, img_h)
        starts_near_left = x1 <= max(20, int(round(img_w * 0.08)))
        text_norm = normalize_text_for_compare(text)
        looks_like_address = (
            "add" in text_norm
            or "west" in text_norm
            or (bottom_ratio >= 0.92 and width_ratio >= 0.35)
        )
        if not looks_like_address or not starts_near_left:
            continue

        score = width_ratio + bottom_ratio
        if "add" in text_norm:
            score += 0.5
        candidates.append((score, idx))

    candidates.sort(reverse=True)
    return [idx for _, idx in candidates[:1]]


def _add_ocr_position_text_diffs(
    *,
    vis_image: np.ndarray,
    visualization_annotations: List[Dict],
    suppressed_annotations: List[Dict],
    template_image: np.ndarray,
    target_image: np.ndarray,
    template_boxes: Sequence[tuple],
    target_boxes: Sequence[tuple],
) -> int:
    template_items = _ocr_items_for_position_matching(
        template_boxes,
        template_image.shape[:2],
    )
    target_items = _ocr_items_for_position_matching(
        target_boxes,
        target_image.shape[:2],
    )
    pairs, unmatched_template, _ = _match_ocr_items_by_position(
        template_items,
        target_items,
    )

    added_count = 0
    for template_item_idx, target_item_idx, cost in pairs:
        template_item = template_items[template_item_idx]
        target_item = target_items[target_item_idx]
        if template_item["normalized"] == target_item["normalized"]:
            continue

        target_box = target_item["box"]
        if _has_existing_text_annotation(target_box, visualization_annotations):
            continue

        poly, _ = _ocr_points_to_poly_and_box(target_boxes[target_item["idx"]][0])
        added = _add_text_visualization_annotation(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_annotations,
            template_image=template_image,
            target_image=target_image,
            box=target_box,
            poly=poly,
            annotation={
                "kind": "text",
                "status": "diff",
                "field": "ocr_position",
                "source": "ocr_position",
                "match_cost": round(float(cost), 4),
                "template_text": template_item["text"],
                "ocr_text": target_item["text"],
                "target_value": target_item["text"],
            },
            template_box=template_item["box"],
        )
        if added:
            added_count += 1
            print(
                "    - OCR位置文字差异: "
                f"'{template_item['text']}' -> '{target_item['text']}'"
            )

    for template_item_idx in unmatched_template:
        template_item = template_items[template_item_idx]
        target_box = _map_box_between_shapes(
            template_item["box"],
            template_image.shape[:2],
            target_image.shape[:2],
        )
        if target_box is None:
            continue
        if _has_existing_text_annotation(target_box, visualization_annotations):
            continue

        added = _add_text_visualization_annotation(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_annotations,
            template_image=template_image,
            target_image=target_image,
            box=target_box,
            poly=None,
            annotation={
                "kind": "text",
                "status": "diff",
                "field": "ocr_position_missing",
                "source": "ocr_position",
                "template_text": template_item["text"],
                "ocr_text": None,
                "target_value": None,
            },
            template_box=template_item["box"],
        )
        if added:
            added_count += 1
            print(f"    - OCR位置文字缺失: '{template_item['text']}'")

    return added_count


def run_llm_extraction(
    image_path,
    ocr_text,
    model_cls: Type[BaseModel],
    label_kind: str,
    max_retries=None,
):
    """
    使用 LLM 提取结构化标签信息

    Args:
        image_path: 图片路径
        ocr_text: OCR 提取的文本
        max_retries: 最大重试次数

    Returns:
        BaseModel: 结构化数据
    """
    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    llm = get_llm()
    b64_img = encode_image(image_path)
    rule_based = _extract_structured_from_text(label_kind, model_cls, ocr_text)
    standard_rule_based = (
        extract_standard_spec_from_text(ocr_text)
        if label_kind == LABEL_KIND_STANDARD
        else None
    )
    if label_kind == LABEL_KIND_COMPACT and rule_based is not None:
        field_names = list(model_cls.model_fields.keys())
        if not needs_compact_llm(rule_based, field_names):
            return model_cls(**rule_based)
        missing = find_missing_fields(rule_based, field_names)
        suspicious = find_suspicious_fields(rule_based, field_names)
        print(
            "    [规则提取] 转 LLM 补洞/校正: "
            f"missing={missing}, suspicious={suspicious}"
        )

    final_prompt = _build_extraction_prompt(label_kind, ocr_text)
    message = {"role": "user", "content": final_prompt}
    if b64_img:
        message["images"] = [b64_img]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.chat([message], json_mode=True, think=False)
            content = response.content

            # 检查空响应
            if not content or not content.strip():
                print(
                    f"    ⚠ 第 {attempt+1} 次尝试: LLM 返回空响应 "
                    f"({response.debug_summary()})，重试..."
                )
                continue

            # [DEBUG] 打印 LLM 返回的原始内容
            print(f"    [DEBUG] LLM 原始返回 (前500字符):")
            print(f"    {content[:500]}")

            # 尝试提取 JSON
            res_dict = _extract_json_object(content)
            if res_dict is not None:
                data = model_cls(**res_dict)
                if label_kind == LABEL_KIND_COMPACT and rule_based is not None:
                    merged = merge_compact_sources(
                        rule_based,
                        data.model_dump(),
                        model_cls.model_fields.keys(),
                    )
                    print(
                        "    [规则+LLM] 合并后字段数: "
                        f"{count_populated_fields(merged)}"
                    )
                    return model_cls(**merged)
                if (
                    label_kind == LABEL_KIND_STANDARD
                    and standard_rule_based is not None
                    and count_populated_fields(standard_rule_based) > 0
                ):
                    merged = merge_standard_sources(
                        standard_rule_based,
                        data.model_dump(),
                        model_cls.model_fields.keys(),
                    )
                    print(
                        "    [OCR锚点+LLM] 标准标签保留 OCR 锚点: "
                        f"{count_populated_fields(standard_rule_based)}"
                    )
                    return model_cls(**merged)
                return data
            else:
                print(
                    f"    ⚠ 第 {attempt+1} 次尝试: 未找到 JSON "
                    f"({response.debug_summary()})，重试..."
                )
                continue

        except Exception as e:
            last_error = e
            print(f"    ⚠ 第 {attempt+1} 次尝试失败: {e}")
            continue

    # 所有重试都失败，返回空数据
    if label_kind == LABEL_KIND_COMPACT and rule_based is not None:
        print("    [规则提取] LLM 补洞失败，回退规则结果")
        return model_cls(**rule_based)
    if (
        label_kind == LABEL_KIND_STANDARD
        and standard_rule_based is not None
        and count_populated_fields(standard_rule_based) > 0
    ):
        print("    [OCR锚点] LLM 提取失败，回退标准标签 OCR 锚点")
        return model_cls(**standard_rule_based)
    print(f"    ✗ LLM 提取失败，使用空数据")
    return model_cls()


def compare_text_results(data1, data2, output_path=None, extra_fields=None):
    """
    对比两张图片的文字检测结果，输出 Excel

    Args:
        data1: 第一张图片的结构化数据
        data2: 第二张图片的结构化数据
        output_path: 输出 Excel 路径
    """
    if output_path is None:
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, "text_comparison.xlsx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    field_names = list(type(data1).model_fields.keys())
    template_values = {field_name: getattr(data1, field_name) for field_name in field_names}
    target_values = {field_name: getattr(data2, field_name) for field_name in field_names}
    extra_fields = dict(extra_fields or {})
    for field_name, values in extra_fields.items():
        field_names.append(field_name)
        template_values[field_name], target_values[field_name] = values

    data_dict = {"参数": [], "模板图片": [], "实拍图片": [], "是否一致": []}
    match_flags: List[bool] = []

    for k in field_names:
        v1 = template_values.get(k)
        v2 = target_values.get(k)
        is_match = text_field_values_match(k, v1, v2)
        data_dict["参数"].append(k)
        data_dict["模板图片"].append(v1 if v1 else "-")
        data_dict["实拍图片"].append(v2 if v2 else "-")
        data_dict["是否一致"].append("✓" if is_match else "✗")
        match_flags.append(is_match)

    df = pd.DataFrame(data_dict)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="文字对比")

        worksheet = writer.sheets["文字对比"]
        red_font = Font(color="FF0000")

        for row in range(2, len(df) + 2):
            cell1 = worksheet.cell(row=row, column=2)
            cell2 = worksheet.cell(row=row, column=3)
            cell3 = worksheet.cell(row=row, column=4)

            if not match_flags[row - 2]:
                cell1.font = red_font
                cell2.font = red_font
                cell3.font = red_font

    print(f"  文字对比结果已保存: {output_path}")
    return output_path


def _print_graphic_decision(prefix: str, result: Dict) -> None:
    decision = result.get("decision", "unknown")
    conf = result.get("confidence")
    source = result.get("judgment_source", "")
    if decision == "match":
        print(f"{prefix}✅ 匹配 (Conf: {conf}, Source: {source})")
    elif decision == "mismatch":
        print(f"{prefix}❌ 不匹配 (Conf: {conf}, Source: {source})")
    else:
        print(f"{prefix}⚠️  需复核 (Conf: {conf}, Source: {source})")


def _make_unresolved_recovery_result(
    source_side: str,
    source_idx: int,
    reason: str,
    inferred_region: Dict | None,
    source_region: Dict | None = None,
    foreground_ratio: float | None = None,
    exc: Exception | None = None,
) -> Dict:
    if source_side == "实拍":
        subject = f"实拍未匹配区域 #{source_idx}"
        side_hint = "疑似实拍多出图形，或模板侧漏检"
    else:
        subject = f"模板未匹配区域 #{source_idx}"
        side_hint = "疑似实拍缺失图形，或实拍侧漏检"

    is_recovered_target_extra = (
        source_side == "实拍"
        and source_region is not None
        and source_region.get("recovered_uncovered_graphic") is True
    )

    if reason == "inference_failed" and is_recovered_target_extra:
        summary = f"{subject} 是补检出的实拍图形，无法推理对应模板区域，判定实拍多出图形"
        error_type = "recovery_extra_target_graphic"
        decision = "mismatch"
        confidence = 0.90
        needs_review = False
        judgment_source = "recovery_uncovered_target_graphic"
        unresolved_unmatched = False
    elif reason == "inference_failed":
        summary = f"{subject} 无法推理对应区域，{side_hint}"
        error_type = "recovery_inference_failed"
        decision = "unknown"
        confidence = 0.0
        needs_review = True
        judgment_source = "recovery_failed"
        unresolved_unmatched = True
    elif reason == "low_foreground":
        if source_side == "实拍":
            summary = (
                f"{subject} 对应模板区域前景过少 ({foreground_ratio:.3f})，"
                "判定实拍多出图形"
            )
            error_type = "recovery_extra_target_graphic"
        else:
            summary = (
                f"{subject} 对应实拍区域前景过少 ({foreground_ratio:.3f})，"
                "判定实拍缺失图形"
            )
            error_type = "recovery_missing_target_graphic"
        decision = "mismatch"
        confidence = 0.95
        needs_review = False
        judgment_source = "recovery_low_foreground"
        unresolved_unmatched = False
    else:
        summary = f"{subject} 恢复对比失败，{side_hint}"
        if exc is not None:
            summary = f"{summary} ({exc})"
        error_type = "recovery_compare_error"
        decision = "unknown"
        confidence = 0.0
        needs_review = True
        judgment_source = "recovery_failed"
        unresolved_unmatched = True

    return {
        "decision": decision,
        "is_match": False,
        "confidence": confidence,
        "needs_review": needs_review,
        "error_type": error_type,
        "differences": [summary],
        "summary": summary,
        "judgment_source": judgment_source,
        "recovered": True,
        "unresolved_unmatched": unresolved_unmatched,
        "recovery_source_side": source_side,
        "recovery_foreground_ratio": foreground_ratio,
        "template_idx": None,
        "target_idx": None,
        "match_distance": None,
        "inferred_template_box": (
            inferred_region["coordinate"] if source_side == "实拍" and inferred_region is not None else None
        ),
        "inferred_target_box": (
            inferred_region["coordinate"] if source_side == "模板" and inferred_region is not None else None
        ),
    }


def _recover_unmatched_regions(
    template_regions: List[Dict],
    target_regions: List[Dict],
    matched_pairs: List[tuple],
    unmatched1: List[int],
    unmatched2: List[int],
    template_image: np.ndarray,
    target_image: np.ndarray,
    output_dir: Optional[str],
    use_vlm: bool,
) -> Dict[str, object]:
    recovery_results: List[Dict] = []
    recovered_template_regions: List[Dict] = []
    recovered_target_regions: List[Dict] = []
    resolved_unmatched1 = set()
    resolved_unmatched2 = set()
    pair_idx = len(matched_pairs)

    def _run_recovery(
        source_side: str,
        source_idx: int,
        source_region: Dict,
        inferred_region: Dict | None,
        compare_box1: List[float],
        compare_box2: List[float],
        foreground_ratio: float,
    ) -> None:
        nonlocal pair_idx

        def _finalize_result(result: Dict) -> None:
            result.setdefault("recovered", True)
            result["recovery_source_side"] = source_side
            result["recovery_foreground_ratio"] = foreground_ratio
            result.setdefault("template_idx", None)
            result.setdefault("target_idx", None)
            result.setdefault("match_distance", None)
            result.setdefault("inferred_template_box", None)
            result.setdefault("inferred_target_box", None)

            if source_side == "实拍":
                result["target_idx"] = source_idx
                if inferred_region is not None:
                    result["inferred_template_box"] = inferred_region["coordinate"]
            else:
                result["template_idx"] = source_idx
                if inferred_region is not None:
                    result["inferred_target_box"] = inferred_region["coordinate"]

            recovery_results.append(result)
            _print_graphic_decision("      恢复判定: ", result)

            if result.get("decision") in {"match", "mismatch"} and not result.get(
                "needs_review", False
            ):
                if source_side == "实拍":
                    resolved_unmatched2.add(source_idx)
                    if result.get("decision") == "match" and inferred_region is not None:
                        recovered_template_regions.append(inferred_region)
                else:
                    resolved_unmatched1.add(source_idx)
                    if result.get("decision") == "match" and inferred_region is not None:
                        recovered_target_regions.append(inferred_region)

        if inferred_region is None:
            print(f"    - {source_side} 未匹配区域 #{source_idx}: 无法推理对应区域")
            _finalize_result(
                _make_unresolved_recovery_result(
                    source_side=source_side,
                    source_idx=source_idx,
                    reason="inference_failed",
                    inferred_region=None,
                    source_region=source_region,
                )
            )
            return

        if foreground_ratio < 0.01:
            print(
                f"    - {source_side} 未匹配区域 #{source_idx}: "
                f"推理区域前景过少 ({foreground_ratio:.3f})"
            )
            _finalize_result(
                _make_unresolved_recovery_result(
                    source_side=source_side,
                    source_idx=source_idx,
                    reason="low_foreground",
                    inferred_region=inferred_region,
                    source_region=source_region,
                    foreground_ratio=foreground_ratio,
                )
            )
            return

        print(
            f"    - {source_side} 未匹配区域 #{source_idx}: "
            f"尝试推理框 {inferred_region['coordinate']}"
        )

        try:
            result = compare_region_pair(
                template_image,
                target_image,
                compare_box1,
                compare_box2,
                output_dir=output_dir,
                pair_idx=pair_idx,
                use_vlm=use_vlm,
            )
        except Exception as exc:
            print(f"      恢复对比出错: {exc}")
            _finalize_result(
                _make_unresolved_recovery_result(
                    source_side=source_side,
                    source_idx=source_idx,
                    reason="compare_error",
                    inferred_region=inferred_region,
                    source_region=source_region,
                    foreground_ratio=foreground_ratio,
                    exc=exc,
                )
            )
            pair_idx += 1
            return

        _finalize_result(result)
        pair_idx += 1

    if unmatched2:
        print("  - 尝试根据实拍未匹配区域恢复模板漏检...")
    for target_idx in unmatched2:
        target_region = target_regions[target_idx]
        inferred_template_region = infer_corresponding_region(
            target_region,
            target_image.shape[:2],
            template_image.shape[:2],
            matched_pairs,
            template_regions,
            target_regions,
            source_side="target",
        )
        foreground_ratio = (
            estimate_region_foreground_ratio(
                template_image,
                inferred_template_region["coordinate"],
            )
            if inferred_template_region is not None
            else 0.0
        )
        compare_box1 = inferred_template_region["coordinate"] if inferred_template_region else [0, 0, 0, 0]
        compare_box2 = target_region["coordinate"]
        _run_recovery(
            "实拍",
            target_idx,
            target_region,
            inferred_template_region,
            compare_box1,
            compare_box2,
            foreground_ratio,
        )

    if unmatched1:
        print("  - 尝试根据模板未匹配区域恢复实拍漏检...")
    for template_idx in unmatched1:
        template_region = template_regions[template_idx]
        inferred_target_region = infer_corresponding_region(
            template_region,
            template_image.shape[:2],
            target_image.shape[:2],
            matched_pairs,
            template_regions,
            target_regions,
            source_side="template",
        )
        foreground_ratio = (
            estimate_region_foreground_ratio(
                target_image,
                inferred_target_region["coordinate"],
            )
            if inferred_target_region is not None
            else 0.0
        )
        compare_box1 = template_region["coordinate"]
        compare_box2 = inferred_target_region["coordinate"] if inferred_target_region else [0, 0, 0, 0]
        _run_recovery(
            "模板",
            template_idx,
            template_region,
            inferred_target_region,
            compare_box1,
            compare_box2,
            foreground_ratio,
        )

    return {
        "recovery_results": recovery_results,
        "recovered_template_regions": recovered_template_regions,
        "recovered_target_regions": recovered_target_regions,
        "resolved_unmatched1": sorted(resolved_unmatched1),
        "resolved_unmatched2": sorted(resolved_unmatched2),
    }


def run_unified_detection(
    template_input_path,
    target_image_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_mode="debug",
):
    """
    完整的整合检测流程（使用 VLM 进行图形对比）

    Args:
        template_input_path: 模板文件路径（支持 PDF 或图片）
        target_image_path: 实拍图片路径
        output_dir: 输出目录
        output_mode: 输出模式，`final` 仅保留最终结果，`debug` 保留详细中间产物

    Returns:
        dict: 包含文字和图形检测结果
    """
    output_options = WorkflowOutputOptions.from_mode(output_mode)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("整合检测：文字检测 + 布局区域图形比较 (VLM Mode)")
    print("=" * 60)

    results = {
        "success": True,
        "text_detection": {},
        "graphic_comparison": {},
        "output_mode": output_options.mode,
    }
    with TemporaryDirectory(
        prefix="label-detection-"
    ) as temp_root, paddle_cache_cleanup_scope(reason="run_unified_detection"):
        work_root = output_root / "debug" if output_options.mode == "debug" else Path(temp_root)
        work_root.mkdir(parents=True, exist_ok=True)

        preprocess_dir = work_root / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        template_assets_dir = work_root / "template_assets"
        graphic_output_dir = (
            work_root / "graphic_comparison" if output_options.save_graphic_debug else None
        )
        if graphic_output_dir is not None:
            graphic_output_dir.mkdir(parents=True, exist_ok=True)
        graphic_output_dir_str = str(graphic_output_dir) if graphic_output_dir is not None else None
        target_preprocessed_path = preprocess_dir / "target_preprocessed.png"
        text_excel_path = None

        # ========== Step 1: 预处理 ==========
        print("\n" + "=" * 40)
        print("Step 1: 预处理")
        print("=" * 40)

        # 1.1 解析模板输入
        print("\n[1.1] 解析模板输入...")
        template_raw_path, template_source_type, err = resolve_template_input(
            template_input_path,
            output_dir=str(template_assets_dir),
        )
        if err:
            return {"success": False, "error": err}
        if template_source_type == "pdf":
            print("  模板来源: PDF，已提取红框区域")
        else:
            print("  模板来源: 图片，直接使用原图")

        # 1.2 预处理模板（检测黑框，去除白边）
        print("\n[1.2] 预处理模板图片（去除黑框外白边）...")
        template_cropped, template_path, err = preprocess_template_image(
            template_raw_path,
            str(preprocess_dir),
        )
        if err:
            return {"success": False, "error": err}

        # 1.3 预处理实拍图片（带角度矫正）
        print("\n[1.3] 预处理实拍图片（带角度矫正）...")
        target_cropped, _, err = preprocess_target(
            target_image_path,
            str(preprocess_dir),
            template_image=template_cropped,
        )
        if err:
            return {"success": False, "error": err}

        # 保存预处理后的实拍图到工作目录，供 OCR 和布局检测复用
        cv2.imwrite(str(target_preprocessed_path), target_cropped)

        # ========== Step 2: 文字检测 ==========
        print("\n" + "=" * 40)
        print("Step 2: 文字检测")
        print("=" * 40)

        # 2.1 OCR 提取模板文字和坐标
        print("\n[2.1] OCR 提取模板图片文字...")
        start_time = time.time()
        template_text, template_boxes = get_ocr_with_boxes(template_path)
        print(f"  识别到 {len(template_boxes)} 个文字区域，共 {len(template_text)} 字符")

        # 2.2 OCR 提取实拍文字和坐标
        print("\n[2.2] OCR 提取实拍图片文字...")
        target_text, target_boxes = get_ocr_with_boxes(str(target_preprocessed_path))
        print(f"  识别到 {len(target_boxes)} 个文字区域，共 {len(target_text)} 字符")

        label_kind = infer_label_kind(template_text or target_text)
        model_cls = get_label_model(label_kind)
        structured_fields = list(model_cls.model_fields.keys())
        print(f"\n[2.3] 识别文字标签类型: {label_kind} ({len(structured_fields)} 个字段)")
        print("[2.3] LLM/规则提取模板结构化信息...")
        llm_start = time.time()
        data1 = run_llm_extraction(template_path, template_text, model_cls, label_kind)
        print(f"  LLM 耗时: {time.time() - llm_start:.2f} 秒")

        print("\n[2.4] LLM/规则提取实拍结构化信息...")
        llm_start = time.time()
        data2 = run_llm_extraction(
            str(target_preprocessed_path),
            target_text,
            model_cls,
            label_kind,
        )
        print(f"  LLM 耗时: {time.time() - llm_start:.2f} 秒")

        template_data = data1.model_dump()
        target_data = data2.model_dump()
        template_label_hits = extract_field_labels_from_ocr_boxes(
            template_boxes,
            structured_fields,
        )
        target_label_hits = extract_field_labels_from_ocr_boxes(
            target_boxes,
            structured_fields,
        )
        label_extra_fields = {}
        label_fields = []
        for field_name in structured_fields:
            if field_name == "address":
                continue
            template_label = (template_label_hits.get(field_name) or {}).get("text")
            target_label = (target_label_hits.get(field_name) or {}).get("text")
            if not template_label:
                continue

            label_field_name = make_label_field_name(field_name)
            label_fields.append(label_field_name)
            label_extra_fields[label_field_name] = (template_label, target_label)
            template_data[label_field_name] = template_label
            target_data[label_field_name] = target_label

        comparison_fields = structured_fields + label_fields
        if label_fields:
            print(f"  附加标签名比对: {len(label_fields)} 项")

        # 2.5 对比文字结果
        print("\n[2.5] 对比文字结果...")
        if output_options.save_text_excel:
            text_excel_path = compare_text_results(
                data1,
                data2,
                str(work_root / "text_comparison.xlsx"),
                extra_fields=label_extra_fields,
            )

        results["text_detection"] = {
            "label_kind": label_kind,
            "fields": comparison_fields,
            "value_fields": structured_fields,
            "label_fields": label_fields,
            "template_data": template_data,
            "target_data": target_data,
            "detected_labels": {
                "template": {
                    field_name: item["text"] for field_name, item in template_label_hits.items()
                },
                "target": {
                    field_name: item["text"] for field_name, item in target_label_hits.items()
                },
            },
            "excel_path": text_excel_path,
            "time": time.time() - start_time,
        }

        # ========== Step 3: 区域检测与对比 ==========
        print("\n" + "=" * 40)
        print("Step 3: 区域检测与对比")
        print("=" * 40)

        # 3.1 检测布局区域 (使用 PP-DocLayoutV3)
        print("\n[3.1] 检测布局区域 (PP-DocLayoutV3)...")
        template_all_regions = detect_layout_regions(template_path)
        template_regions = extract_regions_by_type(template_all_regions, "image")

        target_all_regions = detect_layout_regions(str(target_preprocessed_path))
        target_regions = extract_regions_by_type(target_all_regions, "image")

        skipped_template_regions: List[Dict] = []
        skipped_target_regions: List[Dict] = []
        split_template_regions: List[Dict] = []
        split_target_regions: List[Dict] = []
        skipped_split_template_regions: List[Dict] = []
        skipped_split_target_regions: List[Dict] = []
        recovered_uncovered_template_regions: List[Dict] = []
        recovered_uncovered_target_regions: List[Dict] = []
        template_regions, skipped_template_regions = split_barcode_regions(
            template_regions,
            template_cropped,
            template_boxes,
        )
        target_regions, skipped_target_regions = split_barcode_regions(
            target_regions,
            target_cropped,
            target_boxes,
        )
        if ENABLE_IMAGE_REGION_SPLIT:
            template_regions, split_template_regions = split_composite_image_regions(
                template_regions,
                template_cropped,
                template_boxes,
            )
            template_regions = merge_fragmented_split_regions(template_regions)
            template_regions, skipped_split_template_regions = filter_split_image_regions(
                template_regions,
                template_cropped,
                template_boxes,
            )
            target_regions, split_target_regions = split_composite_image_regions(
                target_regions,
                target_cropped,
                target_boxes,
            )
            target_regions = merge_fragmented_split_regions(target_regions)
            target_regions, skipped_split_target_regions = filter_split_image_regions(
                target_regions,
                target_cropped,
                target_boxes,
            )
            template_regions, recovered_uncovered_template_regions = (
                recover_uncovered_graphic_regions(
                    template_regions,
                    template_cropped,
                    template_boxes,
                    ignored_regions=_ignored_regions_for_uncovered_recovery(
                        skipped_template_regions + skipped_split_template_regions
                    ),
                )
            )
            target_regions, recovered_uncovered_target_regions = (
                recover_uncovered_graphic_regions(
                    target_regions,
                    target_cropped,
                    target_boxes,
                    ignored_regions=_ignored_regions_for_uncovered_recovery(
                        skipped_target_regions + skipped_split_target_regions
                    ),
                )
            )

        print(
            "  - 模板图片区域: "
            f"{len(template_regions)} 个可比对 / {len(skipped_template_regions)} 个条码跳过 "
            f"(总检测 {len(template_all_regions)})"
        )
        print(
            "  - 实拍图片区域: "
            f"{len(target_regions)} 个可比对 / {len(skipped_target_regions)} 个条码跳过 "
            f"(总检测 {len(target_all_regions)})"
        )
        if ENABLE_IMAGE_REGION_SPLIT:
            print(
                "  - 图片区域二次拆分: "
                f"模板拆分 {len(split_template_regions)} 个大框, "
                f"实拍拆分 {len(split_target_regions)} 个大框"
            )
            print(
                "  - 拆分后过滤: "
                f"模板跳过 {len(skipped_split_template_regions)} 个子框, "
                f"实拍跳过 {len(skipped_split_target_regions)} 个子框"
            )
            if recovered_uncovered_template_regions or recovered_uncovered_target_regions:
                print(
                    "  - 未覆盖图形补检: "
                    f"模板补检 {len(recovered_uncovered_template_regions)} 个, "
                    f"实拍补检 {len(recovered_uncovered_target_regions)} 个"
                )

        # 3.2 区域匹配
        print("\n[3.2] 匹配对应区域...")
        matched_pairs, unmatched1, unmatched2 = match_regions(
            template_regions,
            target_regions,
            template_cropped.shape[:2],
            target_cropped.shape[:2],
        )

        print(f"  - 匹配对数: {len(matched_pairs)}")
        print(f"  - 模板未匹配: {len(unmatched1)}")
        print(f"  - 实拍未匹配: {len(unmatched2)}")

        # 3.3 区域内容对比
        print("\n[3.3] 对比匹配区域内容...")
        comparison_results = []

        for idx, (i, j, dist) in enumerate(matched_pairs):
            region1 = template_regions[i]
            region2 = target_regions[j]

            print(f"  >>> 对比区域对 #{idx+1} [T{i} <-> S{j}]...")

            try:
                res = compare_region_pair(
                    template_cropped,
                    target_cropped,
                    region1["coordinate"],
                    region2["coordinate"],
                    output_dir=graphic_output_dir_str,
                    pair_idx=idx,
                    use_vlm=USE_VLM_FOR_GRAPHIC,
                )
                res["template_idx"] = i
                res["target_idx"] = j
                res["match_distance"] = dist

                decision = res.get("decision", "unknown")
                conf = res.get("confidence")
                source = res.get("judgment_source", "")
                if decision == "match":
                    print(f"      判定: ✅ 匹配 (Conf: {conf}, Source: {source})")
                elif decision == "mismatch":
                    print(f"      判定: ❌ 不匹配 (Conf: {conf}, Source: {source})")
                else:
                    print(f"      判定: ⚠️  需复核 (Conf: {conf}, Source: {source})")

                comparison_results.append(res)

            except Exception as e:
                print(f"      对比出错: {e}")

        # 3.4 未匹配区域恢复
        recovered_template_regions: List[Dict] = []
        recovered_target_regions: List[Dict] = []
        remaining_unmatched1 = list(unmatched1)
        remaining_unmatched2 = list(unmatched2)

        if unmatched1 or unmatched2:
            print("\n[3.4] 尝试恢复未匹配区域...")
            recovery_payload = _recover_unmatched_regions(
                template_regions,
                target_regions,
                matched_pairs,
                unmatched1,
                unmatched2,
                template_cropped,
                target_cropped,
                graphic_output_dir_str,
                USE_VLM_FOR_GRAPHIC,
            )
            recovery_results = recovery_payload["recovery_results"]
            recovered_template_regions = recovery_payload["recovered_template_regions"]
            recovered_target_regions = recovery_payload["recovered_target_regions"]
            resolved_unmatched1 = set(recovery_payload["resolved_unmatched1"])
            resolved_unmatched2 = set(recovery_payload["resolved_unmatched2"])
            remaining_unmatched1 = [
                idx for idx in unmatched1 if idx not in resolved_unmatched1
            ]
            remaining_unmatched2 = [
                idx for idx in unmatched2 if idx not in resolved_unmatched2
            ]
            comparison_results.extend(recovery_results)
            print(
                "  - 恢复结果: "
                f"模板补回 {len(recovered_template_regions)} 个, "
                f"实拍补回 {len(recovered_target_regions)} 个"
            )

        if output_options.save_graphic_debug and graphic_output_dir is not None:
            template_vis = draw_regions(template_cropped, template_regions)
            target_vis = draw_regions(target_cropped, target_regions)
            if skipped_template_regions:
                template_vis = draw_regions(
                    template_vis,
                    skipped_template_regions,
                    color=(0, 165, 255),
                )
            if skipped_target_regions:
                target_vis = draw_regions(
                    target_vis,
                    skipped_target_regions,
                    color=(0, 165, 255),
                )
            if skipped_split_template_regions:
                template_vis = draw_regions(
                    template_vis,
                    skipped_split_template_regions,
                    color=(0, 0, 255),
                )
            if skipped_split_target_regions:
                target_vis = draw_regions(
                    target_vis,
                    skipped_split_target_regions,
                    color=(0, 0, 255),
                )
            if recovered_template_regions:
                template_vis = draw_regions(
                    template_vis,
                    recovered_template_regions,
                    color=(255, 128, 0),
                )
            if recovered_target_regions:
                target_vis = draw_regions(
                    target_vis,
                    recovered_target_regions,
                    color=(255, 128, 0),
                )
            cv2.imwrite(
                str(graphic_output_dir / "template_regions_detected.jpg"),
                template_vis,
            )
            cv2.imwrite(
                str(graphic_output_dir / "target_regions_detected.jpg"),
                target_vis,
            )

        resolved_match_count = (
            len(matched_pairs)
            + len(recovered_template_regions)
            + len(recovered_target_regions)
        )
        results["graphic_comparison"] = {
            "template_regions_total_count": len(
                extract_regions_by_type(template_all_regions, "image")
            ),
            "target_regions_total_count": len(
                extract_regions_by_type(target_all_regions, "image")
            ),
            "template_regions_count": len(template_regions),
            "target_regions_count": len(target_regions),
            "skipped_template_regions": skipped_template_regions,
            "skipped_target_regions": skipped_target_regions,
            "split_template_regions": split_template_regions,
            "split_target_regions": split_target_regions,
            "skipped_split_template_regions": skipped_split_template_regions,
            "skipped_split_target_regions": skipped_split_target_regions,
            "recovered_uncovered_template_regions": recovered_uncovered_template_regions,
            "recovered_uncovered_target_regions": recovered_uncovered_target_regions,
            "recovered_template_regions": recovered_template_regions,
            "recovered_target_regions": recovered_target_regions,
            "remaining_unmatched_template": remaining_unmatched1,
            "remaining_unmatched_target": remaining_unmatched2,
            "matched_count": len(matched_pairs),
            "recovered_match_count": len(recovered_template_regions)
            + len(recovered_target_regions),
            "resolved_match_count": resolved_match_count,
            "effective_matched_count": resolved_match_count,
            "comparison_results": comparison_results,
            "region_type": "image",
        }
        results["template_input"] = {
            "source_path": template_input_path,
            "source_type": template_source_type,
            "resolved_image_path": (
                template_raw_path
                if template_source_type == "image"
                or output_options.save_template_assets
                else None
            ),
        }

        # ========== Step 4: 结果可视化 (差异标注) ==========
        print("\n" + "=" * 40)
        print("Step 4: 结果可视化 (标注差异)")
        print("=" * 40)

        vis_image = target_cropped.copy()
        diff_count = 0
        visualization_annotations: List[Dict] = []
        suppressed_visualization_annotations: List[Dict] = []

        print("  寻找并标注差异文字区域...")
        for k in comparison_fields:
            v1 = template_data.get(k)
            v2 = target_data.get(k)

            if not text_field_values_match(k, v1, v2):
                if is_label_field_name(k):
                    base_field_name = get_base_field_name(k)
                    label_hit = target_label_hits.get(base_field_name)
                    if label_hit is None:
                        print(
                            f"    - 字段标签 '{base_field_name}' 差异 (值: {v2}) -> 未找到对应 OCR 框"
                        )
                        continue

                    matched_box_indices = [
                        int(box_idx)
                        for box_idx in (label_hit.get("box_indices") or [])
                    ]
                    if not matched_box_indices:
                        print(
                            f"    - 字段标签 '{base_field_name}' 差异 (值: {v2}) -> 未找到对应 OCR 框"
                        )
                        continue
                    matched_texts = [target_boxes[b_idx][1] for b_idx in matched_box_indices]
                    print(
                        f"    - 字段标签 '{base_field_name}' 差异 (值: {v2}) -> 对应 "
                        f"{len(matched_box_indices)} 个 OCR 框: {matched_texts}"
                    )
                    for b_idx in matched_box_indices:
                        poly, text_box = _ocr_points_to_poly_and_box(target_boxes[b_idx][0])
                        if text_box is None:
                            continue
                        added = _add_text_visualization_annotation(
                            vis_image=vis_image,
                            visualization_annotations=visualization_annotations,
                            suppressed_annotations=suppressed_visualization_annotations,
                            template_image=template_cropped,
                            target_image=target_cropped,
                            box=text_box,
                            poly=poly,
                            annotation={
                                "kind": "text",
                                "status": "diff",
                                "field": k,
                                "base_field": base_field_name,
                                "ocr_text": target_boxes[b_idx][1],
                                "target_value": v2,
                            },
                            force=True,
                        )
                        if added:
                            diff_count += 1
                    continue

                target_val = str(v2) if v2 else ""
                if not target_val or target_val == "None":
                    continue

                matched_box_indices = find_matching_ocr_boxes(
                    target_val,
                    target_boxes,
                    field_name=k,
                )

                if matched_box_indices:
                    matched_texts = [
                        target_boxes[b_idx][1] for b_idx in matched_box_indices
                    ]
                    print(
                        f"    - 字段 '{k}' 差异 (值: {target_val}) -> 对应 "
                        f"{len(matched_box_indices)} 个 OCR 框: {matched_texts}"
                    )
                    for b_idx in matched_box_indices:
                        poly, text_box = _ocr_points_to_poly_and_box(target_boxes[b_idx][0])
                        if text_box is None:
                            continue
                        ocr_text = target_boxes[b_idx][1]
                        draw_box = text_box
                        draw_poly = poly
                        refined_box = _estimate_value_subspan_box(
                            field_name=k,
                            ocr_text=ocr_text,
                            target_value=target_val,
                            box=text_box,
                            image_shape=target_cropped.shape[:2],
                        )
                        if refined_box is not None:
                            draw_box = refined_box
                            draw_poly = None
                        added = _add_text_visualization_annotation(
                            vis_image=vis_image,
                            visualization_annotations=visualization_annotations,
                            suppressed_annotations=suppressed_visualization_annotations,
                            template_image=template_cropped,
                            target_image=target_cropped,
                            box=draw_box,
                            poly=draw_poly,
                            annotation={
                                "kind": "text",
                                "status": "diff",
                                "field": k,
                                "ocr_text": ocr_text,
                                "target_value": target_val,
                            },
                            force=_should_force_confirmed_text_field(k),
                        )
                        if added:
                            diff_count += 1
                else:
                    print(
                        f"    - 字段 '{k}' 差异 (值: {target_val}) -> 未找到对应 OCR 框"
                    )
                    if k not in {"brand", "product_type", "manufacturer", "barcode"}:
                        template_val = str(v1) if v1 else ""
                        template_match_indices = find_matching_ocr_boxes(
                            template_val,
                            template_boxes,
                            field_name=k,
                        )
                        added_mapped_field = False
                        for template_box_idx in template_match_indices[:1]:
                            _, template_box, template_ocr_text = (
                                _ocr_box_and_poly_for_index(
                                    template_boxes,
                                    int(template_box_idx),
                                    template_cropped.shape[:2],
                                )
                            )
                            if template_box is None:
                                continue
                            added = _draw_mapped_template_text_diff(
                                vis_image=vis_image,
                                visualization_annotations=visualization_annotations,
                                suppressed_annotations=suppressed_visualization_annotations,
                                template_image=template_cropped,
                                target_image=target_cropped,
                                field_name=k,
                                template_box=template_box,
                                template_value=template_ocr_text or template_val,
                                target_value=target_val,
                                source="structured_value_mapped",
                                force=True,
                            )
                            if added:
                                diff_count += 1
                                added_mapped_field = True
                                print(
                                    f"    - 字段 '{k}' 使用模板位置补框: "
                                    f"{template_ocr_text or template_val}"
                                )
                                break
                        if not added_mapped_field:
                            template_label_hit = template_label_hits.get(k)
                            label_indices = [
                                int(box_idx)
                                for box_idx in (
                                    (template_label_hit or {}).get("box_indices") or []
                                )
                            ]
                            label_boxes = []
                            for label_box_idx in label_indices:
                                _, label_box, _ = _ocr_box_and_poly_for_index(
                                    template_boxes,
                                    label_box_idx,
                                    template_cropped.shape[:2],
                                )
                                if label_box is not None:
                                    label_boxes.append(label_box)
                            template_label_box = _union_boxes(label_boxes)
                            if template_label_box is not None:
                                added = _draw_mapped_template_text_diff(
                                    vis_image=vis_image,
                                    visualization_annotations=visualization_annotations,
                                    suppressed_annotations=suppressed_visualization_annotations,
                                    template_image=template_cropped,
                                    target_image=target_cropped,
                                    field_name=k,
                                    template_box=template_label_box,
                                    template_value=template_val,
                                    target_value=target_val,
                                    source="structured_label_position_mapped",
                                    force=True,
                                )
                                if added:
                                    diff_count += 1
                                    print(
                                        f"    - 字段 '{k}' 使用模板标签位置补框"
                                    )

        supplemental_text_count = _add_label_anchor_text_diffs(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_visualization_annotations,
            template_image=template_cropped,
            target_image=target_cropped,
            template_boxes=template_boxes,
            target_boxes=target_boxes,
            structured_fields=structured_fields,
            template_label_hits=template_label_hits,
            target_label_hits=target_label_hits,
        )
        supplemental_text_count += _add_value_anchor_text_diffs(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_visualization_annotations,
            template_image=template_cropped,
            target_image=target_cropped,
            template_boxes=template_boxes,
            target_boxes=target_boxes,
            structured_fields=structured_fields,
            template_data=template_data,
        )
        supplemental_text_count += _add_aligned_value_visual_diffs(
            vis_image=vis_image,
            visualization_annotations=visualization_annotations,
            suppressed_annotations=suppressed_visualization_annotations,
            template_image=template_cropped,
            target_image=target_cropped,
            template_boxes=template_boxes,
            structured_fields=structured_fields,
            template_data=template_data,
        )
        diff_count += supplemental_text_count
        if supplemental_text_count:
            print(f"  字段锚点文字补框: {supplemental_text_count} 处")

        if os.getenv("ENABLE_OCR_POSITION_TEXT_FALLBACK", "0") == "1":
            position_text_count = _add_ocr_position_text_diffs(
                vis_image=vis_image,
                visualization_annotations=visualization_annotations,
                suppressed_annotations=suppressed_visualization_annotations,
                template_image=template_cropped,
                target_image=target_cropped,
                template_boxes=template_boxes,
                target_boxes=target_boxes,
            )
            diff_count += position_text_count
            if position_text_count:
                print(f"  OCR位置文字兜底补框: {position_text_count} 处")

        print("  标注差异和需复核的图形区域...")
        needs_review_regions = []
        unresolved_regions = []
        suppressed_graphic_regions = []
        for res in comparison_results:
            decision = res.get("decision", "unknown")
            target_idx = res.get("target_idx")
            template_idx = res.get("template_idx")
            region_box = None
            template_region_box = None

            if target_idx is not None and target_idx < len(target_regions):
                region_box = target_regions[target_idx]["coordinate"]
            elif res.get("inferred_target_box") is not None:
                region_box = res["inferred_target_box"]

            if template_idx is not None and template_idx < len(template_regions):
                template_region_box = template_regions[template_idx]["coordinate"]
            elif res.get("inferred_template_box") is not None:
                template_region_box = res["inferred_template_box"]

            if region_box is not None:
                x1, y1, x2, y2 = [int(v) for v in region_box]
                verification_evidence = None
                if (
                    res.get("unresolved_unmatched")
                    or decision == "unknown"
                    or decision == "mismatch"
                ):
                    mapped_template_box = _map_box_between_shapes(
                        region_box,
                        target_cropped.shape[:2],
                        template_cropped.shape[:2],
                    )
                    verification_template_box = mapped_template_box or template_region_box
                    verification_evidence = _local_graphic_diff_evidence(
                        template_cropped,
                        target_cropped,
                        verification_template_box,
                        region_box,
                    )
                    verification_evidence["detector_template_box"] = template_region_box
                    verification_evidence["mapped_template_box"] = mapped_template_box
                    res["local_diff_evidence"] = verification_evidence
                    if _should_suppress_graphic_visual_box(verification_evidence):
                        res["visualization_suppressed"] = True
                        res["visualization_suppression_reason"] = "local_diff_low"
                        suppressed_annotation = {
                            "kind": "graphic",
                            "status": "suppressed",
                            "decision": decision,
                            "target_idx": target_idx,
                            "template_idx": template_idx,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "reason": "local_diff_low",
                            "local_diff_evidence": verification_evidence,
                            "summary": res.get("summary", ""),
                        }
                        suppressed_graphic_regions.append(suppressed_annotation)
                        suppressed_visualization_annotations.append(suppressed_annotation)
                        print(
                            "    - 图形框过滤: 区域 "
                            f"#{target_idx if target_idx is not None else 'recovered'} "
                            f"(local_diff={verification_evidence.get('diff_ratio')}, "
                            f"component={verification_evidence.get('largest_component_ratio')})"
                        )
                        continue

                visual_region_box = _expand_small_graphic_visual_box(
                    region_box,
                    target_cropped.shape[:2],
                    verification_evidence,
                )
                if visual_region_box is not None:
                    x1, y1, x2, y2 = [int(v) for v in visual_region_box]
                    if verification_evidence is not None:
                        verification_evidence["visual_box"] = visual_region_box

                if res.get("unresolved_unmatched"):
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 128, 255), 3)
                    cv2.putText(
                        vis_image,
                        "Unmatched",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 128, 255),
                        2,
                    )
                    visualization_annotations.append(
                        {
                            "kind": "graphic",
                            "status": "unmatched",
                            "decision": decision,
                            "target_idx": target_idx,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "local_diff_evidence": verification_evidence,
                            "summary": res.get("summary", ""),
                        }
                    )
                    unresolved_regions.append(res)
                    print(
                        "    - 图形未恢复: 区域 "
                        f"#{target_idx if target_idx is not None else 'recovered'} "
                        f"({res.get('summary', '')})"
                    )
                elif decision == "unknown":
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    cv2.putText(
                        vis_image,
                        "Review",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )
                    visualization_annotations.append(
                        {
                            "kind": "graphic",
                            "status": "review",
                            "decision": decision,
                            "target_idx": target_idx,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "local_diff_evidence": verification_evidence,
                            "summary": res.get("summary", ""),
                        }
                    )
                    needs_review_regions.append(res)
                    print(
                        "    - 图形需复核: 区域 "
                        f"#{target_idx if target_idx is not None else 'recovered'} "
                        f"({res.get('summary', '')})"
                    )
                elif decision == "mismatch":
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        vis_image,
                        "Diff",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    visualization_annotations.append(
                        {
                            "kind": "graphic",
                            "status": "diff",
                            "decision": decision,
                            "target_idx": target_idx,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "local_diff_evidence": verification_evidence,
                            "summary": res.get("summary", ""),
                        }
                    )
                    diff_count += 1
                    print(
                        "    - 图形差异: 区域 "
                        f"#{target_idx if target_idx is not None else 'recovered'} "
                        "(确认不匹配)"
                    )

        vis_path = output_root / "visualization_diff.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        print(
            f"  差异可视化已保存: {vis_path} "
            f"(差异 {diff_count} 处, 未恢复 {len(unresolved_regions)} 处, "
            f"待复核 {len(needs_review_regions)} 处, "
            f"过滤 {len(suppressed_graphic_regions)} 处)"
        )
        results["visualization_annotations"] = visualization_annotations
        results["suppressed_visualization_annotations"] = (
            suppressed_visualization_annotations
        )

        # ========== 输出汇总 ==========
        print("\n" + "=" * 60)
        print("检测完成！结果汇总")
        print("=" * 60)

        match_count = sum(
            1
            for k in comparison_fields
            if text_field_values_match(k, template_data.get(k), target_data.get(k))
        )
        total_fields = len(comparison_fields)

        print(f"\n📝 文字检测:")
        print(f"   - 字段匹配: {match_count}/{total_fields}")
        print(f"   - 结果文件: {text_excel_path or '未保存'}")

        print(f"\n🖼️ 图形比较 (基于区域):")
        effective_template_regions = len(template_regions) + len(recovered_template_regions)
        effective_target_regions = len(target_regions) + len(recovered_target_regions)
        effective_matched_pairs = (
            len(matched_pairs)
            + len(recovered_template_regions)
            + len(recovered_target_regions)
        )

        print(
            f"   - 检测区域数: 模板 {len(template_regions)} / 实拍 {len(target_regions)}"
            f" (恢复后 {effective_template_regions} / {effective_target_regions})"
        )
        if skipped_template_regions or skipped_target_regions:
            print(
                f"   - 条码跳过: 模板 {len(skipped_template_regions)} / "
                f"实拍 {len(skipped_target_regions)}"
            )
        if skipped_split_template_regions or skipped_split_target_regions:
            print(
                f"   - 拆分后过滤: 模板 {len(skipped_split_template_regions)} / "
                f"实拍 {len(skipped_split_target_regions)}"
            )
        if recovered_template_regions or recovered_target_regions:
            print(
                f"   - 漏检恢复: 模板 +{len(recovered_template_regions)} / "
                f"实拍 +{len(recovered_target_regions)}"
            )
        print(f"   - 直接匹配: {len(matched_pairs)} 对")
        if recovered_template_regions or recovered_target_regions:
            print(
                f"   - 恢复补配: "
                f"{len(recovered_template_regions) + len(recovered_target_regions)} 对"
            )
        print(f"   - 已解决配对: {effective_matched_pairs} 对")

        confirmed_match = [
            r
            for r in comparison_results
            if r.get("decision") == "match"
        ]
        confirmed_mismatch = [
            r
            for r in comparison_results
            if r.get("decision") == "mismatch"
            and not r.get("visualization_suppressed")
        ]
        review_needed = [
            r
            for r in comparison_results
            if r.get("decision") == "unknown"
            and not r.get("visualization_suppressed")
        ]

        graphic_pass = True
        has_review = len(review_needed) > 0

        if effective_template_regions != effective_target_regions:
            print("   - ⚠️ 区域数量不一致!")
            graphic_pass = False

        if remaining_unmatched1 or remaining_unmatched2:
            print("   - ⚠️ 存在未匹配区域!")
            graphic_pass = False

        if confirmed_mismatch:
            graphic_pass = False
            for res in confirmed_mismatch:
                print(f"   - ❌ 确认不匹配: {res.get('summary')}")

        if review_needed:
            for res in review_needed:
                print(f"   - ⚠️ 需复核: {res.get('summary')}")
        if suppressed_graphic_regions:
            print(f"   - 局部差异过滤: {len(suppressed_graphic_regions)} 个图形框")

        if (
            confirmed_match
            and not confirmed_mismatch
            and not review_needed
            and effective_matched_pairs > 0
        ):
            print("   - ✅ 所有匹配区域均判定一致")
        elif effective_matched_pairs == 0:
            print("   - ⚠️ 无可见图形区域参与对比")

        print(
            f"   - 统计: 一致 {len(confirmed_match)}, 不一致 {len(confirmed_mismatch)}, "
            f"待复核 {len(review_needed)}"
        )
        if remaining_unmatched1 or remaining_unmatched2:
            print(
                f"   - 未恢复区域: 模板 {len(remaining_unmatched1)} / "
                f"实拍 {len(remaining_unmatched2)}"
            )
        print(f"   - 结果目录: {output_root}")
        print(f"   - 可视化图: {vis_path}")

        unresolved_graphics = len(remaining_unmatched1) + len(remaining_unmatched2)
        verdict = build_final_verdict(
            match_count=match_count,
            total_fields=total_fields,
            graphic_pass=graphic_pass,
            review_count=len(review_needed),
            mismatch_count=len(confirmed_mismatch),
            unresolved_graphics=unresolved_graphics,
        )

        print(f"\n🏷️ 综合判定: {verdict}")

        results["verdict"] = verdict
        results["output_dir"] = str(output_root)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        final_json_path = output_root / "final_result.json"
        with final_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description="整合检测脚本 (VLM 模式)")
    parser.add_argument(
        "--template",
        "--pdf",
        dest="template",
        default=DEFAULT_PDF_PATH,
        help="模板文件路径，支持 PDF 或图片",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_PATH, help="实拍图片路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument(
        "--output-mode",
        default="debug",
        choices=["final", "debug"],
        help="输出模式: final 仅保留最终结果, debug 保留详细中间产物",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("运行模式: VLM")
    result = run_unified_detection(
        args.template,
        args.target,
        output_dir=args.output_dir,
        output_mode=args.output_mode,
    )
    if not result.get("success"):
        print(f"检测失败: {result.get('error', '未知错误')}")
        return 1

    print("\n\n" + "=" * 60)
    print("输出文件列表:")
    print("=" * 60)
    output_dir = result["output_dir"]
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")
    return 0
