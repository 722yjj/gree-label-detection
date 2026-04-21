"""CLI for graphic-region comparison experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from label_detection.core.config import (
    LAYOUT_DETECTION_THRESHOLD,
    MATCH_COST_THRESHOLD,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    PROJECT_ROOT as REPO_ROOT,
    VLM_CANVAS_SIZE,
    VLM_MAX_RETRIES,
    VLM_NUM_PREDICT,
    VLM_TIMEOUT,
)
from label_detection.extraction.template_source import resolve_template_input
from label_detection.matching.layout import (
    clip_box_to_image,
    detect_barcode_region,
    detect_layout_regions,
    draw_regions,
    extract_regions_by_type,
    match_regions,
    split_barcode_regions,
    split_composite_image_regions,
)
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template
from label_detection.services.ocr_service import get_ocr_with_boxes
from label_detection.services.vlm_service import get_vlm_comparator


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "custom_graphic_method"
DEFAULT_METHOD_ROOT = Path("/home/data/WinClip")
DEFAULT_METHOD_PYTHON = DEFAULT_METHOD_ROOT / ".venv" / "bin" / "python"
DEFAULT_METHOD_SCRIPT = DEFAULT_METHOD_ROOT / "run_inference.py"

RAW_REGION_VLM_PROMPT = """你是局部图形结构一致性判定器。
你将看到左右两张局部裁剪图：
- 左图是 Template
- 右图是 Target

任务目标：
只比较深色前景图形/符号本身的结构是否一致。
请先忽略背景、纸张底色、阴影、污渍、反光、轻微噪声，再比较前景结构。
不要猜图标语义，不要描述它像什么，只判断结构是否发生真实变化。

判为 mismatch 只允许基于以下前景结构异常：
1. 前景主图形数量不同
2. Target 多出或缺少一个前景符号/组件
3. 主体连通关系不同，例如原本连通的部分断开，或原本分离的部分连在一起
4. 主体轮廓、孔洞数量、分支关系明显不同
5. 原有主体前景结构被替换成另一种结构

以下情况必须忽略，不要判为 mismatch：
- 黄底、灰底、白底、背景颜色或亮度不同
- 纸张纹理、阴影、污渍、反光
- 轻微模糊、压缩噪声、锯齿
- 线条粗细变化
- 轻微裁剪偏移、留白差异
- 小于15%的缩放差异
- 小范围边缘残差、小黑点、小缺口
- 轻微透视残差或拍摄形变
- 条码数字、编码纹理或非主体文字细节

如果主体前景结构大体一致，即使背景不同，也输出 match。
只有当你能明确指出前景结构增删、替换或拓扑变化时，才输出 mismatch。
如果图像质量不足以可靠判断前景结构，输出 unknown，不要猜测。

输出规则（必须严格遵守）：
1. 只输出一个 JSON 对象，不要输出任何其他文本。
2. JSON 必须且仅包含以下 4 个键：
   - "decision": "match" | "mismatch" | "unknown"
   - "confidence": 0.0 到 1.0 的数字
   - "differences": 字符串数组
   - "summary": 字符串
3. 当 decision 为 "match" 时，differences 必须是空数组 []。
4. summary 只描述前景结构结论，推荐使用：
   - "same_foreground_structure"
   - "foreground_structure_changed"
   - "structure_uncertain"

仅输出如下格式的 JSON：
{"decision":"match|mismatch|unknown","confidence":0.0,"differences":[],"summary":"..."}"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="图形区对比实验脚本")
    parser.set_defaults(shape_normalization=False)
    parser.add_argument("--template", required=True, help="模板文件路径，支持 PDF 或图片")
    parser.add_argument("--target", required=True, help="实拍图片路径")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认 results/custom_graphic_method",
    )
    parser.add_argument(
        "--comparison-mode",
        default="vlm_raw",
        choices=["vlm_raw", "custom_method", "both"],
        help="区域对比模式，默认使用原始裁剪图送入 VLM",
    )
    parser.add_argument(
        "--method-python",
        default=str(DEFAULT_METHOD_PYTHON),
        help="自定义方法环境里的 Python 路径",
    )
    parser.add_argument(
        "--method-script",
        default=str(DEFAULT_METHOD_SCRIPT),
        help="自定义方法脚本路径",
    )
    parser.add_argument(
        "--class-name",
        default="printed appliance label",
        help="传给自定义方法的类别文本",
    )
    parser.add_argument(
        "--fusion",
        default="visual",
        choices=["visual", "textual", "textual_visual"],
        help="传给自定义方法的融合模式",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="传给自定义方法的热力图阈值",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=120,
        help="传给自定义方法的最小异常框面积",
    )
    parser.add_argument(
        "--border-ignore",
        type=int,
        default=8,
        help="传给自定义方法的边缘忽略宽度",
    )
    parser.add_argument(
        "--disable-align",
        action="store_true",
        help="禁用自定义方法内部的模板-实拍二次对齐",
    )
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=LAYOUT_DETECTION_THRESHOLD,
        help="版面检测阈值",
    )
    parser.add_argument(
        "--match-cost-threshold",
        type=float,
        default=MATCH_COST_THRESHOLD,
        help="图形区域匹配代价阈值",
    )
    parser.add_argument(
        "--disable-region-split",
        action="store_true",
        help="禁用图形父区域的二次拆分",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=None,
        help="只跑指定匹配对索引（默认跑全部）",
    )
    parser.add_argument(
        "--enable-shape-normalization",
        dest="shape_normalization",
        action="store_true",
        help="启用图标前景二值化/形状归一化预处理（默认关闭）",
    )
    parser.add_argument(
        "--disable-shape-normalization",
        dest="shape_normalization",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shape-canvas-size",
        type=int,
        default=256,
        help="形状归一化画布尺寸",
    )
    parser.add_argument(
        "--shape-margin",
        type=int,
        default=20,
        help="形状归一化时前景到画布边界的留白像素",
    )
    parser.add_argument(
        "--vlm-model",
        default=OLLAMA_MODEL,
        help="VLM 模型名称，默认读取项目配置",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=OLLAMA_API_BASE,
        help="VLM API 地址，默认读取项目配置",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=int,
        default=VLM_TIMEOUT,
        help="VLM 请求超时秒数",
    )
    parser.add_argument(
        "--vlm-panel-size",
        type=int,
        default=VLM_CANVAS_SIZE,
        help="VLM 对比画布单侧面板尺寸",
    )
    parser.add_argument(
        "--vlm-gap",
        type=int,
        default=24,
        help="VLM 对比画布左右面板间距",
    )
    parser.add_argument(
        "--vlm-prompt-file",
        default=None,
        help="自定义 VLM 提示词文件路径（可选）",
    )
    return parser


def ensure_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} 不存在: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} 不是文件: {candidate}")
    return candidate


def box_metrics(box: Sequence[float]) -> Dict[str, float]:
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    width = max(1.0, width)
    height = max(1.0, height)
    return {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / height,
    }


def box_key(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box)


def union_box(box1: Sequence[float], box2: Sequence[float]) -> List[int]:
    return [
        int(min(float(box1[0]), float(box2[0]))),
        int(min(float(box1[1]), float(box2[1]))),
        int(max(float(box1[2]), float(box2[2]))),
        int(max(float(box1[3]), float(box2[3]))),
    ]


def horizontal_overlap_ratio(box1: Sequence[float], box2: Sequence[float]) -> float:
    overlap = min(float(box1[2]), float(box2[2])) - max(float(box1[0]), float(box2[0]))
    if overlap <= 0:
        return 0.0
    width1 = max(1.0, float(box1[2]) - float(box1[0]))
    width2 = max(1.0, float(box2[2]) - float(box2[0]))
    return float(overlap / min(width1, width2))


def vertical_gap(box1: Sequence[float], box2: Sequence[float]) -> float:
    if float(box1[1]) > float(box2[3]):
        return float(box1[1]) - float(box2[3])
    if float(box2[1]) > float(box1[3]):
        return float(box2[1]) - float(box1[3])
    return 0.0


def expand_box(box: Sequence[float], img_shape: Tuple[int, int], pad_ratio: float = 0.06, min_pad: int = 6) -> List[int]:
    metrics = box_metrics(box)
    pad_x = max(min_pad, int(round(metrics["width"] * pad_ratio)))
    pad_y = max(min_pad, int(round(metrics["height"] * pad_ratio)))
    expanded = [
        float(box[0]) - pad_x,
        float(box[1]) - pad_y,
        float(box[2]) + pad_x,
        float(box[3]) + pad_y,
    ]
    clipped = clip_box_to_image(expanded, img_shape, min_size=16)
    if clipped is None:
        raise RuntimeError(f"区域裁剪失败: {box}")
    return clipped


def crop_region(image: "cv2.typing.MatLike", box: Sequence[float]) -> Tuple["cv2.typing.MatLike", List[int]]:
    clipped = expand_box(box, image.shape[:2])
    x1, y1, x2, y2 = clipped
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"区域裁剪为空: {box}")
    return crop, clipped


def prepare_images(template_path: Path, target_path: Path, output_dir: Path) -> Dict[str, object]:
    template_raw_path, template_source_type, err = resolve_template_input(
        str(template_path),
        output_dir=str(output_dir / "template_assets"),
    )
    if err:
        raise RuntimeError(err)

    template_image, _, template_err = preprocess_template(
        template_raw_path,
        output_dir=str(output_dir / "template_preprocess"),
    )
    if template_err or template_image is None:
        raise RuntimeError(template_err or "模板预处理失败")

    target_image, _, target_err = preprocess_target(
        str(target_path),
        output_dir=str(output_dir / "target_preprocess"),
        template_image=template_image,
    )
    if target_err or target_image is None:
        raise RuntimeError(target_err or "实拍图预处理失败")

    template_prepared_path = output_dir / "template_prepared.jpg"
    target_prepared_path = output_dir / "target_prepared.jpg"
    cv2.imwrite(str(template_prepared_path), template_image)
    cv2.imwrite(str(target_prepared_path), target_image)

    return {
        "template_source_type": template_source_type,
        "template_raw_path": str(Path(template_raw_path).resolve()),
        "template_prepared_path": str(template_prepared_path.resolve()),
        "target_prepared_path": str(target_prepared_path.resolve()),
        "template_shape": list(template_image.shape),
        "target_shape": list(target_image.shape),
    }


def is_small_or_sliver_region(region: Dict) -> str | None:
    metrics = box_metrics(region["coordinate"])
    if metrics["area"] < 2500:
        return "small_area"
    if metrics["width"] < 42 or metrics["height"] < 42:
        return "small_side"
    if metrics["aspect_ratio"] >= 2.8 and metrics["height"] < 60:
        return "thin_sliver"
    return None


def looks_like_barcode_cluster(
    parent_box: Sequence[float],
    children: Sequence[Dict],
    image: "cv2.typing.MatLike",
    ocr_boxes: Sequence[Tuple[object, str, float]] | None,
) -> bool:
    if len(children) < 4:
        return False

    parent_region = {"coordinate": list(parent_box)}
    _, barcode_meta = detect_barcode_region(parent_region, image, ocr_boxes)
    parent_metrics = box_metrics(parent_box)
    if parent_metrics["aspect_ratio"] < 2.8:
        return False

    child_metrics = [box_metrics(region["coordinate"]) for region in children]
    narrow_children = sum(1 for item in child_metrics if item["aspect_ratio"] < 0.7)
    tall_children = sum(
        1 for item in child_metrics
        if item["height"] >= parent_metrics["height"] * 0.65
    )
    parent_texture_barcode = (
        float(barcode_meta["transition_density"]) >= 0.18
        and float(barcode_meta["vertical_bias"]) >= 3.0
        and 0.08 <= float(barcode_meta["dark_column_ratio"]) <= 0.85
    )
    center_x_ratio = ((float(parent_box[0]) + float(parent_box[2])) / 2.0) / max(1.0, image.shape[1])
    right_side_barcode = (
        len(children) >= 5
        and center_x_ratio >= 0.62
        and narrow_children / max(1, len(child_metrics)) >= 0.75
        and parent_metrics["aspect_ratio"] >= 3.0
    )
    return (
        (parent_texture_barcode or right_side_barcode)
        and narrow_children / max(1, len(child_metrics)) >= 0.6
        and tall_children / max(1, len(child_metrics)) >= 0.6
    )


def filter_graphic_regions(
    regions: Sequence[Dict],
    image: "cv2.typing.MatLike",
    ocr_boxes: Sequence[Tuple[object, str, float]] | None,
) -> Tuple[List[Dict], List[Dict]]:
    grouped_children: Dict[Tuple[int, int, int, int], List[Dict]] = defaultdict(list)
    for region in regions:
        parent_box = region.get("split_parent_coordinate")
        if parent_box is not None:
            grouped_children[box_key(parent_box)].append(region)

    barcode_like_parents = {
        parent_key
        for parent_key, children in grouped_children.items()
        if looks_like_barcode_cluster(
            children[0]["split_parent_coordinate"],
            children,
            image,
            ocr_boxes,
        )
    }

    filtered: List[Dict] = []
    skipped: List[Dict] = []

    for region in regions:
        region_copy = dict(region)
        parent_box = region_copy.get("split_parent_coordinate")
        if parent_box is not None and box_key(parent_box) in barcode_like_parents:
            region_copy["skip_reason"] = "barcode_cluster"
            skipped.append(region_copy)
            continue

        skip_reason = is_small_or_sliver_region(region_copy)
        if skip_reason:
            region_copy["skip_reason"] = skip_reason
            skipped.append(region_copy)
            continue

        filtered.append(region_copy)

    return filtered, skipped


def merge_fragmented_sibling_regions(regions: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[int, int, int, int], List[Dict]] = defaultdict(list)
    passthrough: List[Dict] = []
    for region in regions:
        parent_box = region.get("split_parent_coordinate")
        if parent_box is None:
            passthrough.append(dict(region))
            continue
        grouped[box_key(parent_box)].append(dict(region))

    merged_regions: List[Dict] = list(passthrough)

    for siblings in grouped.values():
        pending = sorted(
            siblings,
            key=lambda item: (item["coordinate"][0], item["coordinate"][1]),
        )
        changed = True
        while changed:
            changed = False
            next_pending: List[Dict] = []
            used = [False] * len(pending)
            for idx, region in enumerate(pending):
                if used[idx]:
                    continue

                current = dict(region)
                current_indices = set(current.get("merged_child_indices") or [current.get("split_child_idx")])
                used[idx] = True

                for other_idx in range(idx + 1, len(pending)):
                    if used[other_idx]:
                        continue
                    candidate = pending[other_idx]
                    if current.get("split_parent_coordinate") != candidate.get("split_parent_coordinate"):
                        continue

                    parent_metrics = box_metrics(current["split_parent_coordinate"])
                    gap = vertical_gap(current["coordinate"], candidate["coordinate"])
                    overlap = horizontal_overlap_ratio(current["coordinate"], candidate["coordinate"])
                    current_metrics = box_metrics(current["coordinate"])
                    candidate_metrics = box_metrics(candidate["coordinate"])
                    min_area = min(current_metrics["area"], candidate_metrics["area"])
                    max_area = max(current_metrics["area"], candidate_metrics["area"])
                    mergeable = (
                        overlap >= 0.60
                        and gap <= max(14.0, parent_metrics["height"] * 0.14)
                        and min_area / max(1.0, max_area) <= 0.70
                    )
                    if not mergeable:
                        continue

                    current["coordinate"] = union_box(current["coordinate"], candidate["coordinate"])
                    current_indices.update(
                        candidate.get("merged_child_indices") or [candidate.get("split_child_idx")]
                    )
                    current["merged_child_indices"] = sorted(
                        int(item) for item in current_indices if item is not None
                    )
                    used[other_idx] = True
                    changed = True

                next_pending.append(current)
            pending = sorted(next_pending, key=lambda item: (item["coordinate"][0], item["coordinate"][1]))

        merged_regions.extend(pending)

    return merged_regions


def build_foreground_mask(image: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    image_area = mask.shape[0] * mask.shape[1]
    min_component_area = max(12, int(round(image_area * 0.002)))
    filtered = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        filtered[labels == idx] = 255

    return filtered if np.count_nonzero(filtered) > 0 else mask


def normalize_region_shape(
    image: "cv2.typing.MatLike",
    canvas_size: int,
    margin: int,
) -> "cv2.typing.MatLike":
    mask = build_foreground_mask(image)
    ys, xs = np.where(mask > 0)
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    if len(xs) == 0 or len(ys) == 0:
        return canvas

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    fg = mask[y1:y2, x1:x2]
    fg_h, fg_w = fg.shape[:2]
    usable = max(8, canvas_size - 2 * margin)
    scale = min(usable / max(1, fg_w), usable / max(1, fg_h))
    new_w = max(1, int(round(fg_w * scale)))
    new_h = max(1, int(round(fg_h * scale)))
    resized = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = 255 - resized
    return canvas


def load_vlm_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return RAW_REGION_VLM_PROMPT
    return prompt_file.read_text(encoding="utf-8")


def fit_image_to_panel(
    image: "cv2.typing.MatLike",
    panel_size: int,
    background: int = 255,
) -> "cv2.typing.MatLike":
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    scale = min(panel_size / max(1, width), panel_size / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((panel_size, panel_size, 3), background, dtype=np.uint8)
    x_offset = (panel_size - new_w) // 2
    y_offset = (panel_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    cv2.rectangle(canvas, (1, 1), (panel_size - 2, panel_size - 2), (210, 210, 210), 2)
    return canvas


def create_raw_vlm_canvas(
    template_crop: "cv2.typing.MatLike",
    target_crop: "cv2.typing.MatLike",
    panel_size: int,
    gap: int,
) -> "cv2.typing.MatLike":
    header_height = 48
    canvas_h = header_height + panel_size
    canvas_w = panel_size * 2 + gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    left_panel = fit_image_to_panel(template_crop, panel_size=panel_size)
    right_panel = fit_image_to_panel(target_crop, panel_size=panel_size)

    canvas[header_height:, :panel_size] = left_panel
    canvas[header_height:, panel_size + gap : panel_size * 2 + gap] = right_panel

    separator_x = panel_size + gap // 2
    cv2.line(canvas, (separator_x, 0), (separator_x, canvas_h), (185, 185, 185), 2)
    cv2.putText(
        canvas,
        "Template",
        (max(12, panel_size // 2 - 60), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        canvas,
        "Target",
        (panel_size + gap + max(12, panel_size // 2 - 45), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
    )
    return canvas


def compare_canvas_with_vlm(
    comparator,
    canvas: "cv2.typing.MatLike",
    prompt: str,
) -> Dict[str, object]:
    canvas_b64 = comparator._image_to_base64(canvas)
    last_error = None
    session = requests.Session()
    session.trust_env = False

    for attempt in range(VLM_MAX_RETRIES):
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [canvas_b64],
                }
            ]
            response = session.post(
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


def is_background_only_vlm_mismatch(vlm_result: Dict[str, object]) -> bool:
    summary = str(vlm_result.get("summary", "")).lower()
    differences = [str(item).lower() for item in vlm_result.get("differences", [])]
    text = " ".join([summary] + differences)

    background_keywords = [
        "背景",
        "background",
        "灰色",
        "白色",
        "yellow",
        "gray",
        "grey",
        "white",
        "底色",
        "明暗",
        "亮度",
        "阴影",
        "shadow",
    ]
    same_structure_phrases = [
        "主体图形一致",
        "图形一致",
        "前景图形一致",
        "结构一致",
        "轮廓一致",
        "same graphic",
        "same foreground",
        "same structure",
        "same_foreground_structure",
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
        "孔洞",
        "连通",
        "拓扑",
        "extra",
        "additional",
        "missing",
        "replace",
        "replaced",
        "different structure",
        "different graphic",
        "hole",
        "topology",
        "component",
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


def run_vlm_raw_on_pair(
    comparator,
    prompt: str,
    template_crop: "cv2.typing.MatLike",
    target_crop: "cv2.typing.MatLike",
    pair_output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    canvas = create_raw_vlm_canvas(
        template_crop=template_crop,
        target_crop=target_crop,
        panel_size=args.vlm_panel_size,
        gap=args.vlm_gap,
    )
    canvas_path = pair_output_dir / "vlm_raw_canvas.jpg"
    prompt_path = pair_output_dir / "vlm_raw_prompt.txt"
    result_json_path = pair_output_dir / "vlm_raw_result.json"

    cv2.imwrite(str(canvas_path), canvas)
    prompt_path.write_text(prompt, encoding="utf-8")

    result = compare_canvas_with_vlm(
        comparator=comparator,
        canvas=canvas,
        prompt=prompt,
    )

    if result.get("decision") == "mismatch" and is_background_only_vlm_mismatch(result):
        result = dict(result)
        result["decision"] = "match"
        result["is_match"] = True
        result["needs_review"] = False
        result["differences"] = []
        result["summary"] = "same_foreground_structure"
        result["judgment_source"] = "background_override"
    else:
        result = dict(result)
        result.setdefault("judgment_source", "vlm_raw")

    result["canvas_path"] = str(canvas_path.resolve())
    result["prompt_path"] = str(prompt_path.resolve())
    result["panel_size"] = int(args.vlm_panel_size)
    result["gap"] = int(args.vlm_gap)

    with open(result_json_path, "w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)

    return {
        "success": result.get("error_type") is None,
        "result_json_path": str(result_json_path.resolve()),
        "canvas_path": str(canvas_path.resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "final_result": result,
    }


def resolve_backend_decision(backend_name: str, backend_result: Dict[str, object] | None) -> Tuple[str, bool]:
    if not backend_result:
        return "error", False

    if backend_name == "custom_method":
        if not backend_result.get("success"):
            return "error", False
        method_result = ((backend_result.get("final_result") or {}).get("method_result") or {})
        is_different = method_result.get("is_different")
        if is_different is True:
            return "mismatch", True
        if is_different is False:
            return "match", True
        return "unknown", True

    if backend_name == "vlm_raw":
        if not backend_result.get("success"):
            return "error", False
        final_result = backend_result.get("final_result") or {}
        decision = str(final_result.get("decision", "unknown"))
        if decision in {"match", "mismatch", "unknown"}:
            return decision, True
        return "error", False

    return "error", False


def save_region_debug_image(
    image: "cv2.typing.MatLike",
    kept_regions: Sequence[Dict],
    skipped_regions: Sequence[Dict],
    output_path: Path,
) -> None:
    preview = image.copy()
    if skipped_regions:
        preview = draw_regions(preview, list(skipped_regions), color=(0, 165, 255))
    if kept_regions:
        preview = draw_regions(preview, list(kept_regions), color=(0, 255, 0))
    cv2.imwrite(str(output_path), preview)


def extract_graphic_regions(
    prepared_image_path: Path,
    image: "cv2.typing.MatLike",
    output_dir: Path,
    image_label: str,
    threshold: float,
    enable_region_split: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _, ocr_boxes = get_ocr_with_boxes(str(prepared_image_path))
    all_regions = detect_layout_regions(str(prepared_image_path), threshold=threshold)
    image_regions = extract_regions_by_type(all_regions, "image")
    comparable_regions, skipped_barcode_regions = split_barcode_regions(image_regions, image, ocr_boxes)

    split_parent_regions: List[Dict] = []
    candidate_regions = list(comparable_regions)
    if enable_region_split:
        candidate_regions, split_parent_regions = split_composite_image_regions(
            comparable_regions,
            image,
            ocr_boxes,
        )
        candidate_regions = merge_fragmented_sibling_regions(candidate_regions)

    filtered_regions, skipped_filtered_regions = filter_graphic_regions(
        candidate_regions,
        image,
        ocr_boxes,
    )

    cv2.imwrite(
        str(output_dir / f"{image_label}_regions_all.jpg"),
        draw_regions(image, image_regions),
    )
    save_region_debug_image(
        image,
        comparable_regions,
        skipped_barcode_regions,
        output_dir / f"{image_label}_regions_comparable.jpg",
    )
    save_region_debug_image(
        image,
        filtered_regions,
        skipped_filtered_regions,
        output_dir / f"{image_label}_regions_filtered.jpg",
    )

    return {
        "ocr_box_count": len(ocr_boxes),
        "all_regions": all_regions,
        "image_regions": image_regions,
        "comparable_regions": comparable_regions,
        "skipped_barcode_regions": skipped_barcode_regions,
        "split_parent_regions": split_parent_regions,
        "candidate_regions": candidate_regions,
        "filtered_regions": filtered_regions,
        "skipped_filtered_regions": skipped_filtered_regions,
    }


def build_method_command(
    method_python: Path,
    method_script: Path,
    template_path: Path,
    target_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    disable_align: bool,
) -> Tuple[List[str], Path, Path]:
    result_image_path = output_dir / "custom_method_visualization.png"
    result_json_path = output_dir / "custom_method_result.json"
    command = [
        str(method_python),
        str(method_script),
        "--template",
        str(template_path),
        "--query",
        str(target_path),
        "--output",
        str(result_image_path),
        "--result-json",
        str(result_json_path),
        "--class-name",
        args.class_name,
        "--fusion",
        args.fusion,
        "--threshold",
        str(args.threshold),
        "--min-area",
        str(args.min_area),
        "--border-ignore",
        str(args.border_ignore),
    ]
    if disable_align:
        command.append("--disable-align")
    return command, result_image_path, result_json_path


def execute_custom_method(
    command: List[str],
    method_script: Path,
    result_image_path: Path,
    result_json_path: Path,
) -> Dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=str(method_script.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    method_result = None
    if result_json_path.exists():
        with open(result_json_path, "r", encoding="utf-8") as file_obj:
            method_result = json.load(file_obj)

    return {
        "success": completed.returncode == 0 and method_result is not None,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_image_path": str(result_image_path.resolve()) if result_image_path.exists() else None,
        "result_json_path": str(result_json_path.resolve()) if result_json_path.exists() else None,
        "method_result": method_result,
    }


def run_custom_method_on_pair(
    method_python: Path,
    method_script: Path,
    template_crop_path: Path,
    target_crop_path: Path,
    pair_output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    attempts: List[Dict[str, object]] = []
    align_modes = [bool(args.disable_align)]
    if not args.disable_align:
        align_modes.append(True)

    for disable_align in align_modes:
        command, result_image_path, result_json_path = build_method_command(
            method_python=method_python,
            method_script=method_script,
            template_path=template_crop_path,
            target_path=target_crop_path,
            output_dir=pair_output_dir,
            args=args,
            disable_align=disable_align,
        )
        attempt = execute_custom_method(
            command=command,
            method_script=method_script,
            result_image_path=result_image_path,
            result_json_path=result_json_path,
        )
        attempt["disable_align"] = disable_align
        attempts.append(attempt)
        if attempt["success"]:
            return {
                "success": True,
                "attempts": attempts,
                "used_disable_align": disable_align,
                "final_result": attempt,
            }

    return {
        "success": False,
        "attempts": attempts,
        "used_disable_align": attempts[-1]["disable_align"] if attempts else None,
        "final_result": attempts[-1] if attempts else None,
    }


def save_region_inputs(
    template_crop: "cv2.typing.MatLike",
    target_crop: "cv2.typing.MatLike",
    pair_output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path, Dict[str, object]]:
    template_raw_path = pair_output_dir / "template_crop_raw.jpg"
    target_raw_path = pair_output_dir / "target_crop_raw.jpg"
    cv2.imwrite(str(template_raw_path), template_crop)
    cv2.imwrite(str(target_raw_path), target_crop)

    payload: Dict[str, object] = {
        "shape_normalization_enabled": bool(args.shape_normalization),
        "template_raw_path": str(template_raw_path.resolve()),
        "target_raw_path": str(target_raw_path.resolve()),
    }

    if not args.shape_normalization:
        return template_raw_path, target_raw_path, payload

    template_preprocessed_path = pair_output_dir / "template_crop_preprocessed.png"
    target_preprocessed_path = pair_output_dir / "target_crop_preprocessed.png"
    template_normalized = normalize_region_shape(
        template_crop,
        canvas_size=args.shape_canvas_size,
        margin=args.shape_margin,
    )
    target_normalized = normalize_region_shape(
        target_crop,
        canvas_size=args.shape_canvas_size,
        margin=args.shape_margin,
    )
    cv2.imwrite(str(template_preprocessed_path), template_normalized)
    cv2.imwrite(str(target_preprocessed_path), target_normalized)
    payload["template_preprocessed_path"] = str(template_preprocessed_path.resolve())
    payload["target_preprocessed_path"] = str(target_preprocessed_path.resolve())
    payload["shape_canvas_size"] = args.shape_canvas_size
    payload["shape_margin"] = args.shape_margin
    return template_preprocessed_path, target_preprocessed_path, payload


def annotate_target_decisions(
    target_image: "cv2.typing.MatLike",
    target_regions: Sequence[Dict],
    pair_results: Sequence[Dict],
    unmatched_target: Sequence[int],
    output_path: Path,
) -> None:
    canvas = target_image.copy()

    for region in target_regions:
        x1, y1, x2, y2 = [int(v) for v in region["coordinate"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 2)

    for idx in unmatched_target:
        if idx >= len(target_regions):
            continue
        x1, y1, x2, y2 = [int(v) for v in target_regions[idx]["coordinate"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 3)
        cv2.putText(canvas, "Unmatched", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    for pair in pair_results:
        target_idx = pair["target_idx"]
        if target_idx >= len(target_regions):
            continue
        x1, y1, x2, y2 = [int(v) for v in target_regions[target_idx]["coordinate"]]
        decision = str(pair.get("decision", "error"))
        if decision == "mismatch":
            color = (0, 0, 255)
            tag = f"Diff#{pair['pair_idx']}"
        elif decision == "match":
            color = (0, 180, 0)
            tag = f"Same#{pair['pair_idx']}"
        elif decision == "unknown":
            color = (0, 165, 255)
            tag = f"Unk#{pair['pair_idx']}"
        else:
            color = (0, 200, 255)
            tag = f"Err#{pair['pair_idx']}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.putText(canvas, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), canvas)


def write_summary(summary_path: Path, payload: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    template_path = ensure_file(Path(args.template), "模板文件")
    target_path = ensure_file(Path(args.target), "实拍图片")
    method_python = None
    method_script = None
    if args.comparison_mode in {"custom_method", "both"}:
        method_python = ensure_file(Path(args.method_python), "方法 Python")
        method_script = ensure_file(Path(args.method_script), "方法脚本")

    prompt_file = None
    if args.vlm_prompt_file:
        prompt_file = ensure_file(Path(args.vlm_prompt_file), "VLM 提示词文件")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    summary: Dict[str, object] = {
        "success": False,
        "template_path": str(template_path),
        "target_path": str(target_path),
        "output_dir": str(output_dir),
        "comparison_mode": args.comparison_mode,
        "method_python": str(method_python) if method_python else None,
        "method_script": str(method_script) if method_script else None,
        "vlm_model": args.vlm_model,
        "vlm_api_base": args.vlm_api_base,
        "shape_normalization_enabled": bool(args.shape_normalization),
        "prepared": None,
        "template_graphic_regions": None,
        "target_graphic_regions": None,
        "matched_pairs": [],
        "unmatched_template": [],
        "unmatched_target": [],
        "pair_results": [],
    }

    try:
        vlm_prompt = None
        comparator = None
        if args.comparison_mode in {"vlm_raw", "both"}:
            vlm_prompt = load_vlm_prompt(prompt_file)
            comparator = get_vlm_comparator(
                model_name=args.vlm_model,
                api_base=args.vlm_api_base,
            )
            comparator.timeout = args.vlm_timeout
            summary["vlm_prompt"] = vlm_prompt

        prepared = prepare_images(template_path, target_path, output_dir)
        summary["prepared"] = prepared

        template_prepared_path = Path(prepared["template_prepared_path"])
        target_prepared_path = Path(prepared["target_prepared_path"])
        template_image = cv2.imread(str(template_prepared_path))
        target_image = cv2.imread(str(target_prepared_path))
        if template_image is None or target_image is None:
            raise RuntimeError("无法读取预处理后的模板图或实拍图")

        region_debug_dir = output_dir / "graphic_regions"
        template_regions_payload = extract_graphic_regions(
            prepared_image_path=template_prepared_path,
            image=template_image,
            output_dir=region_debug_dir,
            image_label="template",
            threshold=args.layout_threshold,
            enable_region_split=not args.disable_region_split,
        )
        target_regions_payload = extract_graphic_regions(
            prepared_image_path=target_prepared_path,
            image=target_image,
            output_dir=region_debug_dir,
            image_label="target",
            threshold=args.layout_threshold,
            enable_region_split=not args.disable_region_split,
        )

        template_regions = list(template_regions_payload["filtered_regions"])
        target_regions = list(target_regions_payload["filtered_regions"])
        summary["template_graphic_regions"] = template_regions_payload
        summary["target_graphic_regions"] = target_regions_payload

        matched_pairs, unmatched1, unmatched2 = match_regions(
            template_regions,
            target_regions,
            template_image.shape[:2],
            target_image.shape[:2],
            cost_threshold=args.match_cost_threshold,
        )
        summary["matched_pairs"] = matched_pairs
        summary["unmatched_template"] = unmatched1
        summary["unmatched_target"] = unmatched2

        selected_pairs = matched_pairs
        if args.pair_index is not None:
            if args.pair_index < 0 or args.pair_index >= len(matched_pairs):
                raise IndexError(f"pair_index 越界: {args.pair_index}, 总匹配对数 {len(matched_pairs)}")
            selected_pairs = [matched_pairs[args.pair_index]]

        pair_results: List[Dict[str, object]] = []
        for pair_idx, (template_idx, target_idx, match_cost) in enumerate(selected_pairs):
            template_region = template_regions[template_idx]
            target_region = target_regions[target_idx]
            pair_output_dir = output_dir / f"pair_{pair_idx:02d}_t{template_idx}_s{target_idx}"
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            template_crop, template_crop_box = crop_region(template_image, template_region["coordinate"])
            target_crop, target_crop_box = crop_region(target_image, target_region["coordinate"])
            method_template_path, method_target_path, input_payload = save_region_inputs(
                template_crop=template_crop,
                target_crop=target_crop,
                pair_output_dir=pair_output_dir,
                args=args,
            )

            custom_method_result = None
            if args.comparison_mode in {"custom_method", "both"}:
                custom_method_result = run_custom_method_on_pair(
                    method_python=method_python,
                    method_script=method_script,
                    template_crop_path=method_template_path,
                    target_crop_path=method_target_path,
                    pair_output_dir=pair_output_dir,
                    args=args,
                )

            vlm_raw_result = None
            if args.comparison_mode in {"vlm_raw", "both"}:
                vlm_raw_result = run_vlm_raw_on_pair(
                    comparator=comparator,
                    prompt=vlm_prompt,
                    template_crop=template_crop,
                    target_crop=target_crop,
                    pair_output_dir=pair_output_dir,
                    args=args,
                )

            primary_backend = "vlm_raw" if vlm_raw_result is not None else "custom_method"
            decision, success = resolve_backend_decision(
                primary_backend,
                vlm_raw_result if primary_backend == "vlm_raw" else custom_method_result,
            )

            pair_results.append(
                {
                    "pair_idx": pair_idx,
                    "template_idx": int(template_idx),
                    "target_idx": int(target_idx),
                    "match_cost": float(match_cost),
                    "template_region": template_region,
                    "target_region": target_region,
                    "template_crop_box": template_crop_box,
                    "target_crop_box": target_crop_box,
                    "region_inputs": input_payload,
                    "primary_backend": primary_backend,
                    "decision": decision,
                    "success": success,
                    "custom_method": custom_method_result,
                    "vlm_raw": vlm_raw_result,
                }
            )

        summary["pair_results"] = pair_results

        decision_vis_path = output_dir / "target_graphic_region_decisions.jpg"
        annotate_target_decisions(
            target_image=target_image,
            target_regions=target_regions,
            pair_results=pair_results,
            unmatched_target=unmatched2,
            output_path=decision_vis_path,
        )

        successful_pairs = [
            pair for pair in pair_results
            if pair.get("success")
        ]
        different_pairs = [pair for pair in pair_results if pair.get("decision") == "mismatch"]
        same_pairs = [pair for pair in pair_results if pair.get("decision") == "match"]
        unknown_pairs = [pair for pair in pair_results if pair.get("decision") == "unknown"]
        failed_pairs = [
            pair for pair in pair_results
            if pair.get("decision") == "error"
        ]

        overall_decision = "match"
        if unmatched1 or unmatched2 or different_pairs:
            overall_decision = "mismatch"
        elif unknown_pairs or failed_pairs:
            overall_decision = "unknown"

        summary["overall"] = {
            "comparison_mode": args.comparison_mode,
            "template_graphic_region_count": len(template_regions),
            "target_graphic_region_count": len(target_regions),
            "matched_pair_count": len(matched_pairs),
            "processed_pair_count": len(pair_results),
            "successful_pair_count": len(successful_pairs),
            "same_pair_count": len(same_pairs),
            "different_pair_count": len(different_pairs),
            "unknown_pair_count": len(unknown_pairs),
            "failed_pair_count": len(failed_pairs),
            "has_unmatched_regions": bool(unmatched1 or unmatched2),
            "overall_decision": overall_decision,
            "is_different": bool(unmatched1 or unmatched2 or different_pairs),
            "decision_visualization": str(decision_vis_path.resolve()),
        }
        summary["success"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        write_summary(summary_path, summary)
        print(f"实验失败: {exc}")
        print(f"summary: {summary_path}")
        return 1

    write_summary(summary_path, summary)

    overall = summary["overall"]
    print("=" * 60)
    print("图形区对比实验完成")
    print("=" * 60)
    print(f"对比模式: {overall['comparison_mode']}")
    print(f"模板图形区: {overall['template_graphic_region_count']}")
    print(f"实拍图形区: {overall['target_graphic_region_count']}")
    print(f"匹配对数: {overall['matched_pair_count']}")
    print(f"成功对比: {overall['successful_pair_count']}")
    print(f"一致对数: {overall['same_pair_count']}")
    print(f"差异对数: {overall['different_pair_count']}")
    print(f"未知对数: {overall['unknown_pair_count']}")
    print(f"失败对数: {overall['failed_pair_count']}")
    print(f"存在未匹配图形区: {overall['has_unmatched_regions']}")
    print(f"整体结构判定: {overall['overall_decision']}")
    print(f"整体图形差异判定: {overall['is_different']}")
    print(f"判定可视化: {overall['decision_visualization']}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
