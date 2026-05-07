"""Experiment: ask a multimodal LLM to localize differences on aligned labels."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import (  # noqa: E402
    OPENAI_COMPATIBLE_API_BASE,
    OPENAI_COMPATIBLE_API_KEY,
    OPENAI_COMPATIBLE_MODEL,
    PROJECT_ROOT as REPO_ROOT,
    VLM_TIMEOUT,
)
from label_detection.extraction.template_source import resolve_template_input  # noqa: E402
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template  # noqa: E402
from label_detection.services.openai_compatible_client import OpenAICompatibleHTTPClient  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "vlm_diff_localization"
Box = Tuple[int, int, int, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal LLM aligned-difference localization experiment."
    )
    parser.add_argument("--template", required=True, help="模板 PDF 或图片路径")
    parser.add_argument("--target", required=True, help="实拍目标图片路径")
    parser.add_argument(
        "--mode",
        choices=("decision", "point", "box", "both"),
        default="point",
        help="实验模式：只判断、点定位、框定位，或点+框",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="输出根目录，默认 results/vlm_diff_localization",
    )
    parser.add_argument(
        "--run-name",
        help="输出子目录名称；默认使用时间戳",
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
    parser.add_argument("--timeout", type=int, default=VLM_TIMEOUT, help="请求超时秒数")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大输出 token")
    parser.add_argument(
        "--max-differences",
        type=int,
        default=5,
        help="最多要求模型返回多少个差异",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=1024,
        help="送入模型时每侧面板最大宽度，较大更清晰但更耗 token/显存",
    )
    parser.add_argument(
        "--skip-feature-align",
        action="store_true",
        help="跳过 SIFT/ORB 单应性对齐，仅使用预处理透视矫正和缩放",
    )
    parser.add_argument(
        "--skip-ecc",
        action="store_true",
        help="跳过 ECC 微调对齐",
    )
    parser.add_argument(
        "--ecc-motion",
        choices=("translation", "euclidean", "affine"),
        default="translation",
        help="ECC 微调运动模型",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过 /models 连通和模型存在性检查",
    )
    return parser


def prepare_output_dir(root: Path, run_name: str | None) -> Path:
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def imwrite(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写入图片: {path}")
    return str(path)


def write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def resolve_and_preprocess(
    template_path: str,
    target_path: str,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    source_dir = output_dir / "template_source"
    preprocess_dir = output_dir / "preprocess"
    source_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    resolved_template, source_type, error = resolve_template_input(
        template_path,
        output_dir=str(source_dir),
    )
    if error or resolved_template is None:
        raise RuntimeError(error or "无法解析模板输入")

    template_img, _, error = preprocess_template(
        resolved_template,
        output_dir=str(preprocess_dir),
    )
    if error or template_img is None:
        raise RuntimeError(error or "模板预处理失败")

    target_img, _, error = preprocess_target(
        target_path,
        output_dir=str(preprocess_dir),
        template_image=template_img,
    )
    if error or target_img is None:
        raise RuntimeError(error or "目标图预处理失败")

    return template_img, target_img, {
        "template_input": str(template_path),
        "template_source_type": source_type,
        "resolved_template": str(resolved_template),
        "target_input": str(target_path),
        "preprocess_dir": str(preprocess_dir),
    }


def resize_to_reference(moving: np.ndarray, reference_shape: Sequence[int]) -> np.ndarray:
    height, width = int(reference_shape[0]), int(reference_shape[1])
    if moving.shape[:2] == (height, width):
        return moving.copy()
    return cv2.resize(moving, (width, height), interpolation=cv2.INTER_AREA)


def _make_feature_detector() -> Tuple[Any, str]:
    try:
        return cv2.SIFT_create(nfeatures=2500), "sift"
    except Exception:
        return cv2.ORB_create(nfeatures=2500), "orb"


def feature_align(reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    detector, detector_name = _make_feature_detector()

    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    kp_mov, des_mov = detector.detectAndCompute(gray_mov, None)
    stats: Dict[str, Any] = {
        "method": detector_name,
        "reference_keypoints": len(kp_ref),
        "moving_keypoints": len(kp_mov),
        "good_matches": 0,
        "success": False,
    }
    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        stats["reason"] = "not_enough_keypoints"
        return moving.copy(), stats

    norm = cv2.NORM_HAMMING if detector_name == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    try:
        matches = matcher.knnMatch(des_mov, des_ref, k=2)
    except cv2.error as exc:
        stats["reason"] = f"match_failed: {exc}"
        return moving.copy(), stats

    good = []
    for item in matches:
        if len(item) != 2:
            continue
        first, second = item
        if first.distance < 0.75 * second.distance:
            good.append(first)
    stats["good_matches"] = len(good)
    if len(good) < 8:
        stats["reason"] = "not_enough_good_matches"
        return moving.copy(), stats

    src = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None:
        stats["reason"] = "homography_failed"
        return moving.copy(), stats

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    stats["inliers"] = inliers
    if inliers < 8:
        stats["reason"] = "not_enough_inliers"
        return moving.copy(), stats

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(
        moving,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    stats["success"] = True
    stats["homography"] = homography.tolist()
    return aligned, stats


def ecc_align(
    reference: np.ndarray,
    moving: np.ndarray,
    motion: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    motion_type = motion_map[motion]
    warp = np.eye(2, 3, dtype=np.float32)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.equalizeHist(gray_ref).astype(np.float32) / 255.0
    gray_mov = cv2.equalizeHist(gray_mov).astype(np.float32) / 255.0
    stats: Dict[str, Any] = {"motion": motion, "success": False}
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        cc, warp = cv2.findTransformECC(
            gray_ref,
            gray_mov,
            warp,
            motion_type,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        stats["reason"] = str(exc)
        return moving.copy(), stats

    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    stats["success"] = True
    stats["correlation"] = float(cc)
    stats["warp"] = warp.tolist()
    return aligned, stats


def align_target_to_template(
    template: np.ndarray,
    target: np.ndarray,
    *,
    skip_feature_align: bool,
    skip_ecc: bool,
    ecc_motion: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    aligned = resize_to_reference(target, template.shape[:2])
    stats: Dict[str, Any] = {
        "target_original_shape": list(target.shape[:2]),
        "comparison_shape": list(template.shape[:2]),
        "initial_resize": target.shape[:2] != template.shape[:2],
    }
    if not skip_feature_align:
        aligned, feature_stats = feature_align(template, aligned)
        stats["feature"] = feature_stats
    if not skip_ecc:
        aligned, ecc_stats = ecc_align(template, aligned, ecc_motion)
        stats["ecc"] = ecc_stats
    return aligned, stats


def scale_pair_for_canvas(
    template: np.ndarray,
    target: np.ndarray,
    panel_width: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    height, width = template.shape[:2]
    scale = min(1.0, panel_width / float(max(width, 1)))
    if scale >= 0.999:
        return template.copy(), target.copy(), 1.0
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    size = (new_width, new_height)
    return (
        cv2.resize(template, size, interpolation=cv2.INTER_AREA),
        cv2.resize(target, size, interpolation=cv2.INTER_AREA),
        scale,
    )


def make_comparison_canvas(
    template: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    panel_h, panel_w = template.shape[:2]
    title_h = 44
    sep_w = 24
    canvas_h = panel_h + title_h
    canvas_w = panel_w * 2 + sep_w
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    left_box = [0, title_h, panel_w, title_h + panel_h]
    right_box = [panel_w + sep_w, title_h, panel_w + sep_w + panel_w, title_h + panel_h]
    canvas[left_box[1] : left_box[3], left_box[0] : left_box[2]] = template
    canvas[right_box[1] : right_box[3], right_box[0] : right_box[2]] = target
    cv2.line(
        canvas,
        (panel_w + sep_w // 2, 0),
        (panel_w + sep_w // 2, canvas_h),
        (180, 180, 180),
        2,
    )
    cv2.putText(canvas, "Template", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(
        canvas,
        "Target aligned",
        (right_box[0] + 16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return canvas, {
        "canvas_size": {"width": canvas_w, "height": canvas_h},
        "template_panel_px": left_box,
        "target_panel_px": right_box,
        "separator_width": sep_w,
        "title_height": title_h,
    }


def image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("对比图编码失败")
    return base64.b64encode(buffer).decode("ascii")


def build_prompt(
    mode: str,
    canvas_meta: Dict[str, Any],
    max_differences: int,
) -> str:
    size = canvas_meta["canvas_size"]
    template_panel = canvas_meta["template_panel_px"]
    target_panel = canvas_meta["target_panel_px"]
    if mode == "decision":
        detail_rule = 'Each item in "differences" may omit point_1000 and bbox_1000.'
    elif mode == "point":
        detail_rule = 'For each difference, include "point_1000": [x, y]. Omit bbox_1000.'
    elif mode == "box":
        detail_rule = 'For each difference, include "bbox_1000": [x1, y1, x2, y2]. Omit point_1000.'
    else:
        detail_rule = 'For each difference, include both "point_1000": [x, y] and "bbox_1000": [x1, y1, x2, y2].'

    return f"""You are testing visual difference localization on aligned product labels.

You will see one comparison canvas:
- Left panel is the template label.
- Right panel is the target label after preprocessing and alignment.

Canvas geometry:
- full canvas width: {size["width"]} px
- full canvas height: {size["height"]} px
- template panel pixel box: {template_panel}
- target aligned panel pixel box: {target_panel}

Task:
Find real visible content differences between the template and the aligned target.
Prefer graphic/icon/logo/mark/code differences, but include any clearly visible printed-content difference if it is visually localized.

Do NOT mark these as differences:
- remaining tiny alignment shifts
- different brightness, blur, compression, shadows, exposure, or color cast
- global scale/crop changes already handled by the alignment
- tiny edge artifacts on the outer border

Coordinate rules:
- All coordinates must be normalized 0-1000 relative to the FULL comparison canvas, not relative to one panel.
- If the visible differing evidence is on the right target panel, use side "target".
- If it is missing from target but visible on the left template panel, use side "template".
- If both panels need to be referenced, use side "both" and place the point/box on the most diagnostic visible evidence.

Output rules:
1. Output exactly one JSON object and nothing else.
2. Do not output thinking, Markdown, comments, or code fences.
3. Return at most {max_differences} highest-confidence differences.
4. Use this schema:
{{"decision":"same|different|unknown","confidence":0.0,"differences":[{{"side":"template|target|both","label":"short label","point_1000":[x,y],"bbox_1000":[x1,y1,x2,y2],"confidence":0.0}}],"summary":"short summary"}}
5. If decision is "same", differences must be [].
6. If uncertain, use decision "unknown" and differences [].
7. {detail_rule}
"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def point_to_px(point: Any, width: int, height: int) -> Tuple[Optional[List[int]], str]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None, "missing"
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None, "invalid"

    if (x > 1000.0 or y > 1000.0) and 0.0 <= x <= width and 0.0 <= y <= height:
        px = int(round(x))
        py = int(round(y))
        return [min(max(px, 0), width - 1), min(max(py, 0), height - 1)], "pixel"

    px = int(round(max(0.0, min(1000.0, x)) / 1000.0 * width))
    py = int(round(max(0.0, min(1000.0, y)) / 1000.0 * height))
    mode = "norm_1000"
    if x < 0.0 or x > 1000.0 or y < 0.0 or y > 1000.0:
        mode = "norm_1000_clamped"
    return [min(max(px, 0), width - 1), min(max(py, 0), height - 1)], mode


def box_to_px(box: Any, width: int, height: int) -> Tuple[Optional[List[int]], str]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None, "missing"
    try:
        values = [float(value) for value in box[:4]]
    except (TypeError, ValueError):
        return None, "invalid"

    x_values = values[0::2]
    y_values = values[1::2]
    if (
        max(values) > 1000.0
        and min(x_values) >= 0.0
        and min(y_values) >= 0.0
        and max(x_values) <= width
        and max(y_values) <= height
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in values]
        mode = "pixel"
    else:
        p1, mode1 = point_to_px(values[:2], width, height)
        p2, mode2 = point_to_px(values[2:4], width, height)
        x1, y1 = p1 or [0, 0]
        x2, y2 = p2 or [0, 0]
        mode = mode1 if mode1 == mode2 else f"{mode1}+{mode2}"

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left:
        right = min(width - 1, left + 1)
    if bottom <= top:
        bottom = min(height - 1, top + 1)
    return [left, top, right, bottom], mode


def normalize_result(payload: Optional[Dict[str, Any]], canvas_shape: Sequence[int]) -> Dict[str, Any]:
    height, width = int(canvas_shape[0]), int(canvas_shape[1])
    if payload is None:
        return {
            "parse_error": True,
            "decision": "unknown",
            "confidence": 0.0,
            "differences": [],
            "summary": "模型输出无法解析为 JSON",
        }

    decision = str(payload.get("decision") or "unknown").strip().lower()
    if decision not in {"same", "different", "unknown"}:
        decision = "unknown"
    differences = payload.get("differences")
    if not isinstance(differences, list):
        differences = []

    normalized: List[Dict[str, Any]] = []
    for item in differences:
        if not isinstance(item, dict):
            continue
        side = str(item.get("side") or "both").strip().lower()
        if side not in {"template", "target", "both"}:
            side = "both"
        point_px, point_coord_mode = point_to_px(item.get("point_1000"), width, height)
        bbox_px, bbox_coord_mode = box_to_px(item.get("bbox_1000"), width, height)
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "side": side,
                "label": str(item.get("label") or "difference"),
                "confidence": max(0.0, min(1.0, confidence)),
                "point_1000": item.get("point_1000"),
                "point_px": point_px,
                "point_coord_mode": point_coord_mode,
                "bbox_1000": item.get("bbox_1000"),
                "bbox_px": bbox_px,
                "bbox_coord_mode": bbox_coord_mode,
            }
        )

    if decision == "same":
        normalized = []

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "parse_error": False,
        "decision": decision,
        "confidence": max(0.0, min(1.0, confidence)),
        "differences": normalized,
        "summary": str(payload.get("summary") or ""),
    }


def draw_predictions(canvas: np.ndarray, differences: Sequence[Dict[str, Any]]) -> np.ndarray:
    output = canvas.copy()
    for idx, item in enumerate(differences, start=1):
        color = (0, 0, 255) if item.get("side") != "template" else (255, 0, 0)
        bbox = item.get("bbox_px")
        point = item.get("point_px")
        label = f"{idx}:{item.get('side', 'both')}"
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                output,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        if point:
            x, y = [int(v) for v in point]
            cv2.circle(output, (x, y), 8, color, -1)
            cv2.circle(output, (x, y), 16, color, 2)
            cv2.line(output, (x - 22, y), (x + 22, y), color, 2)
            cv2.line(output, (x, y - 22), (x, y + 22), color, 2)
            cv2.putText(
                output,
                label,
                (x + 12, max(24, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    return output


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve(), args.run_name)

    try:
        template_img, target_img, summary = resolve_and_preprocess(
            args.template,
            args.target,
            output_dir,
        )
        imwrite(output_dir / "template_preprocessed.jpg", template_img)
        imwrite(output_dir / "target_preprocessed.jpg", target_img)

        aligned_target, alignment = align_target_to_template(
            template_img,
            target_img,
            skip_feature_align=args.skip_feature_align,
            skip_ecc=args.skip_ecc,
            ecc_motion=args.ecc_motion,
        )
        imwrite(output_dir / "target_aligned.jpg", aligned_target)

        template_panel, target_panel, canvas_scale = scale_pair_for_canvas(
            template_img,
            aligned_target,
            args.panel_width,
        )
        canvas, canvas_meta = make_comparison_canvas(template_panel, target_panel)
        imwrite(output_dir / "comparison_canvas.jpg", canvas)

        prompt = build_prompt(args.mode, canvas_meta, args.max_differences)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        client = OpenAICompatibleHTTPClient(
            model_name=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            timeout=args.timeout,
            num_predict=args.max_tokens,
            check_model=not args.skip_model_check,
        )
        response = client.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_to_base64(canvas)],
                }
            ],
            json_mode=True,
            think=False,
            timeout=args.timeout,
            num_predict=args.max_tokens,
        )
    except Exception as exc:
        error_payload = {
            "success": False,
            "error": str(exc),
            "template": args.template,
            "target": args.target,
        }
        write_json(output_dir / "result.json", error_payload)
        print(f"实验失败: {exc}")
        print(f"Output: {output_dir}")
        return 1

    raw_response_path = output_dir / "raw_response.txt"
    raw_response_path.write_text(response.content, encoding="utf-8")
    parsed_payload = extract_json_object(response.content)
    parsed = normalize_result(parsed_payload, canvas.shape[:2])
    visualization = draw_predictions(canvas, parsed["differences"])
    imwrite(output_dir / "vlm_predictions.jpg", visualization)

    result = {
        "success": True,
        "mode": args.mode,
        "model": response.model,
        "api_base": args.api_base,
        "debug_summary": response.debug_summary(),
        "template": args.template,
        "target": args.target,
        "summary": summary,
        "alignment": alignment,
        "canvas": {
            **canvas_meta,
            "scale_from_aligned_images": canvas_scale,
        },
        "parsed": parsed,
        "raw_response_path": str(raw_response_path),
        "artifacts": {
            "template_preprocessed": str(output_dir / "template_preprocessed.jpg"),
            "target_preprocessed": str(output_dir / "target_preprocessed.jpg"),
            "target_aligned": str(output_dir / "target_aligned.jpg"),
            "comparison_canvas": str(output_dir / "comparison_canvas.jpg"),
            "vlm_predictions": str(output_dir / "vlm_predictions.jpg"),
            "prompt": str(output_dir / "prompt.txt"),
        },
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
    print("VLM diff localization experiment complete")
    print("=" * 72)
    print(f"Decision: {parsed['decision']}")
    print(f"Differences: {len(parsed['differences'])}")
    print(f"Output: {output_dir}")
    print(f"Canvas: {output_dir / 'comparison_canvas.jpg'}")
    print(f"Predictions: {output_dir / 'vlm_predictions.jpg'}")
    print(f"Result: {output_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
