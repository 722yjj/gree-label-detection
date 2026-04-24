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
from typing import Dict, List, Optional, Type

import cv2
import numpy as np
import pandas as pd
from openpyxl.styles import Font
from pydantic import BaseModel

from label_detection.core import langchain_compat as _langchain_compat  # noqa: F401
from label_detection.core.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    VLM_NUM_PREDICT,
    LLM_MAX_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDF_PATH,
    DEFAULT_TARGET_PATH,
    USE_VLM_FOR_GRAPHIC,
    ENABLE_IMAGE_REGION_SPLIT,
    ensure_local_ollama_no_proxy,
    is_local_ollama,
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
from label_detection.services.ocr_service import get_ocr_with_boxes

_llm = None


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
    """Lazily initialize the Ollama client when the workflow actually runs."""
    global _llm
    if _llm is None:
        ensure_local_ollama_no_proxy(OLLAMA_API_BASE)
        from langchain_ollama import ChatOllama

        client_kwargs = {"trust_env": False} if is_local_ollama(OLLAMA_API_BASE) else {}
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API_BASE,
            temperature=0,
            num_predict=VLM_NUM_PREDICT,
            client_kwargs=client_kwargs,
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
    output_path = os.path.join(output_dir, "template_preprocessed.jpg")
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
    import json as json_module

    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    from langchain_core.messages import HumanMessage

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

    msg = HumanMessage(
        content=[
            {"type": "text", "text": final_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
            },
        ]
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke([msg])
            content = response.content

            # 检查空响应
            if not content or not content.strip():
                print(f"    ⚠ 第 {attempt+1} 次尝试: LLM 返回空响应，重试...")
                continue

            # [DEBUG] 打印 LLM 返回的原始内容
            print(f"    [DEBUG] LLM 原始返回 (前500字符):")
            print(f"    {content[:500]}")

            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                res_dict = json_module.loads(json_match.group())
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
                print(f"    ⚠ 第 {attempt+1} 次尝试: 未找到 JSON，重试...")
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
    foreground_ratio: float | None = None,
    exc: Exception | None = None,
) -> Dict:
    if source_side == "实拍":
        subject = f"实拍未匹配区域 #{source_idx}"
        side_hint = "疑似实拍多出图形，或模板侧漏检"
    else:
        subject = f"模板未匹配区域 #{source_idx}"
        side_hint = "疑似实拍缺失图形，或实拍侧漏检"

    if reason == "inference_failed":
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
        target_preprocessed_path = preprocess_dir / "target_preprocessed.jpg"
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
            template_label = (template_label_hits.get(field_name) or {}).get("text")
            target_label = (target_label_hits.get(field_name) or {}).get("text")
            if not template_label or not target_label:
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
                        pts = np.array(target_boxes[b_idx][0], dtype=np.int32)
                        if pts.shape == (4,):
                            x1, y1, x2, y2 = pts
                            poly = np.array(
                                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                dtype=np.int32,
                            )
                        else:
                            poly = pts.reshape((-1, 1, 2))
                        cv2.polylines(
                            vis_image,
                            [poly],
                            isClosed=True,
                            color=(0, 0, 255),
                            thickness=3,
                        )
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
                        pts = np.array(target_boxes[b_idx][0], dtype=np.int32)
                        if pts.shape == (4,):
                            x1, y1, x2, y2 = pts
                            poly = np.array(
                                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                dtype=np.int32,
                            )
                        else:
                            poly = pts.reshape((-1, 1, 2))
                        cv2.polylines(
                            vis_image,
                            [poly],
                            isClosed=True,
                            color=(0, 0, 255),
                            thickness=3,
                        )
                        diff_count += 1
                else:
                    print(
                        f"    - 字段 '{k}' 差异 (值: {target_val}) -> 未找到对应 OCR 框"
                    )

        print("  标注差异和需复核的图形区域...")
        needs_review_regions = []
        unresolved_regions = []
        for res in comparison_results:
            decision = res.get("decision", "unknown")
            target_idx = res.get("target_idx")
            region_box = None

            if target_idx is not None and target_idx < len(target_regions):
                region_box = target_regions[target_idx]["coordinate"]
            elif res.get("inferred_target_box") is not None:
                region_box = res["inferred_target_box"]

            if region_box is not None:
                x1, y1, x2, y2 = [int(v) for v in region_box]

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
            f"待复核 {len(needs_review_regions)} 处)"
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
        ]
        review_needed = [
            r
            for r in comparison_results
            if r.get("decision") == "unknown"
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
