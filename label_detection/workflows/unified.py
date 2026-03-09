"""Unified text and graphic comparison workflow."""

import argparse
import base64
import json
import os
import re
import time
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
)
from label_detection.extraction.pdf import extract_red_box_info
from label_detection.extraction.text import (
    count_populated_fields,
    extract_compact_spec_from_text,
    find_missing_fields,
    merge_compact_sources,
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
)
from label_detection.matching.ocr import find_matching_ocr_boxes
from label_detection.schema import LABEL_KIND_COMPACT, get_label_model, infer_label_kind
from label_detection.preprocessing.border import crop_to_border, find_template_crop_rect
from label_detection.preprocessing.pipeline import preprocess_target
from label_detection.services.ocr_service import get_ocr_with_boxes

_llm = None


def get_llm():
    """Lazily initialize the Ollama client when the workflow actually runs."""
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama

        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API_BASE,
            temperature=0,
            num_predict=VLM_NUM_PREDICT,
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
        return f"""【任务】：提取图中紧凑型标签的文字信息

【参考 OCR 文本】：
{ocr_text}

【字段定义】：
- model_number: 型号
- net_weight: N.W.
- gross_weight: G.W.
- color: Color
- connection_pipes: Connection Pipes
- refrigerant: Refrigerant
- barcode: 条形码下方的数字

【提取规则】：
1. 优先根据图片视觉内容提取，OCR 只用于纠错。
2. 只输出上面 7 个字段；没有就填 null。
3. 保留原始文本，不要补充不存在的字段。
4. 必须输出纯标准 JSON，不要包含 Markdown 代码块或解释。

【输出示例】：
{{"model_number":"GWH24AGD-K6DNA1C/I(WIFI)","net_weight":"14kg","gross_weight":"16.5kg","color":"White","connection_pipes":"1/4\\"/1/2\\"","refrigerant":"R32","barcode":"600001076226"}}"""

    return f"""【任务】：提取图中标签的文字信息

【参考信息】：
为了防止你看不清小字，我已经使用 OCR 技术识别了图中的文字，内容如下（可能存在乱序，仅供参考拼写和数字）：
{ocr_text}

【提取规则】：
1. 优先根据图片视觉内容提取,保留文字原始信息,不做任何修改。
2. 如果图片看不清，参考 OCR 文本。
3. 如果找不到某项，保持为 null。
4. 必须输出纯标准的 JSON 格式，不要包含 Markdown 代码块。
5. 只输出最后的JSON字符串，不要包含任何其他解释性文本和思考过程。

【输出示例】：
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
    print(f"    ✗ LLM 提取失败，使用空数据")
    return model_cls()


def compare_text_results(data1, data2, output_path=None):
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
    data_dict = {"参数": [], "模板图片": [], "实拍图片": [], "是否一致": []}

    for k in field_names:
        v1 = getattr(data1, k)
        v2 = getattr(data2, k)
        data_dict["参数"].append(k)
        data_dict["模板图片"].append(v1 if v1 else "-")
        data_dict["实拍图片"].append(v2 if v2 else "-")
        data_dict["是否一致"].append("✓" if v1 == v2 else "✗")

    df = pd.DataFrame(data_dict)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="文字对比")

        worksheet = writer.sheets["文字对比"]
        red_font = Font(color="FF0000")

        for row in range(2, len(df) + 2):
            cell1 = worksheet.cell(row=row, column=2)
            cell2 = worksheet.cell(row=row, column=3)
            cell3 = worksheet.cell(row=row, column=4)

            if cell1.value != cell2.value:
                cell1.font = red_font
                cell2.font = red_font
                cell3.font = red_font

    print(f"  文字对比结果已保存: {output_path}")
    return output_path


def run_unified_detection(pdf_path, target_image_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    完整的整合检测流程（使用 VLM 进行图形对比）

    Args:
        pdf_path: 模板 PDF 文件路径
        target_image_path: 实拍图片路径
        output_dir: 输出目录

    Returns:
        dict: 包含文字和图形检测结果
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("整合检测：文字检测 + 布局区域图形比较 (VLM Mode)")
    print("=" * 60)

    results = {
        "success": True,
        "text_detection": {},
        "graphic_comparison": {}
    }

    # ========== Step 1: 预处理 ==========
    print("\n" + "=" * 40)
    print("Step 1: 预处理")
    print("=" * 40)

    # 1.1 从 PDF 提取模板图片
    print("\n[1.1] 从 PDF 提取模板图片...")
    template_raw_path = extract_red_box_info(
        pdf_path,
        target_dpi=300,
        output_dir=os.path.join(output_dir, "template_assets"),
    )
    if not template_raw_path:
        return {"success": False, "error": "无法从 PDF 提取模板图片"}

    # 1.2 预处理模板（检测黑框，去除白边）
    print("\n[1.2] 预处理模板图片（去除黑框外白边）...")
    template_cropped, template_path, err = preprocess_template_image(
        template_raw_path, output_dir
    )
    if err:
        return {"success": False, "error": err}

    # 1.3 预处理实拍图片（带角度矫正）
    print("\n[1.3] 预处理实拍图片（带角度矫正）...")
    target_cropped, _, err = preprocess_target(
        target_image_path,
        output_dir,
        template_image=template_cropped
    )
    if err:
        return {"success": False, "error": err}

    # 保存预处理后的实拍图
    target_preprocessed_path = os.path.join(output_dir, "target_preprocessed.jpg")
    cv2.imwrite(target_preprocessed_path, target_cropped)

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
    target_text, target_boxes = get_ocr_with_boxes(target_preprocessed_path)
    print(f"  识别到 {len(target_boxes)} 个文字区域，共 {len(target_text)} 字符")

    label_kind = infer_label_kind(template_text or target_text)
    model_cls = get_label_model(label_kind)
    comparison_fields = list(model_cls.model_fields.keys())
    print(f"\n[2.3] 识别文字标签类型: {label_kind} ({len(comparison_fields)} 个字段)")
    print("[2.3] LLM/规则提取模板结构化信息...")
    llm_start = time.time()
    data1 = run_llm_extraction(template_path, template_text, model_cls, label_kind)
    print(f"  LLM 耗时: {time.time() - llm_start:.2f} 秒")

    print("\n[2.4] LLM/规则提取实拍结构化信息...")
    llm_start = time.time()
    data2 = run_llm_extraction(
        target_preprocessed_path,
        target_text,
        model_cls,
        label_kind,
    )
    print(f"  LLM 耗时: {time.time() - llm_start:.2f} 秒")

    # 2.5 对比文字结果
    print("\n[2.5] 对比文字结果...")
    text_excel_path = compare_text_results(
        data1, data2,
        os.path.join(output_dir, "text_comparison.xlsx")
    )

    results["text_detection"] = {
        "label_kind": label_kind,
        "fields": comparison_fields,
        "template_data": data1.model_dump(),
        "target_data": data2.model_dump(),
        "excel_path": text_excel_path,
        "time": time.time() - start_time
    }

    # ========== Step 3: 区域检测与对比 ==========
    print("\n" + "=" * 40)
    print(f"Step 3: 区域检测与对比")
    print("=" * 40)

    graphic_output_dir = os.path.join(output_dir, "graphic_comparison")
    os.makedirs(graphic_output_dir, exist_ok=True)

    # 3.1 检测布局区域 (使用 PP-DocLayoutV3)
    print("\n[3.1] 检测布局区域 (PP-DocLayoutV3)...")
    template_all_regions = detect_layout_regions(template_path)
    template_regions = extract_regions_by_type(template_all_regions, "image")

    target_all_regions = detect_layout_regions(target_preprocessed_path)
    target_regions = extract_regions_by_type(target_all_regions, "image")

    skipped_template_regions: List[Dict] = []
    skipped_target_regions: List[Dict] = []
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

    # 保存区域检测可视化
    template_vis = draw_regions(template_cropped, template_regions)
    target_vis = draw_regions(target_cropped, target_regions)
    if skipped_template_regions:
        template_vis = draw_regions(template_vis, skipped_template_regions, color=(0, 165, 255))
    if skipped_target_regions:
        target_vis = draw_regions(target_vis, skipped_target_regions, color=(0, 165, 255))
    cv2.imwrite(os.path.join(graphic_output_dir, "template_regions_detected.jpg"), template_vis)
    cv2.imwrite(os.path.join(graphic_output_dir, "target_regions_detected.jpg"), target_vis)

    # 3.2 区域匹配
    print("\n[3.2] 匹配对应区域...")
    matched_pairs, unmatched1, unmatched2 = match_regions(
        template_regions,
        target_regions,
        template_cropped.shape[:2],
        target_cropped.shape[:2]
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
                output_dir=graphic_output_dir,
                pair_idx=idx,
                use_vlm=USE_VLM_FOR_GRAPHIC,
            )
            res["template_idx"] = i
            res["target_idx"] = j
            res["match_distance"] = dist

            # 打印判定结果（使用 decision 字段）
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

    results["graphic_comparison"] = {
        "template_regions_total_count": len(extract_regions_by_type(template_all_regions, "image")),
        "target_regions_total_count": len(extract_regions_by_type(target_all_regions, "image")),
        "template_regions_count": len(template_regions),
        "target_regions_count": len(target_regions),
        "skipped_template_regions": skipped_template_regions,
        "skipped_target_regions": skipped_target_regions,
        "matched_count": len(matched_pairs),
        "comparison_results": comparison_results,
        "region_type": "image"
    }

    # ========== Step 4: 结果可视化 (差异标注) ==========
    print("\n" + "=" * 40)
    print("Step 4: 结果可视化 (标注差异)")
    print("=" * 40)

    # 在实拍预处理图上绘制
    vis_image = target_cropped.copy()
    diff_count = 0

    # 1. 标注差异的文字区域 (红色)
    print("  寻找并标注差异文字区域...")
    for k in comparison_fields:
        v1 = getattr(data1, k)
        v2 = getattr(data2, k)

        if v1 != v2:
            target_val = str(v2) if v2 else ""
            if not target_val or target_val == "None":
                continue

            matched_box_indices = find_matching_ocr_boxes(
                target_val,
                target_boxes,
                field_name=k,
            )

            if matched_box_indices:
                matched_texts = [target_boxes[b_idx][1] for b_idx in matched_box_indices]
                print(
                    f"    - 字段 '{k}' 差异 (值: {target_val}) -> 对应 "
                    f"{len(matched_box_indices)} 个 OCR 框: {matched_texts}"
                )
                for b_idx in matched_box_indices:
                    # 兼容不同格式的 poly 坐标信息，支持四点斜框
                    pts = np.array(target_boxes[b_idx][0], dtype=np.int32)
                    if pts.shape == (4,):
                        # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = pts
                        poly = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
                    else:
                        poly = pts.reshape((-1, 1, 2))
                    cv2.polylines(vis_image, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
                    diff_count += 1
            else:
                print(f"    - 字段 '{k}' 差异 (值: {target_val}) -> 未找到对应 OCR 框")

    # 2. 标注差异的图形区域
    print("  标注差异和需复核的图形区域...")
    needs_review_regions = []
    for res in comparison_results:
        decision = res.get("decision", "unknown")
        target_idx = res.get("target_idx")

        if target_idx is not None and target_idx < len(target_regions):
            region = target_regions[target_idx]
            x1, y1, x2, y2 = [int(v) for v in region["coordinate"]]

            if decision == "mismatch":
                # 确认差异：红框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(vis_image, "Diff", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                diff_count += 1
                print(f"    - 图形差异: 区域 #{target_idx} (确认不匹配)")
            elif decision == "unknown":
                # 需复核：黄框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 200, 255), 3)
                cv2.putText(vis_image, "Review", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                needs_review_regions.append(res)
                print(f"    - 图形需复核: 区域 #{target_idx} ({res.get('summary', '')})")

    # 保存可视化结果
    vis_path = os.path.join(output_dir, "visualization_diff.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"  差异可视化已保存: {vis_path} (差异 {diff_count} 处, 待复核 {len(needs_review_regions)} 处)")


    # ========== 输出汇总 ==========
    print("\n" + "=" * 60)
    print("检测完成！结果汇总")
    print("=" * 60)

    # 文字对比统计
    match_count = sum(
        1 for k in comparison_fields
        if getattr(data1, k) == getattr(data2, k)
    )
    total_fields = len(comparison_fields)

    print(f"\n📝 文字检测:")
    print(f"   - 字段匹配: {match_count}/{total_fields}")
    print(f"   - 结果文件: {text_excel_path}")

    print(f"\n🖼️ 图形比较 (基于区域):")
    print(f"   - 检测区域数: 模板 {len(template_regions)} / 实拍 {len(target_regions)}")
    if skipped_template_regions or skipped_target_regions:
        print(f"   - 条码跳过: 模板 {len(skipped_template_regions)} / 实拍 {len(skipped_target_regions)}")
    print(f"   - 成功匹配: {len(matched_pairs)} 对")

    # 统计分为三类
    confirmed_match = [r for r in comparison_results if r.get("decision") == "match"]
    confirmed_mismatch = [r for r in comparison_results if r.get("decision") == "mismatch"]
    review_needed = [r for r in comparison_results if r.get("decision") == "unknown"]

    graphic_pass = True
    has_review = len(review_needed) > 0

    if len(template_regions) != len(target_regions):
        print(f"   - ⚠️ 区域数量不一致!")
        graphic_pass = False

    if unmatched1 or unmatched2:
        print(f"   - ⚠️ 存在未匹配区域!")
        graphic_pass = False

    if confirmed_mismatch:
        graphic_pass = False
        for res in confirmed_mismatch:
            print(f"   - ❌ 确认不匹配: {res.get('summary')}")

    if review_needed:
        for res in review_needed:
            print(f"   - ⚠️ 需复核: {res.get('summary')}")

    if confirmed_match and not confirmed_mismatch and not review_needed and len(matched_pairs) > 0:
        print("   - ✅ 所有匹配区域均判定一致")
    elif len(matched_pairs) == 0:
         print("   - ⚠️ 无可见图形区域参与对比")

    print(f"   - 统计: 一致 {len(confirmed_match)}, 不一致 {len(confirmed_mismatch)}, 待复核 {len(review_needed)}")
    print(f"   - 结果目录: {output_dir}")
    print(f"   - 可视化图: {vis_path}")

    # 综合判定
    text_match_ratio = match_count / total_fields

    if text_match_ratio == 1.0 and graphic_pass and not has_review:
        verdict = "✅ 标签完全一致"
    elif text_match_ratio == 1.0 and graphic_pass and has_review:
        verdict = f"⚠️ 标签基本一致，{len(review_needed)} 处图形需人工复核"
    elif text_match_ratio >= 0.8 and graphic_pass:
        verdict = f"⚠️ 文字存在差异 ({int((1-text_match_ratio)*total_fields)} 处)，图形一致"
    elif text_match_ratio == 1.0 and not graphic_pass:
        verdict = f"⚠️ 文字一致，图形存在差异 ({len(confirmed_mismatch)} 处不匹配)"
    else:
        verdict = f"❌ 标签差异较大 (文字 {match_count}/{total_fields}，图形 {len(confirmed_mismatch)} 处不匹配)"

    print(f"\n🏷️ 综合判定: {verdict}")

    # 【修复】先补齐 verdict 和 output_dir，再写入 JSON
    results["verdict"] = verdict
    results["output_dir"] = output_dir

    # 保存最终结果 JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    final_json_path = os.path.join(output_dir, "final_result.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description="整合检测脚本 (VLM 模式)")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="模板 PDF 路径")
    parser.add_argument("--target", default=DEFAULT_TARGET_PATH, help="实拍图片路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("运行模式: VLM")
    result = run_unified_detection(args.pdf, args.target, output_dir=args.output_dir)
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
