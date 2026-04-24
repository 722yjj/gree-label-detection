"""Direct-run test for comparing PP-DocLayoutV3 image regions with VLM regions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from label_detection.core.config import GRAPHIC_VLM_MODEL, OLLAMA_API_BASE, PROJECT_ROOT as REPO_ROOT, VLM_TIMEOUT
from label_detection.extraction.pdf import extract_red_box_info
from label_detection.matching.layout import calculate_iou, detect_layout_regions, draw_regions, extract_regions_by_type
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template
from label_detection.services.vlm_detection import VLMObjectDetector


# ==================== 直接在这里改测试参数 ====================
TEST_NAME = "vlm_vs_layout_image_regions"
IMAGE_SOURCES = ("template", "target")  # 可选: ("template",) / ("target",) / ("template", "target")

PDF_PATH = REPO_ROOT / "samples" / "pdfs" / "600004075219-01.pdf"
TARGET_IMAGE_PATH = REPO_ROOT / "samples" / "images" / "produce" / "type1" / "600004075219_1.jpg"

USE_PROJECT_PREPROCESSING = True
REGION_TYPE = "image"
LAYOUT_THRESHOLD = 0.30
IOU_MATCH_THRESHOLD = 0.30
MAX_VLM_REGIONS = 12

OUTPUT_DIR = REPO_ROOT / "results" / "vlm_layout_region_test"
VLM_MODEL = GRAPHIC_VLM_MODEL
VLM_API_BASE = OLLAMA_API_BASE
VLM_TIMEOUT_SECONDS = VLM_TIMEOUT

VLM_REGION_PROMPT = f"""You are a layout graphic-region detector.
Goal: find all graphic regions in the full label image that should correspond to layout-model `image` regions.

Definition of a graphic region / image region:
- icons, certification marks, logos, QR codes, barcodes, diagrams, picture blocks, or other non-text visual regions
- a region may include a small amount of attached text, but the main content must be a graphic, image, or code block
- do not mark pure text paragraphs, pure number lines, table lines, borders, or blank areas as graphic regions

Output rules:
1. Output exactly one JSON object and nothing else. Do not output explanations, analysis, Markdown, code fences, or thinking text.
2. The JSON schema is fixed:
{{"regions":[{{"label":"image","confidence":0.0,"bbox_1000":[x1,y1,x2,y2]}}],"summary":"..."}}
3. `bbox_1000` uses normalized 0-1000 coordinates relative to the full image: [x1, y1, x2, y2].
4. Return at most {MAX_VLM_REGIONS} highest-confidence regions.
5. If uncertain, do not guess. If no image region is detected, return {{"regions":[],"summary":"no image region found"}}.
"""


def prepare_test_images(output_dir: Path) -> Dict[str, Dict[str, object]]:
    """Prepare template/target images used by both layout detection and VLM detection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_payloads: Dict[str, Dict[str, object]] = {}
    common_meta: Dict[str, object] = {
        "image_sources": list(IMAGE_SOURCES),
        "use_project_preprocessing": USE_PROJECT_PREPROCESSING,
    }

    template_raw_path = None
    template_image = None
    if PDF_PATH.exists():
        template_asset_dir = output_dir / "template_assets"
        template_raw_path = extract_red_box_info(
            str(PDF_PATH),
            target_dpi=300,
            output_dir=str(template_asset_dir),
        )
        if template_raw_path:
            template_preprocess_dir = output_dir / "template_preprocess"
            template_image, _, template_err = preprocess_template(
                template_raw_path,
                output_dir=str(template_preprocess_dir),
            )
            if template_err:
                print(f"[WARN] 模板预处理失败: {template_err}")
                template_image = None

    if "template" in IMAGE_SOURCES:
        if not template_raw_path:
            raise RuntimeError(f"无法从 PDF 提取模板图: {PDF_PATH}")
        if USE_PROJECT_PREPROCESSING and template_image is not None:
            prepared_image = template_image
            prepared_from = "template_preprocess"
        else:
            prepared_image = cv2.imread(str(template_raw_path))
            prepared_from = "template_raw_image"

        if prepared_image is None:
            raise RuntimeError("模板测试图片准备失败")

        prepared_path = output_dir / "template_prepared.jpg"
        cv2.imwrite(str(prepared_path), prepared_image)
        image_payloads["template"] = {
            "image": prepared_image,
            "prepared_path": prepared_path,
            "meta": {
                **common_meta,
                "image_source": "template",
                "template_raw_path": template_raw_path,
                "prepared_from": prepared_from,
                "prepared_image_path": str(prepared_path),
            },
        }

    if "target" in IMAGE_SOURCES:
        if USE_PROJECT_PREPROCESSING:
            target_preprocess_dir = output_dir / "target_preprocess"
            prepared_image, _, target_err = preprocess_target(
                str(TARGET_IMAGE_PATH),
                output_dir=str(target_preprocess_dir),
                template_image=template_image,
            )
            if target_err:
                raise RuntimeError(target_err)
            prepared_from = "target_preprocess"
        else:
            prepared_image = cv2.imread(str(TARGET_IMAGE_PATH))
            prepared_from = "target_raw_image"

        if prepared_image is None:
            raise RuntimeError("实拍测试图片准备失败")

        prepared_path = output_dir / "target_prepared.jpg"
        cv2.imwrite(str(prepared_path), prepared_image)
        image_payloads["target"] = {
            "image": prepared_image,
            "prepared_path": prepared_path,
            "meta": {
                **common_meta,
                "image_source": "target",
                "target_image_path": str(TARGET_IMAGE_PATH),
                "template_raw_path": template_raw_path,
                "prepared_from": prepared_from,
                "prepared_image_path": str(prepared_path),
            },
        }

    return image_payloads


def build_vlm_region_dicts(vlm_objects: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    regions: List[Dict[str, object]] = []
    for idx, item in enumerate(vlm_objects):
        bbox = item.get("bbox_px")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        regions.append(
            {
                "label": str(item.get("label") or "image"),
                "score": float(item.get("confidence", 0.0)),
                "coordinate": [int(value) for value in bbox],
                "vlm_idx": idx,
            }
        )
    return regions


def compare_region_sets(
    layout_regions: Sequence[Dict[str, object]],
    vlm_regions: Sequence[Dict[str, object]],
    iou_threshold: float,
) -> Dict[str, object]:
    matches: List[Dict[str, object]] = []
    used_vlm = set()

    for layout_idx, layout_region in enumerate(layout_regions):
        best_vlm_idx = None
        best_iou = 0.0

        for vlm_idx, vlm_region in enumerate(vlm_regions):
            if vlm_idx in used_vlm:
                continue

            iou = calculate_iou(
                layout_region["coordinate"],
                vlm_region["coordinate"],
            )
            if iou > best_iou:
                best_iou = float(iou)
                best_vlm_idx = vlm_idx

        if best_vlm_idx is not None and best_iou >= iou_threshold:
            used_vlm.add(best_vlm_idx)
            matches.append(
                {
                    "layout_idx": layout_idx,
                    "vlm_idx": best_vlm_idx,
                    "iou": round(best_iou, 4),
                    "layout_box": layout_regions[layout_idx]["coordinate"],
                    "vlm_box": vlm_regions[best_vlm_idx]["coordinate"],
                }
            )

    unmatched_layout = [idx for idx in range(len(layout_regions)) if idx not in {m["layout_idx"] for m in matches}]
    unmatched_vlm = [idx for idx in range(len(vlm_regions)) if idx not in used_vlm]

    recall_vs_layout = len(matches) / len(layout_regions) if layout_regions else 0.0
    precision_vs_layout = len(matches) / len(vlm_regions) if vlm_regions else 0.0

    return {
        "matches": matches,
        "matched_count": len(matches),
        "unmatched_layout": unmatched_layout,
        "unmatched_vlm": unmatched_vlm,
        "recall_vs_layout": round(recall_vs_layout, 4),
        "precision_vs_layout": round(precision_vs_layout, 4),
    }


def save_visualizations(
    image,
    layout_regions: Sequence[Dict[str, object]],
    vlm_regions: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_vis = draw_regions(image, list(layout_regions), color=(0, 255, 0))
    vlm_vis = draw_regions(image, list(vlm_regions), color=(0, 0, 255))
    overlay_vis = draw_regions(layout_vis, list(vlm_regions), color=(0, 0, 255))

    layout_path = output_dir / "layout_regions.jpg"
    vlm_path = output_dir / "vlm_regions.jpg"
    overlay_path = output_dir / "layout_vs_vlm_overlay.jpg"

    cv2.imwrite(str(layout_path), layout_vis)
    cv2.imwrite(str(vlm_path), vlm_vis)
    cv2.imwrite(str(overlay_path), overlay_vis)

    return {
        "layout_regions_image": str(layout_path),
        "vlm_regions_image": str(vlm_path),
        "overlay_image": str(overlay_path),
    }


def run_single_image_test(
    image_source: str,
    image_payload: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    image = image_payload["image"]
    prepared_path = image_payload["prepared_path"]
    image_meta = image_payload["meta"]

    print("\n" + "=" * 60)
    print(f"[{image_source}] 测试开始")
    print("=" * 60)
    print(f"  prepared image: {prepared_path}")

    print(f"\n[{image_source}] 运行 PP-DocLayoutV3 获取基线图形区域...")
    layout_all_regions = detect_layout_regions(str(prepared_path), threshold=LAYOUT_THRESHOLD)
    layout_regions = extract_regions_by_type(layout_all_regions, REGION_TYPE)
    print(f"  - [{image_source}] 布局模型总区域数: {len(layout_all_regions)}")
    print(f"  - [{image_source}] {REGION_TYPE} 区域数: {len(layout_regions)}")

    print(f"\n[{image_source}] 运行 VLM 检测图形区域...")
    detector = VLMObjectDetector(
        model_name=VLM_MODEL,
        api_base=VLM_API_BASE,
        timeout=VLM_TIMEOUT_SECONDS,
    )
    vlm_result = detector.detect_objects(
        image=image,
        query="layout image regions",
        max_objects=MAX_VLM_REGIONS,
        custom_prompt=VLM_REGION_PROMPT,
    )
    vlm_regions = build_vlm_region_dicts(vlm_result.get("objects", []))
    print(f"  - [{image_source}] VLM 返回区域数: {len(vlm_regions)}")
    print(f"  - [{image_source}] VLM 摘要: {vlm_result.get('summary', '')}")
    print(f"  - [{image_source}] 解析错误: {vlm_result.get('parse_error', False)}")

    print(f"\n[{image_source}] 对比布局模型区域和 VLM 区域...")
    comparison = compare_region_sets(
        layout_regions=layout_regions,
        vlm_regions=vlm_regions,
        iou_threshold=IOU_MATCH_THRESHOLD,
    )
    print(f"  - [{image_source}] 匹配成功: {comparison['matched_count']}")
    print(f"  - [{image_source}] layout recall: {comparison['recall_vs_layout']:.4f}")
    print(f"  - [{image_source}] vlm precision: {comparison['precision_vs_layout']:.4f}")
    print(f"  - [{image_source}] 未匹配 layout 区域: {comparison['unmatched_layout']}")
    print(f"  - [{image_source}] 未匹配 vlm 区域: {comparison['unmatched_vlm']}")

    print(f"\n[{image_source}] 保存可视化结果...")
    vis_paths = save_visualizations(image, layout_regions, vlm_regions, output_dir)
    for key, value in vis_paths.items():
        print(f"  - {key}: {value}")

    raw_response_path = output_dir / "vlm_raw_response.txt"
    raw_response_path.write_text(vlm_result.get("raw_response", ""), encoding="utf-8")

    summary = {
        "test_name": TEST_NAME,
        "image_source": image_source,
        "prepared_image_path": str(prepared_path),
        "image_meta": image_meta,
        "layout_threshold": LAYOUT_THRESHOLD,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "vlm_model": VLM_MODEL,
        "vlm_api_base": VLM_API_BASE,
        "vlm_prompt": VLM_REGION_PROMPT,
        "vlm_summary": vlm_result.get("summary", ""),
        "vlm_parse_error": vlm_result.get("parse_error", False),
        "layout_region_count": len(layout_regions),
        "vlm_region_count": len(vlm_regions),
        "layout_regions": layout_regions,
        "vlm_regions": vlm_regions,
        "comparison": comparison,
        "visualizations": vis_paths,
        "raw_response_path": str(raw_response_path),
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(f"\n[{image_source}] 汇总结果")
    print(f"  - summary json: {summary_path}")
    print(f"  - raw response: {raw_response_path}")
    print("  - 可以直接看 overlay 图判断 VLM 框和布局模型框是否接近")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    output_dir = OUTPUT_DIR / TEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("测试目标: VLM 是否能替代布局检测模型找出图形区域")
    print("=" * 60)
    print(f"TEST_NAME: {TEST_NAME}")
    print(f"IMAGE_SOURCES: {IMAGE_SOURCES}")
    print(f"OUTPUT_DIR: {output_dir}")

    image_payloads = prepare_test_images(output_dir)
    all_results: Dict[str, Dict[str, object]] = {}

    for image_source in IMAGE_SOURCES:
        if image_source not in image_payloads:
            print(f"[WARN] 未准备好 {image_source} 测试图片，跳过")
            continue

        source_output_dir = output_dir / image_source
        all_results[image_source] = run_single_image_test(
            image_source=image_source,
            image_payload=image_payloads[image_source],
            output_dir=source_output_dir,
        )

    combined_summary = {
        "test_name": TEST_NAME,
        "image_sources": list(IMAGE_SOURCES),
        "output_dir": str(output_dir),
        "results": all_results,
    }
    combined_summary_path = output_dir / "all_results_summary.json"
    with open(combined_summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(combined_summary, file_obj, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("全部测试完成")
    print("=" * 60)
    print(f"combined summary: {combined_summary_path}")
    for image_source, result in all_results.items():
        print(f"  - {image_source}: {result.get('summary_path', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
