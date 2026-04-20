"""CLI experiment for large graphic-region splitting diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from label_detection.core.config import (
    LAYOUT_DETECTION_THRESHOLD,
    MATCH_COST_THRESHOLD,
    PROJECT_ROOT as REPO_ROOT,
)
from label_detection.extraction.template_source import resolve_template_input
from label_detection.matching.layout import (
    detect_barcode_region,
    detect_layout_regions,
    extract_regions_by_type,
    match_regions,
    split_barcode_regions,
    split_composite_image_regions,
)
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template
from label_detection.services.ocr_service import get_ocr_with_boxes


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "composite_graphic_split"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="大图形块二次拆分实验脚本")
    parser.add_argument("--template", required=True, help="模板文件路径，支持 PDF 或图片")
    parser.add_argument("--target", required=True, help="实拍图片路径")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认 results/composite_graphic_split",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LAYOUT_DETECTION_THRESHOLD,
        help="版面检测阈值，默认读取项目配置",
    )
    parser.add_argument(
        "--match-cost-threshold",
        type=float,
        default=MATCH_COST_THRESHOLD,
        help="区域匹配代价阈值，默认读取项目配置",
    )
    parser.add_argument(
        "--min-child-area-ratio",
        type=float,
        default=0.02,
        help="拆分子区域最小父框面积占比，默认 0.02",
    )
    parser.add_argument(
        "--min-child-side",
        type=int,
        default=18,
        help="拆分子区域最小边长，默认 18",
    )
    parser.add_argument(
        "--thin-sliver-aspect",
        type=float,
        default=3.0,
        help="细长碎片的宽高比阈值，默认 3.0",
    )
    parser.add_argument(
        "--thin-sliver-height-ratio",
        type=float,
        default=0.28,
        help="细长碎片的父框高度占比上限，默认 0.28",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="保留已有输出目录，不先清空",
    )
    return parser


def box_metrics(box: Sequence[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / height,
    }


def region_payload(region: Dict[str, object]) -> Dict[str, object]:
    payload = {
        "coordinate": [int(v) for v in region["coordinate"]],
        "label": region.get("label"),
        "score": float(region.get("score", 0.0)),
    }
    for key in (
        "original_idx",
        "split_child_idx",
        "split_child_count",
        "skip_reason",
    ):
        if key in region:
            payload[key] = region[key]
    if region.get("split_parent_coordinate") is not None:
        payload["split_parent_coordinate"] = [
            int(v) for v in region["split_parent_coordinate"]
        ]
    if region.get("split_children") is not None:
        payload["split_children"] = [
            [int(v) for v in child]
            for child in region["split_children"]
        ]
    return payload


def clone_regions(regions: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return [dict(region) for region in regions]


def parent_key(box: Sequence[int] | None) -> Tuple[int, int, int, int] | None:
    if box is None:
        return None
    return tuple(int(v) for v in box)


def looks_like_barcode_cluster(
    parent_box: Sequence[int],
    children: Sequence[Dict[str, object]],
    image,
    ocr_boxes,
) -> bool:
    if len(children) < 4:
        return False

    _, barcode_meta = detect_barcode_region({"coordinate": list(parent_box)}, image, ocr_boxes)
    parent = box_metrics(parent_box)
    if parent["aspect_ratio"] < 2.8:
        return False

    child_metrics = [box_metrics(region["coordinate"]) for region in children]
    narrow_children = sum(1 for item in child_metrics if item["aspect_ratio"] < 0.7)
    tall_children = sum(
        1 for item in child_metrics
        if item["height"] >= parent["height"] * 0.65
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
        and parent["aspect_ratio"] >= 3.0
    )
    return bool(
        (parent_texture_barcode or right_side_barcode)
        and narrow_children / max(1, len(child_metrics)) >= 0.6
        and tall_children / max(1, len(child_metrics)) >= 0.6
    )


def classify_split_fragment(
    region: Dict[str, object],
    min_child_area_ratio: float,
    min_child_side: int,
    thin_sliver_aspect: float,
    thin_sliver_height_ratio: float,
) -> str | None:
    parent_box = region.get("split_parent_coordinate")
    if parent_box is None:
        return None

    child = box_metrics(region["coordinate"])
    parent = box_metrics(parent_box)

    if child["area"] < parent["area"] * min_child_area_ratio:
        return "small_child_area"
    if child["width"] < min_child_side or child["height"] < min_child_side:
        return "small_child_side"
    if (
        child["aspect_ratio"] >= thin_sliver_aspect
        and child["height"] < parent["height"] * thin_sliver_height_ratio
    ):
        return "thin_sliver"
    return None


def filter_split_regions(
    regions: Sequence[Dict[str, object]],
    image,
    ocr_boxes,
    min_child_area_ratio: float,
    min_child_side: int,
    thin_sliver_aspect: float,
    thin_sliver_height_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    grouped_children: Dict[Tuple[int, int, int, int], List[Dict[str, object]]] = defaultdict(list)

    for region in regions:
        key = parent_key(region.get("split_parent_coordinate"))
        if key is not None:
            grouped_children[key].append(dict(region))

    barcode_cluster_parents = {
        key
        for key, children in grouped_children.items()
        if looks_like_barcode_cluster(key, children, image, ocr_boxes)
    }

    for region in regions:
        region_copy = dict(region)
        key = parent_key(region_copy.get("split_parent_coordinate"))
        if key in barcode_cluster_parents:
            region_copy["skip_reason"] = "barcode_cluster"
            skipped.append(region_copy)
            continue

        skip_reason = classify_split_fragment(
            region_copy,
            min_child_area_ratio=min_child_area_ratio,
            min_child_side=min_child_side,
            thin_sliver_aspect=thin_sliver_aspect,
            thin_sliver_height_ratio=thin_sliver_height_ratio,
        )
        if skip_reason:
            region_copy["skip_reason"] = skip_reason
            skipped.append(region_copy)
        else:
            kept.append(region_copy)

    return kept, skipped


def draw_region_groups(
    image,
    groups: Sequence[Tuple[str, Sequence[Dict[str, object]], Tuple[int, int, int]]],
):
    canvas = image.copy()
    for prefix, regions, color in groups:
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = [int(v) for v in region["coordinate"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label_parts = [f"{prefix}{idx}"]
            if region.get("split_child_idx") is not None:
                label_parts.append(f"c{region['split_child_idx']}")
            if region.get("skip_reason"):
                label_parts.append(str(region["skip_reason"]))
            label = " ".join(label_parts)
            cv2.putText(
                canvas,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return canvas


def write_region_debug_images(
    image,
    output_dir: Path,
    image_regions: Sequence[Dict[str, object]],
    comparable_regions: Sequence[Dict[str, object]],
    skipped_barcode_regions: Sequence[Dict[str, object]],
    split_regions: Sequence[Dict[str, object]],
    split_parents: Sequence[Dict[str, object]],
    filtered_regions: Sequence[Dict[str, object]],
    skipped_fragments: Sequence[Dict[str, object]],
) -> None:
    split_children = [region for region in split_regions if region.get("split_parent_coordinate") is not None]
    unsplit_regions = [region for region in split_regions if region.get("split_parent_coordinate") is None]

    cv2.imwrite(
        str(output_dir / "regions_image_all.jpg"),
        draw_region_groups(image, [("I", image_regions, (0, 255, 0))]),
    )
    cv2.imwrite(
        str(output_dir / "regions_barcode_filtered.jpg"),
        draw_region_groups(
            image,
            [
                ("C", comparable_regions, (0, 200, 0)),
                ("B", skipped_barcode_regions, (0, 165, 255)),
            ],
        ),
    )
    cv2.imwrite(
        str(output_dir / "regions_split_raw.jpg"),
        draw_region_groups(
            image,
            [
                ("U", unsplit_regions, (255, 180, 0)),
                ("P", split_parents, (0, 165, 255)),
                ("S", split_children, (0, 255, 0)),
            ],
        ),
    )
    cv2.imwrite(
        str(output_dir / "regions_split_filtered.jpg"),
        draw_region_groups(
            image,
            [
                ("K", filtered_regions, (0, 255, 0)),
                ("X", skipped_fragments, (0, 0, 255)),
            ],
        ),
    )


def write_parent_crops(
    image,
    output_dir: Path,
    split_parents: Sequence[Dict[str, object]],
    split_regions: Sequence[Dict[str, object]],
    filtered_regions: Sequence[Dict[str, object]],
    skipped_fragments: Sequence[Dict[str, object]],
) -> None:
    parent_dir = output_dir / "split_parents"
    parent_dir.mkdir(parents=True, exist_ok=True)

    children_by_parent: Dict[Tuple[int, int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for region in split_regions:
        key = parent_key(region.get("split_parent_coordinate"))
        if key is not None:
            children_by_parent[key].append(region)

    filtered_keys = {
        (parent_key(region.get("split_parent_coordinate")), tuple(int(v) for v in region["coordinate"]))
        for region in filtered_regions
        if region.get("split_parent_coordinate") is not None
    }
    skipped_keys = {
        (parent_key(region.get("split_parent_coordinate")), tuple(int(v) for v in region["coordinate"]))
        for region in skipped_fragments
        if region.get("split_parent_coordinate") is not None
    }

    image_h, image_w = image.shape[:2]
    for idx, parent in enumerate(split_parents):
        px1, py1, px2, py2 = [int(v) for v in parent["coordinate"]]
        pad_x = max(8, int(round((px2 - px1) * 0.06)))
        pad_y = max(8, int(round((py2 - py1) * 0.08)))
        cx1 = max(0, px1 - pad_x)
        cy1 = max(0, py1 - pad_y)
        cx2 = min(image_w, px2 + pad_x)
        cy2 = min(image_h, py2 + pad_y)

        crop = image[cy1:cy2, cx1:cx2].copy()
        crop_overlay = crop.copy()

        for child_idx, child in enumerate(children_by_parent.get(parent_key(parent["coordinate"]), [])):
            x1, y1, x2, y2 = [int(v) for v in child["coordinate"]]
            local_box = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
            state_key = (parent_key(child.get("split_parent_coordinate")), tuple(int(v) for v in child["coordinate"]))
            if state_key in filtered_keys:
                color = (0, 255, 0)
                prefix = "K"
            elif state_key in skipped_keys:
                color = (0, 0, 255)
                prefix = "X"
            else:
                color = (0, 255, 255)
                prefix = "S"
            cv2.rectangle(
                crop_overlay,
                (local_box[0], local_box[1]),
                (local_box[2], local_box[3]),
                color,
                2,
            )
            cv2.putText(
                crop_overlay,
                f"{prefix}{child_idx}",
                (local_box[0], max(16, local_box[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.imwrite(str(parent_dir / f"parent_{idx:02d}_crop.jpg"), crop)
        cv2.imwrite(str(parent_dir / f"parent_{idx:02d}_overlay.jpg"), crop_overlay)
        with (parent_dir / f"parent_{idx:02d}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "parent": region_payload(parent),
                    "children": [
                        region_payload(child)
                        for child in children_by_parent.get(parent_key(parent["coordinate"]), [])
                    ],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def summarize_matching(
    template_regions: Sequence[Dict[str, object]],
    target_regions: Sequence[Dict[str, object]],
    template_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    cost_threshold: float,
) -> Dict[str, object]:
    matched_pairs, unmatched_template, unmatched_target = match_regions(
        clone_regions(template_regions),
        clone_regions(target_regions),
        template_shape,
        target_shape,
        cost_threshold=cost_threshold,
    )
    return {
        "matched_count": len(matched_pairs),
        "unmatched_template_count": len(unmatched_template),
        "unmatched_target_count": len(unmatched_target),
        "matched_pairs": [
            {
                "template_idx": int(i),
                "target_idx": int(j),
                "cost": round(float(cost), 6),
            }
            for i, j, cost in matched_pairs
        ],
        "unmatched_template_indices": [int(idx) for idx in unmatched_template],
        "unmatched_target_indices": [int(idx) for idx in unmatched_target],
    }


def analyze_image(
    image_label: str,
    prepared_path: Path,
    image,
    output_dir: Path,
    threshold: float,
    min_child_area_ratio: float,
    min_child_side: int,
    thin_sliver_aspect: float,
    thin_sliver_height_ratio: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _, ocr_boxes = get_ocr_with_boxes(str(prepared_path))
    all_regions = detect_layout_regions(str(prepared_path), threshold=threshold)
    image_regions = extract_regions_by_type(all_regions, "image")
    comparable_regions, skipped_barcode_regions = split_barcode_regions(
        image_regions,
        image,
        ocr_boxes,
    )
    split_regions, split_parents = split_composite_image_regions(
        comparable_regions,
        image,
        ocr_boxes,
    )
    filtered_regions, skipped_fragments = filter_split_regions(
        split_regions,
        image=image,
        ocr_boxes=ocr_boxes,
        min_child_area_ratio=min_child_area_ratio,
        min_child_side=min_child_side,
        thin_sliver_aspect=thin_sliver_aspect,
        thin_sliver_height_ratio=thin_sliver_height_ratio,
    )

    write_region_debug_images(
        image=image,
        output_dir=output_dir,
        image_regions=image_regions,
        comparable_regions=comparable_regions,
        skipped_barcode_regions=skipped_barcode_regions,
        split_regions=split_regions,
        split_parents=split_parents,
        filtered_regions=filtered_regions,
        skipped_fragments=skipped_fragments,
    )
    write_parent_crops(
        image=image,
        output_dir=output_dir,
        split_parents=split_parents,
        split_regions=split_regions,
        filtered_regions=filtered_regions,
        skipped_fragments=skipped_fragments,
    )

    summary = {
        "image_label": image_label,
        "prepared_path": str(prepared_path),
        "ocr_box_count": len(ocr_boxes),
        "all_region_count": len(all_regions),
        "image_region_count": len(image_regions),
        "comparable_region_count": len(comparable_regions),
        "skipped_barcode_count": len(skipped_barcode_regions),
        "split_region_count": len(split_regions),
        "split_parent_count": len(split_parents),
        "filtered_region_count": len(filtered_regions),
        "skipped_fragment_count": len(skipped_fragments),
        "regions": {
            "image_regions": [region_payload(region) for region in image_regions],
            "comparable_regions": [region_payload(region) for region in comparable_regions],
            "skipped_barcode_regions": [region_payload(region) for region in skipped_barcode_regions],
            "split_regions": [region_payload(region) for region in split_regions],
            "split_parents": [region_payload(region) for region in split_parents],
            "filtered_regions": [region_payload(region) for region in filtered_regions],
            "skipped_fragments": [region_payload(region) for region in skipped_fragments],
        },
    }

    with (output_dir / f"{image_label}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def prepare_images(template_path: str, target_path: str, output_dir: Path) -> Dict[str, object]:
    prepared_dir = output_dir / "prepared"
    template_source_dir = prepared_dir / "template_source"
    template_preprocess_dir = prepared_dir / "template_preprocess"
    target_preprocess_dir = prepared_dir / "target_preprocess"

    resolved_template_path, source_type, template_error = resolve_template_input(
        template_path,
        output_dir=str(template_source_dir),
    )
    if template_error or not resolved_template_path:
        raise RuntimeError(template_error or "模板解析失败")

    template_image, _, template_preprocess_error = preprocess_template(
        resolved_template_path,
        output_dir=str(template_preprocess_dir),
    )
    if template_preprocess_error or template_image is None:
        raise RuntimeError(template_preprocess_error or "模板预处理失败")

    target_image, _, target_preprocess_error = preprocess_target(
        target_path,
        output_dir=str(target_preprocess_dir),
        template_image=template_image,
    )
    if target_preprocess_error or target_image is None:
        raise RuntimeError(target_preprocess_error or "目标预处理失败")

    template_prepared_path = prepared_dir / "template_prepared.jpg"
    target_prepared_path = prepared_dir / "target_prepared.jpg"
    cv2.imwrite(str(template_prepared_path), template_image)
    cv2.imwrite(str(target_prepared_path), target_image)

    return {
        "template_image": template_image,
        "target_image": target_image,
        "template_prepared_path": template_prepared_path,
        "target_prepared_path": target_prepared_path,
        "template_source_type": source_type,
        "resolved_template_path": resolved_template_path,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_images(args.template, args.target, output_dir)

    template_summary = analyze_image(
        image_label="template",
        prepared_path=prepared["template_prepared_path"],
        image=prepared["template_image"],
        output_dir=output_dir / "template",
        threshold=args.threshold,
        min_child_area_ratio=args.min_child_area_ratio,
        min_child_side=args.min_child_side,
        thin_sliver_aspect=args.thin_sliver_aspect,
        thin_sliver_height_ratio=args.thin_sliver_height_ratio,
    )
    target_summary = analyze_image(
        image_label="target",
        prepared_path=prepared["target_prepared_path"],
        image=prepared["target_image"],
        output_dir=output_dir / "target",
        threshold=args.threshold,
        min_child_area_ratio=args.min_child_area_ratio,
        min_child_side=args.min_child_side,
        thin_sliver_aspect=args.thin_sliver_aspect,
        thin_sliver_height_ratio=args.thin_sliver_height_ratio,
    )

    matching_summary = {
        "before_split": summarize_matching(
            template_summary["regions"]["comparable_regions"],
            target_summary["regions"]["comparable_regions"],
            prepared["template_image"].shape[:2],
            prepared["target_image"].shape[:2],
            cost_threshold=args.match_cost_threshold,
        ),
        "after_split": summarize_matching(
            template_summary["regions"]["filtered_regions"],
            target_summary["regions"]["filtered_regions"],
            prepared["template_image"].shape[:2],
            prepared["target_image"].shape[:2],
            cost_threshold=args.match_cost_threshold,
        ),
    }

    summary = {
        "template": {
            "input_path": args.template,
            "resolved_path": prepared["resolved_template_path"],
            "source_type": prepared["template_source_type"],
            "analysis": template_summary,
        },
        "target": {
            "input_path": args.target,
            "analysis": target_summary,
        },
        "matching": matching_summary,
        "parameters": {
            "threshold": args.threshold,
            "match_cost_threshold": args.match_cost_threshold,
            "min_child_area_ratio": args.min_child_area_ratio,
            "min_child_side": args.min_child_side,
            "thin_sliver_aspect": args.thin_sliver_aspect,
            "thin_sliver_height_ratio": args.thin_sliver_height_ratio,
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n[完成] 大图形块拆分实验已输出:")
    print(f"  - Summary: {summary_path}")
    print(f"  - 模板可比对区域: {template_summary['comparable_region_count']} -> {template_summary['filtered_region_count']}")
    print(f"  - 实拍可比对区域: {target_summary['comparable_region_count']} -> {target_summary['filtered_region_count']}")
    print(
        "  - 匹配对数: "
        f"拆分前 {matching_summary['before_split']['matched_count']} / "
        f"拆分后 {matching_summary['after_split']['matched_count']}"
    )


if __name__ == "__main__":
    main()
