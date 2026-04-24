"""CLI for manually testing VLM object detection on a single image."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_detection.core.config import GRAPHIC_VLM_MODEL, OLLAMA_API_BASE, PROJECT_ROOT as REPO_ROOT, VLM_TIMEOUT
from label_detection.services.vlm_detection import VLMObjectDetector, draw_detections


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "vlm_object_detection"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单图 VLM 目标检测测试脚本")
    parser.add_argument("--image", required=True, help="待检测图片路径")
    parser.add_argument(
        "--query",
        default="barcode, qr code, logo, certification icon",
        help="要检测的目标描述，多个目标可用逗号分隔",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认 results/vlm_object_detection",
    )
    parser.add_argument("--model", default=GRAPHIC_VLM_MODEL, help="Ollama 模型名称")
    parser.add_argument("--api-base", default=OLLAMA_API_BASE, help="Ollama API 地址")
    parser.add_argument("--timeout", type=int, default=VLM_TIMEOUT, help="请求超时时间（秒）")
    parser.add_argument("--max-objects", type=int, default=10, help="最多返回多少个目标")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="可视化时过滤低于该置信度的目标",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="额外保存原始模型响应文本",
    )
    parser.add_argument(
        "--prompt",
        help="自定义完整提示词；传入后将跳过内置目标检测提示词模板",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    import cv2

    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片: {image_path}")
        return 1

    detector = VLMObjectDetector(
        model_name=args.model,
        api_base=args.api_base,
        timeout=args.timeout,
    )

    try:
        result = detector.detect_objects(
            image=image,
            query=args.query,
            max_objects=args.max_objects,
            custom_prompt=args.prompt,
        )
    except Exception as exc:
        print(f"检测失败: {exc}")
        return 1

    all_objects = result.get("objects", [])
    filtered_objects = [
        item for item in all_objects
        if float(item.get("confidence", 0.0)) >= args.min_confidence
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_{timestamp}_annotated.jpg"
    json_path = output_dir / f"{stem}_{timestamp}_detections.json"
    raw_response_path = output_dir / f"{stem}_{timestamp}_raw_response.txt"

    annotated = draw_detections(
        image,
        filtered_objects,
        min_confidence=args.min_confidence,
    )
    cv2.imwrite(str(annotated_path), annotated)

    payload = {
        "success": result.get("success", False),
        "parse_error": result.get("parse_error", False),
        "image_path": str(image_path),
        "query": args.query,
        "model": result.get("model", args.model),
        "image_size": result.get("image_size", {}),
        "summary": result.get("summary", ""),
        "object_count": len(all_objects),
        "filtered_object_count": len(filtered_objects),
        "objects": all_objects,
        "filtered_objects": filtered_objects,
        "annotated_image_path": str(annotated_path),
    }

    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    if args.save_raw_response:
        with open(raw_response_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(result.get("raw_response", ""))

    print("=" * 60)
    print("VLM 目标检测测试完成")
    print("=" * 60)
    print(f"图片: {image_path}")
    print(f"查询: {args.query}")
    print(f"模型: {result.get('model', args.model)}")
    print(f"总结: {result.get('summary', '')}")
    print(f"解析状态: {'失败' if result.get('parse_error') else '成功'}")
    print(f"检测目标数: {len(all_objects)}")
    print(f"可视化保留数: {len(filtered_objects)} (min_confidence={args.min_confidence})")
    for idx, item in enumerate(filtered_objects, start=1):
        print(
            f"  [{idx}] {item['label']} "
            f"conf={item['confidence']:.2f} "
            f"bbox_px={item['bbox_px']}"
        )
    print(f"标注图: {annotated_path}")
    print(f"JSON 结果: {json_path}")
    if args.save_raw_response:
        print(f"原始响应: {raw_response_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
