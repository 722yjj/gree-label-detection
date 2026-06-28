"""Export manually reviewed desktop runs into a real-photo dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop_app.repositories.annotation_repository import ANNOTATION_FILENAME
from label_detection.core.config import RESULTS_ROOT


DEFAULT_SOURCE_ROOT = RESULTS_ROOT / "desktop_app"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "real_photo_dataset"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export desktop manual annotations into a real-photo dataset manifest."
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Desktop app results root containing manual_annotation.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for manifest.json.",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include annotations without any reviewed predicted box or manual missing box.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def repo_relative(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def annotation_to_sample(annotation_path: Path, annotation: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(annotation.get("run_dir") or annotation_path.parent))
    image = dict(annotation.get("image") or {})
    predicted_boxes = [
        dict(item)
        for item in annotation.get("predicted_boxes") or []
        if isinstance(item, Mapping)
    ]
    manual_gt_boxes = [
        dict(item)
        for item in annotation.get("manual_gt_boxes") or []
        if isinstance(item, Mapping)
    ]
    sample_id = f"{annotation.get('code') or run_dir.parent.name}_{run_dir.name}"
    return {
        "sample_id": sample_id,
        "code": annotation.get("code"),
        "template_display_name": annotation.get("template_display_name"),
        "run_dir": str(run_dir),
        "annotation_path": str(annotation_path),
        "template": annotation.get("template"),
        "target_image": annotation.get("target_image"),
        "visualization_image": image.get("path"),
        "detection_duration_seconds": annotation.get("detection_duration_seconds"),
        "predicted_boxes": predicted_boxes,
        "manual_gt_boxes": manual_gt_boxes,
        "notes": annotation.get("notes") or "",
        "relative_paths": {
            "run_dir": repo_relative(run_dir),
            "annotation": repo_relative(annotation_path),
            "template": repo_relative(annotation.get("template")),
            "target_image": repo_relative(annotation.get("target_image")),
            "visualization_image": repo_relative(image.get("path")),
        },
    }


def export_annotations(
    source_root: Path,
    output_dir: Path,
    *,
    include_unreviewed: bool,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    annotation_paths = sorted(source_root.rglob(ANNOTATION_FILENAME))
    samples: list[dict[str, Any]] = []
    skipped_unreviewed = 0

    for annotation_path in annotation_paths:
        annotation = read_json(annotation_path)
        predicted_boxes = [
            item
            for item in annotation.get("predicted_boxes") or []
            if isinstance(item, Mapping)
        ]
        has_reviewed_box = any(
            str(item.get("decision") or "unreviewed") in {"true_positive", "false_positive"}
            for item in predicted_boxes
        )
        has_manual_missing = bool(annotation.get("manual_gt_boxes") or [])
        if not include_unreviewed and not has_reviewed_box and not has_manual_missing:
            skipped_unreviewed += 1
            continue
        samples.append(annotation_to_sample(annotation_path, annotation))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": output_dir.name,
        "version": "real_photo_dataset_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "sample_count": len(samples),
        "skipped_unreviewed": skipped_unreviewed,
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "skipped_unreviewed": skipped_unreviewed,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = export_annotations(
        Path(args.source_root),
        Path(args.output_dir),
        include_unreviewed=args.include_unreviewed,
    )
    print(f"manifest={result['manifest']}")
    print(f"sample_count={result['sample_count']}")
    print(f"skipped_unreviewed={result['skipped_unreviewed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
