"""JSON-backed manual annotation storage for real-photo desktop runs."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from desktop_app.models import DetectionJobResult, HistoryRecord


ANNOTATION_FILENAME = "manual_annotation.json"
ANNOTATION_VERSION = "real_photo_annotation_v1"
DEFAULT_BOX_DECISION = "unreviewed"
REVIEWED_BOX_DECISIONS = {"true_positive", "false_positive", "unreviewed"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[AnnotationRepository] failed to read {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_box(value: object) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _path_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _existing_path_or_none(*values: object) -> Path | None:
    for value in values:
        text = _path_or_none(value)
        if not text:
            continue
        path = Path(text)
        if path.exists():
            return path
    return None


def _box_label(item: Mapping[str, Any]) -> str:
    text_line = item.get("matched_template_text_line")
    if isinstance(text_line, Mapping) and text_line.get("text"):
        return str(text_line["text"])
    if item.get("matched_template_text_segment"):
        return str(item["matched_template_text_segment"])
    if item.get("vlm_reason"):
        return str(item["vlm_reason"])[:120]
    return str(item.get("source") or "final_box")


def extract_predicted_boxes(raw_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract final desktop-visible boxes into annotation candidates."""

    source_boxes = raw_result.get("final_boxes") or []
    if not isinstance(source_boxes, Sequence) or isinstance(source_boxes, (str, bytes)):
        return []

    boxes: list[dict[str, Any]] = []
    for index, raw_item in enumerate(source_boxes):
        if not isinstance(raw_item, Mapping):
            continue
        bbox = _normalize_box(raw_item.get("display_box") or raw_item.get("box"))
        if bbox is None:
            continue
        boxes.append(
            {
                "box_id": f"pred_{index}",
                "bbox": bbox,
                "decision": DEFAULT_BOX_DECISION,
                "kind": str(raw_item.get("review_box_source") or raw_item.get("source") or "unknown"),
                "label": _box_label(raw_item),
                "note": "",
                "raw_index": index,
            }
        )
    return boxes


class AnnotationRepository:
    """Persist and load manual annotations beside desktop detection outputs."""

    def annotation_path(self, run_dir: Path) -> Path:
        return Path(run_dir) / ANNOTATION_FILENAME

    def load(self, run_dir: Path) -> dict[str, Any] | None:
        return _read_json(self.annotation_path(run_dir))

    def save(self, annotation: Mapping[str, Any]) -> Path:
        run_dir = Path(str(annotation["run_dir"]))
        path = self.annotation_path(run_dir)
        payload = deepcopy(dict(annotation))
        now = _now_iso()
        payload.setdefault("version", ANNOTATION_VERSION)
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def build_from_detection_result(self, result: DetectionJobResult) -> dict[str, Any]:
        raw_result = dict(result.raw_result or {})
        if not raw_result:
            raw_result = _read_json(result.output_dir / "final_result.json") or {}
        return self._build_draft(
            run_dir=result.output_dir,
            raw_result=raw_result,
            target_image_path=result.target_image_path,
            template_path=result.template_path,
            visualization_path=result.visualization_path,
            code=result.code,
            template_display_name=result.template_display_name,
        )

    def build_from_history_record(self, record: HistoryRecord) -> dict[str, Any]:
        run_dir = Path(record.output_dir)
        raw_result = _read_json(run_dir / "final_result.json") or _read_json(run_dir / "result.json") or {}
        return self._build_draft(
            run_dir=run_dir,
            raw_result=raw_result,
            target_image_path=record.target_image_path,
            template_path=_path_or_none(raw_result.get("template")),
            visualization_path=record.visualization_path or _path_or_none(raw_result.get("visualization_diff")),
            code=record.code,
            template_display_name=record.template_name,
        )

    def _build_draft(
        self,
        *,
        run_dir: Path,
        raw_result: Mapping[str, Any],
        target_image_path: Path | str | None,
        template_path: Path | str | None,
        visualization_path: Path | str | None,
        code: str | None,
        template_display_name: str | None,
    ) -> dict[str, Any]:
        run_dir = Path(run_dir)
        artifacts = raw_result.get("artifacts") if isinstance(raw_result.get("artifacts"), Mapping) else {}
        annotation_image = _existing_path_or_none(
            raw_result.get("annotation_image"),
            artifacts.get("annotation_base") if isinstance(artifacts, Mapping) else None,
            run_dir / "annotation_base.jpg",
        )
        visualization = _existing_path_or_none(
            visualization_path,
            raw_result.get("visualization_diff"),
            run_dir / "visualization_diff.jpg",
            run_dir / "final_result.jpg",
        )
        display_image = annotation_image or visualization
        coordinate_space = (
            str(raw_result.get("annotation_coordinate_space") or "aligned_label_image")
            if annotation_image is not None
            else "visualization_diff"
        )

        draft = {
            "version": ANNOTATION_VERSION,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "annotator": "manual",
            "run_dir": str(run_dir),
            "code": code or _path_or_none(raw_result.get("code")),
            "template_display_name": template_display_name,
            "template": _path_or_none(template_path) or _path_or_none(raw_result.get("template")),
            "target_image": _path_or_none(target_image_path) or _path_or_none(raw_result.get("target")),
            "coordinate_space": coordinate_space,
            "image": {
                "path": str(display_image) if display_image else None,
            },
            "predicted_boxes": extract_predicted_boxes(raw_result),
            "manual_gt_boxes": [],
            "notes": "",
        }
        existing = self.load(run_dir)
        return self.merge_existing(draft, existing)

    @staticmethod
    def merge_existing(
        draft: Mapping[str, Any],
        existing: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload = deepcopy(dict(draft))
        if not existing:
            return payload

        existing_boxes = {
            str(item.get("box_id")): item
            for item in existing.get("predicted_boxes") or []
            if isinstance(item, Mapping) and item.get("box_id")
        }
        merged_boxes: list[dict[str, Any]] = []
        for box in payload.get("predicted_boxes") or []:
            merged = dict(box)
            old = existing_boxes.get(str(merged.get("box_id")))
            if old:
                old_decision = str(old.get("decision") or merged.get("decision") or DEFAULT_BOX_DECISION)
                merged["decision"] = (
                    old_decision
                    if old_decision in REVIEWED_BOX_DECISIONS
                    else DEFAULT_BOX_DECISION
                )
                merged["note"] = old.get("note") or merged.get("note") or ""
            merged_boxes.append(merged)

        payload["created_at"] = existing.get("created_at") or payload.get("created_at")
        payload["updated_at"] = existing.get("updated_at") or payload.get("updated_at")
        payload["annotator"] = existing.get("annotator") or payload.get("annotator")
        payload["predicted_boxes"] = merged_boxes
        payload["manual_gt_boxes"] = list(existing.get("manual_gt_boxes") or [])
        payload["notes"] = existing.get("notes") or payload.get("notes") or ""
        return payload
