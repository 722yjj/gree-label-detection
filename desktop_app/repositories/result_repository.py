"""Load persisted desktop detection results from a run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from desktop_app.models import DetectionJobResult, HistoryRecord
from desktop_app.services.detection_service import DetectionService


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[DetectionResultRepository] failed to read {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _path_if_exists(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DetectionResultRepository:
    """Rebuild UI-facing detection results without rerunning detection."""

    def load_raw_result(self, run_dir: Path) -> dict[str, Any] | None:
        run_dir = Path(run_dir)
        return _read_json(run_dir / "final_result.json") or _read_json(run_dir / "result.json")

    def build_from_history_record(
        self,
        record: HistoryRecord,
    ) -> tuple[DetectionJobResult, float | None] | None:
        run_dir = Path(record.output_dir)
        raw_result = self.load_raw_result(run_dir)
        if raw_result is None:
            return None

        summary = DetectionService._summarize_result(raw_result)
        visualization_path = self._resolve_visualization_path(record, raw_result, run_dir)
        result = DetectionJobResult(
            success=bool(raw_result.get("success", True)),
            verdict=str(raw_result.get("verdict") or record.verdict or "-"),
            output_dir=run_dir,
            summary_text=record.summary_text
            or DetectionService._build_summary_text(raw_result, summary),
            visualization_path=visualization_path,
            raw_result=dict(raw_result),
            error=raw_result.get("error"),
            text_match_count=summary["text_match_count"],
            text_total_count=summary["text_total_count"],
            graphic_match_count=summary["graphic_match_count"],
            graphic_mismatch_count=summary["graphic_mismatch_count"],
            graphic_review_count=summary["graphic_review_count"],
            target_image_path=record.target_image_path,
            template_path=_path_if_exists(raw_result.get("template")),
            code=record.code,
            template_display_name=record.template_name,
        )
        return result, self._load_detection_duration(run_dir, raw_result)

    @staticmethod
    def _resolve_visualization_path(
        record: HistoryRecord,
        raw_result: Mapping[str, Any],
        run_dir: Path,
    ) -> Path | None:
        artifacts = raw_result.get("artifacts")
        artifact_visualization = (
            artifacts.get("visualization_diff")
            if isinstance(artifacts, Mapping)
            else None
        )
        for value in (
            record.visualization_path,
            raw_result.get("visualization_diff"),
            artifact_visualization,
            run_dir / "visualization_diff.jpg",
            run_dir / "final_result.jpg",
        ):
            path = _path_if_exists(value)
            if path is not None:
                return path
        return None

    @staticmethod
    def _load_detection_duration(
        run_dir: Path,
        raw_result: Mapping[str, Any],
    ) -> float | None:
        annotation = _read_json(run_dir / "manual_annotation.json") or {}
        for value in (
            annotation.get("detection_duration_seconds"),
            raw_result.get("detection_duration_seconds"),
            raw_result.get("duration_seconds"),
        ):
            parsed = _float_or_none(value)
            if parsed is not None:
                return parsed
        return None
