"""Desktop-facing wrapper around the unified detection workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from desktop_app.models import DetectionJobRequest, DetectionJobResult
from label_detection.core.config import PROJECT_ROOT
from label_detection.matching.ocr import field_values_match


class DetectionService:
    """Run the existing workflow and normalize the result for the UI."""

    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = Path(output_root or PROJECT_ROOT / "results" / "desktop_app")

    def run(self, request: DetectionJobRequest) -> DetectionJobResult:
        from label_detection.workflows.unified import run_unified_detection

        output_dir = self._build_output_dir(request)
        result = run_unified_detection(
            str(request.template.source_path),
            str(request.target_image_path),
            output_dir=str(output_dir),
            output_mode=request.output_mode,
        )
        resolved_output_dir = Path(result.get("output_dir") or output_dir)
        summary = self._summarize_result(result)

        return DetectionJobResult(
            success=bool(result.get("success")),
            verdict=result.get("verdict"),
            output_dir=resolved_output_dir,
            summary_text=self._build_summary_text(result, summary),
            visualization_path=self._resolve_visualization_path(resolved_output_dir),
            raw_result=dict(result),
            error=result.get("error"),
            text_match_count=summary["text_match_count"],
            text_total_count=summary["text_total_count"],
            graphic_match_count=summary["graphic_match_count"],
            graphic_mismatch_count=summary["graphic_mismatch_count"],
            graphic_review_count=summary["graphic_review_count"],
            target_image_path=request.target_image_path,
            template_path=request.template.source_path,
            code=request.template.code,
            template_display_name=request.template.display_name,
        )

    def _build_output_dir(self, request: DetectionJobRequest) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        variant = request.template.variant or "default"
        return self.output_root / request.template.code / f"{timestamp}_{variant}"

    @classmethod
    def _build_summary_text(cls, result: Mapping[str, Any], summary: Mapping[str, int]) -> str:
        if not result.get("success"):
            return f"检测失败: {result.get('error', '未知错误')}"

        lines = [
            f"综合判定: {result.get('verdict', '-')}",
            f"文字字段匹配: {summary['text_match_count']}/{summary['text_total_count']}",
            f"图形匹配: {summary['graphic_match_count']}",
            f"图形不一致: {summary['graphic_mismatch_count']}",
            f"图形待复核: {summary['graphic_review_count']}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _summarize_result(result: Mapping[str, Any]) -> dict[str, int]:
        text_detection = dict(result.get("text_detection") or {})
        template_data = dict(text_detection.get("template_data") or {})
        target_data = dict(text_detection.get("target_data") or {})
        field_names = list(
            text_detection.get("fields") or template_data.keys() or target_data.keys()
        )
        text_match_count = sum(
            1
            for field_name in field_names
            if field_values_match(template_data.get(field_name), target_data.get(field_name))
        )

        graphic = dict(result.get("graphic_comparison") or {})
        comparison_results = list(graphic.get("comparison_results") or [])
        graphic_match_count = int(
            graphic.get("resolved_match_count")
            or graphic.get("effective_matched_count")
            or 0
        )
        graphic_mismatch_count = sum(
            1 for item in comparison_results if item.get("decision") == "mismatch"
        )
        graphic_review_count = sum(
            1 for item in comparison_results if item.get("decision") == "unknown"
        )
        return {
            "text_match_count": text_match_count,
            "text_total_count": len(field_names),
            "graphic_match_count": graphic_match_count,
            "graphic_mismatch_count": graphic_mismatch_count,
            "graphic_review_count": graphic_review_count,
        }

    @staticmethod
    def _resolve_visualization_path(output_dir: Path) -> Path | None:
        candidate = output_dir / "visualization_diff.jpg"
        return candidate if candidate.exists() else None
