"""Desktop-facing wrapper around the unified detection workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
        )

        return DetectionJobResult(
            success=bool(result.get("success")),
            verdict=result.get("verdict"),
            output_dir=Path(result.get("output_dir") or output_dir),
            summary_text=self._build_summary_text(result),
            visualization_path=self._resolve_visualization_path(
                Path(result.get("output_dir") or output_dir)
            ),
            raw_result=dict(result),
            error=result.get("error"),
        )

    def _build_output_dir(self, request: DetectionJobRequest) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        variant = request.template.variant or "default"
        return self.output_root / request.template.code / f"{timestamp}_{variant}"

    @staticmethod
    def _build_summary_text(result: dict) -> str:
        if not result.get("success"):
            return f"检测失败: {result.get('error', '未知错误')}"

        text_detection = dict(result.get("text_detection") or {})
        template_data = dict(text_detection.get("template_data") or {})
        target_data = dict(text_detection.get("target_data") or {})
        field_names = list(text_detection.get("fields") or template_data.keys() or target_data.keys())
        match_count = sum(
            1
            for field_name in field_names
            if field_values_match(template_data.get(field_name), target_data.get(field_name))
        )
        graphic = dict(result.get("graphic_comparison") or {})
        resolved_match_count = int(
            graphic.get("resolved_match_count")
            or graphic.get("effective_matched_count")
            or 0
        )
        mismatch_count = sum(
            1 for item in list(graphic.get("comparison_results") or [])
            if item.get("decision") == "mismatch"
        )
        review_count = sum(
            1 for item in list(graphic.get("comparison_results") or [])
            if item.get("decision") == "unknown"
        )

        lines = [
            f"综合判定: {result.get('verdict', '-')}",
            f"文字字段匹配: {match_count}/{len(field_names)}",
            f"图形已解决配对: {resolved_match_count}",
            f"图形不一致: {mismatch_count}",
            f"图形待复核: {review_count}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _resolve_visualization_path(output_dir: Path) -> Path | None:
        candidate = output_dir / "visualization_diff.jpg"
        return candidate if candidate.exists() else None
