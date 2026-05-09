"""Desktop-facing wrapper around the unified detection workflow."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import requests

from desktop_app.models import DetectionJobRequest, DetectionJobResult
from label_detection.core.config import (
    GRAPHIC_NUM_PREDICT,
    GRAPHIC_VLM_MODEL,
    OPENAI_COMPATIBLE_API_BASE,
    OPENAI_COMPATIBLE_API_KEY,
    PROJECT_ROOT,
    VLM_TIMEOUT,
)
from label_detection.matching.ocr import text_field_values_match


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_traditional_diff_model(api_base: str, api_key: str) -> str:
    explicit_model = os.getenv("DESKTOP_TRADITIONAL_DIFF_MODEL")
    if explicit_model:
        return explicit_model.strip()
    if os.getenv("GRAPHIC_VLM_MODEL") or os.getenv("OPENAI_COMPATIBLE_MODEL") or os.getenv("VLLM_MODEL"):
        return GRAPHIC_VLM_MODEL

    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(
            f"{api_base.rstrip('/')}/models",
            headers=headers,
            timeout=2,
        )
        response.raise_for_status()
        names = [
            str(item.get("id") or "").strip()
            for item in response.json().get("data", [])
            if str(item.get("id") or "").strip()
        ]
        if len(names) == 1:
            return names[0]
    except requests.RequestException:
        pass

    return GRAPHIC_VLM_MODEL


class DetectionService:
    """Run the existing workflow and normalize the result for the UI."""

    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = Path(output_root or PROJECT_ROOT / "results" / "desktop_app")

    def run(self, request: DetectionJobRequest) -> DetectionJobResult:
        pipeline = os.getenv(
            "DESKTOP_DETECTION_PIPELINE",
            "traditional_full_image_diff",
        ).strip().lower()
        if pipeline in {"unified", "legacy"}:
            return self._run_unified(request)
        if pipeline not in {"traditional_full_image_diff", "traditional", "full_image_diff"}:
            raise ValueError(f"不支持的桌面端检测流程: {pipeline}")
        return self._run_traditional_full_image_diff(request)

    def _run_unified(self, request: DetectionJobRequest) -> DetectionJobResult:
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

    def _run_traditional_full_image_diff(
        self,
        request: DetectionJobRequest,
    ) -> DetectionJobResult:
        from label_detection.workflows.traditional_full_image_diff import (
            run_traditional_full_image_diff,
        )

        output_dir = self._build_output_dir(request)
        api_base = os.getenv(
            "DESKTOP_TRADITIONAL_DIFF_API_BASE",
            OPENAI_COMPATIBLE_API_BASE,
        )
        api_key = os.getenv(
            "DESKTOP_TRADITIONAL_DIFF_API_KEY",
            OPENAI_COMPATIBLE_API_KEY,
        )
        result = run_traditional_full_image_diff(
            request.template.source_path,
            request.target_image_path,
            output_dir=output_dir.parent,
            run_name=output_dir.name,
            output_mode=request.output_mode,
            panel_width=int(os.getenv("DESKTOP_TRADITIONAL_DIFF_PANEL_WIDTH", "900")),
            vlm_filter=_env_bool("DESKTOP_TRADITIONAL_DIFF_VLM_FILTER", True),
            model=_resolve_traditional_diff_model(api_base, api_key),
            api_base=api_base,
            api_key=api_key,
            timeout=int(os.getenv("DESKTOP_TRADITIONAL_DIFF_TIMEOUT", str(VLM_TIMEOUT))),
            max_tokens=int(
                os.getenv("DESKTOP_TRADITIONAL_DIFF_MAX_TOKENS", str(GRAPHIC_NUM_PREDICT))
            ),
            crop_padding=int(os.getenv("DESKTOP_TRADITIONAL_DIFF_CROP_PADDING", "32")),
            review_min_size=int(os.getenv("DESKTOP_TRADITIONAL_DIFF_REVIEW_MIN_SIZE", "128")),
            keep_unknown=_env_bool("DESKTOP_TRADITIONAL_DIFF_KEEP_UNKNOWN", False),
            skip_model_check=_env_bool("DESKTOP_TRADITIONAL_DIFF_SKIP_MODEL_CHECK", False),
        )
        resolved_output_dir = Path(result.get("output_dir") or output_dir)
        result = self._normalize_traditional_desktop_result(
            result,
            resolved_output_dir,
            output_mode=request.output_mode,
        )
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

        if result.get("pipeline") == "traditional_full_image_diff":
            vlm_results = list(result.get("vlm_filter_results") or [])
            kept = sum(1 for item in vlm_results if item.get("vlm_decision") == "keep")
            discarded = sum(
                1 for item in vlm_results if item.get("vlm_decision") == "discard"
            )
            unknown = sum(
                1
                for item in vlm_results
                if item.get("vlm_decision") not in {"keep", "discard"}
            )
            return "\n".join(
                [
                    f"综合判定: {result.get('verdict', '-')}",
                    f"全图差异: {int(result.get('final_box_count') or 0)}",
                    (
                        "候选框: "
                        f"{int(result.get('raw_candidate_box_count') or 0)} -> "
                        f"{int(result.get('merged_candidate_box_count') or 0)}"
                    ),
                    f"VLM过滤: 保留 {kept} / 丢弃 {discarded} / 未定 {unknown}",
                ]
            )

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
        if result.get("pipeline") == "traditional_full_image_diff":
            return {
                "text_match_count": 0,
                "text_total_count": 0,
                "graphic_match_count": 0,
                "graphic_mismatch_count": int(result.get("final_box_count") or 0),
                "graphic_review_count": 0,
            }

        text_detection = dict(result.get("text_detection") or {})
        template_data = dict(text_detection.get("template_data") or {})
        target_data = dict(text_detection.get("target_data") or {})
        field_names = list(
            text_detection.get("fields") or template_data.keys() or target_data.keys()
        )
        text_match_count = sum(
            1
            for field_name in field_names
            if text_field_values_match(
                field_name,
                template_data.get(field_name),
                target_data.get(field_name),
            )
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
    def _normalize_traditional_desktop_result(
        result: Mapping[str, Any],
        output_dir: Path,
        *,
        output_mode: str,
    ) -> dict[str, Any]:
        payload = dict(result)
        payload["pipeline"] = "traditional_full_image_diff"
        payload["output_dir"] = str(output_dir)
        payload["output_mode"] = output_mode
        payload["verdict"] = "不一致" if int(payload.get("final_box_count") or 0) else "通过"

        source_image = output_dir / "final_result.jpg"
        visualization_path = output_dir / "visualization_diff.jpg"
        if source_image.exists() and source_image != visualization_path:
            shutil.copy2(source_image, visualization_path)

        artifacts = dict(payload.get("artifacts") or {})
        if visualization_path.exists():
            artifacts["visualization_diff"] = str(visualization_path)
            artifacts["final_result"] = str(visualization_path)
            payload["visualization_diff"] = str(visualization_path)
        payload["artifacts"] = artifacts

        final_boxes = list(payload.get("final_boxes") or [])
        payload["text_detection"] = {
            "fields": [],
            "template_data": {},
            "target_data": {},
            "excel_path": None,
        }
        payload["graphic_comparison"] = {
            "resolved_match_count": 0,
            "comparison_results": [
                {
                    "decision": "mismatch",
                    "box": item.get("box") if isinstance(item, Mapping) else item,
                    "summary": "traditional full-image diff",
                }
                for item in final_boxes
            ],
        }

        final_json_path = output_dir / "final_result.json"
        with final_json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        if output_mode == "final":
            for transient_path in (output_dir / "result.json", source_image):
                if transient_path.exists() and transient_path not in {
                    final_json_path,
                    visualization_path,
                }:
                    transient_path.unlink()

        return payload

    @staticmethod
    def _resolve_visualization_path(output_dir: Path) -> Path | None:
        candidate = output_dir / "visualization_diff.jpg"
        return candidate if candidate.exists() else None
