"""Shared desktop app data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class TemplateRecord:
    """A template candidate exposed to the desktop UI."""

    code: str
    variant: Optional[str]
    display_name: str
    source_type: str
    source_path: Path
    preview_path: Optional[Path] = None
    label_kind: Optional[str] = None
    is_default: bool = False


@dataclass(frozen=True)
class DetectionJobRequest:
    """A single detection run request from the UI."""

    template: TemplateRecord
    target_image_path: Path
    output_mode: Literal["final", "debug"] = "final"


@dataclass(frozen=True)
class DetectionJobResult:
    """A UI-friendly detection result wrapper."""

    success: bool
    verdict: Optional[str]
    output_dir: Path
    summary_text: str
    visualization_path: Optional[Path] = None
    raw_result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    text_match_count: int = 0
    text_total_count: int = 0
    graphic_match_count: int = 0
    graphic_mismatch_count: int = 0
    graphic_review_count: int = 0
    target_image_path: Optional[Path] = None
    template_path: Optional[Path] = None
    code: Optional[str] = None
    template_display_name: Optional[str] = None


@dataclass(frozen=True)
class HistoryRecord:
    """A persisted desktop detection history entry."""

    created_at: str
    code: str
    template_name: str
    verdict: str
    output_dir: Path
    target_image_path: Path
    visualization_path: Optional[Path] = None
    summary_text: str = ""
