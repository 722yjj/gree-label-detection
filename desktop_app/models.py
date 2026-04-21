"""Shared desktop app data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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
