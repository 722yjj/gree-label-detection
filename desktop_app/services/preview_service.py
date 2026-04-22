"""Template preview resolution helpers for the desktop app."""

from __future__ import annotations

from pathlib import Path

from desktop_app.models import TemplateRecord
from label_detection.core.config import PROJECT_ROOT
from label_detection.extraction.template_source import resolve_template_input


class PreviewService:
    """Resolve a template into a preview image path consumable by the UI."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = Path(
            cache_root
            or PROJECT_ROOT / "results" / "desktop_app" / "template_preview_cache"
        )
        self._cache: dict[Path, Path | None] = {}

    def resolve_preview_path(self, template: TemplateRecord) -> Path | None:
        source_path = template.source_path.resolve()
        cached = self._cache.get(source_path)
        if cached is not None:
            return cached if cached.exists() else None

        if template.preview_path and template.preview_path.exists():
            resolved = template.preview_path.resolve()
            self._cache[source_path] = resolved
            return resolved

        cache_dir = self.cache_root / source_path.stem
        existing = sorted(cache_dir.glob("*.png"))
        if existing:
            resolved = existing[0].resolve()
            self._cache[source_path] = resolved
            return resolved

        resolved_path, _source_type, err = resolve_template_input(
            str(source_path),
            output_dir=str(cache_dir),
            target_dpi=180,
        )
        if err or not resolved_path:
            self._cache[source_path] = None
            return None

        resolved = Path(resolved_path).resolve()
        self._cache[source_path] = resolved if resolved.exists() else None
        return self._cache[source_path]
