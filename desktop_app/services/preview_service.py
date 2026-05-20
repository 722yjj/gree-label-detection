"""Template preview resolution helpers for the desktop app."""

from __future__ import annotations

from pathlib import Path

from desktop_app.models import TemplateRecord
from label_detection.core.config import RESULTS_ROOT
from label_detection.extraction.template_source import resolve_template_input


class PreviewService:
    """Resolve a template into a preview image path consumable by the UI."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = Path(
            cache_root
            or RESULTS_ROOT / "desktop_app" / "template_preview_cache"
        )
        self._cache: dict[Path, tuple[int, Path | None]] = {}

    def resolve_preview_path(self, template: TemplateRecord) -> Path | None:
        source_path = template.source_path.resolve()
        if not source_path.exists():
            self._cache.pop(source_path, None)
            return None
        source_mtime_ns = source_path.stat().st_mtime_ns
        cached_entry = self._cache.get(source_path)
        if cached_entry is not None:
            cached_source_mtime_ns, cached_path = cached_entry
            if cached_source_mtime_ns == source_mtime_ns:
                if cached_path is None:
                    return None
                if self._is_cache_fresh(cached_path, source_mtime_ns):
                    return cached_path

        if template.preview_path and template.preview_path.exists():
            resolved = template.preview_path.resolve()
            self._cache[source_path] = (source_mtime_ns, resolved)
            return resolved

        cache_dir = self.cache_root / source_path.stem
        existing = self._find_cached_preview(cache_dir, source_mtime_ns)
        if existing is not None:
            resolved = existing.resolve()
            self._cache[source_path] = (source_mtime_ns, resolved)
            return resolved
        self._clear_cache_dir(cache_dir)

        resolved_path, _source_type, err = resolve_template_input(
            str(source_path),
            output_dir=str(cache_dir),
            target_dpi=180,
        )
        if err or not resolved_path:
            self._cache[source_path] = (source_mtime_ns, None)
            return None

        resolved = Path(resolved_path).resolve()
        cached_path = resolved if resolved.exists() else None
        self._cache[source_path] = (source_mtime_ns, cached_path)
        return cached_path

    @staticmethod
    def _is_cache_fresh(preview_path: Path, source_mtime_ns: int) -> bool:
        return preview_path.exists() and preview_path.stat().st_mtime_ns >= source_mtime_ns

    def _find_cached_preview(self, cache_dir: Path, source_mtime_ns: int) -> Path | None:
        fresh_previews = [
            path
            for path in sorted(cache_dir.glob("*.png"))
            if self._is_cache_fresh(path, source_mtime_ns)
        ]
        if not fresh_previews:
            return None
        return max(fresh_previews, key=lambda path: path.stat().st_mtime_ns)

    @staticmethod
    def _clear_cache_dir(cache_dir: Path) -> None:
        if not cache_dir.exists():
            return
        for pattern in ("*.png", "*.txt"):
            for path in cache_dir.glob(pattern):
                path.unlink()
