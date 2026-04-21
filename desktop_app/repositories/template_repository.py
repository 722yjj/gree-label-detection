"""Template discovery and query helpers for the desktop app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from desktop_app.models import TemplateRecord
from label_detection.batch_samples import extract_label_code
from label_detection.core.config import PROJECT_ROOT, SAMPLES_DIR
from label_detection.extraction.template_source import SUPPORTED_TEMPLATE_IMAGE_SUFFIXES


class TemplateRepository:
    """Discover template files from the repository and expose query helpers."""

    def __init__(self, roots: Optional[Iterable[Path]] = None) -> None:
        self.roots = tuple(roots or self.default_roots())
        self._templates: list[TemplateRecord] = []
        self.refresh()

    @staticmethod
    def default_roots() -> tuple[Path, ...]:
        return (
            SAMPLES_DIR / "pdfs",
            SAMPLES_DIR / "images" / "original",
        )

    def refresh(self) -> None:
        self._templates = self._scan_templates()

    def find_by_code(self, code: str) -> list[TemplateRecord]:
        normalized = code.strip()
        if not normalized:
            return []
        return [record for record in self._templates if record.code == normalized]

    def list_known_codes(self) -> list[str]:
        return sorted({record.code for record in self._templates})

    def _scan_templates(self) -> list[TemplateRecord]:
        records: list[TemplateRecord] = []
        allowed_suffixes = {".pdf", *SUPPORTED_TEMPLATE_IMAGE_SUFFIXES}

        for root in self.roots:
            if not root.exists():
                continue

            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in allowed_suffixes:
                    continue

                code = extract_label_code(path.stem)
                if not code:
                    continue

                source_type = "pdf" if path.suffix.lower() == ".pdf" else "image"
                preview_path = path if source_type == "image" else None
                records.append(
                    TemplateRecord(
                        code=code,
                        variant=self._extract_variant(path.stem, code),
                        display_name=path.stem,
                        source_type=source_type,
                        source_path=path.resolve(),
                        preview_path=preview_path.resolve() if preview_path else None,
                        is_default=path.stem == code,
                    )
                )

        records.sort(
            key=lambda item: (
                item.code,
                not item.is_default,
                item.variant or "",
                item.source_type,
                item.display_name,
            )
        )
        return records

    @staticmethod
    def _extract_variant(stem: str, code: str) -> Optional[str]:
        if stem == code:
            return None

        for separator in ("-", "_"):
            prefix = f"{code}{separator}"
            if stem.startswith(prefix):
                suffix = stem[len(prefix):].strip()
                return suffix or None

        remainder = stem[len(code):].strip("-_")
        return remainder or None

    @staticmethod
    def make_repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(path.resolve())

