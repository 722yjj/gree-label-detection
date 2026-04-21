"""Mock camera adapter backed by repository sample images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from desktop_app.devices.camera.base import CameraAdapter
from label_detection.batch_samples import extract_label_code
from label_detection.core.config import SAMPLES_DIR
from label_detection.extraction.template_source import SUPPORTED_TEMPLATE_IMAGE_SUFFIXES


class MockCameraAdapter(CameraAdapter):
    """Pick a sample target image from the repo to simulate a camera capture."""

    def __init__(self, roots: Optional[Iterable[Path]] = None) -> None:
        self.roots = tuple(roots or (SAMPLES_DIR / "images" / "produce",))

    @property
    def name(self) -> str:
        return "mock-camera"

    def capture(self, preferred_code: Optional[str] = None) -> Optional[Path]:
        images = self._list_images()
        if preferred_code:
            for path in images:
                if extract_label_code(path.stem) == preferred_code:
                    return path
        return images[0] if images else None

    def _list_images(self) -> list[Path]:
        allowed_suffixes = set(SUPPORTED_TEMPLATE_IMAGE_SUFFIXES)
        found: list[Path] = []
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in allowed_suffixes:
                    found.append(path.resolve())
        return found

