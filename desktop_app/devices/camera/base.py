"""Base camera adapter abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CameraConnectionStatus:
    """Result of a lightweight camera availability check."""

    connected: bool
    message: str


class CameraAdapter(ABC):
    """Small adapter interface that hides camera source differences."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @abstractmethod
    def capture(self, preferred_code: Optional[str] = None) -> Optional[Path]:
        """Return a captured image path when available."""

    def check_connection(self) -> CameraConnectionStatus:
        """Return whether this adapter is ready for a capture attempt."""

        return CameraConnectionStatus(True, f"相机已就绪: {self.name}")

    def start_preview(self):
        """Open a long-lived preview session when the adapter supports streaming."""

        raise RuntimeError(f"相机后端不支持实时预览: {self.name}")
