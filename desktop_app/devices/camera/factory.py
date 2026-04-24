"""Factory helpers for selecting desktop camera adapters."""

from __future__ import annotations

import os
from pathlib import Path

from desktop_app.devices.camera.base import CameraAdapter, CameraConnectionStatus


_DEFAULT_CAMERA_BACKEND = "hikrobot-mvs"
_DISABLED_MOCK_BACKENDS = {"mock", "mock-camera", "sample", "samples"}


class UnavailableCameraAdapter(CameraAdapter):
    """Camera adapter used when configuration is invalid but the UI should start."""

    def __init__(self, requested_backend: str, reason: str) -> None:
        self.requested_backend = requested_backend
        self.reason = reason

    @property
    def name(self) -> str:
        return f"{self.requested_backend} (unavailable)"

    def check_connection(self) -> CameraConnectionStatus:
        return CameraConnectionStatus(False, self.reason)

    def capture(self, preferred_code: str | None = None) -> Path | None:
        raise RuntimeError(self.reason)


def normalize_camera_backend(raw_backend: str | None = None) -> str:
    """Normalize the requested camera backend name."""

    value = (raw_backend or os.getenv("DESKTOP_CAMERA_BACKEND", _DEFAULT_CAMERA_BACKEND))
    value = value.strip().lower()
    value = value.replace("_", "-")
    normalized = value or _DEFAULT_CAMERA_BACKEND
    if normalized in _DISABLED_MOCK_BACKENDS:
        return _DEFAULT_CAMERA_BACKEND
    return normalized


def build_camera_adapter(raw_backend: str | None = None) -> CameraAdapter:
    """Build the configured camera adapter for the desktop application."""

    backend = normalize_camera_backend(raw_backend)
    if backend in {"hikrobot-mvs", "hikrobot", "mvs"}:
        try:
            from desktop_app.devices.camera.hikrobot_mvs import HikrobotMVSCameraAdapter

            return HikrobotMVSCameraAdapter.from_env()
        except Exception as exc:
            return UnavailableCameraAdapter(
                backend,
                f"相机后端 {backend} 初始化失败: {exc}",
            )

    return UnavailableCameraAdapter(
        backend,
        "不支持的桌面相机后端:"
        f" {backend}. 当前桌面端只支持 hikrobot-mvs 真实相机。",
    )


__all__ = [
    "UnavailableCameraAdapter",
    "build_camera_adapter",
    "normalize_camera_backend",
]
