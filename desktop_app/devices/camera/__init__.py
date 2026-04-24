"""Camera adapter implementations."""

from desktop_app.devices.camera.base import CameraAdapter
from desktop_app.devices.camera.mock_camera import MockCameraAdapter

__all__ = ["CameraAdapter", "MockCameraAdapter"]
