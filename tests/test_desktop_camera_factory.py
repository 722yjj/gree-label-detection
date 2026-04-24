import pytest

from desktop_app.devices.camera.factory import (
    UnavailableCameraAdapter,
    build_camera_adapter,
    normalize_camera_backend,
)
from desktop_app.devices.camera.hikrobot_mvs import HikrobotMVSCameraAdapter


def test_normalize_camera_backend_defaults_to_hikrobot_mvs(monkeypatch):
    monkeypatch.delenv("DESKTOP_CAMERA_BACKEND", raising=False)

    assert normalize_camera_backend() == "hikrobot-mvs"


def test_normalize_camera_backend_maps_mock_to_real_camera():
    assert normalize_camera_backend("mock") == "hikrobot-mvs"


def test_normalize_camera_backend_rewrites_underscores():
    assert normalize_camera_backend("hikrobot_mvs") == "hikrobot-mvs"


def test_build_camera_adapter_returns_hikrobot_for_default(monkeypatch):
    monkeypatch.delenv("DESKTOP_CAMERA_BACKEND", raising=False)

    adapter = build_camera_adapter()

    assert isinstance(adapter, HikrobotMVSCameraAdapter)


def test_build_camera_adapter_returns_unavailable_for_unknown_backend():
    adapter = build_camera_adapter("unknown")

    assert isinstance(adapter, UnavailableCameraAdapter)
    assert adapter.name == "unknown (unavailable)"
    assert adapter.check_connection().connected is False
    with pytest.raises(RuntimeError, match="只支持 hikrobot-mvs"):
        adapter.capture()
