import os
from pathlib import Path
from datetime import datetime as real_datetime

import pytest

from desktop_app.devices.camera import hikrobot_mvs as hikrobot_mvs_module
from desktop_app.devices.camera.hikrobot_mvs import (
    HikrobotCameraInfo,
    HikrobotMVSConfig,
    HikrobotMVSCameraAdapter,
    HikrobotMVSError,
    build_capture_output_path,
    select_camera,
)


def test_select_camera_filters_by_model_and_ip():
    cameras = [
        HikrobotCameraInfo(
            index=0,
            transport_layer="GigE",
            model_name="MV-CU060-10GC",
            serial_number="A001",
            user_defined_name="line-a",
            ip_address="192.168.1.10",
        ),
        HikrobotCameraInfo(
            index=1,
            transport_layer="GigE",
            model_name="MV-CU060-10GC",
            serial_number="A002",
            user_defined_name="line-b",
            ip_address="192.168.1.11",
        ),
    ]

    selected = select_camera(
        cameras,
        HikrobotMVSConfig(
            model_name="MV-CU060-10GC",
            ip_address="192.168.1.11",
        ),
    )

    assert selected.serial_number == "A002"


def test_select_camera_uses_filtered_index():
    cameras = [
        HikrobotCameraInfo(
            index=0,
            transport_layer="GigE",
            model_name="MV-CU060-10GC",
            serial_number="A001",
            user_defined_name="line-a",
            ip_address="192.168.1.10",
        ),
        HikrobotCameraInfo(
            index=1,
            transport_layer="GigE",
            model_name="MV-CU060-10GC",
            serial_number="A002",
            user_defined_name="line-b",
            ip_address="192.168.1.11",
        ),
        HikrobotCameraInfo(
            index=2,
            transport_layer="GigE",
            model_name="MV-CU060-10GM",
            serial_number="M001",
            user_defined_name="line-c",
            ip_address="192.168.1.12",
        ),
    ]

    selected = select_camera(
        cameras,
        HikrobotMVSConfig(model_name="MV-CU060-10GC", device_index=1),
    )

    assert selected.serial_number == "A002"


def test_select_camera_raises_for_missing_match():
    cameras = [
        HikrobotCameraInfo(
            index=0,
            transport_layer="GigE",
            model_name="MV-CU060-10GC",
            serial_number="A001",
            user_defined_name="line-a",
            ip_address="192.168.1.10",
        )
    ]

    with pytest.raises(HikrobotMVSError):
        select_camera(cameras, HikrobotMVSConfig(serial_number="missing"))


def test_build_capture_output_path_uses_sanitized_code(tmp_path: Path):
    output_path = build_capture_output_path(tmp_path, "600004075219/01")

    assert output_path.parent.name == "600004075219-01"
    assert output_path.name.endswith("_600004075219-01.jpg")


def test_build_capture_output_path_includes_microseconds(tmp_path: Path, monkeypatch):
    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 4, 24, 14, 49, 25, 123456)

    monkeypatch.setattr(hikrobot_mvs_module, "datetime", FixedDatetime)

    output_path = build_capture_output_path(tmp_path, "600004075219")

    assert output_path.name == "20260424-144925-123456_600004075219.jpg"


def test_config_from_env_reads_capture_settings(monkeypatch):
    monkeypatch.setenv("HIKROBOT_CAMERA_SERIAL", "SN123")
    monkeypatch.setenv("HIKROBOT_CAMERA_INDEX", "2")
    monkeypatch.setenv("HIKROBOT_CAMERA_TIMEOUT_MS", "2200")
    monkeypatch.setenv("HIKROBOT_CAMERA_TRIGGER_MODE", "software")
    monkeypatch.setenv("HIKROBOT_CAMERA_EXPOSURE_US", "1800.5")
    monkeypatch.setenv("HIKROBOT_CAMERA_GAIN", "6.5")
    monkeypatch.setenv("HIKROBOT_CAMERA_EXPOSURE_AUTO", "once")
    monkeypatch.setenv("HIKROBOT_CAMERA_GAIN_AUTO", "continuous")
    monkeypatch.setenv("HIKROBOT_CAMERA_USER_SET", "UserSet1")

    config = HikrobotMVSConfig.from_env()

    assert config.serial_number == "SN123"
    assert config.device_index == 2
    assert config.timeout_ms == 2200
    assert config.trigger_mode == "software"
    assert config.exposure_time_us == 1800.5
    assert config.gain == 6.5
    assert config.exposure_auto == "once"
    assert config.gain_auto == "continuous"
    assert config.user_set == "UserSet1"


def test_config_from_env_defaults_to_continuous_auto_exposure(monkeypatch):
    monkeypatch.delenv("HIKROBOT_CAMERA_EXPOSURE_US", raising=False)
    monkeypatch.delenv("HIKROBOT_CAMERA_EXPOSURE_AUTO", raising=False)
    monkeypatch.delenv("HIKROBOT_CAMERA_GAIN", raising=False)
    monkeypatch.delenv("HIKROBOT_CAMERA_GAIN_AUTO", raising=False)

    config = HikrobotMVSConfig.from_env()

    assert config.exposure_auto == "continuous"
    assert config.gain_auto is None


def test_resolve_mvs_python_path_supports_aarch64_layout(tmp_path: Path, monkeypatch):
    mvs_root = tmp_path / "MVS"
    mv_import = mvs_root / "Samples" / "aarch64" / "Python" / "MvImport"
    mv_import.mkdir(parents=True)

    monkeypatch.delenv("HIKROBOT_MVS_PYTHON_PATH", raising=False)
    monkeypatch.setattr(hikrobot_mvs_module, "_DEFAULT_MVS_ROOT", mvs_root)
    monkeypatch.setattr(hikrobot_mvs_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        hikrobot_mvs_module.platform,
        "architecture",
        lambda: ("64bit", ""),
    )

    assert hikrobot_mvs_module._resolve_mvs_python_path() == mv_import


def test_prepare_mvs_environment_sets_default_paths(tmp_path: Path, monkeypatch):
    mvs_root = tmp_path / "MVS"
    (mvs_root / "lib").mkdir(parents=True)

    monkeypatch.setattr(hikrobot_mvs_module, "_DEFAULT_MVS_ROOT", mvs_root)
    monkeypatch.delenv("MVCAM_SDK_PATH", raising=False)
    monkeypatch.delenv("MVCAM_COMMON_RUNENV", raising=False)
    monkeypatch.delenv("MVCAM_GENICAM_CLPROTOCOL", raising=False)
    monkeypatch.delenv("ALLUSERSPROFILE", raising=False)

    hikrobot_mvs_module._prepare_mvs_environment()

    assert os.environ["MVCAM_SDK_PATH"] == str(mvs_root)
    assert os.environ["MVCAM_COMMON_RUNENV"] == str(mvs_root / "lib")
    assert os.environ["MVCAM_GENICAM_CLPROTOCOL"] == str(
        mvs_root / "lib" / "CLProtocol"
    )
    assert os.environ["ALLUSERSPROFILE"] == str(mvs_root / "MVFG")


def test_adapter_capture_uses_helper_and_returns_saved_path(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_capture(output_path: Path, *, config: HikrobotMVSConfig) -> Path:
        captured["output_path"] = output_path
        captured["config"] = config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        return output_path.resolve()

    monkeypatch.setattr(hikrobot_mvs_module, "capture_hikrobot_image", fake_capture)

    adapter = HikrobotMVSCameraAdapter(
        output_root=tmp_path,
        config=HikrobotMVSConfig(model_name="MV-CU060-10GC"),
    )
    saved_path = adapter.capture(preferred_code="600004075219")

    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.parent.name == "600004075219"
    assert captured["config"].model_name == "MV-CU060-10GC"
