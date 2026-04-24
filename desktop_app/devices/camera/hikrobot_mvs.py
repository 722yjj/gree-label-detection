"""HIKROBOT MVS camera adapter and reusable capture helpers."""

from __future__ import annotations

import importlib
import os
import platform
import re
import sys
import threading
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from desktop_app.devices.camera.base import CameraAdapter, CameraConnectionStatus
from label_detection.core.config import PROJECT_ROOT


_DEFAULT_MVS_ROOT = Path("/opt/MVS")
_DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "desktop_app" / "captures"
_SAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z._-]+")
_MVS_LOCK = threading.Lock()


class HikrobotMVSError(RuntimeError):
    """Raised when the HIKROBOT MVS runtime cannot complete an operation."""


@dataclass(frozen=True)
class HikrobotCameraInfo:
    """Serializable summary of one camera returned by the MVS SDK."""

    index: int
    transport_layer: str
    model_name: str
    serial_number: str
    user_defined_name: str
    vendor_name: str = ""
    ip_address: str | None = None

    @property
    def display_name(self) -> str:
        parts = [self.model_name or f"device-{self.index}"]
        if self.serial_number:
            parts.append(self.serial_number)
        elif self.ip_address:
            parts.append(self.ip_address)
        return " / ".join(parts)


@dataclass(frozen=True)
class HikrobotMVSConfig:
    """Configurable capture settings for the single-frame MVS workflow."""

    serial_number: str | None = None
    ip_address: str | None = None
    model_name: str | None = None
    device_index: int = 0
    timeout_ms: int = 1500
    trigger_mode: Literal["continuous", "software"] = "continuous"
    exposure_time_us: float | None = None
    gain: float | None = None
    user_set: str | None = None

    @classmethod
    def from_env(cls) -> "HikrobotMVSConfig":
        return cls(
            serial_number=_get_env_string("HIKROBOT_CAMERA_SERIAL"),
            ip_address=_get_env_string("HIKROBOT_CAMERA_IP"),
            model_name=_get_env_string("HIKROBOT_CAMERA_MODEL"),
            device_index=_get_env_int("HIKROBOT_CAMERA_INDEX", 0),
            timeout_ms=_get_env_int("HIKROBOT_CAMERA_TIMEOUT_MS", 1500),
            trigger_mode=_normalize_trigger_mode(
                _get_env_string("HIKROBOT_CAMERA_TRIGGER_MODE") or "continuous"
            ),
            exposure_time_us=_get_env_float("HIKROBOT_CAMERA_EXPOSURE_US"),
            gain=_get_env_float("HIKROBOT_CAMERA_GAIN"),
            user_set=_get_env_string("HIKROBOT_CAMERA_USER_SET"),
        )


class HikrobotMVSCameraAdapter(CameraAdapter):
    """Camera adapter that grabs one frame through the HIKROBOT MVS SDK."""

    def __init__(
        self,
        *,
        output_root: Path | None = None,
        config: HikrobotMVSConfig | None = None,
    ) -> None:
        self.output_root = Path(output_root or _DEFAULT_OUTPUT_ROOT)
        self.config = config or HikrobotMVSConfig.from_env()

    @classmethod
    def from_env(cls) -> "HikrobotMVSCameraAdapter":
        return cls()

    @property
    def name(self) -> str:
        return "hikrobot-mvs"

    def check_connection(self) -> CameraConnectionStatus:
        try:
            selected = select_camera(list_hikrobot_cameras(), self.config)
        except HikrobotMVSError as exc:
            return CameraConnectionStatus(False, f"未检测到相机: {exc}")

        return CameraConnectionStatus(True, f"已连接相机: {selected.display_name}")

    def capture(self, preferred_code: str | None = None) -> Path | None:
        output_path = build_capture_output_path(self.output_root, preferred_code)
        return capture_hikrobot_image(output_path, config=self.config)

    def start_preview(self) -> "HikrobotMVSPreviewSession":
        return HikrobotMVSPreviewSession(
            output_root=self.output_root,
            config=self.config,
        )


class HikrobotMVSPreviewSession:
    """Long-lived MVS camera session for desktop live preview."""

    def __init__(self, *, output_root: Path, config: HikrobotMVSConfig) -> None:
        self.output_root = Path(output_root)
        self.config = config
        self._runtime = _MVSRuntime()
        self._sdk = None
        self._camera = None
        self._created = False
        self._opened = False
        self._grabbing = False
        self._frame_buffer = None
        self._frame_buffer_size = 0
        self._frame_info = None
        self.display_name = ""
        self._open()

    def _open(self) -> None:
        self._sdk = self._runtime.__enter__()
        try:
            device_list = _enumerate_devices(self._sdk)
            cameras: list[HikrobotCameraInfo] = []
            device_infos = []
            for index in range(device_list.nDeviceNum):
                device_info = cast(
                    device_list.pDeviceInfo[index],
                    POINTER(self._sdk.MV_CC_DEVICE_INFO),
                ).contents
                cameras.append(_camera_info_from_device(self._sdk, device_info, index))
                device_infos.append(device_info)

            selected = select_camera(cameras, self.config)
            self.display_name = selected.display_name
            device_info = device_infos[selected.index]
            self._camera = self._sdk.MvCamera()

            _check_ret(
                self._camera.MV_CC_CreateHandle(device_info),
                "创建相机句柄失败",
            )
            self._created = True

            _check_ret(
                self._camera.MV_CC_OpenDevice(self._sdk.MV_ACCESS_Exclusive, 0),
                f"打开相机失败: {selected.display_name}",
            )
            self._opened = True

            if selected.transport_layer == "GigE":
                _configure_gige_packet_size(self._camera)

            if self.config.user_set:
                _load_user_set(self._camera, self.config.user_set)
            _apply_manual_capture_settings(self._camera, self.config)
            _configure_trigger(self._camera, self.config.trigger_mode)

            width = _get_int_node(self._camera, self._sdk, "Width")
            height = _get_int_node(self._camera, self._sdk, "Height")
            self._frame_buffer_size = max(width * height * 3, 1)
            self._frame_buffer = (c_ubyte * self._frame_buffer_size)()
            self._frame_info = self._sdk.MV_FRAME_OUT_INFO_EX()
            memset(byref(self._frame_info), 0, sizeof(self._frame_info))

            _check_ret(self._camera.MV_CC_StartGrabbing(), "开始预览取流失败")
            self._grabbing = True
        except Exception:
            self.close()
            raise

    def read_frame_bgr(self):
        if self._camera is None or self._frame_buffer is None or self._frame_info is None:
            raise HikrobotMVSError("相机预览未启动")

        if self.config.trigger_mode == "software":
            _check_ret(
                self._camera.MV_CC_SetCommandValue("TriggerSoftware"),
                "发送软触发失败",
            )

        _check_ret(
            self._camera.MV_CC_GetImageForBGR(
                self._frame_buffer,
                self._frame_buffer_size,
                self._frame_info,
                self.config.timeout_ms,
            ),
            "获取预览图像失败",
        )

        return np.frombuffer(
            self._frame_buffer,
            dtype=np.uint8,
            count=self._frame_info.nWidth * self._frame_info.nHeight * 3,
        ).reshape(self._frame_info.nHeight, self._frame_info.nWidth, 3).copy()

    def save_frame_bgr(self, frame, preferred_code: str | None = None) -> Path:
        output_path = build_capture_output_path(self.output_root, preferred_code)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise HikrobotMVSError(f"写入图片失败: {output_path}")
        return output_path.resolve()

    def close(self) -> None:
        try:
            if self._camera is not None and self._grabbing:
                self._camera.MV_CC_StopGrabbing()
        finally:
            self._grabbing = False
            try:
                if self._camera is not None and self._opened:
                    self._camera.MV_CC_CloseDevice()
            finally:
                self._opened = False
                try:
                    if self._camera is not None and self._created:
                        self._camera.MV_CC_DestroyHandle()
                finally:
                    self._created = False
                    self._camera = None
                    if self._sdk is not None:
                        self._runtime.__exit__(None, None, None)
                        self._sdk = None


def list_hikrobot_cameras() -> list[HikrobotCameraInfo]:
    """Enumerate currently visible cameras through the MVS runtime."""

    with _MVSRuntime() as sdk:
        device_list = _enumerate_devices(sdk)
        cameras: list[HikrobotCameraInfo] = []
        for index in range(device_list.nDeviceNum):
            device_info = cast(
                device_list.pDeviceInfo[index],
                POINTER(sdk.MV_CC_DEVICE_INFO),
            ).contents
            cameras.append(_camera_info_from_device(sdk, device_info, index))
        return cameras


def select_camera(
    cameras: Sequence[HikrobotCameraInfo],
    config: HikrobotMVSConfig,
) -> HikrobotCameraInfo:
    """Select one enumerated camera according to config filters."""

    if not cameras:
        raise HikrobotMVSError(
            "未枚举到任何海康相机，请检查网线、供电、IP 网段和 MVS 环境。"
        )

    matches = list(cameras)
    if config.serial_number:
        expected = config.serial_number.strip().lower()
        matches = [
            camera
            for camera in matches
            if camera.serial_number.strip().lower() == expected
        ]
    if config.ip_address:
        expected = config.ip_address.strip()
        matches = [
            camera for camera in matches if (camera.ip_address or "").strip() == expected
        ]
    if config.model_name:
        expected = config.model_name.strip().lower()
        matches = [
            camera for camera in matches if camera.model_name.strip().lower() == expected
        ]

    if not matches:
        available = ", ".join(camera.display_name for camera in cameras)
        raise HikrobotMVSError(
            "未找到匹配的海康相机。"
            f" serial={config.serial_number or '-'}"
            f", ip={config.ip_address or '-'}"
            f", model={config.model_name or '-'}。"
            f" 当前可见设备: {available or '无'}"
        )

    if config.device_index < 0 or config.device_index >= len(matches):
        raise HikrobotMVSError(
            f"相机索引超出范围: {config.device_index}。当前匹配设备数: {len(matches)}"
        )
    return matches[config.device_index]


def build_capture_output_path(output_root: Path, preferred_code: str | None) -> Path:
    """Create a stable single-frame output path under the desktop capture tree."""

    code_fragment = _sanitize_filename_fragment(preferred_code or "uncoded")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(output_root) / code_fragment / f"{timestamp}_{code_fragment}.jpg"


def capture_hikrobot_image(
    output_path: Path,
    *,
    config: HikrobotMVSConfig | None = None,
) -> Path:
    """Capture one BGR frame from the selected camera and write it to disk."""

    active_config = config or HikrobotMVSConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _MVSRuntime() as sdk:
        device_list = _enumerate_devices(sdk)
        cameras: list[HikrobotCameraInfo] = []
        device_infos = []
        for index in range(device_list.nDeviceNum):
            device_info = cast(
                device_list.pDeviceInfo[index],
                POINTER(sdk.MV_CC_DEVICE_INFO),
            ).contents
            cameras.append(_camera_info_from_device(sdk, device_info, index))
            device_infos.append(device_info)

        selected = select_camera(cameras, active_config)
        device_info = device_infos[selected.index]
        camera = sdk.MvCamera()
        created = False
        opened = False
        grabbing = False

        try:
            _check_ret(
                camera.MV_CC_CreateHandle(device_info),
                "创建相机句柄失败",
            )
            created = True

            _check_ret(
                camera.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
                f"打开相机失败: {selected.display_name}",
            )
            opened = True

            if selected.transport_layer == "GigE":
                _configure_gige_packet_size(camera)

            if active_config.user_set:
                _load_user_set(camera, active_config.user_set)
            _apply_manual_capture_settings(camera, active_config)
            _configure_trigger(camera, active_config.trigger_mode)

            width = _get_int_node(camera, sdk, "Width")
            height = _get_int_node(camera, sdk, "Height")
            buffer_size = max(width * height * 3, 1)
            frame_buffer = (c_ubyte * buffer_size)()
            frame_info = sdk.MV_FRAME_OUT_INFO_EX()
            memset(byref(frame_info), 0, sizeof(frame_info))

            _check_ret(camera.MV_CC_StartGrabbing(), "开始取流失败")
            grabbing = True

            if active_config.trigger_mode == "software":
                _check_ret(
                    camera.MV_CC_SetCommandValue("TriggerSoftware"),
                    "发送软触发失败",
                )

            _check_ret(
                camera.MV_CC_GetImageForBGR(
                    frame_buffer,
                    buffer_size,
                    frame_info,
                    active_config.timeout_ms,
                ),
                "获取 BGR 图像失败",
            )

            image = np.frombuffer(
                frame_buffer,
                dtype=np.uint8,
                count=frame_info.nWidth * frame_info.nHeight * 3,
            ).reshape(frame_info.nHeight, frame_info.nWidth, 3)

            if not cv2.imwrite(str(output_path), image):
                raise HikrobotMVSError(f"写入图片失败: {output_path}")
            return output_path.resolve()
        finally:
            if grabbing:
                camera.MV_CC_StopGrabbing()
            if opened:
                camera.MV_CC_CloseDevice()
            if created:
                camera.MV_CC_DestroyHandle()


class _MVSRuntime:
    """Serialized init/finalize guard for the process-global MVS runtime."""

    def __init__(self) -> None:
        self.sdk = _import_mvs_sdk()

    def __enter__(self):
        _MVS_LOCK.acquire()
        ret = self.sdk.MvCamera.MV_CC_Initialize()
        if ret != 0:
            _MVS_LOCK.release()
            raise HikrobotMVSError(f"MVS SDK 初始化失败: 0x{ret:x}")
        return self.sdk

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.sdk.MvCamera.MV_CC_Finalize()
        finally:
            _MVS_LOCK.release()


def _enumerate_devices(sdk):
    device_list = sdk.MV_CC_DEVICE_INFO_LIST()
    ret = sdk.MvCamera.MV_CC_EnumDevices(_supported_transport_layer_mask(sdk), device_list)
    _check_ret(ret, "枚举设备失败")
    return device_list


def _supported_transport_layer_mask(sdk) -> int:
    names = (
        "MV_GIGE_DEVICE",
        "MV_USB_DEVICE",
        "MV_GENTL_CAMERALINK_DEVICE",
        "MV_GENTL_CXP_DEVICE",
        "MV_GENTL_XOF_DEVICE",
    )
    mask = 0
    for name in names:
        mask |= int(getattr(sdk, name, 0))
    return mask


def _camera_info_from_device(sdk, device_info, index: int) -> HikrobotCameraInfo:
    gige_types = {
        int(getattr(sdk, "MV_GIGE_DEVICE", 0)),
        int(getattr(sdk, "MV_GENTL_GIGE_DEVICE", 0)),
    }
    tlayer = int(device_info.nTLayerType)
    if tlayer in gige_types:
        info = device_info.SpecialInfo.stGigEInfo
        return HikrobotCameraInfo(
            index=index,
            transport_layer="GigE",
            model_name=_decode_char_array(info.chModelName),
            serial_number=_decode_char_array(info.chSerialNumber),
            user_defined_name=_decode_char_array(info.chUserDefinedName),
            vendor_name=_decode_char_array(info.chManufacturerName),
            ip_address=_format_ipv4(info.nCurrentIp),
        )

    if tlayer == int(getattr(sdk, "MV_USB_DEVICE", 0)):
        info = device_info.SpecialInfo.stUsb3VInfo
        return HikrobotCameraInfo(
            index=index,
            transport_layer="USB3",
            model_name=_decode_char_array(info.chModelName),
            serial_number=_decode_char_array(info.chSerialNumber),
            user_defined_name=_decode_char_array(info.chUserDefinedName),
            vendor_name=_decode_char_array(info.chVendorName),
            ip_address=None,
        )

    return HikrobotCameraInfo(
        index=index,
        transport_layer=f"0x{tlayer:x}",
        model_name=f"unknown-{index}",
        serial_number="",
        user_defined_name="",
        vendor_name="",
        ip_address=None,
    )


def _configure_gige_packet_size(camera) -> None:
    packet_size = int(camera.MV_CC_GetOptimalPacketSize())
    if packet_size > 0:
        camera.MV_CC_SetIntValueEx("GevSCPSPacketSize", packet_size)


def _load_user_set(camera, user_set: str) -> None:
    _check_ret(
        camera.MV_CC_SetEnumValueByString("UserSetSelector", user_set),
        f"设置 UserSetSelector 失败: {user_set}",
    )
    _check_ret(
        camera.MV_CC_SetCommandValue("UserSetLoad"),
        f"加载 UserSet 失败: {user_set}",
    )


def _apply_manual_capture_settings(camera, config: HikrobotMVSConfig) -> None:
    if config.exposure_time_us is not None:
        camera.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
        _check_ret(
            camera.MV_CC_SetFloatValue("ExposureTime", config.exposure_time_us),
            f"设置曝光失败: {config.exposure_time_us}",
        )

    if config.gain is not None:
        camera.MV_CC_SetEnumValueByString("GainAuto", "Off")
        _check_ret(
            camera.MV_CC_SetFloatValue("Gain", config.gain),
            f"设置增益失败: {config.gain}",
        )


def _configure_trigger(
    camera,
    trigger_mode: Literal["continuous", "software"],
) -> None:
    if trigger_mode == "software":
        _check_ret(
            camera.MV_CC_SetEnumValueByString("TriggerMode", "On"),
            "开启触发模式失败",
        )
        _check_ret(
            camera.MV_CC_SetEnumValueByString("TriggerSource", "Software"),
            "设置软触发源失败",
        )
        return

    _check_ret(
        camera.MV_CC_SetEnumValueByString("TriggerMode", "Off"),
        "关闭触发模式失败",
    )


def _get_int_node(camera, sdk, node_name: str) -> int:
    value = sdk.MVCC_INTVALUE_EX()
    memset(byref(value), 0, sizeof(value))
    _check_ret(
        camera.MV_CC_GetIntValueEx(node_name, value),
        f"读取节点失败: {node_name}",
    )
    return int(value.nCurValue)


def _import_mvs_sdk():
    _prepare_mvs_environment()
    python_path = _resolve_mvs_python_path()
    if python_path is None or not python_path.exists():
        raise HikrobotMVSError(
            "未找到 MVS Python SDK。"
            " 请确认 /opt/MVS 已安装，或设置 HIKROBOT_MVS_PYTHON_PATH。"
        )
    if str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))
    try:
        return importlib.import_module("MvCameraControl_class")
    except ModuleNotFoundError as exc:
        raise HikrobotMVSError(
            "导入 MVS Python SDK 失败，请先执行 `source /opt/MVS/bin/set_env_path.sh`。"
        ) from exc


def _prepare_mvs_environment() -> None:
    if not os.getenv("MVCAM_COMMON_RUNENV"):
        default_runenv = _DEFAULT_MVS_ROOT / "lib"
        if default_runenv.exists():
            os.environ["MVCAM_COMMON_RUNENV"] = str(default_runenv)


def _resolve_mvs_python_path() -> Path | None:
    override = _get_env_string("HIKROBOT_MVS_PYTHON_PATH")
    if override:
        return Path(override).expanduser()

    bitness = "64" if platform.architecture()[0] == "64bit" else "32"
    candidate = _DEFAULT_MVS_ROOT / "Samples" / bitness / "Python" / "MvImport"
    if candidate.exists():
        return candidate
    return None


def _decode_char_array(char_array) -> str:
    raw = memoryview(char_array).tobytes()
    null_index = raw.find(b"\x00")
    if null_index != -1:
        raw = raw[:null_index]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _format_ipv4(ip_value: int) -> str:
    return ".".join(str((int(ip_value) >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _check_ret(ret: int, message: str) -> None:
    if ret != 0:
        raise HikrobotMVSError(f"{message}: 0x{int(ret):x}")


def _normalize_trigger_mode(value: str) -> Literal["continuous", "software"]:
    normalized = (value or "").strip().lower()
    if normalized in {"continuous", "free-run", "free_run"}:
        return "continuous"
    if normalized in {"software", "soft", "soft-trigger", "soft_trigger"}:
        return "software"
    raise ValueError(f"不支持的 HIKROBOT_CAMERA_TRIGGER_MODE: {value}")


def _sanitize_filename_fragment(value: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("-", value.strip())
    cleaned = cleaned.strip("-.")
    return cleaned or "capture"


def _get_env_string(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _get_env_int(name: str, default: int) -> int:
    value = _get_env_string(name)
    return int(value) if value is not None else default


def _get_env_float(name: str) -> float | None:
    value = _get_env_string(name)
    return float(value) if value is not None else None


__all__ = [
    "HikrobotCameraInfo",
    "HikrobotMVSCameraAdapter",
    "HikrobotMVSConfig",
    "HikrobotMVSError",
    "HikrobotMVSPreviewSession",
    "build_capture_output_path",
    "capture_hikrobot_image",
    "list_hikrobot_cameras",
    "select_camera",
]
