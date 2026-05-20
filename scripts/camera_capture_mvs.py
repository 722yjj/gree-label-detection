"""Capture one image from a HIKROBOT camera through the MVS SDK."""

from __future__ import annotations

import argparse
from pathlib import Path

from desktop_app.devices.camera.hikrobot_mvs import (
    HikrobotMVSConfig,
    HikrobotMVSError,
    build_capture_output_path,
    capture_hikrobot_image,
)
from label_detection.core.config import RESULTS_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用海康 MVS SDK 抓取单张图片")
    parser.add_argument("--output", help="输出图片路径，默认为 results/manual_camera_capture 下自动命名")
    parser.add_argument("--code", help="用于输出目录和文件名的编码前缀")
    parser.add_argument("--serial", help="按序列号选择相机")
    parser.add_argument("--ip", help="按相机 IP 选择相机")
    parser.add_argument("--model", help="按型号选择相机，例如 MV-CU060-10GC")
    parser.add_argument("--index", type=int, default=0, help="匹配结果中的相机索引，默认 0")
    parser.add_argument(
        "--trigger-mode",
        choices=("continuous", "software"),
        default="continuous",
        help="continuous 为自由运行取图，software 为软触发取图",
    )
    parser.add_argument("--timeout-ms", type=int, default=1500, help="抓图超时，单位毫秒")
    parser.add_argument("--exposure-us", type=float, help="曝光时间，单位微秒")
    parser.add_argument("--gain", type=float, help="增益")
    parser.add_argument("--user-set", help="先加载指定 UserSet，例如 UserSet1")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = HikrobotMVSConfig(
        serial_number=args.serial,
        ip_address=args.ip,
        model_name=args.model,
        device_index=args.index,
        timeout_ms=args.timeout_ms,
        trigger_mode=args.trigger_mode,
        exposure_time_us=args.exposure_us,
        gain=args.gain,
        user_set=args.user_set,
    )

    output_path = (
        Path(args.output)
        if args.output
        else build_capture_output_path(
            RESULTS_ROOT / "manual_camera_capture",
            args.code,
        )
    )
    try:
        saved_path = capture_hikrobot_image(output_path, config=config)
    except HikrobotMVSError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"saved: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
