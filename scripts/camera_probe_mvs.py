"""Enumerate HIKROBOT cameras visible to the local MVS runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from desktop_app.devices.camera.hikrobot_mvs import HikrobotMVSError, list_hikrobot_cameras


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="枚举当前机器可见的海康 MVS 相机")
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式输出结果",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cameras = list_hikrobot_cameras()
    except HikrobotMVSError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.json:
        print(
            json.dumps(
                [asdict(camera) for camera in cameras],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"发现相机数量: {len(cameras)}")
    for camera in cameras:
        print(
            f"- idx={camera.index}"
            f", transport={camera.transport_layer}"
            f", model={camera.model_name or '-'}"
            f", serial={camera.serial_number or '-'}"
            f", ip={camera.ip_address or '-'}"
            f", user={camera.user_defined_name or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
