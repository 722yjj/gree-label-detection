#!/usr/bin/env python3
"""Issue an offline Ed25519 license for a collected machine fingerprint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from label_detection.license import build_license_payload, sign_license_payload


def _read_private_key(args: argparse.Namespace) -> str:
    if args.private_key:
        return Path(args.private_key).read_text(encoding="utf-8")
    if args.private_key_env:
        value = os.getenv(args.private_key_env)
        if not value:
            raise RuntimeError(f"环境变量未设置: {args.private_key_env}")
        return value.replace("\\n", "\n")
    raise RuntimeError("请通过 --private-key 或 LICENSE_PRIVATE_KEY_PEM 提供签发私钥")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="签发离线试用/正式授权")
    parser.add_argument("--machine", required=True, help="目标机机器码")
    parser.add_argument("--customer", required=True, help="客户名称")
    parser.add_argument("--days", type=int, default=30, help="有效天数，默认 30")
    parser.add_argument(
        "--feature",
        action="append",
        default=None,
        help="授权功能，可重复。默认 desktop/cli/batch",
    )
    parser.add_argument("--license-id", default="", help="授权 ID；默认自动生成")
    parser.add_argument("--output", default="licenses/license.json", help="输出授权文件路径")
    parser.add_argument("--private-key", help="Ed25519 PEM 私钥文件，仅签发机保存")
    parser.add_argument(
        "--private-key-env",
        default="LICENSE_PRIVATE_KEY_PEM",
        help="读取 PEM 私钥的环境变量，默认 LICENSE_PRIVATE_KEY_PEM",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        private_key = _read_private_key(args)
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(days=args.days)
        payload = build_license_payload(
            machine=args.machine.strip(),
            customer=args.customer.strip(),
            issued_at=issued_at,
            expires_at=expires_at,
            license_id=args.license_id.strip() or f"trial-{uuid4().hex[:12]}",
            features=args.feature or ["desktop", "cli", "batch"],
        )
        payload["signature"] = sign_license_payload(payload, private_key)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"签发失败: {exc}", file=sys.stderr)
        return 1

    print(f"已签发授权: {output_path}")
    print(f"到期时间: {payload['expires_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
