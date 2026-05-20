"""Offline machine fingerprint and license verification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

from label_detection.core.config import DEPLOY_ROOT, PROJECT_ROOT


PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA7AhvWs16BSMA14tZ9kIZ1+ixSdNpHdxeVbuh0IJYi8w=
-----END PUBLIC KEY-----"""

LICENSE_SCHEMA_VERSION = 1
MIN_FINGERPRINT_FIELDS = 3
DEFAULT_LICENSE_PATH = DEPLOY_ROOT / "licenses" / "license.json"


class LicenseError(RuntimeError):
    """Raised when the local license is missing, invalid, or expired."""


@dataclass(frozen=True)
class LicenseStatus:
    ok: bool
    machine_fingerprint: str
    license_path: str
    status: str
    reason: str
    expires_at: str | None = None
    license_id: str | None = None
    customer: str | None = None
    features: list[str] | None = None

    def format_for_user(self) -> str:
        lines = [
            f"授权状态: {self.status}",
            f"机器码: {self.machine_fingerprint}",
            f"授权文件: {self.license_path}",
        ]
        if self.expires_at:
            lines.append(f"到期时间: {self.expires_at}")
        if self.reason:
            lines.append(f"原因: {self.reason}")
        return "\n".join(lines)


def _read_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _run_command(args: Sequence[str], timeout: float = 2.0) -> str | None:
    try:
        proc = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (proc.stdout or proc.stderr or "").strip()
    return output or None


def _clean_identifier(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "",
        "none",
        "unknown",
        "not specified",
        "not available",
        "to be filled by o.e.m.",
        "to be filled by oem",
        "default string",
        "system serial number",
        "base board serial number",
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
    }


def _add_field(fields: dict[str, str], name: str, value: object) -> None:
    cleaned = _clean_identifier(value)
    if not cleaned or _is_placeholder(cleaned):
        return
    fields[name] = cleaned


def _first_existing_text(paths: Sequence[Path]) -> str | None:
    for path in paths:
        text = _read_text(path)
        if text:
            return text
    return None


def _physical_mac_addresses() -> list[str]:
    addresses = []
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        node = uuid.getnode()
        if (node >> 40) % 2 == 0:
            return [":".join(f"{(node >> shift) & 0xff:02x}" for shift in range(40, -1, -8))]
        return []

    for item in sorted(net_root.iterdir()):
        name = item.name
        if name == "lo":
            continue
        address = _read_text(item / "address")
        if not address or address == "00:00:00:00:00:00":
            continue
        if (item / "device").exists() or not (item / "virtual").exists():
            addresses.append(address.lower())
    return sorted(set(addresses))


def _root_disk_serial() -> str | None:
    source = _run_command(["findmnt", "-n", "-o", "SOURCE", "/"])
    if not source:
        return None
    device = source.splitlines()[0].strip()
    device_name = Path(device).name
    if device_name.startswith(("mapper/", "dm-")):
        serial = _run_command(["lsblk", "-ndo", "SERIAL", device])
        return serial.splitlines()[0].strip() if serial else None
    pkname = _run_command(["lsblk", "-ndo", "PKNAME", device])
    block_name = (pkname or device_name).splitlines()[0].strip()
    serial = _run_command(["lsblk", "-ndo", "SERIAL", f"/dev/{block_name}"])
    return serial.splitlines()[0].strip() if serial else None


def collect_machine_fields() -> dict[str, str]:
    """Collect stable local identifiers used to derive a machine fingerprint."""

    fields: dict[str, str] = {}
    _add_field(fields, "machine_id", _first_existing_text([Path("/etc/machine-id")]))
    _add_field(fields, "dmi_product_uuid", _read_text(Path("/sys/class/dmi/id/product_uuid")))
    _add_field(fields, "dmi_board_serial", _read_text(Path("/sys/class/dmi/id/board_serial")))
    _add_field(fields, "dmi_product_serial", _read_text(Path("/sys/class/dmi/id/product_serial")))
    macs = _physical_mac_addresses()
    if macs:
        _add_field(fields, "mac_addresses", ",".join(macs[:4]))
    _add_field(fields, "root_disk_serial", _root_disk_serial())
    _add_field(fields, "platform_machine", platform.machine())
    _add_field(fields, "processor", platform.processor())
    _add_field(fields, "hostname", socket.gethostname())
    return fields


def fingerprint_from_fields(fields: Mapping[str, str]) -> str:
    trusted_fields = {
        key: fields[key]
        for key in sorted(fields)
        if key != "hostname" and _clean_identifier(fields[key])
    }
    if len(trusted_fields) < MIN_FINGERPRINT_FIELDS:
        raise LicenseError(
            f"可用机器码字段不足: {len(trusted_fields)}/{MIN_FINGERPRINT_FIELDS}"
        )
    payload = json.dumps(trusted_fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()
    return "-".join(digest[index : index + 8] for index in range(0, 32, 8))


def machine_fingerprint() -> str:
    return fingerprint_from_fields(collect_machine_fields())


def canonical_license_payload(payload: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sign_license_payload(payload: Mapping[str, Any], private_key_pem: str) -> str:
    key = ECC.import_key(private_key_pem)
    signature = eddsa.new(key, "rfc8032").sign(canonical_license_payload(payload))
    return base64.b64encode(signature).decode("ascii")


def verify_license_signature(payload: Mapping[str, Any], public_key_pem: str | None = None) -> None:
    signature_text = str(payload.get("signature") or "")
    if not signature_text:
        raise LicenseError("许可证缺少 signature 字段")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except ValueError as exc:
        raise LicenseError("许可证 signature 不是有效 Base64") from exc

    key = ECC.import_key(public_key_pem or os.getenv("LICENSE_PUBLIC_KEY_PEM") or PUBLIC_KEY_PEM)
    try:
        eddsa.new(key, "rfc8032").verify(canonical_license_payload(payload), signature)
    except ValueError as exc:
        raise LicenseError("许可证签名无效") from exc


def _parse_datetime(value: object, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise LicenseError(f"许可证缺少 {field_name} 字段")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LicenseError(f"许可证 {field_name} 时间格式无效: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_license(path: Path | str | None = None) -> dict[str, Any]:
    license_path = Path(path or os.getenv("LABEL_DETECTION_LICENSE") or DEFAULT_LICENSE_PATH)
    try:
        with license_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError as exc:
        raise LicenseError(f"许可证文件不存在: {license_path}") from exc
    except json.JSONDecodeError as exc:
        raise LicenseError(f"许可证 JSON 无效: {license_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LicenseError("许可证内容必须是 JSON 对象")
    return payload


def verify_license(path: Path | str | None = None) -> LicenseStatus:
    license_path = Path(path or os.getenv("LABEL_DETECTION_LICENSE") or DEFAULT_LICENSE_PATH)
    try:
        fingerprint = machine_fingerprint()
        payload = load_license(license_path)
        if int(payload.get("version") or 0) != LICENSE_SCHEMA_VERSION:
            raise LicenseError(f"不支持的许可证版本: {payload.get('version')}")
        expected = str(payload.get("machine_fingerprint") or "").strip()
        if expected != fingerprint:
            raise LicenseError("许可证机器码不匹配")
        verify_license_signature(payload)
        expires_at = _parse_datetime(payload.get("expires_at"), "expires_at")
        now = datetime.now(timezone.utc)
        if expires_at < now:
            raise LicenseError(f"许可证已过期: {expires_at.isoformat()}")

        return LicenseStatus(
            ok=True,
            machine_fingerprint=fingerprint,
            license_path=str(license_path),
            status="valid",
            reason="授权有效",
            expires_at=expires_at.isoformat(),
            license_id=str(payload.get("license_id") or ""),
            customer=str(payload.get("customer") or ""),
            features=list(payload.get("features") or []),
        )
    except LicenseError as exc:
        return LicenseStatus(
            ok=False,
            machine_fingerprint=locals().get("fingerprint", ""),
            license_path=str(license_path),
            status="invalid",
            reason=str(exc),
        )


def require_valid_license(path: Path | str | None = None) -> LicenseStatus:
    if os.getenv("LABEL_DETECTION_LICENSE_BYPASS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            fingerprint = machine_fingerprint()
        except LicenseError:
            fingerprint = ""
        return LicenseStatus(
            ok=True,
            machine_fingerprint=fingerprint,
            license_path=str(path or os.getenv("LABEL_DETECTION_LICENSE") or DEFAULT_LICENSE_PATH),
            status="bypassed",
            reason="LABEL_DETECTION_LICENSE_BYPASS 已启用",
        )

    status = verify_license(path)
    if not status.ok:
        raise LicenseError(status.format_for_user())
    return status


def build_license_payload(
    *,
    machine: str,
    customer: str,
    issued_at: datetime,
    expires_at: datetime,
    license_id: str,
    features: Sequence[str],
) -> dict[str, Any]:
    return {
        "version": LICENSE_SCHEMA_VERSION,
        "license_id": license_id,
        "customer": customer,
        "machine_fingerprint": machine,
        "issued_at": issued_at.astimezone(timezone.utc).isoformat(),
        "expires_at": expires_at.astimezone(timezone.utc).isoformat(),
        "features": list(features),
    }


def _print_fingerprint(json_output: bool) -> int:
    fields = collect_machine_fields()
    fingerprint = fingerprint_from_fields(fields)
    if json_output:
        print(
            json.dumps(
                {
                    "machine_fingerprint": fingerprint,
                    "field_count": len([key for key in fields if key != "hostname"]),
                    "fields": fields,
                    "project_root": str(PROJECT_ROOT),
                    "deploy_root": str(DEPLOY_ROOT),
                    "license_path": str(DEFAULT_LICENSE_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(fingerprint)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线授权与机器码工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint_parser = subparsers.add_parser("fingerprint", help="打印当前机器码")
    fingerprint_parser.add_argument("--json", action="store_true", help="输出机器码采集详情")

    verify_parser = subparsers.add_parser("verify", help="验证离线授权文件")
    verify_parser.add_argument(
        "--license",
        dest="license_path",
        default=str(DEFAULT_LICENSE_PATH),
        help="授权文件路径，默认 licenses/license.json",
    )
    verify_parser.add_argument("--json", action="store_true", help="以 JSON 输出验证结果")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "fingerprint":
            return _print_fingerprint(json_output=args.json)
        if args.command == "verify":
            status = verify_license(args.license_path)
            if args.json:
                print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
            else:
                print(status.format_for_user())
            return 0 if status.ok else 1
    except LicenseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
