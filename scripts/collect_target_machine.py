#!/usr/bin/env python3
"""Collect deployment-relevant hardware and OS information from a target Linux host."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from label_detection.license import LicenseError, collect_machine_fields, fingerprint_from_fields


def _run(args: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _python_info() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "compatible_with_project": sys.version_info[:2] == (3, 12),
    }


def collect() -> dict[str, Any]:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "nvcc": ["nvcc", "--version"],
        "lsblk": ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,MODEL,SERIAL,MOUNTPOINTS"],
        "lscpu": ["lscpu"],
        "free": ["free", "-h"],
        "v4l2_devices": ["v4l2-ctl", "--list-devices"],
        "usb_devices": ["lsusb"],
        "input_devices": ["find", "/dev/input", "-maxdepth", "1", "-print"],
    }

    command_results = {}
    for name, args in commands.items():
        if shutil.which(args[0]) is None and args[0] not in {"find"}:
            command_results[name] = {"available": False, "error": f"command not found: {args[0]}"}
            continue
        command_results[name] = _run(args)

    fingerprint_fields = collect_machine_fields()
    try:
        fingerprint = fingerprint_from_fields(fingerprint_fields)
        fingerprint_error = None
    except LicenseError as exc:
        fingerprint = None
        fingerprint_error = str(exc)

    return {
        "machine_fingerprint": fingerprint,
        "fingerprint_error": fingerprint_error,
        "fingerprint_fields": fingerprint_fields,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": _python_info(),
            "os_release": _read_os_release(),
        },
        "commands": command_results,
    }


def main() -> int:
    print(json.dumps(collect(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
