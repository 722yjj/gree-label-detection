"""Preflight checks for launching the desktop app on the current machine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _add(
    results: list[CheckResult],
    name: str,
    ok: bool,
    detail: str,
    required: bool = True,
    warn: bool = False,
) -> None:
    if ok:
        status = "ok"
    elif warn or not required:
        status = "warn"
    else:
        status = "fail"
    results.append(CheckResult(name=name, status=status, detail=detail, required=required))


def _check_llm_service(results: list[CheckResult], strict_services: bool) -> None:
    try:
        from label_detection.core.config import (
            LLM_PROVIDER,
            OLLAMA_API_BASE,
            OPENAI_COMPATIBLE_API_BASE,
            ensure_local_ollama_no_proxy,
            is_openai_compatible_provider,
        )
    except Exception as exc:
        _add(results, "LLM 配置", False, f"读取配置失败: {exc}", required=strict_services)
        return

    if is_openai_compatible_provider(LLM_PROVIDER):
        service_name = "OpenAI-compatible 服务"
        api_base = OPENAI_COMPATIBLE_API_BASE.rstrip("/") + "/"
        check_url = urljoin(api_base, "models")
    else:
        service_name = "Ollama 服务"
        api_base = OLLAMA_API_BASE.rstrip("/") + "/"
        check_url = urljoin(api_base, "api/tags")

    ensure_local_ollama_no_proxy(api_base)
    request = Request(check_url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            ok = 200 <= response.status < 300
    except (OSError, URLError) as exc:
        _add(
            results,
            service_name,
            False,
            f"未连通 {api_base}: {exc}",
            required=strict_services,
            warn=not strict_services,
        )
        return

    _add(
        results,
        service_name,
        ok,
        f"已连通 {api_base}" if ok else f"返回异常状态: {response.status}",
        required=strict_services,
        warn=not strict_services,
    )


def _check_rapidocr_python(results: list[CheckResult], backend: str) -> None:
    from label_detection.core.config import RAPIDOCR_PYTHON

    python = os.getenv("RAPIDOCR_PYTHON", RAPIDOCR_PYTHON).strip() or sys.executable
    python_path = Path(python)
    if python_path.name != python and not python_path.exists():
        _add(results, "RapidOCR Python", False, f"解释器不存在: {python}", required=True)
        return

    proc = subprocess.run(
        [python, "-c", "import rapidocr"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    detail = f"{backend} 使用 {python}"
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "").strip()
        detail = f"{detail}; rapidocr 导入失败: {error}"
    _add(results, "RapidOCR 依赖", proc.returncode == 0, detail, required=True)


def run_checks(strict_services: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []

    _add(results, "项目目录", PROJECT_ROOT.exists(), str(PROJECT_ROOT), required=True)
    _add(
        results,
        "虚拟环境",
        VENV_PYTHON.exists(),
        str(VENV_PYTHON) if VENV_PYTHON.exists() else "缺少 .venv/bin/python，请先运行 uv sync --extra desktop",
        required=True,
    )
    _add(
        results,
        "当前解释器",
        Path(sys.executable).resolve() == VENV_PYTHON.resolve() if VENV_PYTHON.exists() else False,
        sys.executable,
        required=True,
    )

    for module_name, label in [
        ("desktop_app", "桌面端包"),
        ("label_detection", "检测包"),
        ("PySide6", "PySide6"),
        ("cv2", "OpenCV"),
    ]:
        _add(results, label, _module_available(module_name), module_name, required=True)

    try:
        from label_detection.core.config import OCR_BACKEND, OCR_DEVICE
    except Exception as exc:
        _add(results, "项目配置", False, f"读取配置失败: {exc}", required=True)
    else:
        backend = os.getenv("OCR_BACKEND", OCR_BACKEND).strip().lower() or "paddleocr"
        _add(results, "OCR 后端", True, backend, required=True)
        if backend == "paddleocr":
            _add(results, "PaddleOCR", _module_available("paddleocr"), "paddleocr", required=True)
            _add(results, "PaddlePaddle", _module_available("paddle"), "paddle", required=True)
        elif backend in {"rapidocr-onnxruntime-cpu", "rapidocr-tensorrt"}:
            _check_rapidocr_python(results, backend)
        else:
            _add(results, "OCR 后端", False, f"未知 OCR_BACKEND={backend}", required=True)

        if OCR_DEVICE.startswith("gpu") and _module_available("paddle"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import paddle

                cuda_ok = bool(paddle.is_compiled_with_cuda())
            except Exception as exc:
                _add(results, "Paddle GPU", False, f"检查失败: {exc}", required=False, warn=True)
            else:
                detail = (
                    f"OCR_DEVICE={OCR_DEVICE}, CUDA 可用"
                    if cuda_ok
                    else f"OCR_DEVICE={OCR_DEVICE}, 当前 Paddle 非 CUDA 版本，主流程会回退 CPU"
                )
                _add(
                    results,
                    "Paddle GPU",
                    cuda_ok,
                    detail,
                    required=False,
                    warn=True,
                )

    results_dir = PROJECT_ROOT / "results" / "desktop_app"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        writable = os.access(results_dir, os.W_OK)
    except OSError as exc:
        _add(results, "结果目录", False, f"{results_dir}: {exc}", required=True)
    else:
        _add(results, "结果目录", writable, str(results_dir), required=True)

    _check_llm_service(results, strict_services=strict_services)
    return results


def print_human(results: list[CheckResult]) -> None:
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    width = max(len(item.name) for item in results) if results else 0
    for item in results:
        print(f"[{labels[item.status]}] {item.name:<{width}}  {item.detail}")


def has_failures(results: list[CheckResult]) -> bool:
    return any(item.status == "fail" for item in results)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查桌面端本机启动环境")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出检查结果")
    parser.add_argument(
        "--strict-services",
        action="store_true",
        help="把 LLM 等外部服务不可用也视为失败",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_checks(strict_services=args.strict_services)
    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print_human(results)
    return 1 if has_failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
