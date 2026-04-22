"""Helpers for explicit Paddle device selection with safe fallback."""

from __future__ import annotations

from contextlib import contextmanager
import os
from functools import lru_cache

from label_detection.core.config import (
    PADDLE_DEVICE_REQUIRED,
    PADDLE_DISABLE_MODEL_SOURCE_CHECK,
    PADDLE_EMPTY_CACHE_AFTER_RUN,
)


if PADDLE_DISABLE_MODEL_SOURCE_CHECK:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _wants_gpu(device: str) -> bool:
    return str(device or "").strip().lower().startswith("gpu")


@lru_cache(maxsize=None)
def resolve_paddle_device(requested_device: str | None, component: str = "Paddle") -> str:
    """Return a usable Paddle device string, downgrading to CPU when needed."""
    requested = str(requested_device or "gpu:0").strip() or "gpu:0"
    if not _wants_gpu(requested):
        return requested

    try:
        import paddle
    except Exception as exc:  # pragma: no cover - import failures depend on host env
        if PADDLE_DEVICE_REQUIRED:
            raise RuntimeError(
                f"{component} 需要 GPU，但 Paddle 导入失败: {exc}"
            ) from exc
        print(f"[Paddle] {component} 无法导入 Paddle，回退到 CPU: {exc}")
        return "cpu"

    if not paddle.is_compiled_with_cuda():
        message = f"{component} 请求使用 {requested}，但当前 Paddle 不是 CUDA 版本"
        if PADDLE_DEVICE_REQUIRED:
            raise RuntimeError(message)
        print(f"[Paddle] {message}，回退到 CPU")
        return "cpu"

    try:
        device_count = int(paddle.device.cuda.device_count())
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        if PADDLE_DEVICE_REQUIRED:
            raise RuntimeError(
                f"{component} 检查 GPU 数量失败: {exc}"
            ) from exc
        print(f"[Paddle] {component} 检查 GPU 数量失败，回退到 CPU: {exc}")
        return "cpu"

    if device_count < 1:
        message = f"{component} 请求使用 {requested}，但未检测到可用 GPU"
        if PADDLE_DEVICE_REQUIRED:
            raise RuntimeError(message)
        print(f"[Paddle] {message}，回退到 CPU")
        return "cpu"

    return requested


def empty_paddle_cache(reason: str = "workflow") -> None:
    """Release cached Paddle GPU blocks back to the driver after a run."""
    if not PADDLE_EMPTY_CACHE_AFTER_RUN:
        return

    try:
        import paddle
    except Exception:
        return

    try:
        current_device = str(paddle.get_device() or "").strip().lower()
    except Exception:
        current_device = ""

    if not current_device.startswith("gpu"):
        return

    try:
        before_reserved = paddle.device.cuda.memory_reserved()
        before_allocated = paddle.device.cuda.memory_allocated()
        paddle.device.cuda.empty_cache()
        after_reserved = paddle.device.cuda.memory_reserved()
        after_allocated = paddle.device.cuda.memory_allocated()
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        print(f"[Paddle] 清理 GPU 缓存失败 ({reason}): {exc}")
        return

    mib = 1024 * 1024
    print(
        "[Paddle] 已清理 GPU 缓存 "
        f"({reason}): reserved {before_reserved / mib:.1f}MB -> "
        f"{after_reserved / mib:.1f}MB, allocated {before_allocated / mib:.1f}MB -> "
        f"{after_allocated / mib:.1f}MB"
    )


@contextmanager
def paddle_cache_cleanup_scope(reason: str = "workflow"):
    """Ensure Paddle cached GPU memory is released after the wrapped workflow exits."""
    try:
        yield
    finally:
        empty_paddle_cache(reason=reason)
