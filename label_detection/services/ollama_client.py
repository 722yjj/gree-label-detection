"""Small native Ollama HTTP client used by deterministic extraction flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from label_detection.core.config import (
    LLM_TIMEOUT,
    OLLAMA_API_BASE,
    OLLAMA_KEEP_ALIVE,
    TEXT_LLM_MODEL,
    TEXT_NUM_PREDICT,
    ensure_local_ollama_no_proxy,
    is_local_ollama,
)


@dataclass(frozen=True)
class OllamaChatResponse:
    """Normalized response from Ollama's /api/chat endpoint."""

    content: str
    raw: Dict[str, Any]
    model: str
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None

    def debug_summary(self) -> str:
        parts = []
        if self.done_reason:
            parts.append(f"done_reason={self.done_reason}")
        if self.total_duration is not None:
            parts.append(f"total={self.total_duration / 1_000_000_000:.2f}s")
        if self.eval_count is not None:
            parts.append(f"eval_count={self.eval_count}")
        if self.prompt_eval_count is not None:
            parts.append(f"prompt_eval_count={self.prompt_eval_count}")
        return ", ".join(parts) if parts else "no metadata"


class OllamaHTTPClient:
    """Direct Ollama client with explicit JSON mode, timeout, and proxy control."""

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        timeout: int | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        check_model: bool = True,
    ):
        self.model_name = model_name or TEXT_LLM_MODEL
        self.api_base = (api_base or OLLAMA_API_BASE).rstrip("/")
        self.timeout = timeout or LLM_TIMEOUT
        self.num_predict = num_predict or TEXT_NUM_PREDICT
        self.keep_alive = keep_alive or OLLAMA_KEEP_ALIVE

        ensure_local_ollama_no_proxy(self.api_base)
        self.session = requests.Session()
        if is_local_ollama(self.api_base):
            self.session.trust_env = False

        if check_model:
            self._check_model()

    def _check_model(self) -> bool:
        """Check whether the configured model exists in Ollama."""
        try:
            response = self.session.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [item.get("name", "") for item in models]
            if any(self.model_name in name for name in names):
                print(f"[LLM] Ollama 模型就绪: {self.model_name}")
                return True

            print(f"[LLM] 警告: 模型 {self.model_name} 未找到")
            print(f"[LLM] 可用模型: {names}")
            print(f"[LLM] 请运行: ollama pull {self.model_name}")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[LLM] 警告: 无法连接到 Ollama ({self.api_base})")
            print("[LLM] 请确保 Ollama 正在运行: ollama serve")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"[LLM] 警告: 检查模型失败: {exc}")
            return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        json_mode: bool = False,
        think: bool = False,
        timeout: int | None = None,
        num_predict: int | None = None,
    ) -> OllamaChatResponse:
        """Send a non-streaming chat request to Ollama."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": num_predict or self.num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"

        response = self.session.post(
            f"{self.api_base}/api/chat",
            json=payload,
            timeout=timeout or self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        message = raw.get("message", {})
        content = str(message.get("content") or "")

        return OllamaChatResponse(
            content=content,
            raw=raw,
            model=str(raw.get("model") or self.model_name),
            done_reason=raw.get("done_reason"),
            total_duration=raw.get("total_duration"),
            load_duration=raw.get("load_duration"),
            prompt_eval_count=raw.get("prompt_eval_count"),
            eval_count=raw.get("eval_count"),
        )
