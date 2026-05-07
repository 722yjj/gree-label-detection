"""HTTP client for OpenAI-compatible chat completion servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from label_detection.core.config import (
    LLM_TIMEOUT,
    OPENAI_COMPATIBLE_API_BASE,
    OPENAI_COMPATIBLE_API_KEY,
    OPENAI_COMPATIBLE_DISABLE_THINKING,
    OPENAI_COMPATIBLE_MODEL,
    TEXT_NUM_PREDICT,
    ensure_local_ollama_no_proxy,
    is_local_ollama,
)


@dataclass(frozen=True)
class OpenAICompatibleChatResponse:
    """Normalized response from an OpenAI-compatible chat completion endpoint."""

    content: str
    raw: Dict[str, Any]
    model: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def debug_summary(self) -> str:
        parts = []
        if self.finish_reason:
            parts.append(f"finish_reason={self.finish_reason}")
        if self.completion_tokens is not None:
            parts.append(f"completion_tokens={self.completion_tokens}")
        if self.prompt_tokens is not None:
            parts.append(f"prompt_tokens={self.prompt_tokens}")
        if self.total_tokens is not None:
            parts.append(f"total_tokens={self.total_tokens}")
        return ", ".join(parts) if parts else "no metadata"


class OpenAICompatibleHTTPClient:
    """Small non-streaming client for vLLM and other OpenAI-compatible servers."""

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        num_predict: int | None = None,
        disable_thinking: bool | None = None,
        check_model: bool = True,
    ):
        self.model_name = model_name or OPENAI_COMPATIBLE_MODEL
        self.api_base = (api_base or OPENAI_COMPATIBLE_API_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else OPENAI_COMPATIBLE_API_KEY
        self.timeout = timeout or LLM_TIMEOUT
        self.num_predict = num_predict or TEXT_NUM_PREDICT
        self.disable_thinking = (
            OPENAI_COMPATIBLE_DISABLE_THINKING
            if disable_thinking is None
            else disable_thinking
        )

        ensure_local_ollama_no_proxy(self.api_base)
        self.session = requests.Session()
        if is_local_ollama(self.api_base):
            self.session.trust_env = False

        if check_model:
            self._check_model()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _check_model(self) -> bool:
        """Check whether the configured model is advertised by /models."""
        try:
            response = self.session.get(
                f"{self.api_base}/models",
                headers=self._headers(),
                timeout=5,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            names = [str(item.get("id", "")) for item in models]
            if any(self.model_name == name or self.model_name in name for name in names):
                print(f"[LLM] OpenAI-compatible 模型就绪: {self.model_name}")
                return True

            print(f"[LLM] 警告: 模型 {self.model_name} 未找到")
            print(f"[LLM] 可用模型: {names}")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[LLM] 警告: 无法连接到 OpenAI-compatible 服务 ({self.api_base})")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"[LLM] 警告: 检查 OpenAI-compatible 模型失败: {exc}")
            return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        json_mode: bool = False,
        think: bool = False,
        timeout: int | None = None,
        num_predict: int | None = None,
    ) -> OpenAICompatibleChatResponse:
        """Send a non-streaming chat completion request."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._convert_message(message) for message in messages],
            "stream": False,
            "temperature": 0,
            "max_tokens": num_predict or self.num_predict,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking and not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        response = self.session.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout or self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        choices = raw.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") or {}
        content = str(message.get("content") or "")
        usage = raw.get("usage") or {}

        return OpenAICompatibleChatResponse(
            content=content,
            raw=raw,
            model=str(raw.get("model") or self.model_name),
            finish_reason=first_choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _convert_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the project's Ollama-style image field to OpenAI content parts."""
        role = message.get("role", "user")
        content = message.get("content", "")
        images = message.get("images") or []

        if not images:
            return {"role": role, "content": content}

        parts: List[Dict[str, Any]] = []
        if isinstance(content, list):
            parts.extend(content)
        elif content not in (None, ""):
            parts.append({"type": "text", "text": str(content)})

        for image_b64 in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                }
            )

        return {"role": role, "content": parts}
