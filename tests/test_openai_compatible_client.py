from label_detection.services.openai_compatible_client import OpenAICompatibleHTTPClient
from label_detection.workflows import unified


def test_openai_compatible_client_uses_chat_completions_json_and_images(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.trust_env = True
            self.post_url = None
            self.post_headers = None
            self.payload = None

        def post(self, url, headers, json, timeout):
            self.post_url = url
            self.post_headers = headers
            self.payload = json
            return FakeResponse(
                {
                    "model": "/models/qwen",
                    "choices": [
                        {
                            "message": {"content": '{"ok":true}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                }
            )

    fake_session = FakeSession()
    monkeypatch.setattr(
        "label_detection.services.openai_compatible_client.requests.Session",
        lambda: fake_session,
    )

    client = OpenAICompatibleHTTPClient(
        model_name="/models/qwen",
        api_base="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        timeout=12,
        num_predict=123,
        check_model=False,
    )
    response = client.chat(
        [{"role": "user", "content": "Return JSON.", "images": ["abc123"]}],
        json_mode=True,
        think=False,
    )

    assert client.session.trust_env is False
    assert fake_session.post_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert fake_session.post_headers["Authorization"] == "Bearer EMPTY"
    assert fake_session.payload["response_format"] == {"type": "json_object"}
    assert fake_session.payload["max_tokens"] == 123
    assert fake_session.payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert fake_session.payload["messages"][0]["content"] == [
        {"type": "text", "text": "Return JSON."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc123"},
        },
    ]
    assert response.content == '{"ok":true}'
    assert response.debug_summary() == (
        "finish_reason=stop, completion_tokens=4, prompt_tokens=10, total_tokens=14"
    )


def test_get_llm_uses_openai_compatible_client_when_provider_selected(monkeypatch):
    calls = {}

    class FakeOpenAICompatibleHTTPClient:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(unified, "OpenAICompatibleHTTPClient", FakeOpenAICompatibleHTTPClient)
    monkeypatch.setattr(unified, "LLM_PROVIDER", "vllm")
    monkeypatch.setattr(unified, "OPENAI_COMPATIBLE_API_BASE", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(unified, "OPENAI_COMPATIBLE_API_KEY", "EMPTY")
    monkeypatch.setattr(unified, "TEXT_LLM_MODEL", "/models/qwen")
    monkeypatch.setattr(unified, "_llm", None)

    llm = unified.get_llm()

    assert isinstance(llm, FakeOpenAICompatibleHTTPClient)
    assert calls["kwargs"]["api_base"] == "http://127.0.0.1:8000/v1"
    assert calls["kwargs"]["api_key"] == "EMPTY"
    assert calls["kwargs"]["model_name"] == "/models/qwen"
    assert calls["kwargs"]["timeout"] == unified.LLM_TIMEOUT
    assert calls["kwargs"]["num_predict"] == unified.TEXT_NUM_PREDICT
