import os

from label_detection.core.config import ensure_local_ollama_no_proxy, is_local_ollama
from label_detection.services.ollama_client import OllamaHTTPClient
from label_detection.services.vlm_detection import VLMObjectDetector
from label_detection.services.vlm_service import VLMComparator
from label_detection.workflows import unified


def test_is_local_ollama_detects_loopback_hosts():
    assert is_local_ollama("http://localhost:11434") is True
    assert is_local_ollama("http://127.0.0.1:11434") is True
    assert is_local_ollama("http://192.168.1.20:11434") is False


def test_ensure_local_ollama_no_proxy_appends_loopback_hosts(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)

    changed = ensure_local_ollama_no_proxy("http://localhost:11434")

    assert changed is True
    no_proxy = os.environ["NO_PROXY"]
    assert "example.com" in no_proxy
    assert "localhost" in no_proxy
    assert "127.0.0.1" in no_proxy
    assert os.environ["no_proxy"] == no_proxy


def test_vlm_clients_bypass_env_proxy_for_local_ollama(monkeypatch):
    monkeypatch.setattr(VLMComparator, "_check_model", lambda self: True)
    monkeypatch.setattr(VLMObjectDetector, "_check_model", lambda self: True)

    comparator = VLMComparator(api_base="http://localhost:11434")
    detector = VLMObjectDetector(api_base="http://localhost:11434")

    assert comparator.session.trust_env is False
    assert detector.session.trust_env is False


def test_get_llm_uses_native_ollama_client(monkeypatch):
    calls = {}

    class FakeOllamaHTTPClient:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(unified, "OllamaHTTPClient", FakeOllamaHTTPClient)
    monkeypatch.setattr(unified, "_llm", None)

    llm = unified.get_llm()

    assert isinstance(llm, FakeOllamaHTTPClient)
    assert calls["kwargs"]["api_base"] == unified.OLLAMA_API_BASE
    assert calls["kwargs"]["model_name"] == unified.TEXT_LLM_MODEL
    assert calls["kwargs"]["timeout"] == unified.LLM_TIMEOUT
    assert calls["kwargs"]["num_predict"] == unified.TEXT_NUM_PREDICT


def test_native_ollama_client_bypasses_proxy_and_uses_json_mode(monkeypatch):
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
            self.payload = None

        def post(self, url, json, timeout):
            self.payload = json
            return FakeResponse(
                {
                    "model": "qwen3.5:9b",
                    "message": {"content": '{"ok":true}'},
                    "done_reason": "stop",
                    "eval_count": 4,
                }
            )

    fake_session = FakeSession()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(
        "label_detection.services.ollama_client.requests.Session",
        lambda: fake_session,
    )

    client = OllamaHTTPClient(
        model_name="qwen3.5:9b",
        api_base="http://localhost:11434",
        timeout=12,
        num_predict=123,
        check_model=False,
    )
    response = client.chat(
        [{"role": "user", "content": "Return JSON."}],
        json_mode=True,
        think=False,
    )

    assert client.session.trust_env is False
    assert "localhost" in os.environ["NO_PROXY"]
    assert "127.0.0.1" in os.environ["NO_PROXY"]
    assert fake_session.payload["format"] == "json"
    assert fake_session.payload["think"] is False
    assert fake_session.payload["keep_alive"] == client.keep_alive
    assert fake_session.payload["options"]["temperature"] == 0
    assert fake_session.payload["options"]["num_predict"] == 123
    assert response.content == '{"ok":true}'
