import os
import sys
import types

from label_detection.core.config import ensure_local_ollama_no_proxy, is_local_ollama
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


def test_get_llm_sets_no_proxy_before_chatollama_init(monkeypatch):
    calls = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            calls["no_proxy"] = os.environ.get("NO_PROXY")

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_ollama", types.SimpleNamespace(ChatOllama=FakeChatOllama))
    monkeypatch.setattr(unified, "_llm", None)

    llm = unified.get_llm()

    assert isinstance(llm, FakeChatOllama)
    assert calls["kwargs"]["base_url"] == unified.OLLAMA_API_BASE
    assert calls["kwargs"]["client_kwargs"] == {"trust_env": False}
    assert "localhost" in calls["no_proxy"]
    assert "127.0.0.1" in calls["no_proxy"]
