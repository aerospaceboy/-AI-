import json
import urllib.error

from src import llm
from src.llm import OpenAICompatibleProvider


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": " generated "}}]}).encode("utf-8")


def test_retries_transient_connection_errors(monkeypatch) -> None:
    attempts = [0]

    def flaky_urlopen(request, timeout=None):
        attempts[0] += 1
        if attempts[0] < 3:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse()

    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    provider = OpenAICompatibleProvider("http://127.0.0.1:9/v1", "test-model")
    assert provider.summarize("content", "instruction") == "generated"
    assert attempts[0] == 3


def test_retries_retryable_http_status(monkeypatch) -> None:
    attempts = [0]

    def flaky_urlopen(request, timeout=None):
        attempts[0] += 1
        if attempts[0] < 2:
            raise urllib.error.HTTPError("http://127.0.0.1:9/v1", 503, "Service Unavailable", None, None)
        return _FakeResponse()

    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    provider = OpenAICompatibleProvider("http://127.0.0.1:9/v1", "test-model")
    assert provider.summarize("content", "instruction") == "generated"
    assert attempts[0] == 2


def test_does_not_retry_client_errors(monkeypatch) -> None:
    attempts = [0]

    def rejecting_urlopen(request, timeout=None):
        attempts[0] += 1
        raise urllib.error.HTTPError("http://127.0.0.1:9/v1", 400, "Bad Request", None, None)

    monkeypatch.setattr(llm.urllib.request, "urlopen", rejecting_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    provider = OpenAICompatibleProvider("http://127.0.0.1:9/v1", "test-model")
    try:
        provider.summarize("content", "instruction")
    except RuntimeError as exc:
        assert "Bad Request" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert attempts[0] == 1
