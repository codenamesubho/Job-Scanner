
from scanner import llm


def test_client_and_model_prefixes_claude_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-some-model")

    client, model = llm._client_and_model("claude", is_instructor=False)

    assert model == "anthropic/claude-some-model"
    assert isinstance(client, llm._RawLiteLLMClient)


def test_client_and_model_prefixes_gemini_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-some-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    client, model = llm._client_and_model("gemini", is_instructor=False)

    assert model == "gemini/gemini-some-model"


def test_client_and_model_applies_model_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-default-model")

    _, model = llm._client_and_model("claude", is_instructor=False, model_override="claude-cheap-model")

    assert model == "anthropic/claude-cheap-model"


def test_raw_litellm_client_create_calls_through_to_completion_fn(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "test-key")
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return "fake-response"

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)

    client = llm._make_client("claude", is_instructor=False)
    result = client.chat.completions.create(model="anthropic/claude-x", messages=[{"role": "user", "content": "hi"}])

    assert result == "fake-response"
    assert len(calls) == 1
    assert calls[0]["model"] == "anthropic/claude-x"
    assert calls[0]["api_base"] == "http://localhost:8317"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_make_client_gemini_uses_no_api_base(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return "fake-response"

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)

    client = llm._make_client("gemini", is_instructor=False)
    client.chat.completions.create(model="gemini/gemini-x", messages=[])

    assert calls[0]["api_base"] is None


def test_is_rate_limit_error_catches_openai_rate_limit_error():
    import openai

    exc = openai.RateLimitError("rate limited", response=_fake_response(429), body=None)
    assert llm._is_rate_limit_error(exc) is True


def test_is_rate_limit_error_catches_litellm_rate_limit_error():
    exc = llm.litellm.RateLimitError(message="rate limited", llm_provider="anthropic", model="claude-x")
    assert llm._is_rate_limit_error(exc) is True


def test_is_rate_limit_error_catches_status_code_attribute():
    class FakeError(Exception):
        status_code = 529

    assert llm._is_rate_limit_error(FakeError("overloaded")) is True


def test_is_rate_limit_error_catches_message_substring():
    assert llm._is_rate_limit_error(Exception("Service overloaded, try again")) is True


def test_is_rate_limit_error_false_for_unrelated_error():
    assert llm._is_rate_limit_error(ValueError("bad json")) is False


def _fake_response(status_code):
    import httpx

    request = httpx.Request("POST", "http://example.com")
    return httpx.Response(status_code=status_code, request=request)
