import pytest

from scanner import llm


def test_model_raises_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    with pytest.raises(ValueError, match="CLAUDE_MODEL"):
        llm._model("claude")


def test_model_returns_provider_env_var(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-some-model")

    assert llm._model("claude") == "claude-some-model"


def test_model_override_wins_over_env_vars(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "blanket-override")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-some-model")

    assert llm._model("claude", model_override="explicit-override") == "explicit-override"


def test_model_llm_model_wins_over_provider_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "blanket-override")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-some-model")

    assert llm._model("claude") == "blanket-override"


def test_model_class_override_claude_requires_env_var(monkeypatch):
    monkeypatch.delenv("CLAUDE_EXTRACT_MODEL", raising=False)

    with pytest.raises(ValueError, match="CLAUDE_EXTRACT_MODEL"):
        llm._model_class_override("claude", "extract")


def test_model_class_override_claude_returns_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_EXTRACT_MODEL", "claude-haiku-test")

    assert llm._model_class_override("claude", "extract") == "claude-haiku-test"


def test_model_class_override_gemini_is_optional(monkeypatch):
    monkeypatch.delenv("GEMINI_EXTRACT_MODEL", raising=False)

    assert llm._model_class_override("gemini", "extract") is None


def test_model_class_override_gemini_returns_env_var_when_set(monkeypatch):
    monkeypatch.setenv("GEMINI_EXTRACT_MODEL", "gemini-extract-model")

    assert llm._model_class_override("gemini", "extract") == "gemini-extract-model"


def test_model_class_override_none_class_returns_none():
    assert llm._model_class_override("claude", None) is None


def test_model_class_override_structured_score_distinct_from_extract(monkeypatch):
    monkeypatch.setenv("CLAUDE_EXTRACT_MODEL", "claude-haiku-test")
    monkeypatch.setenv("CLAUDE_STRUCTURED_SCORE_MODEL", "claude-sonnet-test")

    assert llm._model_class_override("claude", "extract") == "claude-haiku-test"
    assert llm._model_class_override("claude", "structured_score") == "claude-sonnet-test"


def test_execute_with_breaker_resolves_extraction_model_class(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("CLAUDE_EXTRACT_MODEL", "claude-haiku-test")

    seen = {}

    def fake_client_and_model(provider, is_instructor, model_override=None):
        seen["provider"] = provider
        seen["model_override"] = model_override
        return object(), model_override or "unused"

    monkeypatch.setattr(llm, "_client_and_model", fake_client_and_model)
    monkeypatch.setattr(llm, "provider_breaker_status", lambda provider: {"open": False})
    monkeypatch.setattr(llm, "_breaker_record_success", lambda provider: None)

    result = llm.execute_with_breaker(lambda client, model: "ok", model_class="extract")

    assert result == "ok"
    assert seen["provider"] == "claude"
    assert seen["model_override"] == "claude-haiku-test"


def test_execute_with_breaker_omits_override_when_no_model_class(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    seen = {}

    def fake_client_and_model(provider, is_instructor, model_override=None):
        seen["model_override"] = model_override
        return object(), "some-model"

    monkeypatch.setattr(llm, "_client_and_model", fake_client_and_model)
    monkeypatch.setattr(llm, "provider_breaker_status", lambda provider: {"open": False})
    monkeypatch.setattr(llm, "_breaker_record_success", lambda provider: None)

    llm.execute_with_breaker(lambda client, model: "ok")

    assert seen["model_override"] is None


def test_scoring_mode_defaults_to_raw(monkeypatch):
    monkeypatch.delenv("SCORING_MODE", raising=False)
    assert llm.scoring_mode() == "raw"


def test_scoring_mode_reads_env(monkeypatch):
    monkeypatch.setenv("SCORING_MODE", "Structured")
    assert llm.scoring_mode() == "structured"
