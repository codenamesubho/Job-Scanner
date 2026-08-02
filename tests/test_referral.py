from unittest.mock import MagicMock

import pytest

from scanner.llm import referral


def _response(text, finish_reason="stop"):
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _patch_execute_with_breaker(monkeypatch, client):
    def fake_execute(query_fn, **kw):
        return query_fn(client, "fake-model")
    monkeypatch.setattr(referral, "execute_with_breaker", fake_execute)


_GOOD_MESSAGE = (
    "This is a much longer message that ties background to role and "
    "explicitly asks for a referral, please refer me for this role, thank you so much!"
)


def test_short_response_retries_with_nudge_and_bigger_max_tokens(monkeypatch):
    client = _FakeClient([_response("Too short"), _response(_GOOD_MESSAGE)])
    _patch_execute_with_breaker(monkeypatch, client)

    result = referral.draft_referral_message(
        "Experienced backend engineer.", {"name": "Jane", "title": "Eng Manager"},
        {"title": "Backend Engineer", "company": "Acme"},
    )

    assert result == _GOOD_MESSAGE
    assert len(client.calls) == 2
    assert "too short" in client.calls[1]["messages"][0]["content"].lower()
    assert client.calls[0]["max_tokens"] < client.calls[1]["max_tokens"]


def test_response_without_an_ask_retries(monkeypatch):
    no_ask = "I hope you are doing well and enjoying your role at the company these days."
    client = _FakeClient([_response(no_ask), _response(_GOOD_MESSAGE)])
    _patch_execute_with_breaker(monkeypatch, client)

    result = referral.draft_referral_message(
        "summary", {"name": "Jane"}, {"title": "Eng", "company": "Acme"},
    )

    assert result == _GOOD_MESSAGE
    assert len(client.calls) == 2


def test_truncated_response_retries_without_too_short_nudge(monkeypatch):
    # Includes an ask keyword so truncation is the *only* degenerate signal —
    # isolates the truncation path from the separate "no ask" retry path.
    long_text = "please refer me " + " ".join(["word"] * 50)
    client = _FakeClient([
        _response(long_text, finish_reason="length"),
        _response(long_text + " thanks", finish_reason="stop"),
    ])
    _patch_execute_with_breaker(monkeypatch, client)

    result = referral.draft_referral_message(
        "summary", {"name": "Jane"}, {"title": "Eng", "company": "Acme"},
    )

    assert result.endswith("thanks")
    assert len(client.calls) == 2
    # Truncated-but-substantial output shouldn't get the "too short" nudge —
    # the second prompt must be the unmodified base prompt, not base+nudge.
    assert client.calls[1]["messages"][0]["content"] == client.calls[0]["messages"][0]["content"]


def test_empty_response_twice_raises(monkeypatch):
    client = _FakeClient([_response(""), _response("")])
    _patch_execute_with_breaker(monkeypatch, client)

    with pytest.raises(ValueError, match="empty"):
        referral.draft_referral_message(
            "summary", {"name": "Jane"}, {"title": "Eng", "company": "Acme"},
        )


def test_good_response_on_first_attempt_no_retry(monkeypatch):
    client = _FakeClient([_response(_GOOD_MESSAGE)])
    _patch_execute_with_breaker(monkeypatch, client)

    result = referral.draft_referral_message(
        "summary", {"name": "Jane"}, {"title": "Eng", "company": "Acme"},
    )

    assert result == _GOOD_MESSAGE
    assert len(client.calls) == 1
