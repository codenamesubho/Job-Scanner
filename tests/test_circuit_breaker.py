"""Tests for scanner.llm._breaker.CircuitBreaker.

As module-level state this logic could only be tested by mutating process-wide
globals, so it wasn't. As a class each test gets its own isolated instance.
"""
import time

import litellm
import pytest

from scanner.llm._breaker import CircuitBreaker, is_rate_limit_error


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=2, cooldown_s=300)


# --------------------------------------------------------- rate-limit detection

@pytest.mark.parametrize("message", [
    "Rate limit exceeded", "rate_limit hit", "Model is overloaded",
    "Too many requests", "got 429 back", "503 Service Unavailable",
    "529 overloaded", "please cool down",
])
def test_rate_limit_messages_are_detected(message):
    assert is_rate_limit_error(RuntimeError(message)) is True


@pytest.mark.parametrize("status", [429, 503, 529])
def test_rate_limit_status_codes_are_detected(status):
    exc = RuntimeError("something")
    exc.status_code = status
    assert is_rate_limit_error(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("invalid JSON in response"),
    RuntimeError("connection reset"),
    TimeoutError("timed out"),
])
def test_other_failures_are_not_rate_limits(exc):
    """These say nothing about provider availability and must not trip the breaker."""
    assert is_rate_limit_error(exc) is False


def test_litellm_rate_limit_error_is_detected():
    exc = litellm.RateLimitError("slow down", llm_provider="anthropic", model="x")
    assert is_rate_limit_error(exc) is True


# ------------------------------------------------------------- tripping / reset

def test_starts_closed(breaker):
    assert breaker.is_open("claude") is False
    assert breaker.status("claude")["open"] is False


def test_one_failure_is_not_enough_to_trip(breaker):
    breaker.record_failure("claude", RuntimeError("rate limit"))
    assert breaker.is_open("claude") is False


def test_trips_at_the_failure_threshold(breaker):
    for _ in range(2):
        breaker.record_failure("claude", RuntimeError("rate limit"))

    assert breaker.is_open("claude") is True
    status = breaker.status("claude")
    assert status["open"] is True
    assert 0 < status["retry_in_s"] <= 300
    assert "rate limit" in status["reason"]


def test_non_rate_limit_failures_never_trip_it(breaker):
    for _ in range(10):
        breaker.record_failure("claude", ValueError("bad JSON"))

    assert breaker.is_open("claude") is False


def test_a_single_success_closes_it(breaker):
    for _ in range(2):
        breaker.record_failure("claude", RuntimeError("rate limit"))
    assert breaker.is_open("claude") is True

    breaker.record_success("claude")

    assert breaker.is_open("claude") is False
    assert breaker.status("claude")["reason"] == ""


def test_success_also_resets_the_failure_count(breaker):
    """Otherwise one earlier failure plus one later failure would trip it."""
    breaker.record_failure("claude", RuntimeError("rate limit"))
    breaker.record_success("claude")
    breaker.record_failure("claude", RuntimeError("rate limit"))

    assert breaker.is_open("claude") is False


def test_providers_are_tracked_independently(breaker):
    """Tripping Claude must not disable the Gemini fallback."""
    for _ in range(2):
        breaker.record_failure("claude", RuntimeError("rate limit"))

    assert breaker.is_open("claude") is True
    assert breaker.is_open("gemini") is False


def test_reopens_for_calls_after_the_cooldown_expires():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=0)
    breaker.record_failure("claude", RuntimeError("rate limit"))
    time.sleep(0.01)

    assert breaker.is_open("claude") is False


def test_status_of_an_unknown_provider_is_closed(breaker):
    assert breaker.status("never-seen")["open"] is False
