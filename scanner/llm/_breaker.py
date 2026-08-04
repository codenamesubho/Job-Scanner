"""Per-provider circuit breaker for LLM calls.

Once a provider starts rate-limiting, hammering it just wastes time and can
extend the cooldown. After a couple of consecutive rate-limit-style failures the
breaker trips: further calls are skipped outright, with no API request attempted,
until the cooldown passes. A single success closes it again immediately.

This was module-level mutable state (`_breaker_states` guarded by `_breaker_lock`)
plus five free functions in `scanner/llm/__init__.py`. As a class the lock is
bound to the data it protects, and tests can construct an isolated instance
instead of mutating process-wide state.
"""
import threading
import time
from dataclasses import dataclass, field

import litellm

FAILURE_THRESHOLD = 2      # consecutive rate-limit failures before tripping
COOLDOWN_S        = 300    # 5 minutes


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a rate-limit/overload/cooldown-style failure, as
    opposed to e.g. a validation error or a one-off timeout.

    litellm's exceptions already subclass the equivalent openai ones, but we
    check litellm's directly rather than lean on that inheritance detail.
    `status_code` is read via getattr rather than an isinstance(APIStatusError)
    check, since it is present on both exception hierarchies.
    """
    if isinstance(exc, litellm.RateLimitError):
        return True
    if getattr(exc, "status_code", None) in (429, 503, 529):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "rate limit", "rate_limit", "cooldown", "cool down", "overloaded",
        "too many requests", "429", "503", "529",
    ))


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    last_error: str = ""


@dataclass
class CircuitBreaker:
    """Tracks failures per provider name. Safe to share across threads."""

    failure_threshold: int = FAILURE_THRESHOLD
    cooldown_s: int = COOLDOWN_S
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _states: dict[str, _ProviderState] = field(default_factory=dict, repr=False)

    def _state(self, provider: str) -> _ProviderState:
        """Caller must hold the lock."""
        return self._states.setdefault(provider, _ProviderState())

    def record_failure(self, provider: str, exc: Exception) -> None:
        """Count a failure, tripping the breaker once the threshold is reached.

        Non-rate-limit failures (bad JSON, a one-off timeout) are ignored — they
        say nothing about the provider's availability.
        """
        if not is_rate_limit_error(exc):
            return
        with self._lock:
            state = self._state(provider)
            state.consecutive_failures += 1
            state.last_error = str(exc)
            if state.consecutive_failures >= self.failure_threshold:
                state.open_until = time.time() + self.cooldown_s

    def record_success(self, provider: str) -> None:
        with self._lock:
            state = self._state(provider)
            state.consecutive_failures = 0
            state.open_until = 0.0

    def is_open(self, provider: str) -> bool:
        """True while `provider` is in cooldown and should not be called."""
        with self._lock:
            state = self._states.get(provider)
            return bool(state and state.open_until and time.time() < state.open_until)

    def status(self, provider: str) -> dict:
        """{"open", "retry_at", "retry_in_s", "reason"} — shaped for the UI, which
        shows "scoring not available" rather than trying and failing."""
        with self._lock:
            state = self._states.get(provider)
            if state and state.open_until and time.time() < state.open_until:
                return {
                    "open": True,
                    "retry_at": state.open_until,
                    "retry_in_s": int(state.open_until - time.time()),
                    "reason": state.last_error,
                }
            return {"open": False, "retry_at": None, "retry_in_s": None, "reason": ""}


#: The process-wide breaker every LLM call goes through.
BREAKER = CircuitBreaker()
