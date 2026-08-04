import functools
import hashlib
import json
import os
import re
import threading
import time
import warnings

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

BATCH_SIZE = 4  # max concurrent batch API calls


class EmptyScoringResultError(RuntimeError):
    """Raised when a batch call returns/parses with zero usable scores for a
    non-empty batch — e.g. the model returned an empty scores array, or
    every entry had a non-matching id / null score. Unlike ordinary batch
    failures (timeout, one-off validation error), this is treated as fatal
    for the whole scoring run rather than logged-and-skipped — it usually
    means something structural is wrong (the model silently not responding
    as expected) rather than a one-off blip, and continuing would silently
    under-score the rest of the run without any visible signal."""


def _batch_short_id(text: str, index: int) -> str:
    """Short, easy-for-an-LLM-to-copy-verbatim label for one job within a
    batch, used by raw_scoring/structured_scoring instead of asking the
    model to echo back a job's real (potentially 400+ char opaque) id —
    long ids are prone to single-character transcription errors that make
    the whole entry fail exact-match validation. sha256 of `text`
    (truncated) plus the job's 1-based position in the batch to guarantee
    uniqueness even on a hash collision. Purely ephemeral: scoped to a
    single batch call, never persisted or matched against the DB — contrast
    with scanner.database.content_hash(), a separate save-time dedup
    signal built on the same primitive but a different length threshold and
    persistence model."""
    digest = hashlib.sha256((text or "").encode()).hexdigest()[:10]
    return f"{digest}{index}"


def _raw_completion_text(response) -> str:
    """Best-effort extraction of the raw text/tool-call content and finish
    reason from a litellm ModelResponse, for diagnostic logging when parsing
    produces an unexpectedly empty result. Registered via instructor's
    "completion:response" hook (client.on(...)) rather than read off the
    parsed Pydantic result, since the whole point is to see what the model
    actually said even when the parsed result is empty. Never raises."""
    if response is None:
        return "(no raw response captured)"
    try:
        choice = response.choices[0]
        message = choice.message
        parts = []
        content = getattr(message, "content", None)
        if content:
            parts.append(str(content))
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is not None:
                parts.append(f"[tool_call {getattr(fn, 'name', '?')}] {getattr(fn, 'arguments', '')}")
        text = " | ".join(p for p in parts if p) or "(empty content)"
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        return f"finish_reason={finish_reason} usage={usage} content={text[:3000]!r}"
    except Exception as e:
        return f"(failed to extract raw response: {type(e).__name__}: {e})"


load_dotenv(dotenv_path=os.path.expanduser(os.getenv("ENV_FILE", "~/.env")), override=True)

if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
    # Both litellm's "langfuse" callback and the langfuse SDK itself read
    # LANGFUSE_HOST (defaulting to https://cloud.langfuse.com if unset) —
    # LANGFUSE_BASE_URL isn't a recognized Langfuse env var name anywhere in
    # the ecosystem. This repo's .env uses LANGFUSE_BASE_URL for a
    # self-hosted instance; without this, tracing would silently default to
    # Langfuse Cloud instead. Prefer renaming LANGFUSE_BASE_URL ->
    # LANGFUSE_HOST in .env; this shim just keeps existing .env files working.
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

# ── Langfuse tracing (optional) ────────────────────────────────────────────────
# Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (+ optionally LANGFUSE_HOST,
# default https://cloud.langfuse.com) in .env to enable. Two independent
# pieces, both driven by the same env vars: litellm's built-in "langfuse_otel"
# callback traces every individual litellm.completion() call as a
# "generation" (registered globally below, not per-client — unlike the old
# openai-wrapped-client approach, litellm's callback list is process-wide);
# the @observe decorator groups those generations under one named trace per
# function call (e.g. all calls inside one _score_batch invocation). Uses
# the modern `from langfuse import observe` (langfuse v3+) — the old
# `langfuse.decorators.observe` path (v2) no longer exists.
#
# "langfuse_otel" (not litellm's older "langfuse" callback) deliberately:
# litellm 1.92.0's "langfuse" callback goes through LangfusePromptManagement,
# which calls into the installed langfuse SDK's Python API directly and
# hardcodes assumptions from the SDK's v2.x module layout (e.g.
# langfuse.version.__version__, a Langfuse(sdk_integration=...) kwarg) that
# no longer hold on langfuse v4.x (this repo's installed version) — it fails
# non-fatally, silently leaving tracing uninitialized. "langfuse_otel" sends
# spans over OTLP/HTTP straight to Langfuse's OTel ingestion endpoint
# (LANGFUSE_HOST + "/api/public/otel") and never touches the langfuse
# package's Python API, so it isn't exposed to that SDK-version drift.
from ._breaker import BREAKER, is_rate_limit_error as _is_rate_limit_error  # noqa: E402
from ._tracing import install_callbacks, is_tracing, observe  # noqa: E402

_is_tracing = is_tracing          # kept: referenced by name elsewhere in this module
install_callbacks()


_PROVIDER_CONFIG = {
    "claude": {
        # CLIProxyAPI (bridges a Claude Pro subscription, not a paid API key) —
        # litellm's anthropic/ provider appends /v1/messages itself, so this
        # is bare, unlike the old "http://localhost:8317/v1/" OpenAI-shaped base_url.
        "api_base": "http://localhost:8317",
        "api_key_env": "CLAUDE_API_KEY",
        "litellm_prefix": "anthropic",
    },
    "gemini": {
        "api_base": None,  # litellm's gemini/ provider talks to Google natively — no proxy
        "api_key_env": "GEMINI_API_KEY",
        "litellm_prefix": "gemini",
    },
}

def _is_provider_rate_limited(provider: str) -> bool:
    return BREAKER.is_open(provider)


def _provider() -> str:
    pref = os.getenv("LLM_PROVIDER", "claude").lower()
    if pref == "claude" and _is_provider_rate_limited("claude"):
        return "gemini"
    return pref


def scoring_mode() -> str:
    """'raw' (default) = existing free-text description scoring; 'structured'
    = score structured JD/resume JSON via score_jobs_structured() instead."""
    return os.getenv("SCORING_MODE", "raw").lower()


class _RawLiteLLMClient:
    """Shim exposing client.chat.completions.create(...) over a bound
    litellm.completion(...) call, so the non-instructor call sites
    (generate_summary, draft_referral_message) don't need to change at all —
    litellm's own interface is a free function, not a client object."""
    def __init__(self, completion_fn):
        self._completion_fn = completion_fn
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return self._completion_fn(**kwargs)


def _make_client(provider: str | None = None, is_instructor: bool = False):
    """Return a client for the given LLM provider, bound to that provider's
    api_base/api_key via litellm. Supported providers: "claude" (Anthropic,
    via CLIProxyAPI), "gemini". When is_instructor=True, wraps with
    instructor.from_litellm() for structured-output support; instructor's
    client exposes the same .chat.completions.create(...) shape as the
    plain _RawLiteLLMClient shim, so callers never need to branch on this.
    """
    provider = (provider or _provider()).lower()
    cfg = _PROVIDER_CONFIG.get(provider)
    if not cfg:
        supported = ", ".join(sorted(_PROVIDER_CONFIG))
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {supported}")

    completion_fn = functools.partial(
        litellm.completion, api_base=cfg["api_base"], api_key=_api_key(provider),
    )
    if is_instructor:
        # instructor.from_litellm() always tags the client as Provider.OPENAI
        # internally (litellm's own interface is OpenAI-shaped regardless of
        # the actual backend model) — Mode.TOOLS isn't a registered
        # (provider, mode) pair for Provider.OPENAI in instructor's registry
        # (that's reserved for providers with a non-parallel tool-call API,
        # e.g. native Anthropic), so it fails at call time. Mode.JSON_SCHEMA
        # is registered for Provider.OPENAI, maps to a real structured-output
        # mechanism, and — unlike Mode.PARALLEL_TOOLS, the other
        # OPENAI-registered option — takes a plain response_model instead of
        # requiring Iterable[...], matching our existing single-object
        # response models (BatchScoreResult, StructuredBatchScoreResult, etc.).
        return instructor.from_litellm(completion_fn, mode=instructor.Mode.JSON_SCHEMA)
    return _RawLiteLLMClient(completion_fn)


_warmed_up_providers: set[str] = set()
_warmup_lock = threading.Lock()


class _WarmupModel(BaseModel):
    ok: bool = True


def _warm_up_litellm(provider: str) -> None:
    """One-time, single-threaded instructor.from_litellm(...) call per
    provider to force instructor's mode_registry to fully populate the
    (Provider.OPENAI, Mode.JSON_SCHEMA) handler registration before
    concurrent batch calls start. Confirmed by direct reproduction: firing
    ~20 concurrent first-time instructor.create(mode=Mode.JSON_SCHEMA, ...)
    calls from a cold process raises "RegistryError: Mode Mode.JSON_SCHEMA
    is not registered for provider Provider.OPENAI" on nearly all of
    them — this is a registration race the first time that (provider, mode)
    handler is looked up, not (as first suspected) a litellm lazy-import
    race; a raw, non-instructor warmup call (is_instructor=False) does NOT
    exercise this path and does NOT fix it — the warmup call itself must go
    through instructor with response_model set, matching the real call
    shape. mock_response means this costs no network call or tokens."""
    if provider in _warmed_up_providers:
        return
    with _warmup_lock:
        if provider in _warmed_up_providers:
            return
        try:
            client, model = _client_and_model(provider, is_instructor=True)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "warmup"}],
                response_model=_WarmupModel,
                max_retries=1,
                mock_response='{"ok": true}',
            )
        except Exception:
            pass  # best-effort — a warmup failure shouldn't block real scoring
        _warmed_up_providers.add(provider)


# ── Shared Pydantic models ──────────────────────────────────────────────────
# Shared by both raw (raw_scoring._score_batch) and structured
# (structured_scoring._score_structured_batch) scoring.

class BreakdownItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    score: int = Field(default=0, ge=0)
    max: int = Field(default=0)
    reason: str = Field(default="")


_BREAKDOWN_MAX = {"skills": 60, "company": 10, "remote": 10, "role": 20}


class JobBreakdown(BaseModel):
    """Shared by both raw (_score_batch) and structured (_score_structured_batch)
    scoring. Category max values are fixed by the rubric, not model-supplied —
    the model can get a sub-score wrong (e.g. 24/20, seen in practice), so
    _clamp_to_category_max forces the true max and clips score into [0, max]
    rather than trusting whatever the model echoed back for `max`."""
    model_config = ConfigDict(extra="ignore")
    skills: BreakdownItem = Field(default_factory=BreakdownItem)
    company: BreakdownItem = Field(default_factory=BreakdownItem)
    remote: BreakdownItem = Field(default_factory=BreakdownItem)
    role: BreakdownItem = Field(default_factory=BreakdownItem)

    @model_validator(mode="after")
    def _clamp_to_category_max(self) -> "JobBreakdown":
        for name, cap in _BREAKDOWN_MAX.items():
            item = getattr(self, name)
            item.max = cap
            item.score = max(0, min(item.score, cap))
        return self


_BREAKDOWN_LABELS = (
    ("skills", "Skills"),
    ("company", "Company"),
    ("remote", "Remote"),
    ("role", "Role"),
)


def parse_score_breakdown(breakdown_raw: str, fallback_score=None) -> dict:
    """Parse a stored `score_breakdown` string (the jobs.score_breakdown
    column) into a display-ready structure:
        {"computed_score": int, "items": [(label, score, max, reason), ...]}

    Handles both the current JSON format (a JobBreakdown.model_dump()) and
    the legacy pipe-delimited format from rows scored before that format
    existed — in the legacy case "items" is empty and "legacy_lines" holds
    the raw "label: reason" strings instead.
    """
    breakdown_raw = breakdown_raw or ""

    bd = None
    try:
        bd = json.loads(breakdown_raw)
    except Exception:
        pass

    if bd:
        computed = sum(bd[k]["score"] for k in ("skills", "company", "remote", "role") if k in bd)
        items = [
            (label, bd.get(key, {}).get("score", 0), bd.get(key, {}).get("max", 0),
             bd.get(key, {}).get("reason", ""))
            for key, label in _BREAKDOWN_LABELS
        ]
        return {"computed_score": computed, "items": items, "legacy_lines": []}

    bd_nums = re.findall(r'(\d+)/\d+', breakdown_raw)
    computed = sum(int(x) for x in bd_nums) if bd_nums else int(fallback_score or 0)
    legacy_lines = [p.strip() for p in breakdown_raw.split("|") if p.strip()]
    return {"computed_score": computed, "items": [], "legacy_lines": legacy_lines}


class JobScoreItem(BaseModel):
    """One job's score within a batch response — id ties it back to the job."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default="")
    score: int | None = Field(default=None, ge=0, le=100)
    reason: str = Field(default="")
    breakdown: JobBreakdown = Field(default_factory=JobBreakdown)


class BatchScoreResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # No default — CLIProxyAPI intermittently wraps the whole payload one level
    # deeper (e.g. content='{"input": {"scores": [...]}}' instead of
    # '{"scores": [...]}'), a synthetic-tool-call translation artifact. With a
    # default_factory, that shape silently validates as an empty result instead
    # of failing, which meant the model's fully correct answer was thrown away
    # without triggering instructor's reask/retry. Making the field required
    # turns that into a validation error, which does trigger a reask — the
    # model already produced a correct answer once, so a reask should recover
    # it instead of us discarding it.
    scores: list[JobScoreItem]


# ── Helpers ────────────────────────────────────────────────────────────────

def _api_key(provider: str | None = None) -> str:
    provider = (provider or _provider()).lower()
    cfg = _PROVIDER_CONFIG.get(provider)
    if not cfg:
        supported = ", ".join(sorted(_PROVIDER_CONFIG))
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {supported}")
    env_name = cfg["api_key_env"]
    key = os.getenv(env_name)
    if not key:
        raise ValueError(f"{env_name} is not set. Add it to your .env file.")
    return key


def _model(provider: str | None = None, model_override: str | None = None) -> str:
    """Resolve the model id to use. `model_override` (passed by extraction
    call sites — see _extract_model_override) wins over everything, since
    it's only ever supplied for a specific, deliberately cheaper call class.
    Otherwise: LLM_MODEL (blanket override) > CLAUDE_MODEL/GEMINI_MODEL (per-
    provider). There is no hardcoded fallback — every model id must come
    from .env, so a missing env var fails loudly instead of silently
    guessing a model id that might not exist.
    """
    provider = (provider or _provider()).lower()
    if model_override:
        return model_override
    if os.getenv("LLM_MODEL"):
        return os.getenv("LLM_MODEL")
    env_var = {"claude": "CLAUDE_MODEL", "gemini": "GEMINI_MODEL"}.get(provider)
    value = os.getenv(env_var) if env_var else None
    if not value:
        raise ValueError(f"{env_var} is not set — add it to your .env file.")
    return value


# Model classes distinct from the default scoring-class model _model()
# resolves from CLAUDE_MODEL/GEMINI_MODEL. "extract" is the cheap JD/resume
# extraction call (Haiku-class); "structured_score" is the model that scores
# structured JD/resume JSON (Sonnet-class) — kept as its own env var rather
# than reusing CLAUDE_MODEL/GEMINI_MODEL, since those may already be pointed at
# a cheaper model for the existing raw-text scoring path.
_MODEL_CLASS_ENV = {
    "extract":          {"claude": "CLAUDE_EXTRACT_MODEL", "gemini": "GEMINI_EXTRACT_MODEL"},
    "structured_score": {"claude": "CLAUDE_STRUCTURED_SCORE_MODEL", "gemini": "GEMINI_STRUCTURED_SCORE_MODEL"},
    "referral":         {"claude": "CLAUDE_REFERRAL_MODEL", "gemini": "GEMINI_REFERRAL_MODEL"},
}


def _model_class_override(provider: str, model_class: str | None) -> str | None:
    """Resolve the env var for a given model class + provider. Required
    (raises) under Claude, since each class exists specifically to pick a
    deliberately different tier than whatever CLAUDE_MODEL is set to.
    Optional under Gemini — falls back to GEMINI_MODEL via _model() when unset.
    """
    if not model_class:
        return None
    env_var = _MODEL_CLASS_ENV.get(model_class, {}).get(provider)
    if not env_var:
        return None
    value = os.getenv(env_var)
    if provider == "claude" and not value:
        raise ValueError(f"{env_var} is not set — add it to your .env file.")
    return value


# ── Circuit breaker ────────────────────────────────────────────────────────────
# When the model is rate-limited/overloaded, hammering it with more batch calls
# just wastes time and risks making the cooldown worse. After a couple of
# consecutive rate-limit-style failures, trip the breaker: further scoring is
# skipped outright (no API call attempted) until the cooldown window passes.
# A single success closes the breaker again immediately.

def _breaker_record_failure(provider: str, exc: Exception) -> None:
    BREAKER.record_failure(provider, exc)


def _breaker_record_success(provider: str) -> None:
    BREAKER.record_success(provider)


def provider_breaker_status(provider: str) -> dict:
    """Breaker status for one provider — see CircuitBreaker.status()."""
    return BREAKER.status(provider)


def scoring_breaker_status() -> dict:
    """Breaker status for whichever provider a scoring call would use, so callers
    (e.g. the UI) can show "scoring not available" instead of trying and failing
    against a model that is still in cooldown."""
    return provider_breaker_status(_provider())


def _client_and_model(provider: str, is_instructor: bool, model_override: str | None = None):
    client = _make_client(provider, is_instructor)
    raw_model = _model(provider, model_override)
    prefixed_model = f"{_PROVIDER_CONFIG[provider]['litellm_prefix']}/{raw_model}"
    return client, prefixed_model


def execute_with_breaker(query_fn, is_instructor=False, log_fn=None, model_class=None,
                          provider_override: str | None = None):
    """Executes a query function with circuit breaker protection, automatic provider
    switching, and instant fallback retry if the primary provider hits a rate limit.

    model_class ("extract" or "structured_score") routes the call through a
    model tier distinct from the default scoring-class model (see
    _MODEL_CLASS_ENV) — resolved fresh per provider on each attempt, since
    the correct override differs between the primary provider and a Gemini
    fallback attempt.

    provider_override pins the call to a specific provider regardless of
    LLM_PROVIDER or the rate-limit fallback logic. Use this when a function
    should always use a particular provider (e.g. extract_job_requirements
    always uses Gemini while resume extraction and scoring always use Claude).
    When set, the automatic Claude → Gemini fallback is suppressed — the
    override provider is used as-is and any failure is raised directly.
    """
    if provider_override:
        # Explicit provider pin — honour the breaker for that provider but
        # do NOT fall back automatically (the caller deliberately chose it).
        provider = provider_override.lower()
        breaker = provider_breaker_status(provider)
        if breaker["open"]:
            if log_fn:
                log_fn(f"Skipped — {provider} in cooldown for ~{breaker['retry_in_s']}s.")
            raise RuntimeError(
                f"LLM unavailable — {provider} in cooldown "
                f"(~{breaker['retry_in_s']}s remaining): {breaker['reason']}"
            )
        model_override = _model_class_override(provider, model_class)
        client, model = _client_and_model(provider, is_instructor, model_override)
        try:
            result = query_fn(client, model)
            _breaker_record_success(provider)
            return result
        except Exception as e:
            _breaker_record_failure(provider, e)
            raise

    provider = _provider()
    breaker = provider_breaker_status(provider)
    if breaker["open"]:
        if log_fn:
            log_fn(f"Skipped — {provider} in cooldown for ~{breaker['retry_in_s']}s.")
        raise RuntimeError(
            f"LLM unavailable — {provider} in cooldown "
            f"(~{breaker['retry_in_s']}s remaining): {breaker['reason']}"
        )

    pref_provider = os.getenv("LLM_PROVIDER", "claude").lower()
    if pref_provider == "claude" and provider == "gemini" and log_fn:
        log_fn("Claude is rate-limited; falling back to Gemini.")

    try:
        model_override = _model_class_override(provider, model_class)
        client, model = _client_and_model(provider, is_instructor, model_override)
        result = query_fn(client, model)
        _breaker_record_success(provider)
        return result
    except Exception as e:
        _breaker_record_failure(provider, e)
        if provider == "claude" and _is_rate_limit_error(e):
            gemini_breaker = provider_breaker_status("gemini")
            if not gemini_breaker["open"]:
                msg = "Claude hit rate limit. Retrying query on Gemini immediately..."
                if log_fn:
                    log_fn(msg)
                else:
                    warnings.warn(msg)
                try:
                    gemini_override = _model_class_override("gemini", model_class)
                    client_gemini, model_gemini = _client_and_model("gemini", is_instructor, gemini_override)
                    result = query_fn(client_gemini, model_gemini)
                    _breaker_record_success("gemini")
                    return result
                except Exception as gemini_err:
                    _breaker_record_failure("gemini", gemini_err)
                    raise gemini_err
        raise e


# ── Submodules (public API surface) ─────────────────────────────────────────
# Imported at the bottom, after everything above is fully defined, since each
# submodule reaches back into this package's namespace at import time (e.g.
# `from . import execute_with_breaker`) — importing them earlier would fail
# with a partially-initialized package. `observe` in particular must exist
# above before any of these load, since each applies it as a decorator at
# import time.
from .extraction import (       # noqa: E402
    JobRequirements, ResumeProfile, generate_summary, extract_job_requirements,
    extract_resume_profile,
)
from .raw_scoring import (       # noqa: E402
    score_jobs, _score_batch, _format_job_block, _build_batches,
)
from .structured_scoring import (       # noqa: E402
    score_jobs_structured, _score_structured_batch, _compute_structured_breakdown,
    StructuredBatchScoreResult, StructuredCompanyJudgment, StructuredJobScoreItem,
    StructuredRemoteJudgment, StructuredRequirementJudgment, StructuredRoleJudgment,
    StructuredSkillItem, StructuredSkillsJudgment,
)
from .referral import draft_referral_message, match_form_fields, FormFieldMap       # noqa: E402
