import functools
import json
import os
import re
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Literal

import instructor
import litellm
from dotenv import load_dotenv
from instructor.core import IncompleteOutputException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BATCH_SIZE = 4  # max concurrent batch API calls

load_dotenv(dotenv_path="/tmp/Jobscanner/.env")

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
try:
    from langfuse import observe
    _LF_AVAILABLE = True
except ImportError:
    _LF_AVAILABLE = False

    def observe(_fn=None, **_kw):       # no-op decorator when langfuse not installed
        def _wrap(fn): return fn
        return _wrap(_fn) if _fn else _wrap


def _is_tracing() -> bool:
    return _LF_AVAILABLE and bool(os.getenv("LANGFUSE_SECRET_KEY"))


if _is_tracing():
    litellm.success_callback = ["langfuse_otel"]
    litellm.failure_callback = ["langfuse_otel"]


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

_breaker_lock = threading.Lock()
_breaker_states = {
    "claude": {"consecutive_failures": 0, "open_until": 0.0, "last_error": ""},
    "gemini": {"consecutive_failures": 0, "open_until": 0.0, "last_error": ""},
}


def _is_provider_rate_limited(provider: str) -> bool:
    with _breaker_lock:
        state = _breaker_states.get(provider)
        if state:
            open_until = state.get("open_until", 0.0)
            if open_until and time.time() < open_until:
                return True
        return False


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


# ── Prompts ────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = (
    "Based on the following resume text, write a concise professional summary "
    "in first person. Highlight key skills, years of experience, "
    "and career focus. Highlight my work domain, tech stack and the years I have spent working on them."
    "Output only the summary text — no headings or labels.\n\n"
    "Resume:\n{resume_text}"
)

_SCORE_RUBRIC = """\
Scoring rubric — four categories per job, sub-scores MUST sum to that job's total score:
- skills (0–60): how well the candidate's background fits the STATED requirements, weighing CORE
  fit far more heavily than specific tool/framework overlap. Score only against explicitly listed
  qualifications — these live in sections like "Basic Qualifications"/"Required"/"Preferred"/
  "What you'll need", NOT in company mission/context narrative (e.g. a "we're building supply
  chain tech" intro is marketing framing, not a requirement — do not treat it as one).
    * CORE fit = the same overall engineering discipline/domain as the candidate's background
      (backend/distributed-systems/platform engineering — NOT a fundamentally different
      discipline like QA/test-automation, frontend-only, mobile-only, data-science/ML-research,
      etc.) and a matching general architecture pattern (microservices, event-driven pipelines,
      distributed data systems). This is what actually predicts whether someone can do the job.
      Years-of-experience and seniority-level fit are judged separately (HARD GATE below, and the
      role category) — do not fold them into CORE fit, or the same signal gets scored twice.
    * PERIPHERAL/learnable = a specific named programming language, framework, cloud vendor,
      observability tool, or similar tooling choice (e.g. Go vs Python, React vs Vue, Datadog vs
      Prometheus, a specific AI-coding-assistant, a specific ORM). Most engineers pick these up
      within weeks on the job once the core discipline/architecture already matches — companies
      routinely hire for the core discipline and expect tooling gaps to be learned post-hire.
      Missing several of these should cost meaningful but bounded points; never treat the job as
      "not a match" on their absence alone.
    * Domain-specific compliance/regulatory knowledge (HL7/FHIR/HIPAA, PCI, SOC2, etc.) sits
      between the two — real but learnable-on-the-job unless the JD explicitly requires it as a
      hard prerequisite.
  Do NOT penalise for industry domain (fintech, payments, healthcare, supply chain, etc.) unless
  a qualifications section explicitly requires that domain background. Give partial credit when
  the stack/domain transfers (e.g. distributed systems at HR-tech → payments infrastructure).
  If the JD does NOT name a specific tech stack/language at all (only responsibilities, scope, or
  years), score assuming the company is okay with any stack — no stated stack means no bar to
  fail, and usually signals the org isn't stack-picky, so a strong generalist background is a
  good fit for it. Only reduce skills below what the stated (non-stack) requirements otherwise
  earn when the JD explicitly names specific technologies/languages the candidate's background
  doesn't show, and even then bound the reduction per PERIPHERAL guidance above.
  Scoring bands (pick the band from CORE fit, then adjust within it for how many
  peripheral/tooling requirements are missing):
    * 40–60: core discipline and architecture pattern both match. Deduct within this band for
      missing peripheral tooling/frameworks, but stay at 40+ as long as the core discipline match
      holds — this is a legitimate strong candidate even with tooling gaps.
    * 20–39: core discipline mostly matches but with real gaps — an adjacent-but-different
      architecture pattern, or several peripheral gaps stacked together with a thin overall match.
    * 0–19: reserved for an actual CORE mismatch — a different engineering discipline entirely.
      Do not score this low just because several specific tools/frameworks aren't shown on the
      resume (peripheral gap, not a core mismatch), and do not score it low for a years-of-
      experience shortfall either — that's the HARD GATE's job below, not this band.
  HARD GATE: first check whether the job states an explicit minimum years-of-experience (e.g.
  "12+ years"). Compare it numerically against the candidate's own stated years of experience —
  do not treat "10+ years" as satisfying a "12+ years" requirement just because both are
  large numbers; if the candidate's years fall short of a stated minimum, this is a likely
  automatic screen-out regardless of how well the tech stack matches: cap skills at 15/60 and
  say so explicitly in the reason (name both numbers). If no minimum is stated, or the
  candidate meets/exceeds it, score normally per the bands above.
  The reason MUST justify the point loss, not just the match: name the specific requirements
  that DO match, AND the specific stated requirements that are missing/weak that account for
  why the score isn't higher, distinguishing core gaps from peripheral/learnable ones (e.g.
  "Kafka/Python/AWS and core distributed-systems experience match; missing Go and Kubernetes as
  specific tooling (learnable, minor deduction), no ML-infra domain experience stated — hence
  44/60, not lower, since the core discipline fit holds"). A reason that only lists matches
  without naming what's missing is incomplete — every gap large enough to cost more than ~5
  points must be named.
- company (0–10): big tech (Google, Meta, Apple, Microsoft, Amazon, Netflix, Uber, Airbnb,
  Stripe, OpenAI, Anthropic, Salesforce, Nvidia, Visa, etc.) or a large/well-funded startup
  (unicorn, late-stage, hundreds+ employees) = 10; well-known but smaller/early-stage startup
  = 7; unknown = 0.
- remote (0–10): fully remote = 10; hybrid = 5; on-site = 0.
- role (0–20): how well the job's level matches the candidate's demonstrated CEILING of fit for
  that company's scale — NOT a general scope/impact/growth score. The candidate has
  consistently been hired at **Senior**-level at big tech/large well-funded startups, and at
  **Staff**-level at smaller/early-stage startups (a smaller company gives more scope/impact
  per level, so the achievable ceiling there is one level higher than at a big company). Use
  the same company-scale judgment as the company category above. The key distinction is
  ceiling vs. reach, NOT "exact sweet spot vs. anything else" — a title AT OR BELOW the
  demonstrated ceiling is an equally strong, safe fit, not a downgrade:
    * 15–20: title is AT OR BELOW the demonstrated ceiling for that company's scale. This
      includes the exact sweet spot (Senior at big tech/large startup; Staff at a
      smaller/early-stage startup) AND anything easier than that ceiling (e.g. Senior at a
      smaller/early-stage startup is comfortably below its Staff ceiling — still a strong,
      safe fit, score it here too, not lower).
    * 6–12: company scale itself is ambiguous/borderline (e.g. mid-size, well-funded-but-not-
      huge) so the applicable ceiling is unclear either way.
    * 0–5: title clearly EXCEEDS the demonstrated ceiling for that company's scale — a real
      reach with low odds (e.g. Staff/Principal/Senior Staff at big tech or a large/well-funded
      startup, since the demonstrated ceiling there is Senior). This is not a scope/impact
      bonus — score it low even though the role itself may look impressive.
  Include a one-sentence reason naming both the title level and the company's scale."""

# One LLM call scores every job packed into it (see _build_batches) rather
# than one call per job — the response is a JSON array, one entry per job,
# matched back by the "id" copied verbatim from each job block below.
#
# Split into a system part (rubric + candidate profile — identical across
# every batch call in a scoring run) and a user part (job data + response
# format — different per batch) so the system part can be prompt-cached via
# litellm's cache_control_injection_points (see _score_batch) instead of
# resending ~1,800 tokens of static rubric text on every single batch call.
_BATCH_SCORE_SYSTEM_PROMPT = """\
You are a recruiter scoring job matches for a candidate against MULTIPLE jobs in a single pass. \
Score EVERY job listed below independently, 0–100 each.

{rubric}

Candidate Profile:
{summary}"""

_BATCH_SCORE_USER_PROMPT = """\
There are {n} job(s) below, each starting with "--- Job id=<id> ---". Score every one of them — \
do not skip any and do not invent jobs that aren't listed.

{jobs_block}

IMPORTANT: Respond with ONLY a JSON object — no markdown, no prose, no explanation — of exactly \
this form, with one entry per job above ("id" copied verbatim from that job's block):
{{"scores": [{{"id": "<job id>", "score": <int>, "reason": "<one sentence overall summary>", \
"breakdown": {{"skills": {{"score": <int>, "max": 60, "reason": "<sentence>"}}, \
"company": {{"score": <int>, "max": 10}}, "remote": {{"score": <int>, "max": 10}}, \
"role": {{"score": <int>, "max": 20, "reason": "<sentence>"}}}}}}, ...]}}"""

_JOB_BLOCK = """\
--- Job id={id} ---
Title: {title}
Company: {company}
Remote: {remote}
Description:
{description}
"""

_REFERRAL_PROMPT = (
    "You are helping a job seeker write a referral request message to a LinkedIn connection.\n"
    "Keep it under 150 words, warm but professional, mention the specific role.\n"
    "If a job URL is provided, include it naturally so the contact can view the listing.\n"
    "Output only the message text — no subject line, no greeting label.\n\n"
    "Candidate summary: {summary}\n"
    "Contact: {contact_name}, {contact_title} at {company}\n"
    "Role applying for: {job_title} at {company}\n"
    "Job URL: {job_url}\n"
)

_FORM_FIELD_MATCH_PROMPT = """\
You are matching job-application form fields on a webpage to a candidate's saved profile \
data slots. Below is a list of form fields scanned from the page, each with a tag_id, its \
HTML tag/input type, and a short "blob" of text scraped from its name/id/placeholder/\
aria-label/associated <label> text (lowercased).

Fields:
{fields_block}

Candidate data slots to fill, each needs AT MOST ONE field's tag_id (or none if no field \
clearly matches):
- email: the applicant's email address
- phone: the applicant's phone number
- linkedin: the applicant's LinkedIn profile URL
- first_name: the applicant's first/given name (a field asking ONLY for the first name)
- last_name: the applicant's last/family/surname (a field asking ONLY for the last name)
- full_name: the applicant's full name as one combined field (only if there is a single \
combined name field — do not use this if separate first/last name fields already cover it)
- resume: a file-upload field for the resume/CV document

Rules:
- Only map a field to a slot if the field's blob CLEARLY indicates it collects that exact \
piece of data. If you are not confident, leave the slot null — do not guess.
- Never map a field whose blob refers to something else, even if loosely related: company \
name, school/education, reference contact, emergency contact, cover letter, message/notes, \
salary/compensation expectation, recruiter/referrer name, portfolio/website (unless it \
explicitly says linkedin), veteran/disability/EEO status, or any other unrelated question.
- A field with tag=select (a dropdown) is almost never the right match for email/phone/name/\
linkedin — e.g. a "phone country code" dropdown also contains the word "phone" but is NOT \
the phone-number field. Only map a select field if its blob unmistakably means the exact \
data requested (this will be rare to never for these slots).
- Each field's tag_id may be used for AT MOST ONE slot — never reuse the same tag_id twice.
- Only output tag_ids that appear in the Fields list above — never invent one.
- If no field clearly matches a slot, leave that slot null (JSON null) rather than picking \
the closest-but-imperfect option.

Respond with ONLY a JSON object of exactly this form (JSON null, not the string "null", for \
any slot with no confident match):
{{"email": "<tag_id or null>", "phone": "<tag_id or null>", "linkedin": "<tag_id or null>", \
"first_name": "<tag_id or null>", "last_name": "<tag_id or null>", \
"full_name": "<tag_id or null>", "resume": "<tag_id or null>"}}
"""


# ── Pydantic models ────────────────────────────────────────────────────────────

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
    scores: list[JobScoreItem] = Field(default_factory=list)


class FormFieldMap(BaseModel):
    """One tag_id per candidate-data slot the LLM confidently matched among a
    scanned list of form fields. Any slot with no confident match is left
    null — the model must not guess by elimination."""
    model_config = ConfigDict(extra="ignore")

    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    linkedin: str | None = Field(default=None)
    first_name: str | None = Field(default=None)
    last_name: str | None = Field(default=None)
    full_name: str | None = Field(default=None)
    resume: str | None = Field(default=None)


class JobRequirements(BaseModel):
    """Structured extraction of a raw job description (see
    extract_job_requirements) — the only place raw JD text is read end to
    end. Powers structured scoring (score_jobs_structured) so later scoring
    calls never re-send raw text."""
    model_config = ConfigDict(extra="ignore")

    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    min_yoe: int | None = Field(default=None)
    max_yoe: int | None = Field(default=None)
    seniority_band: str | None = Field(default=None)   # intern/junior/mid/senior/staff/principal/director+
    location: str | None = Field(default=None)
    remote_policy: str | None = Field(default=None)     # remote/hybrid/onsite/unspecified
    work_auth: str | None = Field(default=None)         # e.g. "no sponsorship", "citizenship required"
    tech_stack: list[str] = Field(default_factory=list)
    company_type: str | None = Field(default=None)      # e.g. "big tech", "startup", "agency"
    company_size: str | None = Field(default=None)      # e.g. "1-50", "51-500", "500+"
    industry_domain: str | None = Field(default=None)
    key_responsibilities: list[str] = Field(default_factory=list)
    description: str | None = Field(
        default=None,
        description="A short, consistently-styled synopsis (2-4 plain-prose sentences) of the "
                    "role's core expectations and day-to-day responsibilities — not a copy of "
                    "the raw posting, and not company boilerplate (benefits, EEO statements, "
                    "culture blurbs, legal text). Written in the same length/tone/structure for "
                    "every job regardless of how differently formatted or verbose the original "
                    "source text was, so it reads consistently across jobs from different "
                    "sources (LinkedIn, Greenhouse, Lever, Ashby, JSearch, etc.) and is directly "
                    "comparable to a candidate's per-role resume_profile.work_summary entries "
                    "when matching a resume against this job.",
    )


class ResumeProfile(BaseModel):
    """Structured extraction of a candidate's resume text (see
    extract_resume_profile) — feeds structured scoring and, for the
    fill-in-application fields, a future apply.py form-fill data source."""
    model_config = ConfigDict(extra="ignore")

    # apply.py form-fill slots — field names match scanner/apply.py's _SLOTS
    full_name: str | None = Field(
        default=None,
        description="Candidate's full name, almost always the very first line of the resume "
                    "(the header), often in a larger font/all-caps — extract it even without an "
                    "explicit 'Name:' label.",
    )
    first_name: str | None = Field(
        default=None,
        description="Given name. If not stated as its own field, split it from full_name.",
    )
    last_name: str | None = Field(
        default=None,
        description="Family name. If not stated as its own field, split it from full_name.",
    )
    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    linkedin: str | None = Field(default=None)
    github: str | None = Field(default=None)

    # scoring-relevant fields, mirroring JobRequirements' shape
    years_exp: float | None = Field(default=None)
    seniority_band: str | None = Field(default=None)
    skills: list[str] = Field(default_factory=list)
    skill_years: dict[str, float] = Field(
        default_factory=dict,
        description="Approximate total years of hands-on experience per skill/technology named "
                    "in `skills`, computed from the dated work experience entries where that "
                    "skill was actually used (e.g. a skill used in a role from 2019-2022 and "
                    "again 2023-Present contributes ~4 years) — not from how prominently it's "
                    "listed in a standalone 'Skills' section. Overlapping/concurrent roles that "
                    "use the same skill should not be double-counted. Only include a skill here "
                    "if the resume's dated work history actually shows it being used; a skill "
                    "listed only in a bare skills list with no traceable dated usage can still "
                    "go in `skills` but should be omitted from this mapping.",
    )
    current_title: str | None = Field(
        default=None,
        description="Candidate's current or most recent job title/designation. Usually printed "
                    "right under or beside the name in the header (e.g. 'Jane Doe — Senior "
                    "Backend Engineer'); if the header has no title, use the job title of the "
                    "most recent (topmost/undated 'Present') entry in the work experience "
                    "section instead of leaving this null.",
    )
    current_company_type: str | None = Field(
        default=None,
        description="Category of the candidate's current/most recent employer (e.g. 'big tech', "
                    "'startup', 'agency', 'consultancy') inferred from that employer's name/"
                    "description — not the employer's name itself.",
    )
    work_auth: str | None = Field(default=None)
    location: str | None = Field(default=None)
    domains: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    work_summary: list[str] = Field(
        default_factory=list,
        description="One short summary per work experience entry, same order as listed on the "
                    "resume, each prefixed with company, title, AND the role's date range exactly "
                    "as written on the resume (e.g. 'Acme Corp — Senior Backend Engineer (Nov "
                    "2022 - Present): ...'). The date range is required, not optional — it's what "
                    "lets skill recency be judged later (a skill used in the most recent role "
                    "reads very differently from the same skill only appearing in a role from "
                    "several years ago). Cover whichever of these the resume actually describes "
                    "for that role: the technology/stack used, the architecture or system worked "
                    "on (scale, design, service boundaries), and the specific problem/challenge "
                    "tackled — skip whichever of the three isn't described rather than padding "
                    "with generic filler.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_alternate_shapes(cls, data):
        """extract_resume_profile runs under instructor's MD_JSON mode (a
        prompted-JSON mode, not tool-calling), so the model is free to
        organize richer resumes into nested groupings instead of this flat
        schema — e.g. a "contact": {...} object instead of top-level
        email/phone/linkedin, or a "work_experience": [{company, title,
        work_summary}, ...] array instead of the flat `work_summary` list.
        Without this, `extra="ignore"` silently drops those mismatched keys
        and the fields just come back null/empty — not a validation error,
        so it's easy to mistake for the model failing to find the data at
        all. Normalize the common alternate shapes here before validation.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        contact = data.pop("contact", None)
        if isinstance(contact, dict):
            for key in ("email", "phone", "linkedin", "github", "location", "full_name"):
                if not data.get(key) and contact.get(key):
                    data[key] = contact[key]

        work_experience = data.pop("work_experience", None)
        if isinstance(work_experience, list) and not data.get("work_summary"):
            summaries = []
            for entry in work_experience:
                if not isinstance(entry, dict):
                    continue
                company = entry.get("company") or ""
                prefix = " — ".join(
                    p for p in (company, entry.get("title")) if p
                )
                date_range = " - ".join(
                    str(p) for p in (entry.get("start_date"), entry.get("end_date")) if p
                )
                if date_range:
                    prefix = f"{prefix} ({date_range})" if prefix else f"({date_range})"
                summary = entry.get("work_summary") or entry.get("summary") or ""
                # The model is instructed to prefix work_summary text with
                # "Company — Title (dates): ..." itself, so when it also
                # nests entries under "work_experience" it sometimes embeds
                # that same prefix inside the per-entry summary text too —
                # don't prepend our own computed prefix on top of that.
                already_prefixed = bool(company) and company in summary[:len(prefix) + 20]
                if already_prefixed or not prefix:
                    summaries.append(summary)
                else:
                    summaries.append(f"{prefix}: {summary}")
            if summaries:
                data["work_summary"] = summaries

        education = data.get("education")
        if isinstance(education, list):
            flattened = []
            for entry in education:
                if isinstance(entry, str):
                    flattened.append(entry)
                elif isinstance(entry, dict):
                    parts = [entry.get("degree"), entry.get("field"), entry.get("institution")]
                    date_range = "-".join(
                        str(p) for p in (entry.get("start_date"), entry.get("end_date")) if p
                    )
                    if date_range:
                        parts.append(f"({date_range})")
                    if entry.get("gpa"):
                        parts.append(f"GPA: {entry['gpa']}")
                    flattened.append(", ".join(p for p in parts if p))
            if flattened:
                data["education"] = flattened

        # The model reliably returns full_name but often skips splitting it
        # into first_name/last_name even though the field descriptions ask
        # for it — split deterministically here rather than depending on
        # the model to do it consistently.
        full_name = data.get("full_name")
        if isinstance(full_name, str) and full_name.strip():
            parts = full_name.split()
            if not data.get("first_name") and parts:
                data["first_name"] = parts[0]
            if not data.get("last_name") and len(parts) > 1:
                data["last_name"] = " ".join(parts[1:])

        return data

# ── Helpers ────────────────────────────────────────────────────────────────────

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

_BREAKER_FAILURE_THRESHOLD = 2      # consecutive rate-limit failures before tripping
_BREAKER_COOLDOWN_S        = 300    # 5 minutes


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a rate-limit/overload/cooldown-style failure,
    as opposed to e.g. a validation error or a one-off timeout. litellm's
    exceptions (litellm.RateLimitError etc.) already subclass the equivalent
    openai exception, but check litellm's directly rather than lean on that
    inheritance detail. status_code is checked via getattr rather than an
    isinstance(..., APIStatusError) class check, since it's present on both
    openai's and litellm's exception hierarchies."""
    if isinstance(exc, litellm.RateLimitError):
        return True
    if getattr(exc, "status_code", None) in (429, 503, 529):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "rate limit", "rate_limit", "cooldown", "cool down", "overloaded",
        "too many requests", "429", "503", "529",
    ))


def _breaker_record_failure(provider: str, exc: Exception) -> None:
    if not _is_rate_limit_error(exc):
        return  # other failure kinds (bad JSON, one-off timeout) don't trip the breaker
    with _breaker_lock:
        state = _breaker_states.setdefault(provider, {"consecutive_failures": 0, "open_until": 0.0, "last_error": ""})
        state["consecutive_failures"] += 1
        state["last_error"] = str(exc)
        if state["consecutive_failures"] >= _BREAKER_FAILURE_THRESHOLD:
            state["open_until"] = time.time() + _BREAKER_COOLDOWN_S


def _breaker_record_success(provider: str) -> None:
    with _breaker_lock:
        state = _breaker_states.setdefault(provider, {"consecutive_failures": 0, "open_until": 0.0, "last_error": ""})
        state["consecutive_failures"] = 0
        state["open_until"] = 0.0


def provider_breaker_status(provider: str) -> dict:
    """Check the breaker status for a specific provider. Returns
    {"open": bool, "retry_at": float|None, "retry_in_s": int|None, "reason": str}
    """
    with _breaker_lock:
        state = _breaker_states.get(provider, {"consecutive_failures": 0, "open_until": 0.0, "last_error": ""})
        open_until = state["open_until"]
        if open_until and time.time() < open_until:
            return {
                "open": True,
                "retry_at": open_until,
                "retry_in_s": int(open_until - time.time()),
                "reason": state["last_error"],
            }
        return {"open": False, "retry_at": None, "retry_in_s": None, "reason": ""}


def scoring_breaker_status() -> dict:
    """Check before attempting to score. Returns
    {"open": bool, "retry_at": float|None, "retry_in_s": int|None, "reason": str}
    so callers (e.g. the UI) can show "scoring not available" instead of
    trying and failing against a model that's still in cooldown.
    """
    provider = _provider()
    return provider_breaker_status(provider)


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
        raise RuntimeError(f"LLM unavailable — {provider} in cooldown (~{breaker['retry_in_s']}s remaining): {breaker['reason']}")

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


# ── Public functions ───────────────────────────────────────────────────────────

@observe(name="generate_summary")
def generate_summary(resume_text: str) -> str:
    def _query(client, model):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(resume_text=resume_text[:8000])}],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    return execute_with_breaker(_query)


_JD_EXTRACT_MAX_CHARS = 100000

_JD_EXTRACT_PROMPT = """\
Extract structured information from the following raw job description. Only use information \
explicitly present in the text — leave a field null/empty if it isn't stated; do not guess or \
infer facts not present.

Also write `description`: a short synopsis (2-4 plain-prose sentences) covering only the role's \
core expectations and day-to-day responsibilities — skip benefits, EEO statements, culture blurbs, \
and other company boilerplate that isn't specific to what this person would actually do. Keep the \
same length and tone regardless of how long, short, or differently formatted this particular \
posting is, so descriptions read consistently across jobs from different sources.

For `company_type`/`company_size`: most job descriptions don't state these explicitly. The \
hiring company's name is given below as "Company". If you recognize this company from your own \
general knowledge (e.g. well-known big tech, a large/well-funded startup or unicorn, a well-known \
but smaller/early-stage startup), use that knowledge to fill company_type/company_size — the same \
way an experienced recruiter would recognize a company by name — even though that knowledge isn't \
in the JD text itself. Only leave these null if the company name is generic/unrecognizable and \
the JD text also says nothing about company size/stage.

Company: {company}

Job Description:
{description}
"""


@observe(name="extract_job_requirements")
def extract_job_requirements(description: str, company: str | None = None) -> JobRequirements:
    """The only function that reads a raw job description end to end. Always
    uses Gemini (provider_override="gemini") with the cheap "extract" model
    class (GEMINI_EXTRACT_MODEL) regardless of LLM_PROVIDER — Gemini is the
    designated JD extraction provider while Claude handles resume extraction
    and scoring. `company` is the job's company name (not JD text) — passed
    through so the model can use its own general knowledge of named companies
    to fill company_type/company_size, the same way score_jobs()'s raw path
    recognizes company names directly."""
    prompt = _JD_EXTRACT_PROMPT.format(description=description, company=company or "Not specified")

    def _query(client, model):
        max_tokens = 4000
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=JobRequirements,
                    max_tokens=max_tokens,
                    max_retries=2,
                    timeout=45,
                )
            except IncompleteOutputException:
                if attempt == 0:
                    max_tokens *= 2
                else:
                    raise

    return execute_with_breaker(_query, is_instructor=True, model_class="extract",
                                provider_override="gemini")


_RESUME_EXTRACT_PROMPT = """\
Extract structured candidate information from the following resume text. Only use information \
explicitly present in the text — leave a field null/empty if it isn't stated; do not guess or \
infer facts not present.

The resume's header (its first few lines) is the primary source for full_name and current_title, \
even when it has no labels like "Name:" — treat a standalone line at the very top, often in a \
larger font or all-caps, as the name. A short line right after it (or next to it, or separated by \
a pipe/dash from the name) is usually the current title/designation. If the header has no title, \
fall back to the job title of the most recent (topmost, or marked "Present") entry in the work \
experience section — don't leave current_title null just because it's absent from the header.

For phone/email/linkedin/github, extract exactly as written (don't reformat). For each work \
experience entry, generate one short summary in work_summary, PREFIXED with company, title, and \
that role's date range exactly as written on the resume (e.g. "Acme Corp — Senior Backend \
Engineer (Nov 2022 - Present): ..." or "(2019-2022): ..." if company/title aren't stated) — the \
date range is required on every entry, since it's how a later comparison can tell whether a skill \
was used recently or only in an old role. After the date-range prefix, cover whichever of the \
technology/stack used, the architecture or system worked on (scale, design, service boundaries), \
and the specific problem/challenge that role tackled the resume actually describes; don't invent \
or pad with generic filler for whichever of those three isn't stated for that role.

Duration matters as much as the words used, so read the work experience section's dated entries \
(not just a standalone "Skills" list) to fill skill_years: for each skill/technology actually used \
in a dated role, add up the years of the date range(s) where it was used (e.g. "used in a role \
from 2019-2022" ≈ 3 years); if the same skill recurs in a later role, add that range too, but \
don't double-count time from concurrent/overlapping roles. A skill only ever mentioned in a bare \
skills list, with no traceable dated role behind it, still belongs in `skills` but should be left \
out of skill_years rather than guessed at. This also feeds years_exp and seniority_band — base \
those on the full span of dated work experience, not just the most recent role.

Resume:
{resume_text}
"""


@observe(name="extract_resume_profile")
def extract_resume_profile(resume_text: str) -> ResumeProfile:
    """Structured extraction of resume text. Uses the "structured_score"
    (Sonnet-class) model tier rather than the cheap "extract" tier used by
    extract_job_requirements — there's only one resume per candidate (versus
    many JDs), so it's worth spending more on getting it right."""
    prompt = _RESUME_EXTRACT_PROMPT.format(resume_text=resume_text)

    def _query(client, model):
        max_tokens = 10000  # skill_years/work_summary (one entry per role) can push a
                             # multi-role resume's output well past what a short profile needs
        for attempt in range(3):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=ResumeProfile,
                    max_tokens=max_tokens,
                    max_retries=2,
                    timeout=45,
                )
            except IncompleteOutputException:
                if attempt < 2:
                    max_tokens *= 2
                else:
                    raise

    return execute_with_breaker(_query, is_instructor=True, model_class="structured_score")


_MAX_BATCH_CHARS = 3_999_000  # ~1M-token context budget at ~4 chars/token, minus a small safety margin
_MAX_JOBS_PER_BATCH = 5      # also cap by job count, not just chars — keeps individual calls'
                              # output size (and therefore wall-clock time) bounded, and ensures
                              # a large scoring run actually splits into multiple batches that can
                              # run concurrently instead of one huge serial call
_BASE_TIMEOUT_S     = 60      # per-call timeout floor
_PER_JOB_TIMEOUT_S  = 10      # extra timeout budget per job in the batch
_MAX_TIMEOUT_S      = 600     # hard ceiling regardless of batch size
_HEARTBEAT_S        = 15      # how often to log "still running" while a call is in flight


def _format_job_block(job: dict) -> str:
    return _JOB_BLOCK.format(
        id=job["id"],
        title=job.get("title", ""),
        company=job.get("company", ""),
        remote="yes" if job.get("is_remote") else "no",
        description=(job.get("description") or "").strip(),
    )


def _build_batches(summary: str, jobs: list[dict]) -> list[list[dict]]:
    """Greedily group jobs so each batch's full prompt stays under
    _MAX_BATCH_CHARS AND under _MAX_JOBS_PER_BATCH jobs — whichever limit is
    hit first starts a new batch. A job's description is never split/truncated
    to fit — if the next job doesn't fit in the current batch, it starts the
    next one. A job whose own block alone exceeds the char budget is skipped
    (can't be scored without violating "no partial descriptions").
    """
    header_overhead = len(_BATCH_SCORE_SYSTEM_PROMPT.format(rubric=_SCORE_RUBRIC, summary=summary[:1500])) + \
        len(_BATCH_SCORE_USER_PROMPT.format(n=0, jobs_block=""))

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = header_overhead

    for job in jobs:
        block = _format_job_block(job)
        block_len = len(block)

        if header_overhead + block_len > _MAX_BATCH_CHARS:
            warnings.warn(
                f"Job '{job.get('title')}' description too large to score "
                f"({block_len:,} chars) — skipped."
            )
            continue

        would_exceed_chars = current and current_chars + block_len > _MAX_BATCH_CHARS
        would_exceed_count = current and len(current) >= _MAX_JOBS_PER_BATCH
        if would_exceed_chars or would_exceed_count:
            batches.append(current)
            current = []
            current_chars = header_overhead

        current.append(job)
        current_chars += block_len

    if current:
        batches.append(current)

    return batches


@observe(name="score_batch")
def _score_batch(summary: str, batch: list[dict], log_fn=None, label: str = "",
                  cancel_event: threading.Event | None = None) -> list[dict]:
    """Score every job in `batch` with a single LLM call.

    Bounded by a timeout scaled to batch size (so a hung request fails loudly
    instead of blocking forever), and logs a heartbeat every _HEARTBEAT_S
    seconds while the call is in flight so a slow-but-working batch doesn't
    look indistinguishable from a stuck one. Skips outright (no API call) if
    cancel_event is already set, or if the circuit breaker is open (model in
    cooldown from recent rate-limit failures).

    Returns a list of {id, score, reason, breakdown} dicts for jobs the model
    actually returned a score for (unmatched/failed entries are dropped).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(f"{label}{msg}")

    if cancel_event is not None and cancel_event.is_set():
        _log("Skipped — cancelled.")
        return []

    jobs_block = "\n".join(_format_job_block(job) for job in batch)
    system_prompt = _BATCH_SCORE_SYSTEM_PROMPT.format(rubric=_SCORE_RUBRIC, summary=summary[:1500])
    user_prompt = _BATCH_SCORE_USER_PROMPT.format(n=len(batch), jobs_block=jobs_block)
    timeout_s = min(_MAX_TIMEOUT_S, _BASE_TIMEOUT_S + _PER_JOB_TIMEOUT_S * len(batch))

    _log(f"Starting: {len(batch)} job(s), {len(system_prompt) + len(user_prompt):,} chars, timeout={timeout_s}s…")

    stop_heartbeat = threading.Event()
    t0 = time.time()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(_HEARTBEAT_S):
            _log(f"Still running… ({int(time.time() - t0)}s elapsed, timeout at {timeout_s}s)")

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    def _query(client, model):
        max_tokens = min(1000 * len(batch) + 500, 100_000)
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    # Prompt-cache the system message (rubric + candidate profile —
                    # identical across every batch call in a scoring run) so repeat
                    # calls within the run read from cache instead of reprocessing
                    # ~1,800 tokens of static rubric text every time.
                    cache_control_injection_points=[{"location": "message", "role": "system"}],
                    response_model=BatchScoreResult,
                    max_tokens=max_tokens,
                    max_retries=3,
                    timeout=timeout_s,
                )
            except IncompleteOutputException:
                if attempt == 0:
                    max_tokens = min(max_tokens * 2, 100_000)
                    _log(f"Output truncated at max_tokens — retrying with max_tokens={max_tokens}…")
                else:
                    raise

    try:
        result = execute_with_breaker(_query, is_instructor=True, log_fn=_log)
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1)
    _log(f"Done in {time.time() - t0:.1f}s — {len(result.scores)} score(s) returned.")

    by_id = {job["id"] for job in batch}
    out = []
    for item in result.scores:
        if item.id not in by_id or item.score is None:
            continue
        bd = item.breakdown
        # Enforce score = sum of breakdown to catch LLM arithmetic errors
        computed = bd.skills.score + bd.company.score + bd.remote.score + bd.role.score
        out.append({
            "id": item.id,
            "score": computed,
            "reason": item.reason,
            "breakdown": json.dumps(bd.model_dump()),
        })
    return out


def score_jobs(summary: str, jobs: list[dict], log_fn=None,
               cancel_event: threading.Event | None = None) -> Iterator[list[dict]]:
    """Score jobs by packing as many COMPLETE job descriptions as possible into
    each LLM call (up to _MAX_BATCH_CHARS characters and _MAX_JOBS_PER_BATCH
    jobs — a job is never split or truncated to fit; one that doesn't fit
    starts the next batch). Batches run concurrently in threads (up to
    BATCH_SIZE at once) whenever there's more than one, each with its own
    timeout and heartbeat logging. Yields the list of scored dicts for each
    batch as it completes.

    Each job dict must have: id, title, company, description, is_remote.
    If given, log_fn(str) is called with progress messages from every batch
    (prefixed per-batch) — safe to call from any thread, since each batch
    runs in its own worker thread and log_fn is expected to just append to a
    shared, thread-safe sink (e.g. a locked list), not touch UI widgets directly.
    If given, cancel_event lets a caller stop early (checked before starting
    each batch's API call, and again before yielding each completed result) —
    batches already in flight when cancelled still finish in the background,
    but their results aren't waited on or yielded, and no new batch starts.
    Skips everything outright (yields nothing) if the circuit breaker is open
    (model in a rate-limit cooldown — see scoring_breaker_status()).
    A failed batch yields [] and logs a warning.
    """
    batches = _build_batches(summary, jobs)
    if not batches:
        return

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    breaker = scoring_breaker_status()
    if breaker["open"]:
        _log(f"Scoring unavailable — model in cooldown for ~{breaker['retry_in_s']}s "
             f"({breaker['reason']}).")
        return

    _log(f"{len(jobs)} job(s) split into {len(batches)} batch(es) "
         f"(up to {min(BATCH_SIZE, len(batches))} running concurrently)…")

    _warm_up_litellm(_provider())

    # Not a `with` block on purpose: exiting `with ThreadPoolExecutor()` calls
    # shutdown(wait=True), which blocks until every in-flight batch finishes —
    # defeating cancellation entirely (this function wouldn't return until
    # all batches were done regardless). Managing the pool manually lets the
    # cancelled path shut down without waiting for already-running batches.
    pool = ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(batches)))
    try:
        futures = {
            pool.submit(_score_batch, summary, batch, log_fn,
                        f"[Batch {i}/{len(batches)}] ", cancel_event): batch
            for i, batch in enumerate(batches, start=1)
        }
        pending = set(futures)
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                _log("Cancelled — stopping (any already-running batches finish in the "
                     "background, but their results are discarded).")
                break
            # wait() with a short timeout (rather than as_completed(), which only
            # returns control to this loop once a future actually finishes) is
            # what makes the cancel_event check above actually prompt — with
            # every batch already running and none done yet, as_completed()
            # would otherwise block for however long the slowest one takes
            # before this loop got a chance to notice cancellation at all.
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                batch = futures[future]
                try:
                    yield future.result()
                except Exception as e:
                    _log(f"Batch scoring failed ({len(batch)} job(s)): {e}")
                    warnings.warn(f"Batch scoring failed ({len(batch)} job(s)): {e}")
                    yield []
    finally:
        # cancel_futures=True drops any not-yet-started batch immediately;
        # wait=False means we don't block on ones already running.
        pool.shutdown(wait=False, cancel_futures=True)


# ── Structured scoring (SCORING_MODE=structured) ──────────────────────────────
# Parallel implementation to score_jobs()/_score_batch() above, scoring
# structured JD/resume JSON instead of raw description text. Kept as a fully
# separate code path (not a shared parametrized core) so a bug here can never
# affect the raw-text scoring path, and vice versa.

_STRUCTURED_SCORE_RUBRIC = """\
Judge four categories per job — for EACH one, pick exactly one tier/label and explain your pick \
in one sentence. Do NOT compute or report any point values, sub-scores, or totals: you are \
providing categorical judgments only, and the numeric score is computed deterministically \
afterward, in Python, from the tier(s) you pick — never from anything you add up yourself. Each \
job is provided as structured JSON (job_requirements) instead of raw description text; judge based \
on the structured fields below, not on any text you might otherwise expect to see.

- skills: judge core discipline/architecture fit (`core_fit`) and judge each must-have requirement.
    * `core_fit` (pick exactly one, based ONLY on engineering discipline/architecture fit — \
years-of-experience and seniority-level fit are judged separately, never here, to avoid scoring the \
same signal twice):
        - "core_match": job_requirements implies the same overall engineering discipline/domain as \
resume_profile's background (backend/distributed-systems/platform engineering — NOT a fundamentally \
different discipline like QA/test-automation, frontend-only, mobile-only, data-science/ML-research, \
etc.) with a matching general architecture pattern (microservices, event-driven pipelines, \
distributed data systems).
        - "adjacent": the discipline is adjacent-but-different (overlapping but with a real \
architectural mismatch).
        - "mismatch": an actual different engineering discipline entirely.
      Use the job's raw `title` (not JD text) as the PRIMARY signal. If `title` reads as a general \
software-engineering title (Software Engineer, Software Development Engineer, Backend Engineer, \
Platform Engineer, Full-Stack Engineer, etc. — no QA/test-specific qualifier), judge `core_fit` \
purely on the discipline+architecture match above — do NOT downgrade it just because generic \
software-development-lifecycle duties (writing/maintaining tests, debugging, code review, \
developer-tooling/DX work, mentoring, on-call/incident response) appear in must_haves/\
key_responsibilities. These are normal responsibilities of any senior/staff engineer in any \
discipline and are not, by themselves, a QA/test-automation signal. If `title` explicitly signals a \
different discipline (SDET, QA Engineer, Test Engineer, Quality Assurance Engineer, Automation \
Engineer, or similar), judge `core_fit` for THAT discipline's fit against resume_profile's \
background instead — typically "mismatch" or "adjacent" for a backend-focused candidate, since the \
title itself confirms a genuinely different discipline regardless of how much backend-adjacent \
architecture the JD also describes. Explain your pick in `core_fit_reason`.
    * `must_have_judgments`: for EVERY item in job_requirements.must_haves, produce exactly one \
judgment — same count, same order — deciding a `match_fraction` and a one-line `note`. \
`match_fraction` MUST be exactly one of 0.0, 0.4, 0.6, 0.8, or 1.0 (never any other number) — pick \
the one that best represents how fully resume_profile.skills / domains / skill_years satisfies this \
specific requirement, directly or via a clear transferable/adjacent skill:
        - 1.0 — fully satisfied: every named component is directly covered. Also 1.0 for an \
"or"/alternative requirement (multiple acceptable options joined by "or", or slash-separated, e.g. \
"Java, Python, Node.js, or Ruby/Rails", or "AWS/Azure/GCP") the moment resume_profile matches ANY \
ONE of the listed options — this is one requirement with several acceptable answers, not several \
independent requirements, so a single matched option makes it fully satisfied, never partial, and \
you must NOT treat the other, unmatched options as reducing the fraction or as separate gaps.
        - 0.8 — nearly fully satisfied: the requirement bundles multiple distinct components joined \
by "and" (not "or"), and only one minor, clearly adjacent/learnable component is missing from an \
otherwise well-covered set.
        - 0.6 — majority satisfied: most named components in an "and"-bundled requirement are \
directly covered, but a meaningful piece is missing that is still a clear, adjacent/learnable \
extension of something resume_profile already shows (e.g. Kubernetes vs. Docker — both \
container-tooling, one builds directly on the other; a specific cloud vendor vs. a different one \
already used) — a single missing-but-learnable component should NOT zero out an otherwise \
well-covered bundled requirement; this is exactly the case 0.6 exists for.
        - 0.4 — partially satisfied: some real overlap exists, but a substantial part of the \
requirement is unaddressed and not a clearly adjacent skill.
        - 0.0 — not satisfied: no meaningful coverage, or what's missing is not a clear extension \
of anything on the resume (a genuinely different domain, e.g. healthcare-compliance knowledge or \
LLM/RAG tooling for a candidate with none of it).
        - Years-of-experience: if a must_have is PURELY a numeric years-of-experience threshold \
(e.g. "10+ years of software development experience"), omit it from must_have_judgments entirely — \
it's handled separately by the HARD GATE below, using the real min_yoe/years_exp numbers, and \
including it here would double-count the same signal. If a must_have COMBINES a years-of-experience \
number with other substantive content (e.g. "10+ years, including 2+ years in a Staff/Principal \
role"), strip the numeric-years part but still judge the substantive remainder normally (e.g. does \
resume_profile show equivalent Staff/Principal-level scope?).
        - Only skip an item under the above rule — every other must_have must get a judgment, even \
ones you're not fully certain about (use your best read of resume_profile). Most engineers pick up \
peripheral tooling gaps within weeks once the core discipline already matches, so weigh that when \
picking a fraction on a borderline call, but still record a verdict either way.
    * `nice_to_have_matches`: name any job_requirements.nice_to_haves items resume_profile clearly \
covers (up to 6) — this is a bonus list only. Do NOT name unmatched nice-to-haves; missing a \
nice-to-have is never a gap, by definition of "nice to have".
- company: pick a `tier` from job_requirements.company_type/company_size — "big_or_funded" for big \
tech or a large/well-funded startup; "mid_sized" for an established, recognizable company that \
isn't big-tech scale but also isn't early-stage/high-risk (job_requirements.company_size saying \
something like "Mid-sized" or "Enterprise" without being a big-tech name is the signal — this is a \
DIFFERENT tier from "smaller_startup" below, not the same bucket: a stable, years-old private \
company is not the same risk/scale profile as a small startup, even though neither is big tech); \
"smaller_startup" for a well-known but genuinely small/early-stage startup; "unknown" otherwise. \
Each job also carries its raw `company` name (the actual hiring company, not JD text) — if \
company_type/company_size are null/unclear, use your own general knowledge of that named company \
to judge its scale instead of defaulting to "unknown".
- remote: pick a `tier` from job_requirements.remote_policy — "remote", "hybrid", or "onsite" (use \
"onsite" for unspecified too). The job's own `is_remote` flag, if true, deterministically overrides \
your pick and forces the top score regardless — you don't need to special-case it, just judge \
remote_policy honestly from the structured text.
- role: pick a `tier` comparing job_requirements.seniority_band against resume_profile.\
seniority_band using the same company-scale-ceiling logic as the company category: the candidate's \
demonstrated ceiling is Senior at big tech/large well-funded startups AND at established mid-sized \
companies (mid-sized, years-old private/enterprise companies tend to level titles more formally, \
closer to how a larger org does, NOT with the flat structure of an early-stage startup — being \
"not big tech" does not by itself raise the ceiling), and Staff only at genuinely small/early-stage \
startups (a truly small, flat-structured company gives more scope per level, so the ceiling there \
is one level higher). Each job also carries its raw `title` (the actual job title text, not JD \
text) — if job_requirements.seniority_band is null/ambiguous, read the level directly off `title` \
instead (e.g. "Staff Software Engineer", "Principal Engineer", "Senior Software Development \
Engineer").
  Company scale (from job_requirements.company_type/company_size, or your own knowledge of the \
named `company` when those are unclear — same as the company category) sets a DEFAULT ceiling \
assumption, but organizations vary widely in how they actually level titles, so weigh two more \
signals before picking a tier when the title nominally exceeds that default ceiling:
    * job_requirements.min_yoe AND max_yoe together — a stated, bounded band like "4–8 years" for \
a Staff/Principal title is a modest ask, evidence this org's bar is more attainable than the \
default FAANG-style assumption; an unstated or high min_yoe with no max_yoe reinforces the strict \
default instead.
    * job_requirements.company_type and the raw `company` name itself — use your own knowledge of \
the specific named company/industry (not just its size) to judge whether it actually levels as \
aggressively as a pure big-tech/FAANG-style company. A large fintech/payments/enterprise-IT \
organization, for instance, doesn't necessarily level Staff/Principal titles as strictly as a \
similarly-sized pure tech company.
    * "at_or_below_ceiling": title is at or below the demonstrated ceiling given the full picture \
(company scale/type/name and the stated yoe band together).
    * "ambiguous_scale": title nominally exceeds the default ceiling by company scale, but the yoe \
band (min+max) and/or company type/name evidence suggests this org's bar is more attainable than \
the strict default — a genuine reach, not a near-impossible one. Also covers company scale itself \
being ambiguous/borderline.
    * "exceeds_ceiling": title clearly exceeds the ceiling with no mitigating evidence from the yoe \
band or company type/name — a real, largely unmitigated reach.
  Give a one-sentence `reason` naming the title level, the company's scale/type, and — when it \
drives the tier choice — what the yoe band or company type/name implies about this specific org's \
own bar for the title.
Also give one `overall_reason`: a single sentence summarizing the match across all four categories."""

# Split into system (rubric + resume_json — identical across every batch call
# in a scoring run) and user (job data + response format — different per
# batch) parts, same rationale as _BATCH_SCORE_SYSTEM_PROMPT/_USER_PROMPT
# above: lets the system part be prompt-cached via litellm's
# cache_control_injection_points instead of resending ~2,600 tokens of
# static rubric text (plus the resume JSON) on every single batch call.
_STRUCTURED_BATCH_SCORE_SYSTEM_PROMPT = """\
You are a recruiter judging job matches for a candidate against MULTIPLE jobs in a single pass, \
using STRUCTURED data instead of raw description text. Judge EVERY job listed below independently \
against the categories below — you are providing categorical judgments only, never point values or \
totals (those are computed deterministically afterward from the tiers you pick).

{rubric}

Candidate profile (structured JSON, referred to as resume_profile above):
{resume_json}"""

_STRUCTURED_BATCH_SCORE_USER_PROMPT = """\
There are {n} job(s) below, each a structured JSON object (referred to as job_requirements above) \
with an "id" field copied verbatim into your response. Judge every one of them — do not skip any \
and do not invent jobs that aren't listed.

{jobs_block}

IMPORTANT: Respond with ONLY a JSON object — no markdown, no prose, no explanation — of exactly \
this form, with one entry per job above ("id" copied verbatim from that job's block):
{{"scores": [{{"id": "<job id>", \
"skills": {{"core_fit": "core_match|adjacent|mismatch", "core_fit_reason": "<sentence>", \
"must_have_judgments": [{{"requirement": "<verbatim or short paraphrase>", \
"match_fraction": 0.0|0.4|0.6|0.8|1.0, "note": "<sentence>"}}, ...], \
"nice_to_have_matches": [{{"name": "<item>", "note": "<sentence>"}}, ...]}}, \
"company": {{"tier": "big_or_funded|mid_sized|smaller_startup|unknown", "reason": "<sentence>"}}, \
"remote": {{"tier": "remote|hybrid|onsite", "reason": "<sentence>"}}, \
"role": {{"tier": "at_or_below_ceiling|ambiguous_scale|exceeds_ceiling", "reason": "<sentence>"}}, \
"overall_reason": "<one sentence overall summary>"}}, ...]}}"""


# ── Structured-mode atomic judgment models ──────────────────────────────────
# The LLM never outputs a point value anywhere below — only tier labels
# (Literal fields, enforced by instructor.Mode.TOOLS at the schema level) and
# short named-item lists. All numeric scoring lives in
# _compute_structured_breakdown(), so the LLM's prose and the stored score
# can never diverge (see CLAUDE.md / conversation history for the bug this
# replaces: the old design let the LLM compute CORE_BASE+COVERAGE_BONUS
# itself and self-report a number that didn't have to match its own reason).

class StructuredSkillItem(BaseModel):
    """One named must-have/tech-stack item — used for BOTH matches and gaps
    (same shape either way, just placed in a different list)."""
    model_config = ConfigDict(extra="ignore")
    name: str = Field(default="")
    note: str = Field(default="")


class StructuredRequirementJudgment(BaseModel):
    """One verdict for a single job_requirements.must_haves item (the
    requirement text, verbatim or lightly paraphrased) — a quantized match
    fraction, not a free-form number: this is still an atomic pick from a
    small fixed set (enforced by instructor.Mode.TOOLS as a JSON-schema
    enum), the same kind of safe choice as the tier Literal fields
    elsewhere in this module, NOT the LLM performing arithmetic — Python
    still does 100% of the summing. An "or"/slash-separated requirement
    (e.g. "Java, Python, Node.js, or Ruby/Rails") is ONE judgment at 1.0 if
    any option is covered, not split into per-option items: judging at the
    must_haves sentence level (instead of against the already-flattened
    tech_stack list, which destroys "or" groupings into independent
    tokens) is what keeps a candidate who matches just one option from
    being scored as missing all the others. An "and"-bundled requirement
    (e.g. "microservices, API design, Kubernetes, Docker, messaging
    systems, and distributed architectures") can land at 0.4/0.6/0.8 when
    most-but-not-all named components are covered and the remainder is a
    clear, adjacent/learnable extension of something the resume already
    shows (e.g. Kubernetes vs. Docker) — this is what stops a single
    missing, learnable component from zeroing an otherwise well-covered
    bundled requirement."""
    model_config = ConfigDict(extra="ignore")
    requirement: str = Field(default="")
    match_fraction: Literal[0.0, 0.4, 0.6, 0.8, 1.0] = Field(default=0.0)
    note: str = Field(default="")


class StructuredSkillsJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    core_fit: Literal["core_match", "adjacent", "mismatch"] = Field(default="mismatch")
    core_fit_reason: str = Field(default="")
    must_have_judgments: list[StructuredRequirementJudgment] = Field(default_factory=list)
    nice_to_have_matches: list[StructuredSkillItem] = Field(default_factory=list)

    @field_validator("must_have_judgments", mode="after")
    @classmethod
    def _cap_must_haves(cls, v: list["StructuredRequirementJudgment"]) -> list["StructuredRequirementJudgment"]:
        return v[:12]  # generous — typical extracted must_haves lists run ~3-11 items

    @field_validator("nice_to_have_matches", mode="after")
    @classmethod
    def _cap_nice_to_haves(cls, v: list["StructuredSkillItem"]) -> list["StructuredSkillItem"]:
        # Match-only, no gap counterpart — missing a nice-to-have is never a
        # penalty by definition, so there's nothing to symmetrically cap against.
        return v[:6]


class StructuredCompanyJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tier: Literal["big_or_funded", "mid_sized", "smaller_startup", "unknown"] = Field(default="unknown")
    reason: str = Field(default="")


class StructuredRemoteJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tier: Literal["remote", "hybrid", "onsite"] = Field(default="onsite")
    reason: str = Field(default="")


class StructuredRoleJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tier: Literal["at_or_below_ceiling", "ambiguous_scale", "exceeds_ceiling"] = Field(default="ambiguous_scale")
    reason: str = Field(default="")


class StructuredJobScoreItem(BaseModel):
    """One job's atomic judgments within a structured-mode batch response —
    NO point values anywhere. Numeric scoring is 100% deterministic Python
    (_compute_structured_breakdown), so the LLM's prose and the stored score
    can never diverge again."""
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default="")
    skills: StructuredSkillsJudgment = Field(default_factory=StructuredSkillsJudgment)
    company: StructuredCompanyJudgment = Field(default_factory=StructuredCompanyJudgment)
    remote: StructuredRemoteJudgment = Field(default_factory=StructuredRemoteJudgment)
    role: StructuredRoleJudgment = Field(default_factory=StructuredRoleJudgment)
    overall_reason: str = Field(default="")


class StructuredBatchScoreResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scores: list[StructuredJobScoreItem] = Field(default_factory=list)


_CORE_BASE_BY_TIER     = {"core_match": 35, "adjacent": 20, "mismatch": 0}
_NICE_TO_HAVE_BONUS_PER_ITEM = 1
_NICE_TO_HAVE_BONUS_CAP = 3
_COMPANY_SCORE_BY_TIER = {"big_or_funded": 10, "mid_sized": 7, "smaller_startup": 5, "unknown": 0}
_COMPANY_TIER_LABEL    = {"big_or_funded": "big tech / large well-funded startup",
                           "mid_sized": "established mid-sized company",
                           "smaller_startup": "smaller/early-stage startup", "unknown": "unrecognized company"}
_REMOTE_SCORE_BY_TIER  = {"remote": 10, "hybrid": 5, "onsite": 0}
_REMOTE_TIER_LABEL     = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite/unspecified"}
# Structured rubric's own bands (15-20 / 8-14 / 0-7) -> near-top / band-midpoint / near-bottom picks.
_ROLE_SCORE_BY_TIER    = {"at_or_below_ceiling": 18, "ambiguous_scale": 11, "exceeds_ceiling": 3}
_ROLE_TIER_LABEL       = {"at_or_below_ceiling": "at or below demonstrated ceiling",
                           "ambiguous_scale": "nominally exceeds ceiling but mitigated by yoe/company evidence",
                           "exceeds_ceiling": "clearly exceeds ceiling, unmitigated"}


def _compute_structured_breakdown(item: "StructuredJobScoreItem", job: dict, resume: dict) -> dict:
    """Deterministically compute all four category scores — and a reason
    string built from those SAME numbers — from the LLM's atomic tier/gap
    judgments in `item`, the job dict (requirements/is_remote/title/company
    — see score_jobs_structured), and `resume` (a ResumeProfile.model_dump()
    dict). This is the fix for the root-cause bug: the LLM's prose and its
    numeric score used to be two independent, unsynchronized outputs; now
    both are derived from the same local variables in one place.

    Returns {"skills": {"score": int, "max": 60, "reason": str, ...extra
    debug keys...}, "company": {...}, "remote": {...}, "role": {...}} — the
    same 4-key/score+max+reason shape database.update_structured_scores()
    and llm.parse_score_breakdown() already expect (parse_score_breakdown
    only reads bd[key]["score"/"max"/"reason"] for these 4 keys, so any
    extra keys added here for debuggability are inert JSON).
    """
    requirements = job.get("requirements") or {}
    min_yoe = requirements.get("min_yoe")
    years_exp = resume.get("years_exp")

    core_fit = item.skills.core_fit
    core_base = _CORE_BASE_BY_TIER[core_fit]

    judgments = item.skills.must_have_judgments
    total_must_haves = len(judgments)
    # match_fraction IS the per-requirement credit — no separate lookup table
    # translating a label to a number, so there's one fewer place for the
    # stored reason and the stored score to ever disagree.
    matched_credit = sum(j.match_fraction for j in judgments)
    full_reqs = [j.requirement for j in judgments if j.match_fraction == 1.0 and j.requirement]
    partial_reqs = [(j.requirement, j.match_fraction) for j in judgments
                     if 0.0 < j.match_fraction < 1.0 and j.requirement]
    gap_reqs = [j.requirement for j in judgments if j.match_fraction == 0.0 and j.requirement]

    # Nothing stated to check either way (e.g. JD names no must-haves at
    # all) — no bar to fail, default to full coverage credit. Otherwise a
    # straight ratio: every requirement's credit counts toward coverage,
    # whatever the total — no "up to 6 most material" cherry-picking, no
    # tech_stack flattening that would split an "or" requirement into
    # independent, individually-penalized options, and no single missing-
    # but-learnable component (e.g. Kubernetes vs. Docker) zeroing an
    # otherwise well-covered "and"-bundled requirement.
    base_coverage = 25.0 if total_must_haves == 0 else 25 * matched_credit / total_must_haves

    nice_names = [n.name for n in item.skills.nice_to_have_matches if n.name]
    # Match-only bonus — missing nice-to-haves are never subtracted, only
    # matched ones add a small, capped bonus on top of the must-have ratio.
    nice_bonus = min(_NICE_TO_HAVE_BONUS_CAP, _NICE_TO_HAVE_BONUS_PER_ITEM * len(nice_names))

    floor = 10 if core_fit == "core_match" else 0
    coverage_bonus = max(floor, min(25, round(base_coverage + nice_bonus)))
    pre_gate_score = core_base + coverage_bonus

    hard_gate_hit = min_yoe is not None and years_exp is not None and years_exp < min_yoe
    skills_score = min(pre_gate_score, 15) if hard_gate_hit else pre_gate_score

    partial_desc = ", ".join(f"{req} ({frac:g})" for req, frac in partial_reqs)
    skills_reason = (
        f"CORE_BASE={core_base} ({core_fit}: {item.skills.core_fit_reason}) + "
        f"COVERAGE_BONUS={coverage_bonus} ({matched_credit:g}/{total_must_haves} must-have credit"
        + (f"; gaps: {', '.join(gap_reqs)}" if gap_reqs else "")
        + (f"; partial: {partial_desc}" if partial_reqs else "")
        + (f"; +{nice_bonus} bonus for nice-to-haves: {', '.join(nice_names)}" if nice_names else "")
        + f") = {pre_gate_score}"
    )
    if hard_gate_hit:
        skills_reason += (f". HARD GATE: {years_exp} yrs experience vs {min_yoe}+ required — "
                           f"capped at {skills_score}/60 (would have been {pre_gate_score})")

    company_score = _COMPANY_SCORE_BY_TIER[item.company.tier]
    company_reason = f"{_COMPANY_TIER_LABEL[item.company.tier]} = {company_score}/10"
    if item.company.reason:
        company_reason += f" — {item.company.reason}"

    if job.get("is_remote"):
        remote_score, remote_reason = 10, "is_remote flag is true (ground truth, overrides JD wording) = 10/10"
    else:
        remote_score = _REMOTE_SCORE_BY_TIER[item.remote.tier]
        remote_reason = f"{_REMOTE_TIER_LABEL[item.remote.tier]} = {remote_score}/10"
        if item.remote.reason:
            remote_reason += f" — {item.remote.reason}"

    role_score = _ROLE_SCORE_BY_TIER[item.role.tier]
    role_reason = f"{_ROLE_TIER_LABEL[item.role.tier]} = {role_score}/20"
    if item.role.reason:
        role_reason += f" — {item.role.reason}"

    return {
        "skills": {"score": skills_score, "max": 60, "reason": skills_reason,
                   "core_fit": core_fit, "matched_requirements": full_reqs,
                   "partial_requirements": partial_reqs, "gap_requirements": gap_reqs,
                   "matched_credit": matched_credit, "nice_to_have_matches": nice_names,
                   "core_base": core_base, "coverage_bonus": coverage_bonus,
                   "hard_gate_triggered": hard_gate_hit},
        "company": {"score": company_score, "max": 10, "reason": company_reason, "tier": item.company.tier},
        "remote":  {"score": remote_score,  "max": 10, "reason": remote_reason,  "tier": item.remote.tier},
        "role":    {"score": role_score,    "max": 20, "reason": role_reason,    "tier": item.role.tier},
    }


def _format_structured_job_block(job: dict) -> str:
    payload = dict(job.get("requirements") or {})
    payload["id"] = job["id"]
    payload["is_remote"] = bool(job.get("is_remote"))
    payload["title"] = job.get("title", "")
    payload["company"] = job.get("company", "")
    return json.dumps(payload)


def _build_structured_batches(jobs: list[dict]) -> list[list[dict]]:
    """Structured JD JSON blocks are small (no raw description text), so
    batch purely by job count (_MAX_JOBS_PER_BATCH) rather than the
    char-budget logic _build_batches needs for raw description text."""
    return [jobs[i:i + _MAX_JOBS_PER_BATCH] for i in range(0, len(jobs), _MAX_JOBS_PER_BATCH)]


@observe(name="score_structured_batch")
def _score_structured_batch(resume_json: str, resume: dict, batch: list[dict], log_fn=None, label: str = "",
                             cancel_event: threading.Event | None = None) -> list[dict]:
    """Structured-JSON counterpart to _score_batch() — same heartbeat/
    timeout/cancel/breaker machinery, scoring structured JD JSON against
    structured resume JSON instead of raw text. The LLM only returns atomic
    tier/gap judgments (StructuredJobScoreItem); every point value is then
    computed deterministically by _compute_structured_breakdown()."""
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(f"{label}{msg}")

    if cancel_event is not None and cancel_event.is_set():
        _log("Skipped — cancelled.")
        return []

    jobs_block = "\n".join(_format_structured_job_block(job) for job in batch)
    system_prompt = _STRUCTURED_BATCH_SCORE_SYSTEM_PROMPT.format(
        rubric=_STRUCTURED_SCORE_RUBRIC, resume_json=resume_json,
    )
    user_prompt = _STRUCTURED_BATCH_SCORE_USER_PROMPT.format(n=len(batch), jobs_block=jobs_block)
    timeout_s = min(_MAX_TIMEOUT_S, _BASE_TIMEOUT_S + _PER_JOB_TIMEOUT_S * len(batch))

    _log(f"Starting: {len(batch)} job(s), {len(system_prompt) + len(user_prompt):,} chars, timeout={timeout_s}s…")

    stop_heartbeat = threading.Event()
    t0 = time.time()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(_HEARTBEAT_S):
            _log(f"Still running… ({int(time.time() - t0)}s elapsed, timeout at {timeout_s}s)")

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    def _query(client, model):
        max_tokens = min(1000 * len(batch) + 500, 100_000)
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    # Prompt-cache the system message (rubric + resume_json —
                    # identical across every batch call in a scoring run).
                    cache_control_injection_points=[{"location": "message", "role": "system"}],
                    response_model=StructuredBatchScoreResult,
                    max_tokens=max_tokens,
                    max_retries=3,
                    timeout=timeout_s,
                )
            except IncompleteOutputException:
                if attempt == 0:
                    max_tokens = min(max_tokens * 2, 100_000)
                    _log(f"Output truncated at max_tokens — retrying with max_tokens={max_tokens}…")
                else:
                    raise

    try:
        result = execute_with_breaker(_query, is_instructor=True, log_fn=_log, model_class="structured_score")
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1)
    _log(f"Done in {time.time() - t0:.1f}s — {len(result.scores)} score(s) returned.")

    by_job = {job["id"]: job for job in batch}
    out = []
    for item in result.scores:
        job = by_job.get(item.id)
        if job is None:
            continue
        bd = _compute_structured_breakdown(item, job, resume)
        total = bd["skills"]["score"] + bd["company"]["score"] + bd["remote"]["score"] + bd["role"]["score"]
        out.append({
            "id": item.id,
            "score": total,
            "reason": item.overall_reason,
            "breakdown": json.dumps(bd),
        })
    return out


def score_jobs_structured(resume_profile: ResumeProfile, jobs: list[dict], log_fn=None,
                           cancel_event: threading.Event | None = None) -> Iterator[list[dict]]:
    """Structured-JSON counterpart to score_jobs(). Each job dict must have:
    id, requirements (a JobRequirements-shaped dict already parsed from the
    jobs.jd_extracted column), is_remote (the DB's own boolean flag, used as
    ground truth for the remote sub-score instead of the LLM-extracted
    remote_policy text), title, company (the job's own raw title/company
    name, not JD text — lets the model use its own knowledge of the named
    company and read seniority directly off the title when the extracted
    company_type/seniority_band are null/ambiguous — see
    _STRUCTURED_SCORE_RUBRIC). Same batching/concurrency/cancel/heartbeat/
    breaker machinery as score_jobs(), but scores against
    _STRUCTURED_SCORE_RUBRIC and JSON blocks instead of raw text, and uses
    the "structured_score" model class (see _MODEL_CLASS_ENV)
    rather than the default scoring-class model.
    """
    batches = _build_structured_batches(jobs)
    if not batches:
        return

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    breaker = scoring_breaker_status()
    if breaker["open"]:
        _log(f"Structured scoring unavailable — model in cooldown for ~{breaker['retry_in_s']}s "
             f"({breaker['reason']}).")
        return

    resume_dict = resume_profile.model_dump()
    resume_json = json.dumps(resume_dict)
    _log(f"{len(jobs)} job(s) split into {len(batches)} batch(es) "
         f"(up to {min(BATCH_SIZE, len(batches))} running concurrently)…")

    _warm_up_litellm(_provider())

    pool = ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(batches)))
    try:
        futures = {
            pool.submit(_score_structured_batch, resume_json, resume_dict, batch, log_fn,
                        f"[Batch {i}/{len(batches)}] ", cancel_event): batch
            for i, batch in enumerate(batches, start=1)
        }
        pending = set(futures)
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                _log("Cancelled — stopping (any already-running batches finish in the "
                     "background, but their results are discarded).")
                break
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                batch = futures[future]
                try:
                    yield future.result()
                except Exception as e:
                    _log(f"Structured batch scoring failed ({len(batch)} job(s)): {e}")
                    warnings.warn(f"Structured batch scoring failed ({len(batch)} job(s)): {e}")
                    yield []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


@observe(name="draft_referral_message")
def draft_referral_message(candidate_summary: str, contact: dict, job: dict) -> str:
    job_url = job.get("job_url_direct") or job.get("job_url") or ""
    prompt = _REFERRAL_PROMPT.format(
        summary=candidate_summary[:1000],
        contact_name=contact.get("name", ""),
        contact_title=contact.get("title", ""),
        company=job.get("company", ""),
        job_title=job.get("title", ""),
        job_url=job_url or "not provided",
    )

    def _query(client, model):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    return execute_with_breaker(_query)


def _format_field_line(c: dict) -> str:
    blob = (c.get("blob") or "").replace('"', "'")
    return f'- tag_id={c.get("tag_id")} tag={c.get("tag")} type={c.get("type")} blob="{blob}"'


def match_form_fields(candidates: list[dict]) -> FormFieldMap:
    """Ask the LLM to map scanned job-application form fields to candidate-
    data slots — fallback for scanner.apply's keyword heuristic when it
    finds nothing on an unfamiliar ATS platform. Raises on ANY failure
    (breaker open, missing API key, rate limit, timeout, malformed output)
    — the caller (scanner.apply._llm_match_slots) is responsible for
    catching and degrading gracefully.
    """
    fields_block = "\n".join(_format_field_line(c) for c in candidates)
    prompt = _FORM_FIELD_MATCH_PROMPT.format(fields_block=fields_block)

    def _query(client, model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=FormFieldMap,
            max_tokens=300,
            max_retries=1,
            timeout=15,
        )

    return execute_with_breaker(_query, is_instructor=True)
