import json
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Literal

from instructor.core import IncompleteOutputException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import (
    BATCH_SIZE, EmptyScoringResultError, _provider, _raw_completion_text,
    observe, scoring_breaker_status,
)
from .extraction import ResumeProfile
from .raw_scoring import _BASE_TIMEOUT_S, _HEARTBEAT_S, _MAX_JOBS_PER_BATCH, _MAX_TIMEOUT_S, _PER_JOB_TIMEOUT_S

# Parallel implementation to score_jobs()/_score_batch() in raw_scoring.py,
# scoring structured JD/resume JSON instead of raw description text. Kept as
# a fully separate code path (not a shared parametrized core) so a bug here
# can never affect the raw-text scoring path, and vice versa.

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
# batch) parts, same rationale as _BATCH_SCORE_SYSTEM_PROMPT/_USER_PROMPT in
# raw_scoring.py: lets the system part be prompt-cached via litellm's
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
    # No default — see BatchScoreResult's comment: CLIProxyAPI intermittently
    # wraps the whole payload one level deeper (content='{"input": {"scores":
    # [...]}}'), and a default_factory let that silently validate as an empty
    # result instead of failing and triggering a reask.
    scores: list[StructuredJobScoreItem]


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
    # Call-time lookup — see raw_scoring._score_batch's matching comment:
    # tests monkeypatch `execute_with_breaker` on the scanner.llm package
    # object and expect this function to observe that patch.
    from scanner import llm as _llm

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

    # Captures the raw litellm response via instructor's "completion:response"
    # hook (see _raw_completion_text) so that if the parsed result ends up
    # empty, we can log what the model actually said instead of guessing.
    last_raw_response = {}

    def _query(client, model):
        client.on("completion:response", lambda response: last_raw_response.__setitem__("value", response))
        # Static, generous ceiling rather than scaling off job count alone:
        # structured mode writes one full judgment object per must-have, and
        # must_haves count varies wildly per JD (seen in production: 3 on one
        # job, 14 on another in the same 5-job batch) — a job-count-only
        # formula (the old `1000 * len(batch) + 500`) can undershoot badly
        # for a must-have-heavy batch. 16,000 covers a pessimistic worst case
        # for _MAX_JOBS_PER_BATCH=5 (5 jobs x ~20 must-haves x ~60 tokens/
        # judgment + per-job overhead ≈ 9,000) with real headroom, while
        # staying well under the 100,000 absolute ceiling and under typical
        # model output caps. The IncompleteOutputException doubling below
        # still applies on top of this as a second safety net.
        max_tokens = 16_000
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    # TEMP DIAGNOSTIC: system+user merged into a single user
                    # message (no separate system message) while investigating
                    # whether CLIProxyAPI is dropping the system message on
                    # some calls — production rows showed skills judgments
                    # ("no candidate data") ignoring resume_profile (system-
                    # only) while company/remote/role judgments correctly used
                    # job_requirements (user-only), consistent with the system
                    # message specifically going missing on those calls.
                    # Prompt caching disabled too since cache_control currently
                    # only targets the system message. Revert both once this
                    # diagnostic is resolved.
                    messages=[
                        {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
                    ],
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
        result = _llm.execute_with_breaker(_query, is_instructor=True, log_fn=_log, model_class="structured_score")
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
    if not out and batch:
        raise EmptyScoringResultError(
            f"LLM returned no usable scores for {len(batch)} job(s) "
            f"(ids: {[job['id'] for job in batch]}) — response had zero entries, "
            f"or every entry had a non-matching id. "
            f"Raw response: {_raw_completion_text(last_raw_response.get('value'))}"
        )
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
    # Call-time lookup — see _score_structured_batch's comment: tests
    # monkeypatch `_score_structured_batch`/`_warm_up_litellm`/
    # `scoring_breaker_status` on the scanner.llm package object.
    from scanner import llm as _llm

    batches = _build_structured_batches(jobs)
    if not batches:
        return

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    breaker = _llm.scoring_breaker_status()
    if breaker["open"]:
        _log(f"Structured scoring unavailable — model in cooldown for ~{breaker['retry_in_s']}s "
             f"({breaker['reason']}).")
        return

    resume_dict = resume_profile.model_dump()
    resume_json = json.dumps(resume_dict)
    _log(f"{len(jobs)} job(s) split into {len(batches)} batch(es) "
         f"(up to {min(BATCH_SIZE, len(batches))} running concurrently)…")

    _llm._warm_up_litellm(_provider())

    pool = ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(batches)))
    try:
        futures = {
            pool.submit(_llm._score_structured_batch, resume_json, resume_dict, batch, log_fn,
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
                except EmptyScoringResultError as e:
                    # Fatal for the whole run — see EmptyScoringResultError's
                    # docstring and the matching handling in score_jobs().
                    _log(f"Structured batch scoring returned empty — stopping the run: {e}")
                    raise
                except Exception as e:
                    _log(f"Structured batch scoring failed ({len(batch)} job(s)): {e}")
                    warnings.warn(f"Structured batch scoring failed ({len(batch)} job(s)): {e}")
                    yield []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
