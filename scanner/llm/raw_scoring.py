import json
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from instructor.core import IncompleteOutputException

from . import (
    BATCH_SIZE, BatchScoreResult, EmptyScoringResultError, _provider,
    _raw_completion_text, observe, scoring_breaker_status,
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
# matched back by the "id" copied verbatim from each job block below. That
# id is a short, ephemeral hash-derived label (_batch_short_id), not the
# job's real database id — see _score_batch's id_map for why.
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

_MAX_BATCH_CHARS = 3_999_000  # ~1M-token context budget at ~4 chars/token, minus a small safety margin
_MAX_JOBS_PER_BATCH = 3      # also cap by job count, not just chars — keeps individual calls'
                              # output size (and therefore wall-clock time) bounded, and ensures
                              # a large scoring run actually splits into multiple batches that can
                              # run concurrently instead of one huge serial call
_BASE_TIMEOUT_S     = 60      # per-call timeout floor
_PER_JOB_TIMEOUT_S  = 20      # extra timeout budget per job in the batch
_MAX_TIMEOUT_S      = 600     # hard ceiling regardless of batch size
_HEARTBEAT_S        = 15      # how often to log "still running" while a call is in flight


def _format_job_block(job: dict, short_id: str) -> str:
    return _JOB_BLOCK.format(
        id=short_id,
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
        # Real short id isn't assigned until _score_batch builds the final
        # batch (it depends on position within that batch) — a placeholder
        # of the same length is all this needs for char-budget estimation.
        block = _format_job_block(job, "0" * 11)
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
    # Call-time lookup (not a top-of-file `from . import execute_with_breaker`)
    # so tests that do monkeypatch.setattr(llm, "execute_with_breaker", fake)
    # on the scanner.llm package object actually take effect here — a static
    # import would bind its own independent name in this module's namespace,
    # invisible to a patch applied to the package attribute instead.
    from scanner import llm as _llm

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(f"{label}{msg}")

    if cancel_event is not None and cancel_event.is_set():
        _log("Skipped — cancelled.")
        return []

    # Short, hash-derived per-job label instead of the job's real id — a
    # real id can be a 400+ char opaque token (e.g. JSearch-sourced jobs),
    # and asking the model to echo one back verbatim is prone to single-
    # character transcription errors that fail exact-match validation (see
    # EmptyScoringResultError below). id_map resolves the LLM's short id
    # back to the real job dict once results come back.
    id_map = {_llm._batch_short_id(job.get("description"), i): job
              for i, job in enumerate(batch, start=1)}
    jobs_block = "\n".join(_format_job_block(job, short_id) for short_id, job in id_map.items())
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

    # See _score_structured_batch's matching comment — captures the raw
    # litellm response so an empty-result exception can show what the model
    # actually said instead of just "zero entries."
    last_raw_response = {}

    def _query(client, model):
        client.on("completion:response", lambda response: last_raw_response.__setitem__("value", response))
        max_tokens = min(1000 * len(batch) + 500, 100_000)
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    # CLIProxyAPI (the Anthropic backend behind api_base, see
                    # _PROVIDER_CONFIG) silently drops/overrides the `system`
                    # field — confirmed by sending an unmistakable system-only
                    # instruction both through litellm and via a raw HTTP POST
                    # straight to CLIProxyAPI's /v1/messages, bypassing litellm/
                    # instructor entirely: the model's response showed no sign
                    # of the instruction either way, while its replies
                    # consistently self-identified as "Claude Code" regardless
                    # of what `system` content was sent — consistent with
                    # CLIProxyAPI bridging an actual Claude Code/subscription
                    # session that applies its own fixed system prompt instead.
                    # `messages` content came through faithfully in every test,
                    # so the rubric+candidate profile now rides as a second
                    # leading `user` turn instead of `system` — Anthropic's API
                    # merges consecutive same-role messages into one turn, and
                    # cache_control on its own content block caches it exactly
                    # like a cached system message would.
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                            ],
                        },
                        {"role": "user", "content": user_prompt},
                    ],
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
        result = _llm.execute_with_breaker(_query, is_instructor=True, log_fn=_log)
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1)
    _log(f"Done in {time.time() - t0:.1f}s — {len(result.scores)} score(s) returned.")

    out = []
    for item in result.scores:
        job = id_map.get(item.id)
        if job is None or item.score is None:
            continue
        bd = item.breakdown
        # Enforce score = sum of breakdown to catch LLM arithmetic errors
        computed = bd.skills.score + bd.company.score + bd.remote.score + bd.role.score
        out.append({
            "id": job["id"],  # real job id — item.id is the ephemeral short id, not stored
            "score": computed,
            "reason": item.reason,
            "breakdown": json.dumps(bd.model_dump()),
        })
    if not out and batch:
        raise EmptyScoringResultError(
            f"LLM returned no usable scores for {len(batch)} job(s) "
            f"(ids: {[job['id'] for job in batch]}) — response had zero entries, "
            f"or every entry had a non-matching id / null score. "
            f"Raw response: {_raw_completion_text(last_raw_response.get('value'))}"
        )
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
    # Call-time lookup — see _score_batch's comment on why this can't be a
    # static top-of-file import: tests monkeypatch `_score_batch` and
    # `_warm_up_litellm` on the scanner.llm package object and expect this
    # function to observe those patches.
    from scanner import llm as _llm

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

    _llm._warm_up_litellm(_provider())

    # Not a `with` block on purpose: exiting `with ThreadPoolExecutor()` calls
    # shutdown(wait=True), which blocks until every in-flight batch finishes —
    # defeating cancellation entirely (this function wouldn't return until
    # all batches were done regardless). Managing the pool manually lets the
    # cancelled path shut down without waiting for already-running batches.
    pool = ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(batches)))
    try:
        futures = {
            pool.submit(_llm._score_batch, summary, batch, log_fn,
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
                except EmptyScoringResultError as e:
                    # Fatal for the whole run, not just this batch — see
                    # EmptyScoringResultError's docstring. Re-raising here
                    # propagates out of this generator (and the finally
                    # below still cancels/discards any other pending
                    # batches), stopping score_unscored_jobs() entirely
                    # instead of silently under-scoring the rest of the run.
                    _log(f"Batch scoring returned empty — stopping the run: {e}")
                    raise
                except Exception as e:
                    _log(f"Batch scoring failed ({len(batch)} job(s)): {e}")
                    warnings.warn(f"Batch scoring failed ({len(batch)} job(s)): {e}")
                    yield []
    finally:
        # cancel_futures=True drops any not-yet-started batch immediately;
        # wait=False means we don't block on ones already running.
        pool.shutdown(wait=False, cancel_futures=True)
