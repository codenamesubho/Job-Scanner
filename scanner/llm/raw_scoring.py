import json
import threading
import time
import warnings
from collections.abc import Iterator

from instructor.core import IncompleteOutputException

from . import (
    BATCH_SIZE, BatchScoreResult, EmptyScoringResultError, _provider,
    _raw_completion_text, observe, scoring_breaker_status,
)
from ._batch import breaker_is_open, run_batches
from ._prompt_loader import load_prompt_file

# Prompt text lives in scanner/llm/prompts/raw_scoring.json (versioned per
# entry), loaded once here into the same constant names this module always
# used — everything below (.format() calls, _score_batch, _build_batches)
# is unaffected by prompt text living in JSON instead of inline strings.
#
# One LLM call scores every job packed into it (see _build_batches) rather
# than one call per job — the response is a JSON array, one entry per job,
# matched back by the "id" copied verbatim from each job block below. That
# id is a short, ephemeral hash-derived label (_batch_short_id), not the
# job's real database id — see _score_batch's id_map for why.
#
# batch_score_system/batch_score_user are split into a system part (rubric +
# candidate profile — identical across every batch call in a scoring run)
# and a user part (job data + response format — different per batch) so the
# system part can be prompt-cached via litellm's cache_control_injection_points
# (see _score_batch) instead of resending ~1,800 tokens of static rubric text
# on every single batch call.
_PROMPTS = load_prompt_file("raw_scoring")
_SCORE_RUBRIC = _PROMPTS["score_rubric"]["template"]
_BATCH_SCORE_SYSTEM_PROMPT = _PROMPTS["batch_score_system"]["template"]
_BATCH_SCORE_USER_PROMPT = _PROMPTS["batch_score_user"]["template"]
_JOB_BLOCK = _PROMPTS["job_block"]["template"]

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

    if breaker_is_open(scoring_breaker_status(), log_fn, "Scoring"):
        return

    _log(f"{len(jobs)} job(s) split into {len(batches)} batch(es) "
         f"(up to {min(BATCH_SIZE, len(batches))} running concurrently)…")

    _llm._warm_up_litellm(_provider())

    def _submit(pool, batch, i, total):
        # _llm._score_batch, not the module-level name: resolved at call time so
        # tests patching it on the package object are observed. See _score_batch's
        # own comment for the full rationale.
        return pool.submit(_llm._score_batch, summary, batch, log_fn,
                            f"[Batch {i}/{total}] ", cancel_event)

    yield from run_batches(batches, _submit, log_fn, cancel_event, label="Batch scoring")
