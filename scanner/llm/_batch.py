"""The batch-scoring driver shared by raw and structured scoring.

`score_jobs()` and `score_jobs_structured()` ran near-identical outer loops:
check the circuit breaker, warm up litellm, submit every batch to a thread pool,
then yield each batch's results as it completes while staying responsive to
cancellation. They differed only in wording and in which per-batch function they
submitted, so a fix to one (the FIRST_COMPLETED cancellation handling, the
fatal-EmptyScoringResultError path) had to be mirrored by hand into the other.

The `submit` callable keeps the *call-time* lookup of the per-batch function in
the caller. That indirection is load-bearing: the tests monkeypatch
`_score_batch` / `_score_structured_batch` on the `scanner.llm` package object,
which a static import here would not observe.
"""
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import BATCH_SIZE, EmptyScoringResultError


def run_batches(batches, submit, log_fn=None, cancel_event=None, label="Scoring"):
    """Run `batches` concurrently, yielding each batch's result list as it lands.

    `submit(pool, batch, index, total)` must return a Future for that batch.
    `label` prefixes the user-facing messages ("Scoring" / "Structured scoring").

    A batch that fails yields [] and warns, so one bad batch doesn't lose the
    others — except EmptyScoringResultError, which is fatal for the whole run
    (see its docstring: it signals something structurally wrong, and continuing
    would silently under-score everything else).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    total = len(batches)

    # Not a `with` block on purpose: exiting `with ThreadPoolExecutor()` calls
    # shutdown(wait=True), which would block a cancel/exception unwind until
    # every in-flight batch finished.
    pool = ThreadPoolExecutor(max_workers=min(BATCH_SIZE, total))
    try:
        futures = {submit(pool, batch, i, total): batch
                   for i, batch in enumerate(batches, start=1)}
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
                    _log(f"{label} returned empty — stopping the run: {e}")
                    raise
                except Exception as e:
                    message = f"{label} failed ({len(batch)} job(s)): {e}"
                    _log(message)
                    warnings.warn(message)
                    yield []
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def breaker_is_open(breaker: dict, log_fn, label: str) -> bool:
    """Log and report whether the circuit breaker blocks this run."""
    if not breaker["open"]:
        return False
    if log_fn:
        log_fn(f"{label} unavailable — model in cooldown for ~{breaker['retry_in_s']}s "
               f"({breaker['reason']}).")
    return True
