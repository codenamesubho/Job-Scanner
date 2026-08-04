"""Auto-scoring newly-found jobs, run right after a scan completes."""
import time

import streamlit as st

from scanner import get_candidate, scoring_breaker_status

from ..background import BackgroundJob
from ..constants import AUTO_SCORE_LOG_BOX_HEIGHT_PX, POLL_INTERVAL_S
from .pipeline import ScoringPlan, run_scoring

_PLAN = ScoringPlan(
    only_unscored=True,
    nothing_to_do="No new jobs need scoring.",
    cancellable=False,      # a post-scan run is short and has no Cancel button
)


def _auto_score_new() -> None:
    """Score whatever the scan just added, blocking until it finishes.

    Fetching already deduped cross-source at insert time
    (scanner.database.save_jobs); this additionally skips jobs that already have
    a score from a previous scan, so only genuinely new jobs reach the LLM.

    Everything slow — structured extraction, resume-profile loading, job
    discovery, scoring — happens on the worker thread rather than before it, so
    progress reaches the log box and the UI stays responsive while a large batch
    is extracted one job at a time.
    """
    candidate = get_candidate()
    if not candidate.get("summary"):
        st.caption("⚠ No candidate summary — skipping scoring. Add one in Profile.")
        return

    breaker = scoring_breaker_status()
    if breaker["open"]:
        st.error(
            f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
            f"({breaker['reason']})."
        )
        return

    job = BackgroundJob(name="auto-score").start(
        lambda j: run_scoring(j, candidate, _PLAN)
    )

    progress = st.progress(0, text="Checking for jobs to score…")
    log_box  = st.empty()

    while job.is_alive():
        snap = job.snapshot()
        if snap.total:
            progress.progress(snap.fraction, text=f"Scoring {snap.done} / {snap.total} job(s)…")
        log_box.code(snap.log_text(), height=AUTO_SCORE_LOG_BOX_HEIGHT_PX)
        time.sleep(POLL_INTERVAL_S)

    snap = job.snapshot()
    progress.empty()
    log_box.empty()
    if snap.error:
        st.warning(f"Scoring failed: {snap.error}")
    elif snap.skip:
        st.caption(snap.skip)
    else:
        st.info(f"Scored {snap.done} new job(s).")
