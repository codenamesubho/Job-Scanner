"""The sidebar "Score Jobs" button, which doubles as Cancel while running."""
import time

import streamlit as st

from scanner import get_candidate, scoring_breaker_status

from ..background import BackgroundJob
from ..constants import LOG_BOX_HEIGHT_PX, SCORE_BUTTON_POLL_S
from ..session_keys import SCORE_JOB
from .pipeline import ScoringPlan, run_scoring

_PLAN = ScoringPlan(
    only_unscored=False,    # the explicit button re-scores everything scoreable
    nothing_to_do="No jobs with descriptions to score.",
    cancellable=True,
)


def _start_job() -> bool:
    """Validate preconditions and start a scoring run. Returns True if started."""
    candidate = get_candidate()
    if not candidate.get("summary"):
        st.warning("No candidate summary found. Generate one in the Profile tab first.")
        return False

    breaker = scoring_breaker_status()
    if breaker["open"]:
        st.error(
            f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
            f"({breaker['reason']})."
        )
        return False

    st.session_state[SCORE_JOB] = BackgroundJob(name="score-jobs").start(
        lambda j: run_scoring(j, candidate, _PLAN)
    )
    return True


def _render_running(job: BackgroundJob) -> None:
    """Draw progress for an in-flight run, and finish up once it's done."""
    # This click cancels the run rather than starting a second one.
    if st.button("🛑 Cancel Scoring", use_container_width=True):
        job.cancel()

    snap      = job.snapshot()
    cancelled = job.cancelled

    if snap.total:
        st.progress(snap.fraction, text=f"Scoring {snap.done} / {snap.total} job(s)…")
    else:
        st.progress(0, text="Done." if snap.finished else "Extracting structured JD data…")
    st.code(snap.log_text(), height=LOG_BOX_HEIGHT_PX)

    if not snap.finished:
        # Tick via a short sleep + rerun rather than one long blocking loop: a
        # blocking loop would not return to Streamlit's event loop, so the
        # Cancel click above could never be delivered until the whole run
        # finished — which is exactly what makes this button cancellable.
        time.sleep(SCORE_BUTTON_POLL_S)
        st.rerun()
        return

    del st.session_state[SCORE_JOB]
    if snap.error:
        st.error(f"Scoring failed: {snap.error}")
    elif snap.skip:
        st.info(snap.skip)
    elif cancelled:
        st.warning(f"Scoring cancelled — {snap.done} job(s) scored before stopping.")
    else:
        st.success(f"Scored {snap.done} job(s).")


def _render_score_button() -> None:
    """Renders in the sidebar, just below "Scan All Sources"."""
    job = st.session_state.get(SCORE_JOB)

    if job is None:
        if st.button("🎯 Score Jobs", use_container_width=True) and _start_job():
            st.rerun()
        return

    _render_running(job)
