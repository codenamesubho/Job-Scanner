"""The per-job "Apply" button — runs Autofill-Job-Application's agent in the
background and doubles as Cancel while running, same pattern as the sidebar
Score button (ui/scoring/score_button.py), but keyed per job_id since more
than one job's detail panel can be applied to across a session.
"""
import time

import streamlit as st

from scanner.autofill_bridge import run_apply

from .background import BackgroundJob
from .constants import APPLY_BUTTON_POLL_S, LOG_BOX_HEIGHT_PX
from .session_keys import APPLY_JOBS


def _jobs() -> dict:
    return st.session_state.setdefault(APPLY_JOBS, {})


def _start_job(job_id: str, apply_url: str) -> None:
    _jobs()[job_id] = BackgroundJob(name=f"apply-{job_id}").start(
        lambda j: j.set(result=run_apply(apply_url, log_fn=j.log, cancel_event=j.cancel_event))
    )


def _render_running(job_id: str, job: BackgroundJob) -> None:
    if st.button("🛑 Cancel Apply", key=f"apply_cancel_{job_id}", use_container_width=True):
        job.cancel()

    snap = job.snapshot()
    st.progress(0 if not snap.finished else 1, text="Applying…" if not snap.finished else "Done.")
    st.code(snap.log_text(), height=LOG_BOX_HEIGHT_PX)

    if not snap.finished:
        # Tick via a short sleep + rerun rather than one long blocking loop —
        # same reasoning as _render_score_button: a blocking loop would never
        # return to Streamlit's event loop, so Cancel could never be delivered
        # until the whole (multi-minute) run finished on its own.
        time.sleep(APPLY_BUTTON_POLL_S)
        st.rerun()
        return

    del _jobs()[job_id]
    if snap.error:
        st.error(f"Apply failed: {snap.error}")
        return

    result = snap.result or {}
    filled    = result.get("filled") or []
    escalated = result.get("escalated") or []
    if result.get("success"):
        msg = f"Filled {len(filled)} field(s)"
        if escalated:
            msg += f", {len(escalated)} flagged for your review"
        msg += ". Review & submit in the browser window."
        st.success(msg)
    else:
        st.warning(result.get("error") or "Couldn't fill the application — see the log above.")


def _render_apply_button(job_id: str, apply_url: str) -> None:
    job = _jobs().get(job_id)

    if job is None:
        if st.button("🚀 Apply", key=f"apply_{job_id}", use_container_width=True, type="primary"):
            _start_job(job_id, apply_url)
            st.rerun()
        return

    _render_running(job_id, job)
