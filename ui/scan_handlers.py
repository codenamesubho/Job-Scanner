"""Scan handlers — individual sources (thin wrappers around ui/scan_runners.py)
plus the "Scan All (parallel)" orchestrator.

Unlike ui/scan_runners.py, every function here runs on the main Streamlit
thread and touches st.* directly (progress bars, log placeholders). The one
exception is the per-source worker inside _run_parallel_sources, which runs in
a background daemon thread and reports only through its BackgroundJob — see
ui/background.py for why that separation exists.
"""
import time

import streamlit as st

from scanner import SearchCriteria

from .background import BackgroundJob
from .constants import POLL_INTERVAL_S, SCAN_ALL_LOG_BOX_HEIGHT_PX
from .scan_runners import (
    _SCAN_SOURCES, _run_company_boards, _run_jobspy, _run_jsearch,
    _run_linkedin_login, _run_naukri, _streamlit_log_progress,
)
from .scoring import _auto_score_new


def _run_individual(name: str, runner, criteria: SearchCriteria) -> None:
    progress = st.progress(0, text=f"Starting {name} scan…")
    log_box  = st.empty()
    log_fn, progress_fn = _streamlit_log_progress(log_box, progress)
    try:
        result = runner(criteria, log_fn, progress_fn)
        progress.empty()
        if result.found == 0 and result.new == 0:
            st.info(f"{name}: nothing found (see log above if this is unexpected).")
        else:
            log_box.empty()
            st.success(f"{name}: {result.found} found, {result.new} new.")
        _auto_score_new()
    except Exception as e:
        progress.empty()
        st.error(f"{name} scan failed: {e}")


def handle_jobspy_scan(criteria: SearchCriteria) -> None:
    _run_individual("LinkedIn (jobspy)", _run_jobspy, criteria)


def handle_linkedin_scan(criteria: SearchCriteria) -> None:
    _run_individual("LinkedIn (login)", _run_linkedin_login, criteria)


def handle_naukri_scan(criteria: SearchCriteria) -> None:
    _run_individual("Naukri", _run_naukri, criteria)


def handle_company_boards_scan(criteria: SearchCriteria) -> None:
    _run_individual("Company Boards", _run_company_boards, criteria)


def handle_jsearch_scan(criteria: SearchCriteria) -> None:
    _run_individual("JSearch", _run_jsearch, criteria)


_PARALLEL_SOURCES = [(name, runner) for name, runner in _SCAN_SOURCES if name != "LinkedIn (login)"]


def _run_linkedin_login_alone(criteria: SearchCriteria) -> None:
    """Run LinkedIn (login) sequentially, on its own — see handle_scan_all's
    docstring for why it can't run concurrently with the other sources."""
    st.markdown("---")
    st.caption("Running LinkedIn (login) first, alone (avoids concurrent LinkedIn traffic)…")
    li_progress = st.progress(0, text="LinkedIn (login): starting…")
    li_log_box  = st.empty()
    li_log_fn, li_progress_fn = _streamlit_log_progress(li_log_box, li_progress)
    try:
        result = _run_linkedin_login(criteria, li_log_fn, li_progress_fn)
        li_progress.empty()
        if result.found == 0 and result.new == 0:
            st.info("LinkedIn (login): nothing found (see log above if this is unexpected).")
        else:
            li_log_box.empty()
            st.success(f"LinkedIn (login): {result.found} found, {result.new} new.")
    except Exception as e:
        li_progress.empty()
        st.error(f"LinkedIn (login) failed: {e}")


def _start_source_job(name: str, runner, criteria: SearchCriteria) -> BackgroundJob:
    """Kick off one source on its own thread, reporting through a BackgroundJob.

    `text` carries the progress caption and `result` the ScanResult; both are
    written through the job rather than into a shared dict, so the main thread's
    polling loop below never has to reason about locking.
    """
    def _worker(job: BackgroundJob) -> None:
        def progress_fn(frac: float, text: str) -> None:
            job.set(done=int(frac * 100), total=100, text=text)

        job.set(result=runner(criteria, job.log, progress_fn))

    return BackgroundJob(name=name).start(_worker)


def _run_parallel_sources(criteria: SearchCriteria) -> None:
    """Run every source in _PARALLEL_SOURCES concurrently, one thread each,
    polling each job's snapshot from the main thread to update its progress bar
    and log placeholder — see handle_scan_all's docstring for why worker threads
    can't touch st.* directly."""
    jobs = {name: _start_source_job(name, runner, criteria)
            for name, runner in _PARALLEL_SOURCES}

    st.markdown("---")
    st.caption(f"Running {len(jobs)} more source(s) in parallel…")
    placeholders = {
        name: (st.progress(0, text=f"{name}: queued…"), st.empty())
        for name in jobs
    }

    def _paint(final: bool = False) -> None:
        for name, (pbar, lbox) in placeholders.items():
            snap = jobs[name].snapshot()
            caption = "done" if final else (snap.text or "Queued…")
            pbar.progress(1.0 if final else snap.fraction, text=f"{name}: {caption}")
            lbox.code(snap.log_text(), height=SCAN_ALL_LOG_BOX_HEIGHT_PX)

    while any(job.is_alive() for job in jobs.values()):
        _paint()
        time.sleep(POLL_INTERVAL_S)

    _paint(final=True)
    for name, job in jobs.items():
        snap = job.snapshot()
        if snap.error:
            st.error(f"{name} failed: {snap.error}")
        elif snap.result is not None:
            st.success(f"{name}: {snap.result.found} found, {snap.result.new} new.")


def handle_scan_all(criteria: SearchCriteria) -> None:
    """Run LinkedIn (login) first, sequentially and alone, then every other
    source in _PARALLEL_SOURCES concurrently, one thread each.

    LinkedIn (login) drives a real logged-in Playwright session — running it
    at the same time as other threads hitting linkedin.com (e.g. the jobspy
    source) multiplies concurrent traffic against LinkedIn from this one
    account/IP and increases the chance of getting rate-limited or blocked,
    so it always runs on its own before the parallel batch starts.

    Worker threads never touch st.* directly (Streamlit widgets aren't
    thread-safe) — each reports into its own BackgroundJob, and only the main
    thread reads those to update progress bars and log placeholders.
    """
    _run_linkedin_login_alone(criteria)
    _run_parallel_sources(criteria)
    _auto_score_new()
