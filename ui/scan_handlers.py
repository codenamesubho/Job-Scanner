"""Scan handlers — individual sources (thin wrappers around ui/scan_runners.py)
plus the "Scan All (parallel)" orchestrator.

Unlike ui/scan_runners.py, every function here runs on the main Streamlit
thread and touches st.* directly (progress bars, log placeholders) — except
the per-source `_worker` closures inside _run_parallel_sources, which run in
background daemon threads and write only into a lock-protected shared dict,
never touching st.* themselves.
"""
import threading
import time

import streamlit as st

from .constants import POLL_INTERVAL_S, SCAN_ALL_LOG_BOX_HEIGHT_PX
from .scan_runners import (
    _SCAN_SOURCES, _run_company_boards, _run_jobspy, _run_jsearch,
    _run_linkedin_login, _run_naukri, _streamlit_log_progress,
)
from .scoring import _auto_score_new


def _run_individual(name: str, runner, keywords: str, location: str, results: int, hours: int) -> None:
    progress = st.progress(0, text=f"Starting {name} scan…")
    log_box  = st.empty()
    log_fn, progress_fn = _streamlit_log_progress(log_box, progress)
    try:
        found, new_count = runner(keywords, location, results, hours, log_fn, progress_fn)
        progress.empty()
        if found == 0 and new_count == 0:
            st.info(f"{name}: nothing found (see log above if this is unexpected).")
        else:
            log_box.empty()
            st.success(f"{name}: {found} found, {new_count} new.")
        _auto_score_new()
    except Exception as e:
        progress.empty()
        st.error(f"{name} scan failed: {e}")


def handle_jobspy_scan(keywords: str, location: str, results: int, hours: int) -> None:
    _run_individual("LinkedIn (jobspy)", _run_jobspy, keywords, location, results, hours)


def handle_linkedin_scan(keywords: str, location: str, results: int, hours: int) -> None:
    _run_individual("LinkedIn (login)", _run_linkedin_login, keywords, location, results, hours)


def handle_naukri_scan(keywords: str, location: str, results: int, hours: int) -> None:
    _run_individual("Naukri", _run_naukri, keywords, location, results, hours)


def handle_company_boards_scan(keywords: str, location: str, results: int, hours: int) -> None:
    _run_individual("Company Boards", _run_company_boards, keywords, location, results, hours)


def handle_jsearch_scan(keywords: str, location: str, results: int, hours: int) -> None:
    _run_individual("JSearch", _run_jsearch, keywords, location, results, hours)


_PARALLEL_SOURCES = [(name, runner) for name, runner in _SCAN_SOURCES if name != "LinkedIn (login)"]


def _run_linkedin_login_alone(keywords: str, location: str, results: int, hours: int) -> None:
    """Run LinkedIn (login) sequentially, on its own — see handle_scan_all's
    docstring for why it can't run concurrently with the other sources."""
    st.markdown("---")
    st.caption("Running LinkedIn (login) first, alone (avoids concurrent LinkedIn traffic)…")
    li_progress = st.progress(0, text="LinkedIn (login): starting…")
    li_log_box  = st.empty()
    li_log_fn, li_progress_fn = _streamlit_log_progress(li_log_box, li_progress)
    try:
        found, new_count = _run_linkedin_login(keywords, location, results, hours,
                                                 li_log_fn, li_progress_fn)
        li_progress.empty()
        if found == 0 and new_count == 0:
            st.info("LinkedIn (login): nothing found (see log above if this is unexpected).")
        else:
            li_log_box.empty()
            st.success(f"LinkedIn (login): {found} found, {new_count} new.")
    except Exception as e:
        li_progress.empty()
        st.error(f"LinkedIn (login) failed: {e}")


def _run_parallel_sources(keywords: str, location: str, results: int, hours: int) -> None:
    """Run every source in _PARALLEL_SOURCES concurrently, one thread each,
    polling a lock-protected shared dict from the main thread to update each
    source's progress bar + log placeholder — see handle_scan_all's
    docstring for why worker threads can't touch st.* directly."""
    state_lock = threading.Lock()
    state = {
        name: {"log": [], "pct": 0.0, "text": "Queued…", "done": False,
               "result": None, "error": None}
        for name, _ in _PARALLEL_SOURCES
    }

    def _make_log_fn(name):
        def log_fn(msg: str) -> None:
            with state_lock:
                state[name]["log"].append(msg)
        return log_fn

    def _make_progress_fn(name):
        def progress_fn(frac: float, text: str) -> None:
            with state_lock:
                state[name]["pct"] = frac
                state[name]["text"] = text
        return progress_fn

    def _worker(name, runner):
        try:
            found, new_count = runner(keywords, location, results, hours,
                                       _make_log_fn(name), _make_progress_fn(name))
            with state_lock:
                state[name]["done"]   = True
                state[name]["pct"]    = 1.0
                state[name]["result"] = (found, new_count)
        except Exception as e:
            with state_lock:
                state[name]["done"]  = True
                state[name]["pct"]   = 1.0
                state[name]["error"] = str(e)

    threads = [
        threading.Thread(target=_worker, args=(name, runner), daemon=True)
        for name, runner in _PARALLEL_SOURCES
    ]
    for t in threads:
        t.start()

    st.markdown("---")
    st.caption(f"Running {len(threads)} more source(s) in parallel…")
    placeholders = {
        name: (st.progress(0, text=f"{name}: queued…"), st.empty())
        for name, _ in _PARALLEL_SOURCES
    }

    while any(t.is_alive() for t in threads):
        with state_lock:
            for name, (pbar, lbox) in placeholders.items():
                s = state[name]
                pbar.progress(s["pct"], text=f"{name}: {s['text']}")
                lbox.code("\n".join(s["log"]), height=SCAN_ALL_LOG_BOX_HEIGHT_PX)
        time.sleep(POLL_INTERVAL_S)

    with state_lock:
        for name, (pbar, lbox) in placeholders.items():
            s = state[name]
            pbar.progress(1.0, text=f"{name}: done")
            lbox.code("\n".join(s["log"]), height=SCAN_ALL_LOG_BOX_HEIGHT_PX)
            if s["error"]:
                st.error(f"{name} failed: {s['error']}")
            elif s["result"]:
                found, new_count = s["result"]
                st.success(f"{name}: {found} found, {new_count} new.")


def handle_scan_all(keywords: str, location: str, results: int, hours: int) -> None:
    """Run LinkedIn (login) first, sequentially and alone, then every other
    source in _PARALLEL_SOURCES concurrently, one thread each.

    LinkedIn (login) drives a real logged-in Playwright session — running it
    at the same time as other threads hitting linkedin.com (e.g. the jobspy
    source) multiplies concurrent traffic against LinkedIn from this one
    account/IP and increases the chance of getting rate-limited or blocked,
    so it always runs on its own before the parallel batch starts.

    Worker threads never touch st.* directly (Streamlit widgets aren't
    thread-safe) — each writes into its own slot of a lock-protected shared
    dict, and only the main thread reads that dict to update each source's
    progress bar + log placeholder in a polling loop.
    """
    _run_linkedin_login_alone(keywords, location, results, hours)
    _run_parallel_sources(keywords, location, results, hours)
    _auto_score_new()
