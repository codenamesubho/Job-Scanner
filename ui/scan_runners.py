"""Scan "runners" — one per source, Streamlit-agnostic (log_fn/progress_fn only)
so the exact same logic can run from an individual sidebar button (main
thread, updates st.* placeholders) or from a background thread as part of
"Scan All" (updates a locked shared dict instead). Each returns a ScanResult.

None of the functions in this file may import streamlit or otherwise touch
st.* — that is what lets ui/scan_handlers.py invoke them from a background
thread. Main-thread-only orchestration (progress bars, st.* placeholders)
belongs in ui/scan_handlers.py instead, even though its function names also
start with `_run_` — the prefix alone does not indicate thread-safety here.

The per-keyword search/save loop itself lives in `scanner.search`, shared with
cron_scan.py. What remains here is only the per-source setup each one needs:
credential checks, session/login handling, and API-key gating.
"""
import os

from scanner import (
    ScanResult, SearchCriteria, jsearch_search_jobs, linkedin_login,
    linkedin_playwright_search, naukri_login, naukri_search,
    run_company_board_scan, run_keyword_scan, search_jobs,
)


def _streamlit_log_progress(log_box, progress):
    """Build (log_fn, progress_fn) callables that update st.* placeholders
    directly — only ever call this (and the callables it returns) from the
    main Streamlit script thread, never from a background worker thread."""
    from .constants import LOG_BOX_HEIGHT_PX

    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        log_lines.append(msg)
        if log_box is not None:
            log_box.code("\n".join(log_lines), height=LOG_BOX_HEIGHT_PX)

    def progress_fn(frac: float, text: str) -> None:
        if progress is not None:
            progress.progress(frac, text=text)

    return log_fn, progress_fn


def _run_jobspy(criteria: SearchCriteria, log_fn, progress_fn) -> ScanResult:
    return run_keyword_scan(search_jobs, criteria, log_fn, progress_fn)


def _run_linkedin_login(criteria: SearchCriteria, log_fn, progress_fn) -> ScanResult:
    li_email = os.getenv("LINKEDIN_EMAIL", "")
    li_pass  = os.getenv("LINKEDIN_PASSWORD", "")
    if not li_email or not li_pass:
        log_fn("Skipped — set LINKEDIN_EMAIL and LINKEDIN_PASSWORD in .env.")
        return ScanResult()

    from scanner.linkedin_playwright import SESSION_FILE as _LI_SESSION
    from scanner.linkedin_playwright import set_log_fn
    set_log_fn(log_fn)
    if not _LI_SESSION.exists():
        log_fn("Logging in to LinkedIn…")
        if not linkedin_login(li_email, li_pass):
            log_fn("LinkedIn login failed.")
            return ScanResult()

    def _on_page(pages_done, total_pages, jobs_so_far):
        # Log only — mixing this into progress_fn's fraction would conflict
        # with run_keyword_scan's own per-keyword progress updates above it.
        log_fn(f"Page {pages_done}/{total_pages}: {jobs_so_far} jobs so far")

    return run_keyword_scan(linkedin_playwright_search, criteria, log_fn, progress_fn,
                             on_page_done=_on_page)


def _run_naukri(criteria: SearchCriteria, log_fn, progress_fn) -> ScanResult:
    naukri_email = os.getenv("NAUKRI_EMAIL", "")
    naukri_pass  = os.getenv("NAUKRI_PASSWORD", "")
    if not naukri_email or not naukri_pass:
        log_fn("Skipped — set NAUKRI_EMAIL and NAUKRI_PASSWORD in .env.")
        return ScanResult()

    from scanner.naukri_playwright import SESSION_FILE as _NK_SESSION
    if not _NK_SESSION.exists():
        log_fn("Logging in to Naukri…")
        if not naukri_login(naukri_email, naukri_pass):
            log_fn("Naukri login failed.")
            return ScanResult()

    return run_keyword_scan(naukri_search, criteria, log_fn, progress_fn)


def _run_jsearch(criteria: SearchCriteria, log_fn, progress_fn) -> ScanResult:
    if not os.getenv("JSEARCH_API_KEY"):
        log_fn("Skipped — set JSEARCH_API_KEY in .env.")
        return ScanResult()
    return run_keyword_scan(jsearch_search_jobs, criteria, log_fn, progress_fn)


def _run_company_boards(criteria: SearchCriteria, log_fn, progress_fn) -> ScanResult:
    return run_company_board_scan(criteria, log_fn, progress_fn)


# Registry used by both the individual sidebar buttons and "Scan All (parallel)".
_SCAN_SOURCES = [
    ("LinkedIn (jobspy)", _run_jobspy),
    ("LinkedIn (login)",  _run_linkedin_login),
    ("Naukri",            _run_naukri),
    ("Company Boards",    _run_company_boards),
    ("JSearch",           _run_jsearch),
]
