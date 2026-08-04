"""Scan "runners" — one per source, Streamlit-agnostic (log_fn/progress_fn only)
so the exact same logic can run from an individual sidebar button (main
thread, updates st.* placeholders) or from a background thread as part of
"Scan All" (updates a locked shared dict instead). Each returns (found, new).

None of the functions in this file may import streamlit or otherwise touch
st.* — that is what lets ui/scan_handlers.py invoke them from a background
thread. Main-thread-only orchestration (progress bars, st.* placeholders)
belongs in ui/scan_handlers.py instead, even though its function names also
start with `_run_` — the prefix alone does not indicate thread-safety here.
"""
import os

import pandas as pd

from scanner import (
    ATS_FETCHERS, get_company_boards, jsearch_search_jobs, linkedin_login,
    linkedin_playwright_search, naukri_login, naukri_search, save_jobs,
    search_jobs,
)
from scanner.filters import filter_by_keywords


def _split_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _do_scan_core(scan_fn, keywords: str, location: str, results: int, hours: int,
                   log_fn, progress_fn, **kwargs) -> tuple[int, int]:
    """Run scan_fn once per comma-separated keyword; return (total found, new saved).

    Reports progress via plain callables (log_fn(msg), progress_fn(frac, text))
    instead of touching Streamlit directly — safe to call from any thread,
    including a background worker thread. Callers running on the main thread
    pass in callables that update st.* placeholders; callers running in a
    background thread pass in callables that write into a locked shared dict.
    """
    kw_list = _split_keywords(keywords)

    all_dfs: list[pd.DataFrame] = []
    for i, kw in enumerate(kw_list, start=1):
        progress_fn((i - 1) / len(kw_list), f"Searching '{kw}' ({i}/{len(kw_list)})…")
        log_fn(f"[{i}/{len(kw_list)}] Searching '{kw}' in '{location}'…")
        try:
            df = scan_fn(kw, location, results_wanted=results, hours_old=hours, **kwargs)
            if not df.empty:
                all_dfs.append(df)
                log_fn(f"[{i}/{len(kw_list)}] '{kw}': {len(df)} job(s) found")
            else:
                log_fn(f"[{i}/{len(kw_list)}] '{kw}': no jobs found")
        except Exception as e:
            log_fn(f"[{i}/{len(kw_list)}] '{kw}' FAILED: {e}")
        progress_fn(i / len(kw_list), f"Searched '{kw}' ({i}/{len(kw_list)})")

    if not all_dfs:
        log_fn("No jobs found for any keyword.")
        return 0, 0

    combined = pd.concat(all_dfs, ignore_index=True)
    log_fn(f"Saving {len(combined)} job(s)…")
    new_count = save_jobs(combined)
    log_fn(f"Done — {len(combined)} found, {new_count} new, {len(combined) - new_count} already known.")
    return len(combined), new_count


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


def _run_jobspy(keywords, location, results, hours, log_fn, progress_fn):
    return _do_scan_core(search_jobs, keywords, location, results, hours, log_fn, progress_fn)


def _run_linkedin_login(keywords, location, results, hours, log_fn, progress_fn):
    li_email = os.getenv("LINKEDIN_EMAIL", "")
    li_pass  = os.getenv("LINKEDIN_PASSWORD", "")
    if not li_email or not li_pass:
        log_fn("Skipped — set LINKEDIN_EMAIL and LINKEDIN_PASSWORD in .env.")
        return 0, 0

    from scanner.linkedin_playwright import SESSION_FILE as _LI_SESSION, set_log_fn
    set_log_fn(log_fn)
    if not _LI_SESSION.exists():
        log_fn("Logging in to LinkedIn…")
        if not linkedin_login(li_email, li_pass):
            log_fn("LinkedIn login failed.")
            return 0, 0

    def _on_page(pages_done, total_pages, jobs_so_far):
        # Log only — mixing this into progress_fn's fraction would conflict
        # with _do_scan_core's own per-keyword progress updates above it.
        log_fn(f"Page {pages_done}/{total_pages}: {jobs_so_far} jobs so far")

    return _do_scan_core(linkedin_playwright_search, keywords, location, results, hours,
                          log_fn, progress_fn, on_page_done=_on_page)


def _run_naukri(keywords, location, results, hours, log_fn, progress_fn):
    naukri_email = os.getenv("NAUKRI_EMAIL", "")
    naukri_pass  = os.getenv("NAUKRI_PASSWORD", "")
    if not naukri_email or not naukri_pass:
        log_fn("Skipped — set NAUKRI_EMAIL and NAUKRI_PASSWORD in .env.")
        return 0, 0

    from scanner.naukri_playwright import SESSION_FILE as _NK_SESSION
    if not _NK_SESSION.exists():
        log_fn("Logging in to Naukri…")
        if not naukri_login(naukri_email, naukri_pass):
            log_fn("Naukri login failed.")
            return 0, 0

    return _do_scan_core(naukri_search, keywords, location, results, hours, log_fn, progress_fn)


def _run_jsearch(keywords, location, results, hours, log_fn, progress_fn):
    if not os.getenv("JSEARCH_API_KEY"):
        log_fn("Skipped — set JSEARCH_API_KEY in .env.")
        return 0, 0
    return _do_scan_core(jsearch_search_jobs, keywords, location, results, hours, log_fn, progress_fn)


def _run_company_boards(keywords, location, results, hours, log_fn, progress_fn):
    boards = get_company_boards()
    if not boards:
        log_fn("Skipped — no company boards saved (add some in the Profile tab).")
        return 0, 0

    total = len(boards)
    all_dfs: list[pd.DataFrame] = []
    for i, board in enumerate(boards, start=1):
        progress_fn((i - 1) / total, f"Scanning {board['name']} ({i}/{total})…")
        fetch_fn = ATS_FETCHERS.get(board["ats"])
        if not fetch_fn:
            log_fn(f"[{i}/{total}] {board['name']}: unknown ATS '{board['ats']}', skipped")
            progress_fn(i / total, f"Skipped {board['name']} ({i}/{total})")
            continue
        log_fn(f"[{i}/{total}] Scanning {board['name']} ({board['ats']})…")
        try:
            df = fetch_fn(board["token"], board["name"])
            if not df.empty:
                all_dfs.append(df)
                log_fn(f"[{i}/{total}] {board['name']}: {len(df)} job(s) found")
            else:
                log_fn(f"[{i}/{total}] {board['name']}: no jobs found")
        except Exception as e:
            log_fn(f"[{i}/{total}] {board['name']} FAILED: {e}")
        progress_fn(i / total, f"Scanned {board['name']} ({i}/{total})")

    if not all_dfs:
        log_fn("No jobs found across saved company boards.")
        return 0, 0

    combined = pd.concat(all_dfs, ignore_index=True)
    kw_list  = _split_keywords(keywords)
    if kw_list:
        before   = len(combined)
        combined = filter_by_keywords(combined, kw_list)
        log_fn(f"Keyword filter: {before} → {len(combined)} job(s)")

    log_fn(f"Saving {len(combined)} job(s)…")
    new_count = save_jobs(combined)
    log_fn(f"Done — {len(combined)} found, {new_count} new.")
    return len(combined), new_count


# Registry used by both the individual sidebar buttons and "Scan All (parallel)".
_SCAN_SOURCES = [
    ("LinkedIn (jobspy)", _run_jobspy),
    ("LinkedIn (login)",  _run_linkedin_login),
    ("Naukri",            _run_naukri),
    ("Company Boards",    _run_company_boards),
    ("JSearch",           _run_jsearch),
]
