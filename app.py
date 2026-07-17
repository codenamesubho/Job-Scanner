import json
import os
import threading
import time

import streamlit as st
import pandas as pd

from scanner import (
    search_jobs, save_jobs, get_jobs, update_status, get_stats, update_scores,
    update_structured_scores, parse_jd_extracted,
    scoreable_jobs,
    save_candidate, get_candidate,
    save_resume, get_latest_resume, extract_text,
    save_criteria, get_criteria, get_all_criteria, delete_criteria,
    save_company_board, get_company_boards, delete_company_board,
    generate_summary, score_jobs, scoring_breaker_status,
    scoring_mode, score_jobs_structured, extract_resume_profile, extract_job_requirements,
    extract_missing_job_requirements, load_resume_profile,
    save_referral, get_referrals, delete_referral,
    draft_referral_message,
    linkedin_login, linkedin_playwright_search, find_referral_contacts,
    send_linkedin_message,
    naukri_login, naukri_search,
    ATS_FETCHERS, jsearch_search_jobs,
    apply_and_prefill, parse_score_breakdown,
)
from scanner.filters import filter_by_keywords
from scanner.database import update_job_fields

st.set_page_config(page_title="Job Scanner", page_icon="💼", layout="wide")

STATUSES      = ["new", "saved", "applied", "rejected"]
HOURS_OPTIONS = [24, 48, 72, 168, 336, 720]

# Match-score color thresholds (out of 100) shown next to a job's score.
SCORE_GOOD_THRESHOLD = 80
SCORE_OK_THRESHOLD   = 60
DEFAULT_MIN_SCORE    = 65

# Background-job polling: how often the main thread refreshes progress
# bars/log placeholders while a scan or scoring worker thread is running.
POLL_INTERVAL_S         = 0.4
SCORE_BUTTON_POLL_S     = 0.5  # _render_score_button ticks via st.rerun() instead of a blocking loop
LOG_TAIL_LINES          = 8    # single-source log placeholders (individual scans, score button)
AUTO_SCORE_LOG_TAIL_LINES = 6  # _auto_score_new's log placeholder
SCAN_ALL_LOG_TAIL_LINES = 4    # per-source log placeholders in "Scan All" (narrower, one of several)

# ── Must run before any widget is created ─────────────────────────────────────
# Load/Save buttons in the Profile tab set _profile_load and call st.rerun().
# On the next run this block fires first so the widget keys are set before
# Streamlit instantiates the sidebar widgets (writing to widget keys after
# widget creation raises StreamlitAPIException).
if "_profile_load" in st.session_state:
    _pl = st.session_state.pop("_profile_load")
    st.session_state.sb_keywords = _pl["keywords"]
    st.session_state.sb_location = _pl["location"]
    st.session_state.sb_results  = _pl["results"]
    st.session_state.sb_hours    = _pl["hours"]

# Seed sidebar defaults from DB once per session
_crit = get_criteria()
for _key, _field, _default in (
    ("sb_keywords", "keywords", "software engineer"),
    ("sb_location", "location", "USA"),
    ("sb_results",  "results",  25),
    ("sb_hours",    "hours",    72),
):
    if _key not in st.session_state:
        st.session_state[_key] = _crit.get(_field, _default)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _split_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _auto_score_new() -> None:
    """Phase 2 (dedup already-scored jobs) + Phase 3 (score what's left) of the
    scan pipeline. Phase 1 (fetch) is whatever runner called this — fetching
    itself already dedupes cross-source at insert time (scanner.database.save_jobs);
    this phase additionally skips jobs that already have a score from a
    previous scan, so only genuinely new jobs reach the LLM.

    Under SCORING_MODE=structured, extracts any missing structured JD/resume
    JSON first (blocking — see extract_missing_job_requirements), then scores
    against structured_score instead of score (see scanner.scoring.score_unscored_jobs
    for the same two-mode logic used by the CLI/cron path).
    """
    cand_data = get_candidate()
    if not cand_data.get("summary"):
        st.caption("⚠ No candidate summary — skipping scoring. Add one in Profile.")
        return

    breaker = scoring_breaker_status()
    if breaker["open"]:
        st.error(
            f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
            f"({breaker['reason']})."
        )
        return

    structured = scoring_mode() == "structured"
    resume_profile = None

    if structured:
        st.caption("Extracting structured JD data for jobs missing it…")
        extract_missing_job_requirements(log_fn=lambda msg: None)
        resume_profile = load_resume_profile(cand_data, log_fn=lambda msg: None)
        if resume_profile is None:
            st.caption("Structured scoring skipped — could not load a structured resume profile.")
            return

        st.caption("Phase 2/3: checking for jobs missing a structured score…")
        pending = get_jobs(missing_structured_score=True)
        if "jd_extracted" not in pending.columns or pending.empty:
            st.caption("No jobs need structured scoring.")
            return
        scoreable = scoreable_jobs(pending)
        if scoreable.empty:
            st.caption("No jobs need structured scoring.")
            return
        jobs_list = []
        for _, row in scoreable.iterrows():
            requirements = parse_jd_extracted(row.get("jd_extracted"))
            if requirements is not None:
                jobs_list.append({
                    "id": row["id"], "requirements": requirements,
                    "is_remote": bool(row.get("is_remote")),
                    "title": row.get("title", ""), "company": row.get("company", ""),
                })
        if not jobs_list:
            st.caption("No jobs have structured JD data to score against yet.")
            return
        total = len(jobs_list)
        st.caption(f"{total} new job(s) need structured scoring.")
    else:
        st.caption("Phase 2/3: checking for already-scored jobs…")
        unscored = get_jobs(unscored_only=True)
        if "description" not in unscored.columns or unscored.empty:
            st.caption("No new jobs need scoring.")
            return

        scoreable     = scoreable_jobs(unscored)
        no_desc_count = len(unscored) - len(scoreable)
        if scoreable.empty:
            msg = "No new jobs need scoring."
            if no_desc_count:
                msg += f" ({no_desc_count} new job(s) have no description to score.)"
            st.caption(msg)
            return

        jobs_list  = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")
        total      = len(jobs_list)
        skip_note  = f" ({no_desc_count} skipped — no description)" if no_desc_count else ""
        st.caption(f"{total} new job(s) need scoring{skip_note}.")

    # score_jobs()/score_jobs_structured() run multiple batches concurrently,
    # each with its own heartbeat sub-thread logging "still running…" — so
    # all logging here must go through a locked shared dict, with only the
    # main thread ever touching the log_box/progress widgets (same pattern
    # as handle_scan_all).
    state_lock = threading.Lock()
    state = {"log": [], "scored": 0, "done": False, "error": None}

    def _log_fn(msg: str) -> None:
        with state_lock:
            state["log"].append(msg)

    def _worker() -> None:
        try:
            if structured:
                score_iter = score_jobs_structured(resume_profile, jobs_list, log_fn=_log_fn)
                update_fn = update_structured_scores
            else:
                score_iter = score_jobs(cand_data["summary"], jobs_list, log_fn=_log_fn)
                update_fn = update_scores
            for result in score_iter:
                if result:
                    update_fn(result)
                    with state_lock:
                        state["scored"] += len(result)
        except Exception as e:
            with state_lock:
                state["error"] = str(e)
        finally:
            with state_lock:
                state["done"] = True

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    progress = st.progress(0, text=f"Scoring 0 / {total} job(s)…")
    log_box  = st.empty()

    while thread.is_alive():
        with state_lock:
            scored = state["scored"]
            lines  = list(state["log"][-AUTO_SCORE_LOG_TAIL_LINES:])
        progress.progress(min(scored / total, 1.0), text=f"Scoring {scored} / {total} job(s)…")
        log_box.code("\n".join(lines))
        time.sleep(POLL_INTERVAL_S)

    with state_lock:
        scored = state["scored"]
        error  = state["error"]

    progress.empty()
    log_box.empty()
    if error:
        st.warning(f"Scoring failed: {error}")
    else:
        st.info(f"Scored {scored} new job(s).")


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
    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        log_lines.append(msg)
        if log_box is not None:
            log_box.code("\n".join(log_lines[-LOG_TAIL_LINES:]))

    def progress_fn(frac: float, text: str) -> None:
        if progress is not None:
            progress.progress(frac, text=text)

    return log_fn, progress_fn


# ══════════════════════════════════════════════════════════════════════════════
# Scan "runners" — one per source, Streamlit-agnostic (log_fn/progress_fn only)
# so the exact same logic can run from an individual sidebar button (main
# thread, updates st.* placeholders) or from a background thread as part of
# "Scan All" (updates a locked shared dict instead). Each returns (found, new).
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, str, int, int, bool, bool, bool, bool, bool, bool]:
    with st.sidebar:
        st.title("💼 Job Scanner")
        st.subheader("Search Settings")

        keywords = st.text_input(
            "Keywords (comma-separated)",
            key="sb_keywords",
            help="e.g. backend engineer, python developer",
        )
        location = st.text_input("Location", key="sb_location")
        results  = st.slider("Max results per keyword", 5, 100, key="sb_results")

        hours = st.selectbox(
            "Posted within", HOURS_OPTIONS,
            format_func=lambda h: f"{h}h ({h // 24}d)",
            key="sb_hours",
        )

        st.markdown("**Scan All Sources**")
        scan_all_clicked = st.button(
            "🚀 Scan All (parallel)", use_container_width=True, type="primary",
            help="Runs every source below at once, each in its own thread, "
                 "with a live progress bar + log per source.",
        )

        st.markdown("**Individual Sources**")
        scan_clicked   = st.button("🔍 LinkedIn (jobspy)",  use_container_width=True)
        li_pw_clicked  = st.button("🔗 LinkedIn (login)",   use_container_width=True)
        naukri_clicked = st.button("💼 Naukri",              use_container_width=True)
        boards_clicked = st.button("🏢 Company Boards",     use_container_width=True,
                                    help="Pulls from every saved Greenhouse/Lever/Ashby company board "
                                         "(see Profile tab) — ignores 'Posted within'.")
        jsearch_clicked = st.button("🌐 JSearch",            use_container_width=True,
                                     help="Aggregator covering LinkedIn/Indeed/Glassdoor/ZipRecruiter. "
                                          "Requires JSEARCH_API_KEY in .env.")

    return (keywords, location, results, hours, scan_all_clicked,
            scan_clicked, li_pw_clicked, naukri_clicked, boards_clicked, jsearch_clicked)


# ══════════════════════════════════════════════════════════════════════════════
# Scan handlers — individual sources (thin wrappers around the runners above)
# ══════════════════════════════════════════════════════════════════════════════

def _run_individual(name: str, runner, keywords: str, location: str, results: int, hours: int) -> None:
    progress = st.progress(0, text=f"Starting {name} scan…")
    log_box  = st.empty()
    log_fn, progress_fn = _streamlit_log_progress(log_box, progress)
    try:
        found, new_count = runner(keywords, location, results, hours, log_fn, progress_fn)
        progress.empty()
        log_box.empty()
        if found == 0 and new_count == 0:
            st.info(f"{name}: nothing found (see log above if this is unexpected).")
        else:
            st.success(f"{name}: {found} found, {new_count} new.")
        _auto_score_new()
    except Exception as e:
        progress.empty()
        log_box.empty()
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


# ══════════════════════════════════════════════════════════════════════════════
# Scan handler — all sources in parallel threads
# ══════════════════════════════════════════════════════════════════════════════

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
        li_log_box.empty()
        st.success(f"LinkedIn (login): {found} found, {new_count} new.")
    except Exception as e:
        li_progress.empty()
        li_log_box.empty()
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
                lbox.code("\n".join(s["log"][-SCAN_ALL_LOG_TAIL_LINES:]))
        time.sleep(POLL_INTERVAL_S)

    with state_lock:
        for name, (pbar, lbox) in placeholders.items():
            s = state[name]
            pbar.progress(1.0, text=f"{name}: done")
            lbox.code("\n".join(s["log"][-SCAN_ALL_LOG_TAIL_LINES:]))
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


# ══════════════════════════════════════════════════════════════════════════════
# Jobs tab — sub-components
# ══════════════════════════════════════════════════════════════════════════════

def _render_stats() -> None:
    stats = get_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total",    stats["total"])
    m2.metric("New",      stats["new"])
    m3.metric("Saved",    stats["saved"])
    m4.metric("Applied",  stats["applied"])
    m5.metric("Rejected", stats["rejected"])


def _render_score_button() -> None:
    """Runs scoring in a background thread, tracked in session_state, and
    polls via short sleep + st.rerun() ticks (rather than one long blocking
    loop) so the button click that starts a re-run can actually be delivered
    and processed while a job is in progress — that's what lets a second
    click on this same button act as Cancel instead of being ignored until
    the whole run finishes.
    """
    job = st.session_state.get("_score_job")
    _, score_col = st.columns([4, 1])

    if job is None:
        if not score_col.button("🎯 Score Jobs", use_container_width=True):
            return

        cand    = get_candidate()
        summary = cand.get("summary", "")
        if not summary:
            st.warning("No candidate summary found. Generate one in the Profile tab first.")
            return

        breaker = scoring_breaker_status()
        if breaker["open"]:
            st.error(
                f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
                f"({breaker['reason']})."
            )
            return

        structured = scoring_mode() == "structured"
        resume_profile = None

        if structured:
            extract_missing_job_requirements(log_fn=lambda msg: None)
            resume_profile = load_resume_profile(cand, log_fn=lambda msg: None)
            if resume_profile is None:
                st.info("Structured scoring skipped — could not load a structured resume profile.")
                return
            pending = get_jobs(missing_structured_score=True)
            scoreable = scoreable_jobs(pending) if "jd_extracted" in pending.columns else pending.iloc[0:0]
            if scoreable.empty:
                st.info("No jobs need structured scoring.")
                return
            jobs_list = []
            for _, row in scoreable.iterrows():
                requirements = parse_jd_extracted(row.get("jd_extracted"))
                if requirements is not None:
                    jobs_list.append({
                        "id": row["id"], "requirements": requirements,
                        "is_remote": bool(row.get("is_remote")),
                        "title": row.get("title", ""), "company": row.get("company", ""),
                    })
            if not jobs_list:
                st.info("No jobs have structured JD data to score against yet.")
                return
        else:
            all_jobs  = get_jobs()
            scoreable = scoreable_jobs(all_jobs)
            if scoreable.empty:
                st.info("No jobs with descriptions to score.")
                return
            jobs_list = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")

        cancel_event = threading.Event()
        state_lock   = threading.Lock()
        state = {"log": [], "scored": 0, "done": False, "error": None}

        def _log_fn(msg: str) -> None:
            with state_lock:
                state["log"].append(msg)

        def _worker() -> None:
            try:
                if structured:
                    score_iter = score_jobs_structured(resume_profile, jobs_list, log_fn=_log_fn, cancel_event=cancel_event)
                    update_fn = update_structured_scores
                else:
                    score_iter = score_jobs(summary, jobs_list, log_fn=_log_fn, cancel_event=cancel_event)
                    update_fn = update_scores
                for result in score_iter:
                    if result:
                        update_fn(result)
                        with state_lock:
                            state["scored"] += len(result)
            except Exception as e:
                with state_lock:
                    state["error"] = str(e)
            finally:
                with state_lock:
                    state["done"] = True

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        st.session_state["_score_job"] = {
            "thread": thread, "cancel_event": cancel_event, "lock": state_lock,
            "state": state, "total": len(jobs_list),
        }
        st.rerun()
        return

    # A job is already running — this click cancels it instead of starting another.
    if score_col.button("🛑 Cancel Scoring", use_container_width=True):
        job["cancel_event"].set()

    with job["lock"]:
        scored = job["state"]["scored"]
        lines  = list(job["state"]["log"][-LOG_TAIL_LINES:])
        done   = job["state"]["done"]
        error  = job["state"]["error"]
    total     = job["total"]
    cancelled = job["cancel_event"].is_set()

    st.progress(min(scored / total, 1.0) if total else 1.0, text=f"Scoring {scored} / {total} job(s)…")
    st.code("\n".join(lines))

    if done:
        del st.session_state["_score_job"]
        if error:
            st.error(f"Scoring failed: {error}")
        elif cancelled:
            st.warning(f"Scoring cancelled — {scored} job(s) scored before stopping.")
        else:
            st.success(f"Scored {scored} job(s).")
    else:
        time.sleep(SCORE_BUTTON_POLL_S)
        st.rerun()


def _render_filters() -> tuple[str, str, bool, int]:
    f1, f2, f3 = st.columns([2, 2, 1])
    search_text   = f1.text_input("Search title / company / location",
                                  placeholder="e.g. senior, Google…")
    status_filter = f2.selectbox("Status", ["All"] + STATUSES)
    remote_only   = f3.checkbox("Remote only")
    min_score     = st.slider("Minimum score", 0, 100, value=DEFAULT_MIN_SCORE)
    return search_text, status_filter, remote_only, min_score


def _render_jobs_table(jobs: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if jobs.empty:
        st.info("No jobs match your filters. Run a scan from the sidebar.")
        return pd.DataFrame(), []

    st.caption(f"{len(jobs)} job(s) — click a row to view details")
    # TEMP: structured_score shown alongside score for side-by-side comparison
    # while evaluating SCORING_MODE=structured — remove column once done comparing.
    display_cols = ["title", "company", "location", "score", "structured_score",
                    "is_remote", "date_posted", "status", "first_seen"]
    display_cols = [c for c in display_cols if c in jobs.columns]
    jobs_reset   = jobs.reset_index(drop=True)

    event = st.dataframe(
        jobs_reset[display_cols],
        selection_mode="single-row",
        on_select="rerun",
        key="jobs_table",
        column_config={
            "score":            st.column_config.NumberColumn("Score", format="%d", width="small"),
            "structured_score": st.column_config.NumberColumn("Structured Score", format="%d", width="small"),
            "title":            st.column_config.TextColumn("Title", width="large"),
            "company":          st.column_config.TextColumn("Company"),
            "location":         st.column_config.TextColumn("Location"),
            "is_remote":        st.column_config.CheckboxColumn("Remote"),
            "date_posted":      st.column_config.TextColumn("Posted"),
            "status":           st.column_config.TextColumn("Status"),
            "first_seen":       st.column_config.TextColumn("First Seen"),
        },
        hide_index=True,
        use_container_width=True,
        height=600,
    )
    return jobs_reset, event.selection.rows


def _score_color(score: int) -> str:
    if score >= SCORE_GOOD_THRESHOLD:
        return "🟢"
    if score >= SCORE_OK_THRESHOLD:
        return "🟡"
    return "🔴"


def _render_score_display(sel: pd.Series) -> None:
    score_val = sel.get("score")
    structured_score_val = sel.get("structured_score")
    score_col, btn_col = st.columns([3, 1])
    if pd.notna(score_val):
        breakdown_raw = sel.get("score_breakdown", "") or ""
        reason        = sel.get("score_reason", "") or ""

        parsed   = parse_score_breakdown(breakdown_raw, fallback_score=score_val)
        computed = parsed["computed_score"]
        score_col.metric(label="Match Score", value=f"{_score_color(computed)} {computed} / 100")

        if parsed["items"]:
            for label, sc, mx, rsn in parsed["items"]:
                line = f"**{label}**: {sc}/{mx}"
                if rsn:
                    line += f" — {rsn}"
                st.markdown(line)
        elif parsed["legacy_lines"]:
            st.markdown("\n".join(f"- {p}" for p in parsed["legacy_lines"]))

        if reason:
            st.caption(reason)
    else:
        score_col.caption("Not yet scored.")

    # Structured score (SCORING_MODE=structured) is kept in separate columns
    # so it can be compared against the raw score above for the same job.
    if pd.notna(structured_score_val):
        structured_breakdown_raw = sel.get("structured_score_breakdown", "") or ""
        structured_reason = sel.get("structured_score_reason", "") or ""

        structured_parsed = parse_score_breakdown(structured_breakdown_raw, fallback_score=structured_score_val)
        structured_computed = structured_parsed["computed_score"]
        st.markdown(f"**{_score_color(structured_computed)} Structured: {structured_computed} / 100**")

        if structured_parsed["items"]:
            for label, sc, mx, rsn in structured_parsed["items"]:
                line = f"**{label}**: {sc}/{mx}"
                if rsn:
                    line += f" — {rsn}"
                st.markdown(line)
        elif structured_parsed["legacy_lines"]:
            st.markdown("\n".join(f"- {p}" for p in structured_parsed["legacy_lines"]))

        if structured_reason:
            st.caption(structured_reason)

    job_id = sel.get("id", "")
    if btn_col.button("↺ Rescore", key=f"rescore_{job_id}", use_container_width=True):
        cand = get_candidate()
        summary = cand.get("summary", "")
        if not summary:
            st.warning("No candidate summary — generate one in the Profile tab first.")
        elif not (sel.get("description") or "").strip():
            st.warning("No description available for this job.")
        elif scoring_mode() == "structured":
            with st.spinner("Rescoring (structured)…"):
                requirements = parse_jd_extracted(sel.get("jd_extracted"))
                if requirements is None:
                    try:
                        req = extract_job_requirements(sel.get("description", ""), sel.get("company"))
                        requirements = req.model_dump()
                        update_job_fields(job_id, {"jd_extracted": json.dumps(requirements)})
                    except Exception as e:
                        st.error(f"Structured extraction failed: {e}")
                        requirements = None
                resume_profile = load_resume_profile(cand, log_fn=lambda msg: None) if requirements else None
                if requirements is not None and resume_profile is not None:
                    for result in score_jobs_structured(resume_profile, [{
                        "id": job_id, "requirements": requirements,
                        "is_remote": bool(sel.get("is_remote")),
                        "title": sel.get("title", ""), "company": sel.get("company", ""),
                    }]):
                        if result:
                            update_structured_scores(result)
            st.toast("Rescored (structured)!")
            st.rerun()
        else:
            job_dict = {
                "id": job_id,
                "title": sel.get("title", ""),
                "company": sel.get("company", ""),
                "description": sel.get("description", ""),
                "is_remote": sel.get("is_remote", False),
            }
            with st.spinner("Rescoring…"):
                for result in score_jobs(summary, [job_dict]):
                    if result:
                        update_scores(result)
            st.toast("Rescored!")
            st.rerun()


def _render_status_editor(sel: pd.Series, job_id: str) -> None:
    st_col, sv_col = st.columns([2, 1])
    cur_status = sel.get("status", "new")
    new_status = st_col.selectbox(
        "Status", STATUSES,
        index=STATUSES.index(cur_status) if cur_status in STATUSES else 0,
        key=f"status_sel_{job_id}",
    )
    if sv_col.button("Save status", key=f"status_save_{job_id}", use_container_width=True):
        update_status(job_id, new_status)
        st.toast(f"Status → '{new_status}'.")
        st.rerun()


def _render_find_contacts_button(sel: pd.Series, job_id: str) -> None:
    ref_h, ref_btn_col = st.columns([4, 1])
    ref_h.markdown("#### Referrals")
    if not ref_btn_col.button(
        "🤝 Find contacts",
        key=f"find_ref_{job_id}",
        use_container_width=True,
        help="Search LinkedIn for 1st/2nd-degree connections at this company",
    ):
        return
    with st.spinner(f"Searching LinkedIn at {sel.get('company', '')}…"):
        try:
            contacts = find_referral_contacts(
                company=sel.get("company", ""),
                job_title=sel.get("title", ""),
            )
            if contacts:
                st.session_state[f"_contacts_{job_id}"] = contacts
            else:
                st.info("No 1st/2nd-degree contacts found.")
        except Exception as e:
            st.error(f"Referral search failed: {e}")


def _render_saved_referrals(job_id: str) -> None:
    for ref in get_referrals(job_id):
        degree_badge = f" · {ref['degree']}" if ref.get("degree") else ""
        with st.expander(f"💬 {ref['name']} — {ref.get('title', '')}{degree_badge}"):
            photo_col, info_col = st.columns([1, 4])
            if ref.get("photo_url"):
                photo_col.image(ref["photo_url"], width=64)
            if ref.get("linkedin_url"):
                info_col.markdown(
                    f"**{ref['name']}**  \n"
                    f"{ref.get('title', '')}  \n"
                    f"[View Profile ↗]({ref['linkedin_url']})"
                )
            if ref.get("message"):
                st.text_area("Message", value=ref["message"], height=100,
                             key=f"saved_msg_{ref['id']}", disabled=True)
            if st.button("🗑 Delete", key=f"del_ref_{ref['id']}"):
                delete_referral(ref["id"])
                st.rerun()


def _render_contact_draft_and_send(sel: pd.Series, job_id: str, i: int, contact: dict, cand: dict) -> None:
    draft_key = f"_draft_{job_id}_{i}"
    if st.button("✍️ Draft Message", key=f"draft_btn_{job_id}_{i}"):
        with st.spinner("Drafting…"):
            try:
                msg = draft_referral_message(
                    candidate_summary=cand.get("summary", ""),
                    contact=contact,
                    job={
                        "title":         sel.get("title", ""),
                        "company":       sel.get("company", ""),
                        "job_url":       sel.get("job_url", ""),
                        "job_url_direct": sel.get("job_url_direct", ""),
                    },
                )
                st.session_state[draft_key] = msg
            except Exception as e:
                st.error(f"Draft failed: {e}")

    if draft_key not in st.session_state:
        return

    edited_msg = st.text_area(
        "Message (edit before saving)",
        value=st.session_state[draft_key],
        height=130,
        key=f"ta_{draft_key}",
    )
    send_mode = st.radio(
        "Send mode",
        ["Auto-send", "Fill & send manually"],
        horizontal=True,
        key=f"send_mode_{job_id}_{i}",
        label_visibility="collapsed",
    )
    auto_send = send_mode == "Auto-send"
    btn_save, btn_send = st.columns(2)
    if btn_save.button("💾 Save Referral", key=f"save_ref_{job_id}_{i}",
                       use_container_width=True):
        save_referral(
            job_id=job_id,
            name=contact["name"],
            title=contact.get("title", ""),
            linkedin_url=contact.get("linkedin_url", ""),
            message=edited_msg,
            degree=contact.get("degree", ""),
            photo_url=contact.get("photo_url", ""),
        )
        del st.session_state[draft_key]
        st.session_state.pop(f"_contacts_{job_id}", None)
        st.toast("Referral saved.")
        st.rerun()
    send_label = "📨 Send on LinkedIn" if auto_send else "🖊 Open & pre-fill"
    send_spinner = (
        f"Sending to {contact['name']} on LinkedIn…"
        if auto_send else
        f"Opening LinkedIn for {contact['name']}…"
    )
    if btn_send.button(send_label, key=f"send_li_{job_id}_{i}",
                       use_container_width=True):
        profile_url = contact.get("linkedin_url", "")
        if not profile_url:
            st.error("No LinkedIn URL for this contact.")
        else:
            with st.spinner(send_spinner):
                try:
                    ok = send_linkedin_message(
                        profile_url, edited_msg, auto_send=auto_send,
                    )
                    if ok:
                        if auto_send:
                            st.toast("Message sent on LinkedIn!")
                            save_referral(
                                job_id=job_id,
                                name=contact["name"],
                                title=contact.get("title", ""),
                                linkedin_url=profile_url,
                                message=edited_msg,
                                degree=contact.get("degree", ""),
                                photo_url=contact.get("photo_url", ""),
                            )
                            del st.session_state[draft_key]
                            st.session_state.pop(f"_contacts_{job_id}", None)
                            st.rerun()
                        else:
                            st.info(
                                "Browser opened with message pre-filled. "
                                "Review, edit if needed, then click Send in LinkedIn."
                            )
                    else:
                        st.error(
                            "Message button not found — you may not be connected "
                            "or LinkedIn requires Premium to message this person."
                        )
                except Exception as e:
                    st.error(f"Failed: {e}")


def _render_new_contacts(sel: pd.Series, job_id: str) -> None:
    contacts = st.session_state.get(f"_contacts_{job_id}", [])
    if not contacts:
        return

    st.caption(
        f"{len(contacts)} contact(s) — "
        "1st-degree (any role) → 2nd-degree similar role → 2nd-degree managers → open search"
    )
    cand = get_candidate()
    for i, contact in enumerate(contacts):
        degree_badge = f" · {contact['degree']}" if contact.get("degree") else ""
        with st.expander(f"👤 {contact['name']} — {contact.get('title', '')}{degree_badge}"):
            photo_col, info_col = st.columns([1, 4])
            if contact.get("photo_url"):
                photo_col.image(contact["photo_url"], width=64)
            if contact.get("linkedin_url"):
                info_col.markdown(
                    f"**{contact['name']}**  \n"
                    f"{contact.get('title', '')}  \n"
                    f"[View Profile ↗]({contact['linkedin_url']})"
                )
            _render_contact_draft_and_send(sel, job_id, i, contact, cand)


def _render_referral_section(sel: pd.Series, job_id: str) -> None:
    _render_find_contacts_button(sel, job_id)
    _render_saved_referrals(job_id)
    _render_new_contacts(sel, job_id)


def _render_detail_panel(sel: pd.Series, job_id: str) -> None:
    remote_badge = "  🌐 Remote" if sel.get("is_remote") else ""
    st.markdown(
        f"### {sel.get('title', '')}  \n"
        f"**{sel.get('company', '')}** · {sel.get('location', '')}{remote_badge}"
    )
    job_url    = sel.get("job_url", "") or ""
    direct_url = sel.get("job_url_direct", "") or ""
    apply_url  = direct_url or job_url

    apply_col, listing_col, applied_col = st.columns([2, 2, 1])

    if job_url:
        listing_col.link_button("View Listing ↗", job_url, use_container_width=True)

    if apply_url:
        if apply_col.button("🚀 Apply", key=f"apply_{job_id}", use_container_width=True, type="primary"):
            log_lines: list[str] = []
            with st.spinner("Opening application & prefilling your info…"):
                try:
                    result = apply_and_prefill(apply_url, log_fn=log_lines.append)
                except Exception as e:
                    st.error(f"Could not open application: {e}")
                else:
                    filled = result.get("filled_fields") or []
                    resume_note = " + resume" if result.get("resume_attached") else ""
                    if result.get("success"):
                        st.toast(
                            f"Filled {len(filled)} field(s){resume_note}: {', '.join(filled)}. "
                            "Review & submit in the browser window."
                        )
                    else:
                        st.warning(
                            result.get("error")
                            or "Couldn't auto-detect form fields — browser opened for manual apply."
                        )
            if log_lines:
                with st.expander("Apply log", expanded=False):
                    st.code("\n".join(log_lines))

    if applied_col.button("✅ Mark Applied", key=f"mark_applied_{job_id}", use_container_width=True):
        update_status(job_id, "applied")
        st.toast("Status → 'applied'.")
        st.rerun()

    _render_score_display(sel)
    _render_status_editor(sel, job_id)
    st.divider()

    desc = sel.get("description", "") or ""
    if desc:
        with st.expander("📄 Job Description", expanded=True):
            st.markdown(desc)
    else:
        st.caption("No description saved for this job.")

    st.divider()
    _render_referral_section(sel, job_id)


@st.dialog("Job Details", width="large")
def _job_detail_dialog(sel: pd.Series, job_id: str) -> None:
    _render_detail_panel(sel, job_id)


def render_jobs_tab(keywords: str, location: str, results: int, hours: int,
                     scan_all_clicked: bool, scan_clicked: bool, li_pw_clicked: bool,
                     naukri_clicked: bool, boards_clicked: bool, jsearch_clicked: bool) -> None:
    _render_stats()

    if scan_all_clicked:
        handle_scan_all(keywords, location, results, hours)
    if scan_clicked:
        handle_jobspy_scan(keywords, location, results, hours)
    if li_pw_clicked:
        handle_linkedin_scan(keywords, location, results, hours)
    if naukri_clicked:
        handle_naukri_scan(keywords, location, results, hours)
    if boards_clicked:
        handle_company_boards_scan(keywords, location, results, hours)
    if jsearch_clicked:
        handle_jsearch_scan(keywords, location, results, hours)

    st.divider()
    _render_score_button()

    search_text, status_filter, remote_only, min_score = _render_filters()

    jobs = get_jobs(
        status=None if status_filter == "All" else status_filter,
        search=search_text or None,
    )
    if remote_only:
        jobs = jobs[jobs["is_remote"] == 1]
    if "score" in jobs.columns:
        jobs = jobs[jobs["score"].isna() | (jobs["score"] >= min_score)]

    jobs_reset, selected_rows = _render_jobs_table(jobs)

    row_idx = selected_rows[0] if selected_rows else None
    if row_idx is not None and row_idx < len(jobs_reset):
        sel    = jobs_reset.iloc[row_idx]
        job_id = str(sel["id"])
        _job_detail_dialog(sel, job_id)


# ══════════════════════════════════════════════════════════════════════════════
# Profile tab — sub-components
# ══════════════════════════════════════════════════════════════════════════════

def _render_candidate_section() -> None:
    st.subheader("Candidate Details")
    cand = get_candidate()

    with st.form("candidate_form"):
        c1, c2 = st.columns(2)
        name       = c1.text_input("Full Name",        value=cand.get("name", ""))
        email      = c2.text_input("Email",             value=cand.get("email", ""))
        phone      = c1.text_input("Phone",             value=cand.get("phone", ""))
        linkedin   = c2.text_input("LinkedIn URL",      value=cand.get("linkedin", ""))
        curr_title = c1.text_input("Current Job Title", value=cand.get("title", ""))
        years_exp  = c2.number_input("Years of Experience", min_value=0, max_value=50,
                                     value=int(cand.get("years_exp") or 0))
        summary    = st.text_area("Professional Summary",
                                  value=cand.get("summary", ""),
                                  height=120,
                                  placeholder="Brief description of your background and goals…")
        btn_col1, btn_col2 = st.columns([1, 2])
        save_clicked = btn_col1.form_submit_button("💾 Save Details", type="primary")
        gen_clicked  = btn_col2.form_submit_button("✨ Generate from Resume")

    if save_clicked:
        save_candidate(name, email, phone, linkedin, curr_title, years_exp, summary)
        st.success("Candidate details saved.")

    if gen_clicked:
        save_candidate(name, email, phone, linkedin, curr_title, years_exp, summary)
        resume = get_latest_resume()
        if resume is None:
            st.warning("No resume uploaded yet.")
        else:
            text = extract_text(resume["filename"], resume["raw_content"])
            if not text:
                st.warning("Could not extract text from the resume file.")
            else:
                with st.spinner("Generating professional summary…"):
                    try:
                        new_summary = generate_summary(text)
                    except Exception as e:
                        st.error(f"Could not generate summary: {e}")
                        return

                resume_extracted_json = None
                try:
                    profile = extract_resume_profile(text)
                    resume_extracted_json = json.dumps(profile.model_dump())
                except Exception as e:
                    st.caption(f"Note: structured resume extraction failed ({e}) — summary still saved.")

                save_candidate(name, email, phone, linkedin, curr_title, years_exp,
                               new_summary, resume_extracted=resume_extracted_json)
                st.success("Professional summary generated and saved.")
                st.rerun()


def _render_resume_section() -> None:
    st.subheader("Resume")
    existing_resume = get_latest_resume()
    if existing_resume:
        st.caption(
            f"Current resume: **{existing_resume['filename']}** "
            f"(uploaded {existing_resume['uploaded_at'][:10]})"
        )
        dl_col, _ = st.columns([1, 3])
        dl_col.download_button(
            "⬇ Download",
            data=existing_resume["raw_content"],
            file_name=existing_resume["filename"],
            mime=existing_resume.get("content_type") or "application/octet-stream",
        )

    uploaded = st.file_uploader(
        "Upload resume (PDF or DOCX)", type=["pdf", "docx"],
        help="Replaces the previously stored resume.",
    )
    if uploaded is None:
        return

    file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_last_resume_key") == file_key:
        return

    st.session_state["_last_resume_key"] = file_key
    raw  = uploaded.read()
    save_resume(uploaded.name, uploaded.type, raw)
    text = extract_text(uploaded.name, raw)
    if text:
        try:
            with st.spinner("Generating professional summary from resume…"):
                new_summary = generate_summary(text)

            resume_extracted_json = None
            try:
                profile = extract_resume_profile(text)
                resume_extracted_json = json.dumps(profile.model_dump())
            except Exception as e:
                st.caption(f"Note: structured resume extraction failed ({e}) — summary still saved.")

            cand_now = get_candidate()
            save_candidate(
                cand_now.get("name", ""), cand_now.get("email", ""),
                cand_now.get("phone", ""), cand_now.get("linkedin", ""),
                cand_now.get("title", ""), int(cand_now.get("years_exp") or 0),
                new_summary, resume_extracted=resume_extracted_json,
            )
            st.toast("Resume saved and professional summary auto-generated.")
            st.rerun()
        except Exception as e:
            st.success(f"Resume '{uploaded.name}' saved.")
            if "CLAUDE_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e):
                st.info("Add CLAUDE_API_KEY or GEMINI_API_KEY to .env (and set LLM_PROVIDER) to auto-generate summaries.")
            else:
                st.warning(f"Could not auto-generate summary: {e}")
    else:
        st.success(f"Resume '{uploaded.name}' saved.")


def _render_profile_form() -> None:
    _ep      = st.session_state.get("_editing_profile")
    _is_edit = _ep is not None

    st.markdown(f"**Editing: {_ep['name']}**" if _is_edit else "**Add new profile**")

    _def_name     = _ep["name"]              if _is_edit else ""
    _def_keywords = _ep["keywords"]          if _is_edit else st.session_state.get("sb_keywords", "")
    _def_location = _ep["location"]          if _is_edit else st.session_state.get("sb_location", "USA")
    _def_results  = int(_ep["results"])      if _is_edit else int(st.session_state.get("sb_results", 25))
    _def_hours    = _ep["hours"]             if _is_edit else st.session_state.get("sb_hours", 72)
    _def_remote   = bool(_ep["remote_only"]) if _is_edit else False

    with st.form("profile_form", clear_on_submit=True):
        ap1, ap2 = st.columns(2)
        ap_name     = ap1.text_input("Profile name",
                                     value=_def_name, placeholder="e.g. Remote Python")
        ap_keywords = ap2.text_input("Keywords (comma-separated)",
                                     value=_def_keywords,
                                     placeholder="python developer, backend engineer")
        ap3, ap4, ap5, ap6 = st.columns(4)
        ap_location = ap3.text_input("Location", value=_def_location)
        ap_results  = ap4.number_input("Max results", min_value=5, max_value=100, value=_def_results)
        _ap_h_idx   = HOURS_OPTIONS.index(_def_hours) if _def_hours in HOURS_OPTIONS else 2
        ap_hours    = ap5.selectbox("Posted within", HOURS_OPTIONS, index=_ap_h_idx,
                                    format_func=lambda h: f"{h}h ({h // 24}d)")
        ap_remote   = ap6.checkbox("Remote only", value=_def_remote)

        btn1, btn2    = st.columns([1, 1])
        submit_label  = "💾 Update Profile" if _is_edit else "💾 Save Profile"
        submitted     = btn1.form_submit_button(submit_label, type="primary", use_container_width=True)
        cancelled     = _is_edit and btn2.form_submit_button("✕ Cancel", use_container_width=True)

    if cancelled:
        st.session_state.pop("_editing_profile", None)
        st.rerun()

    if submitted:
        if not ap_name.strip():
            st.warning("Please enter a profile name.")
            return
        _cid = _ep["id"] if _is_edit else None
        save_criteria(ap_name.strip(), ap_keywords, ap_location,
                      ap_results, ap_hours, ap_remote, criteria_id=_cid)
        st.session_state.pop("_editing_profile", None)
        st.session_state._profile_load = {
            "keywords": ap_keywords,
            "location": ap_location,
            "results":  ap_results,
            "hours":    ap_hours,
        }
        action = "updated" if _is_edit else "saved"
        st.success(f"Profile '{ap_name.strip()}' {action}.")
        st.rerun()


def _render_search_profiles_section() -> None:
    st.subheader("Search Profiles")
    st.caption(
        "Save multiple keyword/location combos. "
        "**Load** pushes values to the sidebar instantly."
    )

    all_profiles = get_all_criteria()
    if all_profiles:
        for prof in all_profiles:
            with st.expander(f"**{prof['name']}** — {prof['keywords']} · {prof['location']}"):
                p1, p2 = st.columns([3, 1])
                p1.markdown(
                    f"**Keywords:** {prof['keywords']}  \n"
                    f"**Location:** {prof['location']}  \n"
                    f"**Max results:** {prof['results']} · "
                    f"**Within:** {prof['hours']}h · "
                    f"**Remote only:** {'Yes' if prof['remote_only'] else 'No'}"
                )
                load_col, edit_col, del_col = p2.columns(3)
                if load_col.button("⬆ Load", key=f"load_prof_{prof['id']}",
                                   use_container_width=True):
                    st.session_state._profile_load = {
                        "keywords": prof["keywords"],
                        "location": prof["location"],
                        "results":  prof["results"],
                        "hours":    prof["hours"],
                    }
                    st.rerun()
                if edit_col.button("✏️", key=f"edit_prof_{prof['id']}",
                                   use_container_width=True, help="Edit this profile"):
                    st.session_state._editing_profile = prof
                    st.rerun()
                if del_col.button("🗑", key=f"del_prof_{prof['id']}",
                                  use_container_width=True):
                    if st.session_state.get("_editing_profile", {}).get("id") == prof["id"]:
                        st.session_state.pop("_editing_profile", None)
                    delete_criteria(prof["id"])
                    st.rerun()
    else:
        st.info("No saved profiles yet. Add one below.")

    _render_profile_form()


_ATS_OPTIONS = ["greenhouse", "lever", "ashby"]


def _render_company_boards_section() -> None:
    st.subheader("Company Boards")
    st.caption(
        "Add companies whose Greenhouse/Lever/Ashby job board you want to pull directly — "
        "the '🏢 Company Boards' sidebar button scans all of them at once."
    )

    boards = get_company_boards()
    if boards:
        for board in boards:
            with st.expander(f"**{board['name']}** — {board['ats']} ({board['token']})"):
                b1, b2 = st.columns([3, 1])
                b1.markdown(
                    f"**ATS:** {board['ats']}  \n"
                    f"**Board token:** {board['token']}"
                )
                if b2.button("🗑 Delete", key=f"del_board_{board['id']}",
                             use_container_width=True):
                    delete_company_board(board["id"])
                    st.rerun()
    else:
        st.info("No company boards saved yet. Add one below.")

    with st.form("company_board_form", clear_on_submit=True):
        cb1, cb2, cb3 = st.columns([2, 1, 2])
        cb_name  = cb1.text_input("Company name", placeholder="e.g. dbt Labs")
        cb_ats   = cb2.selectbox("ATS", _ATS_OPTIONS)
        cb_token = cb3.text_input("Board token/slug", placeholder="e.g. dbtlabs")
        submitted = st.form_submit_button("💾 Add Board", type="primary")

    if submitted:
        if not cb_name.strip() or not cb_token.strip():
            st.warning("Please enter both a company name and a board token.")
        else:
            save_company_board(cb_name.strip(), cb_ats, cb_token.strip())
            st.success(f"Added '{cb_name.strip()}' ({cb_ats}).")
            st.rerun()


def render_profile_tab() -> None:
    _render_candidate_section()
    st.divider()
    _render_resume_section()
    st.divider()
    _render_search_profiles_section()
    st.divider()
    _render_company_boards_section()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

(keywords, location, results, hours, scan_all_clicked,
 scan_clicked, li_pw_clicked, naukri_clicked, boards_clicked, jsearch_clicked) = render_sidebar()

tab_jobs, tab_profile = st.tabs(["📋 Jobs", "👤 Profile"])

with tab_jobs:
    render_jobs_tab(keywords, location, results, hours, scan_all_clicked, scan_clicked,
                     li_pw_clicked, naukri_clicked, boards_clicked, jsearch_clicked)

with tab_profile:
    render_profile_tab()
