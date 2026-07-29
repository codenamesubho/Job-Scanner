import pandas as pd
import streamlit as st

from scanner import get_jobs, get_stats

from .constants import DEFAULT_MIN_SCORE, STATUSES
from .detail_panel import _job_detail_dialog
from .scan_handlers import (
    handle_company_boards_scan, handle_jobspy_scan, handle_jsearch_scan,
    handle_linkedin_scan, handle_naukri_scan, handle_scan_all,
)
from .scoring import _render_score_button


def _render_stats() -> None:
    stats = get_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total",    stats["total"])
    m2.metric("New",      stats["new"])
    m3.metric("Saved",    stats["saved"])
    m4.metric("Applied",  stats["applied"])
    m5.metric("Rejected", stats["rejected"])


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
    if "structured_score" in jobs.columns:
        jobs = jobs.sort_values("structured_score", ascending=False, na_position="last")

    jobs_reset, selected_rows = _render_jobs_table(jobs)

    row_idx = selected_rows[0] if selected_rows else None
    if row_idx is not None and row_idx < len(jobs_reset):
        sel    = jobs_reset.iloc[row_idx]
        job_id = str(sel["id"])
        _job_detail_dialog(sel, job_id)
