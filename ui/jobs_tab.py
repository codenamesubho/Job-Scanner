import pandas as pd
import streamlit as st

from scanner import add_job_by_url, get_jobs, get_stats, parse_jd_extracted

from .constants import DEFAULT_MIN_SCORE, STATUSES
from .detail_panel import _job_detail_dialog
from .scan_handlers import (
    handle_company_boards_scan, handle_jobspy_scan, handle_jsearch_scan,
    handle_linkedin_scan, handle_naukri_scan, handle_scan_all,
)


def _render_stats() -> None:
    stats = get_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total",       stats["total"])
    m2.metric("New",         stats["new"])
    m3.metric("Shortlisted", stats["shortlisted"])
    m4.metric("Applied",     stats["applied"])
    m5.metric("Rejected",    stats["rejected"])


def _company_type_options() -> list[str]:
    """Distinct `company_type` values pulled from jobs.jd_extracted (populated
    under SCORING_MODE=structured only). Returns [] otherwise, which collapses
    the filter to just "All"."""
    all_jobs = get_jobs()
    if "jd_extracted" not in all_jobs.columns:
        return []
    types = set()
    for raw in all_jobs["jd_extracted"]:
        parsed = parse_jd_extracted(raw)
        if parsed and parsed.get("company_type"):
            types.add(parsed["company_type"])
    return sorted(types)


def _render_add_job_by_url() -> None:
    with st.expander("➕ Add job by URL"):
        with st.form("add_job_by_url_form", clear_on_submit=True):
            url       = st.text_input("Job URL", placeholder="https://...")
            submitted = st.form_submit_button("Add Job", type="primary")
        if submitted:
            if not url.strip():
                st.warning("Please paste a job URL.")
            else:
                success, message = add_job_by_url(url)
                (st.success if success else st.error)(message)
                if success:
                    st.rerun()


def _render_filters() -> tuple[str, str, bool, str]:
    f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
    search_text   = f1.text_input("Search title / company / location",
                                  placeholder="e.g. senior, Google…")
    status_filter = f2.selectbox("Status", ["All"] + STATUSES)
    f4.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    remote_only   = f4.checkbox("Remote only")
    company_type_filter = f3.selectbox("Company type", ["All"] + _company_type_options())
    return search_text, status_filter, remote_only, company_type_filter


def _render_jobs_table(jobs: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if jobs.empty:
        st.info("No jobs match your filters. Run a scan from the sidebar.")
        return pd.DataFrame(), []

    st.caption(f"{len(jobs)} job(s) — click a row to view details")
    # structured_score leads as the primary score column (see ui/scoring.py's
    # _render_score_display); raw score is kept alongside for comparison.
    display_cols = ["title", "company", "location", "structured_score", "score",
                    "is_remote", "date_posted", "status", "first_seen"]
    display_cols = [c for c in display_cols if c in jobs.columns]
    jobs_reset   = jobs.reset_index(drop=True)

    event = st.dataframe(
        jobs_reset[display_cols],
        selection_mode="single-row",
        on_select="rerun",
        key="jobs_table",
        column_config={
            "structured_score": st.column_config.NumberColumn("Score", format="%d", width="small"),
            "score":            st.column_config.NumberColumn("Raw Score", format="%d", width="small"),
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

    _render_add_job_by_url()

    search_text, status_filter, remote_only, company_type_filter = _render_filters()
    min_score = st.slider("Minimum score", 0, 100, value=DEFAULT_MIN_SCORE)

    jobs = get_jobs(
        status=None if status_filter == "All" else status_filter,
        search=search_text or None,
    )
    if remote_only:
        jobs = jobs[jobs["is_remote"] == 1]
    if company_type_filter != "All" and "jd_extracted" in jobs.columns:
        jobs = jobs[jobs["jd_extracted"].apply(
            lambda raw: (parse_jd_extracted(raw) or {}).get("company_type") == company_type_filter
        )]
    # Primary score is structured_score (falling back to raw score where
    # structured scoring hasn't run) — the slider and sort both key off it.
    if "structured_score" in jobs.columns or "score" in jobs.columns:
        primary_score = jobs.get("structured_score")
        if primary_score is None:
            primary_score = jobs["score"]
        elif "score" in jobs.columns:
            primary_score = primary_score.fillna(jobs["score"])

        jobs = jobs.assign(_primary_score=primary_score)
        jobs = jobs[jobs["_primary_score"].isna() | (jobs["_primary_score"] >= min_score)]
        jobs = jobs.sort_values("_primary_score", ascending=False, na_position="last")
        jobs = jobs.drop(columns="_primary_score")

    jobs_reset, selected_rows = _render_jobs_table(jobs)

    row_idx = selected_rows[0] if selected_rows else None
    if row_idx is not None and row_idx < len(jobs_reset):
        sel    = jobs_reset.iloc[row_idx]
        job_id = str(sel["id"])
        _job_detail_dialog(sel, job_id)
