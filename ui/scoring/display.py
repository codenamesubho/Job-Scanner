"""Rendering a job's score: the structured/raw breakdown panel and Rescore."""
import json

import pandas as pd
import streamlit as st

from scanner import (
    extract_job_requirements, get_candidate, load_resume_profile,
    parse_jd_extracted, parse_score_breakdown, score_jobs, score_jobs_structured,
    scoring_mode, update_scores, update_structured_scores,
)
from scanner.database import update_job_fields

from ..constants import SCORE_GOOD_THRESHOLD, SCORE_OK_THRESHOLD


def _score_color(score: int) -> str:
    if score >= SCORE_GOOD_THRESHOLD:
        return "🟢"
    if score >= SCORE_OK_THRESHOLD:
        return "🟡"
    return "🔴"


def _render_breakdown(parsed: dict) -> None:
    """Render one parsed score breakdown — the per-criterion lines, or the
    legacy pipe-format lines for scores stored before the JSON format."""
    if parsed["items"]:
        for label, score, maximum, reason in parsed["items"]:
            line = f"**{label}**: {score}/{maximum}"
            if reason:
                line += f" — {reason}"
            st.markdown(line)
    elif parsed["legacy_lines"]:
        st.markdown("\n".join(f"- {p}" for p in parsed["legacy_lines"]))


def _render_scored_metric(label: str, raw_breakdown, raw_score, reason: str) -> None:
    """Metric + breakdown + reason for one scoring mode."""
    parsed   = parse_score_breakdown(raw_breakdown or "", fallback_score=raw_score)
    computed = parsed["computed_score"]
    st.metric(label=label, value=f"{_score_color(computed)} {computed} / 100")
    _render_breakdown(parsed)
    if reason:
        st.caption(reason)


def _rescore_structured(sel: pd.Series, job_id: str, cand: dict) -> None:
    """Re-run structured scoring for one job, extracting its JD JSON first if
    it doesn't have any yet."""
    with st.spinner("Rescoring (structured)…"):
        requirements = parse_jd_extracted(sel.get("jd_extracted"))
        if requirements is None:
            try:
                req = extract_job_requirements(sel.get("description", ""), sel.get("company"))
                requirements = req.model_dump()
                update_job_fields(job_id, {"jd_extracted": json.dumps(requirements)})
            except Exception as e:
                st.error(f"Structured extraction failed: {e}")
                return
        resume_profile = load_resume_profile(cand, log_fn=lambda msg: None)
        if resume_profile is None:
            st.error("Could not load a structured resume profile.")
            return
        for result in score_jobs_structured(resume_profile, [{
            "id": job_id, "requirements": requirements,
            "is_remote": bool(sel.get("is_remote")),
            "title": sel.get("title", ""), "company": sel.get("company", ""),
        }]):
            if result:
                update_structured_scores(result)
    st.toast("Rescored (structured)!")
    st.rerun()


def _rescore_raw(sel: pd.Series, job_id: str, summary: str) -> None:
    with st.spinner("Rescoring…"):
        for result in score_jobs(summary, [{
            "id": job_id,
            "title": sel.get("title", ""),
            "company": sel.get("company", ""),
            "description": sel.get("description", ""),
            "is_remote": sel.get("is_remote", False),
        }]):
            if result:
                update_scores(result)
    st.toast("Rescored!")
    st.rerun()


def _handle_rescore(sel: pd.Series, job_id: str) -> None:
    cand    = get_candidate()
    summary = cand.get("summary", "")
    if not summary:
        st.warning("No candidate summary — generate one in the Profile tab first.")
    elif not (sel.get("description") or "").strip():
        st.warning("No description available for this job.")
    elif scoring_mode() == "structured":
        _rescore_structured(sel, job_id, cand)
    else:
        _rescore_raw(sel, job_id, summary)


def _render_score_display(sel: pd.Series) -> None:
    """Show a job's structured score (the primary path) with its raw score kept
    alongside for comparison, plus a Rescore button."""
    score_val            = sel.get("score")
    structured_score_val = sel.get("structured_score")
    has_structured       = pd.notna(structured_score_val)
    score_col, btn_col   = st.columns([3, 1])

    # Structured score leads, since it's the primary scoring path going forward.
    if has_structured:
        with score_col:
            _render_scored_metric(
                "Structured Match Score",
                sel.get("structured_score_breakdown"),
                structured_score_val,
                sel.get("structured_score_reason", "") or "",
            )
    else:
        score_col.caption("Not yet scored (structured).")

    # Raw score collapsed by default once a structured score exists to lead
    # with; expanded when it's the only score there is.
    with st.expander("Raw score details", expanded=not has_structured):
        if pd.notna(score_val):
            _render_scored_metric(
                "Match Score",
                sel.get("score_breakdown"),
                score_val,
                sel.get("score_reason", "") or "",
            )
        else:
            st.caption("Not yet scored.")

    job_id = sel.get("id", "")
    if btn_col.button("↺ Rescore", key=f"rescore_{job_id}", use_container_width=True):
        _handle_rescore(sel, job_id)
