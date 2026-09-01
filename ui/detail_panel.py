import pandas as pd
import streamlit as st

from scanner import update_status

from .apply_button import _render_apply_button
from .constants import STATUSES
from .referrals import _render_referral_section
from .scoring import _render_score_display


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


def _render_detail_panel(sel: pd.Series, job_id: str) -> None:
    remote_badge = "  🌐 Remote" if sel.get("is_remote") else ""
    st.markdown(
        f"### {sel.get('title', '')}  \n"
        f"**{sel.get('company', '')}** · {sel.get('location', '')}{remote_badge}"
    )
    job_url    = sel.get("job_url", "") or ""
    direct_url = sel.get("job_url_direct", "") or ""
    apply_url  = direct_url or job_url

    apply_col, listing_col, shortlist_col, applied_col = st.columns([2, 2, 1, 1])

    if job_url:
        listing_col.link_button("View Listing ↗", job_url, use_container_width=True)

    if apply_url:
        with apply_col:
            _render_apply_button(job_id, apply_url)

    if shortlist_col.button("⭐ Shortlist", key=f"shortlist_{job_id}", use_container_width=True):
        update_status(job_id, "shortlisted")
        st.toast("Status → 'shortlisted'.")
        st.rerun()

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
