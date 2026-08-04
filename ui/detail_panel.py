import pandas as pd
import streamlit as st

from scanner import apply_and_prefill, update_status

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
