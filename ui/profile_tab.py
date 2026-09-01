import json

import streamlit as st

from scanner import (
    delete_company_board, delete_criteria, extract_resume_profile, extract_text,
    generate_summary, get_all_criteria, get_candidate, get_company_boards,
    get_latest_resume, save_candidate, save_company_board, save_criteria,
    save_resume,
)
from scanner.autofill_bridge import build_context_markdown, context_path

from .constants import HOURS_OPTIONS


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

    ctx_path = context_path()
    st.caption(
        f"Apply flow profile: `{ctx_path}` "
        + ("(exists — hand-edited copies are kept)" if ctx_path.exists() else "(created on first Apply click)")
    )
    if st.button("↻ Regenerate apply profile from candidate data", key="regen_autofill_context"):
        build_context_markdown(get_candidate(), force=True)
        st.success(f"Regenerated {ctx_path} from the candidate details above.")

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
        # Consumed by ui/sidebar.py's seed_sidebar_defaults(), which app.py
        # calls before render_sidebar() creates the sidebar widgets — writing
        # to a widget's session_state key after the widget exists raises
        # StreamlitAPIException, so this can't be applied directly here.
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
