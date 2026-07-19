import pandas as pd
import streamlit as st

from scanner import (
    delete_referral, draft_referral_message, find_referral_contacts,
    get_candidate, get_referrals, save_referral, send_linkedin_message,
)


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
