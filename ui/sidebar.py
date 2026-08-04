import streamlit as st

from scanner import SearchCriteria, get_criteria

from .constants import HOURS_OPTIONS
from .models import ScanRequest
from .scoring import _render_score_button
from .session_keys import (
    PROFILE_LOAD, SB_HOURS, SB_KEYWORDS, SB_LOCATION, SB_RESULTS,
)


def seed_sidebar_defaults() -> None:
    """Must be called from app.py before render_sidebar() — writing to a
    widget's session_state key after the widget is created raises
    StreamlitAPIException. ui/profile_tab.py's _render_profile_form() (Load/
    Save Profile) sets st.session_state["_profile_load"] and calls
    st.rerun(); on the next run this function fires first so the widget
    keys are set before render_sidebar() instantiates the sidebar widgets.
    """
    if PROFILE_LOAD in st.session_state:
        _pl = st.session_state.pop(PROFILE_LOAD)
        st.session_state[SB_KEYWORDS] = _pl["keywords"]
        st.session_state[SB_LOCATION] = _pl["location"]
        st.session_state[SB_RESULTS]  = _pl["results"]
        st.session_state[SB_HOURS]    = _pl["hours"]

    # Seed sidebar defaults from DB once per session
    _crit = get_criteria()
    for _key, _field, _default in (
        (SB_KEYWORDS, "keywords", "software engineer"),
        (SB_LOCATION, "location", "USA"),
        (SB_RESULTS,  "results",  25),
        (SB_HOURS,    "hours",    72),
    ):
        if _key not in st.session_state:
            st.session_state[_key] = _crit.get(_field, _default)


def render_sidebar() -> tuple[SearchCriteria, ScanRequest]:
    """Draw the sidebar and return what the user asked for: the search criteria,
    and which scan buttons were clicked on this run."""
    with st.sidebar:
        st.title("💼 Job Scanner")
        st.subheader("Search Settings")

        keywords = st.text_input(
            "Keywords (comma-separated)",
            key=SB_KEYWORDS,
            help="e.g. backend engineer, python developer",
        )
        location = st.text_input("Location", key=SB_LOCATION)
        results  = st.slider("Max results per keyword", 5, 100, key=SB_RESULTS)

        hours = st.selectbox(
            "Posted within", HOURS_OPTIONS,
            format_func=lambda h: f"{h}h ({h // 24}d)",
            key=SB_HOURS,
        )

        st.markdown("**Scan All Sources**")
        scan_all_clicked = st.button(
            "🚀 Scan All (parallel)", use_container_width=True, type="primary",
            help="Runs every source below at once, each in its own thread, "
                 "with a live progress bar + log per source.",
        )

        _render_score_button()

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

    return (
        SearchCriteria(keywords, location, results, hours),
        ScanRequest(
            scan_all=scan_all_clicked,
            jobspy=scan_clicked,
            linkedin_login=li_pw_clicked,
            naukri=naukri_clicked,
            company_boards=boards_clicked,
            jsearch=jsearch_clicked,
        ),
    )
