import streamlit as st

from scanner import get_criteria

from .constants import HOURS_OPTIONS


def seed_sidebar_defaults() -> None:
    """Must be called from app.py before render_sidebar() — writing to a
    widget's session_state key after the widget is created raises
    StreamlitAPIException. ui/profile_tab.py's _render_profile_form() (Load/
    Save Profile) sets st.session_state["_profile_load"] and calls
    st.rerun(); on the next run this function fires first so the widget
    keys are set before render_sidebar() instantiates the sidebar widgets.
    """
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
