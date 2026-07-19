import streamlit as st

from ui.jobs_tab import render_jobs_tab
from ui.profile_tab import render_profile_tab
from ui.sidebar import render_sidebar, seed_sidebar_defaults

st.set_page_config(page_title="Job Scanner", page_icon="💼", layout="wide")

# Must run before render_sidebar() creates the sidebar widgets — see
# seed_sidebar_defaults()'s docstring in ui/sidebar.py.
seed_sidebar_defaults()

(keywords, location, results, hours, scan_all_clicked,
 scan_clicked, li_pw_clicked, naukri_clicked, boards_clicked, jsearch_clicked) = render_sidebar()

tab_jobs, tab_profile = st.tabs(["📋 Jobs", "👤 Profile"])

with tab_jobs:
    render_jobs_tab(keywords, location, results, hours, scan_all_clicked, scan_clicked,
                     li_pw_clicked, naukri_clicked, boards_clicked, jsearch_clicked)

with tab_profile:
    render_profile_tab()
