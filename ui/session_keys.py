"""Names of the `st.session_state` entries the UI uses.

These were bare string literals spread across five modules — a typo produced a
silently-missing value rather than an error, and there was no one place to see
what the UI actually keeps in session state.

Widget-backed keys (SB_*) are special: Streamlit binds them to a widget via
`key=`, and writing to one *after* its widget is created raises
StreamlitAPIException. That is why `seed_sidebar_defaults()` must run before
`render_sidebar()`.
"""

# Sidebar widget bindings — see the note above before writing to these.
SB_KEYWORDS = "sb_keywords"
SB_LOCATION = "sb_location"
SB_RESULTS  = "sb_results"
SB_HOURS    = "sb_hours"

# Pending sidebar values staged by the Profile tab's "Load" button, applied on
# the next run by seed_sidebar_defaults() before the widgets exist.
PROFILE_LOAD = "_profile_load"

# The in-flight scoring job (a ui.background.BackgroundJob), present only while
# scoring is running — its presence is what turns the Score button into Cancel.
SCORE_JOB = "_score_job"

# In-flight Apply runs (a dict of job_id -> ui.background.BackgroundJob) — one
# entry per job currently being applied to via Autofill-Job-Application, keyed
# so applying to job A doesn't disturb an in-flight run for job B.
APPLY_JOBS = "_apply_jobs"

# Profile tab editing state.
EDITING_PROFILE = "_editing_profile"
LAST_RESUME_KEY = "_last_resume_key"
