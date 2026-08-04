"""Typed values passed between the sidebar and the tabs.

`render_sidebar()` used to return a 10-element positional tuple that `app.py`
unpacked and forwarded, unchanged, into `render_jobs_tab(...)` — six of those
elements were consecutive booleans, so a mis-ordering would type-check fine and
silently run the wrong scan.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScanRequest:
    """Which scan buttons the user clicked on this script run.

    Streamlit re-runs the whole script on every interaction and a button reads
    True only on the run where it was clicked, so this is a per-run snapshot,
    not persistent state.
    """

    scan_all: bool = False
    jobspy: bool = False
    linkedin_login: bool = False
    naukri: bool = False
    company_boards: bool = False
    jsearch: bool = False

    @property
    def any_clicked(self) -> bool:
        return any((self.scan_all, self.jobspy, self.linkedin_login,
                    self.naukri, self.company_boards, self.jsearch))
