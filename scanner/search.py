"""The scan kernel: run a job search across keywords or company boards, and save.

This is the single implementation of "search once per keyword, tolerate a failing
keyword, concatenate, save, count what was new". It previously existed twice —
once in `ui/scan_runners._do_scan_core` for the Streamlit app and once in
`cron_scan._run_keyword_scan` for the nightly cron job — which differed only in
their log formatting. Bug fixes to one silently missed the other.

Nothing here imports Streamlit, and nothing here imports the `scanner` package
facade (which imports this module). Progress and logging are reported through
plain callables so the same code runs on Streamlit's main thread, inside a
background worker thread, or from a bare CLI script.
"""
from dataclasses import dataclass

import pandas as pd

from .ats_registry import ATS_FETCHERS
from .database import save_jobs
from .filters import filter_by_keywords
from .profile import get_company_boards


@dataclass(frozen=True)
class SearchCriteria:
    """What to search for. Frozen because a scan should never mutate its own inputs."""

    keywords: str
    location: str = ""
    results: int = 25
    hours: int = 72

    def keyword_list(self) -> list[str]:
        """Split the comma-separated `keywords` field into individual searches.

        Blank segments are dropped, so trailing commas and double commas are
        harmless: `"a,, b ,"` -> `["a", "b"]`.
        """
        return [k.strip() for k in self.keywords.split(",") if k.strip()]


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one source's scan: how many jobs it saw, how many were new."""

    found: int = 0
    new: int = 0

    def __add__(self, other: "ScanResult") -> "ScanResult":
        """Lets callers total up several sources with `sum(results, ScanResult())`."""
        return ScanResult(self.found + other.found, self.new + other.new)


def _noop_progress(frac: float, text: str) -> None:
    """Default progress sink for callers that have no progress bar (e.g. cron)."""


def _save_and_report(all_dfs, log_fn, empty_message: str) -> ScanResult:
    """Concatenate a source's frames, save them, and log the found/new tally."""
    if not all_dfs:
        log_fn(empty_message)
        return ScanResult()

    combined = pd.concat(all_dfs, ignore_index=True)
    log_fn(f"Saving {len(combined)} job(s)…")
    new_count = save_jobs(combined)
    log_fn(f"Done — {len(combined)} found, {new_count} new, "
           f"{len(combined) - new_count} already known.")
    return ScanResult(len(combined), new_count)


def run_keyword_scan(scan_fn, criteria: SearchCriteria, log_fn,
                      progress_fn=None, **kwargs) -> ScanResult:
    """Call `scan_fn` once per keyword in `criteria`, then save everything found.

    `scan_fn(keyword, location, results_wanted=..., hours_old=..., **kwargs)` is the
    uniform signature every source adapter in `scanner` exposes.

    A keyword that raises is logged and skipped rather than aborting the source —
    one bad search term should not lose the results of the others.
    """
    progress_fn = progress_fn or _noop_progress
    kw_list = criteria.keyword_list()
    if not kw_list:
        log_fn("No keywords given.")
        return ScanResult()

    total = len(kw_list)
    all_dfs: list[pd.DataFrame] = []
    for i, kw in enumerate(kw_list, start=1):
        progress_fn((i - 1) / total, f"Searching '{kw}' ({i}/{total})…")
        log_fn(f"[{i}/{total}] Searching '{kw}' in '{criteria.location}'…")
        try:
            df = scan_fn(kw, criteria.location, results_wanted=criteria.results,
                          hours_old=criteria.hours, **kwargs)
            if not df.empty:
                all_dfs.append(df)
                log_fn(f"[{i}/{total}] '{kw}': {len(df)} job(s) found")
            else:
                log_fn(f"[{i}/{total}] '{kw}': no jobs found")
        except Exception as e:
            log_fn(f"[{i}/{total}] '{kw}' FAILED: {e}")
        progress_fn(i / total, f"Searched '{kw}' ({i}/{total})")

    return _save_and_report(all_dfs, log_fn, "No jobs found for any keyword.")


def run_company_board_scan(criteria: SearchCriteria, log_fn, progress_fn=None) -> ScanResult:
    """Scan every saved company ATS board, then keyword-filter what came back.

    Unlike `run_keyword_scan`, the boards are fetched whole (their APIs have no
    keyword parameter) and filtered afterwards, which is also why the "Posted
    within" window does not apply to this source.
    """
    progress_fn = progress_fn or _noop_progress
    boards = get_company_boards()
    if not boards:
        log_fn("Skipped — no company boards saved (add some in the Profile tab).")
        return ScanResult()

    total = len(boards)
    all_dfs: list[pd.DataFrame] = []
    for i, board in enumerate(boards, start=1):
        name = board["name"]
        progress_fn((i - 1) / total, f"Scanning {name} ({i}/{total})…")
        fetch_fn = ATS_FETCHERS.get(board["ats"])
        if not fetch_fn:
            log_fn(f"[{i}/{total}] {name}: unknown ATS '{board['ats']}', skipped")
            progress_fn(i / total, f"Skipped {name} ({i}/{total})")
            continue
        log_fn(f"[{i}/{total}] Scanning {name} ({board['ats']})…")
        try:
            df = fetch_fn(board["token"], name)
            if not df.empty:
                all_dfs.append(df)
                log_fn(f"[{i}/{total}] {name}: {len(df)} job(s) found")
            else:
                log_fn(f"[{i}/{total}] {name}: no jobs found")
        except Exception as e:
            log_fn(f"[{i}/{total}] {name} FAILED: {e}")
        progress_fn(i / total, f"Scanned {name} ({i}/{total})")

    if not all_dfs:
        log_fn("No jobs found across saved company boards.")
        return ScanResult()

    combined = pd.concat(all_dfs, ignore_index=True)
    kw_list = criteria.keyword_list()
    if kw_list:
        before = len(combined)
        combined = filter_by_keywords(combined, kw_list)
        log_fn(f"Keyword filter: {before} → {len(combined)} job(s)")

    return _save_and_report([combined], log_fn, "No jobs found across saved company boards.")


def prefixed_logger(log_fn, prefix: str):
    """Wrap `log_fn` so every line is tagged with `prefix`.

    The Streamlit UI gives each source its own log box, so it needs no prefix; the
    cron job interleaves every source into one stdout stream, so it does. Keeping
    that difference in the logger — rather than as a parameter threaded through the
    scan functions — is what let the two implementations collapse into one.
    """
    def _log(msg: str) -> None:
        log_fn(f"  [{prefix}] {msg}")

    return _log
