"""Standalone CLI script: run a full job search across every enabled source,
then score newly-found jobs. Designed to run headless via cron every morning.

This script does NOT import app.py (a Streamlit script — importing it outside
a Streamlit runtime would fail/misbehave) and does not modify it. It calls
directly into the `scanner` package, same as main.py.

Usage:
    python cron_scan.py                        # uses saved search criteria (Profile tab)
    python cron_scan.py --keywords "backend engineer,staff engineer" --location "USA"
    python cron_scan.py --results 25 --hours 72

Cron entry example (adjust paths):
    0 7 * * * cd /Users/subho/code/project/Job_Scanner && venv/bin/python cron_scan.py >> logs/cron_scan.log 2>&1

LinkedIn (login) and Naukri only run if a session already exists (saved via
the Streamlit app's login flow) — this script never attempts an interactive
login, since a cron job can't complete a visible browser / credentials /
CAPTCHA flow unattended.
"""

import argparse
import os
from datetime import datetime

from scanner import (
    ScanResult, SearchCriteria, get_criteria, jsearch_search_jobs,
    linkedin_playwright_search, naukri_search, prefixed_logger,
    run_company_board_scan, run_keyword_scan, score_unscored_jobs, search_jobs,
)
from scanner.linkedin_playwright import SESSION_FILE as LINKEDIN_SESSION_FILE
from scanner.naukri_playwright import SESSION_FILE as NAUKRI_SESSION_FILE


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args():
    # get_criteria() always returns a fully-populated dict — either the saved
    # search_criteria row, or profile._CRITERIA_DEFAULTS if none was ever saved.
    crit = get_criteria()
    parser = argparse.ArgumentParser(description="Daily job scan across every enabled source")
    parser.add_argument("--keywords", default=crit["keywords"],
                         help="Comma-separated keywords (default: saved search criteria)")
    parser.add_argument("--location", default=crit["location"],
                         help="Job location (default: saved search criteria)")
    parser.add_argument("--results", type=int, default=crit["results"],
                         help="Max results per keyword (default: saved search criteria)")
    parser.add_argument("--hours", type=int, default=crit["hours"],
                         help="Max age of postings in hours (default: saved search criteria)")
    return parser.parse_args()


def _scan(source_name: str, scan_fn, criteria: SearchCriteria) -> ScanResult:
    """Run one keyword-based source, tagging every log line with its name.

    The cron job interleaves all sources into a single stdout stream, so it needs
    the prefix; the Streamlit UI gives each source its own log box and does not.
    That is the only difference between the two callers of run_keyword_scan().
    """
    _log(f"[{source_name}]")
    return run_keyword_scan(scan_fn, criteria, prefixed_logger(_log, source_name))


def main() -> None:
    args = parse_args()
    criteria = SearchCriteria(args.keywords, args.location, args.results, args.hours)
    _log(f"Starting daily scan — keywords='{criteria.keywords}' location='{criteria.location}' "
         f"results={criteria.results} hours={criteria.hours}")

    total = ScanResult()
    total += _scan("LinkedIn (jobspy)", search_jobs, criteria)

    # Login-based sources only run off an already-saved session: a cron job can't
    # complete an interactive visible-browser/CAPTCHA login unattended.
    if LINKEDIN_SESSION_FILE.exists():
        total += _scan("LinkedIn (login)", linkedin_playwright_search, criteria)
    else:
        _log("[LinkedIn (login)]")
        _log("  [LinkedIn (login)] Skipped — no saved session. Log in once via the Streamlit app first.")

    if NAUKRI_SESSION_FILE.exists():
        total += _scan("Naukri", naukri_search, criteria)
    else:
        _log("[Naukri]")
        _log("  [Naukri] Skipped — no saved session. Log in once via the Streamlit app first.")

    _log("[Company Boards]")
    total += run_company_board_scan(criteria, prefixed_logger(_log, "Company Boards"))

    if os.getenv("JSEARCH_API_KEY"):
        total += _scan("JSearch", jsearch_search_jobs, criteria)
    else:
        _log("[JSearch]")
        _log("  [JSearch] Skipped — set JSEARCH_API_KEY in .env.")

    _log(f"Scan complete — {total.found} found, {total.new} new across all sources.")

    scored = score_unscored_jobs(log_fn=_log)

    _log(f"Done. {total.new} new job(s) saved, {scored} scored.")


if __name__ == "__main__":
    main()
