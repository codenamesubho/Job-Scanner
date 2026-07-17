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

import pandas as pd

from scanner import (
    search_jobs, save_jobs, get_criteria,
    linkedin_playwright_search, naukri_search, jsearch_search_jobs,
    get_company_boards, ATS_FETCHERS,
    score_unscored_jobs,
)
from scanner.filters import filter_by_keywords
from scanner.linkedin_playwright import SESSION_FILE as LINKEDIN_SESSION_FILE
from scanner.naukri_playwright import SESSION_FILE as NAUKRI_SESSION_FILE


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _split_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


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


def _run_keyword_scan(source_name: str, scan_fn, keywords: str, location: str,
                       results: int, hours: int, **kwargs) -> tuple[int, int]:
    """Run scan_fn once per comma-separated keyword, concatenate, save. One bad
    keyword doesn't abort the source — same tolerance as app.py's scan loops."""
    kw_list = _split_keywords(keywords)
    all_dfs: list[pd.DataFrame] = []
    for i, kw in enumerate(kw_list, start=1):
        _log(f"  [{source_name}] [{i}/{len(kw_list)}] Searching '{kw}' in '{location}'…")
        try:
            df = scan_fn(kw, location, results_wanted=results, hours_old=hours, **kwargs)
            if not df.empty:
                all_dfs.append(df)
                _log(f"  [{source_name}] [{i}/{len(kw_list)}] '{kw}': {len(df)} job(s) found")
            else:
                _log(f"  [{source_name}] [{i}/{len(kw_list)}] '{kw}': no jobs found")
        except Exception as e:
            _log(f"  [{source_name}] [{i}/{len(kw_list)}] '{kw}' FAILED: {e}")

    if not all_dfs:
        _log(f"  [{source_name}] No jobs found for any keyword.")
        return 0, 0

    combined = pd.concat(all_dfs, ignore_index=True)
    new_count = save_jobs(combined)
    _log(f"  [{source_name}] Done — {len(combined)} found, {new_count} new.")
    return len(combined), new_count


def _run_company_boards(keywords: str, location: str, results: int, hours: int) -> tuple[int, int]:
    boards = get_company_boards()
    if not boards:
        _log("  [Company Boards] Skipped — no company boards saved (add some in the Profile tab).")
        return 0, 0

    all_dfs: list[pd.DataFrame] = []
    for i, board in enumerate(boards, start=1):
        fetch_fn = ATS_FETCHERS.get(board["ats"])
        if not fetch_fn:
            _log(f"  [Company Boards] [{i}/{len(boards)}] {board['name']}: unknown ATS '{board['ats']}', skipped")
            continue
        _log(f"  [Company Boards] [{i}/{len(boards)}] Scanning {board['name']} ({board['ats']})…")
        try:
            df = fetch_fn(board["token"], board["name"])
            if not df.empty:
                all_dfs.append(df)
                _log(f"  [Company Boards] [{i}/{len(boards)}] {board['name']}: {len(df)} job(s) found")
            else:
                _log(f"  [Company Boards] [{i}/{len(boards)}] {board['name']}: no jobs found")
        except Exception as e:
            _log(f"  [Company Boards] [{i}/{len(boards)}] {board['name']} FAILED: {e}")

    if not all_dfs:
        _log("  [Company Boards] No jobs found across saved company boards.")
        return 0, 0

    combined = pd.concat(all_dfs, ignore_index=True)
    kw_list = _split_keywords(keywords)
    if kw_list:
        before = len(combined)
        combined = filter_by_keywords(combined, kw_list)
        _log(f"  [Company Boards] Keyword filter: {before} → {len(combined)} job(s)")

    new_count = save_jobs(combined)
    _log(f"  [Company Boards] Done — {len(combined)} found, {new_count} new.")
    return len(combined), new_count


def main() -> None:
    args = parse_args()
    _log(f"Starting daily scan — keywords='{args.keywords}' location='{args.location}' "
         f"results={args.results} hours={args.hours}")

    total_found = 0
    total_new = 0

    _log("[LinkedIn (jobspy)]")
    found, new = _run_keyword_scan("LinkedIn (jobspy)", search_jobs, args.keywords,
                                    args.location, args.results, args.hours)
    total_found += found
    total_new += new

    _log("[LinkedIn (login)]")
    if LINKEDIN_SESSION_FILE.exists():
        found, new = _run_keyword_scan("LinkedIn (login)", linkedin_playwright_search,
                                        args.keywords, args.location, args.results, args.hours)
        total_found += found
        total_new += new
    else:
        _log("  [LinkedIn (login)] Skipped — no saved session. Log in once via the Streamlit app first.")

    _log("[Naukri]")
    if NAUKRI_SESSION_FILE.exists():
        found, new = _run_keyword_scan("Naukri", naukri_search, args.keywords,
                                        args.location, args.results, args.hours)
        total_found += found
        total_new += new
    else:
        _log("  [Naukri] Skipped — no saved session. Log in once via the Streamlit app first.")

    _log("[Company Boards]")
    found, new = _run_company_boards(args.keywords, args.location, args.results, args.hours)
    total_found += found
    total_new += new

    _log("[JSearch]")
    if os.getenv("JSEARCH_API_KEY"):
        found, new = _run_keyword_scan("JSearch", jsearch_search_jobs, args.keywords,
                                        args.location, args.results, args.hours)
        total_found += found
        total_new += new
    else:
        _log("  [JSearch] Skipped — set JSEARCH_API_KEY in .env.")

    _log(f"Scan complete — {total_found} found, {total_new} new across all sources.")

    scored = score_unscored_jobs(log_fn=_log)

    _log(f"Done. {total_new} new job(s) saved, {scored} scored.")


if __name__ == "__main__":
    main()
