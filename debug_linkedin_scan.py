"""Debug harness for the scanner/linkedin_playwright/ package.

Runs search_jobs() directly, once per comma-separated role in --keywords, and
prints its per-page log lines plus a result summary — no DB writes, no scoring.
Use it to see exactly which pages fail vs. return few cards when the LinkedIn
(login) scan under-delivers.

This is the primary way to validate scraper changes. The test suite deliberately
covers only selector-independent logic (see tests/test_linkedin_pure_logic.py),
because LinkedIn's obfuscated class names can only be verified against the live
site — so any change under scanner/linkedin_playwright/ should be checked with a
run here, in BOTH --keyword-mode values, not by the suite alone.

Requires an existing saved session (log in once via the Streamlit app).
Set SCAN_DEBUG_HEADFUL=1 in your .env to watch the browser windows live.
"""
import argparse
import sys

from scanner.linkedin_playwright import SESSION_FILE, search_jobs
from scanner.search import SearchCriteria


def parse_args():
    parser = argparse.ArgumentParser(description="Debug the LinkedIn Playwright scraper")
    parser.add_argument("--keywords", required=True,
                         help="Job search keywords, comma-separated for multiple roles "
                              "(e.g. 'backend engineer, platform engineer')")
    parser.add_argument("--location", required=True, help="Job location")
    parser.add_argument("--results", type=int, default=25, help="Requested result count per role")
    parser.add_argument("--hours", type=int, default=72, help="Max age of postings in hours")
    parser.add_argument("--pages", type=int, default=None,
                         help="Exact number of search pages to scrape per role, overriding the "
                              "usual results-derived page count (bypasses the 3-4 page clamp)")
    parser.add_argument("--scroll-strategy", choices=["incremental", "all-first"], default="incremental",
                         help="incremental (default): scroll a step, extract, repeat. "
                              "all-first: scroll to the bottom of the list first, then extract once.")
    parser.add_argument("--keyword-mode", choices=["structured", "natural-language"], default="structured",
                         help="structured (default): separate keywords/location/hours query params. "
                              "natural-language: one free-text string like 'backend engineer in "
                              "Remote in last 72 hours', omitting the location/f_TPR params entirely.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not SESSION_FILE.exists():
        print(f"No saved LinkedIn session at {SESSION_FILE} — log in once via the Streamlit app first.")
        sys.exit(1)

    roles = SearchCriteria(args.keywords, args.location).keyword_list()
    seen_ids = set()

    for i, role in enumerate(roles, start=1):
        print(f"\n=== [{i}/{len(roles)}] '{role}' ===")
        jobs = search_jobs(
            keywords=role,
            location=args.location,
            results_wanted=args.results,
            hours_old=args.hours,
            num_pages=args.pages,
            scroll_strategy=args.scroll_strategy.replace("-", "_"),
            keyword_mode=args.keyword_mode.replace("-", "_"),
        )

        new_rows = jobs[~jobs["id"].isin(seen_ids)] if not jobs.empty else jobs
        if not new_rows.empty:
            seen_ids.update(new_rows["id"])

        print(f"\n'{role}': {len(jobs)} unique job(s) returned "
              f"({len(new_rows)} not already seen for an earlier role):")
        for _, row in new_rows.iterrows():
            print(f"  {row.get('title', '?')} | {row.get('company', '?')} | {row.get('id', '?')}")


if __name__ == "__main__":
    main()
