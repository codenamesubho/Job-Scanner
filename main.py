import argparse
import os
from datetime import datetime

from cli_common import add_search_args
from scanner import search_jobs, display_jobs, filter_by_exclude, filter_by_remote_flag, save_jobs
from scanner.config import SEARCH_KEYWORDS, SEARCH_LOCATION, RESULTS_WANTED, HOURS_OLD


def parse_args():
    parser = argparse.ArgumentParser(description="LinkedIn Job Scanner")
    # main.py is the one script whose defaults come from .env rather than the saved
    # search-criteria profile — that is what the SEARCH_* constants are for.
    add_search_args(parser, {"keywords": SEARCH_KEYWORDS, "location": SEARCH_LOCATION,
                              "results": RESULTS_WANTED, "hours": HOURS_OLD})
    parser.add_argument("--remote-only", action="store_true", help="Show only remote jobs (is_remote flag)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Title keywords to exclude")
    parser.add_argument("--save", action="store_true", help="Save results to output/ as CSV")
    parser.add_argument("--db", action="store_true", help="Persist results to SQLite database")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Searching LinkedIn for: '{args.keywords}' in '{args.location}'")
    print(f"Fetching up to {args.results} postings from the last {args.hours}h...\n")

    jobs = search_jobs(
        keywords=args.keywords,
        location=args.location,
        results_wanted=args.results,
        hours_old=args.hours,
    )

    if args.remote_only:
        jobs = filter_by_remote_flag(jobs)

    if args.exclude:
        jobs = filter_by_exclude(jobs, args.exclude)

    display_jobs(jobs)

    if args.db and not jobs.empty:
        new_count = save_jobs(jobs)
        print(f"\nSaved to database — {new_count} new job(s) added.")

    if args.save and not jobs.empty:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"output/jobs_{timestamp}.csv"
        jobs.to_csv(path, index=False)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
