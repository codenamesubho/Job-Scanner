"""Argument-parsing and logging helpers shared by the top-level CLI scripts.

Before this module, the --keywords/--location/--results/--hours quartet was
declared in three scripts, --top in two, and --force in two, each with its own
help text and its own defaults. They had already drifted: debug_linkedin_scan.py
hardcoded 25/72 while main.py read them from .env and cron_scan.py read them from
the saved search-criteria profile.
"""
import argparse
from datetime import datetime

from scanner.search import SearchCriteria


def add_search_args(parser: argparse.ArgumentParser, defaults: dict | None = None) -> None:
    """Add the four search-criteria flags every scanning script accepts.

    `defaults` is any mapping with keywords/location/results/hours keys — the saved
    search-criteria row (`scanner.get_criteria()`) or the `.env`-backed constants in
    `scanner.config`. Omitted keys fall back to SearchCriteria's own defaults, so a
    script can never accidentally introduce a *different* fallback than its peers.
    """
    defaults = defaults or {}
    blank = SearchCriteria(keywords="")
    parser.add_argument("--keywords", default=defaults.get("keywords", blank.keywords),
                         help="Comma-separated job search keywords — each is searched separately")
    parser.add_argument("--location", default=defaults.get("location", blank.location),
                         help="Job location (e.g. 'Remote', 'USA', 'London, UK')")
    parser.add_argument("--results", type=int, default=defaults.get("results", blank.results),
                         help="Max results to fetch per keyword")
    parser.add_argument("--hours", type=int, default=defaults.get("hours", blank.hours),
                         help="Max age of postings in hours")


def add_top_arg(parser: argparse.ArgumentParser, what: str) -> None:
    """Add --top N, the "only the best-scored N jobs" cap the backfills share."""
    parser.add_argument("--top", type=int, default=None,
                         help=f"Only {what} the top N jobs by score (default: all eligible jobs)")


def add_force_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add --force. The help text differs per script, so it is a required argument."""
    parser.add_argument("--force", action="store_true", help=help_text)


def criteria_from_args(args: argparse.Namespace) -> SearchCriteria:
    """Build the SearchCriteria the scan kernel takes from parsed CLI args."""
    return SearchCriteria(args.keywords, args.location, args.results, args.hours)


def log(msg: str) -> None:
    """Timestamped stdout logging, flushed so it interleaves correctly when a
    script's output is redirected to a file (as the cron entry does)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
