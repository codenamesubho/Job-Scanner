"""Utility script to backfill missing job descriptions and scores.

Some LinkedIn jobs get saved without a description when jobspy's per-job
fetch fails transiently during a scan (rate limiting, network hiccup, etc).
Jobs without a description can never be scored (scanner/llm.py requires
description text), so this script re-fetches those directly from LinkedIn's
public job page and, once recovered, scores them.

Usage:
    python backfill_descriptions.py            # backfill descriptions only
    python backfill_descriptions.py --score    # also score newly-recovered jobs
"""
import argparse

from scanner import backfill_missing_descriptions, score_unscored_jobs


def main():
    parser = argparse.ArgumentParser(description="Backfill missing job descriptions")
    parser.add_argument("--score", action="store_true",
                         help="Also score jobs that had their description recovered")
    args = parser.parse_args()

    fixed = backfill_missing_descriptions(log_fn=print)
    print(f"\nRecovered {fixed} description(s).")

    if args.score and fixed:
        scored = score_unscored_jobs(log_fn=print)
        print(f"\nScored {scored} job(s).")


if __name__ == "__main__":
    main()
