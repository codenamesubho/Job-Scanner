"""Utility script to score every job that needs it, without touching JD
extraction or description backfill.

Under SCORING_MODE=raw (default), scores every job with no score yet and a
usable description. Under SCORING_MODE=structured, first extracts any
missing structured JD data, then scores every job missing a structured_score
— see scanner.scoring.score_unscored_jobs for the full branching logic.

Split out from backfill_jd_extraction.py so extraction and scoring can be
run/scheduled independently.

Usage:
    python backfill_score_jobs.py            # score every job that needs it
    python backfill_score_jobs.py --top 50   # only score the top 50 jobs by score
"""
import argparse

from scanner import score_unscored_jobs


def main():
    parser = argparse.ArgumentParser(description="Score jobs that need scoring")
    parser.add_argument("--top", type=int, default=None,
                         help="Only score the top N jobs by score (default: all eligible jobs)")
    args = parser.parse_args()

    scored = score_unscored_jobs(log_fn=print, limit=args.top)
    print(f"\nScored {scored} job(s).")


if __name__ == "__main__":
    main()
