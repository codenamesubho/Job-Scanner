"""Utility script to backfill structured JD extraction (jobs.jd_extracted)
for jobs that already exist in the database from before this feature
existed.

Only relevant when SCORING_MODE=structured — under SCORING_MODE=raw (the
default), extraction never runs and this script is a no-op by design
(mirrors extract_missing_job_requirements()'s own guard).

Extraction only — run backfill_score_jobs.py afterward to score the newly-extracted
jobs (kept as a separate script so extraction and scoring can be run/
scheduled independently).

Usage:
    python backfill_jd_extraction.py             # extract only
    python backfill_jd_extraction.py --top 50    # only the top 50 jobs by score
    python backfill_jd_extraction.py --force     # re-extract every job, even ones already extracted
                                                  # (e.g. after adding a new field to JobRequirements)
"""
import argparse

from scanner import extract_missing_job_requirements
from scanner.llm import scoring_mode


def main():
    parser = argparse.ArgumentParser(description="Backfill structured JD extraction for existing jobs")
    parser.add_argument("--top", type=int, default=None,
                         help="Only extract the top N jobs by score (default: all eligible jobs)")
    parser.add_argument("--force", action="store_true",
                         help="Re-extract every scoreable job, even ones that already have "
                              "jd_extracted populated (default: only jobs missing it)")
    args = parser.parse_args()

    if scoring_mode() != "structured":
        print("SCORING_MODE is not 'structured' — set it in .env first; nothing to backfill.")
        return

    extracted = extract_missing_job_requirements(log_fn=print, limit=args.top, force=args.force)
    print(f"\nExtracted {extracted} job(s).")


if __name__ == "__main__":
    main()
