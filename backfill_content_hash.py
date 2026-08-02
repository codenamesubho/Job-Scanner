"""Backfill jobs.content_hash for jobs saved before this feature existed.
New jobs get a content_hash automatically at save time (scanner.database.
save_jobs); this is only needed for pre-existing rows.

Usage:
    python backfill_content_hash.py             # backfill missing hashes only
    python backfill_content_hash.py --force      # recompute every row's hash
"""
import argparse

from scanner import backfill_content_hashes


def main():
    parser = argparse.ArgumentParser(description="Backfill jobs.content_hash for existing jobs")
    parser.add_argument("--force", action="store_true",
                         help="Recompute content_hash for every job, even ones that already have one")
    args = parser.parse_args()

    updated = backfill_content_hashes(force=args.force)
    print(f"Updated content_hash for {updated} job(s).")


if __name__ == "__main__":
    main()
