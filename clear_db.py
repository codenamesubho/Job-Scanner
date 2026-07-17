"""Utility script to clear jobs and/or referrals from the database."""
import argparse
import sqlite3

from scanner.database import DB_PATH


def clear_table(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"DELETE FROM {table}")
    return cur.rowcount


def main():
    parser = argparse.ArgumentParser(description="Clear database tables")
    parser.add_argument("--jobs", action="store_true", help="Clear jobs table")
    parser.add_argument("--referrals", action="store_true", help="Clear referrals table")
    parser.add_argument("--all", action="store_true", help="Clear both tables")
    args = parser.parse_args()

    if not (args.jobs or args.referrals or args.all):
        parser.print_help()
        return

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        if args.jobs or args.all:
            n = clear_table(conn, "jobs")
            print(f"Cleared {n} rows from jobs")
        if args.referrals or args.all:
            try:
                n = clear_table(conn, "referrals")
                print(f"Cleared {n} rows from referrals")
            except sqlite3.OperationalError:
                print("referrals table does not exist — skipped")
        conn.commit()


if __name__ == "__main__":
    main()
