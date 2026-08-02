import hashlib
import json
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/jobs.db")

_STORE_COLS = [
    "id", "site", "job_url", "job_url_direct", "title", "company",
    "location", "date_posted", "job_type", "min_amount", "max_amount",
    "currency", "is_remote", "job_level", "job_function", "description",
    "company_industry", "company_url",
]

_CREATE_REFERRALS = """
CREATE TABLE IF NOT EXISTS referrals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    name         TEXT NOT NULL,
    title        TEXT,
    linkedin_url TEXT,
    degree       TEXT,
    photo_url    TEXT,
    message      TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
)
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    site             TEXT,
    job_url          TEXT,
    job_url_direct   TEXT,
    title            TEXT,
    company          TEXT,
    location         TEXT,
    date_posted      TEXT,
    job_type         TEXT,
    min_amount       REAL,
    max_amount       REAL,
    currency         TEXT,
    is_remote        INTEGER,
    job_level        TEXT,
    job_function     TEXT,
    description      TEXT,
    company_industry TEXT,
    company_url      TEXT,
    status           TEXT NOT NULL DEFAULT 'new',
    first_seen       TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen        TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

# One-time migration: physically remove rows that the old soft-dedup approach
# only flagged (is_duplicate=1) instead of preventing. Keeps the earliest
# first_seen (tiebreak: lowest rowid) per (title, company) pair.
_DELETE_OLD_DUPLICATES = """
DELETE FROM jobs
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY lower(trim(title)), lower(trim(company))
            ORDER BY first_seen ASC, rowid ASC
        ) AS rn
        FROM jobs
    ) WHERE rn = 1
)
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_REFERRALS)
    for ddl in (
        "ALTER TABLE jobs ADD COLUMN score INTEGER",
        "ALTER TABLE jobs ADD COLUMN score_reason TEXT",
        "ALTER TABLE jobs ADD COLUMN score_breakdown TEXT",
        "ALTER TABLE jobs ADD COLUMN jd_extracted TEXT",
        "ALTER TABLE jobs ADD COLUMN structured_score INTEGER",
        "ALTER TABLE jobs ADD COLUMN structured_score_reason TEXT",
        "ALTER TABLE jobs ADD COLUMN structured_score_breakdown TEXT",
        "ALTER TABLE jobs ADD COLUMN content_hash TEXT",
        "ALTER TABLE referrals ADD COLUMN degree TEXT",
        "ALTER TABLE referrals ADD COLUMN photo_url TEXT",
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    # One-time cleanup: duplicates are now prevented at insert time (see
    # save_jobs), so the old is_duplicate flag column and the rows it used
    # to flag (rather than remove) are no longer needed. This block only
    # runs once — after the column is dropped, `"is_duplicate" in cols` is
    # False on every later _connect() call.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "is_duplicate" in cols:
        conn.execute(_DELETE_OLD_DUPLICATES)
        try:
            conn.execute("ALTER TABLE jobs DROP COLUMN is_duplicate")
        except sqlite3.OperationalError:
            pass  # SQLite < 3.35 — column stays, just unused from here on
        conn.commit()

    return conn


def _dedup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-ID duplicates within a single batch, keeping first occurrence."""
    return df.drop_duplicates(subset=["id"], keep="first")


def _norm_key(title, company) -> str:
    return f"{(title or '').strip().lower()}|{(company or '').strip().lower()}"


_CONTENT_HASH_MIN_CHARS = 200  # a real JD runs hundreds-to-thousands of chars; this
# floor exists to keep short/boilerplate descriptions ("Apply through our careers
# page.") from colliding across UNRELATED jobs at DIFFERENT companies — this hash
# has no company/title scoping of its own, so it needs the input to be specific
# enough that two different real postings landing on the same digest is implausible.
# The cross-source duplicates this is actually meant to catch (the same JD mirrored
# on LinkedIn and a company's Greenhouse board) always have full JD text, so a high
# floor costs nothing there while meaningfully reducing false-merge risk on stubs.


def content_hash(description: str | None) -> str | None:
    """sha256 of normalized description text, truncated to 12 hex chars —
    a save-time dedup signal layered ON TOP OF (not replacing) the existing
    (title, company) key in save_jobs(), for catching the same posting
    mirrored across sources under different titles/casing. None for short/
    trivial text: not a reliable enough identity signal to dedup on (two
    unrelated jobs could share a near-empty description), so save_jobs()
    falls back to the (title, company) key alone in that case."""
    text = (description or "").strip().lower()
    if len(text) < _CONTENT_HASH_MIN_CHARS:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def save_jobs(df: pd.DataFrame) -> int:  # noqa: C901
    """Insert new jobs, skipping rows that duplicate an existing job's
    normalized (title, company) pair, OR its content_hash (sha256 of the
    description — see content_hash()), under a different id — e.g. the same
    role scraped from LinkedIn and mirrored on its Greenhouse/Lever board,
    or the same posting under a differently-cased title across sources.
    Returns the number of genuinely new rows inserted.

    Rules:
    - status and first_seen are never overwritten on same-id upserts.
    - A same-id row upserts normally (existing behavior).
    - A row whose content_hash OR (title, company) matches an existing
      DIFFERENT id is a re-sighting of that job (hash checked first):
      last_seen is bumped and job_url_direct is backfilled if the canonical
      row didn't have one yet. No new row is created.
    - The same rule applies within a single incoming batch, so combining
      multiple sources in one save_jobs() call can't create duplicates.
    """
    if df.empty or "id" not in df.columns:
        return 0

    df = _dedup_dataframe(df)

    cols = [c for c in _STORE_COLS if c in df.columns]
    subset = df[cols].copy()

    # NaN → None so sqlite3 stores NULL, not the string 'nan'
    subset = subset.where(subset.notna(), None)

    # bool → int (SQLite has no native bool)
    if "is_remote" in subset.columns:
        subset["is_remote"] = subset["is_remote"].apply(
            lambda x: int(x) if x is not None else None
        )

    # Sources (jobspy in particular) often derive is_remote by keyword-matching
    # "remote" anywhere in the description — including hybrid-schedule mentions
    # like "remote work on Fridays" — which wrongly flags jobs tied to a real
    # office location as fully remote. The location field is the more reliable
    # signal: if it names a specific place (not blank, not itself "remote"),
    # trust that over a source's own is_remote guess.
    if "is_remote" in subset.columns and "location" in subset.columns:
        def _correct_remote(row):
            loc = (row["location"] or "").strip().lower()
            if row["is_remote"] and loc and "remote" not in loc:
                return 0
            return row["is_remote"]
        subset["is_remote"] = subset.apply(_correct_remote, axis=1)

    # NULLIF/COALESCE: a re-scrape that comes back with a blank/NULL field
    # (e.g. jobspy's per-job description fetch failing transiently) must not
    # clobber a previously-populated value — keep the old value in that case.
    # content_hash is derived (computed per-row below), not sourced from the
    # incoming DataFrame like the _STORE_COLS columns — appended separately.
    upsert_cols = cols + ["content_hash"]
    update_clause = ", ".join(
        f"{c} = COALESCE(NULLIF(excluded.{c}, ''), {c})"
        for c in cols
        if c not in ("id", "status", "first_seen")
    ) + ", content_hash = COALESCE(NULLIF(excluded.content_hash, ''), content_hash)" \
        ", last_seen = datetime('now')"

    col_names = ", ".join(upsert_cols)
    placeholders = ", ".join("?" * len(upsert_cols))

    upsert_sql = f"""
        INSERT INTO jobs ({col_names}, last_seen)
        VALUES ({placeholders}, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET {update_clause}
    """

    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, title, company, job_url_direct, content_hash FROM jobs"
        ).fetchall()
        existing_ids = {row[0] for row in existing}
        # key -> (canonical_id, current job_url_direct)
        canonical = {_norm_key(row[1], row[2]): (row[0], row[3]) for row in existing}
        canonical_by_hash = {row[4]: (row[0], row[3]) for row in existing if row[4]}

        to_upsert: list[list] = []
        backfill: list[tuple] = []  # (new_job_url_direct_or_None, canonical_id)
        new_count = 0

        for _, row in subset.iterrows():
            row = row.to_dict()
            rid = row["id"]
            row_hash = content_hash(row.get("description"))

            if rid in existing_ids:
                to_upsert.append([row[c] for c in cols] + [row_hash])
                continue

            canon = canonical_by_hash.get(row_hash) if row_hash else None
            key = _norm_key(row.get("title"), row.get("company"))
            if canon is None:
                canon = canonical.get(key)
            if canon is not None:
                canon_id, canon_direct = canon
                new_direct = row.get("job_url_direct")
                backfill.append((new_direct if not canon_direct else None, canon_id))
                continue

            to_upsert.append([row[c] for c in cols] + [row_hash])
            new_count += 1
            canonical[key] = (rid, row.get("job_url_direct"))
            if row_hash:
                canonical_by_hash[row_hash] = (rid, row.get("job_url_direct"))
            existing_ids.add(rid)

        if to_upsert:
            conn.executemany(upsert_sql, to_upsert)
        for new_direct, canon_id in backfill:
            if new_direct:
                conn.execute(
                    "UPDATE jobs SET last_seen = datetime('now'), "
                    "job_url_direct = COALESCE(NULLIF(job_url_direct, ''), ?) "
                    "WHERE id = ?",
                    (new_direct, canon_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET last_seen = datetime('now') WHERE id = ?",
                    (canon_id,),
                )

    return new_count


def backfill_content_hashes(force: bool = False) -> int:
    """Populate jobs.content_hash for rows saved before this feature existed
    (new jobs get one automatically at save_jobs() time). Only updates rows
    where content_hash(description) actually returns something — a job with
    too short/no description stays NULL and keeps relying on the (title,
    company) dedup key instead. force=True recomputes every row's hash
    (e.g. after changing content_hash()'s algorithm), not just rows missing
    one. Returns the number of rows updated."""
    with _connect() as conn:
        query = "SELECT id, description FROM jobs"
        if not force:
            query += " WHERE content_hash IS NULL"
        rows = conn.execute(query).fetchall()

        updated = 0
        for job_id, description in rows:
            new_hash = content_hash(description)
            if new_hash is None:
                continue
            conn.execute("UPDATE jobs SET content_hash = ? WHERE id = ?", (new_hash, job_id))
            updated += 1

    return updated


def get_jobs(
    status: str = None,
    search: str = None,
    unscored_only: bool = False,
    missing_structured_score: bool = False,
) -> pd.DataFrame:
    """Return jobs ordered by score desc (nulls last), then newest first.

    `unscored_only` and `missing_structured_score` are independent filters
    (score IS NULL vs. structured_score IS NULL) — structured scoring must
    be able to re-score a job that already has a raw `score`, so its
    job-selection can never be based on the `score` column.
    """
    query = "SELECT * FROM jobs WHERE 1=1"
    params: list = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if search:
        query += " AND (title LIKE ? OR company LIKE ? OR location LIKE ?)"
        params.extend([f"%{search}%"] * 3)
    if unscored_only:
        query += " AND score IS NULL"
    if missing_structured_score:
        query += " AND structured_score IS NULL"

    query += " ORDER BY COALESCE(score, -1) DESC, first_seen DESC"

    with _connect() as conn:
        return pd.read_sql(query, conn, params=params)


def scoreable_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """Rows with a non-empty description — the only ones score_jobs() can use."""
    if jobs.empty or "description" not in jobs.columns:
        return jobs.iloc[0:0]
    return jobs[jobs["description"].notna() & (jobs["description"] != "")]


def parse_jd_extracted(raw: str | None) -> dict | None:
    """Parse a jobs.jd_extracted JSON string back into a dict, or None if
    missing/unparseable."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def update_status(job_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))


def get_stats() -> dict:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
    counts = dict(rows)
    return {
        "total":    sum(counts.values()),
        "new":      counts.get("new", 0),
        "saved":    counts.get("saved", 0),
        "applied":  counts.get("applied", 0),
        "rejected": counts.get("rejected", 0),
    }


def update_job_fields(job_id: str, fields: dict) -> None:
    """Backfill specific columns on one row (e.g. description recovered
    after the fact) without touching anything else. Skips None values so a
    failed re-fetch can't blank out a field that already has data."""
    fields = {k: v for k, v in fields.items() if v not in (None, "")}
    if not fields:
        return
    set_clause = ", ".join(f"{c} = ?" for c in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE jobs SET {set_clause} WHERE id = ?",
            (*fields.values(), job_id),
        )


def update_scores(scores: list[dict]) -> None:
    """Persist scores from score_jobs(). Each dict must have id, score, reason."""
    with _connect() as conn:
        conn.executemany(
            "UPDATE jobs SET score = ?, score_reason = ?, score_breakdown = ? WHERE id = ?",
            [
                (int(s["score"]), s.get("reason", ""), s.get("breakdown", ""), s["id"])
                for s in scores
            ],
        )


def update_structured_scores(scores: list[dict]) -> None:
    """Persist scores from score_jobs_structured(). Each dict must have id,
    score, reason, breakdown. Kept as a fully separate function from
    update_scores() — never touches the raw score/score_reason/
    score_breakdown columns — so the two scores can be compared side by
    side for the same job."""
    with _connect() as conn:
        conn.executemany(
            "UPDATE jobs SET structured_score = ?, structured_score_reason = ?, "
            "structured_score_breakdown = ? WHERE id = ?",
            [
                (int(s["score"]), s.get("reason", ""), s.get("breakdown", ""), s["id"])
                for s in scores
            ],
        )


# ── Referrals ──────────────────────────────────────────────────────────────────

def save_referral(job_id: str, name: str, title: str, linkedin_url: str,
                  message: str, degree: str = "", photo_url: str = "") -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO referrals
               (job_id, name, title, linkedin_url, degree, photo_url, message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, name, title, linkedin_url, degree, photo_url, message),
        )
        return cur.lastrowid


def get_referrals(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, job_id, name, title, linkedin_url, degree, photo_url, "
            "message, created_at "
            "FROM referrals WHERE job_id = ? ORDER BY created_at DESC",
            (job_id,),
        ).fetchall()
    cols = ["id", "job_id", "name", "title", "linkedin_url",
            "degree", "photo_url", "message", "created_at"]
    return [dict(zip(cols, r)) for r in rows]


def delete_referral(referral_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM referrals WHERE id = ?", (referral_id,))
