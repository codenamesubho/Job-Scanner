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


def connect() -> sqlite3.Connection:
    """Open the shared database, creating/migrating the jobs+referrals tables.

    Public because scanner.profile builds its own tables on top of the same
    file and needs a way in that isn't a private name. Reads DB_PATH at call
    time, which is what lets tests repoint it (see the isolated_db fixture).
    """
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


#: Long-standing internal alias for connect(), kept so existing callers
#: (and tests that patch around it) keep working.
_connect = connect


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


class _CanonicalIndex:
    """Which already-known job, if any, an incoming row is a re-sighting of.

    Holds the three lookups that have to move together: known ids, the
    (title, company) key, and the content_hash. `remember()` updating all of
    them is what makes within-batch dedup work — two sources in one save_jobs()
    call can't both insert the same role, because the first one registers itself
    here before the second is classified.

    Values are (canonical_id, that row's current job_url_direct).
    """

    def __init__(self, rows):
        self.ids = {row[0] for row in rows}
        self.by_title_company = {_norm_key(row[1], row[2]): (row[0], row[3]) for row in rows}
        self.by_hash = {row[4]: (row[0], row[3]) for row in rows if row[4]}

    def knows_id(self, job_id) -> bool:
        return job_id in self.ids

    def find(self, row_hash, key):
        """The canonical row this duplicates, or None if it's genuinely new.

        content_hash is checked first: it catches the same posting mirrored
        across sources under a differently-worded title, which the (title,
        company) key would miss.
        """
        if row_hash:
            found = self.by_hash.get(row_hash)
            if found is not None:
                return found
        return self.by_title_company.get(key)

    def remember(self, job_id, key, row_hash, job_url_direct) -> None:
        self.ids.add(job_id)
        self.by_title_company[key] = (job_id, job_url_direct)
        if row_hash:
            self.by_hash[row_hash] = (job_id, job_url_direct)


def _correct_remote_flag(subset: pd.DataFrame) -> pd.DataFrame:
    """Distrust a source's is_remote when the location names a real place.

    Sources (jobspy in particular) often derive is_remote by keyword-matching
    "remote" anywhere in the description — including hybrid mentions like
    "remote work on Fridays" — which wrongly flags office-bound jobs as fully
    remote. A specific location is the more reliable signal.
    """
    if "is_remote" not in subset.columns or "location" not in subset.columns:
        return subset

    def _correct(row):
        loc = (row["location"] or "").strip().lower()
        if row["is_remote"] and loc and "remote" not in loc:
            return 0
        return row["is_remote"]

    subset["is_remote"] = subset.apply(_correct, axis=1)
    return subset


def _normalize_for_storage(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce a scraped frame into what SQLite will accept."""
    subset = df[cols].copy()
    # NaN -> None so sqlite3 stores NULL, not the string 'nan'.
    subset = subset.where(subset.notna(), None)
    # bool -> int (SQLite has no native bool).
    if "is_remote" in subset.columns:
        subset["is_remote"] = subset["is_remote"].apply(
            lambda x: int(x) if x is not None else None
        )
    return _correct_remote_flag(subset)


def _build_upsert_sql(cols: list[str]) -> tuple[str, list[str]]:
    """The INSERT .. ON CONFLICT(id) DO UPDATE used for same-id rows.

    NULLIF/COALESCE means a re-scrape that comes back with a blank field (e.g. a
    transiently failed description fetch) keeps the previously-stored value
    instead of clobbering it. status and first_seen are never in the update
    clause, so a user's own tracking survives every re-scan.
    """
    upsert_cols = cols + ["content_hash", "status"]
    update_clause = ", ".join(
        f"{c} = COALESCE(NULLIF(excluded.{c}, ''), {c})"
        for c in cols
        if c not in ("id", "status", "first_seen")
    ) + ", content_hash = COALESCE(NULLIF(excluded.content_hash, ''), content_hash)" \
        ", last_seen = datetime('now')"

    sql = f"""
        INSERT INTO jobs ({", ".join(upsert_cols)}, last_seen)
        VALUES ({", ".join("?" * len(upsert_cols))}, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET {update_clause}
    """
    return sql, upsert_cols


def _record_resighting(conn, canonical_id, new_direct) -> None:
    """Bump an existing job's last_seen, backfilling job_url_direct if it had none."""
    if new_direct:
        conn.execute(
            "UPDATE jobs SET last_seen = datetime('now'), "
            "job_url_direct = COALESCE(NULLIF(job_url_direct, ''), ?) WHERE id = ?",
            (new_direct, canonical_id),
        )
    else:
        conn.execute("UPDATE jobs SET last_seen = datetime('now') WHERE id = ?", (canonical_id,))


def save_jobs(df: pd.DataFrame, default_status: str = "new") -> int:
    """Insert new jobs, treating duplicates of an existing job as re-sightings.

    A row duplicates an existing job when it matches that job's content_hash, or
    its normalized (title, company) pair, under a *different* id — e.g. the same
    role scraped from LinkedIn and mirrored on its Greenhouse board. Returns the
    number of genuinely new rows inserted.

    `default_status` is the status a genuinely new row gets (e.g. "shortlisted"
    for jobs added by hand via add_job_by_url, vs "new" for scanned ones). It has
    no effect on same-id upserts, where status is never overwritten.

    Rules:
    - status and first_seen are never overwritten on same-id upserts.
    - A same-id row upserts normally.
    - A row matching a DIFFERENT existing id is a re-sighting: last_seen is
      bumped, job_url_direct backfilled if the canonical row lacked one, and no
      new row is created.
    - The same rule applies within one incoming batch — see _CanonicalIndex.
    """
    if df.empty or "id" not in df.columns:
        return 0

    cols = [c for c in _STORE_COLS if c in df.columns]
    subset = _normalize_for_storage(_dedup_dataframe(df), cols)
    upsert_sql, _ = _build_upsert_sql(cols)

    with _connect() as conn:
        index = _CanonicalIndex(conn.execute(
            "SELECT id, title, company, job_url_direct, content_hash FROM jobs"
        ).fetchall())

        to_upsert: list[list] = []
        resightings: list[tuple] = []   # (canonical_id, incoming job_url_direct or None)
        new_count = 0

        for _, raw in subset.iterrows():
            row = raw.to_dict()
            row_id = row["id"]
            row_hash = content_hash(row.get("description"))
            values = [row[c] for c in cols] + [row_hash, default_status]

            if index.knows_id(row_id):
                to_upsert.append(values)
                continue

            key = _norm_key(row.get("title"), row.get("company"))
            canonical = index.find(row_hash, key)
            if canonical is not None:
                canonical_id, canonical_direct = canonical
                incoming_direct = row.get("job_url_direct")
                resightings.append((canonical_id, None if canonical_direct else incoming_direct))
                continue

            to_upsert.append(values)
            new_count += 1
            index.remember(row_id, key, row_hash, row.get("job_url_direct"))

        if to_upsert:
            conn.executemany(upsert_sql, to_upsert)
        for canonical_id, new_direct in resightings:
            _record_resighting(conn, canonical_id, new_direct)

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
    """Return jobs ordered by structured score desc, falling back to the raw
    score, then nulls last, then newest first.

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

    # COALESCE falls back to the raw `score` before -1 so that jobs which have
    # been raw-scored but not structured-scored still sort by how good they are.
    # Without the `score` term they all collapse to -1 and tie, which silently
    # turned every "--top N by score" backfill into "newest N".
    query += " ORDER BY COALESCE(structured_score, score, -1) DESC, first_seen DESC"

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


def reject_low_scores(threshold: int = 30) -> int:
    """Auto-reject jobs whose primary score (structured score, falling back
    to raw score) is below `threshold`. Only touches rows still at the
    default 'new' status, so a job the user has already saved/applied
    to/manually rejected is never overwritten. Returns the number of rows
    updated."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'rejected' "
            "WHERE status = 'new' AND COALESCE(structured_score, score) < ?",
            (threshold,),
        )
        return cur.rowcount


def get_stats() -> dict:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
    counts = dict(rows)
    return {
        "total":       sum(counts.values()),
        "new":         counts.get("new", 0),
        "shortlisted": counts.get("shortlisted", 0),
        "applied":     counts.get("applied", 0),
        "rejected":    counts.get("rejected", 0),
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
