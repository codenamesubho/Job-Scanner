import io
import sqlite3
from scanner.database import _connect as _db_connect

# ── Table definitions ──────────────────────────────────────────────────────────

_CREATE_CANDIDATE = """
CREATE TABLE IF NOT EXISTS candidate (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    name        TEXT,
    email       TEXT,
    phone       TEXT,
    linkedin    TEXT,
    title       TEXT,
    years_exp   INTEGER,
    summary     TEXT,
    updated_at  TEXT DEFAULT (datetime('now'))
)
"""

_CREATE_RESUME = """
CREATE TABLE IF NOT EXISTS resume (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,
    content_type TEXT,
    raw_content  BLOB NOT NULL,
    uploaded_at  TEXT DEFAULT (datetime('now'))
)
"""

# Multi-row criteria table (name column, no id=1 constraint)
_CREATE_CRITERIA = """
CREATE TABLE IF NOT EXISTS search_criteria (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL DEFAULT 'Default',
    keywords    TEXT DEFAULT 'software engineer',
    location    TEXT DEFAULT 'USA',
    results     INTEGER DEFAULT 25,
    hours       INTEGER DEFAULT 72,
    remote_only INTEGER DEFAULT 0,
    updated_at  TEXT DEFAULT (datetime('now'))
)
"""

_CREATE_COMPANY_BOARDS = """
CREATE TABLE IF NOT EXISTS company_boards (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    ats        TEXT NOT NULL,
    token      TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
)
"""

_CRITERIA_DEFAULTS = {
    "id":         None,
    "name":       "Default",
    "keywords":   "software engineer",
    "location":   "USA",
    "results":    25,
    "hours":      72,
    "remote_only": 0,
}


def _setup_criteria(conn: sqlite3.Connection) -> None:
    """Create criteria table, migrating from the old single-row schema if needed."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='search_criteria'"
    ).fetchone()

    if not exists:
        conn.execute(_CREATE_CRITERIA)
        return

    cols = {r[1] for r in conn.execute("PRAGMA table_info(search_criteria)").fetchall()}
    if "name" not in cols:
        # Old single-row schema: save the row, drop, recreate, re-insert as "Default"
        old = conn.execute(
            "SELECT keywords, location, results, hours, remote_only FROM search_criteria LIMIT 1"
        ).fetchone()
        conn.execute("DROP TABLE search_criteria")
        conn.execute(_CREATE_CRITERIA)
        if old:
            conn.execute(
                "INSERT INTO search_criteria (name, keywords, location, results, hours, remote_only)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("Default", *old),
            )


def _connect() -> sqlite3.Connection:
    conn = _db_connect()
    conn.execute(_CREATE_CANDIDATE)
    conn.execute(_CREATE_RESUME)
    conn.execute(_CREATE_COMPANY_BOARDS)
    _setup_criteria(conn)
    for ddl in (
        "ALTER TABLE candidate ADD COLUMN resume_extracted TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


# ── Candidate ──────────────────────────────────────────────────────────────────

def save_candidate(name: str, email: str, phone: str, linkedin: str,
                   title: str, years_exp: int, summary: str,
                   resume_extracted: str | None = None) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO candidate (id, name, email, phone, linkedin, title,
                                    years_exp, summary, resume_extracted, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                name       = excluded.name,
                email      = excluded.email,
                phone      = excluded.phone,
                linkedin   = excluded.linkedin,
                title      = excluded.title,
                years_exp  = excluded.years_exp,
                summary    = excluded.summary,
                resume_extracted = COALESCE(excluded.resume_extracted, candidate.resume_extracted),
                updated_at = datetime('now')
        """, (name, email, phone, linkedin, title, years_exp, summary, resume_extracted))


def get_candidate() -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM candidate WHERE id = 1").fetchone()
    if row is None:
        return {}
    cols = ["id", "name", "email", "phone", "linkedin", "title", "years_exp",
            "summary", "updated_at", "resume_extracted"]
    return dict(zip(cols, row))


# ── Resume ─────────────────────────────────────────────────────────────────────

def save_resume(filename: str, content_type: str, raw_content: bytes) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO resume (filename, content_type, raw_content) VALUES (?, ?, ?)",
            (filename, content_type, raw_content),
        )
        return cur.lastrowid


def get_latest_resume() -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, filename, content_type, raw_content, uploaded_at FROM resume ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["id", "filename", "content_type", "raw_content", "uploaded_at"], row))


def extract_text(filename: str, raw_content: bytes) -> str:
    """Best-effort text extraction from PDF or DOCX. Returns empty string on failure.

    PDFs use pymupdf4llm rather than pypdf/plain PyMuPDF — pypdf's
    extract_text() mangles resumes built with letter-spaced/print-to-PDF
    templates (every glyph its own positioned run, rendered as single
    characters joined by spaces), and even plain PyMuPDF's
    page.get_text(sort=True) interleaves two-column/sidebar resume layouts
    (contact/skills sidebar next to a summary/work-experience main column)
    mid-line, scrambling tokens like emails and phone numbers into
    unrelated sentences. pymupdf4llm is purpose-built for LLM consumption:
    it produces genuinely layout-aware markdown (proper section headers,
    each work-experience entry kept intact) instead of a hand-rolled
    column-splitting heuristic.
    """
    try:
        if filename.lower().endswith(".pdf"):
            import fitz
            import pymupdf4llm
            with fitz.open(stream=raw_content, filetype="pdf") as doc:
                return pymupdf4llm.to_markdown(doc).strip()
        if filename.lower().endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(raw_content))
            return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception:
        pass
    return ""


# ── Search criteria (multi-profile) ───────────────────────────────────────────

def save_criteria(
    name: str,
    keywords: str,
    location: str,
    results: int,
    hours: int,
    remote_only: bool,
    criteria_id: int | None = None,
) -> int:
    """Insert a new profile or update an existing one (by criteria_id). Returns the row id."""
    with _connect() as conn:
        if criteria_id is not None:
            conn.execute(
                """UPDATE search_criteria
                   SET name=?, keywords=?, location=?, results=?, hours=?, remote_only=?,
                       updated_at=datetime('now')
                   WHERE id=?""",
                (name, keywords, location, results, hours, int(remote_only), criteria_id),
            )
            return criteria_id
        cur = conn.execute(
            """INSERT INTO search_criteria (name, keywords, location, results, hours, remote_only)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, keywords, location, results, hours, int(remote_only)),
        )
        return cur.lastrowid


def get_criteria() -> dict:
    """Return the first saved profile (used to seed sidebar defaults)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, keywords, location, results, hours, remote_only FROM search_criteria ORDER BY id LIMIT 1"
        ).fetchone()
    if row is None:
        return _CRITERIA_DEFAULTS.copy()
    cols = ["id", "name", "keywords", "location", "results", "hours", "remote_only"]
    return dict(zip(cols, row))


def get_all_criteria() -> list[dict]:
    """Return all saved search profiles ordered by id."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, keywords, location, results, hours, remote_only FROM search_criteria ORDER BY id"
        ).fetchall()
    cols = ["id", "name", "keywords", "location", "results", "hours", "remote_only"]
    return [dict(zip(cols, r)) for r in rows]


def delete_criteria(criteria_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM search_criteria WHERE id = ?", (criteria_id,))


# ── Company boards (Greenhouse / Lever / Ashby) ───────────────────────────────

def save_company_board(name: str, ats: str, token: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO company_boards (name, ats, token) VALUES (?, ?, ?)",
            (name, ats, token),
        )
        return cur.lastrowid


def get_company_boards() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, ats, token, created_at FROM company_boards ORDER BY id"
        ).fetchall()
    cols = ["id", "name", "ats", "token", "created_at"]
    return [dict(zip(cols, r)) for r in rows]


def delete_company_board(board_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM company_boards WHERE id = ?", (board_id,))
