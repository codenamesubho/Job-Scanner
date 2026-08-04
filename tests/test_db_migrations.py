
import pandas as pd

from scanner import database, profile


def test_jobs_table_gets_new_score_columns(isolated_db):
    conn = database._connect()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    conn.close()

    assert "jd_extracted" in cols
    assert "structured_score" in cols
    assert "structured_score_reason" in cols
    assert "structured_score_breakdown" in cols


def test_jobs_migration_is_idempotent(isolated_db):
    database._connect().close()
    database._connect().close()  # second connect must not raise


def test_candidate_table_gets_resume_extracted_column(isolated_db):
    conn = profile._connect()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(candidate)").fetchall()]
    conn.close()

    assert "resume_extracted" in cols
    # SQLite appends ALTER TABLE ADD COLUMN columns at the physical end of
    # the row, regardless of where you'd conceptually expect it.
    assert cols[-1] == "resume_extracted"


def test_get_candidate_returns_resume_extracted_under_correct_key(isolated_db):
    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "A summary.", resume_extracted='{"skills": ["python"]}')

    cand = profile.get_candidate()
    assert cand["resume_extracted"] == '{"skills": ["python"]}'
    assert cand["name"] == "Test User"


def test_save_candidate_preserves_resume_extracted_on_plain_save(isolated_db):
    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "A summary.", resume_extracted='{"skills": ["python"]}')

    # Plain save, as a normal "Save Details" click would do — no resume_extracted arg.
    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "An updated summary.")

    cand = profile.get_candidate()
    assert cand["resume_extracted"] == '{"skills": ["python"]}'
    assert cand["summary"] == "An updated summary."


def test_update_structured_scores_does_not_touch_raw_score_columns(isolated_db):
    df = pd.DataFrame([{
        "id": "job-1", "site": "test", "job_url": "", "job_url_direct": "",
        "title": "Engineer", "company": "Acme", "location": "Remote",
        "date_posted": None, "is_remote": True, "description": "desc",
    }])
    database.save_jobs(df)
    database.update_scores([{"id": "job-1", "score": 70, "reason": "raw reason", "breakdown": "{}"}])
    database.update_structured_scores([{"id": "job-1", "score": 55, "reason": "structured reason", "breakdown": "{}"}])

    jobs = database.get_jobs()
    row = jobs[jobs["id"] == "job-1"].iloc[0]
    assert row["score"] == 70
    assert row["score_reason"] == "raw reason"
    assert row["structured_score"] == 55
    assert row["structured_score_reason"] == "structured reason"


def test_get_jobs_missing_structured_score_independent_of_raw_score(isolated_db):
    df = pd.DataFrame([{
        "id": "job-1", "site": "test", "job_url": "", "job_url_direct": "",
        "title": "Engineer", "company": "Acme", "location": "Remote",
        "date_posted": None, "is_remote": True, "description": "desc",
    }])
    database.save_jobs(df)
    database.update_scores([{"id": "job-1", "score": 70, "reason": "r", "breakdown": "{}"}])

    # Has a raw score but no structured_score — should still show up as
    # missing_structured_score=True, since the two are independent.
    pending = database.get_jobs(missing_structured_score=True)
    assert "job-1" in set(pending["id"])

    unscored = database.get_jobs(unscored_only=True)
    assert "job-1" not in set(unscored["id"])


def test_parse_jd_extracted_handles_missing_and_invalid(isolated_db):
    assert database.parse_jd_extracted(None) is None
    assert database.parse_jd_extracted("") is None
    assert database.parse_jd_extracted("not json") is None
    assert database.parse_jd_extracted('{"must_haves": ["python"]}') == {"must_haves": ["python"]}


_LONG_DESCRIPTION = (
    "We are looking for a Senior Backend Engineer to join our platform team, "
    "working on distributed systems, event-driven pipelines, and API design. "
    "You will own core services end-to-end, collaborate closely with product "
    "and data teams, and help scale our infrastructure as the company grows."
)


def test_content_hash_none_below_threshold_stable_above(isolated_db):
    assert database.content_hash(None) is None
    assert database.content_hash("") is None
    assert database.content_hash("too short") is None
    # A short boilerplate blurb (the kind of text two DIFFERENT companies'
    # postings could plausibly share) must stay under the floor — this hash
    # has no company/title scoping of its own, so a false merge here would
    # be silent (save_jobs() just returns a lower new_count, nothing flags
    # it). The floor exists specifically to keep stubs like this out.
    boilerplate = "We are hiring a Software Engineer. Apply through our careers page."
    assert len(boilerplate) < database._CONTENT_HASH_MIN_CHARS
    assert database.content_hash(boilerplate) is None

    h1 = database.content_hash(_LONG_DESCRIPTION)
    h2 = database.content_hash(_LONG_DESCRIPTION.upper())  # case-insensitive
    assert h1 is not None
    assert len(h1) == 12
    assert h1 == h2  # normalized (stripped/lowercased) before hashing


def test_save_jobs_dedups_by_content_hash_across_different_titles(isolated_db):
    df1 = pd.DataFrame([{
        "id": "linkedin-1", "site": "linkedin", "job_url": "", "job_url_direct": "",
        "title": "Sr. Backend Engineer", "company": "Acme", "location": "Remote",
        "date_posted": None, "is_remote": True, "description": _LONG_DESCRIPTION,
    }])
    # Different id AND different title (so the (title, company) key alone
    # would NOT catch this as a duplicate) but identical description text.
    df2 = pd.DataFrame([{
        "id": "greenhouse-1", "site": "greenhouse", "job_url": "", "job_url_direct": "https://direct.example/job",
        "title": "Senior Backend Engineer (Platform)", "company": "Acme", "location": "Remote",
        "date_posted": None, "is_remote": True, "description": _LONG_DESCRIPTION,
    }])

    added1 = database.save_jobs(df1)
    added2 = database.save_jobs(df2)

    assert added1 == 1
    assert added2 == 0  # merged via content_hash, not inserted as a new row

    jobs = database.get_jobs()
    assert len(jobs) == 1
    row = jobs.iloc[0]
    assert row["id"] == "linkedin-1"  # canonical row is the first-seen one
    assert row["job_url_direct"] == "https://direct.example/job"  # backfilled from the second sighting


def test_backfill_content_hashes_populates_missing_only_by_default(isolated_db):
    df = pd.DataFrame([{
        "id": "job-1", "site": "test", "job_url": "", "job_url_direct": "",
        "title": "Engineer", "company": "Acme", "location": "Remote",
        "date_posted": None, "is_remote": True, "description": _LONG_DESCRIPTION,
    }])
    database.save_jobs(df)

    # Simulate a pre-existing row saved before content_hash existed.
    conn = database._connect()
    conn.execute("UPDATE jobs SET content_hash = NULL WHERE id = 'job-1'")
    conn.commit()
    conn.close()

    updated = database.backfill_content_hashes()
    assert updated == 1

    jobs = database.get_jobs()
    assert jobs.iloc[0]["content_hash"] == database.content_hash(_LONG_DESCRIPTION)

    # Second run: already populated, not force — no-op.
    assert database.backfill_content_hashes() == 0
    # force=True recomputes regardless.
    assert database.backfill_content_hashes(force=True) == 1


def test_get_jobs_orders_by_raw_score_when_structured_score_is_absent(isolated_db):
    """Regression: get_jobs() used to ORDER BY COALESCE(structured_score, -1), which
    dropped the raw `score` fallback its own docstring promised. Every job without a
    structured score collapsed to -1 and tied, so `--top N by score` on the backfill
    scripts silently degraded to "newest N"."""
    database.save_jobs(pd.DataFrame([
        {"id": "j-low",  "site": "x", "title": "Low",  "company": "A", "description": "d"},
        {"id": "j-high", "site": "x", "title": "High", "company": "B", "description": "d"},
        {"id": "j-mid",  "site": "x", "title": "Mid",  "company": "C", "description": "d"},
    ]))
    database.update_scores([
        {"id": "j-low", "score": 20, "reason": ""},
        {"id": "j-high", "score": 90, "reason": ""},
        {"id": "j-mid", "score": 50, "reason": ""},
    ])

    assert list(database.get_jobs()["id"]) == ["j-high", "j-mid", "j-low"]


def test_get_jobs_prefers_structured_score_over_raw_score(isolated_db):
    """When both exist, structured_score wins — that is the primary sort key."""
    database.save_jobs(pd.DataFrame([
        {"id": "j-a", "site": "x", "title": "A", "company": "A", "description": "d"},
        {"id": "j-b", "site": "x", "title": "B", "company": "B", "description": "d"},
    ]))
    # Raw scores say j-a is better; structured scores say the opposite.
    database.update_scores([
        {"id": "j-a", "score": 90, "reason": ""},
        {"id": "j-b", "score": 10, "reason": ""},
    ])
    database.update_structured_scores([
        {"id": "j-a", "score": 10, "reason": ""},
        {"id": "j-b", "score": 90, "reason": ""},
    ])

    assert list(database.get_jobs()["id"]) == ["j-b", "j-a"]


def test_get_jobs_sorts_unscored_jobs_last(isolated_db):
    database.save_jobs(pd.DataFrame([
        {"id": "j-none", "site": "x", "title": "None", "company": "A", "description": "d"},
        {"id": "j-scored", "site": "x", "title": "Scored", "company": "B", "description": "d"},
    ]))
    database.update_scores([{"id": "j-scored", "score": 5, "reason": ""}])

    assert list(database.get_jobs()["id"]) == ["j-scored", "j-none"]
