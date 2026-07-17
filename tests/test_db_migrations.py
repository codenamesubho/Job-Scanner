import sqlite3

import pandas as pd

from scanner import database, profile


def _isolate_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    return db_path


def test_jobs_table_gets_new_score_columns(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    conn = database._connect()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    conn.close()

    assert "jd_extracted" in cols
    assert "structured_score" in cols
    assert "structured_score_reason" in cols
    assert "structured_score_breakdown" in cols


def test_jobs_migration_is_idempotent(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    database._connect().close()
    database._connect().close()  # second connect must not raise


def test_candidate_table_gets_resume_extracted_column(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    conn = profile._connect()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(candidate)").fetchall()]
    conn.close()

    assert "resume_extracted" in cols
    # SQLite appends ALTER TABLE ADD COLUMN columns at the physical end of
    # the row, regardless of where you'd conceptually expect it.
    assert cols[-1] == "resume_extracted"


def test_get_candidate_returns_resume_extracted_under_correct_key(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "A summary.", resume_extracted='{"skills": ["python"]}')

    cand = profile.get_candidate()
    assert cand["resume_extracted"] == '{"skills": ["python"]}'
    assert cand["name"] == "Test User"


def test_save_candidate_preserves_resume_extracted_on_plain_save(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "A summary.", resume_extracted='{"skills": ["python"]}')

    # Plain save, as a normal "Save Details" click would do — no resume_extracted arg.
    profile.save_candidate("Test User", "t@example.com", "555-1234", "linkedin.com/in/test",
                            "Engineer", 5, "An updated summary.")

    cand = profile.get_candidate()
    assert cand["resume_extracted"] == '{"skills": ["python"]}'
    assert cand["summary"] == "An updated summary."


def test_update_structured_scores_does_not_touch_raw_score_columns(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

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


def test_get_jobs_missing_structured_score_independent_of_raw_score(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

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


def test_parse_jd_extracted_handles_missing_and_invalid(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    assert database.parse_jd_extracted(None) is None
    assert database.parse_jd_extracted("") is None
    assert database.parse_jd_extracted("not json") is None
    assert database.parse_jd_extracted('{"must_haves": ["python"]}') == {"must_haves": ["python"]}
