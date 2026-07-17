import json

import pandas as pd

from scanner import database, profile, scoring


def _isolate_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)


def _seed_job(job_id="job-1", description="A great job description.",
              title="Engineer", company="Acme"):
    df = pd.DataFrame([{
        "id": job_id, "site": "test", "job_url": "", "job_url_direct": "",
        "title": title, "company": company, "location": "Remote",
        "date_posted": None, "is_remote": True, "description": description,
    }])
    database.save_jobs(df)


def test_extract_missing_job_requirements_noop_under_raw_mode(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.delenv("SCORING_MODE", raising=False)
    _seed_job()

    def _boom(description):
        raise AssertionError("extract_job_requirements should not be called under raw mode")

    monkeypatch.setattr(scoring, "extract_job_requirements", _boom)

    assert scoring.extract_missing_job_requirements(log_fn=lambda m: None) == 0


def test_extract_missing_job_requirements_populates_column(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setenv("SCORING_MODE", "structured")
    _seed_job("job-1", title="Backend Engineer", company="Acme")
    _seed_job("job-2", title="Frontend Engineer", company="Beta")

    class FakeRequirements:
        def model_dump(self):
            return {"must_haves": ["python"]}

    calls = []

    def fake_extract(description, company=None):
        calls.append(description)
        return FakeRequirements()

    monkeypatch.setattr(scoring, "extract_job_requirements", fake_extract)

    extracted = scoring.extract_missing_job_requirements(log_fn=lambda m: None)
    assert extracted == 2
    assert len(calls) == 2

    jobs = database.get_jobs()
    for _, row in jobs.iterrows():
        assert json.loads(row["jd_extracted"]) == {"must_haves": ["python"]}

    # Second run should skip already-populated jobs.
    calls.clear()
    extracted_again = scoring.extract_missing_job_requirements(log_fn=lambda m: None)
    assert extracted_again == 0
    assert calls == []

    # force=True re-extracts everything, even already-populated jobs (e.g. to
    # backfill a newly-added JobRequirements field).
    class FakeRequirementsV2:
        def model_dump(self):
            return {"must_haves": ["python"], "description": "A synopsis."}

    monkeypatch.setattr(scoring, "extract_job_requirements", lambda description, company=None: FakeRequirementsV2())
    extracted_forced = scoring.extract_missing_job_requirements(log_fn=lambda m: None, force=True)
    assert extracted_forced == 2

    jobs_after_force = database.get_jobs()
    for _, row in jobs_after_force.iterrows():
        assert json.loads(row["jd_extracted"]) == {"must_haves": ["python"], "description": "A synopsis."}


def test_extract_missing_job_requirements_limit_takes_top_n_by_score(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setenv("SCORING_MODE", "structured")
    _seed_job("job-low", title="Low Score Job", company="Acme")
    _seed_job("job-high", title="High Score Job", company="Beta")
    _seed_job("job-mid", title="Mid Score Job", company="Gamma")
    database.update_scores([
        {"id": "job-low", "score": 20, "reason": ""},
        {"id": "job-high", "score": 90, "reason": ""},
        {"id": "job-mid", "score": 50, "reason": ""},
    ])

    class FakeRequirements:
        def model_dump(self):
            return {"must_haves": ["python"]}

    extracted_ids = []

    def fake_extract(description, company=None):
        return FakeRequirements()

    monkeypatch.setattr(scoring, "extract_job_requirements", fake_extract)

    def fake_update_job_fields(job_id, fields):
        extracted_ids.append(job_id)
        database.update_job_fields(job_id, fields)

    monkeypatch.setattr(scoring, "update_job_fields", fake_update_job_fields)

    extracted = scoring.extract_missing_job_requirements(log_fn=lambda m: None, limit=2)

    assert extracted == 2
    # get_jobs() orders by score desc, so the top 2 by score are job-high, job-mid.
    assert extracted_ids == ["job-high", "job-mid"]

    still_missing = database.get_jobs()
    low_row = still_missing[still_missing["id"] == "job-low"].iloc[0]
    assert pd.isna(low_row["jd_extracted"]) or low_row["jd_extracted"] in (None, "")


def test_score_unscored_jobs_uses_raw_path_by_default(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.delenv("SCORING_MODE", raising=False)
    _seed_job()

    profile.save_candidate("Test User", "t@example.com", "", "", "Engineer", 5,
                            "An experienced engineer.")

    monkeypatch.setattr(scoring, "scoring_breaker_status", lambda: {"open": False})

    structured_calls = []

    def fake_score_jobs(summary, jobs_list, log_fn=None):
        yield [{"id": jobs_list[0]["id"], "score": 80, "reason": "good fit", "breakdown": "{}"}]

    def fake_score_jobs_structured(*args, **kwargs):
        structured_calls.append(1)
        return iter([])

    monkeypatch.setattr(scoring, "score_jobs", fake_score_jobs)
    monkeypatch.setattr(scoring, "score_jobs_structured", fake_score_jobs_structured)

    scored = scoring.score_unscored_jobs(log_fn=lambda m: None)

    assert scored == 1
    assert structured_calls == []
    row = database.get_jobs().iloc[0]
    assert row["score"] == 80


def test_score_unscored_jobs_uses_structured_path_when_enabled(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setenv("SCORING_MODE", "structured")
    _seed_job()

    profile.save_candidate("Test User", "t@example.com", "", "", "Engineer", 5,
                            "An experienced engineer.",
                            resume_extracted=json.dumps({"skills": ["python"]}))

    monkeypatch.setattr(scoring, "scoring_breaker_status", lambda: {"open": False})

    class FakeRequirements:
        def model_dump(self):
            return {"must_haves": ["python"]}

    monkeypatch.setattr(scoring, "extract_job_requirements", lambda description, company=None: FakeRequirements())

    raw_calls = []

    def fake_score_jobs(*args, **kwargs):
        raw_calls.append(1)
        return iter([])

    def fake_score_jobs_structured(resume_profile, jobs_list, log_fn=None):
        yield [{"id": jobs_list[0]["id"], "score": 55, "reason": "structured fit", "breakdown": "{}"}]

    monkeypatch.setattr(scoring, "score_jobs", fake_score_jobs)
    monkeypatch.setattr(scoring, "score_jobs_structured", fake_score_jobs_structured)

    scored = scoring.score_unscored_jobs(log_fn=lambda m: None)

    assert scored == 1
    assert raw_calls == []
    row = database.get_jobs().iloc[0]
    assert row["structured_score"] == 55
    assert pd.isna(row["score"])  # raw score untouched


def test_score_unscored_jobs_raw_path_respects_limit(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.delenv("SCORING_MODE", raising=False)
    _seed_job("job-1", title="Engineer One", company="Acme")
    _seed_job("job-2", title="Engineer Two", company="Beta")
    _seed_job("job-3", title="Engineer Three", company="Gamma")

    profile.save_candidate("Test User", "t@example.com", "", "", "Engineer", 5,
                            "An experienced engineer.")

    monkeypatch.setattr(scoring, "scoring_breaker_status", lambda: {"open": False})

    seen_jobs_lists = []

    def fake_score_jobs(summary, jobs_list, log_fn=None):
        seen_jobs_lists.append(jobs_list)
        yield [{"id": job["id"], "score": 80, "reason": "good fit", "breakdown": "{}"} for job in jobs_list]

    monkeypatch.setattr(scoring, "score_jobs", fake_score_jobs)

    scored = scoring.score_unscored_jobs(log_fn=lambda m: None, limit=1)

    assert scored == 1
    assert len(seen_jobs_lists) == 1
    assert len(seen_jobs_lists[0]) == 1  # limit=1 must cap the jobs actually sent to score_jobs


def test_score_unscored_jobs_structured_path_respects_limit(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setenv("SCORING_MODE", "structured")
    _seed_job("job-1", title="Engineer One", company="Acme")
    _seed_job("job-2", title="Engineer Two", company="Beta")
    _seed_job("job-3", title="Engineer Three", company="Gamma")
    # All three already have jd_extracted populated (e.g. from a prior run),
    # so extract_missing_job_requirements no-ops regardless of `limit` — this
    # isolates the scoring-step limit bug from the extraction-step one
    # already covered by test_extract_missing_job_requirements_limit_takes_top_n_by_score.
    for job_id in ("job-1", "job-2", "job-3"):
        database.update_job_fields(job_id, {"jd_extracted": json.dumps({"must_haves": ["python"]})})

    profile.save_candidate("Test User", "t@example.com", "", "", "Engineer", 5,
                            "An experienced engineer.",
                            resume_extracted=json.dumps({"skills": ["python"]}))

    monkeypatch.setattr(scoring, "scoring_breaker_status", lambda: {"open": False})

    def _boom(description, company=None):
        raise AssertionError("extract_job_requirements should not be called — jd_extracted already set")

    monkeypatch.setattr(scoring, "extract_job_requirements", _boom)

    seen_jobs_lists = []

    def fake_score_jobs_structured(resume_profile, jobs_list, log_fn=None):
        seen_jobs_lists.append(jobs_list)
        yield [{"id": job["id"], "score": 55, "reason": "structured fit", "breakdown": "{}"} for job in jobs_list]

    monkeypatch.setattr(scoring, "score_jobs_structured", fake_score_jobs_structured)

    scored = scoring.score_unscored_jobs(log_fn=lambda m: None, limit=1)

    assert scored == 1
    assert len(seen_jobs_lists) == 1
    assert len(seen_jobs_lists[0]) == 1  # limit=1 must cap the jobs actually sent to score_jobs_structured


def test_load_resume_profile_uses_cached_json(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)

    def _boom(text):
        raise AssertionError("extract_resume_profile should not be called when cache is present")

    monkeypatch.setattr(scoring, "extract_resume_profile", _boom)

    from scanner.llm import ResumeProfile
    candidate = {"resume_extracted": json.dumps({"skills": ["python"], "full_name": "Test User"})}
    profile_obj = scoring.load_resume_profile(candidate, log_fn=lambda m: None)

    assert isinstance(profile_obj, ResumeProfile)
    assert profile_obj.skills == ["python"]
    assert profile_obj.full_name == "Test User"
