import pytest

from scanner import llm


def _raw_job(job_id="job-1"):
    return {"id": job_id, "title": "T", "company": "C", "description": "D", "is_remote": True}


def _structured_job(job_id="job-1"):
    return {"id": job_id, "requirements": {}, "is_remote": True, "title": "T", "company": "C"}


def test_score_batch_raises_on_empty_scores(monkeypatch):
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.BatchScoreResult(scores=[]))

    with pytest.raises(llm.EmptyScoringResultError):
        llm._score_batch("candidate summary", [_raw_job()])


def test_score_batch_does_not_raise_when_scores_present(monkeypatch):
    item = llm.JobScoreItem(
        id="job-1", score=80, reason="good fit",
        breakdown=llm.JobBreakdown(),
    )
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.BatchScoreResult(scores=[item]))

    out = llm._score_batch("candidate summary", [_raw_job()])
    assert len(out) == 1
    assert out[0]["id"] == "job-1"


def test_score_structured_batch_raises_on_empty_scores(monkeypatch):
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.StructuredBatchScoreResult(scores=[]))

    with pytest.raises(llm.EmptyScoringResultError):
        llm._score_structured_batch("{}", {"years_exp": 5}, [_structured_job()])


def test_score_jobs_propagates_empty_scoring_result_and_stops(monkeypatch):
    def fake_score_batch(*a, **kw):
        raise llm.EmptyScoringResultError("LLM returned nothing")

    monkeypatch.setattr(llm, "_score_batch", fake_score_batch)
    monkeypatch.setattr(llm, "_warm_up_litellm", lambda *a, **kw: None)

    gen = llm.score_jobs("summary", [_raw_job()], log_fn=lambda m: None)
    with pytest.raises(llm.EmptyScoringResultError):
        list(gen)


def test_score_jobs_structured_propagates_empty_scoring_result_and_stops(monkeypatch):
    def fake_score_structured_batch(*a, **kw):
        raise llm.EmptyScoringResultError("LLM returned nothing")

    monkeypatch.setattr(llm, "_score_structured_batch", fake_score_structured_batch)
    monkeypatch.setattr(llm, "_warm_up_litellm", lambda *a, **kw: None)
    monkeypatch.setattr(llm, "scoring_breaker_status", lambda: {"open": False})

    resume_profile = llm.ResumeProfile()
    jobs_list = [{
        "id": "job-1", "requirements": {}, "is_remote": True, "title": "T", "company": "C",
    }]
    gen = llm.score_jobs_structured(resume_profile, jobs_list, log_fn=lambda m: None)
    with pytest.raises(llm.EmptyScoringResultError):
        list(gen)


def test_ordinary_exception_still_yields_empty_and_continues(monkeypatch):
    """Non-empty-result failures (timeouts, validation errors, etc.) keep
    the existing tolerant behavior — log, yield [], and keep processing
    other batches — unlike EmptyScoringResultError."""
    def fake_score_batch(*a, **kw):
        raise ValueError("transient blip")

    monkeypatch.setattr(llm, "_score_batch", fake_score_batch)
    monkeypatch.setattr(llm, "_warm_up_litellm", lambda *a, **kw: None)

    gen = llm.score_jobs("summary", [_raw_job()], log_fn=lambda m: None)
    results = list(gen)
    assert results == [[]]
