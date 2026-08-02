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
    # The LLM is asked to echo back a short, hash-derived label (see
    # _batch_short_id), not the job's real id — a single job at position 1
    # in the batch always gets this exact label, matching _score_batch's
    # own id_map construction.
    short_id = llm._batch_short_id("D", 1)  # "D" is _raw_job()'s fixed description
    item = llm.JobScoreItem(
        id=short_id, score=80, reason="good fit",
        breakdown=llm.JobBreakdown(),
    )
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.BatchScoreResult(scores=[item]))

    out = llm._score_batch("candidate summary", [_raw_job()])
    assert len(out) == 1
    assert out[0]["id"] == "job-1"  # real job id comes back out, not the short label


def test_score_structured_batch_raises_on_empty_scores(monkeypatch):
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.StructuredBatchScoreResult(scores=[]))

    with pytest.raises(llm.EmptyScoringResultError):
        llm._score_structured_batch("{}", {"years_exp": 5}, [_structured_job()])


def test_batch_short_id_deterministic_and_unique_per_position():
    a = llm._batch_short_id("some job description", 1)
    b = llm._batch_short_id("some job description", 1)
    c = llm._batch_short_id("some job description", 2)
    d = llm._batch_short_id("a different description", 1)

    assert a == b  # deterministic for the same (text, index)
    assert a != c  # position disambiguates even identical text (collision safety)
    assert a != d


def test_score_batch_matches_job_with_long_opaque_real_id(monkeypatch):
    """Reproduces the bug this fix addresses: a real id long/opaque enough
    that an LLM could garble it (e.g. JSearch's 400+ char ids) must still
    match correctly, because the LLM never has to reproduce that id at all
    — only the short label _score_batch generates itself."""
    long_id = "jsearch-" + "aZ09" * 100  # 408 chars, representative of a real JSearch id
    job = _raw_job(job_id=long_id)
    short_id = llm._batch_short_id(job["description"], 1)

    item = llm.JobScoreItem(id=short_id, score=65, reason="ok fit", breakdown=llm.JobBreakdown())
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.BatchScoreResult(scores=[item]))

    out = llm._score_batch("candidate summary", [job])
    assert len(out) == 1
    assert out[0]["id"] == long_id


def test_score_structured_batch_matches_job_with_long_opaque_real_id(monkeypatch):
    import json as _json

    long_id = "jsearch-" + "aZ09" * 100
    job = _structured_job(job_id=long_id)
    short_id = llm._batch_short_id(_json.dumps(job.get("requirements"), sort_keys=True), 1)

    item = llm.StructuredJobScoreItem(
        id=short_id,
        skills=llm.StructuredSkillsJudgment(core_fit="core_match", core_fit_reason="r"),
        company=llm.StructuredCompanyJudgment(tier="mid_sized", reason="r"),
        remote=llm.StructuredRemoteJudgment(tier="remote", reason="r"),
        role=llm.StructuredRoleJudgment(tier="ambiguous_scale", reason="r"),
        overall_reason="overall",
    )
    monkeypatch.setattr(llm, "execute_with_breaker", lambda *a, **kw: llm.StructuredBatchScoreResult(scores=[item]))

    out = llm._score_structured_batch("{}", {"years_exp": 5}, [job])
    assert len(out) == 1
    assert out[0]["id"] == long_id


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
