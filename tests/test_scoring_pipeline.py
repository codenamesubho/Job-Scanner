"""Tests for ui.scoring.pipeline — the scoring run shared by the Score button
and post-scan auto-scoring.

These two callers each used to carry their own near-identical worker closure.
The tests below pin the three ways they legitimately differ (which jobs are
considered, cancellability, the nothing-to-do message) and the behaviour they
share (per-batch persistence, structured/raw routing).
"""
import pandas as pd
import pytest

from ui.background import BackgroundJob
from ui.models import ScanRequest
from ui.scoring import pipeline
from ui.scoring.pipeline import ScoringPlan, run_scoring

AUTO_PLAN   = ScoringPlan(only_unscored=True,  nothing_to_do="No new jobs need scoring.")
BUTTON_PLAN = ScoringPlan(only_unscored=False, nothing_to_do="No jobs with descriptions to score.",
                           cancellable=True)
CANDIDATE = {"summary": "a candidate summary"}


@pytest.fixture
def raw_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "scoring_mode", lambda: "raw")


def _jobs(n=2, description="a description"):
    return pd.DataFrame([
        {"id": f"j{i}", "title": f"T{i}", "company": "Acme",
         "description": description, "is_remote": 0}
        for i in range(n)
    ])


def _run(plan):
    job = BackgroundJob()
    run_scoring(job, CANDIDATE, plan)   # called directly, not on a thread
    return job.snapshot()


# --------------------------------------------------------- which jobs are used

def test_auto_plan_only_considers_unscored_jobs(monkeypatch, raw_mode):
    seen = {}

    def fake_get_jobs(unscored_only=False, **kw):
        seen["unscored_only"] = unscored_only
        return _jobs()

    monkeypatch.setattr(pipeline, "get_jobs", fake_get_jobs)
    monkeypatch.setattr(pipeline, "score_jobs", lambda *a, **k: iter([]))
    monkeypatch.setattr(pipeline, "update_scores", lambda r: None)

    _run(AUTO_PLAN)

    assert seen["unscored_only"] is True


def test_button_plan_considers_every_job(monkeypatch, raw_mode):
    seen = {}

    def fake_get_jobs(unscored_only=False, **kw):
        seen["unscored_only"] = unscored_only
        return _jobs()

    monkeypatch.setattr(pipeline, "get_jobs", fake_get_jobs)
    monkeypatch.setattr(pipeline, "score_jobs", lambda *a, **k: iter([]))
    monkeypatch.setattr(pipeline, "update_scores", lambda r: None)

    _run(BUTTON_PLAN)

    assert seen["unscored_only"] is False


# ------------------------------------------------------------- nothing to score

@pytest.mark.parametrize("plan", [AUTO_PLAN, BUTTON_PLAN])
def test_each_plan_reports_its_own_nothing_to_do_message(monkeypatch, raw_mode, plan):
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: pd.DataFrame())

    assert _run(plan).skip == plan.nothing_to_do


def test_jobs_without_descriptions_are_not_scored(monkeypatch, raw_mode):
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: _jobs(description=""))
    called = []
    monkeypatch.setattr(pipeline, "score_jobs", lambda *a, **k: called.append(1) or iter([]))

    snap = _run(AUTO_PLAN)

    assert called == []
    assert snap.skip == AUTO_PLAN.nothing_to_do


# ---------------------------------------------------------------- scoring loop

def test_results_are_persisted_per_batch_and_counted(monkeypatch, raw_mode):
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: _jobs(3))
    monkeypatch.setattr(pipeline, "score_jobs",
                         lambda *a, **k: iter([[{"id": "j0"}, {"id": "j1"}], [{"id": "j2"}]]))
    persisted = []
    monkeypatch.setattr(pipeline, "update_scores", persisted.append)

    snap = _run(AUTO_PLAN)

    # Two separate calls, not one at the end — a cancelled run keeps what it did.
    assert [len(batch) for batch in persisted] == [2, 1]
    assert snap.done == 3
    assert snap.total == 3


def test_empty_batches_are_skipped(monkeypatch, raw_mode):
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: _jobs(1))
    monkeypatch.setattr(pipeline, "score_jobs", lambda *a, **k: iter([[], None, [{"id": "j0"}]]))
    persisted = []
    monkeypatch.setattr(pipeline, "update_scores", persisted.append)

    assert _run(AUTO_PLAN).done == 1
    assert len(persisted) == 1


def test_the_candidate_summary_is_what_gets_scored_against(monkeypatch, raw_mode):
    seen = {}
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: _jobs(1))
    monkeypatch.setattr(pipeline, "score_jobs",
                         lambda summary, jobs, **k: seen.update(summary=summary) or iter([]))
    monkeypatch.setattr(pipeline, "update_scores", lambda r: None)

    _run(AUTO_PLAN)

    assert seen["summary"] == CANDIDATE["summary"]


# ------------------------------------------------------------- cancel plumbing

@pytest.mark.parametrize("plan, expects_event", [(AUTO_PLAN, False), (BUTTON_PLAN, True)])
def test_cancel_event_is_passed_only_for_cancellable_plans(monkeypatch, raw_mode,
                                                            plan, expects_event):
    seen = {}
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: _jobs(1))
    monkeypatch.setattr(pipeline, "score_jobs",
                         lambda *a, cancel_event=None, **k: seen.update(ev=cancel_event) or iter([]))
    monkeypatch.setattr(pipeline, "update_scores", lambda r: None)

    _run(plan)

    assert (seen["ev"] is not None) is expects_event


# ------------------------------------------------------------- mode routing

def test_structured_mode_routes_to_the_structured_path(monkeypatch):
    monkeypatch.setattr(pipeline, "scoring_mode", lambda: "structured")
    monkeypatch.setattr(pipeline, "extract_missing_job_requirements", lambda **kw: None)
    monkeypatch.setattr(pipeline, "load_resume_profile", lambda c, log_fn: None)

    # Bails at the resume-profile step, which only the structured path has.
    assert "structured resume profile" in _run(AUTO_PLAN).skip


def test_structured_mode_skips_when_no_jd_data_is_parseable(monkeypatch):
    monkeypatch.setattr(pipeline, "scoring_mode", lambda: "structured")
    monkeypatch.setattr(pipeline, "extract_missing_job_requirements", lambda **kw: None)
    monkeypatch.setattr(pipeline, "load_resume_profile", lambda c, log_fn: object())
    jobs = _jobs(2).assign(jd_extracted=[None, None])
    monkeypatch.setattr(pipeline, "get_jobs", lambda **kw: jobs)
    monkeypatch.setattr(pipeline, "parse_jd_extracted", lambda raw: None)

    assert _run(AUTO_PLAN).skip == "No jobs have structured JD data to score against yet."


# ----------------------------------------------------------------- ScanRequest

def test_scan_request_defaults_to_nothing_clicked():
    assert ScanRequest().any_clicked is False


@pytest.mark.parametrize("field", [
    "scan_all", "jobspy", "linkedin_login", "naukri", "company_boards", "jsearch",
])
def test_any_clicked_detects_each_button(field):
    assert ScanRequest(**{field: True}).any_clicked is True
