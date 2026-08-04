"""Tests for the shared scan kernel (`scanner/search.py`).

This logic previously existed twice — once for Streamlit, once for cron — and had
no tests in either place, despite owning "did we lose a keyword's results?" and
"how many of these were new?".
"""
import dataclasses

import pandas as pd
import pytest

from scanner import database
from scanner.search import (
    ScanResult, SearchCriteria, prefixed_logger, run_keyword_scan,
)


def _job_frame(*ids):
    return pd.DataFrame([
        {"id": i, "site": "test", "title": f"Title {i}", "company": "Acme",
         "description": "A sufficiently long description."}
        for i in ids
    ])


# ---------------------------------------------------------------- SearchCriteria

@pytest.mark.parametrize("raw, expected", [
    ("backend engineer", ["backend engineer"]),
    ("a,b,c", ["a", "b", "c"]),
    ("  spaced  ,  out  ", ["spaced", "out"]),
    ("trailing,", ["trailing"]),
    ("a,,b", ["a", "b"]),
    ("", []),
    ("   ", []),
    (",,,", []),
])
def test_keyword_list_splits_and_cleans(raw, expected):
    assert SearchCriteria(raw, "Remote").keyword_list() == expected


def test_search_criteria_is_frozen():
    """A scan must not be able to mutate the criteria it was handed."""
    criteria = SearchCriteria("a", "Remote")
    with pytest.raises(dataclasses.FrozenInstanceError):
        criteria.keywords = "b"


# -------------------------------------------------------------------- ScanResult

def test_scan_results_add_up():
    assert sum([ScanResult(3, 1), ScanResult(4, 2)], ScanResult()) == ScanResult(7, 3)


# --------------------------------------------------------------- run_keyword_scan

def test_runs_once_per_keyword_and_saves_everything(isolated_db):
    calls = []

    def scan_fn(kw, location, results_wanted, hours_old):
        calls.append((kw, location, results_wanted, hours_old))
        return _job_frame(f"{kw}-1")

    result = run_keyword_scan(scan_fn, SearchCriteria("alpha,beta", "Remote", 10, 24),
                               log_fn=lambda m: None)

    assert calls == [("alpha", "Remote", 10, 24), ("beta", "Remote", 10, 24)]
    assert result == ScanResult(found=2, new=2)


def test_a_failing_keyword_does_not_lose_the_others(isolated_db):
    """One bad search term must not abort the whole source."""
    def scan_fn(kw, location, results_wanted, hours_old):
        if kw == "boom":
            raise RuntimeError("source exploded")
        return _job_frame(f"{kw}-1")

    logs = []
    result = run_keyword_scan(scan_fn, SearchCriteria("good,boom,alsogood", "Remote"),
                               log_fn=logs.append)

    assert result.found == 2
    assert any("boom' FAILED: source exploded" in line for line in logs)


def test_reports_zero_when_every_keyword_fails(isolated_db):
    def scan_fn(kw, location, results_wanted, hours_old):
        raise RuntimeError("down")

    result = run_keyword_scan(scan_fn, SearchCriteria("a,b", "Remote"), log_fn=lambda m: None)

    assert result == ScanResult(0, 0)


def test_empty_frames_are_not_saved(isolated_db):
    result = run_keyword_scan(lambda *a, **k: pd.DataFrame(),
                               SearchCriteria("a", "Remote"), log_fn=lambda m: None)

    assert result == ScanResult(0, 0)
    assert database.get_jobs().empty


def test_counts_already_known_jobs_as_found_but_not_new(isolated_db):
    """Re-scanning a known job still counts as 'found' — only 'new' should drop."""
    scan_fn = lambda *a, **k: _job_frame("dupe-1")  # noqa: E731

    first  = run_keyword_scan(scan_fn, SearchCriteria("a", "Remote"), log_fn=lambda m: None)
    second = run_keyword_scan(scan_fn, SearchCriteria("a", "Remote"), log_fn=lambda m: None)

    assert first == ScanResult(found=1, new=1)
    assert second == ScanResult(found=1, new=0)


def test_no_keywords_short_circuits_without_calling_the_source(isolated_db):
    called = []
    run_keyword_scan(lambda *a, **k: called.append(1),
                      SearchCriteria("  ,, ", "Remote"), log_fn=lambda m: None)

    assert called == []


def test_extra_kwargs_reach_the_source(isolated_db):
    """LinkedIn passes on_page_done through this path."""
    seen = {}

    def scan_fn(kw, location, results_wanted, hours_old, on_page_done=None):
        seen["cb"] = on_page_done
        return pd.DataFrame()

    sentinel = object()
    run_keyword_scan(scan_fn, SearchCriteria("a", "Remote"), log_fn=lambda m: None,
                      on_page_done=sentinel)

    assert seen["cb"] is sentinel


def test_progress_is_reported_and_is_optional(isolated_db):
    seen = []
    run_keyword_scan(lambda *a, **k: pd.DataFrame(), SearchCriteria("a,b", "Remote"),
                      log_fn=lambda m: None, progress_fn=lambda f, t: seen.append(f))
    assert seen == [0.0, 0.5, 0.5, 1.0]

    # Omitting progress_fn entirely (as cron does) must not raise.
    run_keyword_scan(lambda *a, **k: pd.DataFrame(), SearchCriteria("a", "Remote"),
                      log_fn=lambda m: None)


# -------------------------------------------------------------- prefixed_logger

def test_prefixed_logger_tags_each_line():
    lines = []
    log = prefixed_logger(lines.append, "Naukri")
    log("hello")

    assert lines == ["  [Naukri] hello"]
