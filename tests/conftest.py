"""Shared pytest fixtures.

Before this existed, every test hand-rolled its own setup: `test_db_migrations.py`
called a local `_isolate_db(monkeypatch, tmp_path)` as the first line of all ten of
its tests, and the ATS scraper tests each repeated the same `fetch_json` stub.
The fixtures here are the one place that setup lives now.
"""
import pytest

from scanner import database


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point `database.DB_PATH` at a throwaway file for the duration of a test.

    `DB_PATH` is a module-level `Path` precisely so it can be rebound like this —
    every table is created lazily on the first `_connect()`, so moving the path is
    enough to get a clean database, with no fixture teardown needed.

    Yields the path, for tests that need to inspect the file itself.
    """
    db_path = tmp_path / "test_jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    return db_path


@pytest.fixture
def stub_fetch_json(monkeypatch):
    """Stub the `fetch_json()` an ATS module imported from `ats_common`.

    The three public-board scrapers (greenhouse/lever/ashby) all reach the network
    through the same `fetch_json(url, params)` helper, so they all stub it the same
    way. Their *assertions* differ substantially (each maps a different payload
    shape), so only the stubbing is shared — the tests stay separate and readable.

    Usage: `stub_fetch_json(greenhouse, _FIXTURE)`
    """
    def _stub(module, payload):
        monkeypatch.setattr(module, "fetch_json", lambda url, params: payload)

    return _stub
