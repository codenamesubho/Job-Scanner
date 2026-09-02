import pandas as pd

from scanner import manual


def test_match_ats_api_greenhouse(stub_fetch_json):
    """A job embedded via Greenhouse's `for=`/`gh_jid=` style widget on a
    company's own domain fires a boards-api.greenhouse.io request — that
    alone should resolve it, regardless of the page URL's own shape."""
    stub_fetch_json(manual, {
        "id": 123,
        "title": "Backend Engineer",
        "absolute_url": "https://acme.com/careers/job?gh_jid=123",
        "location": {"name": "Remote"},
        "updated_at": "2025-01-01",
        "content": "<p>Do stuff</p>",
    })
    requested = [
        "https://acme.com/some-asset.js",
        "https://boards-api.greenhouse.io/v1/boards/acme/jobs/123",
        "https://analytics.example.com/px.gif",
    ]

    row = manual._match_ats_api(requested, "https://acme.com/careers/job?gh_jid=123")

    assert row["title"] == "Backend Engineer"
    assert row["company"] == "Acme"
    assert row["site"] == "greenhouse"


def test_match_ats_api_lever(stub_fetch_json):
    stub_fetch_json(manual, {
        "id": "9f86d081-884c",
        "text": "Site Reliability Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/9f86d081-884c",
        "categories": {"location": "Remote"},
        "workplaceType": "remote",
        "descriptionPlain": "Keep it running.",
    })
    requested = ["https://api.lever.co/v0/postings/acme/9f86d081-884c?mode=json"]

    row = manual._match_ats_api(requested, "https://acme.com/careers/sre")

    assert row["title"] == "Site Reliability Engineer"
    assert row["site"] == "lever"


def test_match_ats_api_ashby(monkeypatch):
    from scanner import ashby

    board_df = pd.DataFrame([{
        "id": "ash-acme-abc123",
        "job_url": "https://acme.com/careers/pm",
        "title": "Product Manager",
    }])
    monkeypatch.setattr(ashby, "fetch_jobs", lambda board, company_name: board_df)
    requested = ["https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true"]

    row = manual._match_ats_api(requested, "https://acme.com/careers/pm")

    assert row["title"] == "Product Manager"


def test_match_ats_api_returns_none_for_unrecognized_requests():
    requested = ["https://acme.com/careers.css", "https://cdn.example.com/logo.png"]

    assert manual._match_ats_api(requested, "https://acme.com/careers/job") is None


def test_match_ats_api_returns_none_on_empty_list():
    assert manual._match_ats_api([], "https://acme.com/careers/job") is None
