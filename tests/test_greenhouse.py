from scanner import greenhouse

_FIXTURE = {
    "jobs": [
        {
            "id": 12345,
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/12345",
            "title": "Backend Engineer",
            "location": {"name": "Remote - US"},
            "updated_at": "2026-07-01T00:00:00Z",
            "content": "<p>We build <b>things</b>.</p>",
        },
        {
            "id": 67890,
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/67890",
            "title": "Office Manager",
            "location": {"name": "New York, NY"},
            "updated_at": "2026-07-02T00:00:00Z",
            "content": "",
        },
    ]
}


def test_fetch_jobs_maps_rows(monkeypatch):
    monkeypatch.setattr(greenhouse, "fetch_json", lambda url, params: _FIXTURE)

    df = greenhouse.fetch_jobs("acme", "Acme Inc")

    assert list(df.columns) == [
        "id", "site", "job_url", "job_url_direct", "title", "company",
        "location", "date_posted", "is_remote", "description",
    ]
    assert len(df) == 2

    remote_row = df.iloc[0]
    assert remote_row["id"] == "gh-acme-12345"
    assert remote_row["site"] == "greenhouse"
    assert remote_row["company"] == "Acme Inc"
    assert bool(remote_row["is_remote"]) is True
    assert remote_row["description"] == "We build things ."

    onsite_row = df.iloc[1]
    assert onsite_row["id"] == "gh-acme-67890"
    assert bool(onsite_row["is_remote"]) is False
    assert onsite_row["description"] == ""


def test_fetch_jobs_empty(monkeypatch):
    monkeypatch.setattr(greenhouse, "fetch_json", lambda url, params: {"jobs": []})

    df = greenhouse.fetch_jobs("acme", "Acme Inc")

    assert len(df) == 0
