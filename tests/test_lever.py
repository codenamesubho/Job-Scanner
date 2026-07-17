import pandas as pd

from scanner import lever

_FIXTURE = [
    {
        "id": "lv-1",
        "hostedUrl": "https://jobs.lever.co/acme/lv-1",
        "applyUrl": "https://jobs.lever.co/acme/lv-1/apply",
        "text": "Site Reliability Engineer",
        "categories": {"location": "Remote", "commitment": "Full-time", "team": "Infra"},
        "workplaceType": "remote",
        "createdAt": "1735689600000",  # 2025-01-01T00:00:00Z in ms
        "descriptionPlain": "Keep things running.",
        "salaryRange": {"min": 120000, "max": 180000, "currency": "USD"},
    },
    {
        "id": "lv-2",
        "hostedUrl": "https://jobs.lever.co/acme/lv-2",
        "text": "Office Coordinator",
        "categories": {"location": "Austin, TX", "commitment": "Full-time", "team": "Ops"},
        "workplaceType": "onsite",
        "createdAt": "not-a-number",
        "descriptionPlain": "Keep the office running.",
        "salaryRange": {},
    },
]


def test_fetch_jobs_maps_rows(monkeypatch):
    monkeypatch.setattr(lever, "fetch_json", lambda url, params: _FIXTURE)

    df = lever.fetch_jobs("acme", "Acme Inc")

    assert len(df) == 2
    assert set(["job_type", "job_function", "min_amount", "max_amount", "currency"]) <= set(df.columns)

    remote_row = df.iloc[0]
    assert remote_row["id"] == "lv-acme-lv-1"
    assert remote_row["job_url_direct"] == "https://jobs.lever.co/acme/lv-1/apply"
    assert bool(remote_row["is_remote"]) is True
    assert remote_row["date_posted"] == "2025-01-01"
    assert remote_row["min_amount"] == 120000
    assert remote_row["currency"] == "USD"

    onsite_row = df.iloc[1]
    assert onsite_row["job_url_direct"] == "https://jobs.lever.co/acme/lv-2"  # falls back to hostedUrl
    assert bool(onsite_row["is_remote"]) is False
    assert onsite_row["date_posted"] is None  # bad createdAt -> None, not a crash
    assert pd.isna(onsite_row["min_amount"])
