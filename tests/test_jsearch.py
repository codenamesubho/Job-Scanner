from scanner import jsearch

_FIXTURE = {
    "data": {
        "jobs": [
            {
                "job_id": "abc123",
                "job_publisher": "LinkedIn",
                "job_apply_link": "https://example.com/apply/abc123",
                "job_title": "Platform Engineer",
                "employer_name": "Acme Inc",
                "job_city": "Austin",
                "job_state": "TX",
                "job_country": "US",
                "job_employment_type": "FULLTIME",
                "job_description": "Build the platform.",
                "job_is_remote": True,
                "job_posted_at_datetime_utc": "2026-07-01T00:00:00Z",
                "job_min_salary": 100000,
                "job_max_salary": 150000,
                "job_salary_currency": "USD",
                "employer_website": "https://acme.example.com",
            }
        ]
    }
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_search_jobs_maps_rows(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(
        jsearch.requests, "get", lambda url, headers, params, timeout: _FakeResponse(_FIXTURE)
    )

    df = jsearch.search_jobs("platform engineer", "Austin, TX")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["id"] == "jsearch-abc123"
    assert row["location"] == "Austin, TX, US"
    assert bool(row["is_remote"]) is True
    assert row["min_amount"] == 100000
    assert row["company"] == "Acme Inc"


def test_map_hours_to_date_posted():
    assert jsearch._map_hours_to_date_posted(24) == "3days"
    assert jsearch._map_hours_to_date_posted(72) == "3days"
    assert jsearch._map_hours_to_date_posted(100) == "week"
    assert jsearch._map_hours_to_date_posted(200) == "month"
    assert jsearch._map_hours_to_date_posted(10_000) == "all"
