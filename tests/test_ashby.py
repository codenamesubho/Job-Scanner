import hashlib

from scanner import ashby

_FIXTURE = {
    "jobs": [
        {
            "jobUrl": "https://jobs.ashbyhq.com/acme/abc",
            "applyUrl": "https://jobs.ashbyhq.com/acme/abc/apply",
            "title": "Data Engineer",
            "location": "Remote",
            "workplaceType": "Remote",
            "employmentType": "FullTime",
            "team": "Data",
            "publishedAt": "2026-07-01T00:00:00Z",
            "descriptionPlain": "Plain text description.",
            "isRemote": True,
        },
        {
            "jobUrl": "https://jobs.ashbyhq.com/acme/def",
            "title": "Recruiter",
            "location": "San Francisco",
            "workplaceType": "Onsite",
            "employmentType": "FullTime",
            "department": "People",
            "publishedAt": "2026-07-02T00:00:00Z",
            "descriptionHtml": "<p>Hire <i>great</i> people.</p>",
            "isRemote": False,
        },
    ]
}


def test_fetch_jobs_maps_rows(stub_fetch_json):
    stub_fetch_json(ashby, _FIXTURE)

    df = ashby.fetch_jobs("acme", "Acme Inc")

    assert len(df) == 2
    assert "job_type" in df.columns and "job_function" in df.columns

    plain_row = df.iloc[0]
    expected_digest = hashlib.md5(b"https://jobs.ashbyhq.com/acme/abc").hexdigest()[:12]
    assert plain_row["id"] == f"ash-acme-{expected_digest}"
    assert plain_row["job_url_direct"] == "https://jobs.ashbyhq.com/acme/abc/apply"
    assert bool(plain_row["is_remote"]) is True
    assert plain_row["description"] == "Plain text description."
    assert plain_row["job_function"] == "Data"

    html_row = df.iloc[1]
    assert html_row["job_url_direct"] == "https://jobs.ashbyhq.com/acme/def"  # falls back to jobUrl
    assert bool(html_row["is_remote"]) is False
    assert html_row["description"] == "Hire great people."
    assert html_row["job_function"] == "People"  # falls back to department
