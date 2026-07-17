"""Fetch jobs from a company's public Greenhouse job board.

GET https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true
is a free, unauthenticated JSON endpoint — no login, no Playwright.
"""

import pandas as pd

from .ats_common import fetch_json, html_to_text, build_job_row

_BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def fetch_jobs(board_token: str, company_name: str) -> pd.DataFrame:
    jobs = fetch_json(
        _BASE_URL.format(token=board_token),
        params={"content": "true"},
    ).get("jobs", [])

    rows = []
    for job in jobs:
        location = (job.get("location") or {}).get("name", "") or ""
        rows.append(build_job_row(
            id=f"gh-{board_token}-{job['id']}",
            site="greenhouse",
            job_url=job.get("absolute_url", ""),
            job_url_direct=job.get("absolute_url", ""),
            title=job.get("title", ""),
            company=company_name,
            location=location,
            date_posted=job.get("updated_at"),
            is_remote="remote" in location.lower(),
            description=html_to_text(job.get("content") or ""),
        ))

    return pd.DataFrame(rows)
