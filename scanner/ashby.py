"""Fetch jobs from a company's public Ashby job board.

GET https://api.ashbyhq.com/posting-api/job-board/{job_board_name} is a free,
unauthenticated JSON endpoint — no login, no Playwright. Ashby doesn't return
a job `id`, so one is derived from a hash of the job's URL.
"""

import hashlib

import pandas as pd

from .ats_common import fetch_json, html_to_text, build_job_row

_BASE_URL = "https://api.ashbyhq.com/posting-api/job-board/{board}"


def fetch_jobs(job_board_name: str, company_name: str) -> pd.DataFrame:
    jobs = fetch_json(
        _BASE_URL.format(board=job_board_name),
        params={"includeCompensation": "true"},
    ).get("jobs", [])

    rows = []
    for job in jobs:
        job_url = job.get("jobUrl", "")
        digest = hashlib.md5(job_url.encode()).hexdigest()[:12]
        location = job.get("location", "") or ""
        workplace_type = job.get("workplaceType", "") or ""

        description = job.get("descriptionPlain", "")
        if not description and job.get("descriptionHtml"):
            description = html_to_text(job["descriptionHtml"])

        rows.append(build_job_row(
            id=f"ash-{job_board_name}-{digest}",
            site="ashby",
            job_url=job_url,
            job_url_direct=job.get("applyUrl") or job_url,
            title=job.get("title", ""),
            company=company_name,
            location=location,
            job_type=job.get("employmentType"),
            job_function=job.get("team") or job.get("department"),
            date_posted=job.get("publishedAt"),
            description=description,
            is_remote=bool(job.get("isRemote")) or workplace_type.lower() == "remote",
        ))

    return pd.DataFrame(rows)
