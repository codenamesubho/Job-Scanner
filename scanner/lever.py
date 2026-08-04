"""Fetch jobs from a company's public Lever postings board.

GET https://api.lever.co/v0/postings/{company}?mode=json is a free,
unauthenticated JSON endpoint — no login, no Playwright.
"""

import datetime as dt

import pandas as pd

from .ats_common import fetch_json, build_job_row

_BASE_URL = "https://api.lever.co/v0/postings/{company}"


def fetch_jobs(company: str, company_name: str) -> pd.DataFrame:
    jobs = fetch_json(_BASE_URL.format(company=company), params={"mode": "json"})

    rows = []
    for job in jobs:
        categories = job.get("categories") or {}
        location = categories.get("location", "") or ""
        workplace_type = job.get("workplaceType", "") or ""
        salary = job.get("salaryRange") or {}

        created_at = job.get("createdAt")
        date_posted = None
        if created_at:
            try:
                # fromtimestamp(..., dt.UTC) rather than the deprecated
                # utcfromtimestamp(); the result is tz-aware but strftime()
                # renders the same date string.
                date_posted = dt.datetime.fromtimestamp(
                    int(created_at) / 1000, dt.UTC
                ).strftime("%Y-%m-%d")
            except (ValueError, TypeError, OSError):
                date_posted = None

        rows.append(build_job_row(
            id=f"lv-{company}-{job['id']}",
            site="lever",
            job_url=job.get("hostedUrl", ""),
            job_url_direct=job.get("applyUrl") or job.get("hostedUrl", ""),
            title=job.get("text", ""),
            company=company_name,
            location=location,
            job_type=categories.get("commitment", ""),
            job_function=categories.get("team", ""),
            date_posted=date_posted,
            description=job.get("descriptionPlain", ""),
            is_remote=workplace_type.lower() == "remote" or "remote" in location.lower(),
            min_amount=salary.get("min"),
            max_amount=salary.get("max"),
            currency=salary.get("currency"),
        ))

    return pd.DataFrame(rows)
