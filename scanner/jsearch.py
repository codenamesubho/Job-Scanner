"""Search jobs via JSearch (RapidAPI), an aggregator covering LinkedIn,
Indeed, Glassdoor, ZipRecruiter and more through one keyed HTTP endpoint.

Reads JSEARCH_API_KEY directly from the environment at call time — same
pattern scanner/llm.py uses for CLAUDE_API_KEY — not routed through
scanner/config.py, which is CLI-only.
"""

import math
import os

import requests
import pandas as pd

# /search (v1) 404s under some RapidAPI subscription plans — /search-v2 is the
# endpoint actually reachable on those plans. Confirmed live: same query params
# (query, num_pages, date_posted) work on both, but v2 nests the job list one
# level deeper (data.jobs, plus a data.cursor for pagination) instead of
# putting it directly under "data", and drops job_salary_currency (v2 has
# job_salary/job_salary_period/job_salary_string instead — not parsed here).
_URL = "https://jsearch.p.rapidapi.com/search-v2"

_JOBS_PER_PAGE = 10  # JSearch's fixed page size
_MIN_PAGES     = 3   # always request at least this many pages per keyword
_MAX_PAGES     = 4   # keep a ceiling so one keyword can't blow through RapidAPI quota


def _map_hours_to_date_posted(hours_old: int) -> str:
    if hours_old <= 72:
        return "3days"
    if hours_old <= 168:
        return "week"
    if hours_old <= 720:
        return "month"
    return "all"


def search_jobs(
    keywords: str,
    location: str,
    results_wanted: int = 25,
    hours_old: int = 72,
) -> pd.DataFrame:
    api_key = os.environ["JSEARCH_API_KEY"]
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    num_pages = min(_MAX_PAGES, max(_MIN_PAGES, math.ceil(results_wanted / _JOBS_PER_PAGE)))
    params = {
        "query": f"{keywords} in {location}",
        "num_pages": str(num_pages),
        "date_posted": _map_hours_to_date_posted(hours_old),
    }

    # A multi-page request takes JSearch proportionally longer to fulfill
    # server-side (it's scraping upstream sources live) — 30s was enough for
    # a single page but timed out at num_pages=3, so allow more headroom.
    resp = requests.get(_URL, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    jobs = resp.json().get("data", {}).get("jobs", [])

    rows = []
    for job in jobs:
        location_str = ", ".join(
            part for part in (
                job.get("job_city"), job.get("job_state"), job.get("job_country"),
            ) if part
        )
        rows.append({
            "id":               f"jsearch-{job['job_id']}",
            "site":             job.get("job_publisher", "jsearch"),
            "job_url":          job.get("job_apply_link", ""),
            "job_url_direct":   job.get("job_apply_link", ""),
            "title":            job.get("job_title", ""),
            "company":          job.get("employer_name", ""),
            "location":         location_str,
            "job_type":         job.get("job_employment_type"),
            "description":      job.get("job_description", ""),
            "is_remote":        bool(job.get("job_is_remote", False)),
            "date_posted":      job.get("job_posted_at_datetime_utc"),
            "min_amount":       job.get("job_min_salary"),
            "max_amount":       job.get("job_max_salary"),
            "currency":         job.get("job_salary_currency"),
            "company_url":      job.get("employer_website"),
        })

    # Not trimmed to results_wanted: num_pages already guarantees at least
    # _MIN_PAGES pages are requested, so returning everything found (rather
    # than cutting back down to results_wanted) is the point of that over-fetch.
    return pd.DataFrame(rows)
