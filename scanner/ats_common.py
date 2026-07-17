"""Shared helpers for the public, unauthenticated ATS job-board scrapers
(greenhouse.py, ashby.py, lever.py). Each hits a free JSON endpoint, strips
HTML from the description, and builds the same row shape for the jobs table.
"""

import requests
from bs4 import BeautifulSoup


def fetch_json(url: str, params: dict, timeout: int = 30):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def build_job_row(
    *,
    id: str,
    site: str,
    job_url: str,
    job_url_direct: str,
    title: str,
    company: str,
    location: str,
    date_posted,
    is_remote: bool,
    description: str,
    **extra,
) -> dict:
    """Base row shape shared by all ATS scrapers. Callers pass source-specific
    fields (job_type, job_function, min_amount, ...) as extra kwargs — only
    fields a source actually has are included, matching each source's original
    column set."""
    return {
        "id": id,
        "site": site,
        "job_url": job_url,
        "job_url_direct": job_url_direct,
        "title": title,
        "company": company,
        "location": location,
        "date_posted": date_posted,
        "is_remote": is_remote,
        "description": description,
        **extra,
    }
