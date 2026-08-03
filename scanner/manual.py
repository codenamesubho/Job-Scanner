"""Manually add a single job to the database from a pasted URL.

Recognizes known ATS board URLs (Greenhouse/Lever/Ashby — the same public
APIs the scan sources already use) and LinkedIn job-view URLs (the same
public HTML scrape linkedin.py already uses to backfill missing
descriptions). Anything else falls back to a plain HTTP GET + page-text
extraction.
"""
import hashlib
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .ats_common import fetch_json, html_to_text, build_job_row
from .database import save_jobs

_GREENHOUSE_RE = re.compile(r"greenhouse\.io/([^/]+)/jobs/(\d+)")
_LEVER_RE      = re.compile(r"lever\.co/([^/]+)/([0-9a-fA-F-]+)")
_ASHBY_RE      = re.compile(r"ashbyhq\.com/([^/]+)/([0-9a-fA-F-]+)")
_LINKEDIN_RE   = re.compile(r"linkedin\.com/jobs/view/(\d+)")


def _slug_to_name(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _from_greenhouse(token: str, job_id: str, url: str) -> dict | None:
    try:
        data = fetch_json(
            f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{job_id}",
            params={"content": "true"},
        )
    except Exception:
        return None
    if not data or not data.get("title"):
        return None
    location = (data.get("location") or {}).get("name", "") or ""
    return build_job_row(
        id=f"gh-{token}-{data['id']}",
        site="greenhouse",
        job_url=data.get("absolute_url", url),
        job_url_direct=data.get("absolute_url", url),
        title=data.get("title", ""),
        company=_slug_to_name(token),
        location=location,
        date_posted=data.get("updated_at"),
        is_remote="remote" in location.lower(),
        description=html_to_text(data.get("content") or ""),
    )


def _from_lever(company: str, job_id: str, url: str) -> dict | None:
    try:
        data = fetch_json(
            f"https://api.lever.co/v0/postings/{company}/{job_id}",
            params={"mode": "json"},
        )
    except Exception:
        return None
    if not data or not data.get("text"):
        return None
    categories     = data.get("categories") or {}
    location       = categories.get("location", "") or ""
    workplace_type = data.get("workplaceType", "") or ""
    salary         = data.get("salaryRange") or {}
    return build_job_row(
        id=f"lv-{company}-{data['id']}",
        site="lever",
        job_url=data.get("hostedUrl", url),
        job_url_direct=data.get("applyUrl") or data.get("hostedUrl", url),
        title=data.get("text", ""),
        company=_slug_to_name(company),
        location=location,
        job_type=categories.get("commitment", ""),
        job_function=categories.get("team", ""),
        description=data.get("descriptionPlain", ""),
        is_remote=workplace_type.lower() == "remote" or "remote" in location.lower(),
        min_amount=salary.get("min"),
        max_amount=salary.get("max"),
        currency=salary.get("currency"),
    )


def _from_ashby(board: str, url: str) -> dict | None:
    """Ashby has no per-job endpoint — fetch the whole board and match by URL."""
    from .ashby import fetch_jobs as ashby_fetch_jobs
    try:
        df = ashby_fetch_jobs(board, _slug_to_name(board))
    except Exception:
        return None
    if df.empty:
        return None
    match = df[df["job_url"].str.rstrip("/") == url.rstrip("/")]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _from_linkedin(job_id: str, url: str) -> dict | None:
    from .linkedin import fetch_missing_description
    try:
        data = fetch_missing_description(job_id)
    except Exception:
        return None
    if not data.get("description"):
        return None
    return build_job_row(
        id=f"li-{job_id}",
        site="linkedin",
        job_url=url,
        job_url_direct="",
        title=data.get("title") or "",
        company=data.get("company") or "",
        location="",
        date_posted=None,
        is_remote=False,
        description=data.get("description") or "",
    )


def _from_generic(url: str) -> dict | None:
    """Last-resort fallback for any URL that doesn't match a known ATS/site:
    plain HTTP GET + whole-page text as the description."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    soup        = BeautifulSoup(resp.text, "html.parser")
    title       = soup.title.get_text(strip=True) if soup.title else ""
    description = html_to_text(resp.text)
    if len(description) < 50:
        return None
    job_id = f"manual-{hashlib.md5(url.encode()).hexdigest()[:12]}"
    return build_job_row(
        id=job_id,
        site="manual",
        job_url=url,
        job_url_direct=url,
        title=title,
        company="",
        location="",
        date_posted=None,
        is_remote=False,
        description=description,
    )


def add_job_by_url(url: str) -> tuple[bool, str]:
    """Fetch and save a single job posting from a pasted URL.

    Tries known ATS/LinkedIn patterns first (structured, reliable), then
    falls back to a generic page-text scrape. Returns (success, message)
    for the caller to display.
    """
    url = url.strip()
    if not url:
        return False, "Please paste a job URL."

    row = None
    if m := _GREENHOUSE_RE.search(url):
        row = _from_greenhouse(m.group(1), m.group(2), url)
    elif m := _LEVER_RE.search(url):
        row = _from_lever(m.group(1), m.group(2), url)
    elif m := _ASHBY_RE.search(url):
        row = _from_ashby(m.group(1), url)
    elif m := _LINKEDIN_RE.search(url):
        row = _from_linkedin(m.group(1), url)

    if row is None:
        row = _from_generic(url)

    if row is None:
        return False, ("Couldn't fetch a description from that URL — the page may "
                        "require login or block automated requests.")

    new_count = save_jobs(pd.DataFrame([row]))
    company = row.get("company") or "unknown company"
    if new_count:
        return True, f'Added "{row["title"] or url}" at {company}.'
    return True, "That job was already in the database — last-seen updated."
