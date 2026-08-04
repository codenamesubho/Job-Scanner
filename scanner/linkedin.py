from jobspy import scrape_jobs
from jobspy.linkedin import LinkedIn
from jobspy.model import Location, Country
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import requests


def _patched_get_location(self, metadata_card):
    """Patched version of LinkedIn._get_location that skips unrecognised countries."""

    location = Location(country=Country.from_string(self.country))
    if metadata_card is not None:
        location_tag = metadata_card.find("span", class_="job-search-card__location")
        location_string = location_tag.text.strip() if location_tag else "N/A"
        parts = location_string.split(", ")
        if len(parts) == 2:
            city, state = parts
            location = Location(city=city, state=state, country=Country.from_string(self.country))
        elif len(parts) == 3:
            city, state, country_str = parts
            try:
                country = Country.from_string(country_str)
            except ValueError:
                country = Country.from_string(self.country)
            location = Location(city=city, state=state, country=country)
    return location


# Monkey-patch the library so unrecognised countries don't crash the scraper.
LinkedIn._get_location = _patched_get_location


# jobspy's Indeed scraper requires country_indeed to be one of its known
# country names/aliases (a city string raises ValueError) — matching is
# case-insensitive. "worldwide" is a valid value, used as the safe fallback.
_INDEED_COUNTRY_ALIASES = {
    "usa": "usa", "us": "usa", "united states": "usa", "america": "usa",
    "uk": "uk", "united kingdom": "uk", "england": "uk",
    "india": "india", "remote": "worldwide", "worldwide": "worldwide",
}


def _infer_indeed_country(location: str) -> str:
    env_override = os.getenv("INDEED_COUNTRY", "").strip().lower()
    if env_override:
        return env_override

    loc = (location or "").lower().strip()
    if "remote" in loc or not loc:
        return "worldwide"
    last_segment = loc.split(",")[-1].strip()
    return _INDEED_COUNTRY_ALIASES.get(last_segment, "worldwide")


def search_jobs(
    keywords: str,
    location: str,
    results_wanted: int = 25,
    hours_old: int = 72,
) -> pd.DataFrame:
    jobs = scrape_jobs(
        site_name=os.getenv("JOBS_SOURCE", "linkedin").split(","),
        keywords=keywords,
        location=location,
        search_term=keywords,
        results_wanted=results_wanted,
        hours_old=hours_old,
        linkedin_fetch_description=True,
        country_indeed=_infer_indeed_country(location),
    )

    # LinkedIn/jobspy treats location="Remote" as a weak global fallback, not
    # a real geo-filter — confirmed live: a "Remote" search returns mostly
    # city-based jobs (only ~1 in 5 actually flagged remote). jobspy already
    # computes a per-job is_remote signal (from title/description/location
    # together), so when the caller clearly wants remote-only, use that
    # signal to filter down rather than trusting LinkedIn's own results.
    if "remote" in (location or "").lower() and not jobs.empty and "is_remote" in jobs.columns:
        jobs = jobs[jobs["is_remote"] == True]  # noqa: E712

    return jobs


def fetch_missing_description(job_id: str) -> dict:
    """Re-fetch title/company/description straight from a LinkedIn job's
    public view page (no auth, no headless browser) — used to backfill rows
    where jobspy's own per-job description fetch failed during the scan
    (transient network error, rate limiting, etc). Returns {} on any
    failure so callers can just skip the row."""
    url = f"https://www.linkedin.com/jobs/view/{job_id}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return {}
    if "linkedin.com/signup" in resp.url:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    div = soup.find("div", class_=lambda x: x and "show-more-less-html__markup" in x)
    description = div.get_text(" ", strip=True) if div else None

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None

    company_tag = soup.find("a", class_=lambda x: x and "topcard__org-name-link" in x)
    company = company_tag.get_text(strip=True) if company_tag else None

    return {"description": description, "title": title, "company": company}


def backfill_missing_descriptions(log_fn=None) -> int:
    """Find LinkedIn jobs stored without a description and try to recover
    them via fetch_missing_description(). Returns the count actually fixed."""
    from . import database

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    jobs = database.get_jobs()
    if jobs.empty:
        return 0
    missing = jobs[
        (jobs["site"] == "linkedin")
        & (jobs["description"].isna() | (jobs["description"] == ""))
    ]
    if missing.empty:
        _log("No jobs missing a description.")
        return 0

    _log(f"Attempting to backfill {len(missing)} job(s) missing a description…")
    fixed = 0
    for _, row in missing.iterrows():
        job_id = row["id"].removeprefix("li-")
        details = fetch_missing_description(job_id)
        if details.get("description"):
            database.update_job_fields(row["id"], details)
            fixed += 1
            _log(f"  fixed: {row['id']} ({details.get('title') or row['title']})")
        else:
            _log(f"  still unavailable: {row['id']}")
        time.sleep(0.5)  # polite spacing between requests to LinkedIn's public page
    _log(f"Backfill done — recovered {fixed}/{len(missing)}.")
    return fixed


def display_jobs(jobs: pd.DataFrame) -> None:
    if jobs.empty:
        print("No jobs found.")
        return

    cols = ["title", "company", "location", "date_posted", "job_url"]
    display_cols = [c for c in cols if c in jobs.columns]
    print(jobs[display_cols].to_string(index=False))
    print(f"\nTotal: {len(jobs)} job(s) found.")
