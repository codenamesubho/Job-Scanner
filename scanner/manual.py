"""Manually add a single job to the database from a pasted URL.

Recognizes known ATS board URLs (Greenhouse/Lever/Ashby — the same public
APIs the scan sources already use) and LinkedIn job-view URLs (the same
public HTML scrape linkedin.py already uses to backfill missing
descriptions) directly, with no browser needed.

A growing number of companies instead embed one of these same ATS's job
data on their own careers domain (query param, iframe, custom widget —
the embed style varies per company and per ATS release, so matching it by
URL shape doesn't scale). Whatever the embed style, the page's JS still has
to call the ATS's own public API to get the job data — so instead of
special-casing each embed style, `_sniff_and_match()` renders the page with
a real browser and matches ANY request it makes against the same handful of
known ATS API endpoints. This one mechanism covers every embed style for
every supported ATS, present and future, without new per-site code.
"""
import datetime as dt
import hashlib
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from .ats_common import fetch_json, html_to_text, build_job_row
from .database import save_jobs

# Tags that carry no job-description content on a generic page (nav/chrome/
# scripting) — stripped before the remaining HTML is converted to Markdown.
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "iframe", "button"]


def _clean_html_to_markdown(html: str) -> str:
    """Strip page chrome (nav/header/footer/scripts) from raw page HTML and
    convert what's left to Markdown, rather than flattening everything to
    one plain-text blob. Keeps headings and bold survive as `#`/`**text**`
    instead of vanishing into the surrounding word soup — the same
    structure a real job description relies on to separate e.g.
    "Requirements" from "Nice to have"."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    md = markdownify(str(soup), heading_style="ATX", strip=["img"])
    return re.sub(r"\n{3,}", "\n\n", md).strip()

# Canonical hosted-board URLs — one pattern per ATS platform, matched
# directly against the pasted URL with no browser needed.
_GREENHOUSE_RE = re.compile(r"greenhouse\.io/([^/]+)/jobs/(\d+)")
_LEVER_RE      = re.compile(r"lever\.co/([^/]+)/([0-9a-fA-F-]+)")
_ASHBY_RE      = re.compile(r"ashbyhq\.com/([^/]+)/([0-9a-fA-F-]+)")
_LINKEDIN_RE   = re.compile(r"linkedin\.com/jobs/view/(\d+)")

# The same ATS platforms' underlying public API endpoints — matched against
# every request a rendered page makes, to catch that same job data however
# a third-party site embeds it (see _sniff_and_match).
_GREENHOUSE_API_RE = re.compile(r"boards-api\.greenhouse\.io/v1/boards/([^/]+)/jobs/(\d+)")
_LEVER_API_RE      = re.compile(r"api\.lever\.co/v0/postings/([^/]+)/([0-9a-fA-F-]+)")
_ASHBY_API_RE      = re.compile(r"api\.ashbyhq\.com/posting-api/job-board/([^/?]+)")


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
    created_at     = data.get("createdAt")
    date_posted    = None
    if created_at:
        try:
            date_posted = dt.datetime.utcfromtimestamp(int(created_at) / 1000).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            date_posted = None
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
        date_posted=date_posted,
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
    description = _clean_html_to_markdown(resp.text)
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


def _match_ats_api(requested_urls: list[str], page_url: str) -> dict | None:
    """Check a list of URLs a rendered page requested against the known ATS
    API endpoints and, on the first match, delegate to that ATS's own
    fetcher for the real structured data. Pure/no I/O beyond the fetcher
    calls, so this is unit-testable with a fake URL list — no browser
    needed in tests."""
    for req_url in requested_urls:
        if m := _GREENHOUSE_API_RE.search(req_url):
            if row := _from_greenhouse(m.group(1), m.group(2), page_url):
                return row
        elif m := _LEVER_API_RE.search(req_url):
            if row := _from_lever(m.group(1), m.group(2), page_url):
                return row
        elif m := _ASHBY_API_RE.search(req_url):
            if row := _from_ashby(m.group(1), page_url):
                return row
    return None


def _sniff_and_match(url: str) -> dict | None:
    """Render `url` with a real browser, capture every request the page
    makes while loading, and match those against known ATS APIs
    (_match_ats_api) — this is what catches a job embedded on a company's
    own domain regardless of the embed style, without needing a new rule
    per company/style. If nothing matches a known ATS, fall back to the
    rendered page's own HTML, cleaned and converted to Markdown
    (_clean_html_to_markdown) — still an improvement over a raw HTTP GET for
    a client-rendered (JS-only) page, since it reflects what a real visitor
    sees rather than an empty page shell, with headings/bold intact rather
    than flattened into one run-on paragraph.
    Returns None (never raises) on any Playwright failure, so callers can
    fall through to the plain-HTTP _from_generic()."""
    from playwright.sync_api import sync_playwright
    from .browser import launch_stealth_browser

    requested: list[str] = []
    try:
        with sync_playwright() as p:
            browser, context = launch_stealth_browser(p, headless=True)
            page = context.new_page()
            page.on("request", lambda req: requested.append(req.url))
            try:
                page.goto(url, timeout=20000, wait_until="networkidle")
            except Exception:
                pass  # match whatever fired before the timeout/navigation error
            if row := _match_ats_api(requested, url):
                browser.close()
                return row
            title = page.title()
            html = page.content()
            browser.close()
    except Exception:
        return None

    description = _clean_html_to_markdown(html)
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

    Tries known ATS/LinkedIn hosted-board URLs first (cheap, no browser).
    Anything else is rendered in a real browser and matched against the
    same ATS APIs by what the page actually requests (_sniff_and_match) —
    catches the job however a company embeds it on its own domain. Only if
    that fails too (e.g. Playwright unavailable) does this fall back to a
    plain HTTP GET + static page-text scrape. A genuinely new row is saved
    with status "shortlisted" rather than the scan-sourced default "new" —
    a job someone bothered to paste a URL for is already past triage.
    Returns (success, message) for the caller to display.
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
        row = _sniff_and_match(url)

    if row is None:
        row = _from_generic(url)

    if row is None:
        return False, ("Couldn't fetch a description from that URL — the page may "
                        "require login or block automated requests.")

    new_count = save_jobs(pd.DataFrame([row]), default_status="shortlisted")
    company = row.get("company") or "unknown company"
    if new_count:
        return True, f'Added "{row["title"] or url}" at {company}.'
    return True, "That job was already in the database — last-seen updated."
