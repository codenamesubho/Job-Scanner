"""Naukri.com job scraper using Playwright (headless Chromium).

Session cookies are persisted to data/playwright_sessions/naukri.json.
"""

import re
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, BrowserContext, Page

from .browser import SessionBrowser

SESSION_FILE = Path("data/playwright_sessions/naukri.json")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


_SESSION = SessionBrowser(SESSION_FILE, viewport={"width": 1280, "height": 800})


def _launch(p, *, headless: bool | None = None, load_session: bool = True):
    return _SESSION.launch(p, headless=headless, load_session=load_session)


def _save_session(context: BrowserContext) -> None:
    _SESSION.save(context)


def _is_logged_in(page: Page) -> bool:
    url = page.url.lower()
    return (
        "naukri.com/mnjuser" in url
        or page.query_selector("a[href*='myjobs']") is not None
        or page.query_selector(".nI-gNb-sb__main") is not None
    )


def login(email: str, password: str) -> bool:
    """Open a visible browser, log in to Naukri, and persist the session.

    A Chrome window will appear — complete any OTP or CAPTCHA there.
    Returns True on success.
    """
    with sync_playwright() as p:
        browser, context = _launch(p, headless=False, load_session=False)
        page = context.new_page()
        try:
            page.goto("https://www.naukri.com/nlogin/login", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)

            for sel, val in (
                ("input[placeholder*='Email']", email),
                ("input[placeholder*='email']", email),
                ("input[type='email']", email),
            ):
                el = page.query_selector(sel)
                if el:
                    el.click()
                    el.fill(val)
                    break

            for sel, val in (
                ("input[placeholder*='password']", password),
                ("input[placeholder*='Password']", password),
                ("input[type='password']", password),
            ):
                el = page.query_selector(sel)
                if el:
                    el.click()
                    el.fill(val)
                    break

            for sel in ("button[type='submit'].loginButton", "button[type='submit']"):
                btn = page.query_selector(sel)
                if btn:
                    btn.click()
                    break

            # Wait up to 30s for dashboard to load (user may need to complete OTP)
            page.wait_for_timeout(6000)

            if _is_logged_in(page):
                _save_session(context)
                return True
            return False
        finally:
            browser.close()


_JOBS_PER_PAGE = 20  # Naukri's typical page size for job search
_MIN_PAGES     = 3   # always scan at least this many pages per keyword
_MAX_PAGES     = 4   # keep a ceiling consistent with the other scan sources


def _parse_job_card(card) -> dict:
    title_el   = card.query_selector("a.title")
    company_el = card.query_selector("a.comp-name")
    loc_el     = card.query_selector("span.locWdth")

    title   = title_el.inner_text().strip()   if title_el   else ""
    company = company_el.inner_text().strip() if company_el else ""
    loc_str = loc_el.inner_text().strip()     if loc_el     else ""
    href    = title_el.get_attribute("href")  if title_el   else ""

    # Derive a stable ID from the URL slug
    job_id = f"naukri-{re.sub(r'[^a-z0-9]', '', title.lower()[:20])}-{abs(hash(href)) % 10**8}"

    return {
        "id":        job_id,
        "site":      "naukri",
        "job_url":   href,
        "title":     title,
        "company":   company,
        "location":  loc_str,
        "is_remote": int("remote" in loc_str.lower()),
    }


def _scrape_cards(cards, seen_urls: set[str]) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        try:
            row = _parse_job_card(card)
            if row["job_url"] in seen_urls:
                continue
            seen_urls.add(row["job_url"])
            rows.append(row)
        except Exception:
            continue
    return rows


def _fetch_job_description(page: Page, row: dict) -> None:
    """Best-effort: navigate to the job's own page and fill in description/date_posted."""
    if not row.get("job_url"):
        return
    try:
        page.goto(row["job_url"], wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        # Naukri's current site uses CSS-modules class names with an
        # unstable hash suffix (e.g. "styles_JDC__dang-inner-html__h0K4t")
        # — match on the stable substring instead of the full name.
        desc_els = page.query_selector_all("div[class*='dang-inner-html']")
        if desc_els:
            row["description"] = desc_els[0].inner_text().strip()
        for stat_el in page.query_selector_all("[class*='jhc__stat']"):
            text = stat_el.inner_text().strip()
            if text.lower().startswith("posted"):
                row["date_posted"] = text.split(":", 1)[-1].strip()
                break
    except Exception:
        pass


def search_jobs(
    keywords: str,
    location: str,
    results_wanted: int = 25,
    experience: int = 0,
) -> pd.DataFrame:
    """Scrape Naukri job listings across multiple search-result pages.

    Returns DataFrame matching jobspy schema. Not trimmed to results_wanted:
    num_pages already guarantees at least _MIN_PAGES pages are scanned, so
    returning everything found (rather than cutting back down) is the point.
    """
    kw_slug  = _slugify(keywords)
    loc_slug = _slugify(location)
    base_url = f"https://www.naukri.com/{kw_slug}-jobs-in-{loc_slug}?experience={experience}"

    num_pages = min(_MAX_PAGES, max(_MIN_PAGES,
                    (results_wanted + _JOBS_PER_PAGE - 1) // _JOBS_PER_PAGE))

    rows: list[dict] = []
    seen_urls: set[str] = set()

    with sync_playwright() as p:
        browser, context = _launch(p)
        page = context.new_page()
        try:
            for page_num in range(1, num_pages + 1):
                url = base_url if page_num == 1 else f"{base_url}&page={page_num}"
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)

                cards = page.query_selector_all("div.cust-job-tuple")
                if not cards:
                    break  # no more result pages

                rows.extend(_scrape_cards(cards, seen_urls))

            for row in rows:
                _fetch_job_description(page, row)

            _save_session(context)
        finally:
            browser.close()

    return pd.DataFrame(rows) if rows else pd.DataFrame()
