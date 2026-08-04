"""LinkedIn scraper and referral finder using Playwright.

Login uses a visible browser window (headless=False) so LinkedIn doesn't block it.
Session cookies are persisted to data/playwright_sessions/linkedin.json — login is
only required once (or when the session expires).

Progress/status lines go through _log(), which prints to stdout by default.
Call set_log_fn() before a scrape/login/referral-search to route those lines
into a UI log panel instead.
"""

import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, quote_plus

import pandas as pd
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from .browser import debug_headful, launch_stealth_browser, STEALTH_SCRIPT_WITH_PLUGINS

SESSION_FILE = Path("data/playwright_sessions/linkedin.json")


_log_fn = None  # overridable via set_log_fn() — see module docstring


def set_log_fn(fn) -> None:
    """Redirect _log() output (e.g. into a Streamlit log panel) instead of
    stdout. Set before starting a scrape/login/referral-search call; the
    value is shared by worker threads spawned during that call. Pass None
    to go back to plain stdout printing."""
    global _log_fn
    _log_fn = fn


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    if _log_fn:
        _log_fn(line)
    else:
        print(line, flush=True)

_DESC_SELECTORS = (
    "#job-details",
    ".jobs-description__content",
    ".jobs-description-content__text",
    ".jobs-description-content__text--stretch",
    ".jobs-box__html-content",
    "article.jobs-description",
    "section[class*='description']",
    "div[class*='description-content']",
)

_DATE_SELECTORS = (
    ".jobs-unified-top-card__posted-date",
    ".tvm__text--positive",
    "span[class*='posted']",
    "span[class*='date']",
)

# Selectors for the right-side detail panel that appears when clicking a job card
_PANEL_SELECTORS = (
    ".jobs-search__job-details--wrapper",
    ".jobs-search-two-pane__detail-view",
    ".scaffold-layout__detail",
    ".jobs-details",
)


# ── Browser helpers ────────────────────────────────────────────────────────────

def _launch(p, *, headless: bool | None = None, load_session: bool = True) -> tuple[Browser, BrowserContext]:
    """headless=None (the default for every scan call site) resolves to
    `not debug_headful()`, so setting SCAN_DEBUG_HEADFUL=1 in .env makes
    every LinkedIn scan browser launch visibly. Callers with a hard
    requirement (login() always headed, send_linkedin_message()'s
    auto_send toggle) pass headless explicitly and are unaffected."""
    if headless is None:
        headless = not debug_headful()
    storage_state_path = str(SESSION_FILE) if load_session and SESSION_FILE.exists() else None
    return launch_stealth_browser(
        p,
        headless=headless,
        storage_state_path=storage_state_path,
        viewport={"width": 1280, "height": 800},
        timezone_id="Asia/Kolkata",
        stealth_script=STEALTH_SCRIPT_WITH_PLUGINS,
        try_real_chrome=False,
    )


def _save_session(context: BrowserContext) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(SESSION_FILE))


def _is_logged_in(page: Page) -> bool:
    url = page.url.lower()
    return (
        "linkedin.com/feed" in url
        or "linkedin.com/in/" in url
        or "linkedin.com/jobs" in url
        or page.query_selector("nav.global-nav") is not None
        or page.query_selector(".feed-identity-module") is not None
    )


def _fill_input(page: Page, selectors: tuple, value: str) -> None:
    """Try each selector in order, fill the first one found."""
    for sel in selectors:
        el = page.query_selector(sel)
        if el:
            el.click()
            el.fill(value)
            return
    raise RuntimeError(f"Could not find input field. Tried: {selectors}")


def _extract_job_id(href: str) -> str | None:
    m = re.search(r"/jobs/view/(\d+)", href)
    return m.group(1) if m else None


# ── Profile / people helpers ───────────────────────────────────────────────────

def _degree_rank(degree: str) -> int:
    if "1st" in degree: return 0
    if "2nd" in degree: return 1
    if "3rd" in degree: return 2
    return 3


_DEGREE_RE = re.compile(r'^\d+(st|nd|rd|th)(\s|$)', re.I)


def _is_real_name(text: str) -> bool:
    """Return False for degree badges (1st, 2nd, 3rd…) that share aria-hidden spans."""
    return len(text) > 1 and not _DEGREE_RE.match(text)


def _wait_for_search_results(page: Page) -> None:
    try:
        page.wait_for_selector(
            ".entity-result__item, li[class*='result'], .search-results-container",
            timeout=3000,
        )
    except Exception:
        page.wait_for_timeout(1500)


def _profile_container(link):
    return link.evaluate_handle(
        "el => el.closest('li') "
        "|| el.closest('div[class*=\"result\"]') "
        "|| el.closest('div[class*=\"card\"]') "
        "|| el.closest('div[class*=\"member\"]') "
        "|| el.parentElement"
    ).as_element()


def _extract_name(link, container) -> str:
    # Link text first, then specific container selectors. Avoid
    # "span[aria-hidden='true']" without scope — degree badges like "2nd"
    # also use aria-hidden and would be matched first.
    name = link.inner_text().strip().split("\n")[0].strip()
    if not _is_real_name(name):
        name = ""
    if not name and container:
        for sel in (
            ".entity-result__title-text span[aria-hidden='true']",
            "span[class*='name']",
            "strong",
            "span[class*='title']",
        ):
            el = container.query_selector(sel)
            if el:
                candidate = el.inner_text().strip().split("\n")[0].strip()
                if _is_real_name(candidate):
                    name = candidate
                    break
    return name


def _extract_photo_url(container) -> str:
    if not container:
        return ""
    for img_sel in (
        "img.presence-entity__image",
        "img.evi-image",
        "img[class*='profile-photo']",
        "img[class*='entity-image']",
    ):
        img_el = container.query_selector(img_sel)
        if img_el:
            src = (img_el.get_attribute("src")
                   or img_el.get_attribute("data-delayed-url") or "")
            if src and src.startswith("http") and "ghost" not in src.lower():
                return src
    return ""


def _extract_title(container) -> str:
    if not container:
        return ""
    for sel in (
        ".entity-result__primary-subtitle",
        "div[class*='subtitle']",
        "span[class*='subtitle']",
        "div[class*='headline']",
        "span[class*='headline']",
    ):
        el = container.query_selector(sel)
        if el:
            title = el.inner_text().strip()
            if title:
                return title
    return ""


def _extract_degree(container) -> str:
    if not container:
        return ""
    for sel in (
        ".entity-result__badge-text",
        "span[class*='badge']",
        "span[class*='distance']",
        "span[class*='degree']",
    ):
        el = container.query_selector(sel)
        if el:
            degree = el.inner_text().strip()
            if degree:
                return degree
    return ""


def _merge_or_add_profile(results: list[dict], seen: dict[str, int], profile_url: str,
                           name: str, photo_url: str, title: str, degree: str) -> None:
    """Add a new profile result, or — if this is the second link for a
    person already in results — patch in whatever field was missing."""
    if profile_url in seen:
        existing = results[seen[profile_url]]
        if name and not _is_real_name(existing["name"]):
            existing["name"] = name
        if photo_url and not existing["photo_url"]:
            existing["photo_url"] = photo_url
        return

    if not name:
        return  # can't build a useful result without a name

    seen[profile_url] = len(results)
    results.append({
        "name":         name,
        "title":        title,
        "linkedin_url": profile_url,
        "degree":       degree,
        "photo_url":    photo_url,
    })


def _scrape_profiles(page: Page) -> list[dict]:
    """Generic profile scraper: anchors on /in/ links, walks up to the card for context.

    Uses a dict (URL → result index) instead of a set so that the second link for
    the same person (photo link or name link, whichever comes second) can patch in a
    missing name or photo_url without duplicating the result.
    """
    _wait_for_search_results(page)

    results: list[dict] = []
    seen: dict[str, int] = {}  # profile_url -> index in results

    for link in page.query_selector_all("a[href*='/in/']"):
        try:
            href = link.get_attribute("href") or ""
            profile_url = href.split("?")[0]
            if not profile_url or "/in/" not in profile_url:
                continue

            container = _profile_container(link)
            name = _extract_name(link, container)
            photo_url = _extract_photo_url(container)

            if profile_url in seen:
                _merge_or_add_profile(results, seen, profile_url, name, photo_url, "", "")
                continue

            if not name:
                continue  # can't build a useful result without a name

            title = _extract_title(container)
            degree = _extract_degree(container)
            _merge_or_add_profile(results, seen, profile_url, name, photo_url, title, degree)
        except Exception:
            continue

    return results


def _apply_current_company_filter(page: Page, company: str) -> str | None:
    """Click the 'Current company' filter chip in the people-search bar,
    search for the company, select the first autocomplete suggestion, and apply.

    Returns the numeric company ID extracted from the post-filter URL
    (e.g. '1586' from currentCompany=%5B%221586%22%5D), or None on failure.
    The page is left on the filtered results.
    """
    # Wait for the filter bar to render
    filter_btn = None
    for sel in (
        "button[aria-label*='Current company' i]",
        "button:has-text('Current company')",
        "button[id*='current-company' i]",
    ):
        try:
            page.wait_for_selector(sel, timeout=2000)
            filter_btn = page.query_selector(sel)
            break
        except Exception:
            continue

    if not filter_btn:
        return None

    filter_btn.click()

    # Wait for the company input to appear
    company_input = None
    for sel in (
        "input[placeholder*='company' i]",
        "input[aria-label*='company' i]",
        "input[type='text']",
    ):
        try:
            page.wait_for_selector(sel, timeout=2000)
            el = page.query_selector(sel)
            if el and el.is_visible():
                company_input = el
                break
        except Exception:
            continue

    if not company_input:
        return None

    company_input.fill(company)

    # Wait for autocomplete suggestions
    suggestion = None
    for sel in (
        "[role='option']",
        "[class*='typeahead'] li",
        "[class*='autocomplete'] li",
        "ul[role='listbox'] li",
    ):
        try:
            page.wait_for_selector(sel, timeout=2000)
            el = page.query_selector(sel)
            if el and el.is_visible():
                suggestion = el
                break
        except Exception:
            continue

    if not suggestion:
        page.wait_for_timeout(800)
        for sel in ("[role='option']", "ul[role='listbox'] li"):
            el = page.query_selector(sel)
            if el and el.is_visible():
                suggestion = el
                break

    if suggestion:
        suggestion.click()
        page.wait_for_timeout(600)

    # Confirm / show results
    for sel in (
        "button:has-text('Show results')",
        "button:has-text('Done')",
        "button[aria-label*='Apply' i]",
        "button:has-text('Apply')",
    ):
        btn = page.query_selector(sel)
        if btn and btn.is_visible():
            btn.click()
            # Wait for results to reload rather than sleeping a fixed amount
            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                page.wait_for_timeout(1500)
            break

    m = re.search(r"currentCompany=%5B%22(\d+)%22%5D", page.url)
    return m.group(1) if m else None


def _search_people_at_company(page: Page, company_id: str, keywords: str = "") -> list[dict]:
    """Search people using LinkedIn's currentCompany filter (e.g. currentCompany=["1586"]).

    Uses LinkedIn's own faceted search — results show connection-degree badges
    and only include current employees (no skill-name false positives).
    """
    params: dict = {
        "origin": "FACETED_SEARCH",
        "currentCompany": f'["{company_id}"]',
    }
    if keywords:
        params["keywords"] = keywords
    page.goto(
        "https://www.linkedin.com/search/results/people/?" + urlencode(params),
        wait_until="domcontentloaded",
    )
    return _scrape_profiles(page)


def _scrape_company_people(page: Page, slug: str, keywords: str = "") -> list[dict]:
    """Fallback: navigate to /company/{slug}/people/ and scrape profiles."""
    kw_param = f"?keywords={quote_plus(keywords)}" if keywords else ""
    page.goto(
        f"https://www.linkedin.com/company/{slug}/people/{kw_param}",
        wait_until="domcontentloaded",
    )
    return _scrape_profiles(page)


def _get_company_slug(page: Page, company: str) -> str | None:
    """Return the LinkedIn URL slug for a company (used as fallback when filter UI fails)."""
    page.goto(
        "https://www.linkedin.com/search/results/companies/"
        f"?keywords={quote_plus(company)}&origin=GLOBAL_SEARCH_HEADER",
        wait_until="domcontentloaded",
    )
    page.wait_for_timeout(2000)
    for link in page.query_selector_all("a[href*='/company/']"):
        href = link.get_attribute("href") or ""
        m = re.search(r"/company/([^/?#]+)", href)
        if m and m.group(1) not in ("search", "results", "create"):
            return m.group(1)
    return None


def _role_keywords(job_title: str) -> str:
    """Return search keywords that match people in the same role."""
    words = [w for w in job_title.split() if w.lower() not in
             ("the", "a", "an", "and", "or", "of", "in", "at", "for")][:3]
    return " ".join(words) if words else job_title


def _manager_keywords(job_title: str) -> str:
    """Derive search keywords targeting managers and seniors for the given title."""
    title_lower = job_title.lower()
    if any(w in title_lower for w in ("engineer", "developer", "programmer")):
        return "engineering manager OR staff engineer OR principal engineer"
    if any(w in title_lower for w in ("analyst", "data", "scientist")):
        return "data science manager OR senior analyst OR lead analyst"
    if any(w in title_lower for w in ("product", "pm", "product manager")):
        return "director of product OR senior product manager OR group pm"
    if any(w in title_lower for w in ("design", "ux", "ui")):
        return "design manager OR senior designer OR head of design"
    words = job_title.split()[:3]
    return f"senior {' '.join(words)} OR {' '.join(words)} manager"


# ── Public API ─────────────────────────────────────────────────────────────────

def login(email: str, password: str) -> bool:
    """Open a visible browser, log in to LinkedIn, and persist the session.

    A Chrome window will appear — complete any CAPTCHA or 2-FA there, then the
    window closes automatically once the feed loads. Returns True on success.
    """
    with sync_playwright() as p:
        browser, context = _launch(p, headless=False, load_session=False)
        page = context.new_page()
        try:
            page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)

            for sel in ("button[action-type='ACCEPT']", "button#onetrust-accept-btn-handler"):
                btn = page.query_selector(sel)
                if btn:
                    btn.click()
                    page.wait_for_timeout(1000)
                    break

            _fill_input(page, (
                "#username",
                "input[name='session_key']",
                "input[autocomplete='username']",
                "input[type='email']",
            ), email)

            _fill_input(page, (
                "#password",
                "input[name='session_password']",
                "input[autocomplete='current-password']",
                "input[type='password']",
            ), password)

            page.click("button[type='submit']")

            try:
                page.wait_for_url("**/feed**", timeout=60000)
            except Exception:
                pass  # may land on /in/ or /jobs instead

            page.wait_for_timeout(2000)

            if _is_logged_in(page):
                _save_session(context)
                return True
            return False
        finally:
            browser.close()


_JOBS_PER_PAGE = 25  # LinkedIn's fixed page size for job search
_MIN_PAGES     = 3   # always scan at least this many pages per keyword
_MAX_PAGES     = 4   # LinkedIn search results get unreliable/rate-limited beyond this


def _scrape_job_cards(page: Page, seen_ids: set[str], limit: int) -> list[dict]:
    """Collect job cards from the current search-results page.

    Scrolls up to 3 times to trigger lazy-loading, then returns up to `limit`
    new (unseen) job rows. `seen_ids` is updated in place.
    """
    rows: list[dict] = []

    for _ in range(3):
        for link in page.query_selector_all("a[href*='/jobs/view/']"):
            if len(rows) >= limit:
                break
            try:
                href       = link.get_attribute("href") or ""
                numeric_id = _extract_job_id(href)
                if not numeric_id:
                    continue
                job_id = f"li-{numeric_id}"
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

                raw_title = link.inner_text().strip()
                title     = raw_title.split("\n")[0].replace(" with verification", "").strip()
                if not title:
                    card = link.evaluate_handle(
                        "el => el.closest('li') || el.closest('div[data-job-id]')"
                    ).as_element()
                    if card:
                        for sel in ("[class*='title']", "strong", "h3", "h2"):
                            t = card.query_selector(sel)
                            if t:
                                title = t.inner_text().strip().split("\n")[0]
                                break

                card = link.evaluate_handle(
                    "el => el.closest('li') || el.closest('[data-job-id]') || el.parentElement"
                ).as_element()

                company = location_str = ""
                if card:
                    for sel in (
                        ".job-card-container__primary-description",
                        ".artdeco-entity-lockup__subtitle",
                        "[class*='company']",
                        "span[class*='subtitle']",
                    ):
                        el = card.query_selector(sel)
                        if el:
                            company = el.inner_text().strip()
                            break
                    for sel in (
                        ".job-card-container__metadata-item",
                        ".artdeco-entity-lockup__caption",
                        "[class*='location']",
                        "span[class*='metadata']",
                    ):
                        el = card.query_selector(sel)
                        if el:
                            location_str = el.inner_text().strip()
                            break

                rows.append({
                    "id":        job_id,
                    "site":      "linkedin",
                    "job_url":   f"https://www.linkedin.com/jobs/view/{numeric_id}/",
                    "title":     title or "(no title)",
                    "company":   company,
                    "location":  location_str,
                    "is_remote": int("remote" in location_str.lower()),
                })
            except Exception:
                continue

        if len(rows) >= limit:
            break  # don't scroll if we already have enough

        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)

    return rows


def _read_description_from_page(page: Page, row: dict) -> None:
    """Read description, date_posted, and direct apply URL from the current page."""
    for sel in _DESC_SELECTORS:
        desc_el = page.query_selector(sel)
        if desc_el:
            text = desc_el.inner_text().strip()
            if len(text) > 50:
                row["description"] = text
                break

    for sel in _DATE_SELECTORS:
        posted_el = page.query_selector(sel)
        if posted_el:
            row["date_posted"] = posted_el.inner_text().strip()
            break

    # Extract the direct career-page URL for non-EasyApply jobs.
    # EasyApply opens a LinkedIn modal (button, no href); external apply links are <a href=...>.
    if not row.get("job_url_direct"):
        for sel in (
            "a.jobs-apply-button--top-card[href]",
            ".jobs-s-apply a[href]",
            "a[class*='apply-button'][href]",
            "a[data-tracking-control-name*='apply'][href]",
        ):
            el = page.query_selector(sel)
            if el:
                href = el.get_attribute("href") or ""
                if href and "linkedin.com" not in href:
                    row["job_url_direct"] = href
                    break


def _fetch_job_description(page: Page, row: dict) -> None:
    """Fallback: navigate to the job page to get description + date_posted."""
    title = row.get("title", row["id"])
    _log(f"  Fallback nav for: {title!r}")
    try:
        page.goto(row["job_url"], wait_until="domcontentloaded")
        try:
            page.wait_for_selector(
                "#job-details, .jobs-description__content, "
                ".jobs-description-content__text",
                timeout=8000,
            )
        except Exception:
            page.wait_for_timeout(3000)
        _read_description_from_page(page, row)
    except Exception:
        pass


def _fetch_descriptions_via_panel(page: Page, rows: list[dict]) -> list[dict]:
    """Click each job card to load description in the right panel — no page navigation.

    Returns rows that still need descriptions (panel didn't load) so the caller
    can fall back to individual page navigation for them.
    """
    needs_fallback: list[dict] = []

    total = len([r for r in rows if not r.get("description")])
    _log(f"  Fetching descriptions via panel for {total} jobs...")

    for i, row in enumerate(rows):
        if row.get("description"):
            continue

        job_id = row["id"]
        numeric_id = job_id.replace("li-", "")
        title = row.get("title", job_id)

        # Click the job card link — the right panel updates in place
        clicked = False
        for link_sel in (
            f"a[href*='/jobs/view/{numeric_id}/']",
            f"a[href*='currentJobId={numeric_id}']",
        ):
            link = page.query_selector(link_sel)
            if link:
                try:
                    link.click()
                    clicked = True
                    break
                except Exception:
                    continue

        if not clicked:
            _log(f"  [{i+1}/{total}] No card link found: {title!r} — will fallback")
            needs_fallback.append(row)
            continue

        # Wait for the detail panel to populate (combined selector — single wait)
        try:
            page.wait_for_selector(", ".join(_PANEL_SELECTORS), timeout=4000)
        except Exception:
            page.wait_for_timeout(1500)

        _read_description_from_page(page, row)

        if not row.get("description"):
            _log(f"  [{i+1}/{total}] Panel empty for {title!r} — will fallback")
            needs_fallback.append(row)
        else:
            _log(f"  [{i+1}/{total}] Got description via panel: {title!r}")

        page.wait_for_timeout(500)  # brief pause between card clicks

    return needs_fallback


def _scrape_one_page(
    page_num: int,
    keywords: str,
    location: str,
    seconds: int,
) -> list[dict]:
    """Scrape one search-results page + fetch all descriptions.

    Each call runs inside its own sync_playwright() so it is safe to call
    from a thread — Playwright's sync API is not thread-safe to share.
    """
    seen_ids: set[str] = set()
    rows: list[dict] = []

    _log(f"Page {page_num}: starting (offset={page_num * _JOBS_PER_PAGE})")
    with sync_playwright() as p:
        browser, context = _launch(p)
        pg = context.new_page()
        try:
            params = urlencode({
                "keywords": keywords,
                "location": location,
                "f_TPR":    f"r{seconds}",
                "start":    page_num * _JOBS_PER_PAGE,
            })
            pg.goto(
                f"https://www.linkedin.com/jobs/search/?{params}",
                wait_until="domcontentloaded",
            )
            pg.wait_for_timeout(2000)

            rows = _scrape_job_cards(pg, seen_ids, _JOBS_PER_PAGE)
            _log(f"Page {page_num}: found {len(rows)} job cards")

            if rows:
                needs_fallback = _fetch_descriptions_via_panel(pg, rows)
                if needs_fallback:
                    _log(f"Page {page_num}: {len(needs_fallback)} jobs need fallback nav")
                for row in needs_fallback:
                    _fetch_job_description(pg, row)
        finally:
            browser.close()

    _log(f"Page {page_num}: done — {len(rows)} jobs with descriptions")
    return rows


def search_jobs(
    keywords: str,
    location: str,
    results_wanted: int = 25,
    hours_old: int = 72,
    on_page_done=None,
) -> pd.DataFrame:
    """Scrape LinkedIn Jobs while logged in. Returns DataFrame matching jobspy schema.

    Runs up to 4 search pages in parallel threads (one Playwright instance per thread),
    then deduplicates and trims to results_wanted.

    on_page_done(pages_done, total_pages, jobs_so_far) is called from the main thread
    after each page completes — safe to use for Streamlit UI updates.
    """
    seconds    = hours_old * 3600
    num_pages  = min(_MAX_PAGES, max(_MIN_PAGES,
                     (results_wanted + _JOBS_PER_PAGE - 1) // _JOBS_PER_PAGE))

    _log(f"Searching LinkedIn: {keywords!r} in {location!r} | "
         f"{hours_old}h window | {num_pages} page(s) in parallel")

    all_rows: list[dict] = []
    seen_ids: set[str]   = set()
    pages_done = 0

    # Not a `with` block on purpose: exiting `with ThreadPoolExecutor()` calls
    # shutdown(wait=True), which blocks a KeyboardInterrupt/exception unwind
    # until every in-flight page scrape finishes — see the same pattern (and
    # rationale) in scanner/llm/raw_scoring.py's batch-scoring pool.
    pool = ThreadPoolExecutor(max_workers=num_pages)
    try:
        futures = {
            pool.submit(_scrape_one_page, pn, keywords, location, seconds): pn
            for pn in range(num_pages)
        }
        for future in as_completed(futures):
            pn = futures[future]
            pages_done += 1
            try:
                page_rows = future.result()
                before = len(all_rows)
                for row in page_rows:
                    if row["id"] not in seen_ids:
                        seen_ids.add(row["id"])
                        all_rows.append(row)
                added = len(all_rows) - before
                _log(f"Page {pn} merged: +{added} unique jobs (total so far: {len(all_rows)})")
            except Exception as exc:
                warnings.warn(f"Search page {pn} failed: {exc}")
                _log(f"Page {pn} FAILED: {exc}")
            if on_page_done:
                on_page_done(pages_done, num_pages, len(all_rows))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    with sync_playwright() as p:
        browser, context = _launch(p)
        _save_session(context)
        browser.close()

    # Not trimmed to results_wanted: num_pages already guarantees at least
    # _MIN_PAGES pages are fetched, so returning everything found (rather than
    # cutting back down to results_wanted) is the point of that over-fetch.
    _log(f"Search complete: {len(all_rows)} unique job(s) across {num_pages} page(s)")
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _collect_via_company_filter(page: Page, company: str, mgr_kw: str, limit: int,
                                 pool: list[dict], add_fn) -> str | None:
    """Step 1: apply the 'Current company' filter (role-keyword search already
    on the page) and, while still short of `limit`, reuse the resolved company
    ID for a manager-keyword pass and then an all-employees pass. Returns the
    company ID, or None if the filter UI failed to resolve one."""
    company_id = _apply_current_company_filter(page, company)
    if not company_id:
        return None

    add_fn(_scrape_profiles(page))
    if len(pool) < limit:
        add_fn(_search_people_at_company(page, company_id, mgr_kw))
    if len(pool) < limit:
        add_fn(_search_people_at_company(page, company_id))
    return company_id


def _collect_via_company_slug(page: Page, company: str, role_kw: str, mgr_kw: str,
                               limit: int, pool: list[dict], add_fn) -> None:
    """Fallback when the filter UI fails: scrape the /company/{slug}/people/
    page directly, same role/manager/all-employees passes as the filter path."""
    slug = _get_company_slug(page, company)
    if not slug:
        return
    add_fn(_scrape_company_people(page, slug, role_kw))
    if len(pool) < limit:
        add_fn(_scrape_company_people(page, slug, mgr_kw))
    if len(pool) < limit:
        add_fn(_scrape_company_people(page, slug))


def _collect_via_keyword_search(page: Page, company: str, keywords: str, add_fn) -> None:
    """Supplement: plain "{company} {keywords}" people search, no company filter."""
    page.goto(
        "https://www.linkedin.com/search/results/people/"
        f"?keywords={quote_plus(company + ' ' + keywords)}"
        "&origin=GLOBAL_SEARCH_HEADER",
        wait_until="domcontentloaded",
    )
    add_fn(_scrape_profiles(page))


def find_referral_contacts(
    company: str,
    job_title: str,
    limit: int = 10,
) -> list[dict]:
    """Find LinkedIn contacts at a company, ranked by connection degree.

    Strategy:
      1. Navigate to people search, open "All filters", fill "Current company"
         with the company name, and let LinkedIn's autocomplete resolve the
         numeric ID. Apply the filter and scrape results with role keywords.
      2. Once the company ID is known from the URL, reuse it for manager-keyword
         and no-keyword passes without reopening the filter panel.
      3. Fall back to the /company/{slug}/people/ page if the filter UI fails.
      4. Supplement with keyword-only search if still below limit.
    Results are sorted: 1st > 2nd > 3rd+ > unknown degree.
    """
    role_kw = _role_keywords(job_title)
    mgr_kw  = _manager_keywords(job_title)

    seen_urls: set[str] = set()
    pool: list[dict]    = []

    def _add(results: list[dict]) -> None:
        for c in results:
            if c["linkedin_url"] not in seen_urls:
                seen_urls.add(c["linkedin_url"])
                pool.append(c)

    with sync_playwright() as p:
        browser, context = _launch(p)
        page = context.new_page()
        try:
            page.goto(
                "https://www.linkedin.com/search/results/people/"
                f"?keywords={quote_plus(role_kw)}&origin=GLOBAL_SEARCH_HEADER",
                wait_until="domcontentloaded",
            )
            page.wait_for_timeout(2000)

            company_id = _collect_via_company_filter(page, company, mgr_kw, limit, pool, _add)
            if not company_id:
                _collect_via_company_slug(page, company, role_kw, mgr_kw, limit, pool, _add)

            if len(pool) < limit:
                _collect_via_keyword_search(page, company, role_kw, _add)
            if len(pool) < limit:
                _collect_via_keyword_search(page, company, mgr_kw, _add)

            _save_session(context)
        finally:
            browser.close()

    pool.sort(key=lambda c: _degree_rank(c["degree"]))
    return pool[:limit]


def send_linkedin_message(
    profile_url: str,
    message: str,
    auto_send: bool = True,
) -> bool:
    """Open the LinkedIn profile, click Message, and pre-fill the compose box.

    auto_send=True  — headless browser clicks Send automatically, then closes.
    auto_send=False — visible browser opens with the message ready; the user
                      edits and sends manually. The browser stays open (Chromium
                      runs as an OS process) until the user closes it.

    Returns True once the compose box is filled (auto_send=False) or the message
    is sent (auto_send=True). Returns False if the Message button or compose box
    could not be found.
    """
    # Use start()/stop() directly so we can choose not to stop in manual mode.
    pw = sync_playwright().start()
    browser, context = _launch(pw, headless=auto_send)
    pg = context.new_page()

    def _cleanup() -> None:
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass

    try:
        pg.goto(profile_url, wait_until="domcontentloaded")
        pg.wait_for_timeout(3000)

        # LinkedIn's Message link has href="/messaging/compose/..." — find it
        # regardless of obfuscated class names (which change with each deploy).
        try:
            pg.wait_for_selector('a[href*="/messaging/compose/"]', timeout=8000)
        except Exception:
            pg.wait_for_timeout(1000)

        msg_btn = pg.query_selector('a[href*="/messaging/compose/"]')
        if not msg_btn:
            _cleanup()
            return False

        # Scroll the button into view and let React's event handler run.
        # Do NOT use expect_navigation — LinkedIn's SPA intercepts the click,
        # calls event.preventDefault(), and opens the compose dialog as an
        # overlay via history.pushState (no full page navigation).
        # Fire the click via JavaScript — bypasses any overlapping element
        # (e.g. LinkedIn Learning banner) that intercepts pointer events,
        # while still triggering React's onClick so the compose overlay opens.
        pg.evaluate("""
            () => {
                const btn = Array.from(document.querySelectorAll('a'))
                    .find(a => (a.href || '').includes('/messaging/compose/'));
                if (btn) btn.click();
            }
        """)

        # Give React time to render the compose overlay.
        pg.wait_for_timeout(3000)

        compose = None
        for sel in (
            "[role='textbox'][contenteditable='true']",
            ".msg-form__contenteditable",
            "[contenteditable='true']:not([aria-label='Subject'])",
        ):
            try:
                compose = pg.wait_for_selector(sel, timeout=8000)
                if compose:
                    break
            except Exception:
                continue

        if not compose:
            # Compose box never appeared — return False so caller can warn user
            if auto_send:
                _cleanup()
            return False

        # Click to focus, then type — works with React's synthetic event system
        # regardless of obfuscated class names on the compose box.
        compose.click()
        pg.wait_for_timeout(300)
        pg.keyboard.type(message, delay=10)
        pg.wait_for_timeout(500)

        if auto_send:
            sent = False
            for sel in (
                "button.msg-form__send-button",
                ".msg-form button[type='submit']",
                ".msg-overlay-conversation-bubble button[type='submit']",
                "button:has-text('Send')",
            ):
                try:
                    send_btn = pg.query_selector(sel)
                    if send_btn and send_btn.is_visible():
                        send_btn.click()
                        pg.wait_for_timeout(1500)
                        sent = True
                        break
                except Exception:
                    continue
            _save_session(context)
            _cleanup()
            return sent
        else:
            # Manual mode: keep the browser open until the user closes it.
            # A daemon thread holds references to pw/browser so they aren't
            # garbage-collected when this function returns.
            def _wait_for_close():
                try:
                    browser.wait_for_event("disconnected", timeout=3_600_000)
                except Exception:
                    pass
                finally:
                    _cleanup()

            threading.Thread(target=_wait_for_close, daemon=True).start()
            return True

    except Exception:
        _cleanup()
        raise
