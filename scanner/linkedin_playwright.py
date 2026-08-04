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
        try_real_chrome=True,
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


def _wait_for_job_results(page: Page) -> None:
    """Wait for the job-search results list (or a 'no results' banner) to
    actually render before scraping — `domcontentloaded` fires well before
    LinkedIn's client-side JS has populated any job cards."""
    try:
        page.wait_for_selector(
            "a[href*='/jobs/view/'], .jobs-search-no-results-banner, "
            "li[data-occludable-job-id]",
            timeout=8000,
        )
    except Exception:
        page.wait_for_timeout(2000)


def _wait_for_first_card_ready(page: Page, timeout: int = 10000) -> None:
    """Make sure the first job card in the list is actually loaded and its
    description panel reflects THAT job — not just any leftover content from
    the previous page — before scrolling/scraping starts.

    LinkedIn auto-selects a job on page load, but right after a pagination
    click the panel can still be showing the previous page's description
    while the new list renders underneath it; a plain "is there description
    text yet" check can pass on that stale content, since it never goes
    empty in between. Explicitly (re-)clicking the first visible card and
    waiting for the panel to refresh after that click is the only way to be
    sure the description shown actually belongs to the current first card —
    without it, that first card was intermittently read as empty later
    ("Panel empty ... will fallback"), especially on page 2+.

    Skipped on /jobs/search-results/ (natural_language mode): unlike the
    classic /jobs/search/ list, its one real <a href="/jobs/view/..."> is a
    full navigation link, not an in-place panel switch — clicking it
    navigates the whole page away from the results list instead of just
    updating the side panel, which left the list-reading code that runs
    right after this looking at an empty page. That endpoint's panel is
    already populated for the auto-selected job via `currentJobId` on load,
    so there's nothing to click there in the first place.
    """
    if "/jobs/search-results/" in page.url:
        desc_selector = ", ".join(_DESC_SELECTORS)
        try:
            page.wait_for_function(
                """(sel) => {
                    const el = document.querySelector(sel);
                    return !!el && el.innerText.trim().length > 50;
                }""",
                arg=desc_selector,
                timeout=timeout,
            )
        except Exception:
            pass
        return

    try:
        page.wait_for_selector("a[href*='/jobs/view/']", timeout=timeout)
    except Exception:
        return

    link = page.query_selector("a[href*='/jobs/view/']")
    if link:
        try:
            link.click()
        except Exception:
            pass

    desc_selector = ", ".join(_DESC_SELECTORS)
    try:
        page.wait_for_function(
            """(sel) => {
                const el = document.querySelector(sel);
                return !!el && el.innerText.trim().length > 50;
            }""",
            arg=desc_selector,
            timeout=timeout,
        )
    except Exception:
        pass


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
_MAX_SCROLL_ATTEMPTS = 8  # cards load ~7 at a time; enough headroom to reach 25 per page


def _find_scroll_container(page: Page):
    """Find the scrollable ancestor of the job-card list.

    LinkedIn renders the results list inside its own overflow container, not
    the page body, so `window.scrollTo` alone never triggers its lazy-loading
    — it silently no-ops, capping extraction at whatever rendered on initial
    load (~7 cards). Walks up from a rendered card generically (nearest
    ancestor with scrollHeight > clientHeight and overflow-y auto/scroll)
    rather than a hardcoded class name, since LinkedIn's wrapper classes
    shift often. Returns None if no card/scrollable ancestor is found yet,
    in which case callers fall back to scrolling the window.
    """
    handle = page.evaluate_handle("""
        () => {
            const card = document.querySelector("li[data-occludable-job-id]")
                      || document.querySelector("a[href*='/jobs/view/']");
            if (!card) return null;
            let node = card.closest('li[data-occludable-job-id]') || card;
            while (node && node !== document.body) {
                const style = window.getComputedStyle(node);
                if (node.scrollHeight > node.clientHeight + 10
                        && /(auto|scroll)/.test(style.overflowY)) {
                    return node;
                }
                node = node.parentElement;
            }
            return null;
        }
    """)
    return handle.as_element()


def _scroll_step(page: Page, container) -> None:
    """Scroll one step toward the bottom of the job list — the container if
    found, else the window as a fallback — then wait for lazy content.

    Moves by 3/4 of the viewport height per step rather than jumping straight
    to scrollHeight, so lazy-loaded cards render in smaller, more human-like
    increments instead of one large jump.
    """
    if container:
        container.evaluate(
            "el => el.scrollTo(0, Math.min(el.scrollTop + el.clientHeight * 0.75, el.scrollHeight))"
        )
    else:
        page.evaluate(
            "() => window.scrollTo(0, Math.min(window.scrollY + window.innerHeight * 0.75, document.body.scrollHeight))"
        )
    page.wait_for_timeout(1500)


def _extract_cards(page: Page, seen_ids: set[str], limit: int) -> list[dict]:
    """One extraction pass over whatever job cards are currently in the DOM.

    Returns up to `limit` new (unseen) job rows. `seen_ids` is updated in place.
    """
    rows: list[dict] = []

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

    return rows


def _extract_semantic_cards(page: Page, seen_ids: set[str], limit: int) -> list[dict]:
    """One extraction pass over LinkedIn's natural-language/semantic search
    results UI (keyword_mode="natural_language"), which renders
    /jobs/search-results/ with an entirely different, obfuscated component
    tree than the classic /jobs/search/ list _extract_cards() handles — no
    stable class names, no per-card <a href>, no title/company text anywhere
    in the card markup itself. The one thing it does expose reliably is each
    card's job id, via a `componentkey="job-card-component-ref-{id}"`
    attribute. Returns id-only rows; _fetch_job_description() fills in
    title/company (from the job page's <title> tag) and description once
    each row is visited.
    """
    numeric_ids: list[str] = page.evaluate("""
        () => [...new Set(
            [...document.querySelectorAll("[componentkey^='job-card-component-ref-']")]
                .map(e => e.getAttribute('componentkey').replace('job-card-component-ref-', ''))
        )]
    """)

    rows: list[dict] = []
    for numeric_id in numeric_ids:
        if len(rows) >= limit:
            break
        job_id = f"li-{numeric_id}"
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        rows.append({
            "id":        job_id,
            "site":      "linkedin",
            "job_url":   f"https://www.linkedin.com/jobs/view/{numeric_id}/",
            "title":     "",
            "company":   "",
            "location":  "",
            "is_remote": 0,
        })
    return rows


def _scrape_job_cards(page: Page, seen_ids: set[str], limit: int,
                       scroll_strategy: str = "incremental") -> list[dict]:
    """Collect job cards from the current search-results page.

    scroll_strategy:
    - "incremental" (default): extract whatever's rendered, scroll one step,
      extract again, repeating until `limit` is reached, no new cards appear
      for two consecutive rounds (end of list), or _MAX_SCROLL_ATTEMPTS is
      hit. LinkedIn's list appends new cards to the DOM rather than
      replacing/virtualizing old ones, so nothing is lost by extracting
      between scrolls — and it matches how the list actually loads
      (~7 cards per chunk).
    - "all_first": scroll to the bottom repeatedly until the card count
      stops growing (same cap), then extract once. Kept only for debug A/B
      comparison against "incremental" (see debug_linkedin_scan.py) — no
      production caller uses this.
    """
    container = _find_scroll_container(page)

    if scroll_strategy == "all_first":
        prev_count = -1
        for _ in range(_MAX_SCROLL_ATTEMPTS):
            count = page.evaluate("document.querySelectorAll(\"a[href*='/jobs/view/']\").length")
            if count == prev_count:
                break
            prev_count = count
            _scroll_step(page, container)
        return _extract_cards(page, seen_ids, limit)

    rows: list[dict] = []
    stale_rounds = 0
    for _ in range(_MAX_SCROLL_ATTEMPTS):
        new_rows = _extract_cards(page, seen_ids, limit - len(rows))
        rows.extend(new_rows)
        if len(rows) >= limit:
            break
        if new_rows:
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= 2:
                break
        _scroll_step(page, container)

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

_JOB_DESCRIPTION_PAGE_SETTLE_MS = 750

def _fetch_job_description(page: Page, row: dict) -> None:
    """Fallback: navigate to the job page to get description + date_posted.

    Also fills in title/company when the caller didn't already have them
    (e.g. _extract_semantic_cards() rows, which start out id-only) by
    parsing the page's own <title> tag — "{Job Title} | {Company} |
    LinkedIn" — since that's stable regardless of whatever obfuscated class
    names the page body itself is using.
    """
    title = row.get("title", row["id"])
    _log(f"  Fallback nav for: {title!r}")
    try:
        page.goto(row["job_url"], wait_until="domcontentloaded")
        page.wait_for_timeout(_JOB_DESCRIPTION_PAGE_SETTLE_MS)
        try:
            page.wait_for_selector(
                "#job-details, .jobs-description__content, "
                ".jobs-description-content__text",
                timeout=8000,
            )
        except Exception:
            page.wait_for_timeout(3000)

        if not row.get("title") or not row.get("company"):
            parts = [part.strip() for part in page.title().split("|")]
            if len(parts) >= 2:
                if not row.get("title"):
                    row["title"] = parts[0]
                if not row.get("company"):
                    row["company"] = parts[1]

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

        page.wait_for_timeout(1500)  # brief pause between card clicks

    return needs_fallback


def _fetch_descriptions_via_semantic_click(page: Page, rows: list[dict]) -> list[dict]:
    """Click each job card in LinkedIn's natural-language/semantic search UI
    (keyword_mode="natural_language") to load its description into the
    right-side panel in place — no page navigation — instead of visiting
    each job's own page one at a time via _fetch_job_description(). That
    full-navigation fallback is what _scrape_pages() used exclusively before
    this (there's no per-card link to click there, unlike the classic list),
    and measured live at ~13s/job; clicking through the cards here measured
    at ~2s/job instead — about 7x faster for a mode that's already debug-only
    and slow to iterate on.

    Cards are found via the same `componentkey="job-card-component-ref-{id}"`
    attribute _extract_semantic_cards() uses to discover them — this UI has
    no other stable way to target a specific card. After a click, the
    description streams into a job-id-scoped container,
    `#JobDetails_AboutTheJob_{id}`; waiting for that specific id (rather than
    a generic "is there description text anywhere" check, which this UI's
    obfuscated markup can't support the way _DESC_SELECTORS does for the
    classic list) avoids reading stale content left over from whichever job
    was selected before this click. Title/company come from the page's own
    <title> tag ("{Job Title} | {Company} | LinkedIn") — LinkedIn keeps that
    updated on selection even though the click never triggers a real
    navigation.

    Returns rows the click approach couldn't fill in (card not found, or the
    panel didn't load in time) so the caller can fall back to
    _fetch_job_description()'s slower full navigation for just those.
    """
    needs_fallback: list[dict] = []
    total = len(rows)
    _log(f"  Fetching descriptions via card click for {total} jobs...")

    for i, row in enumerate(rows):
        numeric_id = row["id"].replace("li-", "")
        title = row.get("title") or row["id"]

        card = page.query_selector(
            f"div[role='button'][componentkey='job-card-component-ref-{numeric_id}']"
        )
        if not card:
            _log(f"  [{i+1}/{total}] No card found: {title!r} — will fallback")
            needs_fallback.append(row)
            continue
        try:
            card.click()
        except Exception:
            _log(f"  [{i+1}/{total}] Click failed: {title!r} — will fallback")
            needs_fallback.append(row)
            continue

        try:
            page.wait_for_function(
                """(id) => {
                    const el = document.querySelector('#JobDetails_AboutTheJob_' + id);
                    return !!el && el.innerText.trim().length > 50;
                }""",
                arg=numeric_id,
                timeout=8000,
            )
        except Exception:
            _log(f"  [{i+1}/{total}] Panel empty for {title!r} — will fallback")
            needs_fallback.append(row)
            continue

        parts = [part.strip() for part in page.title().split("|")]
        if len(parts) >= 2:
            row["title"] = parts[0]
            row["company"] = parts[1]

        desc_el = page.query_selector(f"#JobDetails_AboutTheJob_{numeric_id}")
        if desc_el:
            row["description"] = desc_el.inner_text().strip()

        if not row.get("description"):
            _log(f"  [{i+1}/{total}] Panel empty for {title!r} — will fallback")
            needs_fallback.append(row)
        else:
            _log(f"  [{i+1}/{total}] Got description via card click: {row.get('title', title)!r}")

        page.wait_for_timeout(300)  # brief pause between card clicks

    return needs_fallback


def _find_pagination_control(page: Page, target_page: int):
    """Find LinkedIn's own "Page N" control in the bottom pagination bar.

    target_page is 1-indexed, matching the number shown on screen (unlike our
    internal 0-indexed page_num)."""
    for sel in (
        f"li[data-test-pagination-page-btn='{target_page}'] button",
        f"button[aria-label='Page {target_page}']",
        f"a[aria-label='Page {target_page}']",
    ):
        el = page.query_selector(sel)
        if el:
            return el
    return None


def _goto_next_results_page(page: Page, target_page: int) -> bool:
    """Advance to results page `target_page` (1-indexed) by clicking LinkedIn's
    own bottom pagination control instead of constructing a `start=` URL
    ourselves.

    A hand-built `?start=N` URL doesn't carry whatever session state (e.g.
    currentJobId) LinkedIn's own pagination click threads between pages —
    without it, later "pages" were silently re-rendering page 1's results
    instead of advancing. Returns False (caller should stop paginating) if no
    control for that page number is found, e.g. fewer results than pages.
    """
    control = _find_pagination_control(page, target_page)
    if not control:
        return False
    try:
        control.scroll_into_view_if_needed()
        control.click()
    except Exception:
        return False
    _wait_for_job_results(page)

    # LinkedIn's client-side pagination click can leave the list in a stale
    # or partially-rendered state (see _wait_for_first_card_ready) — a full
    # reload of the page it just navigated to (the click updates the URL's
    # `start=` param via history.pushState) forces a clean server render of
    # that page instead of relying on the in-place JS update.
    #
    # Skipped on /jobs/search-results/ (natural_language mode): verified
    # live that this endpoint's `start=` param is client-side/history-only —
    # the SPA reads it from in-memory state, not from the URL on a fresh
    # load. Reloading there doesn't refresh page N's content; it silently
    # resets the whole page back to page 1 (same job ids, `start=` param
    # dropped from the URL), which looked like every "next page" was really
    # just re-scraping page 1. The classic /jobs/search/ list has no such
    # problem — its `start=` is a real server-side query param.
    if "/jobs/search-results/" not in page.url:
        try:
            page.reload(wait_until="domcontentloaded")
        except Exception:
            pass
        _wait_for_job_results(page)

    _wait_for_first_card_ready(page)
    return True


def _scrape_pages(
    keywords: str,
    location: str,
    seconds: int,
    num_pages: int,
    scroll_strategy: str = "incremental",
    keyword_mode: str = "structured",
    on_page_done=None,
) -> list[dict]:
    """Scrape `num_pages` search-results pages + fetch all descriptions,
    sequentially in one browser session — advancing page-to-page via
    LinkedIn's own bottom pagination controls (see _goto_next_results_page)
    rather than independently constructed `start=` URLs per page.

    keyword_mode:
    - "structured" (default): the usual /jobs/search/ page with separate
      keywords/location/f_TPR query params, as LinkedIn's own search form
      submits them. Cards are read via _extract_cards()/_scrape_job_cards()
      (real <a href="/jobs/view/...">  per card) and descriptions fetched
      via the right-side panel (_fetch_descriptions_via_panel()), falling
      back to per-job navigation only when the panel doesn't load.
    - "natural_language": LinkedIn's AI-powered /jobs/search-results/ page
      instead, with location and the time window folded into one free-text
      keywords string (e.g. "Staff Software Engineer in Bengaluru in last 72
      hours") and the location/f_TPR params omitted entirely. Debug-only,
      for comparing against "structured"; not used by any production
      caller.

      This UI's card markup is a completely different, obfuscated component
      tree with no scrapable per-card text or links at all — the only
      reliable signal is a `componentkey="job-card-component-ref-{id}"`
      attribute (see _extract_semantic_cards()). So every row here starts
      out id-only, and descriptions are fetched by clicking each card
      in place (_fetch_descriptions_via_semantic_click() — ~7x faster than
      full navigation), falling back to per-job navigation
      (_fetch_job_description(), which fills in title/company from the job
      page's own <title> tag) only for cards that approach couldn't fill in.
    """
    seen_ids: set[str] = set()
    all_rows: list[dict] = []
    semantic = keyword_mode == "natural_language"

    with sync_playwright() as p:
        browser, context = _launch(p)
        pg = context.new_page()
        try:
            if semantic:
                base_url = "https://www.linkedin.com/jobs/search-results/"
                hours_old = seconds // 3600
                query = {"keywords": f"{keywords} in {location} in last {hours_old} hours"}
            else:
                base_url = "https://www.linkedin.com/jobs/search/"
                query = {"keywords": keywords, "location": location, "f_TPR": f"r{seconds}"}
            params = urlencode(query)
            pg.goto(f"{base_url}?{params}", wait_until="domcontentloaded")
            _wait_for_job_results(pg)
            _wait_for_first_card_ready(pg)

            for page_num in range(num_pages):
                ui_page = page_num + 1
                _log(f"Page {ui_page}: starting")

                if semantic:
                    rows = _extract_semantic_cards(pg, seen_ids, _JOBS_PER_PAGE)
                    if not rows:
                        pg.wait_for_timeout(2000)
                        rows = _extract_semantic_cards(pg, seen_ids, _JOBS_PER_PAGE)
                else:
                    rows = _scrape_job_cards(pg, seen_ids, _JOBS_PER_PAGE, scroll_strategy)
                    if not rows:
                        # Job list can populate via a follow-up XHR after the
                        # initial paint — give it one more beat before giving up.
                        pg.wait_for_timeout(2000)
                        rows = _scrape_job_cards(pg, seen_ids, _JOBS_PER_PAGE, scroll_strategy)
                _log(f"Page {ui_page}: found {len(rows)} job cards")

                if rows:
                    # Fallback nav navigates pg away from the results list —
                    # save the URL LinkedIn's own pagination put us on so we
                    # can return to this exact page before advancing further.
                    results_url = pg.url
                    needs_fallback = (
                        _fetch_descriptions_via_semantic_click(pg, rows) if semantic
                        else _fetch_descriptions_via_panel(pg, rows)
                    )
                    if needs_fallback:
                        _log(f"Page {ui_page}: {len(needs_fallback)} jobs need fallback nav")
                        for row in needs_fallback:
                            _fetch_job_description(pg, row)
                        pg.goto(results_url, wait_until="domcontentloaded")
                        _wait_for_job_results(pg)

                all_rows.extend(rows)
                _log(f"Page {ui_page}: done — {len(rows)} jobs with descriptions")
                if on_page_done:
                    on_page_done(ui_page, num_pages, len(all_rows))

                if page_num < num_pages - 1:
                    next_page = ui_page + 1
                    if not _goto_next_results_page(pg, next_page):
                        _log(f"Page {next_page}: no pagination link found — stopping early")
                        break

            _save_session(context)
        finally:
            browser.close()

    return all_rows


def search_jobs(
    keywords: str,
    location: str,
    results_wanted: int = 25,
    hours_old: int = 72,
    on_page_done=None,
    num_pages: int | None = None,
    scroll_strategy: str = "incremental",
    keyword_mode: str = "structured",
) -> pd.DataFrame:
    """Scrape LinkedIn Jobs while logged in. Returns DataFrame matching jobspy schema.

    Scrapes up to 4 search-results pages sequentially in one browser session,
    advancing via LinkedIn's own bottom pagination controls (see
    _goto_next_results_page) rather than independently constructed `start=`
    URLs per page, then deduplicates and trims to results_wanted.

    on_page_done(pages_done, total_pages, jobs_so_far) is called after each
    page completes — safe to use for Streamlit UI updates.

    num_pages overrides the results_wanted-derived page count entirely (no _MIN_PAGES/
    _MAX_PAGES clamping) — for debugging/testing an exact page count only; production
    callers should leave this None and let results_wanted drive it as usual.

    scroll_strategy is passed straight through to _scrape_job_cards() — see its
    docstring. Production callers should leave this at the default "incremental";
    "all_first" exists only for debug_linkedin_scan.py's A/B comparison.

    keyword_mode is passed straight through to _scrape_pages() — see its
    docstring. Production callers should leave this at the default "structured";
    "natural_language" exists only for debug_linkedin_scan.py's A/B comparison.
    """
    seconds = hours_old * 3600
    if num_pages is None:
        num_pages = min(_MAX_PAGES, max(_MIN_PAGES,
                         (results_wanted + _JOBS_PER_PAGE - 1) // _JOBS_PER_PAGE))

    _log(f"Searching LinkedIn: {keywords!r} in {location!r} | "
         f"{hours_old}h window | {num_pages} page(s)")

    seen_ids: set[str] = set()
    all_rows: list[dict] = []

    try:
        page_rows = _scrape_pages(
            keywords, location, seconds, num_pages, scroll_strategy, keyword_mode,
            on_page_done=on_page_done,
        )
        for row in page_rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                all_rows.append(row)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc!r}"
        warnings.warn(f"Search failed: {detail}")
        _log(f"Search FAILED: {detail}")

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
