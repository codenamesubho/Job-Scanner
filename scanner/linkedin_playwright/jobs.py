"""Job search: scrolling the results list, extracting cards, paginating."""
import re
import warnings
from urllib.parse import urlencode

import pandas as pd
from playwright.sync_api import Page, sync_playwright

from .descriptions import (
    _fetch_descriptions_via_panel, _fetch_descriptions_via_semantic_click, _fetch_job_description,
)
from .selectors import (
    _DESC_SELECTORS, _JOBS_PER_PAGE, _MAX_PAGES, _MAX_SCROLL_ATTEMPTS, _MIN_PAGES,
    SEARCH_URL, SEMANTIC_SEARCH_URL,
)
from .session import _launch, _log, _save_session, _wait_for_job_results

def _extract_job_id(href: str) -> str | None:
    m = re.search(r"/jobs/view/(\d+)", href)
    return m.group(1) if m else None

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

def build_search_url(keywords: str, location: str, seconds: int,
                      keyword_mode: str = "structured") -> str:
    """Build the results URL for one search.

    The two modes hit genuinely different endpoints, not just different params:

    - "structured" — /jobs/search/ with separate keywords/location/f_TPR params,
      the way LinkedIn's own search form submits them.
    - "natural_language" — /jobs/search-results/, LinkedIn's AI-powered semantic
      search, with the location and time window folded into one free-text
      keywords string and no location/f_TPR params at all.

    Getting this wrong is not loud: pointing the natural-language query at a URL
    shape that endpoint does not accept collapses the page to a single job's
    detail view, which scrapes as one bogus "job" rather than as an error.
    """
    if keyword_mode == "natural_language":
        hours_old = seconds // 3600
        query = {"keywords": f"{keywords} in {location} in last {hours_old} hours"}
        return f"{SEMANTIC_SEARCH_URL}?{urlencode(query)}"

    query = {"keywords": keywords, "location": location, "f_TPR": f"r{seconds}"}
    return f"{SEARCH_URL}?{urlencode(query)}"


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
            pg.goto(build_search_url(keywords, location, seconds, keyword_mode),
                    wait_until="domcontentloaded")
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
