"""Finding referral contacts: profile scraping, company filters, search strategies."""
import re

from playwright.sync_api import Page, sync_playwright
from urllib.parse import quote_plus, urlencode

from .session import _launch, _save_session, _wait_for_search_results

def _degree_rank(degree: str) -> int:
    if "1st" in degree: return 0
    if "2nd" in degree: return 1
    if "3rd" in degree: return 2
    return 3


_DEGREE_RE = re.compile(r'^\d+(st|nd|rd|th)(\s|$)', re.I)


def _is_real_name(text: str) -> bool:
    """Return False for degree badges (1st, 2nd, 3rd…) that share aria-hidden spans."""
    return len(text) > 1 and not _DEGREE_RE.match(text)

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
