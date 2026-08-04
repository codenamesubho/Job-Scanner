"""Getting a job's description, by three escalating strategies.

The classic results list can load a description into its side panel on click;
the semantic (natural-language) UI needs a different click + a job-id-scoped
container; and when neither works we navigate to the job's own page. Panel
clicks are ~7x faster than navigation, so navigation is the last resort.
"""
from playwright.sync_api import Page

from .selectors import (
    _DATE_SELECTORS, _DESC_SELECTORS, _JOB_DESCRIPTION_PAGE_SETTLE_MS, _PANEL_SELECTORS,
)
from .session import _log

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
