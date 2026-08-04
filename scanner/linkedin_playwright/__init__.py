"""LinkedIn scraper and referral finder using Playwright.

Login uses a visible browser window (headless=False) so LinkedIn doesn't block
it. Session cookies persist to data/playwright_sessions/linkedin.json — login is
only required once, or when the session expires.

Progress/status lines go through _log(), which prints to stdout by default. Call
set_log_fn() before a scrape/login/referral-search to route them into a UI log
panel instead.

This was a single 1,509-line module. It is split by concern now — the public
surface is unchanged and re-exported below, so `from scanner.linkedin_playwright
import search_jobs` keeps working:

    selectors.py     every CSS selector and tuning constant (the fragile part)
    session.py       browser/session lifecycle, login, the logging sink
    jobs.py          search: scrolling, card extraction, pagination
    descriptions.py  the three description-fetch strategies
    people.py        profile scraping and referral-contact discovery
    messaging.py     sending a LinkedIn DM
"""
from .descriptions import _fetch_descriptions_via_semantic_click, _fetch_job_description
from .jobs import (
    SEARCH_URL, SEMANTIC_SEARCH_URL, _extract_cards, _extract_job_id,
    _extract_semantic_cards, _scrape_job_cards, build_search_url, search_jobs,
)
from .messaging import send_linkedin_message
from .people import _degree_rank, _is_real_name, _manager_keywords, _role_keywords, find_referral_contacts
from .session import SESSION_FILE, _launch, _log, _save_session, login, set_log_fn

__all__ = [
    # Public API
    "SESSION_FILE", "login", "search_jobs", "find_referral_contacts",
    "send_linkedin_message", "set_log_fn", "build_search_url",
    "SEARCH_URL", "SEMANTIC_SEARCH_URL",
]
