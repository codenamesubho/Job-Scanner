"""Browser session, login, and the progress-logging sink.

`_log_fn` is deliberately still module-level state rather than per-session: the
scrapers are driven one at a time (handle_scan_all runs LinkedIn login alone,
specifically to avoid concurrent LinkedIn traffic), and threading a session
object through every scraping helper to remove a global that nothing currently
races would be a large change for a theoretical gain. It stays a known
limitation rather than a pretend fix.
"""
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from ..browser import STEALTH_SCRIPT_WITH_PLUGINS, SessionBrowser

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

_SESSION = SessionBrowser(
    SESSION_FILE,
    viewport={"width": 1280, "height": 800},
    timezone_id="Asia/Kolkata",
    stealth_script=STEALTH_SCRIPT_WITH_PLUGINS,
    try_real_chrome=True,
)


def _launch(p, *, headless: bool | None = None, load_session: bool = True) -> tuple[Browser, BrowserContext]:
    return _SESSION.launch(p, headless=headless, load_session=load_session)


def _save_session(context: BrowserContext) -> None:
    _SESSION.save(context)


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
