"""Shared Playwright launch configuration for the browser-automation modules
(linkedin_playwright.py, naukri_playwright.py, apply.py). Each of those
previously redefined the same user agent, stealth script, and launch args.
"""

import os


def debug_headful() -> bool:
    """True if SCAN_DEBUG_HEADFUL=1 (or true/yes) is set in the environment —
    forces LinkedIn/Naukri scan browsers to launch visibly instead of
    headless, so a failing/flaky scan can be watched live. Set it in .env."""
    return os.getenv("SCAN_DEBUG_HEADFUL", "").strip().lower() in ("1", "true", "yes")


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.navigator.chrome = {runtime: {}};
"""

# linkedin_playwright.py additionally masks the plugins array and several
# other properties that differ between headless and headed Chrome and that
# LinkedIn's bot detection checks: an empty window.chrome object, mismatched
# permissions.query behavior for Notification, the SwiftShader/ANGLE
# software-renderer WebGL string headless Chromium reports, and outerWidth/
# outerHeight reporting 0 (no real browser chrome around a headless window).
STEALTH_SCRIPT_WITH_PLUGINS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.navigator.chrome = {
    runtime: {
        connect: () => {},
        sendMessage: () => {},
    },
    loadTimes: () => {},
    csi: () => {},
};

const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters)
);

const getParameterProxy = (getParameter) => function (parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.call(this, parameter);
};
if (window.WebGLRenderingContext) {
    WebGLRenderingContext.prototype.getParameter = getParameterProxy(WebGLRenderingContext.prototype.getParameter);
}
if (window.WebGL2RenderingContext) {
    WebGL2RenderingContext.prototype.getParameter = getParameterProxy(WebGL2RenderingContext.prototype.getParameter);
}

if (!window.outerWidth || !window.outerHeight) {
    Object.defineProperty(window, 'outerWidth', {get: () => window.innerWidth});
    Object.defineProperty(window, 'outerHeight', {get: () => window.innerHeight + 85});
}
"""

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]


def launch_stealth_browser(
    p,
    *,
    headless: bool = True,
    storage_state_path: str | None = None,
    viewport: dict | None = None,
    timezone_id: str | None = None,
    stealth_script: str = STEALTH_SCRIPT,
    try_real_chrome: bool = True,
):
    """Launch a Chromium browser/context configured to blend in with a real
    browser: fixed user agent, viewport, locale, and a stealth init script.

    Several ATS/job-board pages run Akamai/Cloudflare bot detection that
    blocks Playwright's bundled Chromium specifically (confirmed with
    Naukri: 403 via bundled Chromium, 200 via a plain HTTP request with the
    same UA/IP) — launching the real installed Chrome binary via
    channel="chrome" avoids that fingerprint mismatch. Falls back to bundled
    Chromium if Chrome isn't installed on this machine.
    """
    if try_real_chrome:
        try:
            browser = p.chromium.launch(headless=headless, channel="chrome", args=LAUNCH_ARGS)
        except Exception:
            browser = p.chromium.launch(headless=headless, args=LAUNCH_ARGS)
    else:
        browser = p.chromium.launch(headless=headless, args=LAUNCH_ARGS)

    ctx_kwargs: dict = {
        "user_agent": USER_AGENT,
        "viewport": viewport or {"width": 1280, "height": 900},
        "locale": "en-US",
    }
    if timezone_id:
        ctx_kwargs["timezone_id"] = timezone_id
    if storage_state_path:
        ctx_kwargs["storage_state"] = storage_state_path

    context = browser.new_context(**ctx_kwargs)
    context.add_init_script(stealth_script)
    return browser, context


class SessionBrowser:
    """Launch config for one site, bound to where that site's session is saved.

    `linkedin_playwright`, `naukri_playwright` and `apply` each had their own
    `_launch()` (and two had their own `_save_session()`) wrapping the single
    `launch_stealth_browser()` below. They agreed on everything that matters —
    resolve `headless=None` from SCAN_DEBUG_HEADFUL, load the saved session if
    the file exists — and differed only in viewport, timezone and stealth
    options, which are constructor arguments here.
    """

    def __init__(self, session_file, *, viewport=None, timezone_id=None,
                 stealth_script=None, try_real_chrome=True):
        self.session_file = session_file
        self.viewport = viewport or {"width": 1280, "height": 800}
        self.timezone_id = timezone_id
        self.stealth_script = stealth_script
        self.try_real_chrome = try_real_chrome

    def has_session(self) -> bool:
        return self.session_file is not None and self.session_file.exists()

    def launch(self, p, *, headless: bool | None = None, load_session: bool = True):
        """Launch a browser + context for this site.

        `headless=None` (the default at every scan call site) resolves to
        `not debug_headful()`, so setting SCAN_DEBUG_HEADFUL=1 in .env makes
        scan browsers launch visibly. Callers with a hard requirement — login()
        is always headed — pass `headless` explicitly and are unaffected.
        """
        if headless is None:
            headless = not debug_headful()
        kwargs = {
            "headless": headless,
            "storage_state_path": str(self.session_file) if (load_session and self.has_session()) else None,
            "viewport": self.viewport,
            # Always passed explicitly: if this were only forwarded when true,
            # try_real_chrome=False could never actually disable it, since
            # launch_stealth_browser's own default is True.
            "try_real_chrome": self.try_real_chrome,
        }
        if self.timezone_id:
            kwargs["timezone_id"] = self.timezone_id
        if self.stealth_script:
            kwargs["stealth_script"] = self.stealth_script
        return launch_stealth_browser(p, **kwargs)

    def save(self, context) -> None:
        """Persist cookies/localStorage so the next run skips the login flow."""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(self.session_file))
