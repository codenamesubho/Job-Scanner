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
