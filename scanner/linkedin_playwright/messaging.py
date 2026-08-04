"""Sending a referral message as a LinkedIn DM."""
import threading

from playwright.sync_api import sync_playwright

from .session import _launch, _save_session

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
