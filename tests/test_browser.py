from scanner import browser
from tests.fakes import FakePlaywright


def test_constants_are_non_empty():
    assert browser.USER_AGENT
    assert browser.STEALTH_SCRIPT.strip()
    assert browser.STEALTH_SCRIPT_WITH_PLUGINS.strip()
    assert browser.LAUNCH_ARGS


def test_stealth_script_with_plugins_is_a_superset():
    assert "webdriver" in browser.STEALTH_SCRIPT_WITH_PLUGINS
    assert "plugins" in browser.STEALTH_SCRIPT_WITH_PLUGINS
    assert "plugins" not in browser.STEALTH_SCRIPT


def test_launch_stealth_browser_tries_real_chrome_first():
    p = FakePlaywright(chrome_fails=False)

    browser_obj, context = browser.launch_stealth_browser(p, headless=True)

    assert len(p.launch_calls) == 1
    assert p.launch_calls[0]["channel"] == "chrome"
    assert context.kwargs["user_agent"] == browser.USER_AGENT
    assert context.init_scripts == [browser.STEALTH_SCRIPT]


def test_launch_stealth_browser_falls_back_without_real_chrome():
    p = FakePlaywright(chrome_fails=True)

    browser_obj, context = browser.launch_stealth_browser(p, headless=True)

    assert len(p.launch_calls) == 2
    assert p.launch_calls[1].get("channel") is None


def test_launch_stealth_browser_skips_real_chrome_when_disabled():
    p = FakePlaywright(chrome_fails=False)

    browser.launch_stealth_browser(p, headless=True, try_real_chrome=False)

    assert len(p.launch_calls) == 1
    assert "channel" not in p.launch_calls[0]


def test_launch_stealth_browser_passes_storage_state_and_timezone():
    p = FakePlaywright()

    _, context = browser.launch_stealth_browser(
        p, storage_state_path="session.json", timezone_id="Asia/Kolkata",
        viewport={"width": 1, "height": 2},
    )

    assert context.kwargs["storage_state"] == "session.json"
    assert context.kwargs["timezone_id"] == "Asia/Kolkata"
    assert context.kwargs["viewport"] == {"width": 1, "height": 2}


# ------------------------------------------------------------- SessionBrowser
# One launch path replacing the three near-identical `_launch()` functions that
# linkedin_playwright, naukri_playwright and apply each carried.

def _session_browser(tmp_path, **kw):
    from scanner.browser import SessionBrowser
    return SessionBrowser(tmp_path / "session.json", **kw)


def test_session_browser_loads_a_saved_session_when_the_file_exists(tmp_path):
    sb = _session_browser(tmp_path)
    sb.session_file.write_text("{}")
    p = FakePlaywright()

    _, context = sb.launch(p, headless=True)

    assert context.kwargs["storage_state"] == str(sb.session_file)


def test_session_browser_starts_clean_when_there_is_no_session(tmp_path):
    _, context = _session_browser(tmp_path).launch(FakePlaywright(), headless=True)

    # launch_stealth_browser omits the key entirely rather than passing None.
    assert "storage_state" not in context.kwargs


def test_load_session_false_ignores_an_existing_session(tmp_path):
    """login() re-authenticates from scratch rather than reusing stale cookies."""
    sb = _session_browser(tmp_path)
    sb.session_file.write_text("{}")

    _, context = sb.launch(FakePlaywright(), headless=True, load_session=False)

    assert "storage_state" not in context.kwargs


def test_headless_none_follows_the_debug_headful_setting(tmp_path, monkeypatch):
    from scanner import browser as browser_mod

    monkeypatch.setattr(browser_mod, "debug_headful", lambda: True)
    p = FakePlaywright()
    _session_browser(tmp_path).launch(p, headless=None)
    assert p.launch_calls[0]["headless"] is False      # headful for debugging

    monkeypatch.setattr(browser_mod, "debug_headful", lambda: False)
    p2 = FakePlaywright()
    _session_browser(tmp_path).launch(p2, headless=None)
    assert p2.launch_calls[0]["headless"] is True


def test_explicit_headless_overrides_the_debug_setting(tmp_path, monkeypatch):
    from scanner import browser as browser_mod
    monkeypatch.setattr(browser_mod, "debug_headful", lambda: True)

    p = FakePlaywright()
    _session_browser(tmp_path).launch(p, headless=True)

    assert p.launch_calls[0]["headless"] is True


def test_site_specific_options_are_passed_through(tmp_path):
    """LinkedIn needs a timezone, the plugins stealth script and real Chrome;
    Naukri needs none of them."""
    from scanner import browser as browser_mod

    sb = _session_browser(tmp_path, timezone_id="Asia/Kolkata",
                           stealth_script=browser_mod.STEALTH_SCRIPT_WITH_PLUGINS,
                           try_real_chrome=True)
    p = FakePlaywright()
    _, context = sb.launch(p, headless=True)

    assert context.kwargs["timezone_id"] == "Asia/Kolkata"
    assert context.init_scripts == [browser_mod.STEALTH_SCRIPT_WITH_PLUGINS]
    assert p.launch_calls[0]["channel"] == "chrome"


def test_defaults_omit_the_optional_site_settings(tmp_path):
    p = FakePlaywright()
    _, context = _session_browser(tmp_path).launch(p, headless=True)

    assert context.kwargs.get("timezone_id") is None
    # Real Chrome IS attempted by default — Naukri relies on it to avoid a 403
    # from bot detection that fingerprints Playwright's bundled Chromium.
    assert p.launch_calls[0]["channel"] == "chrome"


def test_try_real_chrome_false_actually_disables_it(tmp_path):
    """Regression: forwarding this kwarg only when true made False a no-op,
    because launch_stealth_browser's own default is True."""
    p = FakePlaywright()
    _session_browser(tmp_path, try_real_chrome=False).launch(p, headless=True)

    assert "channel" not in p.launch_calls[0]


def test_save_writes_the_session_and_creates_its_directory(tmp_path):
    sb = _session_browser(tmp_path / "nested" / "dir")
    written = {}

    class _Ctx:
        def storage_state(self, path):
            written["path"] = path

    sb.save(_Ctx())

    assert written["path"] == str(sb.session_file)
    assert sb.session_file.parent.is_dir()


def test_has_session_reflects_the_file(tmp_path):
    sb = _session_browser(tmp_path)
    assert sb.has_session() is False
    sb.session_file.write_text("{}")
    assert sb.has_session() is True
