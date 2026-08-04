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
