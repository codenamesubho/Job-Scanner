from scanner import browser


def test_constants_are_non_empty():
    assert browser.USER_AGENT
    assert browser.STEALTH_SCRIPT.strip()
    assert browser.STEALTH_SCRIPT_WITH_PLUGINS.strip()
    assert browser.LAUNCH_ARGS


def test_stealth_script_with_plugins_is_a_superset():
    assert "webdriver" in browser.STEALTH_SCRIPT_WITH_PLUGINS
    assert "plugins" in browser.STEALTH_SCRIPT_WITH_PLUGINS
    assert "plugins" not in browser.STEALTH_SCRIPT


class _FakeContext:
    def __init__(self):
        self.init_scripts = []

    def add_init_script(self, script):
        self.init_scripts.append(script)


class _FakeBrowser:
    def __init__(self):
        self.contexts = []

    def new_context(self, **kwargs):
        ctx = _FakeContext()
        ctx.kwargs = kwargs
        self.contexts.append(ctx)
        return ctx


class _FakePlaywright:
    def __init__(self, chrome_fails=False):
        self.chrome_fails = chrome_fails
        self.launch_calls = []

    class chromium:
        pass

    def _launch_impl(self, **kwargs):
        self.launch_calls.append(kwargs)
        if self.chrome_fails and kwargs.get("channel") == "chrome":
            raise RuntimeError("no chrome installed")
        return _FakeBrowser()


def _make_fake_playwright(chrome_fails=False):
    p = _FakePlaywright(chrome_fails=chrome_fails)
    p.chromium = type("chromium", (), {"launch": staticmethod(p._launch_impl)})()
    return p


def test_launch_stealth_browser_tries_real_chrome_first():
    p = _make_fake_playwright(chrome_fails=False)

    browser_obj, context = browser.launch_stealth_browser(p, headless=True)

    assert len(p.launch_calls) == 1
    assert p.launch_calls[0]["channel"] == "chrome"
    assert context.kwargs["user_agent"] == browser.USER_AGENT
    assert context.init_scripts == [browser.STEALTH_SCRIPT]


def test_launch_stealth_browser_falls_back_without_real_chrome():
    p = _make_fake_playwright(chrome_fails=True)

    browser_obj, context = browser.launch_stealth_browser(p, headless=True)

    assert len(p.launch_calls) == 2
    assert p.launch_calls[1].get("channel") is None


def test_launch_stealth_browser_skips_real_chrome_when_disabled():
    p = _make_fake_playwright(chrome_fails=False)

    browser.launch_stealth_browser(p, headless=True, try_real_chrome=False)

    assert len(p.launch_calls) == 1
    assert "channel" not in p.launch_calls[0]


def test_launch_stealth_browser_passes_storage_state_and_timezone():
    p = _make_fake_playwright()

    _, context = browser.launch_stealth_browser(
        p, storage_state_path="session.json", timezone_id="Asia/Kolkata",
        viewport={"width": 1, "height": 2},
    )

    assert context.kwargs["storage_state"] == "session.json"
    assert context.kwargs["timezone_id"] == "Asia/Kolkata"
    assert context.kwargs["viewport"] == {"width": 1, "height": 2}
