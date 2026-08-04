"""Fake Playwright objects for tests that exercise scraping logic without a browser.

The real Playwright API surface these stand in for is small and stable: elements
answer `inner_text()`/`get_attribute()`, and pages/elements answer
`query_selector()`/`query_selector_all()`. That is enough to test every field-mapping
and parsing helper in the scrapers, which is where the actual logic (and the actual
bugs) live — as opposed to the selector strings themselves, which only a live run
can validate.

These were previously two separate, partly-overlapping sets: `_FakeContext`/
`_FakeBrowser`/`_FakePlaywright` in `test_browser.py` and `_FakeEl`/`_FakeCard` in
`test_naukri_playwright.py`.
"""


class FakeElement:
    """A single DOM element: text, attributes, and optional children by selector."""

    def __init__(self, text="", attrs=None, children=None):
        self._text = text
        self._attrs = attrs or {}
        self._children = children or {}

    def inner_text(self):
        return self._text

    def get_attribute(self, name):
        return self._attrs.get(name)

    def query_selector(self, selector):
        found = self._children.get(selector)
        return found[0] if isinstance(found, list) else found

    def query_selector_all(self, selector):
        found = self._children.get(selector)
        if found is None:
            return []
        return found if isinstance(found, list) else [found]


class FakePage:
    """A page whose `query_selector*` calls are answered from a selector->element map.

    `elements` maps a selector string to either one FakeElement or a list of them.
    `url` and `title_text` back the `url` property and `title()` method that the
    LinkedIn scraper reads to detect navigation and pull title/company.
    """

    def __init__(self, elements=None, url="", title_text=""):
        self._elements = elements or {}
        self.url = url
        self._title = title_text
        self.evaluated = []
        self.timeouts_waited = []

    def title(self):
        return self._title

    def query_selector(self, selector):
        found = self._elements.get(selector)
        return found[0] if isinstance(found, list) else found

    def query_selector_all(self, selector):
        found = self._elements.get(selector)
        if found is None:
            return []
        return found if isinstance(found, list) else [found]

    def evaluate(self, script, arg=None):
        """Record the script and return whatever `evaluate_result` was set to.

        Scrapers use `evaluate()` for DOM work that has no query_selector
        equivalent (scroll position, collecting componentkey attributes).
        """
        self.evaluated.append(script)
        return getattr(self, "evaluate_result", None)

    def wait_for_timeout(self, ms):
        self.timeouts_waited.append(ms)


class FakeContext:
    def __init__(self):
        self.init_scripts = []
        self.kwargs = {}

    def add_init_script(self, script):
        self.init_scripts.append(script)


class FakeBrowser:
    def __init__(self):
        self.contexts = []

    def new_context(self, **kwargs):
        ctx = FakeContext()
        ctx.kwargs = kwargs
        self.contexts.append(ctx)
        return ctx


class FakePlaywright:
    """Stands in for a `sync_playwright()` handle.

    `chrome_fails=True` simulates the "real Chrome channel not installed" case, so
    tests can assert the launch helper falls back to bundled Chromium.
    """

    def __init__(self, chrome_fails=False):
        self.chrome_fails = chrome_fails
        self.launch_calls = []

        outer = self

        class _Chromium:
            @staticmethod
            def launch(**kwargs):
                return outer._launch_impl(**kwargs)

        self.chromium = _Chromium()

    def _launch_impl(self, **kwargs):
        self.launch_calls.append(kwargs)
        if self.chrome_fails and kwargs.get("channel") == "chrome":
            raise RuntimeError("no chrome installed")
        return FakeBrowser()
