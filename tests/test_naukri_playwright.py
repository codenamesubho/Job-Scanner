from scanner import naukri_playwright as np_module


class _FakeEl:
    def __init__(self, text="", attrs=None):
        self._text = text
        self._attrs = attrs or {}

    def inner_text(self):
        return self._text

    def get_attribute(self, name):
        return self._attrs.get(name)


class _FakeCard:
    def __init__(self, els: dict):
        self._els = els

    def query_selector(self, selector):
        return self._els.get(selector)


def test_parse_job_card_maps_fields():
    card = _FakeCard({
        "a.title": _FakeEl("Backend Engineer", {"href": "https://naukri.com/job/123"}),
        "a.comp-name": _FakeEl("Acme Inc"),
        "span.locWdth": _FakeEl("Remote"),
    })

    row = np_module._parse_job_card(card)

    assert row["title"] == "Backend Engineer"
    assert row["company"] == "Acme Inc"
    assert row["location"] == "Remote"
    assert row["job_url"] == "https://naukri.com/job/123"
    assert row["site"] == "naukri"
    assert row["is_remote"] == 1
    assert row["id"].startswith("naukri-backendengineer-")


def test_parse_job_card_handles_missing_elements():
    card = _FakeCard({})

    row = np_module._parse_job_card(card)

    assert row["title"] == ""
    assert row["company"] == ""
    assert row["location"] == ""
    assert row["job_url"] == ""
    assert row["is_remote"] == 0


def test_scrape_cards_dedupes_by_job_url():
    card_a = _FakeCard({"a.title": _FakeEl("Job A", {"href": "https://naukri.com/a"})})
    card_b = _FakeCard({"a.title": _FakeEl("Job A dup", {"href": "https://naukri.com/a"})})
    card_c = _FakeCard({"a.title": _FakeEl("Job C", {"href": "https://naukri.com/c"})})

    seen: set[str] = set()
    rows = np_module._scrape_cards([card_a, card_b, card_c], seen)

    assert len(rows) == 2
    assert {r["job_url"] for r in rows} == {"https://naukri.com/a", "https://naukri.com/c"}
