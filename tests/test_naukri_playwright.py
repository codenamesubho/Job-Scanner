from scanner import naukri_playwright as np_module
from tests.fakes import FakeElement, FakePage


def test_parse_job_card_maps_fields():
    card = FakePage({
        "a.title": FakeElement("Backend Engineer", {"href": "https://naukri.com/job/123"}),
        "a.comp-name": FakeElement("Acme Inc"),
        "span.locWdth": FakeElement("Remote"),
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
    card = FakePage({})

    row = np_module._parse_job_card(card)

    assert row["title"] == ""
    assert row["company"] == ""
    assert row["location"] == ""
    assert row["job_url"] == ""
    assert row["is_remote"] == 0


def test_scrape_cards_dedupes_by_job_url():
    card_a = FakePage({"a.title": FakeElement("Job A", {"href": "https://naukri.com/a"})})
    card_b = FakePage({"a.title": FakeElement("Job A dup", {"href": "https://naukri.com/a"})})
    card_c = FakePage({"a.title": FakeElement("Job C", {"href": "https://naukri.com/c"})})

    seen: set[str] = set()
    rows = np_module._scrape_cards([card_a, card_b, card_c], seen)

    assert len(rows) == 2
    assert {r["job_url"] for r in rows} == {"https://naukri.com/a", "https://naukri.com/c"}
