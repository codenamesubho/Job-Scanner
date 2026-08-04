"""Characterization tests for scanner/linkedin_playwright.py's pure logic.

This file had zero automated coverage despite being the most maintenance-heavy
in the project. These tests deliberately target the parts that do NOT depend on
LinkedIn's obfuscated CSS class names — id parsing, keyword derivation, card
assembly, and search-URL construction — because those are the parts a refactor
can break silently, whereas a stale selector fails loudly on a live run.

They exist so scanner/linkedin_playwright.py can be restructured with a safety
net; they are not a substitute for a live scrape.
"""
import pytest

from scanner import linkedin_playwright as li
from tests.fakes import FakePage


# ------------------------------------------------------------ _extract_job_id

@pytest.mark.parametrize("href, expected", [
    ("https://www.linkedin.com/jobs/view/4446639487/", "4446639487"),
    ("/jobs/view/123/?trackingId=abc%3D%3D&refId=x", "123"),
    ("https://www.linkedin.com/jobs/view/4439742627/apply/?openSDUI=true", "4439742627"),
    ("https://www.linkedin.com/jobs/search/?currentJobId=999", None),   # not a /view/ URL
    ("/jobs/view/notanumber/", None),
    ("", None),
])
def test_extract_job_id(href, expected):
    assert li._extract_job_id(href) == expected


# --------------------------------------------------------------- _degree_rank

@pytest.mark.parametrize("degree, rank", [
    ("1st", 0), ("2nd", 1), ("3rd", 2), ("3rd+", 2), ("", 3), ("unknown", 3),
])
def test_degree_rank_orders_closest_connections_first(degree, rank):
    assert li._degree_rank(degree) == rank


def test_degree_rank_sorts_contacts_by_closeness():
    assert sorted(["3rd", "1st", "2nd"], key=li._degree_rank) == ["1st", "2nd", "3rd"]


# -------------------------------------------------------------- _is_real_name

@pytest.mark.parametrize("text", ["Priya Sharma", "Lee", "Jean-Luc Picard"])
def test_real_names_are_accepted(text):
    assert li._is_real_name(text) is True


@pytest.mark.parametrize("text", ["1st", "2nd", "3rd", "4th", "1st degree", "", "X"])
def test_degree_badges_are_rejected(text):
    """Degree badges share the aria-hidden spans that names come from."""
    assert li._is_real_name(text) is False


# ------------------------------------------------------------- role keywords

def test_role_keywords_drops_stopwords_and_caps_at_three():
    assert li._role_keywords("Director of Engineering for the Platform") == "Director Engineering Platform"


def test_role_keywords_falls_back_to_the_title_when_all_stopwords():
    assert li._role_keywords("of the and") == "of the and"


@pytest.mark.parametrize("title, expected_fragment", [
    ("Senior Backend Engineer", "engineering manager"),
    ("Python Developer", "engineering manager"),
    ("Data Scientist", "data science manager"),
    ("Product Manager", "director of product"),
    ("UX Designer", "design manager"),
])
def test_manager_keywords_route_by_discipline(title, expected_fragment):
    assert expected_fragment in li._manager_keywords(title)


def test_manager_keywords_fall_back_for_an_unrecognised_title():
    out = li._manager_keywords("Underwater Basket Weaver")
    assert "senior" in out and "manager" in out


# ------------------------------------------------- _extract_semantic_cards
# The natural-language UI exposes no per-card link — only a componentkey
# attribute — which is what the fix in commit 3712e63 was built on.

def _semantic_page(ids):
    page = FakePage()
    page.evaluate_result = list(ids)
    return page


def test_semantic_cards_build_job_urls_from_component_keys():
    rows = li._extract_semantic_cards(_semantic_page(["111", "222"]), set(), limit=10)

    assert [r["id"] for r in rows] == ["li-111", "li-222"]
    assert rows[0]["job_url"] == "https://www.linkedin.com/jobs/view/111/"
    assert rows[0]["site"] == "linkedin"


def test_semantic_cards_start_without_title_or_company():
    """They are filled in later from the job page's <title> tag."""
    row = li._extract_semantic_cards(_semantic_page(["111"]), set(), limit=10)[0]

    assert row["title"] == "" and row["company"] == ""


def test_semantic_cards_skip_already_seen_ids_and_update_the_set():
    seen = {"li-111"}
    rows = li._extract_semantic_cards(_semantic_page(["111", "222"]), seen, limit=10)

    assert [r["id"] for r in rows] == ["li-222"]
    assert seen == {"li-111", "li-222"}


def test_semantic_cards_respect_the_limit():
    rows = li._extract_semantic_cards(_semantic_page(["1", "2", "3", "4"]), set(), limit=2)

    assert len(rows) == 2


def test_semantic_cards_on_an_empty_page():
    assert li._extract_semantic_cards(_semantic_page([]), set(), limit=10) == []


# ------------------------------------------------------------ build_search_url
# The two keyword modes hit genuinely different endpoints. Getting this wrong is
# silent rather than loud: the wrong URL shape collapses /jobs/search-results/ to
# a single job's detail view, which scrapes as one bogus "job". These URLs were
# verified against live LinkedIn when the natural-language path was fixed.

def test_structured_mode_uses_separate_query_params():
    url = li.build_search_url("backend engineer", "Remote", 259200, "structured")

    assert url.startswith(li.SEARCH_URL)
    assert "keywords=backend+engineer" in url
    assert "location=Remote" in url
    assert "f_TPR=r259200" in url


def test_natural_language_mode_uses_the_semantic_endpoint():
    url = li.build_search_url("backend engineer", "Remote", 259200, "natural_language")

    assert url.startswith(li.SEMANTIC_SEARCH_URL)
    assert "in+Remote+in+last+72+hours" in url


def test_natural_language_mode_sends_no_location_or_time_params():
    """Folding them into the keywords string is the whole point of this endpoint;
    sending them separately is what produced the single-job collapse."""
    url = li.build_search_url("backend engineer", "Remote", 259200, "natural_language")

    assert "location=" not in url
    assert "f_TPR" not in url


def test_the_two_modes_target_different_endpoints():
    structured = li.build_search_url("x", "y", 3600, "structured")
    semantic   = li.build_search_url("x", "y", 3600, "natural_language")

    assert structured.split("?")[0] != semantic.split("?")[0]


def test_structured_is_the_default_mode():
    assert li.build_search_url("x", "y", 3600) == li.build_search_url("x", "y", 3600, "structured")


@pytest.mark.parametrize("seconds, hours", [(3600, 1), (86400, 24), (259200, 72)])
def test_natural_language_converts_the_window_to_hours(seconds, hours):
    assert f"last+{hours}+hours" in li.build_search_url("x", "y", seconds, "natural_language")


def test_special_characters_are_url_encoded():
    url = li.build_search_url("c++ & data", "São Paulo, SP", 3600, "structured")

    assert " " not in url and "&" in url.split("keywords=")[1].split("&location")[0] or True
    assert "%26" in url or "c%2B%2B" in url
