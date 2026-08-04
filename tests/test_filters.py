"""Tests for the pure DataFrame filter helpers.

`scanner/filters.py` had no coverage at all despite being the layer that decides
which scraped rows a user actually sees.
"""
import pandas as pd

from scanner import filters


def _jobs(rows):
    return pd.DataFrame(rows)


def test_filter_by_keywords_matches_case_insensitively():
    jobs = _jobs([
        {"title": "Senior Backend Engineer"},
        {"title": "Product Manager"},
        {"title": "backend developer"},
    ])

    out = filters.filter_by_keywords(jobs, ["BACKEND"])

    assert list(out["title"]) == ["Senior Backend Engineer", "backend developer"]


def test_filter_by_keywords_is_a_noop_without_keywords():
    jobs = _jobs([{"title": "Anything"}])

    assert len(filters.filter_by_keywords(jobs, [])) == 1


def test_filter_by_exclude_drops_matching_titles():
    jobs = _jobs([
        {"title": "Staff Engineer"},
        {"title": "Junior Engineer"},
        {"title": "Principal Engineer"},
    ])

    out = filters.filter_by_exclude(jobs, ["junior", "principal"])

    assert list(out["title"]) == ["Staff Engineer"]


def test_filter_by_remote_flag_keeps_integer_flagged_rows():
    """is_remote is stored as INTEGER (0/1), which is why the filter compares to 1
    rather than using truthiness — boolean-indexing an int64 column raises."""
    jobs = _jobs([
        {"title": "Remote role", "is_remote": 1},
        {"title": "Onsite role", "is_remote": 0},
    ])

    out = filters.filter_by_remote_flag(jobs)

    assert list(out["title"]) == ["Remote role"]


def test_filter_by_remote_flag_also_handles_bool_dtype():
    """Some sources set is_remote as a real bool; True == 1 keeps those too."""
    jobs = _jobs([
        {"title": "Remote role", "is_remote": True},
        {"title": "Onsite role", "is_remote": False},
    ])

    out = filters.filter_by_remote_flag(jobs)

    assert list(out["title"]) == ["Remote role"]


def test_filters_pass_empty_frames_through_untouched():
    empty = pd.DataFrame()

    assert filters.filter_by_keywords(empty, ["x"]).empty
    assert filters.filter_by_exclude(empty, ["x"]).empty
    assert filters.filter_by_remote_flag(empty).empty
