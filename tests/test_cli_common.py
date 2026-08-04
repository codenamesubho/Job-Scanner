"""Tests for the shared CLI argument helpers.

The point of `cli_common` is that every script declares these flags identically,
so these tests mostly guard against the defaults drifting apart again.
"""
import argparse

import pytest

from cli_common import (
    add_force_arg, add_search_args, add_top_arg, criteria_from_args,
)
from scanner.search import SearchCriteria


def _parser():
    return argparse.ArgumentParser()


def test_search_args_fall_back_to_search_criteria_defaults():
    """A script that passes no defaults must land on the same values as its peers,
    rather than inventing its own (which is how 25/72 got hardcoded before)."""
    p = _parser()
    add_search_args(p)
    args = p.parse_args([])

    blank = SearchCriteria(keywords="")
    assert args.results == blank.results
    assert args.hours == blank.hours


def test_search_args_take_supplied_defaults():
    p = _parser()
    add_search_args(p, {"keywords": "kw", "location": "Remote", "results": 5, "hours": 12})
    args = p.parse_args([])

    assert (args.keywords, args.location, args.results, args.hours) == ("kw", "Remote", 5, 12)


def test_partial_defaults_fall_back_per_key():
    """Supplying only some keys must not blank out the rest."""
    p = _parser()
    add_search_args(p, {"keywords": "kw"})
    args = p.parse_args([])

    assert args.keywords == "kw"
    assert args.results == SearchCriteria(keywords="").results


def test_command_line_overrides_defaults():
    p = _parser()
    add_search_args(p, {"keywords": "from-defaults", "results": 5})
    args = p.parse_args(["--keywords", "from-cli", "--results", "99"])

    assert args.keywords == "from-cli"
    assert args.results == 99


def test_results_and_hours_are_typed_as_int():
    p = _parser()
    add_search_args(p)
    args = p.parse_args(["--results", "7", "--hours", "48"])

    assert args.results == 7 and isinstance(args.results, int)
    assert args.hours == 48 and isinstance(args.hours, int)


def test_criteria_from_args_builds_the_kernel_input():
    p = _parser()
    add_search_args(p)
    criteria = criteria_from_args(p.parse_args(
        ["--keywords", "a,b", "--location", "Remote", "--results", "3", "--hours", "9"]))

    assert criteria == SearchCriteria("a,b", "Remote", 3, 9)
    assert criteria.keyword_list() == ["a", "b"]


def test_top_defaults_to_none_meaning_all():
    p = _parser()
    add_top_arg(p, "score")

    assert p.parse_args([]).top is None
    assert p.parse_args(["--top", "50"]).top == 50


def test_force_is_an_off_by_default_flag():
    p = _parser()
    add_force_arg(p, "re-do everything")

    assert p.parse_args([]).force is False
    assert p.parse_args(["--force"]).force is True


@pytest.mark.parametrize("flag", ["--keywords", "--location", "--results", "--hours"])
def test_every_search_flag_is_registered(flag):
    p = _parser()
    add_search_args(p)

    assert flag in p.format_help()
