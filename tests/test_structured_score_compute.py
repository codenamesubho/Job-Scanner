import itertools
import re

from scanner import llm
from scanner.llm import (
    StructuredCompanyJudgment,
    StructuredJobScoreItem,
    StructuredRemoteJudgment,
    StructuredRequirementJudgment,
    StructuredRoleJudgment,
    StructuredSkillItem,
    StructuredSkillsJudgment,
)


def _req(text, fraction):
    return StructuredRequirementJudgment(requirement=text, match_fraction=fraction, note="")


def _item(core_fit="core_match", judgments=None, nice=None, company_tier="big_or_funded",
          remote_tier="remote", role_tier="at_or_below_ceiling"):
    return StructuredJobScoreItem(
        id="job-1",
        skills=StructuredSkillsJudgment(
            core_fit=core_fit, core_fit_reason="matches",
            must_have_judgments=judgments or [],
            nice_to_have_matches=[StructuredSkillItem(name=n, note="") for n in (nice or [])],
        ),
        company=StructuredCompanyJudgment(tier=company_tier, reason=""),
        remote=StructuredRemoteJudgment(tier=remote_tier, reason=""),
        role=StructuredRoleJudgment(tier=role_tier, reason=""),
        overall_reason="overall",
    )


def _job(min_yoe=None, is_remote=False):
    return {"id": "job-1", "requirements": {"min_yoe": min_yoe}, "is_remote": is_remote,
            "title": "Backend Engineer", "company": "Acme"}


def _resume(years_exp=8.0):
    return {"years_exp": years_exp}


def test_core_match_no_must_haves_defaults_to_full_coverage():
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=[]), _job(), _resume())
    assert bd["skills"]["score"] == 60  # 35 + 25 (nothing stated -> no bar to fail)


def test_core_match_all_matched_scores_top_of_band():
    judgments = [_req(f"req{i}", True) for i in range(6)]
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    assert bd["skills"]["score"] == 60  # 35 + 25 (ratio = 1.0)


def test_core_match_all_gaps_hits_coverage_floor():
    judgments = [_req(f"req{i}", False) for i in range(6)]
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    assert bd["skills"]["score"] == 45  # 35 + max(10, round(25*0/6))=10
    assert bd["skills"]["coverage_bonus"] == 10


def test_mixed_ratio_matches_expected_score():
    judgments = [_req(f"req{i}", i < 6) for i in range(12)]  # 6/12 matched
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    # round(25*6/12) == round(12.5) == 12 (Python banker's rounding), above the 10 floor
    assert bd["skills"]["coverage_bonus"] == 12
    assert bd["skills"]["score"] == 47


def test_adjacent_tier_no_floor():
    judgments = [_req(f"req{i}", False) for i in range(5)]
    bd = llm._compute_structured_breakdown(_item(core_fit="adjacent", judgments=judgments), _job(), _resume())
    assert bd["skills"]["score"] == 20  # 20 + max(0, round(25*0/5))=0


def test_mismatch_tier_scores_low():
    bd = llm._compute_structured_breakdown(_item(core_fit="mismatch", judgments=[]), _job(), _resume())
    assert bd["skills"]["score"] == 25  # 0 + 25 (nothing stated)


def test_or_requirement_is_one_matched_item_not_split_into_gaps():
    """Regression test for the actual bug reported: 'Strong expertise in
    Java, Python, Node.js, or Ruby/Rails' must be ONE judgment (matched via
    Python), not four independent items where three show up as gaps."""
    judgments = [_req("Strong expertise in Java, Python, Node.js, or Ruby/Rails", True)]
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    assert bd["skills"]["matched_requirements"] == ["Strong expertise in Java, Python, Node.js, or Ruby/Rails"]
    assert bd["skills"]["gap_requirements"] == []
    assert bd["skills"]["score"] == 60  # 1/1 matched -> full coverage, not penalized for Java/Node/Ruby


def test_partial_credit_fractions_contribute_proportionally():
    # Single requirement, one fraction at a time -> matched_credit == the fraction itself.
    for fraction in (0.0, 0.4, 0.6, 0.8, 1.0):
        judgments = [_req("bundled requirement", fraction)]
        bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
        assert bd["skills"]["matched_credit"] == fraction


def test_kubernetes_docker_partial_match_regression():
    """Regression test for the real case reported: an "and"-bundled
    requirement ("microservices, API design, Kubernetes, Docker, messaging
    systems, and distributed architectures") where the candidate covers
    most components and Docker experience is a clear adjacent/learnable
    base for the missing Kubernetes piece — this must land as a PARTIAL
    match (0.4/0.6/0.8), not a full 0.0 gap that wipes out credit for
    everything else genuinely covered in the bundle."""
    judgments = [_req(
        "Deep understanding of microservices, API design, Kubernetes, Docker, "
        "messaging systems, and distributed architectures", 0.6,
    )]
    bd = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    assert bd["skills"]["gap_requirements"] == []  # NOT filed as a full gap
    assert len(bd["skills"]["partial_requirements"]) == 1
    assert bd["skills"]["partial_requirements"][0][1] == 0.6
    assert bd["skills"]["matched_credit"] == 0.6
    # Strictly better than if it had been judged a full 0.0 gap.
    full_gap_bd = llm._compute_structured_breakdown(
        _item(core_fit="core_match", judgments=[_req("same requirement", 0.0)]), _job(), _resume(),
    )
    assert bd["skills"]["score"] > full_gap_bd["skills"]["score"]


def test_nice_to_have_matches_add_bonus_but_never_subtract():
    judgments = [_req(f"req{i}", i < 2) for i in range(5)]  # 2/5 matched -> base coverage = 25*2/5 = 10.0 exactly

    no_bonus = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    with_bonus = llm._compute_structured_breakdown(
        _item(core_fit="core_match", judgments=judgments, nice=["docker", "graphql", "terraform"]), _job(), _resume(),
    )
    assert with_bonus["skills"]["score"] > no_bonus["skills"]["score"]
    assert with_bonus["skills"]["score"] - no_bonus["skills"]["score"] == 3  # capped bonus


def test_nice_to_have_bonus_capped_at_three():
    judgments = [_req(f"req{i}", True) for i in range(1)] + [_req("gap", False)] * 5  # base coverage well below max
    bd = llm._compute_structured_breakdown(
        _item(core_fit="adjacent", judgments=judgments, nice=["a", "b", "c", "d", "e", "f"]), _job(), _resume(),
    )
    # base_coverage = round(25*1/6) = 4; bonus capped at 3 -> coverage_bonus = 7
    assert bd["skills"]["coverage_bonus"] == 7


def test_missing_nice_to_haves_do_not_lower_score_vs_baseline():
    judgments = [_req(f"req{i}", True) for i in range(3)]
    baseline = llm._compute_structured_breakdown(_item(core_fit="core_match", judgments=judgments), _job(), _resume())
    fewer_nice = llm._compute_structured_breakdown(
        _item(core_fit="core_match", judgments=judgments, nice=[]), _job(), _resume(),
    )
    assert fewer_nice["skills"]["score"] == baseline["skills"]["score"]


def test_hard_gate_caps_skills_even_with_perfect_core_fit():
    judgments = [_req(f"req{i}", True) for i in range(3)]
    bd = llm._compute_structured_breakdown(
        _item(core_fit="core_match", judgments=judgments), _job(min_yoe=12), _resume(years_exp=5),
    )
    assert bd["skills"]["score"] == 15
    assert bd["skills"]["hard_gate_triggered"] is True
    assert "HARD GATE" in bd["skills"]["reason"]
    assert "12" in bd["skills"]["reason"] and "5" in bd["skills"]["reason"]


def test_hard_gate_not_triggered_when_years_exp_meets_minimum():
    judgments = [_req(f"req{i}", True) for i in range(3)]
    bd = llm._compute_structured_breakdown(
        _item(core_fit="core_match", judgments=judgments), _job(min_yoe=5), _resume(years_exp=8),
    )
    assert bd["skills"]["score"] == 60
    assert bd["skills"]["hard_gate_triggered"] is False


def test_remote_is_remote_flag_overrides_llm_onsite_judgment():
    bd = llm._compute_structured_breakdown(_item(remote_tier="onsite"), _job(is_remote=True), _resume())
    assert bd["remote"]["score"] == 10


def test_remote_uses_llm_tier_when_is_remote_flag_false():
    bd = llm._compute_structured_breakdown(_item(remote_tier="hybrid"), _job(is_remote=False), _resume())
    assert bd["remote"]["score"] == 5


def test_company_tier_scores():
    for tier, expected in [("big_or_funded", 10), ("mid_sized", 7), ("smaller_startup", 5), ("unknown", 0)]:
        bd = llm._compute_structured_breakdown(_item(company_tier=tier), _job(), _resume())
        assert bd["company"]["score"] == expected


def test_role_tier_scores():
    for tier, expected in [("at_or_below_ceiling", 18), ("ambiguous_scale", 11), ("exceeds_ceiling", 3)]:
        bd = llm._compute_structured_breakdown(_item(role_tier=tier), _job(), _resume())
        assert bd["role"]["score"] == expected


def test_must_have_judgments_truncated_to_twelve_even_if_llm_names_more():
    judgment = StructuredSkillsJudgment(
        core_fit="core_match", core_fit_reason="",
        must_have_judgments=[_req(f"req{i}", True) for i in range(15)],
    )
    assert len(judgment.must_have_judgments) == 12


def test_nice_to_have_matches_truncated_to_six_even_if_llm_names_more():
    judgment = StructuredSkillsJudgment(
        core_fit="core_match", core_fit_reason="",
        nice_to_have_matches=[StructuredSkillItem(name=f"nice{i}") for i in range(9)],
    )
    assert len(judgment.nice_to_have_matches) == 6


def test_reason_arithmetic_matches_returned_score_regression():
    """Regression test for the original arithmetic-divergence bug: the
    reason string's stated arithmetic must equal the returned pre-gate
    score, across many tier/fraction-mix/nice-to-have/remote combinations."""
    fraction_mixes = [
        [],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.6, 0.8, 1.0, 0.0, 0.0],
        [0.6] * 6,
    ]
    for core_fit, fractions, n_nice, is_remote, remote_tier in itertools.product(
        ["core_match", "adjacent", "mismatch"], fraction_mixes, [0, 2],
        [True, False], ["remote", "hybrid", "onsite"],
    ):
        judgments = [_req(f"req{i}", f) for i, f in enumerate(fractions)]
        item = _item(
            core_fit=core_fit, judgments=judgments,
            nice=[f"n{i}" for i in range(n_nice)], remote_tier=remote_tier,
        )
        bd = llm._compute_structured_breakdown(item, _job(is_remote=is_remote), _resume())
        # pre_gate_score always directly follows the coverage-bonus clause's
        # closing paren, e.g. "...gaps: x) = 45[. HARD GATE: ...]" — match
        # that first "= <digits>" rather than splitting on "." (partial
        # fractions like "0.6" contain periods too, so a naive split breaks).
        m = re.search(r"\)\s*=\s*(\d+)", bd["skills"]["reason"])
        assert m and int(m.group(1)) == bd["skills"]["core_base"] + bd["skills"]["coverage_bonus"]
