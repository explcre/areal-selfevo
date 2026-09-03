"""Tests for the reward equations and for the size-matched random controls."""

from __future__ import annotations

import math

import pytest

from ornith_repro.controls import (
    measure_proportions,
    random_scaffold_control,
    random_task_control,
)
from ornith_repro.grpo import (
    degenerate_fraction,
    grpo_advantages,
    predicted_binary_degeneracy,
)
from ornith_repro.guards import GuardViolation
from ornith_repro.rewards import (
    P_STAR_PUBLISHED,
    difficulty_reward,
    harness_reward,
    task_reward,
)
from ornith_repro.types import Scaffold, Task


# ------------------------------------------------------------------ equations
def test_published_p_star_is_used_verbatim():
    assert P_STAR_PUBLISHED == 0.2


def test_difficulty_reward_peaks_exactly_at_p_star():
    assert difficulty_reward(0.2) == pytest.approx(1.0)
    assert difficulty_reward(0.2, sigma=0.05) == pytest.approx(1.0)
    assert difficulty_reward(0.5) < difficulty_reward(0.3) < difficulty_reward(0.2)


def test_difficulty_reward_matches_the_closed_form():
    p, ps, s = 0.5, 0.2, 0.15
    assert difficulty_reward(p, ps, s) == pytest.approx(
        math.exp(-((p - ps) ** 2) / (2 * s**2))
    )


def test_any_zero_factor_annihilates_the_product():
    """The multiplicative form is Ornith's and is load-bearing; keep it exact."""
    assert task_reward(0.0, 1.0, 1.0) == 0.0
    assert task_reward(1.0, 0.0, 1.0) == 0.0
    assert harness_reward(1.0, 1.0, 0.0) == 0.0
    assert task_reward(0.5, 0.5, 0.5) == pytest.approx(0.125)


# ------------------------------------------------------------------ GRPO
def test_degenerate_group_has_exactly_zero_advantage_and_is_flagged():
    g = grpo_advantages([1.0] * 8)
    assert g.degenerate is True
    assert g.reward_std == 0.0
    assert all(a == 0.0 for a in g.advantages), "exactly zero, not merely small"


def test_informative_group_is_not_flagged_and_advantages_sum_to_zero():
    g = grpo_advantages([1.0] * 2 + [0.0] * 6)
    assert g.degenerate is False
    assert sum(g.advantages) == pytest.approx(0.0)
    assert g.reward_std > 0.0


def test_binary_degeneracy_prediction_matches_simulation():
    """theta^G + (1-theta)^G, checked against a simulation rather than asserted."""
    import random

    rng = random.Random(7)
    for theta in (0.2, 0.5):
        trials = 20000
        deg = sum(
            1
            for _ in range(trials)
            if grpo_advantages(
                [1.0 if rng.random() < theta else 0.0 for _ in range(8)]
            ).degenerate
        )
        assert deg / trials == pytest.approx(
            predicted_binary_degeneracy(theta, 8), abs=0.01
        )


def test_ornith_p_star_wastes_far_more_groups_than_p_star_half():
    """The published p*=0.2 sits where ~21x more groups carry zero gradient."""
    at_02 = predicted_binary_degeneracy(0.2, 8)
    at_05 = predicted_binary_degeneracy(0.5, 8)
    assert at_02 == pytest.approx(0.16777, abs=1e-4)
    assert at_05 == pytest.approx(0.00781, abs=1e-4)
    assert at_02 / at_05 == pytest.approx(21.47, abs=0.05)


def test_degenerate_fraction_refuses_empty_batch():
    with pytest.raises(ValueError, match="no groups"):
        degenerate_fraction([])


# ------------------------------------------------------------------ controls
def _tasks(spec):
    return [
        Task(task_id=f"t{i}", text=f"task text number {i}", family=fam)
        for i, fam in enumerate(spec)
    ]


def test_control_matches_the_treatments_measured_proportions_not_uniform():
    treatment = _tasks(["algebra"] * 8 + ["geometry"] * 2)
    target = measure_proportions(treatment, "family")
    assert target == {"algebra": 0.8, "geometry": 0.2}

    pool = _tasks(["algebra"] * 50 + ["geometry"] * 50)
    control = random_task_control(treatment, pool, key="family", seed=1)

    assert len(control) == len(treatment), "size-matched"
    assert measure_proportions(control, "family") == pytest.approx(target)
    assert all(t.source == "pool" for t in control), "arms must never be mixable"


def test_control_construction_fails_loudly_if_pool_lacks_a_stratum():
    treatment = _tasks(["algebra"] * 5 + ["geometry"] * 5)
    pool = _tasks(["algebra"] * 20)
    with pytest.raises(ValueError, match="no items with family"):
        random_task_control(treatment, pool, key="family", seed=1)


def test_scaffold_control_matches_grader_kind_proportions():
    treatment = [
        Scaffold(scaffold_id=f"s{i}", instructions="do it",
                 grader_kind="exact" if i < 6 else "any")
        for i in range(10)
    ]
    pool = [
        Scaffold(scaffold_id=f"p{i}", instructions="do it",
                 grader_kind="exact" if i % 2 == 0 else "any")
        for i in range(40)
    ]
    control = random_scaffold_control(treatment, pool, key="grader_kind", seed=3)
    assert len(control) == 10
    assert measure_proportions(control, "grader_kind") == pytest.approx(
        {"exact": 0.6, "any": 0.4}
    )


def test_a_deliberately_mismatched_control_is_rejected():
    """Mutation test for the matching guard itself."""
    from ornith_repro.guards import assert_proportions_match

    with pytest.raises(GuardViolation):
        assert_proportions_match({"a": 0.8, "b": 0.2}, {"a": 0.6, "b": 0.4}, tol=0.02)


def test_measure_proportions_refuses_empty_input():
    with pytest.raises(ValueError, match="empty set"):
        measure_proportions([], "family")
