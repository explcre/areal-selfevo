"""Mutation tests: every guard must be OBSERVED to fire on the input it refuses.

A guard that has never been seen to fail is not evidence. Three guards in one day in this
project passed exactly what they were written to refuse. So each guard gets a pair:

  * a NEGATIVE case that must raise -- this is the mutation, and it is the real test;
  * a POSITIVE case that must NOT raise -- so a guard that always raises (which would
    also "pass" every negative case) is caught.

Neither half is sufficient alone. A guard that only has the negative test could be
`raise GuardViolation` unconditionally.
"""

from __future__ import annotations

import pytest

from ornith_repro.grpo import grpo_advantages
from ornith_repro.guards import (
    GuardViolation,
    assert_batch_not_all_degenerate,
    assert_group_nonempty,
    assert_model_served,
    assert_proportions_match,
    assert_task_not_in_buffer,
    assert_token_budget_fits,
)
from ornith_repro.types import Task


# ---------------------------------------------------------------- G2 token budget
def test_g2_fires_when_budget_exceeds_served_context():
    with pytest.raises(GuardViolation, match="exceeds the served context"):
        assert_token_budget_fits(
            prompt_tokens=4000, max_new_tokens=1000, served_context_len=4096
        )


def test_g2_passes_when_budget_fits():
    assert_token_budget_fits(
        prompt_tokens=1000, max_new_tokens=1000, served_context_len=4096
    )


def test_g2_fires_on_nonpositive_arguments():
    with pytest.raises(GuardViolation, match="must be positive"):
        assert_token_budget_fits(0, 100, 4096)


# ---------------------------------------------------------------- G3 empty batch
@pytest.mark.parametrize("rewards", [[], [1.0]])
def test_g3_fires_on_empty_or_singleton_group(rewards):
    with pytest.raises(GuardViolation, match="must be refused"):
        assert_group_nonempty(rewards)


def test_g3_passes_on_a_real_group():
    assert_group_nonempty([0.0, 1.0])


def test_g3_grpo_refuses_singleton_rather_than_returning_zero():
    """The silent-zero version of this bug returns [0.0] and looks like an update."""
    with pytest.raises(ValueError, match="at least 2 group members"):
        grpo_advantages([1.0])


# ---------------------------------------------------------------- G4 degenerate batch
def test_g4_fires_when_every_group_is_degenerate():
    groups = [grpo_advantages([1.0] * 8) for _ in range(16)]
    assert all(g.degenerate for g in groups)
    with pytest.raises(GuardViolation, match="zero reward-directed gradient"):
        assert_batch_not_all_degenerate(groups)


def test_g4_passes_when_one_group_is_informative():
    groups = [grpo_advantages([1.0] * 8) for _ in range(15)]
    groups.append(grpo_advantages([1.0] * 7 + [0.0]))
    assert_batch_not_all_degenerate(groups)


def test_g4_fires_on_empty_batch():
    with pytest.raises(GuardViolation, match="no groups"):
        assert_batch_not_all_degenerate([])


# ---------------------------------------------------------------- G5 buffer ordering
def test_g5_fires_when_task_already_in_buffer():
    t = Task(task_id="t1", text="prove that the sum is even")
    with pytest.raises(GuardViolation, match="already in the buffer"):
        assert_task_not_in_buffer(t, [t.text, "something else"])


def test_g5_passes_for_a_fresh_task():
    t = Task(task_id="t1", text="prove that the sum is even")
    assert_task_not_in_buffer(t, ["something else"])


# ---------------------------------------------------------------- served model
def test_model_guard_fires_on_unregistered_id():
    """An unregistered id is answered by the base model with a 200 and no warning."""
    with pytest.raises(GuardViolation, match="is not served"):
        assert_model_served("Qwen/Qwen3.8-27B", ["Qwen/Qwen3-32B"])


def test_model_guard_fires_when_backend_lists_nothing():
    with pytest.raises(GuardViolation, match="no served models"):
        assert_model_served("Qwen/Qwen3.8-27B", [])


def test_model_guard_passes_when_id_is_served():
    assert_model_served("Qwen/Qwen3.8-27B", ["Qwen/Qwen3.8-27B"])


# ---------------------------------------------------------------- proportion match
def test_proportion_guard_fires_on_unmatched_control():
    treatment = {"algebra": 0.8, "geometry": 0.2}
    control = {"algebra": 0.5, "geometry": 0.5}
    with pytest.raises(GuardViolation, match="deviates by"):
        assert_proportions_match(treatment, control, tol=0.02)


def test_proportion_guard_fires_when_control_drops_a_stratum():
    with pytest.raises(GuardViolation, match="supports differ"):
        assert_proportions_match({"a": 0.5, "b": 0.5}, {"a": 1.0}, tol=0.02)


def test_proportion_guard_passes_on_a_matched_control():
    assert_proportions_match({"a": 0.75, "b": 0.25}, {"a": 0.75, "b": 0.25}, tol=0.02)
