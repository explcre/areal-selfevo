"""Tests for the specific silent-zero paths named in the brief.

Each test constructs the exact situation in which a number would be produced that looks
like a measurement but is not, and asserts the code refuses it. These are the paths that
have actually happened in this project.
"""

from __future__ import annotations

import pytest

from ornith_repro.buffer import TaskBuffer
from ornith_repro.judges import Judges
from ornith_repro.llm import StubClient
from ornith_repro.loop import OrnithConfig, run_iteration
from ornith_repro.rewards import difficulty_reward, novelty_reward, success_rate
from ornith_repro.types import Rollout, RolloutOutcome, Scaffold, Task


def _r(outcome: RolloutOutcome, i: int = 0) -> Rollout:
    return Rollout(rollout_id=f"r{i}", text="x", outcome=outcome,
                   reward=1.0 if outcome is RolloutOutcome.SUCCESS else 0.0)


# -------------------------------------------------- aborted graded as wrong (G1)
def test_aborted_rollouts_are_not_counted_as_failures_by_default():
    """The bug: 6 aborts among 8 rollouts silently yield a plausible-looking p_hat."""
    rollouts = [_r(RolloutOutcome.SUCCESS, 0), _r(RolloutOutcome.SUCCESS, 1)] + [
        _r(RolloutOutcome.ABORTED, i) for i in range(2, 8)
    ]
    p_excl, n_valid, n_abort = success_rate(rollouts, abort_policy="exclude")
    assert (n_valid, n_abort) == (2, 6)
    assert p_excl == pytest.approx(1.0), "both gradeable rollouts succeeded"

    p_fail, n_valid_f, _ = success_rate(rollouts, abort_policy="failure")
    assert n_valid_f == 8
    assert p_fail == pytest.approx(0.25), "the known-bad policy reports 0.25 instead"

    # The two differ by 0.75 in p_hat, which at p*=0.2, sigma=0.15 is the difference
    # between a near-worthless task and a near-perfect one.
    assert difficulty_reward(p_fail) > difficulty_reward(p_excl)


def test_all_aborted_is_refused_not_scored_zero():
    rollouts = [_r(RolloutOutcome.ABORTED, i) for i in range(8)]
    with pytest.raises(ValueError, match="every rollout aborted"):
        success_rate(rollouts, abort_policy="exclude")


def test_too_few_gradeable_rollouts_refuses_the_task():
    """A task graded on 2 surviving rollouts is a task graded on noise."""
    cfg = OrnithConfig(base_model="stub", k_rollouts=8, min_valid_rollouts=4)
    task = Task(task_id="q", text="a task with enough words to be valid here")
    scaffold = Scaffold(scaffold_id="h", instructions="a task with enough words")
    # 6 of 8 truncated.
    rollout_texts = [("SOLVED a", False), ("SOLVED b", False)] + [("", True)] * 6
    with pytest.raises(ValueError, match="only 2 gradeable rollouts"):
        run_iteration(cfg, task, scaffold, rollout_texts, TaskBuffer(), Judges())


def test_strict_abort_policy_raises_on_any_abort():
    rollouts = [_r(RolloutOutcome.SUCCESS, 0), _r(RolloutOutcome.ABORTED, 1)]
    with pytest.raises(ValueError, match="aborted and abort_policy"):
        success_rate(rollouts, abort_policy="strict")


# -------------------------------------------------- empty reward batch (G3)
def test_empty_rollout_list_is_refused_not_scored_zero():
    with pytest.raises(ValueError, match="empty rollout list"):
        success_rate([])


# -------------------------------------------------- empty-buffer novelty (G6)
def test_empty_buffer_novelty_is_one_not_zero():
    """Returning 0.0 here would annihilate R_task for the first task of every run."""
    n, empty = novelty_reward("some brand new task", [])
    assert (n, empty) == (1.0, True)


def test_nonempty_buffer_novelty_is_measured():
    n, empty = novelty_reward("alpha beta gamma", ["alpha beta gamma"])
    assert empty is False
    assert n == pytest.approx(0.0), "an exact duplicate has zero novelty"


# -------------------------------------------------- reward-range validation
@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_out_of_range_factors_are_refused(bad):
    from ornith_repro.rewards import harness_reward, task_reward

    with pytest.raises(ValueError):
        task_reward(bad, 0.5, 0.5)
    with pytest.raises(ValueError):
        harness_reward(0.5, bad, 0.5)


def test_difficulty_reward_refuses_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        difficulty_reward(0.2, sigma=0.0)


def test_stub_client_abort_rate_actually_produces_aborts():
    """Guard the guard: if the stub never aborted, the abort tests would be vacuous."""
    client = StubClient(abort_rate=1.0)
    _, truncated = client.generate("p", 128, 0)
    assert truncated is True

    client_never = StubClient(abort_rate=0.0)
    _, truncated2 = client_never.generate("p", 128, 0)
    assert truncated2 is False


# ------------------------------------------------- the DEFAULT policy, not just explicit
def test_default_abort_policy_excludes_aborts_rather_than_failing_them():
    """Regression for a hole the mutation harness found (M2).

    Every other test in this file passes `abort_policy=` explicitly, so flipping the
    DEFAULT from "exclude" to "failure" changed nothing observable and survived mutation.
    The default is the value that ships, so it is asserted here directly, with no keyword.
    """
    rollouts = [_r(RolloutOutcome.SUCCESS, 0), _r(RolloutOutcome.SUCCESS, 1)] + [
        _r(RolloutOutcome.ABORTED, i) for i in range(2, 8)
    ]
    p_default, n_valid, n_abort = success_rate(rollouts)  # NO abort_policy argument
    assert (n_valid, n_abort) == (2, 6), "the default must exclude aborts from the denominator"
    assert p_default == pytest.approx(1.0)


def test_config_default_abort_policy_is_exclude():
    """The loop's own default is asserted too, since it overrides the function default."""
    assert OrnithConfig().abort_policy == "exclude"
