"""Nested sampling: two comparison levels, budget matching, and the scaffold-level curse."""

from __future__ import annotations

import pytest

from ornith_repro.buffer import TaskBuffer
from ornith_repro.judges import Judges
from ornith_repro.llm import StubClient
from ornith_repro.loop import OrnithConfig
from ornith_repro.nested import NestedConfig, run_iteration_nested
from ornith_repro.types import Scaffold, Task


def _task():
    return Task(task_id="q1", text="count the distinct primes below the given bound",
                family="number_theory")


def _scaffolds(m):
    return [Scaffold(scaffold_id=f"h{j}",
                     instructions=f"count the distinct primes below the given bound v{j}",
                     grader_kind="exact") for j in range(m)]


def _blocks(m, n, success_rate=0.5, seed=0, abort_rate=0.0):
    c = StubClient(success_rate=success_rate, abort_rate=abort_rate)
    return [[c.generate(f"s{j}", 128, seed + j * 100 + i) for i in range(n)]
            for j in range(m)]


def test_nested_forms_both_comparison_levels():
    cfg, ncfg = OrnithConfig(base_model="stub"), NestedConfig(n_scaffolds=3,
                                                              n_rollouts_per_scaffold=8)
    r = run_iteration_nested(cfg, ncfg, _task(), _scaffolds(3), _blocks(3, 8),
                             TaskBuffer(), Judges())
    # level 2: one rollout group per scaffold
    assert len(r.rollout_advantages) == 3
    assert all(len(g.advantages) == 8 for g in r.rollout_advantages)
    # level 1: one group across scaffolds
    assert r.scaffold_advantages is not None
    assert len(r.scaffold_advantages.advantages) == 3
    # difficulty uses every rollout of the task, not one scaffold's
    assert len(r.task_record.rollouts) == 24
    assert r.task_record.n_valid == 24
    assert len(r.harness_records) == 3


def test_flat_ablation_has_no_scaffold_level():
    cfg = OrnithConfig(base_model="stub")
    ncfg = NestedConfig(n_scaffolds=1, n_rollouts_per_scaffold=8, sampling="flat")
    r = run_iteration_nested(cfg, ncfg, _task(), _scaffolds(1), _blocks(1, 8),
                             TaskBuffer(), Judges())
    assert r.scaffold_advantages is None, "flat has exactly one comparison level"
    assert len(r.rollout_advantages) == 1
    assert len(r.task_record.rollouts) == 8


def test_nested_and_flat_differ_in_budget_so_arms_must_be_matched():
    """The two readings are not the same experiment at the same n_rollouts_per_scaffold."""
    nested = NestedConfig(n_scaffolds=3, n_rollouts_per_scaffold=8)
    flat = NestedConfig(n_scaffolds=1, n_rollouts_per_scaffold=8, sampling="flat")
    assert nested.total_rollouts == 24
    assert flat.total_rollouts == 8
    # budget-matched flat comparison spends the same rollouts on one scaffold
    matched = NestedConfig(n_scaffolds=1, n_rollouts_per_scaffold=24, sampling="flat")
    assert matched.total_rollouts == nested.total_rollouts


def test_nested_config_refuses_degenerate_group_structures():
    with pytest.raises(ValueError, match="needs >= 2 scaffolds"):
        NestedConfig(n_scaffolds=1, sampling="nested")
    with pytest.raises(ValueError, match=">= 2 rollouts per scaffold"):
        NestedConfig(n_scaffolds=3, n_rollouts_per_scaffold=1)


def test_malformed_nesting_is_refused_not_silently_reshaped():
    cfg, ncfg = OrnithConfig(base_model="stub"), NestedConfig(n_scaffolds=3,
                                                              n_rollouts_per_scaffold=8)
    with pytest.raises(ValueError, match="rollout blocks; the nesting is malformed"):
        run_iteration_nested(cfg, ncfg, _task(), _scaffolds(3), _blocks(2, 8),
                             TaskBuffer(), Judges())
    with pytest.raises(ValueError, match="expected 8"):
        run_iteration_nested(cfg, ncfg, _task(), _scaffolds(3), _blocks(3, 4),
                             TaskBuffer(), Judges())


def test_scaffold_reward_is_an_aggregate_of_its_own_rollouts_so_a_holdout_differs():
    """The winner's curse one level up: same scaffolds, independent rollouts, new scores.

    Asserts on observable state -- that discovery and holdout scaffold rewards are not
    identical -- which is the precondition for the curse existing at all. If they were
    identical the scaffold score would not depend on the drawn rollouts and there would be
    nothing to correct.
    """
    cfg, ncfg = OrnithConfig(base_model="stub"), NestedConfig(n_scaffolds=3,
                                                              n_rollouts_per_scaffold=8)
    r = run_iteration_nested(
        cfg, ncfg, _task(), _scaffolds(3), _blocks(3, 8, seed=0),
        TaskBuffer(), Judges(),
        holdout_texts_by_scaffold=_blocks(3, 8, seed=9999),
    )
    assert len(r.scaffold_rewards_discovery) == 3
    assert len(r.scaffold_rewards_holdout) == 3
    assert r.scaffold_rewards_discovery != r.scaffold_rewards_holdout, (
        "a scaffold's reward must depend on the rollouts it drew, or there is no curse"
    )


def test_aborted_rollouts_are_pooled_correctly_across_scaffolds():
    cfg = OrnithConfig(base_model="stub", min_valid_rollouts=4)
    ncfg = NestedConfig(n_scaffolds=3, n_rollouts_per_scaffold=8)
    r = run_iteration_nested(cfg, ncfg, _task(), _scaffolds(3),
                             _blocks(3, 8, abort_rate=0.5, seed=3),
                             TaskBuffer(), Judges())
    assert r.task_record.n_aborted > 0, "fixture must actually produce aborts"
    assert r.task_record.n_valid + r.task_record.n_aborted == 24
    assert 0.0 <= r.task_record.p_hat <= 1.0
