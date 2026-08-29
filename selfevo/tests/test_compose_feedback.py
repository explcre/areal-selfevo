"""Tests for the composition axes and the routing feedback channel."""

from __future__ import annotations

import pytest

from selfevo.compose import (
    GATES,
    PipelineConfig,
    SHAPERS,
    is_valid,
    register_shaper,
    validate,
)
from selfevo.routing.base import RoutingContext, TrainingMode
from selfevo.routing.feedback import (
    BanditRouter,
    ConfoundedUpdate,
    DecisionOutcome,
    LearningRouter,
)

# --------------------------------------------------------------------------- compose


def test_plain_config_is_valid():
    assert is_valid(PipelineConfig())


def test_meds_style_shaper_with_prefix_gate_is_rejected():
    """The rule this module exists for: entropy bonus voids sum_i A_i = 0."""
    register_shaper("entropy_bonus", None, breaks_centring=True)
    cfg = PipelineConfig(shaper="entropy_bonus", gate="prefix_dead")
    problems = validate(cfg)
    assert problems, "MEDS-style shaping + prefix gating must be rejected"
    assert any(p.axes == ("shaper", "gate") for p in problems)
    assert "sum_i A_i" in str(problems[0])
    assert "dp_actor.py:560" in str(problems[0]), "must cite where the claim is grounded"


def test_the_same_shaper_is_fine_without_the_gate():
    register_shaper("entropy_bonus", None, breaks_centring=True)
    assert is_valid(PipelineConfig(shaper="entropy_bonus", gate="none"))


def test_the_same_gate_is_fine_without_the_shaper():
    assert is_valid(PipelineConfig(shaper="none", gate="prefix_dead"))


def test_learned_policy_without_feedback_is_rejected():
    problems = validate(PipelineConfig(evolve_policy="learned_weights"))
    assert any("require_feedback" in p.reason for p in problems)
    assert is_valid(
        PipelineConfig(evolve_policy="learned_weights", require_feedback=True)
    )


def test_learned_code_policy_is_held_to_the_same_rule():
    assert validate(PipelineConfig(evolve_policy="learned_code"))
    assert is_valid(PipelineConfig(evolve_policy="learned_code", require_feedback=True))


def test_evolving_both_under_a_fixed_rule_is_unattributable():
    problems = validate(PipelineConfig(evolve_target="both", evolve_policy="rule"))
    assert any(p.axes == ("evolve_target", "evolve_policy") for p in problems)
    # A policy that records its choice fixes attribution -- but "both" now includes the
    # reward, so it must also carry the frozen-reward guards. Tightened deliberately:
    # the old config is genuinely unsafe, not merely newly-rejected.
    assert is_valid(
        PipelineConfig(
            evolve_target="both",
            evolve_policy="learned_weights",
            require_feedback=True,
            frozen_eval_reward=True,
            policy_scored_by_frozen_reward=True,
        )
    )


def test_unknown_names_are_reported_not_ignored():
    problems = validate(
        PipelineConfig(shaper="nope", gate="nope", evolve_target="nope", evolve_policy="nope")
    )
    assert len(problems) >= 4


def test_validate_reports_every_problem_not_just_the_first():
    register_shaper("entropy_bonus", None, breaks_centring=True)
    problems = validate(
        PipelineConfig(
            shaper="entropy_bonus",
            gate="prefix_dead",
            evolve_policy="learned_weights",
        )
    )
    assert len(problems) >= 2, "a sweep needs all rejected reasons at once"


def test_registering_a_shaper_inconsistently_raises():
    register_shaper("probe_shaper", None, breaks_centring=True)
    register_shaper("probe_shaper", None, breaks_centring=True)  # idempotent
    with pytest.raises(ValueError):
        register_shaper("probe_shaper", None, breaks_centring=False)
    with pytest.raises(ValueError):
        register_shaper("", None, breaks_centring=False)


def test_a_new_safe_shaper_composes_with_the_gate():
    register_shaper("rescale_only", None, breaks_centring=False)
    assert is_valid(PipelineConfig(shaper="rescale_only", gate="prefix_dead"))
    assert "rescale_only" in SHAPERS and "prefix_dead" in GATES


# -------------------------------------------------------------------------- feedback


def _out(mode: str, value: float, batch: str = "b0", cost: float = 1.0) -> DecisionOutcome:
    return DecisionOutcome(mode=mode, value=value, batch_id=batch, cost=cost)


def test_outcome_validation():
    with pytest.raises(ValueError):
        DecisionOutcome(mode="nope", value=1.0, batch_id="b")
    with pytest.raises(ValueError):
        DecisionOutcome(mode=TrainingMode.RL, value=1.0, batch_id="")
    with pytest.raises(ValueError):
        DecisionOutcome(mode=TrainingMode.RL, value=1.0, batch_id="b", cost=0.0)
    with pytest.raises(ValueError):
        DecisionOutcome(mode=TrainingMode.RL, value=float("nan"), batch_id="b")


def test_bandit_satisfies_the_learning_protocol():
    assert isinstance(BanditRouter(), LearningRouter)


def test_zero_exploration_is_rejected():
    """Without exploration the router only ever observes the mode it already prefers."""
    with pytest.raises(ValueError, match="explore"):
        BanditRouter(explore_prob=0.0)


def test_single_mode_batch_is_refused_as_confounded():
    b = BanditRouter()
    with pytest.raises(ConfoundedUpdate, match="confounded"):
        b.observe({"u1": _out(TrainingMode.RL, 1.0), "u2": _out(TrainingMode.RL, 2.0)})


def test_outcomes_spanning_batches_are_refused():
    b = BanditRouter()
    with pytest.raises(ConfoundedUpdate, match="span"):
        b.observe(
            {
                "u1": _out(TrainingMode.RL, 1.0, batch="b0"),
                "u2": _out(TrainingMode.SFT, 2.0, batch="b1"),
            }
        )


def test_empty_observation_is_a_noop():
    b = BanditRouter()
    b.observe({})
    assert b.value_estimates() == {}


def test_estimates_are_value_per_cost_so_cheap_modes_compete():
    b = BanditRouter()
    b.observe({"u1": _out(TrainingMode.RL, 2.0, cost=4.0), "u2": _out(TrainingMode.SKIP, 1.0, cost=1.0)})
    est = b.value_estimates()
    assert est[TrainingMode.RL] == pytest.approx(0.5)
    assert est[TrainingMode.SKIP] == pytest.approx(1.0)


def test_unobserved_modes_are_absent_not_zero():
    b = BanditRouter()
    b.observe({"u1": _out(TrainingMode.RL, 1.0), "u2": _out(TrainingMode.SFT, 1.0)})
    assert TrainingMode.SKIP not in b.value_estimates()


def test_bandit_eventually_prefers_the_better_mode():
    b = BanditRouter(explore_prob=0.05, min_observations=2, seed=3)
    for i in range(60):
        b.observe(
            {
                "a": _out(TrainingMode.RL, 1.0, batch=f"b{i}"),
                "c": _out(TrainingMode.SKIP, 0.0, batch=f"b{i}"),
                "b": _out(TrainingMode.SFT, 5.0, batch=f"b{i}"),
            }
        )
    ctx = RoutingContext(solve_rate=0.5, group_size=4, has_teacher=True)
    picks = [b.route(ctx).argmax() for _ in range(200)]
    assert picks.count(TrainingMode.SFT) > 150, f"did not converge: {set(picks)}"


def test_bandit_honours_the_teacher_invariant():
    b = BanditRouter(explore_prob=1.0, seed=0)
    ctx = RoutingContext(solve_rate=0.5, group_size=4, has_teacher=False)
    for _ in range(100):
        assert b.route(ctx).argmax() != TrainingMode.SFT


def test_bandit_is_reproducible_given_a_seed():
    ctx = RoutingContext(solve_rate=0.5, group_size=4, has_teacher=True)
    a = [BanditRouter(seed=11).route(ctx).argmax() for _ in range(1)]
    c = [BanditRouter(seed=11).route(ctx).argmax() for _ in range(1)]
    assert a == c


# --------------------------------------------- the rule, against the SHIPPED registry


def test_entropy_bonus_is_marked_unsafe_by_the_shipped_registry():
    """Checked in a FRESH interpreter, because register_shaper mutates a module global.

    An in-session assert is worthless here: other tests call register_shaper() and re-add
    the very entry under test, so emptying the module-level seed left the whole suite
    green. Only a subprocess sees the shipped registry.
    """
    import subprocess
    import sys as _sys

    prog = (
        "from selfevo.compose import _BREAKS_CENTRING, validate, PipelineConfig;"
        "assert 'entropy_bonus' in _BREAKS_CENTRING, 'not in shipped registry';"
        "p = validate(PipelineConfig(shaper='entropy_bonus', gate='prefix_dead'));"
        "assert p, 'MEDS shaper + prefix gate not rejected out of the box';"
        "assert any(x.axes == ('shaper', 'gate') for x in p), 'wrong axes';"
        "print('OK')"
    )
    r = subprocess.run([_sys.executable, "-c", prog], capture_output=True, text=True)
    assert r.returncode == 0, (
        "shipped registry does not reject the MEDS combination.\n"
        f"stdout={r.stdout!r} stderr={r.stderr[-400:]!r}"
    )


def test_meds_combination_is_rejected_without_any_test_setup():
    """The exact MEDS cell, validated with nothing registered by the test."""
    problems = validate(PipelineConfig(shaper="entropy_bonus", gate="prefix_dead"))
    assert problems, "the MEDS shaper + prefix gate must be rejected out of the box"
    assert any(p.axes == ("shaper", "gate") for p in problems)
    assert "dp_actor.py:560" in str(problems[0])


# ------------------------------------------------ reward evolution and critic axes


def test_evolving_the_reward_without_a_frozen_one_is_rejected():
    """A rising curve cannot be told from a reward that got easier."""
    problems = validate(PipelineConfig(evolve_target="reward"))
    assert any("unmeasurable" in p.reason for p in problems)
    assert is_valid(PipelineConfig(evolve_target="reward", frozen_eval_reward=True))


def test_learned_policy_evolving_its_own_reward_is_rejected():
    """The degenerate fixed point: lowering the bar is the optimum, not a risk."""
    cfg = PipelineConfig(
        evolve_target="reward",
        evolve_policy="learned_weights",
        require_feedback=True,
        frozen_eval_reward=True,
    )
    problems = validate(cfg)
    assert any(p.axes == ("evolve_target", "evolve_policy") for p in problems), problems
    assert any("easier" in p.reason for p in problems)
    # scoring the policy by the frozen reward removes the incentive
    assert is_valid(
        PipelineConfig(
            evolve_target="reward",
            evolve_policy="learned_weights",
            require_feedback=True,
            frozen_eval_reward=True,
            policy_scored_by_frozen_reward=True,
        )
    )


def test_both_target_inherits_the_reward_guards():
    """'both' includes the reward, so it must carry the same protections."""
    problems = validate(
        PipelineConfig(evolve_target="both", evolve_policy="learned_weights",
                       require_feedback=True)
    )
    assert any("unmeasurable" in p.reason for p in problems)


def test_a_rule_based_policy_may_evolve_the_reward_with_a_frozen_anchor():
    """No learned objective to game, so only the measurement guard applies."""
    assert is_valid(
        PipelineConfig(evolve_target="reward", evolve_policy="rule",
                       frozen_eval_reward=True)
    )


def test_unknown_critic_is_rejected():
    problems = validate(PipelineConfig(critic="nope"))
    assert any(p.axes == ("critic",) for p in problems)


def test_critics_compose_freely_with_the_other_axes():
    from selfevo.compose import CRITICS

    assert {"none", "scalar", "two_level"} <= CRITICS
    for c in CRITICS:
        assert is_valid(PipelineConfig(critic=c))


def test_frozen_reward_flag_alone_is_harmless():
    """Setting the guard when not evolving the reward must not reject anything."""
    assert is_valid(PipelineConfig(frozen_eval_reward=True))
    assert is_valid(PipelineConfig(policy_scored_by_frozen_reward=True))
