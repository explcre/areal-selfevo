"""The M9 rule evolve-policy: driven through the REGISTRY, because that is what training builds.

``actor._route_groups`` constructs a router with ``factory()`` and no arguments, so the
dataclass defaults ARE the arm. Two arms in this project have already been lost in exactly
that seam -- ``random`` shipped an unusable ``{rl: 1.0}`` default and ran bit-identical to the
off arm, ``contextual`` shipped ``cold_start_rounds=0`` and routed every group to RL for 149
steps -- so every behavioural test here goes through ``ROUTERS["rule"]()`` rather than
through a hand-constructed object.

The properties under test are the three the requirement names: it consumes the same
features as the learned controller, it is deterministic, and each branch is the one the
measurements support.
"""

from __future__ import annotations

import math

import pytest
import torch

from areal.api.cli_args import GroupRoutingConfig
from selfevo import compose
from selfevo.compose import EVOLVE_POLICY_FACTORIES, PipelineConfig, ROUTERS, validate
from selfevo.observability import FEATURE_NAMES, group_features
from selfevo.routing.base import Granularity, HarnessAction, RoutingContext, TrainingMode
from selfevo.routing.contextual import MissingFeatures
from selfevo.routing.rule_policy import (
    READ_FEATURES,
    InconsistentFeatures,
    RulePolicyRouter,
)

# Imported rather than re-derived: two definitions of "an actor configured like the live
# runs" drift, and the drift is silent. Same reasoning as test_actor_router_seam.py.
from selfevo.tests.test_group_routing import (  # noqa: E402
    G,
    MIXED,
    PROMPT,
    SOLVED_AND_UNSOLVED,
    advantages,
    make_actor,
)


# --------------------------------------------------------------------- fixtures ------


def _extra(**over: float) -> dict[str, float]:
    """All seven observability features, with overrides.

    Args:
        **over: Feature values to replace. Passing an unknown name raises, so a typo in a
            test cannot quietly assert nothing.

    Returns:
        A feature mapping suitable for ``RoutingContext.extra``.

    Raises:
        KeyError: If ``over`` names something that is not a feature.
    """
    base = {
        "solve_rate": 0.5,
        "reward_std": 0.5,
        "mean_response_len": 100.0,
        "len_dispersion": 0.2,
        "mean_logprob": -0.5,
        "logprob_dispersion": 0.1,
        "truncated_fraction": 0.0,
    }
    for k in over:
        if k not in base:
            raise KeyError(f"{k!r} is not an observability feature: {sorted(base)}")
    base.update(over)
    return base


def _ctx(
    *,
    solve_rate: float = 0.5,
    reward_std: float = 0.5,
    truncated_fraction: float = 0.0,
    group_size: int = 8,
    extra: dict[str, float] | None = None,
    **kw: object,
) -> RoutingContext:
    """A routable group whose context and features agree, as the real producer makes them.

    Args:
        solve_rate: Written into BOTH ``ctx.solve_rate`` and ``extra["solve_rate"]``, which
            is the invariant ``actor._route_groups`` maintains.
        reward_std: Group reward dispersion; zero means RL-silent.
        truncated_fraction: Fraction of samples that hit the token budget.
        group_size: Samples in the group.
        extra: Complete replacement for the feature mapping, for the malformed-input tests.
        **kw: Passed through to :class:`RoutingContext` (``has_teacher``,
            ``can_evolve_harness``, ``granularity``, ``unit_id``).

    Returns:
        The context.
    """
    feats = (
        _extra(
            solve_rate=solve_rate,
            reward_std=reward_std,
            truncated_fraction=truncated_fraction,
        )
        if extra is None
        else extra
    )
    return RoutingContext(solve_rate=solve_rate, group_size=group_size, extra=feats, **kw)


def _features_from_rewards(rewards: list[float], length: int = 6) -> dict[str, float]:
    """Run the REAL feature producer, so the contract is tested against it and not a mock.

    Args:
        rewards: One raw reward per sample in a single group.
        length: Response tokens per sample.

    Returns:
        ``GroupFeatures.as_extra()`` for the one group.
    """
    n = len(rewards)
    feats = group_features(
        torch.tensor(rewards, dtype=torch.float32),
        torch.ones(n, length),
        torch.full((n, length), -0.5),
        [n],
    )
    return feats[0].as_extra()


# ------------------------------------------------------------- registry / wiring ------


def test_the_registry_builds_the_rule_router_with_no_arguments():
    """Production calls ``factory()``; a factory needing arguments is an arm that never runs."""
    r = ROUTERS["rule"]()
    assert type(r).__name__ == "RulePolicyRouter"
    assert tuple(r.feature_names) == tuple(FEATURE_NAMES)


def test_the_shipped_defaults_are_the_measured_ones():
    """The defaults ARE the arm, so they are asserted rather than trusted.

    ``skip`` on the solved branch is the measured position (inert at 0.5, harmful at 2.0);
    ``1.0`` is the truncation value at which every sample in the group failed to terminate.
    """
    r = ROUTERS["rule"]()
    assert r.solved_mode == TrainingMode.SKIP
    assert r.truncated_threshold == 1.0


def test_the_evolve_policy_axis_selects_this_router_and_not_the_one_scalar_one():
    """``evolve_policy="rule"`` must build the seven-feature rule.

    Discriminating assertion, not an identity check: the previous occupant of this slot
    (``SolveRateRouter``) ignores ``ctx.extra`` entirely, so it routes a featureless context
    happily. Anything that still does is not a like-for-like baseline for
    ``learned_weights``, which refuses the same context.
    """
    r = EVOLVE_POLICY_FACTORIES["rule"]()
    with pytest.raises(MissingFeatures):
        r.route(RoutingContext(solve_rate=0.5, group_size=8, extra={}))
    assert not validate(PipelineConfig(evolve_policy="rule", router="rule"))


def test_the_rule_is_selectable_as_a_router_by_config():
    """The baseline has to be runnable as an arm, which means passing ``validate``."""
    assert not validate(PipelineConfig(router="rule"))


# --------------------------------------------------------------------- the rule ------


def test_an_informative_group_goes_to_rl():
    """``reward_std > 0`` means some advantage is non-zero, so RL carries signal."""
    d = ROUTERS["rule"]().route(_ctx(solve_rate=0.5, reward_std=0.5))
    assert d.argmax() == TrainingMode.RL
    assert d.harness is HarnessAction.NONE


def test_a_group_unanimous_in_outcome_but_not_in_reward_still_goes_to_rl():
    """The case that separates keying on ``reward_std`` from keying on ``solve_rate``.

    Rewards ``[1.0, 0.8, 1.0, 0.8]`` grade as all-correct (``solve_rate == 1.0``) while the
    advantages ``r_i - rbar`` are +-0.1 and the gradient is live -- the exact group
    ``test_reward_std_is_zero_exactly_when_the_group_is_unanimous`` pins in the observability
    suite. A rule that read ``solve_rate`` would delete that gradient, and one that compared
    ``reward_std`` against any threshold at or above 0.1 would too.
    """
    extra = _features_from_rewards([1.0, 0.8, 1.0, 0.8])
    assert extra["solve_rate"] == 1.0 and extra["reward_std"] > 0.0
    d = ROUTERS["rule"]().route(_ctx(solve_rate=1.0, extra=extra))
    assert d.argmax() == TrainingMode.RL, d.reason


def test_a_barely_informative_binary_group_goes_to_rl():
    """One solve out of eight is the smallest non-zero signal a binary group can carry."""
    extra = _features_from_rewards([1.0] + [0.0] * 7)
    d = ROUTERS["rule"]().route(_ctx(solve_rate=extra["solve_rate"], extra=extra))
    assert d.argmax() == TrainingMode.RL, extra


def test_a_solved_silent_group_is_skipped_rather_than_sharpened():
    """Measured: the free self-target is inert at 0.5 and harmful at 2.0.

    So the branch that HAS a target is still the branch that spends nothing, and the rule
    encodes the measurement rather than the intuition that a free target must be worth using.
    """
    extra = _features_from_rewards([1.0] * 8)
    assert extra["reward_std"] == 0.0 and extra["solve_rate"] == 1.0
    d = ROUTERS["rule"]().route(_ctx(solve_rate=1.0, extra=extra))
    assert d.argmax() == TrainingMode.SKIP, d.reason


def test_the_solved_branch_can_be_switched_to_sft_for_the_ab():
    """GOAL.md critical-path item 1 is SFT-on-own-sample versus SKIP, at matched compute.

    It has to be an argument to this router, not a different router, or the A/B changes two
    things at once.
    """
    r = ROUTERS["rule"](solved_mode=TrainingMode.SFT)
    d = r.route(_ctx(solve_rate=1.0, reward_std=0.0))
    assert d.argmax() == TrainingMode.SFT


def test_an_unsolved_silent_group_without_a_teacher_is_skipped():
    """No sample was correct, so no self-target exists and RL is identically zero."""
    extra = _features_from_rewards([0.0] * 8)
    d = ROUTERS["rule"]().route(_ctx(solve_rate=0.0, extra=extra))
    assert d.argmax() == TrainingMode.SKIP, d.reason


def test_an_unsolved_silent_group_with_an_external_teacher_goes_to_the_teacher():
    """The M7/M24 branch: the only thing that turns this unit from SKIP into signal."""
    d = ROUTERS["rule"]().route(_ctx(solve_rate=0.0, reward_std=0.0, has_teacher=True))
    assert d.argmax() == TrainingMode.SFT, d.reason


# ------------------------------------------------------------------ harness axis ------


def test_a_fully_truncated_unsolved_group_proposes_to_the_harness():
    """Truncation is non-termination, not a budget shortfall (79 -> 78 at twice the cap).

    A scaffold change is the consumer of that failure, which is what PROPOSE records.
    """
    d = ROUTERS["rule"]().route(
        _ctx(solve_rate=0.0, reward_std=0.0, truncated_fraction=1.0, can_evolve_harness=True)
    )
    assert d.harness is HarnessAction.PROPOSE
    assert d.argmax() == TrainingMode.SKIP


def test_a_partly_truncated_unsolved_group_does_not_propose_at_the_default():
    """1.0 is chosen so the group's failure mode is unambiguous; 7/8 truncated is not."""
    d = ROUTERS["rule"]().route(
        _ctx(solve_rate=0.0, reward_std=0.0, truncated_fraction=0.875, can_evolve_harness=True)
    )
    assert d.harness is HarnessAction.NONE, d.reason


def test_an_untruncated_unsolved_group_does_not_propose():
    """PROPOSE is gated on truncation, not merely on being unsolved -- an unsolved group
    that terminates is a knowledge failure, whose consumer is a teacher, not the harness."""
    d = ROUTERS["rule"]().route(
        _ctx(solve_rate=0.0, reward_std=0.0, truncated_fraction=0.0, can_evolve_harness=True)
    )
    assert d.harness is HarnessAction.NONE, d.reason


def test_no_harness_action_when_the_run_has_no_harness_arm():
    """``_route_groups`` never sets ``can_evolve_harness``, so this is the LIVE path.

    A run without a harness arm must behave exactly as it would without this axis, and the
    reason must say the action was dropped rather than silently omitting it.
    """
    d = ROUTERS["rule"]().route(
        _ctx(solve_rate=0.0, reward_std=0.0, truncated_fraction=1.0)
    )
    assert d.harness is HarnessAction.NONE
    assert "dropped" in d.reason


def test_a_truncated_group_that_still_solved_something_does_not_propose():
    """A group with a correct sample has a demonstrated path to success, so its truncated
    siblings are not evidence that the scaffold is at fault."""
    d = ROUTERS["rule"]().route(
        _ctx(solve_rate=0.5, reward_std=0.5, truncated_fraction=1.0, can_evolve_harness=True)
    )
    assert d.harness is HarnessAction.NONE


# ------------------------------------------------------------------ determinism ------


def test_the_same_context_always_gives_the_same_decision():
    """The property that makes this a baseline at all."""
    r = ROUTERS["rule"]()
    ctx = _ctx(solve_rate=0.25, reward_std=0.43, unit_id="u0")
    first = r.route(ctx)
    for _ in range(50):
        again = r.route(ctx)
        assert again.weights == first.weights and again.reason == first.reason


def test_decisions_do_not_depend_on_call_order_or_on_which_instance_decided():
    """No state and no RNG: a shuffled batch must produce the same per-unit decisions.

    ``RandomRouter`` fails this by construction, which is the difference between a control
    and a baseline.
    """
    units = [
        _ctx(solve_rate=s, reward_std=v, truncated_fraction=t, unit_id=f"u{i}")
        for i, (s, v, t) in enumerate(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.125, 0.33, 0.5)] * 8
        )
    ]
    forward = [ROUTERS["rule"]().route(c).argmax() for c in units]
    backward = list(reversed([ROUTERS["rule"]().route(c).argmax() for c in reversed(units)]))
    one_router = ROUTERS["rule"]()
    shared = [one_router.route(c).argmax() for c in units]
    assert forward == backward == shared


def test_a_batch_at_the_measured_silent_fraction_carries_more_than_one_mode():
    """The cold-start precondition, checked at the measured composition rather than assumed.

    ``batch_outcomes`` refuses a single-mode batch as confounded, so a cold-start policy that
    collapses to one mode starves the learner it is seeding. ``silent_group_fraction`` was
    measured at 0.5906 (G=8) and 0.4553 (G=16) on the matched GSM8K pair, so at 0.5906
    (``step0m-off``) a batch of 64 groups contains both branches. The wider 0.34-0.61 range
    quoted elsewhere came from the retroactive-sizing table, which the 2026-08-31 RETRACTION
    withdraws, so it is deliberately not cited here.
    """
    r = ROUTERS["rule"]()
    silent = 38  # 0.5906 of 64, the step0m-off measurement
    batch = [_ctx(solve_rate=1.0, reward_std=0.0) for _ in range(silent)]
    batch += [_ctx(solve_rate=0.5, reward_std=0.5) for _ in range(64 - silent)]
    modes = {r.route(c).argmax() for c in batch}
    assert modes == {TrainingMode.RL, TrainingMode.SKIP}, modes


# ------------------------------------------------------- validation, loudly ----------


@pytest.mark.parametrize("dropped", FEATURE_NAMES)
def test_every_feature_the_learned_router_requires_is_required_here_too(dropped):
    """Like-for-like means failing on exactly the contexts the learned controller fails on.

    Parametrised over all seven, including the four this rule does not read: a baseline that
    accepted a context the contextual arm rejects would run on batches the other arm could
    not, and the arms would no longer be comparable.
    """
    extra = _extra()
    extra.pop(dropped)
    with pytest.raises(MissingFeatures):
        ROUTERS["rule"]().route(_ctx(extra=extra))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_feature_is_refused_rather_than_defaulted(bad):
    """``group_features`` guarantees finite fields, so a NaN means another producer.

    It matters more here than in a learned router: ``nan >= threshold`` is False, so a NaN
    truncation fraction would silently read as 'this group terminated'.
    """
    with pytest.raises(InconsistentFeatures):
        ROUTERS["rule"]().route(_ctx(extra=_extra(truncated_fraction=bad)))


def test_features_describing_a_different_unit_are_refused():
    """One group's decision must not be taken on another group's evidence."""
    extra = _extra(solve_rate=1.0)
    with pytest.raises(InconsistentFeatures):
        ROUTERS["rule"]().route(
            RoutingContext(solve_rate=0.5, group_size=8, extra=extra)
        )


def test_an_arithmetically_impossible_group_is_refused():
    """Unanimous rewards grade all-correct or all-wrong, never half.

    Both silent branches assume every sample agreed, so a unit that claims otherwise belongs
    to neither and must not be assigned to one by whichever check ran first.
    """
    with pytest.raises(InconsistentFeatures):
        ROUTERS["rule"]().route(_ctx(solve_rate=0.5, reward_std=0.0))


def test_token_granularity_is_refused():
    """``has_self_target`` is forced False for a token, so a solved token would silently take
    the unsolved branch -- a misroute with nothing in the log to show it."""
    with pytest.raises(ValueError, match="TOKEN"):
        ROUTERS["rule"]().route(
            _ctx(solve_rate=1.0, reward_std=0.0, granularity=Granularity.TOKEN)
        )


def test_a_malformed_router_configuration_raises_at_construction():
    """Every rejected configuration is rejected when it is built, not at the first batch."""
    with pytest.raises(ValueError):
        ROUTERS["rule"](solved_mode="not_a_mode")
    with pytest.raises(ValueError, match="does not require a teacher"):
        ROUTERS["rule"](teacher_mode=TrainingMode.RL)
    with pytest.raises(ValueError, match="SKIP with extra compute"):
        ROUTERS["rule"](solved_mode=TrainingMode.RL)
    with pytest.raises(ValueError):
        ROUTERS["rule"](truncated_threshold=1.5)
    with pytest.raises(ValueError):
        ROUTERS["rule"](feature_names=())


def test_feature_names_must_cover_every_feature_the_rule_reads():
    """A feature the rule reads but never requires would be absent at decision time."""
    for missing in READ_FEATURES:
        names = tuple(f for f in FEATURE_NAMES if f != missing)
        with pytest.raises(ValueError, match="READS"):
            ROUTERS["rule"](feature_names=names)


def test_a_narrower_feature_set_is_allowed_when_it_covers_what_is_read():
    """The requirement is on what the rule reads, not on the full seven, so an ablation that
    deliberately narrows the input contract stays expressible."""
    r = ROUTERS["rule"](feature_names=READ_FEATURES)
    d = r.route(
        RoutingContext(
            solve_rate=1.0,
            group_size=8,
            extra={"solve_rate": 1.0, "reward_std": 0.0, "truncated_fraction": 0.0},
        )
    )
    assert d.argmax() == TrainingMode.SKIP


def test_the_reason_names_the_branch_that_fired():
    """A decision that cannot be audited after the fact is not evidence about an arm."""
    r = ROUTERS["rule"]()
    assert "informative" in r.route(_ctx(reward_std=0.5)).reason
    assert "solved" in r.route(_ctx(solve_rate=1.0, reward_std=0.0)).reason
    assert "unsolved" in r.route(_ctx(solve_rate=0.0, reward_std=0.0)).reason


def test_features_are_read_but_not_all_of_them_decide():
    """The claim READ_FEATURES makes, tested rather than asserted in prose.

    Varying any feature outside ``READ_FEATURES`` must not move a decision; varying one
    inside it must be able to. This is what lets the rule-vs-learned comparison attribute a
    difference to the features the learned arm uses and this one does not.
    """
    r = ROUTERS["rule"]()
    base = r.route(_ctx(solve_rate=0.0, reward_std=0.0)).argmax()
    for name in FEATURE_NAMES:
        if name in READ_FEATURES:
            continue
        moved = r.route(_ctx(solve_rate=0.0, reward_std=0.0, extra=_extra(
            solve_rate=0.0, reward_std=0.0, **{name: 42.0}
        ))).argmax()
        assert moved == base, name
    assert r.route(_ctx(solve_rate=0.0, reward_std=0.7)).argmax() != base


def test_every_decision_is_a_registered_mode_with_a_finite_weight():
    """A decision naming an unregistered mode would be refused downstream mid-batch."""
    r = ROUTERS["rule"]()
    for ctx in (
        _ctx(solve_rate=0.5, reward_std=0.5),
        _ctx(solve_rate=1.0, reward_std=0.0),
        _ctx(solve_rate=0.0, reward_std=0.0),
        _ctx(solve_rate=0.0, reward_std=0.0, has_teacher=True),
    ):
        d = r.route(ctx)
        assert sum(d.weights.values()) == 1.0
        assert all(math.isfinite(w) for w in d.weights.values())


# ------------------------------------------------------ the REAL actor path ----------


def test_the_rule_arm_is_bit_identical_to_the_off_arm_on_a_fully_silent_batch():
    """Driven through ``PPOActor._compute_advantages``, the entry point training calls.

    A test that calls ``_route_groups`` directly cannot catch ``_route_groups`` being
    unreachable, which is the failure this project has hit twice. And the assertion is
    stronger than "it ran": under the shipped defaults every silent group is SKIPped, SKIP
    on an already-silent group writes nothing, so a rule arm must reproduce the off arm
    EXACTLY on a batch of one solved and one unsolved group. If it does not, the baseline is
    spending gradient somewhere it claims not to.
    """
    off = advantages(make_actor(), SOLVED_AND_UNSOLVED)
    on = advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5, router="rule")),
        SOLVED_AND_UNSOLVED,
    )
    assert torch.equal(off, on), (off, on)


def test_the_rule_leaves_an_informative_group_to_rl_on_the_real_path():
    """The RL branch must be a genuine no-op on the tensor, not merely a label in the log."""
    off = advantages(make_actor(), MIXED)
    on = advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5, router="rule")),
        MIXED,
    )
    assert off[G:].abs().max() > 1e-6          # the group really is informative
    assert torch.equal(off, on), (off, on)


def test_the_solved_branch_ab_actually_reaches_the_loss_when_switched_on():
    """``solved_mode="sft"`` is GOAL.md critical-path item 1, so it has to move the tensor.

    Registered through ``compose.ROUTERS`` rather than constructed, because
    ``_route_groups`` builds routers with no arguments -- a variant that cannot be reached
    that way is not a runnable arm. The registry is process-wide state, so it is restored.
    """
    name = "_rule_sft_ab"
    previous = compose.ROUTERS.get(name, KeyError)
    compose.ROUTERS[name] = lambda **kw: RulePolicyRouter(solved_mode=TrainingMode.SFT, **kw)
    try:
        adv = advantages(
            make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5, router=name)),
            SOLVED_AND_UNSOLVED,
        )
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(name, None)
        else:
            compose.ROUTERS[name] = previous
    solved, unsolved = adv[:G], adv[G:]
    assert torch.allclose(solved[:, PROMPT:], torch.full_like(solved[:, PROMPT:], 0.5))
    assert solved[:, :PROMPT].abs().max() == 0.0   # never write into the prompt
    assert unsolved.abs().max() == 0.0             # no target of any kind -> still SKIP
