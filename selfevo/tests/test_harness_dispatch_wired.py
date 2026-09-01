"""The harness axis through the REAL actor, which is the only place the claim is testable.

``selfevo/tests/test_harness_dispatch.py`` proves the dispatcher works. It cannot prove the
thing that matters, because the failure this axis has actually suffered was never in a
component: ``RoutingContext.can_evolve_harness`` existed, ``RoutingDecision.harness`` existed,
``HarnessVariant`` existed, every one of them was tested, and the axis was still inert end to
end because no production code connected them. ``_refuse_dropped_harness`` was added for
exactly that reason -- to make the disconnection loud -- and it is still the thing that has to
keep working.

So everything here goes through ``PPOActor._compute_advantages``, the entry point training
calls, with a real ``GroupRoutingConfig``. A test that called ``_route_groups`` directly, or
that built its own ``RoutingContext``, could not tell a wired seam from an unwired one, and
that distinction is the entire subject of this file.

Four things are pinned:

* ``can_evolve_harness`` is True in production exactly when a 2+ variant set is configured.
* The refusal guard still fires when no dispatcher exists, so the axis cannot go back to
  being silently dropped.
* A router's PROPOSE reaches the harness and moves it.
* Two arms differing ONLY in ``harness_variants`` differ in something observable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import areal.trainer.ppo.actor as actor_mod

from areal.api.cli_args import GroupRoutingConfig
from selfevo import compose
from selfevo.routing.base import HarnessAction, RoutingDecision, TrainingMode

from selfevo.tests.test_group_routing import (  # noqa: E402
    MIXED,
    advantages,
    make_actor,
)

STUB = "_harness_seam_stub"


class HarnessRouter:
    """A Router that records what it was TOLD and emits what it was CONFIGURED to emit.

    Deliberately not a real router. ``CoHarnessRouter`` and ``RulePolicyRouter`` both decide
    for themselves whether to propose, so a failure under one of them would be ambiguous
    between "the actor did not set can_evolve_harness" and "the router chose not to act". This
    one separates the two: :attr:`told` is what the actor put in the context, and ``action`` is
    fixed by the test.

    Args:
        action: The harness action every decision carries.
        respect_flag: If True, downgrade to NONE when the context says the harness cannot
            evolve -- the shape every shipped router uses. If False, emit regardless, which
            is what a misbehaving or newly written router looks like and is the case
            ``_refuse_dropped_harness`` exists to catch.
    """

    def __init__(
        self, action: HarnessAction = HarnessAction.NONE, *, respect_flag: bool = True
    ) -> None:
        self.action = action
        self.respect_flag = respect_flag
        self.told: list[bool] = []

    def route(self, ctx) -> RoutingDecision:
        """Record ``ctx.can_evolve_harness`` and return the configured action."""
        self.told.append(ctx.can_evolve_harness)
        action = self.action
        if self.respect_flag and not ctx.can_evolve_harness:
            action = HarnessAction.NONE
        return RoutingDecision({TrainingMode.RL: 1.0}, reason="stub", harness=action)


@pytest.fixture
def router_factory():
    """Register a configurable router under ``STUB`` and restore the registry afterwards.

    ``compose.ROUTERS`` is module-level state shared with every other test in the process, so
    a test that mutated it without restoring would make an unrelated test fail later, in a
    different file, with no visible connection to this one.

    Yields:
        A callable taking the router's constructor kwargs; the router it produces is what the
        actor will build, and the same object is returned for inspection.
    """
    made: list[HarnessRouter] = []
    kwargs: dict = {}

    def factory(*_a, **_kw):
        r = HarnessRouter(**kwargs)
        made.append(r)
        return r

    def configure(**kw):
        kwargs.clear()
        kwargs.update(kw)
        return made

    previous = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = factory
    try:
        yield configure
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = previous


class Recorder:
    """Captures ``stats_tracker.scalar`` kwargs so the LOGGED values can be asserted."""

    def __init__(self) -> None:
        self.seen: dict = {}

    def scalar(self, **kw) -> None:
        """Accumulate every logged key, last write winning."""
        self.seen.update(kw)


@pytest.fixture
def recorder(monkeypatch):
    """Swap the actor's module-level stats tracker for one that records."""
    r = Recorder()
    monkeypatch.setattr(actor_mod, "stats_tracker", r)
    return r


def actor_with(variants, **kw):
    """An actor whose group routing uses the stub router and the given variant set.

    Args:
        variants: Value for ``GroupRoutingConfig.harness_variants``.
        **kw: Further ``GroupRoutingConfig`` overrides.

    Returns:
        A CPU ``PPOActor`` whose ``_compute_advantages`` can be called directly.
    """
    return make_actor(
        GroupRoutingConfig(
            enabled=True,
            solved_advantage=0.5,
            router=STUB,
            harness_variants=variants,
            **kw,
        )
    )


# ------------------------------------------------------------- the config itself ---------


def test_the_shipped_default_has_no_harness_arm():
    """Every run written before this field existed must be untouched by it."""
    assert GroupRoutingConfig().harness_variants is None


def test_a_repeated_variant_name_is_refused_by_the_config():
    """``["plain", "plain"]`` would pass a length check and be a one-scaffold arm.

    Caught at config time rather than at the first batch, because a run whose harness arm is
    secretly its own control is worth failing before the GPUs are allocated.
    """
    with pytest.raises(ValueError, match="repeats"):
        GroupRoutingConfig(harness_variants=["plain", "plain"])


def test_an_unregistered_variant_name_is_refused_by_the_config():
    """A dropped unknown name would shrink a two-variant arm into a one-variant control."""
    with pytest.raises(ValueError, match="unregistered variant"):
        GroupRoutingConfig(harness_variants=["plain", "no-such-variant"])


def test_a_one_name_variant_set_is_still_a_legal_config():
    """It is the matched control for a dispatching arm and has to be runnable."""
    cfg = GroupRoutingConfig(router="solve_rate", harness_variants=["plain"])
    assert cfg.harness_variants == ["plain"]


def test_a_harness_arm_without_a_router_is_refused():
    """The fixed solved/unsolved rule emits no decision, so it can emit no harness action.

    Without a router ``_route_groups`` never runs, no dispatcher is ever built, and the run
    would carry a ``harness_variants`` line in its config while dispatching nothing. That is
    an arm reporting itself as something it is not, which is the exact failure the refusal
    guard exists to make loud -- and the guard cannot catch it, because a router that never
    runs emits no action to refuse.
    """
    with pytest.raises(ValueError, match="requires a router"):
        GroupRoutingConfig(enabled=True, harness_variants=["plain", "long"])


# ------------------------------------------------ can_evolve_harness, in production ------


def test_two_configured_variants_make_can_evolve_harness_true(router_factory):
    """The line that was missing. Nothing in production had ever set this True.

    Asserted on EVERY group, not just the first: the flag is built into each context
    separately, so a per-unit condition that happened to be true once would pass a weaker
    check while leaving most of the batch unable to propose.
    """
    made = router_factory()
    advantages(actor_with(["plain", "long"]), MIXED)
    assert made[0].told == [True, True]


def test_a_single_configured_variant_leaves_can_evolve_harness_false(router_factory):
    """A dispatcher exists, but it has one possible answer, so it is not an evolvable harness.

    This is the case a length check on the config alone would get wrong, and it is the
    matched control for the arm above: same code path, same metrics, one scaffold.
    """
    made = router_factory()
    advantages(actor_with(["plain"]), MIXED)
    assert made[0].told == [False, False]


def test_no_variant_set_leaves_can_evolve_harness_false(router_factory):
    """The shipped default. Every run before this feature existed must be unchanged."""
    made = router_factory()
    advantages(actor_with(None), MIXED)
    assert made[0].told == [False, False]


def test_three_variants_also_make_it_true(router_factory):
    """Two is a floor, not a magic number, and the seam must not hardcode it as equality."""
    made = router_factory()
    advantages(actor_with(["plain", "long", "short"]), MIXED)
    assert all(made[0].told)


# ------------------------------------------------------------- the refusal guard ---------


def test_the_guard_still_fires_when_no_dispatcher_exists(router_factory):
    """Without a consumer, an emitted harness action must still be refused, loudly.

    The whole point of building the dispatcher was to satisfy this guard rather than remove
    it. If wiring the consumer had also disarmed the unwired case, the next inert axis would
    go unnoticed for exactly as long as this one did.
    """
    router_factory(action=HarnessAction.PROPOSE, respect_flag=False)
    with pytest.raises(ValueError, match="nothing consumes"):
        advantages(actor_with(None), MIXED)


def test_the_guard_is_satisfied_once_a_dispatcher_is_configured(router_factory):
    """And it must step aside when a consumer exists, or it blocks the feature it guards."""
    router_factory(action=HarnessAction.PROPOSE, respect_flag=False)
    actor = actor_with(["plain", "long"])
    advantages(actor, MIXED)  # must not raise
    assert actor._selfevo_harness.active.name == "long"


def test_a_one_variant_dispatcher_is_a_consumer_too(router_factory):
    """It consumes the action and REFUSES it, which is a different run from dropping it.

    The alternative -- declaring the consumer on ``can_evolve`` rather than on the
    dispatcher's presence -- would raise here, and the control arm could not be run.
    """
    router_factory(action=HarnessAction.PROPOSE, respect_flag=False)
    actor = actor_with(["plain"])
    advantages(actor, MIXED)  # must not raise
    assert actor._selfevo_harness.active.name == "plain"


def test_a_validate_action_is_consumed_too(router_factory):
    """PROPOSE is not the only non-NONE action; the guard covers VALIDATE identically."""
    router_factory(action=HarnessAction.VALIDATE, respect_flag=False)
    with pytest.raises(ValueError, match="nothing consumes"):
        advantages(actor_with(None), MIXED)


# ------------------------------------------------------- the action reaches the harness ---


def test_a_proposal_moves_the_live_harness(router_factory):
    """End to end: a router's PROPOSE changes which scaffold the actor holds.

    The dispatcher is built lazily on the first batch, so its absence beforehand is asserted
    too: an actor that constructed one at ``__init__`` would hide a config read that never
    happened.
    """
    router_factory(action=HarnessAction.PROPOSE)
    actor = actor_with(["plain", "long"])
    assert not hasattr(actor, "_selfevo_harness")
    advantages(actor, MIXED)
    assert actor._selfevo_harness.active.name == "long"


def test_a_batch_without_proposals_leaves_the_harness_alone(router_factory):
    """The overwhelmingly common case must be exactly inert."""
    router_factory(action=HarnessAction.NONE)
    actor = actor_with(["plain", "long"])
    advantages(actor, MIXED)
    assert actor._selfevo_harness.active.name == "plain"


def test_the_dispatcher_is_built_once_and_keeps_its_selection(router_factory):
    """The active variant is persistent state; rebuilding per batch would erase the axis.

    Three variants and three batches, so a dispatcher reset each step would sit on ``plain``
    forever while still logging a switch every batch -- passing any test that only counted
    switches.
    """
    router_factory(action=HarnessAction.PROPOSE)
    actor = actor_with(["plain", "long", "short"])
    seen = []
    for _ in range(3):
        advantages(actor, MIXED)
        seen.append(actor._selfevo_harness.active.name)
    assert seen == ["long", "short", "plain"]


def test_the_harness_axis_does_not_disturb_the_advantage_tensor(router_factory):
    """The two coordinates are orthogonal, and this is what orthogonal has to mean.

    Same router, same mode, same everything except that one arm has a harness to move. If the
    tensors differed, the harness axis would be confounded with the model axis and no gap
    between two harness arms could be attributed.
    """
    router_factory(action=HarnessAction.PROPOSE)
    torch.manual_seed(0)
    without = advantages(actor_with(None), MIXED)
    torch.manual_seed(0)
    with_harness = advantages(actor_with(["plain", "long"]), MIXED)
    assert torch.equal(without, with_harness)


# ----------------------------------------------------------------------- metrics ---------


def test_the_harness_metrics_reach_the_logs(router_factory, recorder):
    """An axis whose effect cannot be seen in the logs cannot be verified after the run."""
    router_factory(action=HarnessAction.PROPOSE)
    advantages(actor_with(["plain", "long"]), MIXED)
    assert recorder.seen["route/harness_propose"] == 2.0
    assert recorder.seen["route/harness_switches"] == 1.0
    assert recorder.seen["route/harness_can_evolve"] == 1.0
    assert recorder.seen["route/harness_active_long"] == 1.0
    assert recorder.seen["route/harness_active_plain"] == 0.0


def test_a_quiet_harness_arm_still_logs_its_keys(router_factory, recorder):
    """Otherwise "the axis did nothing" and "the axis was never enabled" look identical."""
    router_factory(action=HarnessAction.NONE)
    advantages(actor_with(["plain", "long"]), MIXED)
    assert recorder.seen["route/harness_none"] == 2.0
    assert recorder.seen["route/harness_switches"] == 0.0
    assert recorder.seen["route/harness_active_plain"] == 1.0


def test_a_run_with_no_harness_arm_logs_no_harness_keys(router_factory, recorder):
    """The namespace announces the axis, so an unconfigured run must not claim to have one."""
    router_factory(action=HarnessAction.NONE)
    advantages(actor_with(None), MIXED)
    assert not [k for k in recorder.seen if k.startswith("route/harness")], recorder.seen


# --------------------------------------------------------- the arms actually differ -------


def test_two_arms_differing_only_in_harness_routing_behave_differently(
    router_factory, recorder
):
    """The headline claim, stated as the smallest experiment that could refute it.

    Identical config, identical router, identical batch. The only difference is whether the
    variant set has two members or one. If these two runs were indistinguishable, the harness
    axis would be a label, which is precisely the state this work started from.
    """
    router_factory(action=HarnessAction.PROPOSE)

    arm = actor_with(["plain", "long"])
    advantages(arm, MIXED)
    arm_metrics = dict(recorder.seen)

    recorder.seen.clear()
    control = actor_with(["plain"])
    advantages(control, MIXED)
    control_metrics = dict(recorder.seen)

    assert arm._selfevo_harness.active.name != control._selfevo_harness.active.name
    assert arm_metrics["route/harness_switches"] == 1.0
    assert control_metrics["route/harness_switches"] == 0.0
    assert arm_metrics["route/harness_can_evolve"] == 1.0
    assert control_metrics["route/harness_can_evolve"] == 0.0
