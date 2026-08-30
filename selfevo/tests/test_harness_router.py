"""Tests for harness-aware routing.

The router exists because ``I_RL(p, G) = 1 - p**G - (1 - p)**G`` is exactly zero at ``p = 0``
and ``p = 1``: the unanimous groups are precisely the ones GRPO cannot learn from, and they
are the only ones this router redirects. These tests pin the whole DECISION TABLE rather
than the classification, because a router that identifies the three regimes correctly and
then sends them to the wrong destination is indistinguishable from a working one if only
the classification is checked -- the same failure the cluster tests were written against.

Three properties are load-bearing and are swept rather than sampled:

* ROLLBACK -- with ``can_evolve_harness=False`` the harness axis is inert for every
  configuration, every granularity and every solve rate, so a run with no harness arm
  behaves as it did before the axis existed;
* SAFETY -- a mode registered ``needs_teacher=True`` is never emitted for a unit with no
  target, external or self;
* PARTITION -- ``partition=True`` is Co-Harness exactly: a unit feeds the model or the
  harness, never both.
"""

from __future__ import annotations

import itertools

import pytest

from selfevo.compose import PipelineConfig, ROUTERS, validate
from selfevo.routing.base import (
    Granularity,
    HarnessAction,
    Router,
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)
from selfevo.routing.harness import CoHarnessRouter

GRANS = tuple(Granularity)
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def ctx(p, *, teacher=False, evolve=False, g=4, gran=Granularity.SAMPLE):
    """A routing context, built by keyword so no field order can silently rebind."""
    return RoutingContext(
        solve_rate=p, group_size=g, granularity=gran,
        has_teacher=teacher, can_evolve_harness=evolve,
    )


def mode_of(d):
    """The single mode a hard routing decision selected."""
    assert len(d.weights) == 1, f"expected a one-hot decision, got {d.weights}"
    return next(iter(d.weights))


# ------------------------------------------------------------------- the decision table

# (solve_rate, can_evolve_harness, has_teacher) -> (mode, harness action), defaults.
_TABLE = {
    (0.0, False, False): (TrainingMode.SKIP, HarnessAction.NONE),
    (0.0, False, True):  (TrainingMode.SFT,  HarnessAction.NONE),
    (0.0, True,  False): (TrainingMode.SKIP, HarnessAction.PROPOSE),
    (0.0, True,  True):  (TrainingMode.SFT,  HarnessAction.PROPOSE),
    (0.5, False, False): (TrainingMode.RL,   HarnessAction.NONE),
    (0.5, False, True):  (TrainingMode.RL,   HarnessAction.NONE),
    (0.5, True,  False): (TrainingMode.RL,   HarnessAction.NONE),
    (0.5, True,  True):  (TrainingMode.RL,   HarnessAction.NONE),
    (1.0, False, False): (TrainingMode.SFT,  HarnessAction.NONE),
    (1.0, False, True):  (TrainingMode.SFT,  HarnessAction.NONE),
    (1.0, True,  False): (TrainingMode.SFT,  HarnessAction.VALIDATE),
    (1.0, True,  True):  (TrainingMode.SFT,  HarnessAction.VALIDATE),
}


def test_the_default_decision_table_is_pinned_cell_by_cell():
    """Both axes of every (solve_rate x can_evolve_harness x has_teacher) cell."""
    r = CoHarnessRouter()
    for (p, evolve, teacher), expected in _TABLE.items():
        d = r.route(ctx(p, teacher=teacher, evolve=evolve))
        got = (mode_of(d), d.harness)
        assert got == expected, f"p={p} evolve={evolve} teacher={teacher}: {got} != {expected}"


def test_an_all_solved_group_is_sft_on_its_own_sample_with_no_teacher():
    """p=1 kills the RL gradient but guarantees a correct sample: rejection-sampling FT."""
    d = CoHarnessRouter().route(ctx(1.0, teacher=False))
    assert mode_of(d) == TrainingMode.SFT
    assert "own correct sample" in d.reason


def test_an_all_failed_group_with_no_teacher_is_the_harness_only_case():
    """RL is dead and there is no self-target, so the harness is the only consumer left."""
    d = CoHarnessRouter().route(ctx(0.0, teacher=False, evolve=True))
    assert mode_of(d) == TrainingMode.SKIP
    assert d.harness is HarnessAction.PROPOSE
    assert "harness" in d.reason


def test_one_unit_feeds_both_the_gradient_and_the_harness():
    """The point of a separate action axis: a partition throws away one of two readings."""
    d = CoHarnessRouter().route(ctx(0.0, teacher=True, evolve=True))
    assert mode_of(d) == TrainingMode.SFT
    assert d.harness is HarnessAction.PROPOSE


def test_a_mixed_group_is_left_to_rl_and_never_reaches_the_harness():
    """0 < p < 1 is where I_RL > 0; redirecting it would remove signal RL can use."""
    for teacher, evolve in itertools.product((False, True), repeat=2):
        d = CoHarnessRouter().route(ctx(0.5, teacher=teacher, evolve=evolve))
        assert mode_of(d) == TrainingMode.RL
        assert d.harness is HarnessAction.NONE


def test_validate_on_success_off_keeps_solved_units_out_of_the_harness():
    """The regression-case offer is opt-out, and opting out must not disturb the mode."""
    d = CoHarnessRouter(validate_on_success=False).route(ctx(1.0, evolve=True))
    assert d.harness is HarnessAction.NONE
    assert mode_of(d) == TrainingMode.SFT


def test_the_self_target_mode_is_configurable_not_hardcoded():
    """A mutant that hardcodes SFT passes every default-configuration test."""
    r = CoHarnessRouter(self_target_mode=TrainingMode.DISTILL)
    assert mode_of(r.route(ctx(1.0, teacher=True))) == TrainingMode.DISTILL


def test_a_self_target_mode_needing_no_teacher_is_not_gated_on_a_target():
    """known_modes() is the gate, not the mode name; a needs_teacher=False mode is free."""
    d = CoHarnessRouter(self_target_mode=TrainingMode.RL).route(
        ctx(1.0, teacher=False, gran=Granularity.TOKEN)
    )
    assert mode_of(d) == TrainingMode.RL


# ------------------------------------------------------------------------ partition mode


def test_partition_never_sends_a_unit_to_both_destinations():
    """Co-Harness is either/or; a cell that trains AND evolves is not a reproduction."""
    r = CoHarnessRouter(partition=True)
    for p, teacher, gran in itertools.product(RATES, (False, True), GRANS):
        d = r.route(ctx(p, teacher=teacher, evolve=True, gran=gran))
        trains = mode_of(d) != TrainingMode.SKIP
        evolves = d.harness is not HarnessAction.NONE
        assert not (trains and evolves), f"both at p={p} teacher={teacher} gran={gran}"


def test_partition_sends_failure_to_the_harness_and_success_to_the_model():
    """The direction matters: swapping the two branches still satisfies 'never both'."""
    r = CoHarnessRouter(partition=True)
    fail = r.route(ctx(0.0, teacher=True, evolve=True))
    assert mode_of(fail) == TrainingMode.SKIP
    assert fail.harness is HarnessAction.PROPOSE
    ok = r.route(ctx(1.0, teacher=True, evolve=True))
    assert mode_of(ok) == TrainingMode.SFT
    assert ok.harness is HarnessAction.NONE


def test_partition_overrides_validate_on_success():
    """Documented as ignored under partition -- checked, because a doc is not a test."""
    d = CoHarnessRouter(partition=True, validate_on_success=True).route(ctx(1.0, evolve=True))
    assert d.harness is HarnessAction.NONE


def test_the_relaxed_default_does_emit_both_which_is_why_the_axis_is_separate():
    """If the default also partitioned, the orthogonal axis would buy nothing."""
    d = CoHarnessRouter(partition=False).route(ctx(0.0, teacher=True, evolve=True))
    assert mode_of(d) != TrainingMode.SKIP
    assert d.harness is not HarnessAction.NONE


def test_partition_without_a_harness_arm_discards_failed_units():
    """PINNED HAZARD, not an endorsement.

    Under ``partition`` a failed unit is handed to a harness that, with
    ``can_evolve_harness=False``, does not exist -- so it trains nothing even when a teacher
    could have supplied a target. Only reachable by running this router with
    ``evolve_target='model'``, which :func:`selfevo.compose.validate` rejects. Pinned so the
    sink cannot appear or disappear silently.
    """
    sunk = CoHarnessRouter(partition=True).route(ctx(0.0, teacher=True, evolve=False))
    assert mode_of(sunk) == TrainingMode.SKIP
    assert sunk.harness is HarnessAction.NONE
    kept = CoHarnessRouter(partition=False).route(ctx(0.0, teacher=True, evolve=False))
    assert mode_of(kept) == TrainingMode.SFT


# ------------------------------------------------------------------ threshold validation


def test_thresholds_that_cross_or_touch_are_refused():
    """At failed == solved a unit on the boundary is both, and the mixed band vanishes."""
    for solved, failed in ((0.5, 0.5), (0.2, 0.8), (0.0, 0.0), (1.0, 1.0)):
        with pytest.raises(ValueError, match="must be below"):
            CoHarnessRouter(solved_threshold=solved, failed_threshold=failed)


def test_thresholds_outside_the_unit_interval_are_refused():
    with pytest.raises(ValueError, match="must be in"):
        CoHarnessRouter(solved_threshold=1.5)
    with pytest.raises(ValueError, match="must be in"):
        CoHarnessRouter(failed_threshold=-0.1)
    with pytest.raises(ValueError, match="must be in"):
        CoHarnessRouter(solved_threshold=-0.5, failed_threshold=-0.7)


def test_an_unregistered_self_target_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        CoHarnessRouter(self_target_mode="telepathy")


def test_no_accepted_threshold_pair_makes_a_unit_both_solved_and_failed():
    """The guarantee is about the accepted REGION, not about the two defaults."""
    grid = [i / 8 for i in range(9)]
    accepted = 0
    for solved, failed in itertools.product(grid, grid):
        try:
            CoHarnessRouter(solved_threshold=solved, failed_threshold=failed)
        except ValueError:
            continue
        accepted += 1
        for p in grid:
            assert not (p >= solved and p <= failed), \
                f"p={p} is both solved and failed at solved={solved} failed={failed}"
    assert accepted > 0, "the sweep accepted no configuration, so it proved nothing"


def test_every_solve_rate_reaches_exactly_one_regime():
    """'Neither' would be a unit that silently produces no decision at all."""
    r = CoHarnessRouter()
    for i in range(21):
        d = r.route(ctx(i / 20, teacher=True, evolve=True))
        assert len(d.weights) == 1
        assert sum(d.weights.values()) == 1.0


def test_the_thresholds_are_inclusive_at_the_boundary():
    """A unanimous group sits exactly ON the default threshold; an exclusive comparison
    would classify the one regime this router exists for as mixed and change nothing."""
    r = CoHarnessRouter(solved_threshold=0.75, failed_threshold=0.25)
    assert r.route(ctx(0.75, teacher=True, evolve=True)).harness is HarnessAction.VALIDATE
    assert r.route(ctx(0.25, teacher=True, evolve=True)).harness is HarnessAction.PROPOSE
    assert r.route(ctx(0.5, teacher=True, evolve=True)).harness is HarnessAction.NONE
    d = CoHarnessRouter().route(ctx(1.0, teacher=True, evolve=True))
    assert d.harness is HarnessAction.VALIDATE, "solve_rate == solved_threshold must count"
    d = CoHarnessRouter().route(ctx(0.0, teacher=True, evolve=True))
    assert d.harness is HarnessAction.PROPOSE, "solve_rate == failed_threshold must count"


def test_a_relaxed_failed_threshold_still_uses_a_self_target_when_one_exists():
    """Only p=0 is genuinely target-free. With failed_threshold > 0 a 'failed' group can
    still contain a correct sample, and skipping it throws away a free target."""
    r = CoHarnessRouter(failed_threshold=0.25)
    d = r.route(ctx(0.25, teacher=False, evolve=True))
    assert mode_of(d) == TrainingMode.SFT
    assert "own correct sample" in d.reason
    assert d.harness is HarnessAction.PROPOSE
    assert mode_of(r.route(ctx(0.0, teacher=False, evolve=True))) == TrainingMode.SKIP


# ----------------------------------------------------------------------- TOKEN semantics


def test_token_granularity_has_no_self_target_but_coarser_ones_do():
    """A group mean does not describe a sibling sample at token resolution."""
    for p in RATES:
        assert ctx(p, gran=Granularity.TOKEN).has_self_target is False
    for gran in (Granularity.TASK, Granularity.CLUSTER, Granularity.SAMPLE):
        assert ctx(1.0, gran=gran).has_self_target is True
        assert ctx(0.25, gran=gran).has_self_target is True
        assert ctx(0.0, gran=gran).has_self_target is False


def test_has_target_is_external_or_self():
    assert ctx(1.0).has_target is True
    assert ctx(0.0).has_target is False
    assert ctx(0.0, teacher=True).has_target is True
    assert ctx(1.0, gran=Granularity.TOKEN).has_target is False
    assert ctx(1.0, teacher=True, gran=Granularity.TOKEN).has_target is True


def test_a_solved_token_unit_without_a_teacher_is_skipped_not_sft():
    """SFT is registered needs_teacher=True; emitting it here trains on a missing target."""
    d = CoHarnessRouter().route(ctx(1.0, teacher=False, evolve=True, gran=Granularity.TOKEN))
    assert mode_of(d) == TrainingMode.SKIP
    assert "no target is available" in d.reason
    assert d.harness is HarnessAction.VALIDATE, "the harness axis is independent of the mode"


def test_a_solved_token_unit_with_a_teacher_does_not_claim_a_self_target():
    """The reason is an audit record; at TOKEN there is no self-target to attribute to."""
    d = CoHarnessRouter().route(ctx(1.0, teacher=True, gran=Granularity.TOKEN))
    assert mode_of(d) == TrainingMode.SFT
    assert "external teacher" in d.reason
    assert "own correct sample" not in d.reason


def test_a_teacher_requiring_mode_is_never_emitted_without_a_target():
    """The one safety property, swept over every configuration and granularity."""
    checked = 0
    for part, vos, teacher, evolve, gran, p in itertools.product(
        (False, True), (False, True), (False, True), (False, True), GRANS, RATES
    ):
        c = ctx(p, teacher=teacher, evolve=evolve, gran=gran)
        for r in (CoHarnessRouter(partition=part, validate_on_success=vos),
                  CoHarnessRouter(partition=part, validate_on_success=vos,
                                  failed_threshold=0.25, solved_threshold=0.75)):
            mode = mode_of(r.route(c))
            checked += 1
            if known_modes()[mode]:
                assert c.has_target, \
                    f"{mode} needs a teacher but p={p} gran={gran.value} teacher={teacher}"
    assert checked == 2 * 2 * 2 * 2 * len(GRANS) * len(RATES) * 2


# ------------------------------------------------------------------- the rollback property


def test_the_harness_axis_is_inert_without_a_harness_arm():
    """ROLLBACK: can_evolve_harness=False gives NONE for every configuration and context."""
    checked = 0
    for part, vos, teacher, gran, p in itertools.product(
        (False, True), (False, True), (False, True), GRANS, RATES
    ):
        d = CoHarnessRouter(partition=part, validate_on_success=vos).route(
            ctx(p, teacher=teacher, evolve=False, gran=gran)
        )
        assert d.harness is HarnessAction.NONE, \
            f"harness action leaked: part={part} vos={vos} p={p} gran={gran.value}"
        checked += 1
    assert checked == 2 * 2 * 2 * len(GRANS) * len(RATES)


def test_a_dropped_harness_action_says_so_rather_than_looking_deliberate():
    """A silently absent action is indistinguishable from a unit the router declined."""
    d = CoHarnessRouter().route(ctx(0.0, teacher=True, evolve=False))
    assert d.harness is HarnessAction.NONE
    assert "no harness arm" in d.reason


def test_a_decision_built_the_old_way_still_means_none():
    """Every RoutingDecision constructed before this field existed meant HarnessAction.NONE."""
    d = RoutingDecision({TrainingMode.RL: 1.0}, reason="static")
    assert d.harness is HarnessAction.NONE
    assert d == RoutingDecision({TrainingMode.RL: 1.0}, reason="static")
    assert d != RoutingDecision({TrainingMode.RL: 1.0}, reason="static",
                                harness=HarnessAction.PROPOSE)
    assert d.normalised() == {TrainingMode.RL: 1.0}
    assert d.argmax() == TrainingMode.RL


def test_a_context_built_the_old_way_has_no_harness_arm():
    assert RoutingContext(solve_rate=0.5, group_size=4).can_evolve_harness is False


def test_adding_the_harness_flag_did_not_shift_the_positional_signature():
    """unit_id was the fifth positional argument before can_evolve_harness existed. If the
    new field took that slot, a unit id would silently become a truthy harness arm and the
    id would be lost -- both silently, since neither field is type-checked."""
    c = RoutingContext(0.5, 4, Granularity.SAMPLE, True, "unit-7")
    assert c.unit_id == "unit-7"
    assert c.can_evolve_harness is False


def test_the_routers_that_predate_this_axis_never_emit_an_action():
    """Rollback across the whole router registry, not just this one router."""
    from selfevo.routing.routers import (
        InvertedRouter, RandomRouter, SolveRateRouter, StaticRouter,
    )
    for r in (StaticRouter(), SolveRateRouter(), RandomRouter(), InvertedRouter()):
        for p in RATES:
            d = r.route(ctx(p, teacher=True, evolve=True))
            assert d.harness is HarnessAction.NONE, f"{type(r).__name__} at p={p}"


# ------------------------------------------------------------------------- wiring


def test_the_router_satisfies_the_protocol_and_is_registered():
    assert isinstance(CoHarnessRouter(), Router)
    made = ROUTERS["coharness"](partition=True, failed_threshold=0.1)
    assert isinstance(made, CoHarnessRouter)
    assert made.partition is True
    assert made.failed_threshold == 0.1


def test_the_factory_surfaces_a_bad_configuration_instead_of_swallowing_it():
    with pytest.raises(ValueError):
        ROUTERS["coharness"](solved_threshold=2.0)


def test_harness_action_is_importable_from_the_routing_package():
    """base.__all__ is the package's public surface and every other name in it is re-exported."""
    import selfevo.routing as routing

    assert routing.HarnessAction is HarnessAction


def test_a_coharness_config_with_nothing_to_evolve_is_rejected():
    """With evolve_target='model' every action is dropped and the arm degenerates into a
    solve-rate split under a different name -- a null arm that still reports as harness."""
    problems = validate(PipelineConfig(router="coharness", evolve_target="model"))
    assert any(p.axes == ("router", "evolve_target") for p in problems), problems
    assert not validate(PipelineConfig(router="coharness", evolve_target="harness"))
    assert not validate(PipelineConfig(router="solve_rate", evolve_target="model"))
