"""The harness axis, from ``can_evolve_harness`` to a variant that actually changed.

Two claims are on trial here, and they are different claims.

The first is that ``selfevo.harness.dispatch.HarnessDispatcher`` does what a dispatcher must:
a proposal moves the active variant, a validation does not, nothing moves anywhere outside
the configured set, and a set with fewer than two members reports that it cannot move rather
than pretending. Those are unit tests and they live in the first half.

The second is the one that matters for the paper, and it can only be tested through the REAL
actor: ``RoutingContext.can_evolve_harness`` becomes True in PRODUCTION when and only when a
variant set with two or more members is configured, ``route_all`` is told there is a consumer
only when one exists, and ``_refuse_dropped_harness`` still fires when one does not. A test
that constructed the contexts itself would prove nothing about the seam -- that is exactly the
mistake ``test_actor_router_seam`` was written to stop repeating -- so everything in the second
half goes through ``PPOActor._compute_advantages``, the entry point training calls.

Nothing here touches :mod:`selfevo.harness.mini_swe`. That adapter needs Docker images, a
SWE-bench download and a served model, so a test suite that depended on it could not run on
CPU, and the axis would go untested on every box this project actually uses. The dispatcher
dispatches over CONFIGURATIONS for that reason, and the adapter seam is exercised with a fake.
"""

from __future__ import annotations

import pytest

from selfevo.harness.base import HarnessRollout, HarnessVariant
from selfevo.harness.dispatch import (
    DispatchBatch,
    HarnessDispatcher,
    build_dispatcher,
    round_robin,
)
from selfevo.routing.base import HarnessAction

A = HarnessAction

PLAIN = HarnessVariant("plain", "default budget", step_limit=40)
LONG = HarnessVariant("long", "2.5x budget", step_limit=100)
SHORT = HarnessVariant("short", "the cheap arm", step_limit=15)
PAIR = [PLAIN, LONG]
TRIO = [PLAIN, LONG, SHORT]


# ------------------------------------------------------------------ can_evolve ---------


def test_two_variants_make_the_harness_evolvable():
    """The premise of the whole axis: with somewhere to go, the axis says so.

    This is the value production copies into ``can_evolve_harness``, so if it is wrong every
    router in the repo silently drops its harness action and the arm is its own control.
    """
    assert HarnessDispatcher(PAIR).can_evolve is True


def test_three_variants_are_still_evolvable():
    """Two is a floor, not a special case; a larger set must not trip the same guard."""
    assert HarnessDispatcher(TRIO).can_evolve is True


def test_one_variant_is_not_an_evolvable_harness():
    """A dispatch rule with one possible answer is not a decision.

    The single most important negative in this file. A one-variant arm labelled
    "harness-evolving" would train identically to a run with no harness arm at all, and the
    only thing standing between that arm and a published gap attributed to it is this
    returning False.
    """
    assert HarnessDispatcher([PLAIN]).can_evolve is False


def test_an_empty_variant_set_is_not_evolvable():
    """Zero is below the floor too, and must not read as "unconfigured, so allow it"."""
    d = HarnessDispatcher([])
    assert d.can_evolve is False
    assert d.active is None


def test_a_single_variant_set_still_has_an_active_variant():
    """The control arm must still name a scaffold, or its rollouts have no configuration."""
    assert HarnessDispatcher([PLAIN]).active is PLAIN


def test_the_first_configured_variant_starts_active():
    """Deterministic and positional, so two runs with the same config start in the same place."""
    assert HarnessDispatcher(TRIO).active is PLAIN
    assert HarnessDispatcher([LONG, PLAIN]).active is LONG


# ------------------------------------------------------- what an action does -----------


def test_propose_changes_the_active_variant():
    """The claim the axis rests on: a PROPOSE is observable in the harness afterwards."""
    d = HarnessDispatcher(PAIR)
    rec = d.apply(A.PROPOSE)
    assert d.active is LONG
    assert rec.changed is True
    assert (rec.before, rec.after) == ("plain", "long")


def test_validate_does_not_change_the_active_variant():
    """VALIDATE is the regression case -- measure what we have -- so moving would destroy it."""
    d = HarnessDispatcher(PAIR)
    rec = d.apply(A.VALIDATE)
    assert d.active is PLAIN
    assert rec.changed is False
    assert (rec.before, rec.after) == ("plain", "plain")


def test_none_does_not_change_the_active_variant():
    """NONE is the default on every decision ever constructed; it must be exactly inert."""
    d = HarnessDispatcher(PAIR)
    rec = d.apply(A.NONE)
    assert d.active is PLAIN
    assert rec.changed is False


def test_a_long_run_of_validates_and_nones_never_moves_the_harness():
    """One inert call proves little; a run of them is what a real batch looks like."""
    d = HarnessDispatcher(TRIO)
    for action in [A.NONE, A.VALIDATE] * 25:
        d.apply(action)
    assert d.active is PLAIN


def test_a_proposal_with_nowhere_to_go_is_refused_and_says_so():
    """A refused proposal must be visibly refused, not quietly reported as a switch.

    ``changed`` is asserted separately from ``active`` on purpose: an implementation that
    moved nothing but reported ``changed=True`` would make the switch count -- the number the
    whole axis is read off -- lie in exactly the direction that flatters the method.
    """
    d = HarnessDispatcher([PLAIN])
    rec = d.apply(A.PROPOSE)
    assert d.active is PLAIN
    assert rec.changed is False
    assert "nothing to move to" in rec.reason


def test_a_proposal_against_an_empty_set_is_refused():
    """No variants means no active variant; proposing must not invent one."""
    d = HarnessDispatcher([])
    rec = d.apply(A.PROPOSE)
    assert d.active is None
    assert rec.changed is False


def test_a_bare_string_is_not_an_action():
    """A typo that dispatched as NONE would silence the axis and look like a quiet run."""
    with pytest.raises(ValueError, match="HarnessAction"):
        HarnessDispatcher(PAIR).apply("propose")


# --------------------------------------------------- never outside the set -------------


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20])
def test_the_dispatcher_never_leaves_its_configured_set(n):
    """Whatever the proposal history, the active variant is one that was configured.

    Parametrised over lengths that wrap the three-member set an unequal number of times, so
    an off-by-one in the rotation cannot hide behind a run length that happens to land back
    on a legal member.
    """
    d = HarnessDispatcher(TRIO)
    names = {v.name for v in TRIO}
    for _ in range(n):
        d.apply(A.PROPOSE)
        assert d.active.name in names
    assert d.active in TRIO


def test_a_selector_that_leaves_the_set_is_refused():
    """A custom rule must not be able to dispatch to a scaffold nobody configured.

    ``selector`` is a seam, and the whole point of a seam is that someone else writes the
    thing that plugs into it. A rule returning an unconfigured variant would produce rollouts
    under a scaffold that appears in no config and no log.
    """
    foreign = HarnessVariant("foreign", "never configured", step_limit=5)
    d = HarnessDispatcher(PAIR, selector=lambda variants, current: foreign)
    with pytest.raises(ValueError, match="not in the configured set"):
        d.apply(A.PROPOSE)
    assert d.active is PLAIN, "a refused proposal must not have moved anything"


def test_a_selector_that_stays_put_is_refused():
    """A proposal that cannot move is a no-op the log would report as a proposal.

    This is the same failure mode as a one-variant arm, arriving through a different door,
    and it is worse because ``can_evolve`` is True: the run reports an evolvable harness,
    counts proposals, and is byte-identical to its control.
    """
    d = HarnessDispatcher(PAIR, selector=lambda variants, current: current)
    with pytest.raises(ValueError, match="already-active"):
        d.apply(A.PROPOSE)


def test_a_custom_selector_is_actually_used():
    """The seam has to be reachable, or the parameter is decoration.

    Asserted by choosing a variant round-robin would NOT have chosen: over three members the
    default successor of ``plain`` is ``long``, so landing on ``short`` can only have come
    from the supplied rule.
    """
    d = HarnessDispatcher(TRIO, selector=lambda variants, current: variants[2])
    d.apply(A.PROPOSE)
    assert d.active is SHORT


# -------------------------------------------------------------- round robin -------------


def test_round_robin_visits_every_variant_before_repeating():
    """A measurement arm needs data on every variant, not on whichever two it oscillates over."""
    d = HarnessDispatcher(TRIO)
    seen = [d.active.name]
    for _ in range(2):
        d.apply(A.PROPOSE)
        seen.append(d.active.name)
    assert seen == ["plain", "long", "short"]


def test_round_robin_wraps_back_to_the_start():
    """The rule must be total: there is no last variant with nowhere to go."""
    d = HarnessDispatcher(TRIO)
    for _ in range(3):
        d.apply(A.PROPOSE)
    assert d.active is PLAIN


def test_round_robin_refuses_a_current_it_does_not_know():
    """Caller and dispatcher disagreeing about the set must not resolve by guessing."""
    with pytest.raises(ValueError, match="not in the configured set"):
        round_robin(PAIR, SHORT)


def test_round_robin_refuses_an_empty_set():
    """There is no successor in an empty sequence, and index arithmetic would divide by zero."""
    with pytest.raises(ValueError, match="non-empty"):
        round_robin([], PLAIN)


# ---------------------------------------------------------- construction guards ---------


def test_duplicate_variant_names_are_refused():
    """``["plain", "plain"]`` would report an evolvable harness and dispatch to itself."""
    with pytest.raises(ValueError, match="duplicate variant names"):
        HarnessDispatcher([PLAIN, PLAIN])


def test_variants_that_behave_identically_are_refused():
    """Two names for one configuration is the silent-identity failure with a label on it.

    The members differ in ``name`` and ``description`` and in nothing a rollout can observe,
    so an arm dispatching between them would report switches and produce exactly the
    trajectories of its own control.
    """
    twin = HarnessVariant("twin", "different words, same scaffold", step_limit=40)
    with pytest.raises(ValueError, match="same behaviour"):
        HarnessDispatcher([PLAIN, twin])


def test_variants_differing_only_in_settings_are_accepted():
    """The behaviour check must read ``settings`` too, not just ``step_limit``.

    Otherwise the one axis an adapter actually varies -- its pass-through configuration --
    would be rejected as a duplicate, and the guard would block the feature it protects.
    """
    a = HarnessVariant("a", "tool-heavy", step_limit=40, settings={"tools": "many"})
    b = HarnessVariant("b", "tool-light", step_limit=40, settings={"tools": "few"})
    assert HarnessDispatcher([a, b]).can_evolve is True


def test_a_single_variant_is_not_rejected_as_its_own_duplicate():
    """The control arm must remain constructible, or it cannot be run as a comparison."""
    assert HarnessDispatcher([PLAIN]).active is PLAIN


def test_unhashable_settings_do_not_break_construction():
    """``settings`` is passed through verbatim, so a list in there is legal input."""
    a = HarnessVariant("a", "list settings", step_limit=40, settings={"cmds": ["x"]})
    b = HarnessVariant("b", "other list", step_limit=40, settings={"cmds": ["y"]})
    assert HarnessDispatcher([a, b]).can_evolve is True


# ------------------------------------------------------------------- batches ------------


def test_a_batch_moves_the_harness_at_most_once():
    """N proposals in one batch are N pieces of evidence for ONE decision, not N decisions.

    This is the guard against the nastiest silent failure available here. A harness is a
    shared artefact, so applying every group's proposal in turn would rotate the active
    variant once per proposal -- and with an even number of proposals over a two-variant set
    the step would END where it began. The run would log four switches per batch and be
    indistinguishable, at every step boundary, from an arm that never switched.
    """
    d = HarnessDispatcher(PAIR)
    batch = d.consume([A.PROPOSE] * 4)
    assert d.active is LONG
    assert batch.switches == 1
    assert batch.count(A.PROPOSE) == 4, "every proposal is still counted as evidence"


def test_consecutive_batches_each_move_once():
    """Aggregation is per batch, not per run: the axis must keep moving across steps."""
    d = HarnessDispatcher(TRIO)
    assert [d.consume([A.PROPOSE, A.PROPOSE]).active for _ in range(3)] == [
        "long",
        "short",
        "plain",
    ]


def test_a_validation_earlier_in_the_batch_does_not_block_a_later_proposal():
    """Aggregation keys on what MOVED the harness, not on "some action happened".

    Not a corner case: routers emit VALIDATE for solved groups and PROPOSE for failed ones,
    in group order, so a batch whose first non-inert action is a validation is the ordinary
    shape. If one solved group at the head of a batch could silence every proposal behind it,
    the axis would go quiet for a reason no metric names.
    """
    d = HarnessDispatcher(PAIR)
    batch = d.consume([A.VALIDATE, A.NONE, A.PROPOSE])
    assert d.active is LONG
    assert batch.switches == 1


def test_a_batch_of_validates_moves_nothing():
    """A batch that only validates is a measurement step and must leave the harness alone."""
    d = HarnessDispatcher(PAIR)
    batch = d.consume([A.VALIDATE, A.VALIDATE, A.NONE])
    assert d.active is PLAIN
    assert batch.switches == 0


def test_the_active_variant_persists_across_batches():
    """The selection is state, not a per-batch computation; resetting it erases the axis."""
    d = HarnessDispatcher(PAIR)
    d.consume([A.PROPOSE])
    d.consume([A.NONE, A.NONE])
    assert d.active is LONG


def test_a_refused_proposal_does_not_consume_the_batch_s_one_move():
    """Aggregation must key on what HAPPENED, not on what was asked for.

    A one-variant dispatcher refuses every proposal, so a batch full of them must report zero
    switches rather than "we already moved this batch" after the first refusal.
    """
    d = HarnessDispatcher([PLAIN])
    batch = d.consume([A.PROPOSE] * 3)
    assert batch.switches == 0
    assert batch.count(A.PROPOSE) == 3


# ------------------------------------------------------------------- metrics ------------


def test_metrics_count_each_action_separately():
    """Three counters, because a single "harness actions" total hides which axis moved."""
    d = HarnessDispatcher(PAIR)
    m = d.consume([A.NONE, A.NONE, A.PROPOSE, A.VALIDATE]).as_metrics()
    assert m["route/harness_none"] == 2.0
    assert m["route/harness_propose"] == 1.0
    assert m["route/harness_validate"] == 1.0


def test_metrics_report_switches_separately_from_proposals():
    """What the router ASKED for and what the harness DID are different numbers.

    An arm whose every proposal is refused has a healthy propose count and zero switches, and
    a log that reported only the first would describe it as a working harness arm.
    """
    refused = HarnessDispatcher([PLAIN]).consume([A.PROPOSE] * 2).as_metrics()
    acted = HarnessDispatcher(PAIR).consume([A.PROPOSE] * 2).as_metrics()
    assert refused["route/harness_propose"] == acted["route/harness_propose"] == 2.0
    assert refused["route/harness_switches"] == 0.0
    assert acted["route/harness_switches"] == 1.0


def test_metrics_name_the_active_variant():
    """The active variant is in the KEY, so a dashboard can plot which scaffold was live."""
    d = HarnessDispatcher(PAIR)
    before = d.consume([A.NONE]).as_metrics()
    assert before["route/harness_active_plain"] == 1.0
    assert before["route/harness_active_long"] == 0.0
    after = d.consume([A.PROPOSE]).as_metrics()
    assert after["route/harness_active_plain"] == 0.0
    assert after["route/harness_active_long"] == 1.0


def test_metrics_report_whether_the_harness_could_move_at_all():
    """A one-variant control and a two-variant arm must be separable from the logs alone."""
    control = HarnessDispatcher([PLAIN]).consume([A.NONE]).as_metrics()
    arm = HarnessDispatcher(PAIR).consume([A.NONE]).as_metrics()
    assert control["route/harness_can_evolve"] == 0.0
    assert control["route/harness_n_variants"] == 1.0
    assert arm["route/harness_can_evolve"] == 1.0
    assert arm["route/harness_n_variants"] == 2.0


def test_a_quiet_batch_still_emits_the_whole_key_set():
    """A key that appears only on eventful steps is a key no panel can plot against.

    This repo has already shipped a routed run with an empty ``route/`` namespace and had to
    reconstruct what it did from the config afterwards.
    """
    busy = set(HarnessDispatcher(PAIR).consume([A.PROPOSE, A.VALIDATE]).as_metrics())
    quiet = set(HarnessDispatcher(PAIR).consume([A.NONE, A.NONE]).as_metrics())
    assert busy == quiet
    assert set(HarnessDispatcher(PAIR).consume([]).as_metrics()) == busy


def test_every_metric_is_a_float():
    """``stats_tracker.scalar`` takes numbers; a stray bool or str would fail in a live run."""
    m = HarnessDispatcher(PAIR).consume([A.PROPOSE]).as_metrics()
    assert all(type(v) is float for v in m.values()), m


def test_metric_keys_live_under_the_route_namespace():
    """The actor's own convention, so the harness axis lands on the same panel as the rest."""
    m = HarnessDispatcher(PAIR).consume([A.PROPOSE]).as_metrics()
    assert all(k.startswith("route/") for k in m), sorted(m)


def test_records_are_kept_one_per_action_in_order():
    """The audit trail: a switch has to be reconstructable from the batch after the fact."""
    batch = HarnessDispatcher(PAIR).consume([A.NONE, A.PROPOSE, A.VALIDATE])
    assert isinstance(batch, DispatchBatch)
    assert [r.action for r in batch.records] == [A.NONE, A.PROPOSE, A.VALIDATE]
    assert [r.changed for r in batch.records] == [False, True, False]


# ------------------------------------------------------------- the adapter seam ---------


class _FakeAdapter:
    """Records which variant a rollout was requested under, with no Docker in sight."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def variants(self):
        """Part of the HarnessAdapter protocol; unused by the dispatcher."""
        return tuple(PAIR)

    def run(self, task_id: str, variant: HarnessVariant) -> HarnessRollout:
        """Record the pairing and return a usable rollout."""
        self.seen.append(variant.name)
        return HarnessRollout(task_id, variant.name, solved=True, steps=1)


def test_running_without_an_adapter_raises_rather_than_fabricating_a_rollout():
    """A fabricated unsolved rollout would score an infrastructure failure as reward 0."""
    with pytest.raises(RuntimeError, match="no HarnessAdapter"):
        HarnessDispatcher(PAIR).run("some__task-1")


def test_a_rollout_runs_under_whatever_variant_is_active():
    """The dispatch has to reach the thing being dispatched, or it selects nothing."""
    fake = _FakeAdapter()
    d = HarnessDispatcher(PAIR, adapter=fake)
    d.run("t1")
    d.apply(A.PROPOSE)
    d.run("t2")
    assert fake.seen == ["plain", "long"]


def test_the_dispatch_path_needs_no_adapter():
    """The premise of testing this at all: Docker is not available where this runs."""
    d = HarnessDispatcher(PAIR)
    assert d.adapter is None
    d.consume([A.PROPOSE])
    assert d.active is LONG


# ------------------------------------------------------------------- build ---------------


def test_no_variant_set_builds_no_dispatcher():
    """``None`` in, ``None`` out: what every run before this module existed had."""
    assert build_dispatcher(None) is None
    assert build_dispatcher([]) is None


def test_a_configured_pair_builds_an_evolvable_dispatcher():
    """The production path from a config list of names to a live consumer."""
    d = build_dispatcher(["plain", "long"])
    assert d.can_evolve is True
    assert [v.name for v in d.variants] == ["plain", "long"]


def test_an_unknown_variant_name_is_refused_rather_than_skipped():
    """Skipping it would shrink a two-variant arm into its own one-variant control."""
    with pytest.raises(ValueError, match="unknown harness variant"):
        build_dispatcher(["plain", "no-such-variant"])


def test_the_configured_order_is_the_dispatch_order():
    """The proposal rule is positional, so config order is reproducible run history."""
    assert build_dispatcher(["long", "plain"]).active.name == "long"
