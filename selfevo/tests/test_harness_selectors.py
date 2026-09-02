"""The feature-driven harness rule, and the control that makes it evidence rather than a story.

Two claims are on trial, and they are not the same claim.

The first is that :class:`~selfevo.harness.selectors.TruncationStepLimitSelector` does what
the paper predicts: a batch that keeps running out of steps moves the harness to a LONGER step
budget, a batch that never runs out moves it to a shorter one, and everything in between moves
nothing. The negatives matter as much as the positive -- a rule that moved on every proposal
would be round-robin with a feature-shaped docstring, and this repo has shipped that mistake
before under other names.

The second is that :class:`~selfevo.harness.selectors.RateMatchedControlSelector` switches at
the SAME RATE while reading nothing. Without that, a treatment arm is uninterpretable: this
project has three separate findings in which a targeted rule turned out to be
indistinguishable from a random one applied at the same rate, so an arm reported against a
no-harness control measures the intervention rate and nothing else. The rate-matching tests
below are therefore not book-keeping -- they are the tests that decide whether the experiment
is a comparison at all.

Everything here runs through a real
:class:`~selfevo.harness.dispatch.HarnessDispatcher` wherever a dispatcher is what production
would use, because a test that called the selector directly would prove nothing about the
seam -- the refusal path in particular exists only because ``apply`` translates it.
"""

from __future__ import annotations

import math
import random

import pytest

from selfevo.harness.base import HarnessVariant
from selfevo.harness.dispatch import HarnessDispatcher, HarnessSelectionRefused
from selfevo.harness.selectors import (
    SELECTOR_METRIC_KEYS,
    TRUNCATION_FEATURE,
    RateMatchedControlSelector,
    TruncationStepLimitSelector,
)
from selfevo.observability import GroupFeatures
from selfevo.routing.base import HarnessAction
from selfevo.routing.contextual import MissingFeatures

A = HarnessAction

# A LADDER, not a pair. Four rungs is the smallest set on which "move one step" and "move to
# the extreme" are different behaviours in both directions, and the difference between them is
# the rule's central design choice.
TINY = HarnessVariant("tiny", "smallest budget", step_limit=5)
SHORT = HarnessVariant("short", "the cheap arm", step_limit=15)
PLAIN = HarnessVariant("plain", "default budget", step_limit=40)
LONG = HarnessVariant("long", "2.5x budget", step_limit=100)
LADDER = [TINY, SHORT, PLAIN, LONG]
PAIR = [PLAIN, LONG]


def rows(*fractions: float) -> list[dict[str, float]]:
    """One feature mapping per group, carrying the feature the rule reads.

    Args:
        fractions: ``truncated_fraction`` for each group in the batch.

    Returns:
        The shape ``RoutingContext.extra`` carries, one entry per routed group.
    """
    return [{TRUNCATION_FEATURE: f} for f in fractions]


def treatment(**kwargs) -> TruncationStepLimitSelector:
    """A treatment selector at the shipped thresholds unless a test overrides them."""
    return TruncationStepLimitSelector(**kwargs)


def drive(dispatcher, selector, batches, proposals: int = 1) -> list:
    """Run batches through ``observe`` then ``consume``, in production's order.

    Args:
        dispatcher: The dispatcher under test.
        selector: The selector plugged into it, observed once per batch.
        batches: One feature-row list per batch.
        proposals: PROPOSE actions per batch. More than one is the ordinary case -- routers
            emit one action per group -- and is what makes the once-per-observation rule load
            bearing.

    Returns:
        One :class:`~selfevo.harness.dispatch.DispatchBatch` per batch.
    """
    out = []
    for batch in batches:
        selector.observe(batch)
        out.append(dispatcher.consume([A.PROPOSE] * proposals))
    return out


def realistic_batches(seed: int = 11, groups: int = 4) -> list[list[dict[str, float]]]:
    """A truncation stream shaped like a run whose budget is first too small, then too large.

    Three regimes, twice over: heavily truncated (the budget binds), middling (it binds for
    some), and almost never truncated (it is slack). That is the sequence the rule is FOR --
    a fixed regime would let a rule that always moved and a rule that never moved look alike
    over most of the run -- and it is what a rate measured off it is a rate over.

    Deterministic given ``seed``: an unreproducible decision sequence could not be replayed by
    a matched control, which is the whole point of the control.
    """
    rng = random.Random(seed)
    out: list[list[dict[str, float]]] = []
    for centre in (0.9, 0.3, 0.02) * 2:
        for _ in range(8):
            out.append(
                rows(*[min(1.0, max(0.0, rng.gauss(centre, 0.05))) for _ in range(groups)])
            )
    return out


# ------------------------------------------------- the rule moves where the feature points --


def test_high_truncation_moves_to_a_longer_step_limit():
    """The paper's prediction, in one assertion: out of steps means give it more steps.

    Everything else in this file is a qualification of this line. If it fails, the harness
    axis is round-robin with a different docstring.
    """
    sel = treatment()
    d = HarnessDispatcher(PAIR, selector=sel)
    batch = drive(d, sel, [rows(1.0, 0.9, 1.0, 0.8)])[0]
    assert d.active is LONG
    assert batch.switches == 1
    assert d.active.step_limit > PLAIN.step_limit


def test_the_move_is_one_variant_not_a_jump_to_the_largest_budget():
    """Direction is measured, magnitude is not, so the rule must not claim a magnitude.

    From ``tiny`` on a four-rung ladder a jump to the extreme and a single step are different
    variants, and only the single step keeps the arm's trajectory a function of the FEATURE
    rather than of how wide the configured set happens to be.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    drive(d, sel, [rows(1.0)])
    assert d.active is SHORT, "one rung up from tiny, not straight to long"


def test_sustained_truncation_walks_up_the_ladder_one_rung_per_batch():
    """A run that keeps truncating keeps climbing, and stops at the top rather than wrapping.

    The wrap is what round-robin does, and it is exactly wrong here: wrapping from the longest
    budget to the shortest on a batch that is starved of steps would take the harness in the
    direction the feature argues against, while reporting a switch.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    seen = [d.active.name]
    for _ in range(5):
        sel.observe(rows(1.0))
        d.consume([A.PROPOSE])
        seen.append(d.active.name)
    assert seen == ["tiny", "short", "plain", "long", "long", "long"]


def test_low_truncation_moves_to_a_shorter_step_limit():
    """The symmetric half, and the clause that makes the axis about adaptation, not compute.

    Without it, "the harness follows the feature" is confounded with "the harness gets a
    bigger budget", and a gain over a fixed-budget control could be bought entirely with the
    extra steps.
    """
    sel = treatment()
    # Configured with the longest budget first, so the run STARTS at the top and the downward
    # branch has somewhere to go. Ordering, not private state: the first configured variant is
    # the active one, which is the documented production behaviour.
    d = HarnessDispatcher([LONG, PLAIN, SHORT, TINY], selector=sel)
    drive(d, sel, [rows(0.0, 0.0, 0.02, 0.0)])
    assert d.active is PLAIN


def test_the_statistic_is_the_mean_over_groups_not_the_max():
    """One pathological group must not move a SHARED artefact.

    ``[1.0, 0.0, 0.0, 0.0]`` has mean 0.25 -- inside the dead band -- while its max is 1.0 and
    its min is 0.0. A rule reducing by max would raise the budget for the whole batch on the
    evidence of one group; by min it would cut it on the evidence of one group. Only the mean
    answers the question the step limit acts on, which is what fraction of ROLLOUTS ran out.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batch = drive(d, sel, [rows(1.0, 0.0, 0.0, 0.0)])[0]
    assert d.active is TINY
    assert batch.switches == 0
    assert batch.refusals == 1


def test_a_batch_truncated_for_most_of_its_groups_moves_up():
    """The threshold is about the median rollout, so three groups in four is enough."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    drive(d, sel, [rows(1.0, 1.0, 1.0, 0.0)])
    assert d.active is SHORT


def test_production_shaped_feature_rows_are_accepted():
    """The rule must read the mapping ``group_features`` really produces, not a test fixture.

    ``GroupFeatures.as_extra`` is what ``RoutingContext.extra`` carries in the actor, so a
    rule that only worked on a hand-built single-key dict would be a rule with no production
    input.
    """
    feats = GroupFeatures(
        solve_rate=0.0,
        reward_std=0.0,
        mean_response_len=900.0,
        len_dispersion=0.1,
        mean_logprob=-0.5,
        logprob_dispersion=0.2,
        truncated_fraction=1.0,
    )
    sel = treatment()
    d = HarnessDispatcher(PAIR, selector=sel)
    drive(d, sel, [[feats.as_extra()]])
    assert d.active is LONG


# --------------------------------------- it does not move when the feature says nothing -----


def test_dead_band_truncation_does_not_move_the_harness():
    """The negative the whole design rests on: a proposal the feature does not justify.

    ``apply`` records it as a refusal rather than a switch, so the arm's own metrics separate
    "the rule declined" from "no proposal arrived" -- and the run continues, because a dead
    band is a data condition and not a bug.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batch = drive(d, sel, [rows(0.3, 0.25, 0.3, 0.2)])[0]
    assert d.active is TINY
    assert batch.switches == 0
    assert batch.refusals == 1
    assert "dead band" in batch.records[0].reason
    assert batch.records[0].refused is True
    assert batch.records[0].changed is False


def test_a_long_run_inside_the_dead_band_never_moves():
    """One quiet batch proves little; twenty is what a run in a settled regime looks like."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batches = drive(d, sel, [rows(0.3, 0.3)] * 20)
    assert d.active is TINY
    assert sum(b.switches for b in batches) == 0
    assert sel.moves == 0
    assert sel.decisions == 20


def test_the_dead_band_edges_are_where_the_docstring_says_they_are():
    """A threshold whose boundary is off by one comparison is a different rule.

    ``>=`` and ``<=`` are asserted at the exact configured values, because the band is the
    part of the rule that decides how often the arm intervenes at all -- which is the quantity
    the control has to match.
    """
    # From tiny: 0.5 is AT raise_above and moves up; 0.49999 is inside the band; 0.05 is AT
    # lower_below and moves down, which from the shortest rung is a refusal.
    for value, expected in ((0.5, SHORT), (0.49999, TINY), (0.05, TINY)):
        sel = treatment()
        d = HarnessDispatcher(LADDER, selector=sel)
        drive(d, sel, [rows(value)])
        assert d.active is expected, value
    sel = treatment()
    d = HarnessDispatcher([PLAIN, TINY, SHORT, LONG], selector=sel)
    drive(d, sel, [rows(0.05)])
    assert d.active is SHORT, "lower_below is inclusive and moves DOWN"


# --------------------------------------------------------------------------- boundaries -----


def test_already_at_the_longest_refuses_and_names_the_ceiling():
    """The honest answer when the feature asks for something the configuration cannot give.

    Not a silent stay and not a wrap: the record says the set has nothing longer, so a run
    whose refusals are all of this kind reports a variant set too small for its own rule
    rather than looking like a rule that stopped firing.
    """
    sel = treatment()
    d = HarnessDispatcher([LONG, PLAIN, SHORT, TINY], selector=sel)
    batch = drive(d, sel, [rows(1.0)])[0]
    assert d.active is LONG
    assert batch.switches == 0
    assert batch.refusals == 1
    assert "nothing longer" in batch.records[0].reason
    assert "long" in batch.records[0].reason


def test_already_at_the_shortest_refuses_and_names_the_floor():
    """The mirror case, which a rule written only for the interesting direction would miss."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batch = drive(d, sel, [rows(0.0)])[0]
    assert d.active is TINY
    assert batch.refusals == 1
    assert "nothing shorter" in batch.records[0].reason


def test_a_single_variant_dispatcher_never_reaches_the_selector():
    """The control arm's shape: the dispatcher refuses first, so the rule takes no decision.

    It matters that ``decisions`` stays 0. If a one-variant arm accumulated decisions that
    could never move, its switch rate would be a denominator with no numerator available, and
    a control matched to it would be matched to an artefact of the configuration.
    """
    sel = treatment()
    d = HarnessDispatcher([PLAIN], selector=sel)
    batch = drive(d, sel, [rows(1.0)])[0]
    assert batch.switches == 0
    assert batch.refusals == 1
    assert "nothing to move to" in batch.records[0].reason
    assert sel.decisions == 0


def test_calling_the_rule_on_a_single_variant_set_refuses_rather_than_returning_current():
    """Called directly, the rule must still decline explicitly -- returning ``current`` would
    make ``apply`` raise a message about the wrong thing."""
    sel = treatment()
    sel.observe(rows(1.0))
    with pytest.raises(HarnessSelectionRefused, match="nothing longer"):
        sel([PLAIN], PLAIN)


def test_an_empty_variant_set_is_a_programmer_error_not_a_refusal():
    """A set that does not exist is a disagreement between caller and dispatcher, not data."""
    sel = treatment()
    sel.observe(rows(1.0))
    with pytest.raises(ValueError, match="non-empty") as exc:
        sel([], PLAIN)
    assert not isinstance(exc.value, HarnessSelectionRefused), (
        "a refusal here would be recorded as a data condition and the run would continue "
        "dispatching under a set the caller and the dispatcher disagree about"
    )


def test_a_current_outside_the_configured_set_is_a_programmer_error():
    """Guessing which side is right would dispatch under a configuration nobody declared."""
    sel = treatment()
    sel.observe(rows(1.0))
    with pytest.raises(ValueError, match="not in the configured set") as exc:
        sel(PAIR, TINY)
    assert not isinstance(exc.value, HarnessSelectionRefused)


def test_variants_at_the_same_step_limit_are_neither_longer_nor_shorter():
    """A set may legitimately hold two variants at one budget; moving between them would be a
    change the feature did not ask for, wearing the label of one it did.

    The dispatcher explicitly allows variants that differ only in ``settings``, so this is a
    reachable configuration and not a hypothetical.
    """
    twin = HarnessVariant("twin", "same budget, other tools", step_limit=40, settings={"t": 1})
    down = treatment()
    dd = HarnessDispatcher([PLAIN, twin, LONG], selector=down)
    batch = drive(dd, down, [rows(0.0)])[0]
    assert dd.active is PLAIN
    assert batch.refusals == 1
    assert "nothing shorter" in batch.records[0].reason
    # And the same in the other direction: the twin is not the nearest LONGER variant either,
    # because it is not a longer one at all. Checked separately because "longer" and "shorter"
    # are two comparisons and an off-by-one in one of them says nothing about the other.
    up = treatment()
    du = HarnessDispatcher([PLAIN, twin, LONG], selector=up)
    drive(du, up, [rows(1.0)])
    assert du.active is LONG


def test_a_set_whose_variants_all_share_one_step_limit_is_refused():
    """The one configuration in which this rule is its own control, caught loudly.

    The dispatcher deliberately ACCEPTS variants that differ only in ``settings``, because that
    is an axis an adapter really varies. This rule moves along the step budget, so over such a
    set it can never move: every proposal would be refused, and the arm would report a
    feature-driven harness while training exactly like the control it is supposed to be
    compared against. That is the failure the whole axis exists to prevent, so it stops the run
    rather than becoming the most common line in the log.
    """
    a = HarnessVariant("a", "tool-heavy", step_limit=40, settings={"tools": "many"})
    b = HarnessVariant("b", "tool-light", step_limit=40, settings={"tools": "few"})
    sel = treatment()
    d = HarnessDispatcher([a, b], selector=sel)
    assert d.can_evolve is True, "the dispatcher allows this set; the rule is what cannot use it"
    sel.observe(rows(1.0))
    with pytest.raises(ValueError, match="can never move") as exc:
        d.consume([A.PROPOSE])
    assert not isinstance(exc.value, HarnessSelectionRefused)


def test_the_nearest_rung_wins_over_a_further_one():
    """"Nearest longer", not "first longer in configured order": configuration order must not
    silently become the rule."""
    sel = treatment()
    d = HarnessDispatcher([SHORT, LONG, PLAIN], selector=sel)
    drive(d, sel, [rows(1.0)])
    assert d.active is PLAIN, "40 is nearer to 15 than 100 is, whatever the configured order"


# ------------------------------------------------------- features that did not arrive -------


def test_a_missing_feature_raises_rather_than_defaulting_to_zero():
    """The most dangerous default available here, which is why it must not exist.

    A substituted 0.0 reads as "nothing truncated" and would propose a SHORTER budget on a
    batch whose observability never arrived -- an intervention in the wrong direction, taken
    on no evidence, and reported as a feature-driven decision.
    """
    sel = treatment()
    with pytest.raises(MissingFeatures, match="truncated_fraction") as exc:
        sel.observe([{"solve_rate": 0.0}])
    assert not isinstance(exc.value, HarnessSelectionRefused)


def test_an_empty_observation_is_refused():
    """A batch with no groups carries no evidence; deciding anyway would report a decision
    that was taken on the previous batch's statistic."""
    sel = treatment()
    with pytest.raises(ValueError, match="no group features"):
        sel.observe([])
    with pytest.raises(ValueError, match="no group features"):
        sel.observe(None)


def test_deciding_before_any_observation_raises():
    """A selection taken before a batch was observed is a decision on evidence that never
    arrived, and a default here would report an arm that never ran."""
    sel = treatment()
    with pytest.raises(ValueError, match="before observe") as exc:
        sel(LADDER, TINY)
    assert not isinstance(exc.value, HarnessSelectionRefused)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_a_feature_outside_its_guaranteed_range_is_refused(bad):
    """``group_features`` guarantees a finite fraction in [0, 1]; a violation means these rows
    came from somewhere else, and the rule must not decide on them."""
    sel = treatment()
    with pytest.raises(ValueError, match="came from somewhere else"):
        sel.observe(rows(bad))


# ------------------------------------------------------------ configuration guards ----------


def test_thresholds_that_cross_are_refused():
    """At or above ``raise_above`` a single value satisfies both branches, and the rule's
    answer would depend on which branch happens to be written first."""
    with pytest.raises(ValueError, match="strictly below"):
        treatment(raise_above=0.4, lower_below=0.4)
    with pytest.raises(ValueError, match="strictly below"):
        treatment(raise_above=0.2, lower_below=0.6)


@pytest.mark.parametrize(
    "kwargs", [{"raise_above": 1.2}, {"raise_above": -0.1}, {"lower_below": 2.0}]
)
def test_thresholds_outside_zero_one_are_refused(kwargs):
    """A threshold on a fraction that no fraction can reach is a branch that never fires."""
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        treatment(**kwargs)


def test_an_empty_feature_name_is_refused():
    """A rule pointed at nothing would raise MissingFeatures on every batch and read as a
    features-never-arrived failure rather than as a misconfiguration."""
    with pytest.raises(ValueError, match="non-empty"):
        treatment(feature="")


def test_custom_thresholds_actually_change_the_decision():
    """The thresholds are parameters, and a parameter that changes no behaviour is
    decoration -- this repo has shipped several and had to go looking for them."""
    quiet = treatment(raise_above=0.9)
    d = HarnessDispatcher(LADDER, selector=quiet)
    drive(d, quiet, [rows(0.6)])
    assert d.active is TINY, "0.6 is below a 0.9 trigger"
    eager = treatment(raise_above=0.5)
    d2 = HarnessDispatcher(LADDER, selector=eager)
    drive(d2, eager, [rows(0.6)])
    assert d2.active is SHORT


# --------------------------------------------------- determinism and set membership ---------


def test_the_treatment_is_deterministic_with_no_seed_at_all():
    """Stronger than reproducible-under-a-seed: there is no seed to get wrong.

    Two instances shown the same observations produce identical histories, so two arms
    differing only in seed cannot differ in harness history.
    """
    batches = realistic_batches()
    histories = []
    for _ in range(2):
        sel = treatment()
        d = HarnessDispatcher(LADDER, selector=sel)
        drive(d, sel, batches)
        histories.append([(r.statistic, r.before, r.after, r.moved) for r in sel.records])
    assert histories[0] == histories[1]
    assert not any(isinstance(v, random.Random) for v in vars(treatment()).values())


def test_the_treatment_never_proposes_outside_the_configured_set():
    """Whatever the feature stream, the active variant is one that was configured."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    names = {v.name for v in LADDER}
    rng = random.Random(3)
    for _ in range(120):
        sel.observe(rows(rng.random(), rng.random()))
        d.consume([A.PROPOSE, A.PROPOSE])
        assert d.active.name in names
    assert {r.after for r in sel.records if r.moved} <= names


# ------------------------------------------------------------------- the control ------------


def test_the_control_matches_the_treatments_switch_count_over_a_realistic_sequence():
    """The test the experiment stands on.

    The treatment is run over a realistic truncation stream; its REALISED moves and decisions
    -- measured, never nominal, for the reason ``selfevo.routing.proportions`` documents at
    length -- become the control's deck. Over the same number of decisions the control makes
    exactly as many switches, having read nothing. Any gap between the two arms in a real run
    is then attributable to WHERE the switches went, which is the claim under test.
    """
    batches = realistic_batches()
    t = treatment()
    dt = HarnessDispatcher(LADDER, selector=t)
    t_batches = drive(dt, t, batches)

    assert t.moves > 0, "a treatment that never moved would make this test vacuous"
    assert t.moves < t.decisions, "and one that always moved would too"
    assert sum(b.switches for b in t_batches) == t.moves

    c = RateMatchedControlSelector.from_treatment(t, seed=0)
    dc = HarnessDispatcher(LADDER, selector=c)
    c_batches = drive(dc, c, batches)

    assert c.decisions == t.decisions
    assert c.moves == t.moves
    assert sum(b.switches for b in c_batches) == sum(b.switches for b in t_batches)
    assert c.switch_rate == t.switch_rate


def test_the_control_matches_exactly_at_every_multiple_of_its_deck():
    """Matched by CONSTRUCTION, not in expectation: a Bernoulli draw at the same probability
    would have a count with standard deviation ``sqrt(n p (1-p))``, and a control that missed
    the rate it was built to hold is not a rate-matched control."""
    c = RateMatchedControlSelector(moves=3, decisions=8, seed=7)
    d = HarnessDispatcher(LADDER, selector=c)
    drive(d, c, [None] * 8)
    assert c.moves == 3
    drive(d, c, [None] * 8)
    assert c.moves == 6
    drive(d, c, [None] * 8)
    assert c.moves == 9


def test_the_control_mismatch_on_a_partial_deck_stays_inside_the_documented_bound():
    """The residual mismatch is real and is stated rather than hidden.

    After ``n = q * decisions + r`` calls the realised rate differs from the target by at most
    ``r / n``. Asserted at a length that is deliberately not a multiple of the deck, because a
    bound that is only checked where it is tight is not checked.
    """
    moves, deck, extra = 3, 8, 5
    c = RateMatchedControlSelector(moves=moves, decisions=deck, seed=7)
    d = HarnessDispatcher(LADDER, selector=c)
    n = deck + extra
    drive(d, c, [None] * n)
    assert abs(c.switch_rate - c.target_rate) <= extra / n
    assert moves <= c.moves <= moves + min(extra, moves)


def test_the_control_reads_no_feature_at_all():
    """The defining property of the arm: two runs whose feature streams could not be more
    different produce the same schedule, because the schedule is a function of the seed.

    A control whose timing depended on the feature would have conceded the hypothesis before
    the run started -- it would be intervening where the feature points, which is the
    treatment.
    """
    histories = []
    for stream in (
        [rows(1.0, 1.0)] * 24,
        [rows(0.0, 0.0)] * 24,
        [rows(0.3, 0.7)] * 24,
        [None] * 24,
    ):
        c = RateMatchedControlSelector(moves=9, decisions=24, seed=5)
        d = HarnessDispatcher(LADDER, selector=c)
        drive(d, c, stream)
        histories.append([(r.moved, r.before, r.after) for r in c.records])
    assert histories.count(histories[0]) == len(histories)


def test_the_control_is_reproducible_under_a_fixed_seed():
    """An unreproducible arm is not evidence."""
    runs = []
    for _ in range(2):
        c = RateMatchedControlSelector(moves=9, decisions=24, seed=5)
        d = HarnessDispatcher(LADDER, selector=c)
        drive(d, c, [None] * 24)
        runs.append([(r.moved, r.after) for r in c.records])
    assert runs[0] == runs[1]


def test_a_different_seed_gives_a_different_schedule():
    """The seed has to be load-bearing, or the control is one fixed schedule wearing a
    parameter -- and a single schedule cannot be replicated across control runs."""
    schedules = []
    for seed in (0, 1, 2):
        c = RateMatchedControlSelector(moves=9, decisions=24, seed=seed)
        d = HarnessDispatcher(LADDER, selector=c)
        drive(d, c, [None] * 24)
        schedules.append(tuple(c.outcomes()))
        assert c.moves == 9, "every seed realises the same rate"
    assert len(set(schedules)) == 3


def test_consecutive_decks_are_not_the_same_schedule_replayed():
    """Each exhausted deck is reshuffled, so the schedule is not periodic with the deck.

    A schedule that repeated exactly every ``decisions`` calls would hold the rate while
    aligning its interventions to a fixed phase, which is a way for a "feature-independent"
    control to correlate with anything in the run that has the same period.
    """
    c = RateMatchedControlSelector(moves=9, decisions=24, seed=5)
    d = HarnessDispatcher(LADDER, selector=c)
    drive(d, c, [None] * 48)
    first, second = c.outcomes()[:24], c.outcomes()[24:]
    assert sum(first) == sum(second) == 9
    assert first != second


def test_the_control_never_proposes_the_variant_that_is_already_active():
    """A move that resolves to the current scaffold is the no-op the dispatcher raises on, and
    it would silently cost the control one matched switch."""
    c = RateMatchedControlSelector(moves=20, decisions=24, seed=3)
    d = HarnessDispatcher(LADDER, selector=c)
    drive(d, c, [None] * 24)
    assert all(r.after != r.before for r in c.records if r.moved)
    assert c.moves == 20


def test_the_control_never_proposes_outside_the_configured_set():
    """Same invariant as the treatment, checked separately: the control chooses its
    destination by drawing, and a draw is exactly where an off-by-one lands out of range."""
    names = {v.name for v in LADDER}
    c = RateMatchedControlSelector(moves=16, decisions=24, seed=9)
    d = HarnessDispatcher(LADDER, selector=c)
    for _ in range(24):
        c.observe(None)
        d.consume([A.PROPOSE])
        assert d.active.name in names
    assert {r.after for r in c.records if r.moved} <= names


def test_the_control_spreads_its_destinations_rather_than_pinning_one():
    """The destination is drawn uniformly among the other members.

    A control that always moved to the same variant would be a second targeted rule -- a worse
    one -- rather than the absence of targeting, and over a four-rung ladder it would visit a
    third of the set the treatment visits.
    """
    c = RateMatchedControlSelector(moves=20, decisions=24, seed=1)
    d = HarnessDispatcher(LADDER, selector=c)
    drive(d, c, [None] * 24)
    assert len({r.after for r in c.records if r.moved}) >= 3


def test_from_treatment_reads_the_measured_rate_off_the_arm_it_matches():
    """Configuring a control with the probability the treatment was SUPPOSED to realise is the
    failure ``MatchedPermutationControl`` was written to avoid; this reads what it DID."""
    t = treatment()
    d = HarnessDispatcher(LADDER, selector=t)
    drive(d, t, realistic_batches())
    c = RateMatchedControlSelector.from_treatment(t, seed=2)
    assert (c.target_moves, c.block) == (t.moves, t.decisions)
    assert c.target_rate == t.switch_rate


def test_a_control_matched_to_a_treatment_that_never_moved_never_moves():
    """Matching is the point, including when the matched rate is zero: a control that moved
    where the treatment did not would be the treatment's opposite, not its null."""
    t = treatment()
    d = HarnessDispatcher(LADDER, selector=t)
    drive(d, t, [rows(0.3)] * 12)
    assert t.moves == 0
    c = RateMatchedControlSelector.from_treatment(t, seed=4)
    dc = HarnessDispatcher(LADDER, selector=c)
    batches = drive(dc, c, [None] * 12)
    assert c.moves == 0
    assert sum(b.switches for b in batches) == 0
    assert dc.active is TINY


def test_a_control_with_no_decisions_to_match_is_refused():
    """An empty deck would be a no-op arm wearing a control's name -- the same guard, for the
    same reason, as ``MatchedPermutationControl``'s empty-decisions check."""
    with pytest.raises(ValueError, match="decisions must be positive"):
        RateMatchedControlSelector(moves=0, decisions=0)
    t = treatment()
    with pytest.raises(ValueError, match="decisions must be positive"):
        RateMatchedControlSelector.from_treatment(t)


@pytest.mark.parametrize("moves", [-1, 9])
def test_a_control_rate_outside_zero_one_is_refused(moves):
    """A switch count above the decision count is not a rate this control could realise."""
    with pytest.raises(ValueError, match="moves must be in"):
        RateMatchedControlSelector(moves=moves, decisions=8)


# ------------------------------------------- one observation is one decision -----------------


def test_one_observation_is_one_decision_however_many_proposals_arrive():
    """The denominator of the matched rate, pinned.

    ``consume`` stops at the first proposal that MOVES, so a batch whose rule declines calls
    the selector once per proposing group while a batch whose rule moves calls it once.
    Counting raw calls would make the denominator a function of the outcome, and the two arms'
    rates would not be comparable even when they behaved identically.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batch = drive(d, sel, [rows(0.3)], proposals=5)[0]
    assert sel.decisions == 1
    assert sel.repeat_calls == 4
    assert batch.refusals == 5, "every proposal is still recorded as refused"
    assert batch.switches == 0


def test_a_moving_batch_and_a_declining_batch_contribute_one_decision_each():
    """The pairing the previous test protects, stated as the equality it implies."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    drive(d, sel, [rows(1.0), rows(0.3)], proposals=4)
    assert sel.decisions == 2
    assert sel.moves == 1


def test_the_control_spends_one_deck_token_per_observation_not_per_proposal():
    """Otherwise the control would burn its deck faster than the treatment took decisions and
    the rates would diverge for a reason that has nothing to do with either rule."""
    c = RateMatchedControlSelector(moves=4, decisions=8, seed=6)
    d = HarnessDispatcher(LADDER, selector=c)
    drive(d, c, [None] * 8, proposals=6)
    assert c.decisions == 8
    assert c.moves == 4


def test_a_missing_observe_is_visible_as_repeat_calls_rather_than_a_silent_freeze():
    """The failure mode of an observe-then-decide seam: a caller that forgets to observe.

    The harness then cannot move, and the honest signal for that is a repeat count that climbs
    while the decision count does not -- an arm frozen for a reason the metrics name.
    """
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    sel.observe(rows(1.0))
    d.consume([A.PROPOSE])
    assert d.active is SHORT
    batch = d.consume([A.PROPOSE])  # no observe(): still the same epoch
    assert d.active is SHORT
    assert batch.refusals == 1
    assert "already been taken" in batch.records[0].reason
    assert sel.decisions == 1
    assert sel.repeat_calls == 1


# ---------------------------------------------------------------------- metrics --------------


def test_both_selectors_emit_the_same_metric_key_set():
    """Two arms that emit different keys cannot be put on one panel, and this axis is read as
    the difference between exactly these two arms."""
    t = treatment()
    c = RateMatchedControlSelector(moves=1, decisions=2)
    assert set(t.as_metrics()) == set(c.as_metrics()) == set(SELECTOR_METRIC_KEYS)


def test_the_selector_key_set_does_not_depend_on_what_happened():
    """A key that appears only on eventful steps is a key no panel can plot against."""
    busy = treatment()
    d = HarnessDispatcher(LADDER, selector=busy)
    drive(d, busy, [rows(1.0), rows(0.3)])
    assert set(busy.as_metrics()) == set(treatment().as_metrics())


def test_every_selector_metric_is_a_float():
    """``stats_tracker.scalar`` takes numbers; a stray bool or int would fail in a live run."""
    t = treatment()
    d = HarnessDispatcher(LADDER, selector=t)
    drive(d, t, [rows(1.0)])
    assert all(type(v) is float for v in t.as_metrics().values()), t.as_metrics()


def test_selector_metric_keys_live_under_the_route_namespace():
    """The actor's convention, so the selector lands on the same panel as the rest of routing."""
    assert all(k.startswith("route/") for k in SELECTOR_METRIC_KEYS)


def test_an_undefined_switch_rate_is_nan_rather_than_zero():
    """An arm that was never asked and an arm that always declined are different runs, and a
    0.0 in this slot merges them into one number that reads as "the rule does nothing"."""
    t = treatment()
    assert math.isnan(t.switch_rate)
    assert math.isnan(t.as_metrics()["route/harness_sel_rate"])
    assert t.as_metrics()["route/harness_sel_decisions"] == 0.0


def test_refusals_are_counted_by_kind():
    """"The rule declined" and "the set could not oblige" are different diagnoses: the first is
    about the data, the second is about a variant set too small for the rule it was given."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    drive(d, sel, [rows(0.3), rows(0.0)])
    m = sel.as_metrics()
    assert m["route/harness_sel_refused_no_move_wanted"] == 1.0
    assert m["route/harness_sel_refused_no_variant"] == 1.0
    assert m["route/harness_sel_decisions"] == 2.0
    assert m["route/harness_sel_moves"] == 0.0


def test_the_dispatcher_counts_a_selector_refusal_as_a_refusal_not_a_switch():
    """The number the axis is read off must not be flattered by decisions that declined."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    m = drive(d, sel, [rows(0.3)], proposals=3)[0].as_metrics()
    assert m["route/harness_propose"] == 3.0
    assert m["route/harness_switches"] == 0.0
    assert m["route/harness_refused"] == 3.0
    assert m["route/harness_active_tiny"] == 1.0


def test_an_ordinary_batch_reports_no_refusals():
    """The counter must be able to be zero, or it is a constant rather than a measurement."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    m = drive(d, sel, [rows(1.0)], proposals=3)[0].as_metrics()
    assert m["route/harness_switches"] == 1.0
    assert m["route/harness_refused"] == 0.0


# ------------------------------------------------------- the seam is narrow -------------------


def test_the_refusal_seam_does_not_swallow_other_selector_errors():
    """``apply`` catches the refusal BY TYPE. A blanket ``except ValueError`` here would turn
    every programmer error the dispatcher guards against -- an unconfigured variant, a set the
    caller disagrees about -- into a quiet refusal, which is the exact failure this axis was
    built to make impossible."""

    def boom(variants, current):
        raise ValueError("features never arrived")

    with pytest.raises(ValueError, match="features never arrived"):
        HarnessDispatcher(PAIR, selector=boom).apply(A.PROPOSE)

    foreign = HarnessVariant("foreign", "never configured", step_limit=7)
    with pytest.raises(ValueError, match="not in the configured set"):
        HarnessDispatcher(PAIR, selector=lambda v, c: foreign).apply(A.PROPOSE)


def test_a_rule_that_refuses_with_an_unknown_category_is_refused_loudly():
    """A refusal counted under no category would vanish from the metrics.

    ``_decide`` is a subclass hook, so this is the defect a future rule introduces by mistyping
    its own category name -- and the failure it would otherwise produce is a refusal that is
    recorded, raised, and counted nowhere, which is precisely the silent-no-op shape this axis
    exists to prevent. Caught where the counter is incremented, not three frames away.
    """

    class MistypedCategory(TruncationStepLimitSelector):
        """A rule whose refusal category is not one of the two that are counted."""

        def _decide(self, variants, current):
            """Refuse with a category name nothing increments."""
            return (None, 0.0, "declined", "typo_category")

    sel = MistypedCategory()
    sel.observe(rows(0.3))
    with pytest.raises(ValueError, match="unknown category"):
        sel(LADDER, TINY)


def test_a_refusal_is_a_value_error_so_existing_callers_keep_working():
    """Adding a new exception type to a seam must not break a caller that already handled the
    old one."""
    assert issubclass(HarnessSelectionRefused, ValueError)


def test_a_refused_batch_leaves_the_harness_exactly_where_it_was():
    """The strongest form of the negative: after twenty declining batches the active variant is
    the one the run started on, and the metrics say why."""
    sel = treatment()
    d = HarnessDispatcher(LADDER, selector=sel)
    batches = drive(d, sel, [rows(0.2, 0.4)] * 20, proposals=2)
    assert d.active is TINY
    assert sum(b.switches for b in batches) == 0
    assert sum(b.refusals for b in batches) == 40
