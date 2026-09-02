"""The harness selectors, and the two ways a selector arm can be vacuous.

Two claims are under test and they are not the same claim.

The first is mechanical: the selectors implement the contract their call site relies on --
one observation opens one epoch, a second inside it is refused and COUNTED rather than
silently overwriting the first, a refusal never reaches ``HarnessDispatcher.apply`` as a
PROPOSE, and the ladder is ordered by budget rather than by config order.

The second is the one that matters for the experiment: a variant set whose members differ
only in a field nothing reads produces two arms that are byte-identical. ``gen96``/``gen160``
/``gen256`` differ in ``settings['max_new_tokens']``; ``plain``/``long``/``short`` differ
only in ``step_limit``, which on a single-turn run nothing reads. The tests below assert
that the FIRST set has budgets and the second does not, so a future edit that pointed the
selector at the step-limit set would fail here rather than in a run whose two arms came back
identical.
"""

import pytest

from selfevo.harness.base import VARIANTS, HarnessVariant
from selfevo.harness.dispatch import HarnessDispatcher, build_dispatcher, round_robin
from selfevo.harness.selectors import (
    GENERATION_BUDGET_KEY,
    RateMatchedControlSelector,
    SELECTORS,
    TruncationStepLimitSelector,
    budget_of,
    build_selector,
    ladder,
)
from selfevo.routing.base import HarnessAction

SHORT = VARIANTS["gen96"]
MID = VARIANTS["gen160"]
LONG = VARIANTS["gen256"]
LADDER = [SHORT, MID, LONG]


# -- the mapping: which field a variant changes ------------------------------------------


def test_budget_variants_carry_a_budget():
    """The registered ladder names a generation budget, in ascending order."""
    assert [budget_of(v) for v in (SHORT, MID, LONG)] == [96, 160, 256]


def test_step_limit_variants_have_no_budget():
    """The original variants are refused: they differ only in a field nothing reads here.

    This is the guard against the silent no-op. ``plain``/``long``/``short`` are legitimate
    variants for an AGENTIC harness, where ``step_limit`` is real; pointing a single-turn
    generation-budget selector at them would dispatch between scaffolds that produce
    identical rollouts, and the run would report switches the whole time.
    """
    for name in ("plain", "long", "short"):
        with pytest.raises(ValueError, match=GENERATION_BUDGET_KEY):
            budget_of(VARIANTS[name])


def test_budget_must_be_a_positive_int():
    """A float or a non-positive budget is refused rather than coerced."""
    for bad in (0, -1, 160.0, True):
        with pytest.raises(ValueError):
            budget_of(HarnessVariant("x", "d", settings={GENERATION_BUDGET_KEY: bad}))


def test_ladder_sorts_by_budget_not_config_order():
    """"One rung longer" is defined against the budget, not against the config's order."""
    assert [v.name for v in ladder([LONG, SHORT, MID])] == ["gen96", "gen160", "gen256"]


def test_ladder_refuses_one_rung_and_duplicate_budgets():
    """A ladder that cannot move, or whose rungs are the same length, is refused."""
    with pytest.raises(ValueError, match="at least 2"):
        ladder([MID])
    twin = HarnessVariant("twin", "same budget", settings={GENERATION_BUDGET_KEY: 160})
    with pytest.raises(ValueError, match="share generation budget"):
        ladder([MID, twin])


# -- the treatment rule ------------------------------------------------------------------


def _observe(sel, epoch, frac, current=MID, variants=LADDER):
    return sel.observe(epoch, frac, variants=variants, current=current)


def test_treatment_moves_up_down_and_refuses_between():
    """The three branches, at the thresholds and inside the dead band."""
    sel = TruncationStepLimitSelector()
    assert _observe(sel, 0, 0.9).move == 1
    assert _observe(sel, 1, 0.5).move == 1, "the upper threshold is inclusive"
    assert _observe(sel, 2, 0.3).move == 0
    assert _observe(sel, 3, 0.05).move == -1, "the lower threshold is inclusive"
    assert _observe(sel, 4, 0.0).move == -1
    assert sel.decisions == 5 and sel.moves == 4 and sel.refusals == 1
    assert sel.up_moves == 2 and sel.down_moves == 2


def test_treatment_is_blocked_at_the_ends_and_says_so():
    """A move the ladder cannot take becomes a refusal, distinguishable from a dead-band one."""
    sel = TruncationStepLimitSelector()
    top = _observe(sel, 0, 0.9, current=LONG)
    assert top.move == 0 and top.blocked and "longest rung" in top.reason
    bottom = _observe(sel, 1, 0.0, current=SHORT)
    assert bottom.move == 0 and bottom.blocked and "shortest rung" in bottom.reason
    assert sel.blocked == 2 and sel.refusals == 2 and sel.moves == 0


def test_crossed_thresholds_are_refused_at_construction():
    """Overlapping branches would make the outcome depend on test order, not on the batch."""
    with pytest.raises(ValueError):
        TruncationStepLimitSelector(up_threshold=0.2, down_threshold=0.5)
    with pytest.raises(ValueError):
        TruncationStepLimitSelector(up_threshold=0.5, down_threshold=0.5)


def test_a_nan_observation_is_refused_not_treated_as_a_refusal():
    """NaN compares False against every threshold and would be logged as a considered stay."""
    sel = TruncationStepLimitSelector()
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            _observe(sel, 0, bad)


# -- the epoch contract ------------------------------------------------------------------


def test_second_observation_in_an_epoch_is_refused_and_counted():
    """The decision does not change, and the run can see that the call site misbehaved."""
    sel = TruncationStepLimitSelector()
    first = _observe(sel, 7, 0.9)
    again = _observe(sel, 7, 0.0)
    assert again is first, "the standing decision is returned unchanged"
    assert sel.repeat_observations == 1
    assert sel.decisions == 1, "a refused observation is not a second decision"


def test_a_stale_epoch_is_refused_too():
    """Epochs move forward; an out-of-order call cannot re-open one."""
    sel = TruncationStepLimitSelector()
    _observe(sel, 5, 0.9)
    _observe(sel, 4, 0.0)
    assert sel.repeat_observations == 1 and sel.decisions == 1


def test_choosing_without_observing_raises():
    """``__call__`` reports a decision; it must never invent one."""
    sel = TruncationStepLimitSelector()
    with pytest.raises(RuntimeError, match="before any observation"):
        sel(LADDER, MID)


def test_choosing_after_a_refusal_raises():
    """A caller that proposes on a refusal is a bug, not a coin flip."""
    sel = TruncationStepLimitSelector()
    _observe(sel, 0, 0.3)
    with pytest.raises(RuntimeError, match="refused to move"):
        sel(LADDER, MID)


def test_call_returns_the_neighbouring_rung_in_the_decided_direction():
    """The dispatcher seam returns a variant, and it is the one the observation chose."""
    sel = TruncationStepLimitSelector()
    _observe(sel, 0, 0.9)
    assert sel(LADDER, MID) is LONG
    _observe(sel, 1, 0.0)
    assert sel(LADDER, MID) is SHORT


# -- the dispatcher, driven by a selector ------------------------------------------------


def test_dispatcher_moves_the_active_variant_the_way_the_batch_says():
    """End to end through the production entry point: observe, propose, consume, active."""
    d = build_dispatcher(["gen96", "gen160", "gen256"], selector="truncation_step_limit")
    assert d.active is SHORT and d.can_evolve
    d.selector.observe(0, 0.9, variants=d.variants, current=d.active)
    batch = d.consume([HarnessAction.PROPOSE])
    assert batch.switches == 1 and d.active is MID
    assert batch.as_metrics()["route/harness_active_gen160"] == 1.0


def test_a_refusal_consumed_as_none_leaves_the_budget_alone():
    """The call site expresses a refusal as NONE, and the key set is unchanged."""
    d = build_dispatcher(["gen96", "gen160", "gen256"], selector="truncation_step_limit")
    before = d.active
    batch = d.consume([HarnessAction.NONE])
    assert batch.switches == 0 and d.active is before
    assert set(batch.as_metrics()) == set(
        d.consume([HarnessAction.NONE]).as_metrics()
    ), "every step emits the same keys"


def test_build_dispatcher_defaults_to_round_robin():
    """Omitting the selector leaves the shipped behaviour exactly as it was."""
    d = build_dispatcher(["plain", "long"])
    assert d.selector is round_robin


def test_build_dispatcher_refuses_a_selector_over_a_set_it_cannot_walk():
    """A budgetless or single-rung set is refused before any GPU is touched."""
    with pytest.raises(ValueError, match=GENERATION_BUDGET_KEY):
        build_dispatcher(["plain", "long"], selector="truncation_step_limit")
    with pytest.raises(ValueError, match="at least 2"):
        build_dispatcher(["gen160"], selector="truncation_step_limit")


def test_unknown_selector_is_refused():
    """An unregistered name must not fall back to the default rule."""
    with pytest.raises(ValueError, match="unknown harness selector"):
        build_selector("no_such_rule")
    assert set(SELECTORS) == {"truncation_step_limit", "rate_matched_control"}


# -- the matched control -----------------------------------------------------------------


def test_control_ignores_the_batch_entirely():
    """Two control runs seeing opposite batches make identical decisions from one seed."""
    a = RateMatchedControlSelector(move_rate=0.5, up_share=0.5, seed=11)
    b = RateMatchedControlSelector(move_rate=0.5, up_share=0.5, seed=11)
    moves_a = [_observe(a, i, 0.99).move for i in range(50)]
    moves_b = [_observe(b, i, 0.01).move for i in range(50)]
    assert moves_a == moves_b


def test_control_realises_a_rate_near_its_target():
    """The match is stochastic; the residual is what a report must quote, not hide."""
    sel = RateMatchedControlSelector(move_rate=0.4, up_share=0.5, seed=3)
    for i in range(2000):
        _observe(sel, i, 0.3)
    assert abs(sel.moves / sel.decisions - 0.4) < 0.03


def test_control_never_refuses_a_drawn_move_at_a_ladder_end():
    """A blocked draw is flipped inward, so the move COUNT -- the matched quantity -- holds."""
    sel = RateMatchedControlSelector(move_rate=1.0, up_share=1.0, seed=5)
    d = _observe(sel, 0, 0.3, current=LONG)
    assert d.move == -1 and not d.blocked and "flipped inward" in d.reason
    assert sel.moves == 1 and sel.refusals == 0 and sel.flips == 1
    assert sel.as_metrics()["route/harness_move_rate"] == 1.0


def test_control_rate_of_one_moves_every_decision():
    """The counters describe what the arm DID, including after a flip."""
    sel = RateMatchedControlSelector(move_rate=1.0, up_share=0.5, seed=9)
    current = MID
    for i in range(30):
        d = _observe(sel, i, 0.3, current=current)
        assert d.move != 0
        current = sel(LADDER, current)
    assert sel.moves == 30 and sel.decisions == 30 and sel.refusals == 0


def test_control_probabilities_are_validated():
    """A rate outside [0, 1] is a configuration error, not a clamp."""
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            RateMatchedControlSelector(move_rate=bad)
        with pytest.raises(ValueError):
            RateMatchedControlSelector(move_rate=0.5, up_share=bad)


def test_selector_args_reach_the_factory():
    """A control is configured from measured numbers, through the same config seam."""
    sel = build_selector(
        "rate_matched_control", {"move_rate": 0.25, "up_share": 0.75, "seed": 4.0}
    )
    assert isinstance(sel, RateMatchedControlSelector)
    assert sel.move_rate == 0.25 and sel.up_share == 0.75 and sel.seed == 4


def test_bad_selector_args_are_refused_with_the_name():
    """An argument the factory does not take must not be silently dropped."""
    with pytest.raises(ValueError, match="do not fit selector"):
        build_selector("truncation_step_limit", {"move_rate": 0.5})
