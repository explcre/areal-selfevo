"""Tests for the contextual (LinUCB) router.

The point of this router is that it decides from FEATURES. A policy that sees only what a
hand-written threshold sees cannot beat that threshold by more than noise, so the failure
worth testing for is not "it crashes" -- it is the router quietly degrading to something
context-free or arbitrary while still reporting itself as contextual. Four ways it could do
that are each swept:

* MISSING FEATURES -- ``require_features=True`` must raise, never substitute a zero, and it
  must raise on every path including the one that short-circuits before the argmax. Zeros
  in place of features IS the context-free bandit.
* POISONING -- one non-finite entry makes an arm's ``theta`` NaN forever, and a NaN never
  wins an argmax, so a poisoned arm silently stops being selectable and ``updates`` keeps
  rising. Both the feature side and the outcome side are checked.
* MISCREDIT -- an outcome is credited only when the remembered decision names the same
  mode. Crediting the wrong arm corrupts the model in a way no metric shows.
* TIE-BREAK -- a fresh router's arms are exactly tied on every context, so the tie-break
  decides every early decision in a run. It has to be the package-wide rule (first mode by
  name, the same one ``RoutingDecision.argmax`` and ``BanditRouter`` use) rather than
  whichever mode the caller happened to list first.

:func:`test_the_router_learns_a_rule_a_solve_rate_threshold_cannot_express` is the one that
justifies the module: it runs the real route/observe loop against a synthetic environment
whose optimal mode depends on a feature ``solve_rate`` cannot see, and checks the
context-free ``BanditRouter`` against the same stream as the bound to beat.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from selfevo.observability import FEATURE_NAMES, group_features
from selfevo.routing.base import (
    Granularity,
    Router,
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)
from selfevo.routing.contextual import ContextualBanditRouter, MissingFeatures
from selfevo.routing.feedback import (
    BanditRouter,
    ConfoundedUpdate,
    DecisionOutcome,
    LearningRouter,
)

GRANS = tuple(Granularity)
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_MODES = (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP)


def feats(**overrides):
    """A complete feature mapping, so a test that omits one omits it deliberately."""
    f = {n: 0.1 for n in FEATURE_NAMES}
    f.update(overrides)
    return f


def ctx(p=0.5, *, teacher=False, uid=None, gran=Granularity.SAMPLE, g=4, extra=None, **kw):
    """A routing context, built by keyword so no field order can silently rebind."""
    return RoutingContext(
        solve_rate=p, group_size=g, granularity=gran, has_teacher=teacher,
        unit_id=uid, extra=feats(**kw) if extra is None else extra,
    )


def mode_of(d):
    """The single mode a hard routing decision selected."""
    assert len(d.weights) == 1, f"expected a one-hot decision, got {d.weights}"
    return next(iter(d.weights))


def outcome(mode, value, unit_id, *, batch_id="b0", cost=1.0):
    """A DecisionOutcome for a unit this router routed."""
    return DecisionOutcome(mode=mode, value=value, batch_id=batch_id, unit_id=unit_id,
                           cost=cost)


# ------------------------------------------------------------------- the protocols ----


def test_it_is_both_a_router_and_a_learning_router():
    """Swapping a fixed criterion for this one must need no change anywhere else."""
    r = ContextualBanditRouter()
    assert isinstance(r, Router) and isinstance(r, LearningRouter)


def test_a_decision_is_one_hot_and_names_a_registered_mode():
    r = ContextualBanditRouter()
    for p, teacher, gran in itertools.product(RATES, (False, True), GRANS):
        d = r.route(ctx(p, teacher=teacher, gran=gran))
        assert mode_of(d) in known_modes()
        assert sum(d.weights.values()) == 1.0
        assert d.reason.startswith("contextual:")


# ---------------------------------------------------------------- missing features ----


@pytest.mark.parametrize("dropped", FEATURE_NAMES)
def test_a_missing_feature_raises_and_the_message_names_it(dropped):
    """Substituting a zero would turn this into the context-free bandit while it still
    reported itself as contextual -- the failure this package keeps finding in other guises."""
    partial = {k: v for k, v in feats().items() if k != dropped}
    with pytest.raises(MissingFeatures, match=dropped):
        ContextualBanditRouter().route(ctx(extra=partial, teacher=True))


def test_an_empty_extra_raises_rather_than_routing_on_zeros():
    with pytest.raises(MissingFeatures):
        ContextualBanditRouter().route(ctx(extra={}, teacher=True))


def test_missing_features_is_a_key_error():
    """Pinned because callers catch KeyError around ``extra`` lookups; if this stopped
    being one, a caller's handler would stop firing without any signature changing."""
    assert issubclass(MissingFeatures, KeyError)


def test_the_requirement_cannot_be_bypassed_by_a_unit_with_no_target():
    """``route`` short-circuits when no mode is selectable. Checking the features only
    after that would let a run whose features never arrived hide behind its target-free
    units and report SKIP -- a context-free decision -- for every one of them."""
    r = ContextualBanditRouter(modes=(TrainingMode.SFT, TrainingMode.DISTILL))
    for p, gran in itertools.product((0.0,), GRANS):
        with pytest.raises(MissingFeatures):
            r.route(ctx(p, teacher=False, gran=gran, extra={}))
    # ...and with the features present the same context does short-circuit.
    d = r.route(ctx(0.0, teacher=False, gran=Granularity.TOKEN))
    assert mode_of(d) == TrainingMode.SKIP and "no mode has a target" in d.reason


@pytest.mark.parametrize("dropped", FEATURE_NAMES)
def test_require_features_false_substitutes_a_zero_and_nothing_else(dropped):
    """The only difference between the two flags: a missing feature becomes 0.0. Compared
    against the SAME context with an explicit zero, so a flag that also changed the
    exploration term or the intercept would show up here."""
    partial = {k: v for k, v in feats().items() if k != dropped}
    explicit = feats(**{dropped: 0.0})
    lax = ContextualBanditRouter(require_features=False)
    strict = ContextualBanditRouter()
    assert np.array_equal(lax._vector(ctx(extra=partial)), strict._vector(ctx(extra=explicit)))
    a = lax.route(ctx(teacher=True, extra=partial))
    b = strict.route(ctx(teacher=True, extra=explicit))
    assert (a.weights, a.reason) == (b.weights, b.reason)


def test_with_every_feature_present_the_two_flags_are_indistinguishable():
    """Swept over a whole stream, decisions and learning included: the flag must not be a
    second behaviour switch."""
    lax = ContextualBanditRouter(require_features=False)
    strict = ContextualBanditRouter(require_features=True)
    for i, p in enumerate(RATES * 4):
        c = ctx(p, teacher=True, uid=f"u{i}", solve_rate=p, mean_logprob=-0.1 * i)
        a, b = lax.route(c), strict.route(c)
        assert (a.weights, a.reason) == (b.weights, b.reason), i
        lax.observe({c.unit_id: outcome(mode_of(a), 1.0, c.unit_id, batch_id=f"b{i}")})
        strict.observe({c.unit_id: outcome(mode_of(b), 1.0, c.unit_id, batch_id=f"b{i}")})
    assert lax.updates == strict.updates > 0


# --------------------------------------------------------------------- poisoning ------


@pytest.mark.parametrize("name", FEATURE_NAMES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_feature_is_zeroed_before_it_reaches_the_model(name, bad):
    """One NaN in ``x`` makes ``A`` and ``b`` NaN forever, and no metric would show it."""
    r = ContextualBanditRouter()
    x = r._vector(ctx(extra=feats(**{name: bad})))
    assert np.isfinite(x).all(), x
    assert x[FEATURE_NAMES.index(name)] == 0.0


def test_a_non_finite_feature_does_not_stop_the_router_deciding():
    r = ContextualBanditRouter()
    for name, bad in itertools.product(FEATURE_NAMES, (float("nan"), float("inf"))):
        d = r.route(ctx(teacher=True, uid=None, extra=feats(**{name: bad})))
        assert mode_of(d) in DEFAULT_MODES


def test_an_outcome_whose_update_is_not_finite_is_rejected_and_counted():
    """``DecisionOutcome`` rejects a non-finite ``value``, but ``value / cost`` overflows.
    Without the guard the arm's theta is NaN forever; a NaN never wins an argmax, so the
    arm becomes unselectable and ``updates`` still counts the update as applied."""
    r = ContextualBanditRouter(pending_cap=8)
    c = ctx(teacher=True, uid="u0")
    m = mode_of(r.route(c))
    r.observe({"u0": outcome(m, 1e300, "u0", cost=1e-300)})
    assert (r.updates, r.rejected) == (0, 1)
    assert np.isfinite(r._A[m]).all() and np.isfinite(r._b[m]).all()
    assert all(math.isfinite(v) for v in r.weights(m).values())
    # the arm is still selectable, which is the property the guard is really protecting
    fresh = ContextualBanditRouter()
    assert mode_of(r.route(ctx(teacher=True))) == mode_of(fresh.route(ctx(teacher=True)))


def test_a_finite_but_enormous_feature_cannot_overflow_the_model_either():
    """``x`` passes the finiteness check and ``np.outer(x, x)`` still overflows.

    RESIDUAL, recorded here rather than fixed: the UCB score inside ``route`` overflows on
    the same input (``x @ A_inv @ x`` is ~1e400), so the DECISION for such a context is
    arbitrary-but-deterministic even though the MODEL stays clean. Features are checked for
    finiteness, not for magnitude; every real feature is a rate in [0, 1] or a token count,
    so nothing reaches 1e154. ``errstate`` here keeps that overflow from being reported as a
    test warning, not from happening.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        r = ContextualBanditRouter(pending_cap=8)
        c = ctx(teacher=True, uid="u0", mean_response_len=1e200)
        m = mode_of(r.route(c))
        r.observe({"u0": outcome(m, 1.0, "u0")})
    assert (r.updates, r.rejected) == (0, 1)
    assert np.isfinite(r._A[m]).all()


def test_a_normal_outcome_is_not_rejected():
    """A guard that rejected everything would pass both tests above."""
    r = ContextualBanditRouter(pending_cap=8)
    c = ctx(teacher=True, uid="u0")
    m = mode_of(r.route(c))
    r.observe({"u0": outcome(m, 1.0, "u0")})
    assert (r.updates, r.rejected) == (1, 0)
    assert not np.array_equal(r._b[m], np.zeros_like(r._b[m]))


def test_features_straight_from_a_real_batch_never_reject():
    """End to end over the seam: the producer's output must be updatable by the consumer,
    including the degenerate batches (empty responses, unanimous groups)."""
    r = ContextualBanditRouter(pending_cap=64)
    mask = torch.zeros(4, 9)
    mask[0, 3:5] = 1.0
    mask[1, 3:9] = 1.0
    mask[2, 3:9] = 1.0                              # row 3 keeps an EMPTY response
    lp = torch.full((4, 9), float("nan"))
    lp[mask > 0] = -0.4
    for i, g in enumerate(group_features(torch.tensor([1.0, 0.0, 1.0, 0.0]), mask, lp,
                                         [2, 2], max_response_len=6)):
        c = ctx(g.solve_rate, teacher=True, uid=f"g{i}", extra=g.as_extra())
        r.observe({c.unit_id: outcome(mode_of(r.route(c)), 1.0, c.unit_id)})
    assert (r.updates, r.rejected) == (2, 0)


# ------------------------------------------------------------------- credit -----------


@pytest.mark.parametrize("claimed", DEFAULT_MODES)
def test_an_outcome_is_credited_only_when_it_names_the_mode_that_was_routed(claimed):
    """A mismatch means the caller applied something other than what was routed. Crediting
    the wrong arm corrupts the model in a way no metric would show, so it is dropped."""
    r = ContextualBanditRouter(pending_cap=8)
    c = ctx(teacher=True, uid="u0")
    routed = mode_of(r.route(c))
    before = {m: r._b[m].copy() for m in r.modes}
    r.observe({"u0": outcome(claimed, 1.0, "u0")})
    if claimed == routed:
        assert r.updates == 1 and not np.array_equal(before[routed], r._b[routed])
    else:
        assert r.updates == 0
        assert all(np.array_equal(before[m], r._b[m]) for m in r.modes), claimed
    assert "u0" not in r._pending, "a consumed decision must not be reusable"


def test_an_outcome_for_a_unit_that_was_never_routed_is_dropped():
    r = ContextualBanditRouter()
    r.observe({"never-seen": outcome(TrainingMode.RL, 1.0, "never-seen")})
    assert (r.updates, r.rejected) == (0, 0)


def test_the_same_outcome_observed_twice_is_credited_once():
    """Double credit is how one lucky batch becomes a permanent preference."""
    r = ContextualBanditRouter(pending_cap=8)
    c = ctx(teacher=True, uid="u0")
    o = outcome(mode_of(r.route(c)), 1.0, "u0")
    r.observe({"u0": o})
    r.observe({"u0": o})
    assert r.updates == 1


def test_an_empty_batch_of_outcomes_is_a_no_op():
    r = ContextualBanditRouter()
    r.observe({})
    assert (r.updates, r.rejected, r.evicted) == (0, 0, 0)


# -------------------------------------------------------------- the pending cache ------


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 8])
def test_the_pending_cache_is_bounded_and_every_eviction_is_counted(cap):
    """A cache that grew without bound would leak one entry per unrouted unit for the whole
    run; one that dropped silently would make an unobservable arm look merely unlucky."""
    r = ContextualBanditRouter(pending_cap=cap)
    n = 20
    for i in range(n):
        r.route(ctx(teacher=True, uid=f"u{i}"))
    assert len(r._pending) == cap
    assert r.evicted == n - cap
    assert list(r._pending) == [f"u{i}" for i in range(n - cap, n)], "FIFO, oldest first"


def test_an_evicted_units_outcome_is_dropped_rather_than_crashing():
    r = ContextualBanditRouter(pending_cap=2)
    for i in range(5):
        r.route(ctx(teacher=True, uid=f"u{i}"))
    r.observe({f"u{i}": outcome(TrainingMode.RL, 1.0, f"u{i}") for i in range(3)})
    assert r.updates == 0 and r.evicted == 3
    # the survivors are still creditable
    live = list(r._pending)
    mode, _ = r._pending[live[-1]]
    r.observe({live[-1]: outcome(mode, 1.0, live[-1])})
    assert r.updates == 1


def test_re_routing_a_remembered_unit_refreshes_it_instead_of_evicting_a_bystander():
    """Evicting to make room for a key the cache already holds loses a live decision AND
    overcounts ``evicted``, so the counter that exists to make eviction visible lies."""
    r = ContextualBanditRouter(pending_cap=3)
    for u in ("a", "b", "c"):
        r.route(ctx(teacher=True, uid=u))
    r.route(ctx(teacher=True, uid="a"))
    assert set(r._pending) == {"a", "b", "c"} and r.evicted == 0
    assert list(r._pending)[-1] == "a", "a refresh must move the unit to the newest end"
    r.route(ctx(teacher=True, uid="d"))
    assert r.evicted == 1 and set(r._pending) == {"c", "a", "d"}


def test_a_re_routed_unit_is_credited_against_its_LATEST_decision():
    r = ContextualBanditRouter(pending_cap=8)
    first = ctx(teacher=True, uid="u0", solve_rate=0.0)
    r.route(first)
    second = ctx(teacher=True, uid="u0", solve_rate=1.0, mean_logprob=-5.0)
    m2 = mode_of(r.route(second))
    r.observe({"u0": outcome(m2, 1.0, "u0")})
    assert r.updates == 1
    assert np.allclose(r._b[m2], r._vector(second))


def test_a_unit_with_no_id_is_never_remembered():
    """There is nothing to key an outcome on, so remembering it would credit the next
    unit's outcome to this unit's decision."""
    r = ContextualBanditRouter(pending_cap=4)
    for _ in range(10):
        r.route(ctx(teacher=True, uid=None))
    assert len(r._pending) == 0 and r.evicted == 0


# ------------------------------------------------------------------ determinism --------


def test_the_same_features_and_history_give_the_same_decisions():
    """Reason strings included: the reason is what a run is audited from."""
    def play():
        r = ContextualBanditRouter()
        out = []
        for i in range(40):
            c = ctx(i / 40, teacher=True, uid=f"u{i}",
                    solve_rate=i / 40, len_dispersion=(i % 7) / 7, mean_logprob=-i / 40)
            d = r.route(c)
            out.append((d.weights, d.reason))
            r.observe({c.unit_id: outcome(mode_of(d), (i % 3) / 2, c.unit_id,
                                          batch_id=f"b{i}")})
        return out, r
    a, ra = play()
    b, rb = play()
    assert a == b
    for m in DEFAULT_MODES:
        assert ra.weights(m) == rb.weights(m)


@pytest.mark.parametrize(
    "modes",
    [DEFAULT_MODES, (TrainingMode.SKIP, TrainingMode.SFT, TrainingMode.RL),
     (TrainingMode.SFT, TrainingMode.SKIP), (TrainingMode.DISTILL, TrainingMode.SKIP),
     (TrainingMode.SKIP, TrainingMode.DISTILL), (TrainingMode.RL, TrainingMode.SKIP)],
)
def test_a_fresh_router_ties_on_every_arm_and_breaks_by_mode_name(modes):
    """Every arm starts at theta = 0, so a fresh router's arms are EXACTLY tied on every
    context and the tie-break decides every early decision in a run. It must be the
    package's rule -- the first mode by name, as ``RoutingDecision.argmax`` and
    ``BanditRouter`` both resolve it -- and not the order the caller listed them in."""
    r = ContextualBanditRouter(modes=modes)
    expected = RoutingDecision({m: 1.0 for m in modes}).argmax()
    assert expected == min(modes)
    for p, lp in itertools.product(RATES, (-0.1, -5.0, 0.0)):
        got = mode_of(r.route(ctx(p, teacher=True, solve_rate=p, mean_logprob=lp)))
        assert got == expected, f"p={p} lp={lp}: {got} != {expected}"


def test_the_arms_really_are_tied_so_the_test_above_is_about_the_tie_break():
    """The premise. If the scores differed, the assertion above would be about the argmax
    and a ``>=`` mutant would survive it."""
    r = ContextualBanditRouter()
    x = r._vector(ctx(teacher=True))
    scores = set()
    for m in r.modes:
        A_inv = np.linalg.inv(r._A[m])
        scores.add(float(A_inv @ r._b[m] @ x + r.alpha * np.sqrt(x @ A_inv @ x)))
    assert len(scores) == 1, scores


def test_a_learned_preference_beats_the_tie_break():
    """A tie-break that also decided untied cases would make the router constant."""
    r = ContextualBanditRouter(pending_cap=64)
    for i in range(30):
        c = ctx(teacher=True, uid=f"u{i}")
        m = mode_of(r.route(c))
        r.observe({c.unit_id: outcome(m, 1.0 if m == TrainingMode.SKIP else 0.0,
                                      c.unit_id, batch_id=f"b{i}")})
    r.alpha = 0.0                       # freeze exploration and read the preference off
    assert mode_of(r.route(ctx(teacher=True))) == TrainingMode.SKIP


# ---------------------------------------------------------------- the teacher guard ----


def test_a_teacher_requiring_mode_is_never_selected_without_a_target():
    """The same guard every other router in this package applies, swept over the whole
    context space rather than the one case that motivated it."""
    checked = 0
    for p, teacher, gran, modes in itertools.product(
        RATES, (False, True), GRANS,
        (DEFAULT_MODES, (TrainingMode.SFT, TrainingMode.SKIP),
         (TrainingMode.DISTILL, TrainingMode.RL, TrainingMode.SKIP)),
    ):
        c = ctx(p, teacher=teacher, gran=gran)
        m = mode_of(ContextualBanditRouter(modes=modes).route(c))
        assert not (known_modes()[m] and not c.has_target), f"{m} at p={p} teacher={teacher}"
        checked += 1
    assert checked == len(RATES) * 2 * len(GRANS) * 3


def test_the_guard_beats_a_learned_preference():
    """The interesting direction: a router that has learned SFT is the best mode must still
    not choose it for a unit with no target. A guard applied before learning starts would
    pass the sweep above and fail here."""
    r = ContextualBanditRouter(pending_cap=256)
    for i in range(60):
        c = ctx(1.0, teacher=True, uid=f"u{i}")
        m = mode_of(r.route(c))
        r.observe({c.unit_id: outcome(m, 5.0 if m == TrainingMode.SFT else 0.0,
                                      c.unit_id, batch_id=f"b{i}")})
    r.alpha = 0.0
    assert mode_of(r.route(ctx(1.0, teacher=True))) == TrainingMode.SFT
    for gran in GRANS:
        c = ctx(0.0, teacher=False, gran=gran)
        assert not c.has_target
        assert mode_of(r.route(c)) != TrainingMode.SFT, gran


def test_a_self_target_counts_as_a_target_but_not_at_token_granularity():
    """``_usable`` reads ``has_target``, so a solved group's own correct sample makes SFT
    legal with no teacher -- except at TOKEN granularity, where there is no sibling sample
    and ``has_self_target`` is forced False."""
    r = ContextualBanditRouter(modes=(TrainingMode.SFT, TrainingMode.SKIP))
    assert mode_of(r.route(ctx(0.5, teacher=False))) == TrainingMode.SFT
    assert mode_of(r.route(ctx(0.5, teacher=False, gran=Granularity.TOKEN))) == TrainingMode.SKIP
    assert mode_of(r.route(ctx(0.0, teacher=False))) == TrainingMode.SKIP
    # With SKIP removed there is no fallback arm at all, and the short-circuit fires.
    only_teacher = ContextualBanditRouter(modes=(TrainingMode.SFT, TrainingMode.DISTILL))
    assert known_modes()[mode_of(only_teacher.route(ctx(0.5, teacher=False)))]
    d = only_teacher.route(ctx(0.5, teacher=False, gran=Granularity.TOKEN))
    assert mode_of(d) == TrainingMode.SKIP and "no mode has a target" in d.reason
    assert TrainingMode.SKIP not in only_teacher.modes, "the SKIP above is the fallback"


def test_a_forced_skip_is_not_remembered_and_cannot_be_credited():
    """SKIP-because-no-target is not a decision this router made among alternatives, so
    crediting an outcome to it would train the model on a choice it never had."""
    r = ContextualBanditRouter(modes=(TrainingMode.SFT,), pending_cap=8)
    d = r.route(ctx(0.0, teacher=False, uid="u0"))
    assert mode_of(d) == TrainingMode.SKIP and len(r._pending) == 0
    r.observe({"u0": outcome(TrainingMode.SKIP, 1.0, "u0")})
    assert r.updates == 0


# --------------------------------------------------------------------- learning --------
#
# The claim the module is built on: a controller that decides from features can express a
# rule a threshold on solve_rate cannot. The environment below is built so that claim is
# falsifiable -- solve_rate is IDENTICAL in both regions, so a context-free policy is
# provably unable to be right in both, and the context-free BanditRouter is run against the
# same stream as the bound to beat.


def _regions(seed, n):
    """A stream of (region, context): the regions differ ONLY in ``truncated_fraction``."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        hi = bool(rng.integers(0, 2))
        yield hi, ctx(0.0, teacher=True, uid=f"u{i}",
                      **{n_: 0.0 for n_ in FEATURE_NAMES} | {"truncated_fraction": 0.9 if hi else 0.0})


def _payoff(hi, mode):
    """SFT is optimal for a truncated group, RL for one that is not. SKIP never is."""
    if mode == TrainingMode.SFT:
        return 1.0 if hi else 0.0
    if mode == TrainingMode.RL:
        return 0.0 if hi else 1.0
    return 0.2


def _score(log):
    """Fraction of decisions that named the optimal mode, per region."""
    hi = [m == TrainingMode.SFT for r, m in log if r]
    lo = [m == TrainingMode.RL for r, m in log if not r]
    assert hi and lo, "the stream did not cover both regions"
    return sum(hi) / len(hi), sum(lo) / len(lo)


def _run_contextual(alpha, seed, n=400):
    r = ContextualBanditRouter(alpha=alpha, pending_cap=64)
    log = []
    for i, (hi, c) in enumerate(_regions(seed, n)):
        m = mode_of(r.route(c))
        r.observe({c.unit_id: outcome(m, _payoff(hi, m), c.unit_id, batch_id=f"b{i}")})
        log.append((hi, m))
    return r, log


def _run_context_free(seed, n=400, batch=16):
    """The same stream through ``BanditRouter``, the context-free policy this one extends.

    Outcomes go in fixed batches, and a batch ``BanditRouter`` refuses as confounded (one
    mode only) is dropped, which is what its guard asks a caller to do. The batch has to be
    reasonably wide: flushing as soon as two modes appear starves the router instead --
    ``under_observed`` can narrow to a single mode, that mode is then the only one emitted,
    no batch ever contains two modes again, and its count never rises. That is a property
    of the caller's batching rather than of this module, but it is easy to reproduce and
    would look exactly like a learner that refused to learn.
    """
    r = BanditRouter(explore_prob=0.1, seed=0)
    log, buf = [], {}
    for i, (hi, c) in enumerate(_regions(seed, n)):
        m = mode_of(r.route(c))
        log.append((hi, m))
        buf[c.unit_id] = outcome(m, _payoff(hi, m), c.unit_id, batch_id=f"batch{i // batch}")
        if len(buf) == batch:
            try:
                r.observe(buf)
            except ConfoundedUpdate:
                pass
            buf = {}
    return r, log


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_router_learns_a_rule_a_solve_rate_threshold_cannot_express(seed):
    """LinUCB, driven through the real route/observe loop.

    One mode is optimal in one feature region and another in the other, and ``solve_rate``
    is held constant across both, so nothing a context-free policy can see distinguishes
    them. The router must end up right in BOTH regions, and its fitted weights must carry
    the sign the environment implies -- a router that merely landed on the right mode by
    luck would not.
    """
    r, log = _run_contextual(alpha=1.0, seed=seed)
    early_hi, early_lo = _score(log[:60])
    late_hi, late_lo = _score(log[-100:])
    assert late_hi >= 0.9, f"truncated region: {early_hi:.2f} -> {late_hi:.2f}"
    assert late_lo >= 0.9, f"clean region: {early_lo:.2f} -> {late_lo:.2f}"
    assert r.updates == 400 and r.rejected == 0

    w_sft, w_rl = r.weights(TrainingMode.SFT), r.weights(TrainingMode.RL)
    assert w_sft["truncated_fraction"] > 0.25, w_sft
    assert w_rl["truncated_fraction"] < -0.25, w_rl
    assert w_rl["intercept"] > w_sft["intercept"], (w_rl, w_sft)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_context_free_bandit_cannot_be_right_in_both_regions_on_the_same_stream(seed):
    """The bound. Without this the learning test above proves only that the environment is
    easy, not that the features are what solved it.

    ``solve_rate`` is identical in the two regions, so a policy that cannot see the feature
    must emit the same distribution in both: picking the truncated region's optimum with
    probability q scores q in one region and 1 - q in the other, and ``min`` of those is at
    most 0.5 for every q. The bandit run here confirms it does the profitable thing -- it
    converges on ONE mode and is right about 90% of the time in that mode's region, so what
    it fails is the other region, not the task.
    """
    bandit, log = _run_context_free(seed)
    hi, lo = _score(log[-100:])
    assert max(hi, lo) > 0.7, f"the control did not converge at all: {hi:.2f}/{lo:.2f}"
    assert len(bandit.value_estimates()) == 3, "the control never observed every arm"
    assert min(hi, lo) < 0.2, f"context-free router scored {hi:.2f}/{lo:.2f}"
    _, ctx_log = _run_contextual(alpha=1.0, seed=seed)
    assert min(_score(ctx_log[-100:])) >= 0.9 > min(hi, lo)


def test_the_learned_policy_acts_on_the_feature_at_decision_time():
    """Learning the right weights is not the same as using them: with exploration frozen
    off, the SAME router must decide differently for the two regions."""
    r, _ = _run_contextual(alpha=1.0, seed=0, n=600)
    r.alpha = 0.0
    hi = ctx(0.0, teacher=True, **{n_: 0.0 for n_ in FEATURE_NAMES} | {"truncated_fraction": 0.9})
    lo = ctx(0.0, teacher=True, **{n_: 0.0 for n_ in FEATURE_NAMES} | {"truncated_fraction": 0.0})
    assert mode_of(r.route(hi)) == TrainingMode.SFT
    assert mode_of(r.route(lo)) == TrainingMode.RL


def test_greedy_never_explores_at_all():
    """PINNED HAZARD, not an endorsement.

    ``alpha = 0`` removes the confidence term, every arm starts at theta = 0, so the arms
    tie forever, the tie-break picks the first mode by name, and that mode is the only one
    ever observed. ``BanditRouter`` REJECTS ``explore_prob = 0`` for exactly this reason;
    here it is legal, so a run configured with ``alpha = 0`` is a constant router that
    still reports itself as contextual. Pinned so it cannot change silently.
    """
    r, log = _run_contextual(alpha=0.0, seed=0)
    assert {m for _, m in log} == {TrainingMode.RL}
    assert _score(log[-100:]) == (0.0, 1.0)
    assert r.weights(TrainingMode.SFT) == {k: 0.0 for k in (*FEATURE_NAMES, "intercept")}


def test_exploration_has_to_be_wide_enough_to_reach_the_second_region():
    """PINNED HAZARD. A small but non-zero alpha still locks in: the confidence term has to
    outweigh a learned advantage before the arm is ever tried. The default of 1.0 does."""
    _, small = _run_contextual(alpha=0.05, seed=0)
    _, default = _run_contextual(alpha=1.0, seed=0)
    assert min(_score(small[-100:])) < 0.5
    assert min(_score(default[-100:])) >= 0.9


# ------------------------------------------------------------------- construction ------


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(modes=()), "non-empty"),
        (dict(modes=("telepathy",)), "unknown mode"),
        (dict(modes=(TrainingMode.RL, "telepathy")), "unknown mode"),
        (dict(feature_names=()), "non-empty"),
        (dict(alpha=-0.1), "alpha must be >= 0"),
        (dict(ridge=0.0), "ridge must be > 0"),
        (dict(ridge=-1.0), "ridge must be > 0"),
        (dict(pending_cap=0), "pending_cap must be >= 1"),
        (dict(pending_cap=-5), "pending_cap must be >= 1"),
    ],
)
def test_bad_configurations_are_refused_at_construction(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ContextualBanditRouter(**kwargs)


def test_two_routers_do_not_share_a_model():
    """A mutable default would make every router in a sweep the same router."""
    a, b = ContextualBanditRouter(), ContextualBanditRouter()
    c = ctx(teacher=True, uid="u0")
    a.observe({"u0": outcome(mode_of(a.route(c)), 1.0, "u0")})
    assert a.updates == 1 and b.updates == 0
    assert not np.array_equal(a._A[TrainingMode.RL], b._A[TrainingMode.RL])
    assert len(b._pending) == 0


@pytest.mark.parametrize("names", [FEATURE_NAMES, ("solve_rate",),
                                   ("solve_rate", "truncated_fraction")])
def test_the_model_has_one_dimension_per_named_feature_plus_an_intercept(names):
    """Without the intercept an arm's value is forced through the origin, so a mode that is
    uniformly good cannot be represented at all."""
    r = ContextualBanditRouter(feature_names=names)
    d = len(names) + 1
    assert r._A[TrainingMode.RL].shape == (d, d)
    assert list(r.weights(TrainingMode.RL)) == [*names, "intercept"]
    x = r._vector(ctx(teacher=True))
    assert len(x) == d and x[-1] == 1.0


def test_the_intercept_lets_an_arm_be_uniformly_preferred_from_an_all_zero_context():
    """The case the intercept exists for: with every feature at 0 the feature part of the
    score is 0 for every arm, so only the intercept can express a preference."""
    zero = {n: 0.0 for n in FEATURE_NAMES}
    r = ContextualBanditRouter(pending_cap=64)
    for i in range(40):
        c = ctx(0.0, teacher=True, uid=f"u{i}", extra=dict(zero))
        m = mode_of(r.route(c))
        r.observe({c.unit_id: outcome(m, 1.0 if m == TrainingMode.SFT else 0.0, c.unit_id,
                                      batch_id=f"b{i}")})
    r.alpha = 0.0
    assert mode_of(r.route(ctx(0.0, teacher=True, extra=dict(zero)))) == TrainingMode.SFT
    assert r.weights(TrainingMode.SFT)["intercept"] > 0.5


def test_weights_names_every_coefficient_and_refuses_an_unknown_arm():
    """Exposed so a run can be audited for WHICH feature drove a decision."""
    r = ContextualBanditRouter()
    w = r.weights(TrainingMode.RL)
    assert set(w) == {*FEATURE_NAMES, "intercept"}
    assert all(v == 0.0 for v in w.values()), "an unobserved arm is the prior, not noise"
    with pytest.raises(ValueError, match="not one of this router's modes"):
        r.weights(TrainingMode.DISTILL)


def test_ridge_is_the_prior_strength_and_damps_a_single_observation():
    """A larger ridge must move theta LESS for the same evidence; ``A`` starting at zero
    instead of ``ridge * I`` would make the first observation infinitely confident."""
    seen = []
    for ridge in (0.5, 1.0, 10.0):
        r = ContextualBanditRouter(ridge=ridge, pending_cap=8)
        c = ctx(teacher=True, uid="u0")
        r.observe({"u0": outcome(mode_of(r.route(c)), 1.0, "u0")})
        seen.append(max(abs(v) for v in r.weights(TrainingMode.RL).values()))
    assert seen[0] > seen[1] > seen[2] > 0.0, seen


# -------------------------------------------------------- confounding, not guarded ------


def test_observe_accepts_a_batch_a_context_free_bandit_would_refuse():
    """PINNED HAZARD, not an endorsement.

    ``BanditRouter.observe`` raises ``ConfoundedUpdate`` when a batch used a single mode
    (the mode is then perfectly confounded with the batch) or when outcomes span several
    ``batch_id``s (they are not comparable). This router applies both without complaint --
    ``batch_id`` is not read at all. Pinned so the asymmetry is a decision on the record
    rather than an oversight, because the same confound applies to a contextual policy.
    """
    r = ContextualBanditRouter(pending_cap=8)
    single = {}
    for i in range(6):
        c = ctx(teacher=True, uid=f"u{i}")
        m = mode_of(r.route(c))
        single[c.unit_id] = outcome(m, 1.0, c.unit_id, batch_id=f"mixed{i}")
    assert len({o.batch_id for o in single.values()}) > 1
    r.observe(single)
    assert r.updates == 6, "outcomes from different batches were pooled without complaint"

    one_mode = {k: v for k, v in single.items() if v.mode == list(single.values())[0].mode}
    with pytest.raises(Exception):
        BanditRouter(seed=0).observe(
            {k: outcome(o.mode, o.value, k, batch_id="one") for k, o in one_mode.items()}
        )
