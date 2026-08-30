"""Tests for the outcome-calibrated meta-critic.

The class exists to distinguish "no signal" from "inverted signal", so the tests are built
around that distinction rather than around AUC arithmetic.
"""
from __future__ import annotations

import pytest

from selfevo.critics import CriticScore
from selfevo.meta_critics import (
    CalibrationReport,
    CalibrationVerdict,
    OutcomeCalibratedMetaCritic,
    _auc,
)
from selfevo.routing.criteria import SilenceSide


def sc(value: float, unit_id: str | None, *, coarse: bool = False) -> CriticScore:
    return CriticScore(value=value, basis="test", side=SilenceSide.INFORMATIVE,
                       group_size=8, unit_id=unit_id, coarse=coarse)


def _series(n: int):
    """n descending scores whose first half succeed: a perfect ranker."""
    scores = [sc(1.0 - i / n, f"u{i}") for i in range(n)]
    outcomes = {f"u{i}": i < n // 2 for i in range(n)}
    return scores, outcomes


def test_perfect_ranker_is_informative():
    s, o = _series(20)
    r = OutcomeCalibratedMetaCritic().assess(s, o)
    assert r.auc == 1.0
    assert r.verdict is CalibrationVerdict.INFORMATIVE


def test_inverted_ranker_is_mis_ordering_not_merely_uninformative():
    """The distinction the class exists for: an inverted critic is worse than a silent one."""
    s, o = _series(20)
    flipped = {k: not v for k, v in o.items()}
    r = OutcomeCalibratedMetaCritic().assess(s, flipped)
    assert r.auc == 0.0
    assert r.verdict is CalibrationVerdict.MIS_ORDERING
    assert r.verdict is not CalibrationVerdict.UNINFORMATIVE


def test_constant_scorer_is_uninformative_not_perfect():
    """Ties must count a half, or a constant scorer reads as perfect or as inverted."""
    scores = [sc(0.5, f"u{i}") for i in range(20)]
    outcomes = {f"u{i}": i < 10 for i in range(20)}
    r = OutcomeCalibratedMetaCritic().assess(scores, outcomes)
    assert r.auc == 0.5
    assert r.verdict is CalibrationVerdict.UNINFORMATIVE
    assert "identical value" in r.basis


def test_auc_ties_count_a_half():
    assert _auc([1.0, 1.0], [True, False]) == 0.5
    assert _auc([2.0, 1.0], [True, False]) == 1.0
    assert _auc([1.0, 2.0], [True, False]) == 0.0


def test_partial_ties_are_scored_between_the_extremes():
    """A partly-tied ranker must land strictly between chance and perfect."""
    scores = [sc(1.0, "a"), sc(0.5, "b"), sc(0.5, "c"), sc(0.0, "d")]
    out = {"a": True, "b": True, "c": False, "d": False}
    r = OutcomeCalibratedMetaCritic(min_paired=4).assess(scores, out)
    assert 0.5 < r.auc < 1.0


def test_too_few_pairs_refuses_rather_than_guessing():
    s, o = _series(6)
    r = OutcomeCalibratedMetaCritic(min_paired=20).assess(s, o)
    assert r.verdict is CalibrationVerdict.INSUFFICIENT
    assert r.auc is None
    assert "need 20" in r.basis


def test_uniform_outcomes_refuse_because_ranking_is_untestable():
    """All-success or all-failure says nothing about the critic, and must not read as 0.5."""
    scores = [sc(i / 20, f"u{i}") for i in range(20)]
    for const in (True, False):
        r = OutcomeCalibratedMetaCritic().assess(scores, {f"u{i}": const for i in range(20)})
        assert r.verdict is CalibrationVerdict.INSUFFICIENT
        assert r.auc is None


def test_coarse_scores_are_dropped_and_counted():
    """The critic flags coarse when a single group cannot support a ranking."""
    s, o = _series(20)
    s = s + [sc(0.99, "junk", coarse=True)]
    o = dict(o, junk=False)
    r = OutcomeCalibratedMetaCritic().assess(s, o)
    assert r.n_dropped_coarse == 1
    assert r.n_paired == 20


def test_coarse_scores_are_used_when_asked():
    s, o = _series(20)
    s = s + [sc(0.99, "junk", coarse=True)]
    o = dict(o, junk=False)
    r = OutcomeCalibratedMetaCritic(use_coarse=True).assess(s, o)
    assert r.n_dropped_coarse == 0
    assert r.n_paired == 21


def test_unpaired_scores_are_reported_not_silently_dropped():
    """A critic scoring units nobody observed is its own failure mode."""
    s, o = _series(20)
    s = s + [sc(0.4, "never_observed"), sc(0.4, None)]
    r = OutcomeCalibratedMetaCritic().assess(s, o)
    assert r.n_unpaired == 2
    assert r.n_scored == 22
    assert r.n_paired == 20


def test_margin_defines_the_uninformative_band():
    """Just inside the margin is UNINFORMATIVE; widening the margin absorbs a real signal."""
    scores = [sc(1.0, "a"), sc(0.9, "b"), sc(0.8, "c"), sc(0.7, "d")]
    out = {"a": True, "b": False, "c": True, "d": False}  # AUC = 0.75
    assert OutcomeCalibratedMetaCritic(min_paired=4, margin=0.05).assess(scores, out).verdict \
        is CalibrationVerdict.INFORMATIVE
    assert OutcomeCalibratedMetaCritic(min_paired=4, margin=0.30).assess(scores, out).verdict \
        is CalibrationVerdict.UNINFORMATIVE


def test_rejects_incoherent_configuration():
    with pytest.raises(ValueError):
        OutcomeCalibratedMetaCritic(min_paired=1)
    with pytest.raises(ValueError):
        OutcomeCalibratedMetaCritic(margin=0.5)
    with pytest.raises(ValueError):
        OutcomeCalibratedMetaCritic(margin=-0.1)


def test_report_rejects_incoherent_states():
    """An INSUFFICIENT verdict carrying an AUC would imply a conclusion it disclaims."""
    with pytest.raises(ValueError):
        CalibrationReport(auc=0.7, verdict=CalibrationVerdict.INSUFFICIENT,
                          n_scored=1, n_paired=1, n_dropped_coarse=0, n_unpaired=0, basis="b")
    with pytest.raises(ValueError):
        CalibrationReport(auc=None, verdict=CalibrationVerdict.INFORMATIVE,
                          n_scored=1, n_paired=1, n_dropped_coarse=0, n_unpaired=0, basis="b")
    with pytest.raises(ValueError):
        CalibrationReport(auc=1.5, verdict=CalibrationVerdict.INFORMATIVE,
                          n_scored=1, n_paired=1, n_dropped_coarse=0, n_unpaired=0, basis="b")
    with pytest.raises(ValueError):
        CalibrationReport(auc=0.7, verdict=CalibrationVerdict.INFORMATIVE,
                          n_scored=1, n_paired=1, n_dropped_coarse=0, n_unpaired=0, basis="")


def test_auc_refuses_single_class():
    with pytest.raises(ValueError):
        _auc([1.0, 2.0], [True, True])


def test_registry_builds_it():
    """The factory must construct the real class, not stay a stub."""
    from selfevo.compose import META_CRITIC_FACTORIES
    f = META_CRITIC_FACTORIES["outcome_calibrated"]
    assert f is not None
    assert isinstance(f(min_paired=4), OutcomeCalibratedMetaCritic)


def test_verdict_boundaries_are_inclusive():
    """AUC exactly on the margin counts as informative / mis-ordering, not as chance.

    The four scores below give AUC 0.75 and, with outcomes flipped, 0.25. Setting
    margin=0.25 puts both exactly on their boundary, which is the only place a >= / >
    confusion is observable -- every interior test passes either way.
    """
    scores = [sc(1.0, "a"), sc(0.9, "b"), sc(0.8, "c"), sc(0.7, "d")]
    up = {"a": True, "b": False, "c": True, "d": False}
    m = OutcomeCalibratedMetaCritic(min_paired=4, margin=0.25)

    hi = m.assess(scores, up)
    assert hi.auc == 0.75
    assert hi.verdict is CalibrationVerdict.INFORMATIVE

    lo = m.assess(scores, {k: not v for k, v in up.items()})
    assert lo.auc == 0.25
    assert lo.verdict is CalibrationVerdict.MIS_ORDERING
