"""Which of the seven observability features are functions of the reward vector, checked.

The 102/102 rule-versus-solve-rate equivalence (EXPERIMENTS.md 2026-08-31, GOAL.md M9) is
either a corollary of a published result or a new negative result about covariates. It is
published -- arXiv 2607.00152 Thm 1, 2605.05112 Eq. 1, 2510.13651 -- that for a binary
reward vector ``r`` in ``{0,1}^G`` every group-relative advantage and every reward-derived
statistic is a function of ``k = #correct`` alone. So the whole question is which of the
seven features in :mod:`selfevo.observability` are pure functions of ``r``, and which read a
covariate -- response length, truncation, log-probabilities -- that ``r`` does not determine.

This module answers that as a CHECKED FACT rather than as a reading of the source. The
partition is asserted in both directions: a k-function must not move when a covariate moves
with ``r`` held fixed, and a covariate must move. Then the partition is applied to the sweep
itself, so "the 102 contexts could not have detected covariate information" is a test rather
than a paragraph. A feature changing sides becomes a failure here instead of a stale
sentence in a findings file.

One distinction is decisive and worth stating up front, because a feature named "entropy"
would be ambiguous -- the entropy of the binary reward distribution ``H(k/G)`` is a
k-function, the sampler's token-level entropy is a covariate. Neither is here: nothing named
entropy is in ``FEATURE_NAMES`` at all, which
:func:`test_no_feature_is_an_entropy_of_either_kind` pins so the ambiguity cannot be
reintroduced silently.

Scope. This audits the RULE's decision function. The learned controller
(:class:`selfevo.routing.contextual.ContextualRouter`) defaults to the same full
``FEATURE_NAMES``, so its INPUT space is not k-collapsed even though the rule's decision is;
that asymmetry is the reason the collapse measured here says nothing about what a learned
router could in principle read.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import pytest
import torch

from selfevo.compose import ROUTERS
from selfevo.observability import FEATURE_NAMES, group_features
from selfevo.routing.base import HarnessAction, RoutingContext, TrainingMode
from selfevo.routing.routers import SolveRateRouter
from selfevo.routing.rule_policy import READ_FEATURES

# The classification this file exists to prove. Stated as data so every test below reads
# from one place and a reclassification is a one-line edit that many assertions check.
#
#   solve_rate  = #(r > 0.5) / G                    -> k / G
#   reward_std  = population std of r               -> sqrt(k(G-k)) / G  for binary r
#
# Neither touches loss_mask, logprobs or the token budget. The other five each read one of
# those and nothing about r at all.
K_FUNCTIONS: tuple[str, ...] = ("solve_rate", "reward_std")
COVARIATES: tuple[str, ...] = (
    "mean_response_len",
    "len_dispersion",
    "mean_logprob",
    "logprob_dispersion",
    "truncated_fraction",
)

# Group sizes swept. Even throughout, because two of the covariate pairs below hold a mean
# fixed while moving its dispersion and that construction needs pairing.
GROUP_SIZES: tuple[int, ...] = (2, 4, 8, 16)


def _features(
    rewards: list[float],
    *,
    lengths: list[int] | None = None,
    logprobs: list[float] | None = None,
    max_response_len: int | None = None,
) -> dict[str, float]:
    """Run the REAL producer on one group, with every covariate under the caller's control.

    A mock would make this file test its own idea of the features rather than the ones the
    actor computes, which is the failure mode the whole audit exists to avoid.

    Args:
        rewards: One raw reward per sample. This list IS the reward vector ``r``.
        lengths: Response length per sample, written as a ragged loss mask exactly as a real
            batch carries it. Defaults to 6 tokens for every sample.
        logprobs: Per-token log-probability for each sample, constant within the sample so
            the sample's mean log-probability is the value given. Defaults to -0.5.
        max_response_len: Token budget, or None for "not supplied", which the producer
            reports as ``truncated_fraction`` 0.0 rather than guessing.

    Returns:
        ``GroupFeatures.as_extra()`` for the single group.
    """
    n = len(rewards)
    lens = [6] * n if lengths is None else lengths
    lps = [-0.5] * n if logprobs is None else logprobs
    width = max(lens)
    mask = torch.zeros(n, width)
    lp = torch.zeros(n, width)
    for i, (ln, value) in enumerate(zip(lens, lps)):
        mask[i, :ln] = 1.0
        # Written across the FULL row, padding included, so the producer's masking is
        # exercised rather than assumed: an unmasked producer would report a different mean.
        lp[i, :] = value
    return group_features(
        torch.tensor(rewards, dtype=torch.float32),
        mask,
        lp,
        [n],
        max_response_len=max_response_len,
    )[0].as_extra()


# For each covariate, two producer keyword sets that differ ONLY in inputs the reward vector
# does not determine. Each is a function of the group size so the same pair can be applied at
# every G in the sweep.
_MOVES: dict[str, tuple[Callable[[int], dict], Callable[[int], dict]]] = {
    # Longer responses, same shape.
    "mean_response_len": (
        lambda n: {"lengths": [4] * n},
        lambda n: {"lengths": [9] * n},
    ),
    # Same MEAN length, different spread: 0 dispersion against 0.5, so this pair moves the
    # dispersion without moving the mean it is normalised by.
    "len_dispersion": (
        lambda n: {"lengths": [4] * n},
        lambda n: {"lengths": [2 if i % 2 else 6 for i in range(n)]},
    ),
    # A less confident sample of the same length.
    "mean_logprob": (
        lambda n: {"logprobs": [-0.5] * n},
        lambda n: {"logprobs": [-1.5] * n},
    ),
    # Same MEAN log-probability, different spread within the group: the "unanimous in answer,
    # diverse in reasoning" case the feature's docstring names.
    "logprob_dispersion": (
        lambda n: {"logprobs": [-0.5] * n},
        lambda n: {"logprobs": [-0.25 if i % 2 else -0.75 for i in range(n)]},
    ),
    # Identical responses; only the token budget they are measured against changes.
    "truncated_fraction": (
        lambda n: {"lengths": [6] * n, "max_response_len": None},
        lambda n: {"lengths": [6] * n, "max_response_len": 6},
    ),
}


def _binary(k: int, g: int) -> list[float]:
    """A binary reward vector with ``k`` of ``g`` samples correct."""
    return [1.0] * k + [0.0] * (g - k)


# ------------------------------------------------------------------- the partition ----


def test_the_partition_covers_every_feature_exactly_once():
    """The classification is exhaustive and disjoint, so no feature escapes it unclassified.

    A feature added to ``GroupFeatures`` without being classified fails here, which is the
    point: the verdict in ``FINDINGS_feature_audit.md`` is only as good as its coverage of
    the feature set, and an eighth feature would silently invalidate it otherwise.
    """
    assert set(K_FUNCTIONS) | set(COVARIATES) == set(FEATURE_NAMES)
    assert set(K_FUNCTIONS) & set(COVARIATES) == set()
    assert len(K_FUNCTIONS) + len(COVARIATES) == len(FEATURE_NAMES) == 7


def test_no_feature_is_an_entropy_of_either_kind():
    """The decisive naming check: there is no entropy feature, so it cannot be miscounted.

    Had one existed, its classification would decide the whole audit -- reward entropy
    ``H(k/G)`` is a function of ``k`` and adds nothing, while the sampler's token-level
    entropy is a covariate and would be the escape route. Asserting its absence stops a
    future feature called ``entropy`` from being read as either.
    """
    assert not [n for n in FEATURE_NAMES if "entropy" in n.lower()]


@pytest.mark.parametrize("covariate", COVARIATES)
def test_each_k_function_is_invariant_when_a_covariate_moves_with_the_rewards_fixed(
    covariate: str,
):
    """A k-function must not notice a covariate. Exact equality, over the whole k range.

    The comparison is ``==`` rather than ``approx``: both sides run the same float32
    reduction over the same reward tensor, so any difference at all would mean the feature
    read something other than ``r``.
    """
    lo, hi = _MOVES[covariate]
    for g in GROUP_SIZES:
        for k in range(g + 1):
            r = _binary(k, g)
            a = _features(r, **lo(g))
            b = _features(r, **hi(g))
            for name in K_FUNCTIONS:
                assert a[name] == b[name], (covariate, name, g, k, a[name], b[name])


@pytest.mark.parametrize("covariate", COVARIATES)
def test_each_covariate_moves_while_the_reward_vector_is_held_fixed(covariate: str):
    """The other direction: a covariate must actually differ, or the label is unearned.

    Without this half, "covariate" would be an unfalsifiable label -- a feature that happens
    to be constant is trivially invariant under everything. Each pair is built to move the
    named feature and nothing about ``r``, and the k-functions are re-asserted equal on the
    same two contexts so the pair is known to be a pure covariate perturbation.
    """
    lo, hi = _MOVES[covariate]
    for g in GROUP_SIZES:
        for k in range(g + 1):
            r = _binary(k, g)
            a = _features(r, **lo(g))
            b = _features(r, **hi(g))
            assert a[covariate] != b[covariate], (covariate, g, k, a[covariate])
            for name in K_FUNCTIONS:
                assert a[name] == b[name], (covariate, name, g, k)


def test_the_two_k_functions_are_the_closed_forms_in_k_and_g():
    """Not just invariant -- the exact published forms, so "k-function" is quantitative.

    ``solve_rate = k/G`` and, for a binary group, ``reward_std = sqrt(k(G-k))/G``. Pinning
    the closed forms is what makes the corollary argument concrete: these two numbers carry
    no more information than ``k`` does, at any group size.
    """
    for g in GROUP_SIZES:
        for k in range(g + 1):
            f = _features(_binary(k, g))
            assert f["solve_rate"] == pytest.approx(k / g, abs=1e-7)
            assert f["reward_std"] == pytest.approx(math.sqrt(k * (g - k)) / g, abs=1e-6)


# ------------------------------------------------------- what the rule actually reads ----


def test_the_rule_reads_two_k_functions_and_exactly_one_covariate():
    """``READ_FEATURES`` against the partition, so the module constant is audited too.

    ``rule_policy.READ_FEATURES`` is the file's own claim about itself. Checking it against a
    classification derived from the producer turns two independent statements into one.
    """
    assert set(READ_FEATURES) == {"solve_rate", "reward_std", "truncated_fraction"}
    assert set(READ_FEATURES) & set(K_FUNCTIONS) == {"solve_rate", "reward_std"}
    assert set(READ_FEATURES) & set(COVARIATES) == {"truncated_fraction"}


def _ctx(feats: dict[str, float], g: int, **kw: object) -> RoutingContext:
    """A context whose ``solve_rate`` and features agree, as ``actor._route_groups`` builds it.

    Args:
        feats: The full seven-feature mapping from :func:`_features`.
        g: Group size.
        **kw: Passed through to :class:`RoutingContext` (``has_teacher``,
            ``can_evolve_harness``).

    Returns:
        The context.
    """
    return RoutingContext(
        solve_rate=feats["solve_rate"], group_size=g, extra=feats, **kw
    )


def test_the_one_covariate_the_rule_reads_cannot_change_the_mode():
    """``truncated_fraction`` moves no mode, on any branch, with or without a harness arm.

    This is what confines the rule's MODE to the k-functions: the single covariate it reads
    is wired to the harness axis only. Swept over both silent branches and the informative
    one, both teacher settings, and both harness settings.
    """
    rule = ROUTERS["rule"]()
    for g in GROUP_SIZES:
        for k in (0, g // 2, g):
            base = _features(_binary(k, g))
            for teacher, harness_arm in itertools.product((False, True), repeat=2):
                modes = {
                    rule.route(
                        _ctx(
                            base | {"truncated_fraction": t},
                            g,
                            has_teacher=teacher,
                            can_evolve_harness=harness_arm,
                        )
                    ).argmax()
                    for t in (0.0, 0.5, 1.0)
                }
                assert len(modes) == 1, (g, k, teacher, harness_arm, modes)


def test_truncated_fraction_moves_the_harness_axis_only_when_a_harness_arm_exists():
    """The covariate is not inert in the code -- it is inert in the CONFIGURATION.

    With ``can_evolve_harness`` True the full decision ``(mode, harness)`` is NOT a function
    of ``k`` alone: two unsolved-and-silent groups with the same ``k`` differ in ``.harness``
    when their truncation differs. With it False -- the default, and what every context in
    the 102-sweep used -- the action is dropped and the decision collapses to ``k``. So the
    equivalence measured on ``.argmax()`` is strictly narrower than "the router is a
    k-function", and the difference is exactly one unconsumed axis.
    """
    rule = ROUTERS["rule"]()
    base = _features(_binary(0, 8))

    live = [
        rule.route(_ctx(base | {"truncated_fraction": t}, 8, can_evolve_harness=True))
        for t in (0.0, 1.0)
    ]
    assert [d.harness for d in live] == [HarnessAction.NONE, HarnessAction.PROPOSE]
    assert live[0].argmax() == live[1].argmax() == TrainingMode.SKIP

    dead = [
        rule.route(_ctx(base | {"truncated_fraction": t}, 8, can_evolve_harness=False))
        for t in (0.0, 1.0)
    ]
    assert [d.harness for d in dead] == [HarnessAction.NONE, HarnessAction.NONE]


def test_the_rule_decision_is_a_function_of_k_alone_under_the_shipped_configuration():
    """The corollary, executed: hold ``k`` fixed, move every covariate, nothing moves.

    Each of the five covariate perturbations is applied at every ``(G, k)`` and both teacher
    settings, with ``can_evolve_harness`` at its default False, and the WHOLE decision --
    mode and harness -- is required to be identical. That is the theorem's content
    (2607.00152 Thm 1) reduced to this router: its decision function factors through ``k``,
    so no measurement taken with the shipped configuration can say anything about covariates.
    """
    rule = ROUTERS["rule"]()
    for g in GROUP_SIZES:
        for k in range(g + 1):
            r = _binary(k, g)
            decisions = set()
            for lo, hi in _MOVES.values():
                for kwargs in (lo(g), hi(g)):
                    feats = _features(r, **kwargs)
                    for teacher in (False, True):
                        d = rule.route(_ctx(feats, g, has_teacher=teacher))
                        decisions.add((teacher, d.argmax(), d.harness))
            # Two teacher settings, one decision each: 2 distinct tuples, never more.
            assert len(decisions) == 2, (g, k, sorted(map(str, decisions)))


# ------------------------------------------------------- the 102 contexts themselves ----


def _sweep() -> list[dict]:
    """Rebuild the 2026-08-31 audit sweep exactly as ``test_rule_policy`` runs it.

    Returns:
        One record per context: ``k``, ``group_size``, ``has_teacher``, the truncation value
        that was substituted, and the full feature mapping the producer returned.

    Rebuilt rather than imported because the point is to inspect the CONTEXTS, which that
    test discards. The construction is kept identical -- same group sizes, same k range, same
    truncation values, same producer inputs (uniform length 6, uniform log-probability -0.5)
    -- and :func:`test_the_rebuilt_sweep_reproduces_the_published_102_over_102` checks the
    rebuild against the published result before anything is concluded from it.
    """
    out = []
    for has_teacher in (False, True):
        for g in (2, 4, 8, 16):
            for k in range(g + 1):
                for trunc in (0.0, 0.5, 1.0):
                    feats = _features(_binary(k, g))
                    feats["truncated_fraction"] = trunc
                    out.append(
                        {
                            "k": k,
                            "group_size": g,
                            "has_teacher": has_teacher,
                            "trunc": trunc,
                            "feats": feats,
                        }
                    )
    return out


def test_the_rebuilt_sweep_reproduces_the_published_102_over_102():
    """The rebuild is the audited object, so it must agree with the audited number first."""
    rule, solve_rate = ROUTERS["rule"](), SolveRateRouter()
    per_teacher: dict[bool, int] = {False: 0, True: 0}
    disagreements = []
    for row in _sweep():
        ctx = _ctx(row["feats"], row["group_size"], has_teacher=row["has_teacher"])
        per_teacher[row["has_teacher"]] += 1
        if rule.route(ctx).argmax() != solve_rate.route(ctx).argmax():
            disagreements.append(row)
    assert per_teacher == {False: 102, True: 102}
    assert disagreements == []


def test_the_102_contexts_span_every_k_including_the_degenerate_ends():
    """k coverage, reported as an assertion: 0..G inclusive at every G, ends included.

    RL-ZVP (2509.21880) and HIVE (2603.25184) locate covariate gains specifically at
    ``k in {0, G}``, so a sweep that skipped the ends could be dismissed on coverage alone.
    It does not skip them -- every k from 0 to G appears, at all four group sizes, under both
    teacher settings. Coverage of ``k`` is not the audit's limitation; see the next test for
    what is.
    """
    rows = _sweep()
    for has_teacher in (False, True):
        for g in (2, 4, 8, 16):
            ks = sorted(
                {
                    r["k"]
                    for r in rows
                    if r["group_size"] == g and r["has_teacher"] is has_teacher
                }
            )
            assert ks == list(range(g + 1)), (g, has_teacher, ks)
            assert 0 in ks and g in ks
    # 34 compositions x 3 truncation values x 2 teacher settings.
    assert len(rows) == 204


def test_the_102_contexts_hold_four_of_the_five_covariates_constant():
    """The audit's real limitation, made checkable: only one covariate ever varied.

    ``mean_response_len``, ``len_dispersion``, ``mean_logprob`` and ``logprob_dispersion``
    take a SINGLE value across all 204 contexts, because the producer was fed a uniform
    length-6 mask and a uniform -0.5 log-probability every time. The fifth,
    ``truncated_fraction``, varied over {0, 0.5, 1} but is read only on the harness axis,
    which no context in the sweep enabled and which the comparison (``.argmax()``) did not
    look at.

    Consequence, and this is the verdict this whole file supports: the sweep never had a
    covariate vary while ``k`` was held fixed AND the varying covariate reach the compared
    output. It therefore cannot have detected covariate information, and the 102/102 is a
    corollary rather than a constraint on covariates.
    """
    rows = _sweep()
    for name in ("mean_response_len", "len_dispersion", "mean_logprob", "logprob_dispersion"):
        values = {r["feats"][name] for r in rows}
        assert len(values) == 1, (name, sorted(values))
    assert {r["trunc"] for r in rows} == {0.0, 0.5, 1.0}
    # And the one that did vary reached nothing: no context enabled the harness arm, and the
    # compared output was the mode.
    rule = ROUTERS["rule"]()
    by_key: dict[tuple, set] = {}
    for row in rows:
        ctx = _ctx(row["feats"], row["group_size"], has_teacher=row["has_teacher"])
        key = (row["group_size"], row["k"], row["has_teacher"])
        d = rule.route(ctx)
        by_key.setdefault(key, set()).add((d.argmax(), d.harness))
    assert all(len(v) == 1 for v in by_key.values())
