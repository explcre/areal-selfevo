"""Tests for the per-group observability features.

These features exist so a controller can decide from something other than ``solve_rate``.
That makes two properties load-bearing, and both are swept rather than sampled:

* MASKING -- every feature must be a property of the RESPONSE. The prompt region is written
  with a distinct value (and with NaN, and with infinities) in every test that touches
  log-probabilities, because ``logprobs * loss_mask`` does NOT mask a non-finite value:
  ``nan * 0 == nan``, so an unmasked-by-multiplication implementation contaminates a sum the
  prompt is not part of, and the contamination then hides behind ``_safe_div``'s default.
* FINITENESS -- ``group_features`` guarantees every returned field is finite. A NaN in a
  feature vector poisons a linear policy's arm permanently and nothing downstream reports
  it, so the guarantee is checked over an adversarial cross-product of inputs rather than on
  the one case that motivated it.

The third property, GROUPING, is checked by computing a group's features twice -- once
inside a batch and once on its own rows -- rather than by re-deriving the slice arithmetic
in the test file. A test that pins a copy of the expression cannot notice the copy drifting.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from selfevo.observability import (
    FEATURE_NAMES,
    GroupFeatures,
    _safe_div,
    group_features,
)
from selfevo.routing.base import RoutingContext
from selfevo.routing.contextual import ContextualBanditRouter

PROMPT = 3                      # columns before the response; loss_mask == 0 there
OUTSIDE = -9.0                  # log-prob written everywhere the mask is 0
NON_FINITE = (float("nan"), float("inf"), float("-inf"))


def batch(lengths, *, outside=OUTSIDE, inside=-0.5):
    """A ``(B, T)`` ``(loss_mask, logprobs)`` pair with per-row response lengths.

    Args:
        lengths: Response token count for each row. Rows may differ, so padding columns
            exist and are masked out exactly like the prompt.
        outside: Log-prob written at every masked position -- prompt AND padding. Distinct
            from anything inside the response, so a feature that reads it is visible.
        inside: Per-token log-prob inside the response; a scalar, or one value per row.

    Returns:
        ``(loss_mask, logprobs)``, both ``(B, PROMPT + max(lengths))``.
    """
    n = len(lengths)
    width = PROMPT + max(max(lengths), 1)
    mask = torch.zeros(n, width)
    lp = torch.full((n, width), float(outside))
    ins = list(inside) if isinstance(inside, (list, tuple)) else [inside] * n
    for i, ln in enumerate(lengths):
        mask[i, PROMPT:PROMPT + ln] = 1.0
        lp[i, PROMPT:PROMPT + ln] = float(ins[i])
    return mask, lp


def feats(rewards, lengths, *, group_sizes=None, **kw):
    """Run the real feature computation on a batch built from ``rewards``/``lengths``."""
    mask, lp = batch(lengths, **{k: v for k, v in kw.items() if k in ("outside", "inside")})
    rest = {k: v for k, v in kw.items() if k not in ("outside", "inside")}
    return group_features(
        torch.tensor(rewards, dtype=torch.float32), mask, lp,
        group_sizes or [len(rewards)], **rest,
    )


def one(rewards, lengths, **kw):
    """Features of a single-group batch."""
    return feats(rewards, lengths, **kw)[0]


# ------------------------------------------------------------------------- masking ----


def test_mean_logprob_reads_response_tokens_only():
    """Dropping the mask, or dividing by the sequence length, both change this number."""
    for outside in (-9.0, 0.0, 5.0, -1e6):
        f = one([1.0, 0.0], [4, 4], outside=outside, inside=-0.25)
        assert f.mean_logprob == pytest.approx(-0.25), f"outside={outside} leaked"


def test_a_non_finite_logprob_outside_the_response_cannot_reach_the_mean():
    """``nan * 0 == nan``: multiplying by the mask is not the same as masking.

    A contaminated sum does not show up as a NaN feature -- ``_safe_div`` turns it into
    0.0 -- so this is only visible as a wrong, plausible number.
    """
    for outside in NON_FINITE:
        f = one([1.0, 0.0], [4, 4], outside=outside, inside=-0.25)
        assert f.mean_logprob == pytest.approx(-0.25), f"outside={outside} contaminated"
        assert f.logprob_dispersion == 0.0


def test_no_feature_depends_on_anything_outside_the_response():
    """The whole masked region -- prompt and padding -- swept against a fixed response."""
    ref = one([1.0, 0.0, 1.0], [5, 2, 4], outside=OUTSIDE, inside=-0.5)
    for outside in (-9.0, 0.0, 1.0, 1e9, *NON_FINITE):
        got = one([1.0, 0.0, 1.0], [5, 2, 4], outside=outside, inside=-0.5)
        assert got == ref, f"outside={outside} moved a feature: {got} != {ref}"


def test_mean_logprob_is_a_per_token_mean_not_a_per_row_sum():
    """Rows of different lengths at the same per-token value must agree."""
    short = one([1.0], [2], inside=-0.3)
    long = one([1.0], [7], inside=-0.3)
    assert short.mean_logprob == pytest.approx(long.mean_logprob) == pytest.approx(-0.3)
    assert short.mean_response_len != long.mean_response_len


def test_mean_response_len_counts_mask_tokens_not_columns():
    """Padding must not inflate a length; a batch is as wide as its longest row."""
    assert one([1.0, 1.0], [2, 6]).mean_response_len == pytest.approx(4.0)
    assert one([1.0, 1.0], [2, 2]).mean_response_len == pytest.approx(2.0)


# ---------------------------------------------------------------------- dispersion ----


def test_dispersions_are_population_not_sample():
    """Bessel's correction on a group of four is a 15% error and never NaNs, so a test
    that only checks 'not NaN' cannot tell the two apart."""
    f = one([1.0, 1.0, 0.0, 0.0], [4, 4, 4, 4])
    assert f.reward_std == pytest.approx(0.5)                # sample std would be 0.5774
    g = one([1.0] * 4, [2, 4, 6, 8])
    assert g.len_dispersion == pytest.approx(math.sqrt(5.0) / 5.0)   # pop std 2.2360 / 5


def test_logprob_dispersion_is_population_not_sample():
    """The third dispersion has its own ``std`` call, so it needs its own pin: a sample
    standard deviation over four samples is 15% high and never NaN."""
    f = one([1.0] * 4, [4] * 4, inside=[-0.2, -0.4, -0.6, -0.8])
    assert f.mean_logprob == pytest.approx(-0.5)
    assert f.logprob_dispersion == pytest.approx(math.sqrt(0.05) / 0.5)   # 0.4472, not 0.5164


def test_a_singleton_group_has_zero_dispersion_not_nan():
    """Every dispersion, not only the one that motivated the population std."""
    f = one([1.0], [4])
    for name in ("reward_std", "len_dispersion", "logprob_dispersion"):
        assert getattr(f, name) == 0.0, f"{name} on a singleton: {getattr(f, name)}"
    assert all(math.isfinite(v) for v in f.as_extra().values())


@pytest.mark.parametrize("g", [1, 2, 3, 4, 8])
def test_singleton_groups_inside_a_batch_are_also_finite(g):
    """A batch split entirely into singletons is the degenerate grouping a caller reaches
    for when routing per sample."""
    rewards = [float(i % 2) for i in range(g)]
    for f in feats(rewards, [i + 1 for i in range(g)], group_sizes=[1] * g):
        assert f.reward_std == 0.0
        assert all(math.isfinite(v) for v in f.as_extra().values())


def test_reward_std_is_zero_exactly_when_the_group_is_unanimous():
    """The docstring's claim, swept over every binary pattern plus non-binary rewards.

    Zero must mean 'RL-silent' and nothing else: a group that is unanimous in OUTCOME but
    not in reward still carries advantages, and reporting it as silent would flag a live
    gradient as empty.
    """
    for bits in itertools.product((0.0, 1.0), repeat=4):
        f = one(list(bits), [4] * 4)
        assert (f.reward_std == 0.0) == (len(set(bits)) == 1), bits
    graded = [1.0, 0.8, 1.0, 0.8]                    # unanimous in outcome, not in reward
    f = one(graded, [4] * 4)
    assert f.solve_rate == 1.0 and f.reward_std > 0.0


def test_len_dispersion_is_scale_free():
    """A linear policy over raw lengths would mostly learn the tokenizer, which is why the
    dispersion is a ratio; multiplying every length must leave it alone."""
    base = one([1.0] * 4, [2, 4, 6, 8])
    for k in (2, 3, 5):
        scaled = one([1.0] * 4, [2 * k, 4 * k, 6 * k, 8 * k])
        assert scaled.len_dispersion == pytest.approx(base.len_dispersion), k
        assert scaled.mean_response_len == pytest.approx(k * base.mean_response_len)


def test_logprob_dispersion_divides_by_the_ABSOLUTE_mean():
    """Log-probabilities are negative, so a missing ``abs`` gives a negative dispersion --
    a sign no downstream consumer checks, and a feature that reads backwards."""
    neg = one([1.0] * 4, [4] * 4, inside=[-0.2, -0.4, -0.6, -0.8])
    pos = one([1.0] * 4, [4] * 4, inside=[0.2, 0.4, 0.6, 0.8])
    assert neg.logprob_dispersion > 0.0
    assert neg.logprob_dispersion == pytest.approx(pos.logprob_dispersion)
    assert neg.mean_logprob == pytest.approx(-pos.mean_logprob)


def test_a_uniform_group_has_zero_dispersion_in_every_channel():
    f = one([1.0] * 4, [4] * 4, inside=-0.5)
    assert (f.reward_std, f.len_dispersion, f.logprob_dispersion) == (0.0, 0.0, 0.0)


# -------------------------------------------------------------------- solve rate ------


@pytest.mark.parametrize("thr", [0.0, 0.25, 0.5, 0.9])
def test_solve_rate_counts_rewards_STRICTLY_above_the_threshold(thr):
    """A reward exactly at the threshold is not a solve; an inclusive comparison would
    score every all-zero group as solved at the default threshold of 0."""
    at = one([thr] * 4, [4] * 4, reward_threshold=thr)
    above = one([thr + 0.01] * 4, [4] * 4, reward_threshold=thr)
    assert at.solve_rate == 0.0 and above.solve_rate == 1.0


def test_solve_rate_is_a_fraction_of_the_group():
    for k in range(5):
        f = one([1.0] * k + [0.0] * (4 - k), [4] * 4)
        assert f.solve_rate == pytest.approx(k / 4), k


# ------------------------------------------------------------------- truncation -------


def test_truncated_fraction_counts_rows_that_REACHED_the_budget():
    """The budget is a length a response can attain, so the comparison is inclusive: an
    exclusive one reports 0% truncation on a batch where every row hit the wall."""
    f = one([0.0] * 4, [2, 4, 6, 8], max_response_len=6)
    assert f.truncated_fraction == pytest.approx(0.5)
    assert one([0.0] * 4, [6] * 4, max_response_len=6).truncated_fraction == 1.0
    assert one([0.0] * 4, [5] * 4, max_response_len=6).truncated_fraction == 0.0


def test_an_absent_budget_reports_zero_rather_than_guessing():
    """Guessing the budget from the longest row would report truncation on every batch."""
    f = one([0.0] * 4, [2, 4, 6, 8])
    assert f.truncated_fraction == 0.0


def test_truncation_is_the_feature_a_solve_rate_router_cannot_see():
    """The motivating case: two groups that look identical to a threshold on solve_rate
    and differ in why they failed."""
    out_of_budget = one([0.0] * 4, [8] * 4, max_response_len=8)
    genuinely_wrong = one([0.0] * 4, [3] * 4, max_response_len=8)
    assert out_of_budget.solve_rate == genuinely_wrong.solve_rate == 0.0
    assert out_of_budget.truncated_fraction == 1.0
    assert genuinely_wrong.truncated_fraction == 0.0


# --------------------------------------------------------------------- grouping -------


@pytest.mark.parametrize("sizes", [[2, 2], [1, 3], [3, 1], [1, 1, 1, 1], [4], [2, 1, 1]])
def test_a_group_gets_the_same_features_inside_a_batch_as_on_its_own_rows(sizes):
    """Grouping is checked by recomputation, not by re-deriving the slice arithmetic: a
    test that pins a copy of the expression cannot notice the copy drifting."""
    rewards = [1.0, 0.0, 1.0, 1.0]
    lengths = [2, 5, 3, 8]
    mask, lp = batch(lengths)
    r = torch.tensor(rewards)
    whole = group_features(r, mask, lp, sizes, max_response_len=8)
    start = 0
    for i, g in enumerate(sizes):
        sl = slice(start, start + g)
        start += g
        alone = group_features(r[sl], mask[sl], lp[sl], [g], max_response_len=8)[0]
        assert whole[i] == alone, f"group {i} of {sizes} bled: {whole[i]} != {alone}"


def test_groups_are_returned_in_row_order():
    """Two groups whose every feature differs, so a swap cannot pass."""
    a, b = feats([1.0, 1.0, 0.0, 0.0], [2, 2, 9, 9], group_sizes=[2, 2], max_response_len=9)
    assert (a.solve_rate, a.truncated_fraction) == (1.0, 0.0)
    assert (b.solve_rate, b.truncated_fraction) == (0.0, 1.0)
    assert a.mean_response_len < b.mean_response_len


def test_padding_width_changes_no_feature():
    """A group's features must not depend on how long the batch's longest OTHER row was."""
    narrow = group_features(torch.ones(2), *batch([3, 3]), [2])[0]
    mask, lp = batch([3, 3, 40])
    wide = group_features(torch.ones(3), mask, lp, [2, 1])[0]
    assert narrow == wide


# ------------------------------------------------------------------- validation -------


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(rewards=torch.zeros(4, 1)), "must be 1-D"),
        (dict(loss_mask=torch.ones(4)), "must be 2-D"),
        (dict(logprobs=torch.ones(4)), "must be 2-D"),
        (dict(loss_mask=torch.ones(4, 5, 1)), "must be 2-D"),
        (dict(rewards=torch.zeros(5)), "rows"),
        (dict(logprobs=torch.zeros(4, 9)), "same shape"),
        (dict(group_sizes=[3]), "batch has 4 rows"),
        (dict(group_sizes=[2, 3]), "batch has 4 rows"),
        (dict(group_sizes=[0, 4]), "must be >= 1"),
        (dict(group_sizes=[-1, 5]), "must be >= 1"),
        (dict(max_response_len=0), "max_response_len must be >= 1"),
        (dict(max_response_len=-3), "max_response_len must be >= 1"),
    ],
)
def test_bad_inputs_are_refused_with_a_message_that_names_the_problem(kwargs, needle):
    """A silently wrong grouping attributes one group's features to another, which is
    unrecoverable downstream, so every disagreement is a refusal rather than a guess."""
    mask, lp = batch([4, 4, 4, 4])
    call = dict(rewards=torch.zeros(4), loss_mask=mask, logprobs=lp, group_sizes=[4])
    call.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        group_features(**call)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_reward_is_refused_rather_than_scored(bad):
    """``nan > threshold`` is False, so a NaN reward is silently graded WRONG and makes
    ``reward_std`` NaN. Both are wrong answers to a grader bug; refusing is not."""
    mask, lp = batch([4, 4])
    with pytest.raises(ValueError, match="not finite"):
        group_features(torch.tensor([1.0, bad]), mask, lp, [2])


def test_a_reward_that_is_finite_only_before_the_cast_is_refused():
    """Features are computed in float32. A float64 reward of 1e300 passes ``isfinite`` and
    becomes ``inf`` on the cast, after which ``reward_std`` is NaN -- and a guard that
    replaces a NaN with 0.0 would report this group as UNANIMOUS, which is the opposite of
    what it is. Checking after the cast is what makes 'std == 0 iff unanimous' true."""
    mask, lp = batch([4, 4])
    rewards = torch.tensor([1e300, 0.0], dtype=torch.float64)
    assert bool(torch.isfinite(rewards).all()), "the premise: finite in float64"
    with pytest.raises(ValueError, match="not finite in float32"):
        group_features(rewards, mask, lp, [2])
    ok = group_features(torch.tensor([1e30, 0.0], dtype=torch.float64), mask, lp, [2])[0]
    assert ok.reward_std > 0.0, "a large-but-representable reward is still accepted"


def test_the_partition_check_accepts_every_valid_partition():
    """The guarantee is about the accepted region, not about the two shapes that motivated
    it: a sweep that only rejected would pass with the function rejecting everything."""
    mask, lp = batch([4] * 6)
    r = torch.zeros(6)
    accepted = 0
    for k in range(1, 7):
        for sizes in itertools.product(range(1, 7), repeat=k):
            if sum(sizes) != 6:
                continue
            assert len(group_features(r, mask, lp, list(sizes))) == k
            accepted += 1
    assert accepted == 32, accepted        # compositions of 6


# -------------------------------------------------------------------- finiteness ------


def test_every_field_is_finite_for_every_adversarial_input():
    """The output contract, swept: a non-finite mask, a non-finite log-prob, an empty
    response and a degenerate budget, crossed. One NaN here poisons an arm forever."""
    checked = 0
    for maskval, lpval, ln in itertools.product(
        (0.0, 1.0, *NON_FINITE), (-0.5, *NON_FINITE), ([4, 4], [0, 4], [0, 0]),
    ):
        mask, lp = batch([max(x, 1) for x in ln], outside=lpval, inside=lpval)
        mask = torch.where(mask > 0, torch.tensor(maskval), mask)
        for i, k in enumerate(ln):
            if k == 0:
                mask[i] = 0.0
        for f in group_features(torch.tensor([1.0, 0.0]), mask, lp, [2], max_response_len=4):
            bad = {k: v for k, v in f.as_extra().items() if not math.isfinite(v)}
            assert not bad, f"mask={maskval} lp={lpval} len={ln}: {bad}"
        checked += 1
    assert checked == 5 * 4 * 3


def test_a_zero_length_response_gives_zero_not_nan():
    """An empty mask divides by zero in three places."""
    mask = torch.zeros(2, 6)
    f = group_features(torch.tensor([1.0, 0.0]), mask, torch.full((2, 6), -0.5), [2],
                       max_response_len=4)[0]
    assert (f.mean_response_len, f.len_dispersion) == (0.0, 0.0)
    assert (f.mean_logprob, f.logprob_dispersion) == (0.0, 0.0)
    assert f.truncated_fraction == 0.0
    assert f.solve_rate == 0.5 and f.reward_std == 0.5


@pytest.mark.parametrize(
    "num, den",
    list(itertools.product(
        (0.0, 1.0, -1.0, 1e308, -1e308, float("nan"), float("inf"), float("-inf")),
        (0.0, 1e-13, 1.0, -1.0, 1e-308, float("nan"), float("inf")),
    )),
)
def test_safe_div_never_returns_a_non_finite_value(num, den):
    """Every numerator x denominator pair that can reach it, including the overflow that a
    ``den == 0`` guard alone does not catch."""
    assert math.isfinite(_safe_div(num, den))


def test_safe_div_still_divides():
    """A guard that returned the default unconditionally would pass the sweep above."""
    assert _safe_div(1.0, 4.0) == 0.25
    assert _safe_div(-3.0, 2.0) == -1.5
    assert _safe_div(1.0, 0.0, default=7.0) == 7.0


def test_safe_div_treats_a_near_zero_denominator_as_zero():
    """``den == 0`` alone is not enough: a denominator of 1e-13 is a group whose mean is
    zero to floating-point precision, and dividing by it yields a 1e13 'dispersion' that is
    finite, plausible and meaningless."""
    assert _safe_div(1.0, 1e-13) == 0.0
    assert _safe_div(1.0, -1e-13) == 0.0
    assert _safe_div(1.0, 1e-11) == pytest.approx(1e11), "the guard must not swallow real values"


# -------------------------------------------------------------- the consumer seam -----


def test_feature_names_is_the_dataclass_field_order():
    """The router indexes ``ctx.extra`` by these names in this order; a rename here that
    does not reach FEATURE_NAMES would raise MissingFeatures at the first decision."""
    assert FEATURE_NAMES == tuple(GroupFeatures.__dataclass_fields__)
    assert len(FEATURE_NAMES) == 7
    assert tuple(one([1.0], [4]).as_extra()) == FEATURE_NAMES


def test_as_extra_is_plain_floats():
    extra = one([1.0, 0.0], [4, 6], max_response_len=6).as_extra()
    assert all(type(v) is float for v in extra.values()), extra


def test_the_producer_satisfies_the_contextual_router_without_any_adaptation():
    """The seam this module exists for: features straight from a batch must be routable by
    a router built with ``require_features=True`` and no extra plumbing."""
    router = ContextualBanditRouter(require_features=True)
    mask, lp = batch([2, 5, 8, 8])
    for g in group_features(torch.tensor([1.0, 0.0, 0.0, 1.0]), mask, lp, [2, 2],
                            max_response_len=8):
        ctx = RoutingContext(solve_rate=g.solve_rate, group_size=2, has_teacher=True,
                             extra=g.as_extra())
        d = router.route(ctx)
        assert len(d.weights) == 1 and sum(d.weights.values()) == 1.0
