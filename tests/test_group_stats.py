# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``areal.utils.group_stats``.

Written from the specification alone -- the author of this file has not read the
parallel implementation.  Every test documents the defect it is meant to catch.
Wherever the specification leaves a decision open, the test encodes one reading
and flags it with a ``# SPEC AMBIGUITY:`` comment.

Global conventions assumed by this file (each also flagged at its use site):

* ``n_positive`` counts strictly positive rewards (``x > 0``) *after* trailing
  dims are reduced; ``p_hat = n_positive / size``.
* ``is_silent`` means the group carries no positive/negative contrast, i.e.
  ``p_hat in {0.0, 1.0}``.  A singleton group is therefore always silent.
* ``reward_std`` is the *population* (ddof=0) standard deviation.  This is the
  only convention under which ``between_group_var + within_group_var`` equals
  the total population variance, and the only one recoverable from the stored
  ``(size, reward_std)`` pairs, so it is the reading taken here.
* ``between_group_var`` / ``within_group_var`` are the size-weighted ANOVA
  components (law of total variance), both with ddof=0.
"""

import dataclasses
import json
import math
import pathlib

import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization
from areal.utils.group_stats import GroupStats, GroupStatsRecorder

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GROUP_STATS_FIELDS = (
    "step",
    "group_index",
    "size",
    "n_positive",
    "p_hat",
    "is_silent",
    "reward_mean",
    "reward_std",
)

SUMMARY_KEYS = (
    "n_groups",
    "silent_rate",
    "all_zero_rate",
    "all_one_rate",
    "p_hat_hist",
    "between_group_var",
    "within_group_var",
)

NORM_LEVELS = ["batch", "group", None]


def _t(values) -> torch.Tensor:
    """Deterministic float32 CPU tensor (no RNG anywhere in this file)."""
    return torch.tensor(values, dtype=torch.float32)


def _slices(sizes: list[int]) -> list[slice]:
    """Contiguous slices for the given group sizes, as ``_build_group_slices``."""
    out, offset = [], 0
    for sz in sizes:
        out.append(slice(offset, offset + sz))
        offset += sz
    return out


def _read_jsonl(path) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _make_recorder(tmp_path, enabled: bool = True, name: str = "stats.jsonl"):
    path = tmp_path / name
    return GroupStatsRecorder(out_path=str(path), enabled=enabled), path


def _flushed_records(recorder, path) -> list[dict]:
    """Per-group records, read back through the only channel the spec exposes.

    # SPEC AMBIGUITY: the spec gives no public accessor for the individual
    # ``GroupStats`` objects (only ``summary()``), so every per-group assertion
    # in this file goes through ``flush()`` + the JSONL file.  Always call
    # ``summary()`` *before* this helper: it is unspecified whether ``flush()``
    # also drains the in-memory state that ``summary()`` reads.
    """
    recorder.flush()
    return _read_jsonl(path)


def _hist_counts(hist):
    """Normalise ``p_hat_hist`` (list of counts or dict keyed by bin) to a list.

    # SPEC AMBIGUITY: neither the container type, the number of bins, nor
    # whether the entries are counts or densities is specified.  Only
    # order-and-mass properties that hold for *any* ascending binning of [0, 1]
    # are asserted below (plus one separately flagged count test).
    """
    if isinstance(hist, dict):

        def _key(k):
            try:
                return (0, float(k))
            except (TypeError, ValueError):
                return (1, str(k))

        return [hist[k] for k in sorted(hist, key=_key)]
    return list(hist)


def _is_finite_number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _assert_all_numbers_finite(obj, where: str) -> None:
    for k, v in obj.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            assert math.isfinite(v), f"{where}[{k!r}] is not finite: {v!r}"
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    assert math.isfinite(item), (
                        f"{where}[{k!r}][{i}] is not finite: {item!r}"
                    )


class _SpyRecorder(GroupStatsRecorder):
    """A real recorder that also captures the arguments ``record`` was called with.

    Uses ``*args/**kwargs`` so it works whether the caller passes the arguments
    positionally or by keyword, and delegates to the real implementation so the
    recorder still behaves normally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def record(self, *args, **kwargs):  # type: ignore[override]
        x = args[0] if len(args) > 0 else kwargs.get("x")
        group_slices = args[1] if len(args) > 1 else kwargs.get("group_slices")
        step = args[2] if len(args) > 2 else kwargs.get("step")
        self.calls.append(
            (
                None if x is None else x.detach().clone(),
                None if group_slices is None else list(group_slices),
                step,
            )
        )
        return super().record(*args, **kwargs)


# ---------------------------------------------------------------------------
# 0. surface conformance
# ---------------------------------------------------------------------------


def test_group_stats_is_a_dataclass_with_the_specified_fields():
    """Catches: a ``GroupStats`` that drops/renames a field the spec enumerates
    (e.g. ships only ``p_hat`` and forgets ``n_positive``/``reward_std``), which
    would silently make the JSONL unusable downstream."""
    assert dataclasses.is_dataclass(GroupStats)
    names = {f.name for f in dataclasses.fields(GroupStats)}
    missing = set(GROUP_STATS_FIELDS) - names
    assert not missing, f"GroupStats is missing spec'd fields: {sorted(missing)}"


def test_summary_reports_every_specified_key(tmp_path):
    """Catches: a ``summary()`` that omits one of the seven required keys."""
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(_t([0.0, 1.0, 1.0, 0.0]), _slices([2, 2]), 0)
    summary = recorder.summary()
    assert isinstance(summary, dict)
    missing = set(SUMMARY_KEYS) - set(summary)
    assert not missing, f"summary() is missing spec'd keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 1. THE CRITICAL ONE: instrumentation must be a bitwise no-op
# ---------------------------------------------------------------------------


def _noop_inputs(shape: str):
    """bs=12 in three groups of 4; a 1-D reward vector and a (B, T) variant.

    Deliberately mixes negative, fractional and identical values: an in-place
    ``clamp_``/``abs_``/``round_``/``mul_`` inside the recorder, or a silent
    precision change, then cannot hide behind 0/1-only data.
    """
    if shape == "1d":
        return (
            _t([-1.0, 0.5, 1.0, 0.0, 2.25, -0.75, 1.0, 1.0, 0.0, -0.25, 0.0, 3.5]),
            None,
        )
    x = _t(
        [
            [-1.0, 1.0],
            [0.5, 1.0],
            [1.0, -0.25],
            [0.0, 0.0],
            [2.25, 1.0],
            [1.0, -0.75],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, -0.5],
            [-0.25, 1.0],
            [0.0, 0.0],
            [3.5, 1.0],
        ]
    )
    mask = _t(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    return x, mask


@pytest.mark.parametrize("shape", ["1d", "2d"])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.parametrize(
    "std_level", NORM_LEVELS, ids=["std-batch", "std-group", "std-none"]
)
@pytest.mark.parametrize(
    "mean_level", NORM_LEVELS, ids=["mean-batch", "mean-group", "mean-none"]
)
def test_recorder_is_a_bitwise_noop_on_normalization_output(
    mean_level, std_level, enabled, shape, tmp_path
):
    """THE critical property.  Catches any instrumentation that perturbs the
    numbers it is supposed to only observe: casting ``x`` to float64/float32 in
    place, reusing a recorder-side reduced tensor for the math, reordering the
    mean/std computation around the record call, or short-circuiting a branch
    when a recorder is present.  Compared with ``torch.equal`` (exact values +
    dtype + shape), never ``allclose``.  Also asserts the second call through
    the same instance is identical, so accumulated recorder state cannot leak
    into the math."""
    x, mask = _noop_inputs(shape)
    cfg_kwargs = dict(mean_level=mean_level, std_level=std_level, group_size=4)
    group_sizes = [4, 4, 4]

    # Reference: constructed exactly as before this feature existed.
    baseline = Normalization(NormConfig(**cfg_kwargs))(
        x.clone(), loss_mask=mask, group_sizes=group_sizes
    )

    recorder, _ = _make_recorder(tmp_path, enabled=enabled)
    norm = Normalization(NormConfig(**cfg_kwargs), recorder=recorder)
    first = norm(x.clone(), loss_mask=mask, group_sizes=group_sizes, step=3)
    second = norm(x.clone(), loss_mask=mask, group_sizes=group_sizes, step=4)

    assert first.dtype == baseline.dtype
    assert first.shape == baseline.shape
    assert torch.equal(first, baseline), (
        f"attaching a recorder (enabled={enabled}) changed the normalized output "
        f"for mean_level={mean_level!r}, std_level={std_level!r}, shape={shape}"
    )
    assert torch.equal(second, baseline), "recorder state leaked into a later call"


@pytest.mark.parametrize("std_unbiased", [False, True])
@pytest.mark.parametrize("mean_leave1out", [False, True])
def test_recorder_is_a_bitwise_noop_on_singleton_group_paths(
    mean_leave1out, std_unbiased, tmp_path
):
    """Catches: instrumentation that disturbs the special-cased degenerate
    branches (``group_size == 1`` with leave-one-out mean / unbiased std), which
    are exactly the branches a recorder is most likely to be spliced next to."""
    cfg_kwargs = dict(
        mean_level="group",
        std_level="group",
        group_size=1,
        mean_leave1out=mean_leave1out,
        std_unbiased=std_unbiased,
    )
    x = _t([-1.0, 1.0, 0.5, 0.0, 2.25, -0.75])

    baseline = Normalization(NormConfig(**cfg_kwargs))(x.clone())

    recorder, _ = _make_recorder(tmp_path)
    got = Normalization(NormConfig(**cfg_kwargs), recorder=recorder)(x.clone(), step=0)

    assert torch.equal(got, baseline)


def test_fully_masked_batch_is_still_a_bitwise_noop(tmp_path):
    """Catches: a record call inserted ahead of the all-masked early return that
    crashes or rewrites the returned tensor.

    # SPEC AMBIGUITY: ``__call__`` returns early (before any group slice exists)
    # when ``loss_mask.sum() == 0``, so whether anything is recorded on that path
    # is unspecified.  Only the no-op property is asserted here.
    """
    x = _t([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.zeros_like(x)
    cfg_kwargs = dict(mean_level="group", std_level="group", group_size=2)

    baseline = Normalization(NormConfig(**cfg_kwargs))(x.clone(), loss_mask=mask)
    recorder, _ = _make_recorder(tmp_path)
    got = Normalization(NormConfig(**cfg_kwargs), recorder=recorder)(
        x.clone(), loss_mask=mask, group_sizes=[2, 2], step=0
    )
    assert torch.equal(got, baseline)


# ---------------------------------------------------------------------------
# 2. no mutation of the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["1d", "2d"])
def test_record_does_not_mutate_its_input_tensor(shape, tmp_path):
    """Catches: in-place reduction/normalisation inside ``record`` -- e.g.
    ``x.clamp_(0)``, ``x /= n``, ``x.squeeze_()`` or an in-place dtype cast --
    which would silently corrupt the rewards the caller still needs."""
    # negative and fractional values, so clamp_/abs_/round_/floor_ style
    # in-place damage cannot pass unnoticed
    x = _t([-1.5, 0.0, 2.25, 1.0, -0.5, 0.75])
    if shape == "2d":
        x = x.unsqueeze(1).repeat(1, 3)
    before = x.clone()

    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, _slices([3, 3]), 0)

    assert x.shape == before.shape
    assert x.dtype == before.dtype
    assert torch.equal(x, before), "record() mutated its input tensor"


def test_normalization_with_recorder_does_not_mutate_its_input_tensor(tmp_path):
    """Same defect, observed through the integration point."""
    x = _t([-1.5, 0.0, 2.25, 1.0, -0.5, 0.75])
    before = x.clone()
    recorder, _ = _make_recorder(tmp_path)
    Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=3),
        recorder=recorder,
    )(x, group_sizes=[3, 3], step=1)
    assert torch.equal(x, before)


# ---------------------------------------------------------------------------
# 3. silent-group detection
# ---------------------------------------------------------------------------


def test_silent_group_detection_and_rates(tmp_path):
    """Catches: off-by-one silent detection (e.g. ``n_positive == size - 1``),
    an all-one group not counted as silent, rates divided by the wrong
    denominator, or the singleton groups being dropped instead of counted.

    Groups: [0,0,0] all-zero, [1,1,1] all-one, [0,1,1] mixed, [1] singleton,
    [0] singleton.

    # DECISION (spec is silent on this): a size-1 group IS silent -- it has no
    # peer to contrast against, so its advantage is identically zero.  It also
    # counts toward all_zero_rate / all_one_rate according to its own value, and
    # it counts in the denominator ``n_groups`` like any other group.
    # SPEC AMBIGUITY: all_zero_rate / all_one_rate are taken as fractions of
    # *all* groups (denominator ``n_groups``), not of the silent groups.  With
    # 5 groups / 4 silent / 2 all-zero the two readings give 0.4 vs 0.5.
    """
    x = _t([0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    group_slices = _slices([3, 3, 3, 1, 1])

    recorder, path = _make_recorder(tmp_path)
    recorder.record(x, group_slices, 5)

    summary = recorder.summary()
    assert summary["n_groups"] == 5
    assert summary["silent_rate"] == pytest.approx(4 / 5)
    assert summary["all_zero_rate"] == pytest.approx(2 / 5)
    assert summary["all_one_rate"] == pytest.approx(2 / 5)

    records = _flushed_records(recorder, path)
    assert len(records) == 5
    assert [r["group_index"] for r in records] == [0, 1, 2, 3, 4]
    assert all(r["step"] == 5 for r in records)
    assert [r["size"] for r in records] == [3, 3, 3, 1, 1]
    assert [r["is_silent"] for r in records] == [True, True, False, True, True]
    # SPEC AMBIGUITY: n_positive counts strictly-positive rewards (x > 0).
    assert [r["n_positive"] for r in records] == [0, 3, 2, 1, 0]
    assert [r["p_hat"] for r in records] == pytest.approx([0.0, 1.0, 2 / 3, 1.0, 0.0])
    assert [r["reward_mean"] for r in records] == pytest.approx(
        [0.0, 1.0, 2 / 3, 1.0, 0.0]
    )
    # population (ddof=0) std: sqrt(2)/3 for [0,1,1], 0 everywhere else.
    assert [r["reward_std"] for r in records] == pytest.approx(
        [0.0, 0.0, math.sqrt(2) / 3, 0.0, 0.0], abs=1e-6
    )


def test_a_group_with_no_silent_members_reports_zero_rates(tmp_path):
    """Catches: rates hard-coded / inverted, or a mixed group misclassified as
    silent (which would make the whole instrument report ~100% collapse)."""
    x = _t([0, 1, 1, 0, 0, 1])
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, _slices([3, 3]), 0)
    summary = recorder.summary()
    assert summary["n_groups"] == 2
    assert summary["silent_rate"] == pytest.approx(0.0)
    assert summary["all_zero_rate"] == pytest.approx(0.0)
    assert summary["all_one_rate"] == pytest.approx(0.0)


def test_constant_nonbinary_group_is_silent_but_a_split_positive_group_is_not(
    tmp_path,
):
    """Catches: ``is_silent`` implemented as ``reward_std == 0`` instead of
    ``p_hat in {0, 1}``.

    # SPEC AMBIGUITY: two readings of "silent" coincide on binary rewards and
    # diverge on continuous ones.  This file takes silence to be an absence of
    # positive/negative *contrast* (``p_hat in {0, 1}``), because that is what
    # makes a group's advantages collapse under group-mean centering with a
    # binary verifier.  Under that reading [0.3, 0.7] is silent (both positive)
    # while a ``reward_std == 0`` reading would call it non-silent.
    """
    x = _t([0.25, 0.25, 0.3, 0.7, 0.0, -1.0])
    recorder, path = _make_recorder(tmp_path)
    recorder.record(x, _slices([2, 2, 2]), 0)
    summary = recorder.summary()
    records = _flushed_records(recorder, path)
    assert [r["is_silent"] for r in records] == [True, True, True]
    assert [r["n_positive"] for r in records] == [2, 2, 0]
    assert summary["silent_rate"] == pytest.approx(1.0)
    assert summary["all_one_rate"] == pytest.approx(2 / 3)
    assert summary["all_zero_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 4. variable group sizes
# ---------------------------------------------------------------------------


def test_variable_group_sizes_produce_one_record_per_group(tmp_path):
    """Catches: an implementation that assumes a fixed group size (slicing with
    ``bs // group_size`` or ``x.view(-1, group_size)``), which silently
    mis-attributes rewards to the wrong groups whenever rollouts are filtered."""
    x = _t([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    group_sizes = [3, 5, 2]
    assert sum(group_sizes) == x.numel() == 10

    recorder, path = _make_recorder(tmp_path)
    recorder.record(x, _slices(group_sizes), 7)

    summary = recorder.summary()
    assert summary["n_groups"] == 3
    records = _flushed_records(recorder, path)
    assert len(records) == 3
    assert [r["group_index"] for r in records] == [0, 1, 2]
    assert [r["size"] for r in records] == [3, 5, 2]
    assert [r["n_positive"] for r in records] == [1, 3, 0]
    assert [r["p_hat"] for r in records] == pytest.approx([1 / 3, 3 / 5, 0.0])
    assert [r["reward_mean"] for r in records] == pytest.approx([1 / 3, 3 / 5, 0.0])
    assert [r["is_silent"] for r in records] == [False, False, True]


def test_variable_group_sizes_through_normalization(tmp_path):
    """Same defect at the integration point: the recorder must receive the
    ``group_sizes``-derived slices, not fixed-size ones."""
    x = _t([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    recorder, path = _make_recorder(tmp_path)
    Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=5),
        recorder=recorder,
    )(x.clone(), group_sizes=[3, 5, 2], step=7)

    summary = recorder.summary()
    assert summary["n_groups"] == 3
    records = _flushed_records(recorder, path)
    assert [r["size"] for r in records] == [3, 5, 2]
    assert [r["p_hat"] for r in records] == pytest.approx([1 / 3, 3 / 5, 0.0])


# ---------------------------------------------------------------------------
# 5. shape handling
# ---------------------------------------------------------------------------


def test_1d_and_2d_rewards_with_constant_time_axis_give_identical_stats(tmp_path):
    """Catches: a (B, T) tensor flattened to B*T before slicing (which would
    make groups T times too large and shift every group boundary), or trailing
    dims left unreduced."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    group_slices = _slices([3, 3])

    rec1, p1 = _make_recorder(tmp_path, name="one_d.jsonl")
    rec1.record(_t(values), group_slices, 2)
    s1 = rec1.summary()
    r1 = _flushed_records(rec1, p1)

    rec2, p2 = _make_recorder(tmp_path, name="two_d.jsonl")
    rec2.record(_t(values).unsqueeze(1).repeat(1, 4), group_slices, 2)
    s2 = rec2.summary()
    r2 = _flushed_records(rec2, p2)

    assert r1 == r2, "(B,) and constant-along-T (B,T) rewards gave different records"
    assert _hist_counts(s1["p_hat_hist"]) == _hist_counts(s2["p_hat_hist"])
    for key in SUMMARY_KEYS:
        if key != "p_hat_hist":
            assert s1[key] == pytest.approx(s2[key]), key
    assert [r["size"] for r in r1] == [3, 3]


def test_2d_rewards_are_reduced_over_trailing_dims_by_mean(tmp_path):
    """Catches: reduction by ``sum`` instead of ``mean``, taking ``x[:, 0]`` or
    ``x[:, -1]``, or counting positives *before* the reduction -- all of which
    agree with the mean on constant rows and disagree here.

    Rows [1,1,0], [0,0,0], [1,1,1], [0,1,0] reduce to 2/3, 0, 1, 1/3.
    Counting positives before reduction would give 2 of 6 for group 0 instead of
    1 of 2; summing would give 2, 0, 3, 1.
    """
    x2d = _t([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    x1d = _t([2 / 3, 0.0, 1.0, 1 / 3])
    group_slices = _slices([2, 2])

    rec1, p1 = _make_recorder(tmp_path, name="reduced.jsonl")
    rec1.record(x1d, group_slices, 0)
    r1 = _flushed_records(rec1, p1)

    rec2, p2 = _make_recorder(tmp_path, name="raw2d.jsonl")
    rec2.record(x2d, group_slices, 0)
    r2 = _flushed_records(rec2, p2)

    assert [r["size"] for r in r2] == [2, 2]
    assert [r["n_positive"] for r in r2] == [1, 2]
    assert [r["p_hat"] for r in r2] == pytest.approx([0.5, 1.0])
    assert [r["reward_mean"] for r in r2] == pytest.approx([1 / 3, 2 / 3], abs=1e-6)
    for a, b in zip(r1, r2):
        for field in GROUP_STATS_FIELDS:
            assert a[field] == pytest.approx(b[field], abs=1e-6), field


# ---------------------------------------------------------------------------
# 6. variance components
# ---------------------------------------------------------------------------


def test_variance_components_match_analytic_values(tmp_path):
    """The test most likely to catch a wrong implementation.

    Groups (equal sizes, so weighted and unweighted between-group variance
    coincide and only the ddof convention is under test):

        g0 = [0,0,0,0]  mean 0.0   pop-var 0.00
        g1 = [1,1,1,1]  mean 1.0   pop-var 0.00
        g2 = [0,0,1,1]  mean 0.5   pop-var 0.25

        N = 12, grand mean M = (0 + 4 + 2)/12 = 0.5
        between = sum_g n_g (m_g - M)^2 / N = (4*.25 + 4*.25 + 4*0)/12 = 1/6
        within  = sum_g n_g var_g      / N = (0 + 0 + 4*.25)/12       = 1/12
        total   = 1/6 + 1/12 = 1/4, which is the population variance of the 12
                  values (every value sits 0.5 away from the grand mean).

    Catches: sample (ddof=1) variance anywhere -- the unbiased between-group
    variance of the three group means is 0.25, a 50% error; the unbiased within
    for g2 is 1/3 rather than 0.25 -- as well as swapping the two components,
    dividing by ``n_groups`` instead of ``N``, or centring the group means on
    an unweighted grand mean.
    """
    x = _t([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, _slices([4, 4, 4]), 0)

    summary = recorder.summary()
    assert summary["n_groups"] == 3
    assert summary["silent_rate"] == pytest.approx(2 / 3)
    assert summary["between_group_var"] == pytest.approx(1 / 6, abs=1e-7)
    assert summary["within_group_var"] == pytest.approx(1 / 12, abs=1e-7)
    # law of total variance -- must hold for a self-consistent decomposition
    total = float(x.var(unbiased=False))
    assert total == pytest.approx(0.25, abs=1e-7)
    assert summary["between_group_var"] + summary["within_group_var"] == pytest.approx(
        total, abs=1e-7
    )


def test_variance_components_are_size_weighted(tmp_path):
    """Catches: between/within computed as unweighted averages over groups,
    which is wrong whenever group sizes differ (the common case once rollouts
    are filtered).

        g0 = [0,0]      n=2  mean 0.00  pop-var 0.0000
        g1 = [1,1,1,0]  n=4  mean 0.75  pop-var 0.1875

        N = 6, grand mean M = 3/6 = 0.5
        between = (2*(0-.5)^2 + 4*(.75-.5)^2)/6 = (0.5 + 0.25)/6 = 0.125
        within  = (2*0 + 4*0.1875)/6            = 0.75/6          = 0.125
        between + within = 0.25 = population variance of [0,0,1,1,1,0]

    An unweighted between-group variance would report 0.140625, and an
    unweighted within would report 0.09375 -- both outside the tolerance.

    # SPEC AMBIGUITY: the spec does not say whether the components are weighted
    # by group size.  Size weighting is assumed here because it is the only
    # choice for which the two components sum to the total population variance.
    """
    x = _t([0, 0, 1, 1, 1, 0])
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, _slices([2, 4]), 0)

    summary = recorder.summary()
    assert summary["between_group_var"] == pytest.approx(0.125, abs=1e-7)
    assert summary["within_group_var"] == pytest.approx(0.125, abs=1e-7)
    assert summary["between_group_var"] + summary["within_group_var"] == pytest.approx(
        float(x.var(unbiased=False)), abs=1e-7
    )


def test_all_silent_batch_has_zero_within_group_variance(tmp_path):
    """Catches: a within-group term that never reaches 0 (e.g. an eps floor or a
    std/var mix-up), which would hide exactly the collapse this instrument
    exists to detect.  All groups constant => within = 0, between = total."""
    x = _t([0, 0, 0, 1, 1, 1])
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, _slices([3, 3]), 0)
    summary = recorder.summary()
    assert summary["within_group_var"] == pytest.approx(0.0, abs=1e-9)
    assert summary["between_group_var"] == pytest.approx(0.25, abs=1e-7)
    assert summary["silent_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. + 8. flush
# ---------------------------------------------------------------------------


def test_flush_without_out_path_is_a_noop(tmp_path, monkeypatch):
    """Catches: a ``flush`` that raises (``open(None)``) or quietly drops a file
    in the process cwd when no path was configured."""
    monkeypatch.chdir(tmp_path)
    for recorder in (
        GroupStatsRecorder(out_path=None, enabled=True),
        GroupStatsRecorder(enabled=True),  # default out_path
    ):
        recorder.record(_t([0.0, 1.0, 1.0, 0.0]), _slices([2, 2]), 0)
        recorder.flush()
        recorder.flush()
        # SPEC AMBIGUITY: out_path=None is read as "in-memory only" -- it
        # disables writing, not recording, so summary() still sees the groups.
        assert recorder.summary()["n_groups"] == 2
    assert sorted(child.name for child in tmp_path.iterdir()) == []


def test_flush_writes_one_json_object_per_group_and_appends(tmp_path):
    """Catches: JSON-array-per-flush instead of JSONL, one line per *step*
    instead of per group, a ``"w"`` open mode that truncates earlier steps away,
    and a flush that re-emits already-written records (duplicating history)."""
    path = tmp_path / "stats.jsonl"
    recorder = GroupStatsRecorder(out_path=str(path), enabled=True)

    recorder.record(_t([0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), _slices([3, 3]), 0)
    recorder.flush()
    first_bytes = path.read_bytes()
    lines = _read_jsonl(path)
    assert len(lines) == 2
    for i, obj in enumerate(lines):
        assert isinstance(obj, dict)
        missing = set(GROUP_STATS_FIELDS) - set(obj)
        assert not missing, f"record is missing spec'd fields: {sorted(missing)}"
        assert obj["group_index"] == i
        assert obj["step"] == 0
        assert obj["size"] == 3

    recorder.record(
        _t([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]), _slices([3, 3, 3]), 1
    )
    recorder.flush()

    assert path.read_bytes().startswith(first_bytes), (
        "second flush() truncated or rewrote what the first one had written"
    )
    lines = _read_jsonl(path)
    # SPEC AMBIGUITY: "one object per group record" is read as *exactly* one --
    # a flush that re-writes the whole buffer would yield 7 lines here.
    assert len(lines) == 5
    assert [o["step"] for o in lines] == [0, 0, 1, 1, 1]
    assert [o["group_index"] for o in lines] == [0, 1, 0, 1, 2]


def test_flushed_records_are_plain_json_types(tmp_path):
    """Catches: torch tensors / numpy scalars leaking into the record, which
    makes ``json.dumps`` raise (or silently emit ``NaN``, which is not valid
    JSON and breaks strict parsers downstream)."""
    recorder, path = _make_recorder(tmp_path)
    recorder.record(_t([0.0, 1.0, 1.0]), _slices([1, 2]), 3)
    records = _flushed_records(recorder, path)
    assert len(records) == 2
    for obj in records:
        _assert_all_numbers_finite(obj, "record")
        assert isinstance(obj["size"], int)
        assert isinstance(obj["n_positive"], int)
        assert isinstance(obj["is_silent"], bool)
        assert _is_finite_number(obj["p_hat"])
        assert _is_finite_number(obj["reward_mean"])
        assert _is_finite_number(obj["reward_std"])
    # the file itself must be strictly parseable line by line
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip():
            json.loads(line)


# ---------------------------------------------------------------------------
# 9. reset
# ---------------------------------------------------------------------------


def test_reset_clears_accumulated_state(tmp_path):
    """Catches: a ``reset`` that only zeroes a counter while leaving the record
    buffer in place (so the next flush re-emits the previous step's groups), or
    one that leaves ``summary()`` reporting stale rates."""
    path = tmp_path / "stats.jsonl"
    recorder = GroupStatsRecorder(out_path=str(path), enabled=True)
    recorder.record(_t([0.0, 0.0, 1.0, 1.0]), _slices([2, 2]), 0)
    assert recorder.summary()["n_groups"] == 2

    recorder.reset()

    summary = recorder.summary()
    assert summary["n_groups"] == 0
    # SPEC AMBIGUITY: an empty recorder reports 0.0 rates/variances rather than
    # None or NaN, and summary() must not raise on a zero denominator.
    for key in (
        "silent_rate",
        "all_zero_rate",
        "all_one_rate",
        "between_group_var",
        "within_group_var",
    ):
        assert summary[key] == pytest.approx(0.0), key
    _assert_all_numbers_finite(summary, "summary")
    assert sum(_hist_counts(summary["p_hat_hist"])) == pytest.approx(0.0)

    recorder.flush()
    assert _read_jsonl(path) == [], "reset() left records behind for flush() to emit"

    # SPEC AMBIGUITY: reset() clears data, not configuration -- the recorder is
    # still enabled and still usable afterwards.
    recorder.record(_t([1.0, 0.0]), _slices([2]), 1)
    assert recorder.summary()["n_groups"] == 1


# ---------------------------------------------------------------------------
# 10. disabled recorder
# ---------------------------------------------------------------------------


def test_disabled_recorder_records_nothing(tmp_path):
    """Catches: the enabled flag checked only at the Normalization call site, so
    a directly-reachable ``record`` still accumulates (and still writes files) --
    the instrument would then cost memory/IO in every production run."""
    path = tmp_path / "stats.jsonl"
    recorder = GroupStatsRecorder(out_path=str(path), enabled=False)

    recorder.record(_t([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), _slices([3, 3]), 0)
    assert recorder.summary()["n_groups"] == 0

    Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=3),
        recorder=recorder,
    )(_t([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), group_sizes=[3, 3], step=1)
    assert recorder.summary()["n_groups"] == 0

    recorder.flush()
    assert _read_jsonl(path) == []
    assert not path.exists() or path.stat().st_size == 0


def test_default_constructed_recorder_is_disabled(tmp_path):
    """Catches: ``enabled`` defaulting to True, which would turn the instrument
    on for every existing caller."""
    recorder = GroupStatsRecorder()
    recorder.record(_t([0.0, 1.0]), _slices([2]), 0)
    assert recorder.summary()["n_groups"] == 0


# ---------------------------------------------------------------------------
# integration: what Normalization hands to the recorder
# ---------------------------------------------------------------------------


def test_normalization_records_raw_rewards_not_normalized_ones(tmp_path):
    """Catches: the record call placed *after* the normalization math.  Under
    group mean/std normalization every group's output is exactly zero-mean, so a
    late recorder would report reward_mean == 0 and roughly half the samples
    positive for every group -- a plausible-looking but meaningless instrument."""
    x = _t([1, 1, 0, 0, 1, 0, 0, 0])
    recorder, path = _make_recorder(tmp_path)
    out = Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=4),
        recorder=recorder,
    )(x.clone(), group_sizes=[4, 4], step=11)

    summary = recorder.summary()
    assert summary["n_groups"] == 2
    records = _flushed_records(recorder, path)
    assert [r["step"] for r in records] == [11, 11]
    assert [r["n_positive"] for r in records] == [2, 1]
    assert [r["p_hat"] for r in records] == pytest.approx([0.5, 0.25])
    assert [r["reward_mean"] for r in records] == pytest.approx([0.5, 0.25])
    # sanity: the normalized output really is zero-mean per group, which is why
    # the assertions above discriminate.
    assert float(out[:4].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out[4:].mean()) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "group_sizes,expected",
    [
        ([3, 5, 4], [(0, 3), (3, 8), (8, 12)]),
        (None, [(0, 4), (4, 8), (8, 12)]),
    ],
    ids=["variable-sizes", "fixed-group-size"],
)
def test_normalization_passes_the_built_slices_and_step_to_record(
    group_sizes, expected, tmp_path
):
    """Catches: the recorder being handed the raw batch (no slicing), slices
    rebuilt from a stale ``group_size``, the step silently dropped, or ``record``
    invoked more than once per call."""
    x = _t([1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    spy = _SpyRecorder(out_path=str(tmp_path / "spy.jsonl"), enabled=True)
    Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=4), recorder=spy
    )(x.clone(), group_sizes=group_sizes, step=42)

    assert len(spy.calls) == 1, f"record() called {len(spy.calls)} times, expected once"
    recorded_x, slices, step = spy.calls[0]
    assert slices is not None
    assert [(s.start, s.stop) for s in slices] == expected
    assert step == 42
    assert torch.equal(recorded_x, x), "record() received something other than raw x"


def test_recorder_fires_under_batch_level_normalization(tmp_path):
    """# SPEC AMBIGUITY: ``Normalization`` today builds group slices only when
    ``mean_level`` or ``std_level`` is ``"group"``.  The spec says the recorder
    runs "after slices are built", which leaves batch-level configs undefined.
    This file assumes the instrument is orthogonal to the normalization level --
    an enabled recorder must produce group stats under batch normalization too,
    otherwise the silent-group measurement is unavailable in exactly the configs
    it is most needed for.  Catches: recording gated behind the group-norm
    branch."""
    x = _t([1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    recorder, path = _make_recorder(tmp_path)
    Normalization(
        NormConfig(mean_level="batch", std_level="batch", group_size=4),
        recorder=recorder,
    )(x.clone(), group_sizes=[4, 4, 4], step=0)

    summary = recorder.summary()
    assert summary["n_groups"] == 3
    records = _flushed_records(recorder, path)
    assert [r["size"] for r in records] == [4, 4, 4]


def test_recorder_tolerates_a_missing_step(tmp_path):
    """Catches: an implementation that assumes ``step`` is an int and crashes
    (or writes non-JSON) when ``__call__`` is invoked without one, which is the
    default for every existing caller.

    # SPEC AMBIGUITY: ``__call__``'s ``step`` defaults to None while
    # ``GroupStats.step`` is nominally an int; None / -1 / 0 are all accepted
    # here as the stored value.
    """
    x = _t([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    recorder, path = _make_recorder(tmp_path)
    Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=3),
        recorder=recorder,
    )(x.clone(), group_sizes=[3, 3])

    assert recorder.summary()["n_groups"] == 2
    records = _flushed_records(recorder, path)
    assert len(records) == 2
    assert all(r["step"] in (None, -1, 0) for r in records)


def test_records_accumulate_across_steps(tmp_path):
    """Catches: each ``record`` call overwriting the previous one, so
    ``summary()`` only ever describes the latest step."""
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(_t([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), _slices([3, 3]), 0)
    recorder.record(
        _t([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), _slices([3, 3, 3]), 1
    )
    summary = recorder.summary()
    assert summary["n_groups"] == 5
    # groups: [0,0,1] mixed, [1,1,1] silent/one, [0,1,0] mixed, [0,0,0] silent/zero,
    #         [1,1,1] silent/one  -> 3 silent, 1 all-zero, 2 all-one
    assert summary["silent_rate"] == pytest.approx(3 / 5)
    assert summary["all_zero_rate"] == pytest.approx(1 / 5)
    assert summary["all_one_rate"] == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


def test_degenerate_groups_produce_finite_stats(tmp_path):
    """Catches: ``torch.std`` used with its default ``correction=1``, which
    returns NaN for a size-1 group -- NaN then poisons ``summary()`` and is not
    representable in JSON.

    # SPEC AMBIGUITY: reward_std of a singleton group is taken to be 0.0
    # (population convention), not NaN and not 1.0.
    """
    x = _t([1.0, 0.0, 0.5, 0.5])
    recorder, path = _make_recorder(tmp_path)
    recorder.record(x, _slices([1, 1, 2]), 0)

    summary = recorder.summary()
    _assert_all_numbers_finite(summary, "summary")
    records = _flushed_records(recorder, path)
    assert [r["size"] for r in records] == [1, 1, 2]
    assert [r["reward_std"] for r in records] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-9
    )
    assert [r["reward_mean"] for r in records] == pytest.approx([1.0, 0.0, 0.5])
    assert [r["is_silent"] for r in records] == [True, True, True]
    for obj in records:
        _assert_all_numbers_finite(obj, "record")


def test_empty_group_slice_is_tolerated(tmp_path):
    """Catches: a 0/0 division on an empty group producing NaN, or an
    IndexError that takes down the training step.

    # SPEC AMBIGUITY: "tolerates degenerate groups" does not say whether an
    # empty slice is skipped or recorded.  ``_build_group_slices`` rejects
    # non-positive sizes, so empty slices cannot arrive from Normalization;
    # this test only requires that nothing raises and that no NaN/inf escapes.
    """
    x = _t([1.0, 0.0, 1.0, 1.0])
    recorder, path = _make_recorder(tmp_path)
    recorder.record(x, [slice(0, 2), slice(2, 2), slice(2, 4)], 0)

    summary = recorder.summary()
    _assert_all_numbers_finite(summary, "summary")
    assert summary["n_groups"] in (2, 3)
    for obj in _flushed_records(recorder, path):
        _assert_all_numbers_finite(obj, "record")


def test_single_group_covering_the_whole_batch(tmp_path):
    """Catches: a between/within decomposition that divides by ``n_groups - 1``
    -- with one group that is a division by zero.  With a single group all of
    the variance is within-group and none is between."""
    x = _t([0.0, 1.0, 1.0, 0.0])
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(x, [slice(0, 4)], 0)
    summary = recorder.summary()
    assert summary["n_groups"] == 1
    assert summary["between_group_var"] == pytest.approx(0.0, abs=1e-9)
    assert summary["within_group_var"] == pytest.approx(0.25, abs=1e-7)


# ---------------------------------------------------------------------------
# p_hat histogram
# ---------------------------------------------------------------------------


def test_p_hat_hist_places_mass_in_the_extreme_bins(tmp_path):
    """Catches: a histogram whose bins run the wrong way, that clips p_hat == 1
    out of the top bin (a classic ``int(p * n_bins)`` off-by-one that drops
    every fully-solved group), or that is keyed by something other than p_hat.
    Deliberately agnostic to bin count and to counts-vs-density."""
    zeros, _ = _make_recorder(tmp_path, name="zeros.jsonl")
    zeros.record(_t([0.0] * 9), _slices([3, 3, 3]), 0)
    counts = _hist_counts(zeros.summary()["p_hat_hist"])
    assert len(counts) >= 2, "a p_hat histogram needs more than one bin"
    assert counts[0] > 0
    assert sum(counts[1:]) == pytest.approx(0.0)

    ones, _ = _make_recorder(tmp_path, name="ones.jsonl")
    ones.record(_t([1.0] * 9), _slices([3, 3, 3]), 0)
    counts = _hist_counts(ones.summary()["p_hat_hist"])
    assert counts[-1] > 0, "p_hat == 1.0 fell outside the top histogram bin"
    assert sum(counts[:-1]) == pytest.approx(0.0)

    mixed, _ = _make_recorder(tmp_path, name="mixed.jsonl")
    mixed.record(_t([0.0, 0.0, 1.0, 1.0]), _slices([2, 2]), 0)
    counts = _hist_counts(mixed.summary()["p_hat_hist"])
    assert counts[0] > 0 and counts[-1] > 0


def test_p_hat_hist_counts_every_group_exactly_once(tmp_path):
    """# SPEC AMBIGUITY: ``p_hat_hist`` is assumed to hold integer counts that
    sum to ``n_groups`` (not normalised densities, and with no group dropped for
    landing on a bin edge).  Catches a histogram that silently loses groups."""
    recorder, _ = _make_recorder(tmp_path)
    recorder.record(_t([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]), _slices([3, 3, 3, 3]), 0)
    summary = recorder.summary()
    counts = _hist_counts(summary["p_hat_hist"])
    assert sum(counts) == summary["n_groups"] == 4
