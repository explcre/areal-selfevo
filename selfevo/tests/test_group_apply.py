"""The decision -> advantage seam, swept over the masks, modes and dtypes it will meet.

``apply_decisions`` is the only place a routing decision becomes a number in the tensor the
loss reads, so a defect here is invisible everywhere else: the routers still pass their own
tests, the run still trains, and the arm reports a reach it never had. These tests therefore
assert on the RETURNED TENSOR, and derive the expected statistics FROM that tensor rather
than from the expression the implementation uses -- a stats check written as a copy of the
implementation's own arithmetic cannot notice the arithmetic being wrong.

Two things are swept rather than spot-checked, because both have been the hiding place for a
defect in this project before: the mask shape (all-ones, prompted, ragged, an all-zero row,
an all-zero mask) and the group partition (random partitions, not only the tidy 4+4).

One test drives the REAL ``PPOActor._compute_advantages``. The property that matters for M22
is not internal consistency: it is that routing a silent group to SFT produces the same
tensor the actor's hardcoded rule already produces, so moving the actor onto this seam is a
refactor on silent groups and a decision only on informative ones.
"""

from __future__ import annotations

import pathlib
import random
import subprocess
import sys

import pytest
import torch

from selfevo.integration.group_apply import ApplyStats, apply_decisions
from selfevo.routing.base import TrainingMode, known_modes

# The actor fixture is IMPORTED rather than rebuilt: it already mirrors the live config
# (group-level reward norm, adv_norm off), and a second copy of it here would drift from the
# first without either copy failing.
from selfevo.tests.test_group_routing import (
    B,
    G,
    MIXED,
    PROMPT,
    T,
    advantages,
    make_actor,
)

from areal.api.cli_args import GroupRoutingConfig

MODES = [TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP]
WEIGHTS = [0.0, 0.1, 0.5, 1.0, 7.5]


def masks(b: int, t: int, prompt: int = 2) -> dict[str, torch.Tensor]:
    """The mask shapes a real batch produces, plus the degenerate ones.

    Args:
        b: Rows.
        t: Sequence length.
        prompt: Columns that are prompt in the ``prompted`` mask.

    Returns:
        ``{name: (b, t) float mask}``. ``ragged`` gives every row a different response start,
        which is what a real batch looks like; ``one-zero-row`` contains a row that can
        receive no gradient at all; ``all-zero`` is the batch where every decision is a no-op.
    """
    prompted = torch.zeros(b, t)
    prompted[:, prompt:] = 1.0
    ragged = torch.zeros(b, t)
    for i in range(b):
        ragged[i, (i % max(t - 1, 1)) + 1 :] = 1.0
    one_zero_row = prompted.clone()
    one_zero_row[0] = 0.0
    return {
        "all-ones": torch.ones(b, t),
        "prompted": prompted,
        "ragged": ragged,
        "one-zero-row": one_zero_row,
        "all-zero": torch.zeros(b, t),
    }


def informative(b: int = 4, t: int = 6, prompt: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """A group RL can already learn from: non-zero advantages, prompt columns included.

    The prompt columns are non-zero on purpose. That is what the actor's GAE actually leaves
    there -- it carries ``lastgaelam`` backwards through the masked positions -- so a test
    using an all-zero prompt region would not notice a write that clobbers it.
    """
    lm = torch.zeros(b, t)
    lm[:, prompt:] = 1.0
    adv = torch.arange(1, b * t + 1, dtype=torch.float32).reshape(b, t) / 10.0
    return adv, lm


def silent(b: int = 4, t: int = 6, prompt: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """A group GRPO cannot learn from: every advantage exactly zero."""
    lm = torch.zeros(b, t)
    lm[:, prompt:] = 1.0
    return torch.zeros(b, t), lm


def random_partition(rng: random.Random, b: int) -> list[int]:
    """A random partition of ``b`` rows into groups, so the tests never only see 4 + 4."""
    sizes, left = [], b
    while left:
        g = rng.randint(1, left)
        sizes.append(g)
        left -= g
    return sizes


# ------------------------------------------------------------------------- semantics ---


@pytest.mark.parametrize("name", list(masks(4, 6)))
def test_rl_returns_the_input_values_untouched(name):
    """RL is the identity on values. If it is not, every rollback claim is void."""
    adv, _ = informative()
    out, stats = apply_decisions(adv, masks(4, 6)[name], [4], [TrainingMode.RL], sft_weight=0.5)
    assert torch.equal(out, adv), (out - adv).abs().max()
    assert stats.changed_rows == 0


def test_sft_replaces_the_response_advantages_rather_than_adding_to_them():
    """The semantic the mode exists for: 'train this unit by SFT INSTEAD OF RL'.

    Adding would leave the RL gradient in place and superimpose a supervised one, which is a
    different method -- and one that no test keyed on a silent group can tell apart, because
    there the two coincide.
    """
    adv, lm = informative()
    w = 0.5
    out, _ = apply_decisions(adv, lm, [4], [TrainingMode.SFT], sft_weight=w)
    response = lm.bool()
    assert torch.equal(out[response], torch.full_like(out[response], w)), out
    added = adv + w * lm
    assert not torch.allclose(out, added), "SFT added to the RL advantages instead of replacing"


def test_on_a_silent_group_replace_and_add_coincide():
    """The premise for the actor-consistency test below."""
    adv, lm = silent()
    w = 0.5
    out, _ = apply_decisions(adv, lm, [4], [TrainingMode.SFT], sft_weight=w)
    assert torch.equal(out, adv + w * lm)


def test_skip_zeroes_the_response_advantages():
    adv, lm = informative()
    out, _ = apply_decisions(adv, lm, [4], [TrainingMode.SKIP], sft_weight=0.5)
    assert torch.equal(out[lm.bool()], torch.zeros_like(out[lm.bool()])), out
    assert torch.equal(out[~lm.bool()], adv[~lm.bool()]), out


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_a_weight_valued_mask_scales_the_write_as_the_actor_scales_its_constant(scale):
    """``loss_mask`` is 0/1 today, so multiplying by it and writing through a boolean of it
    agree -- and a test that only ever sees 0/1 cannot tell the two apart (a mutant that
    dropped the multiply survived until this test existed). The actor writes
    ``row_adv * loss_mask``, which scales with the mask value; pinning the seam to the same
    arithmetic keeps the two interchangeable if a weighted mask ever arrives, instead of
    silently changing the magnitude of every SFT write on the day it does.
    """
    adv, lm = silent()
    lm, w = lm * scale, 0.5
    out, _ = apply_decisions(adv, lm, [4], [TrainingMode.SFT], sft_weight=w)
    assert torch.equal(out, adv + w * lm), out
    assert float(out[0, -1]) == w * scale


def test_skip_is_not_rl_and_rl_is_not_skip():
    """Swapping the two branches must not pass: on an informative group they differ."""
    adv, lm = informative()
    skipped, _ = apply_decisions(adv, lm, [4], [TrainingMode.SKIP], sft_weight=0.0)
    kept, _ = apply_decisions(adv, lm, [4], [TrainingMode.RL], sft_weight=0.0)
    assert not torch.equal(skipped, kept)


def test_a_decision_reaches_only_its_own_group():
    """Three groups, one decision: the other two must come back bit-identical."""
    adv = torch.arange(1, 6 * 4 + 1, dtype=torch.float32).reshape(6, 4) / 10.0
    lm = torch.ones(6, 4)
    out, stats = apply_decisions(
        adv, lm, [2, 2, 2], [TrainingMode.RL, TrainingMode.SFT, TrainingMode.RL], sft_weight=0.5
    )
    assert torch.equal(out[:2], adv[:2])
    assert torch.equal(out[4:], adv[4:])
    assert torch.equal(out[2:4], torch.full((2, 4), 0.5))
    assert stats.changed_rows == 2


@pytest.mark.parametrize("w", WEIGHTS)
def test_the_written_magnitude_is_the_configured_weight(w):
    """A weight that never reaches the tensor would still pass a presence check."""
    adv, lm = silent()
    out, _ = apply_decisions(adv, lm, [4], [TrainingMode.SFT], sft_weight=w)
    assert torch.equal(out[lm.bool()], torch.full((int(lm.sum()),), w))


# ------------------------------------------------------------------ the mask bounds ----


@pytest.mark.parametrize("name", list(masks(4, 6)))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("w", [0.0, 0.5, 7.5])
def test_the_mask_bounds_every_write(name, mode, w):
    """Nothing outside the mask moves, in either direction.

    Two failures hide here. Writing a non-zero value on a prompt token puts gradient where
    the model was never asked to generate; writing a ZERO there is invisible to the loss but
    erases the GAE value the actor left behind and makes ``changed_rows`` count a row whose
    gradient did not move.
    """
    lm = masks(4, 6)[name]
    adv, _ = informative()
    out, _ = apply_decisions(adv, lm, [2, 2], [mode, mode], sft_weight=w)
    off, on = lm == 0, lm.bool()
    assert torch.equal(out[off], adv[off]), (out - adv).abs().max()
    if mode == TrainingMode.RL:
        assert torch.equal(out, adv)
    elif mode == TrainingMode.SFT:
        assert torch.equal(out[on], torch.full_like(out[on], w))
    else:
        assert torch.equal(out[on], torch.zeros_like(out[on]))


def test_a_fully_masked_row_is_neither_written_nor_counted():
    """A row that can receive no gradient is not reach, whatever the decision says."""
    adv = torch.full((4, 6), 7.0)
    lm = torch.zeros(4, 6)
    lm[1:, 2:] = 1.0                      # row 0 is entirely masked out
    for mode in (TrainingMode.SFT, TrainingMode.SKIP):
        out, stats = apply_decisions(adv, lm, [4], [mode], sft_weight=0.5)
        assert torch.equal(out[0], adv[0]), out[0]
        assert stats.changed_rows == 3, (mode, stats)


def test_an_all_zero_mask_makes_every_decision_a_no_op():
    adv, _ = informative()
    for mode in MODES:
        out, stats = apply_decisions(adv, torch.zeros(4, 6), [4], [mode], sft_weight=0.5)
        assert torch.equal(out, adv), mode
        assert stats.changed_rows == 0, mode


# --------------------------------------------------------------------- no in-place ----


@pytest.mark.parametrize("mode", MODES)
def test_the_caller_s_tensor_is_not_modified(mode):
    """By identity, by storage and by value: the caller may still hold the original."""
    adv, lm = informative()
    snapshot = adv.clone()
    out, _ = apply_decisions(adv, lm, [4], [mode], sft_weight=0.5)
    assert out is not adv
    assert out.data_ptr() != adv.data_ptr()
    assert torch.equal(adv, snapshot), (adv - snapshot).abs().max()


# -------------------------------------------------------------------- changed_rows ----


@pytest.mark.parametrize(
    "mode, w, tensor, expected",
    [
        (TrainingMode.SKIP, 0.5, "silent", 0),        # already zero: nothing to zero
        (TrainingMode.SFT, 0.0, "silent", 0),         # writes the value already there
        (TrainingMode.SFT, 0.5, "silent", 4),         # a real intervention
        (TrainingMode.SKIP, 0.0, "informative", 4),
        (TrainingMode.SFT, 0.5, "informative", 4),
        (TrainingMode.RL, 0.5, "informative", 0),
    ],
    ids=["skip-silent", "sft0-silent", "sft-silent", "skip-info", "sft-info", "rl-info"],
)
def test_changed_rows_counts_interventions_not_decisions(mode, w, tensor, expected):
    """A no-op must count zero, or the reported reach is inflated by decisions that did
    nothing -- the specific error this field exists to avoid."""
    adv, lm = silent() if tensor == "silent" else informative()
    _, stats = apply_decisions(adv, lm, [4], [mode], sft_weight=w)
    assert stats.changed_rows == expected, stats


def test_changed_rows_counts_rows_not_groups():
    """Half a group moving is half a group's rows, not one group."""
    adv, lm = silent()
    adv = adv.clone()
    adv[2:, 2:] = 1.0                     # rows 2 and 3 are informative, rows 0 and 1 silent
    _, stats = apply_decisions(adv, lm, [4], [TrainingMode.SKIP], sft_weight=0.0)
    assert stats.changed_rows == 2, stats


def test_changed_rows_ignores_a_difference_the_loss_cannot_see():
    """Rows silent where the loss reads them, non-zero only outside the mask.

    Zeroing the prompt columns is a change to the tensor and no change to the update. Counted
    as reach, it would report a fully-covered batch for a batch nothing happened in.
    """
    lm = torch.zeros(4, 6)
    lm[:, 2:] = 1.0
    adv = torch.zeros(4, 6)
    adv[:, :2] = 3.0
    for mode, w in ((TrainingMode.SKIP, 0.0), (TrainingMode.SFT, 0.0)):
        _, stats = apply_decisions(adv, lm, [4], [mode], sft_weight=w)
        assert stats.changed_rows == 0, (mode, stats)


# --------------------------------------------------------------------- validation -----


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 4), group_sizes=[4],
              modes=["rl"], sft_weight=0.0), "same shape"),
        (dict(advantages=torch.zeros(4), loss_mask=torch.ones(4), group_sizes=[4],
              modes=["rl"], sft_weight=0.0), r"must be \(B, T\)"),
        (dict(advantages=torch.zeros(4, 3, 2), loss_mask=torch.ones(4, 3, 2), group_sizes=[4],
              modes=["rl"], sft_weight=0.0), r"must be \(B, T\)"),
        (dict(advantages=torch.zeros(4, 3, dtype=torch.int64),
              loss_mask=torch.ones(4, 3), group_sizes=[4], modes=["sft"], sft_weight=0.7),
         "floating point"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[3],
              modes=["rl"], sft_weight=0.0), "sums to 3"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[2, 2],
              modes=["rl"], sft_weight=0.0), "one decision per group"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[-1, 5],
              modes=["skip", "rl"], sft_weight=0.0), "must be >= 1"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[0, 4],
              modes=["skip", "rl"], sft_weight=0.0), "must be >= 1"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[4],
              modes=["sft"], sft_weight=-0.1), "must be >= 0"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[4],
              modes=["sft"], sft_weight=float("nan")), "must be finite"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[4],
              modes=["sft"], sft_weight=float("inf")), "must be finite"),
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[4],
              modes=["banana"], sft_weight=0.0), "cannot apply modes"),
    ],
    ids=["shape", "1d", "3d", "int-advantages", "sizes-sum", "modes-len", "negative-size",
         "zero-size", "negative-weight", "nan-weight", "inf-weight", "unknown-mode"],
)
def test_invalid_input_is_refused(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        apply_decisions(
            kwargs["advantages"], kwargs["loss_mask"], kwargs["group_sizes"],
            kwargs["modes"], sft_weight=kwargs["sft_weight"],
        )


def test_the_seam_and_the_mode_registry_cannot_drift():
    """``_APPLIED`` exists so an unimplemented mode cannot reach the update. It was a
    hand-maintained literal, so a mode registered without a branch here stayed fully
    routable: it cost a whole rollout and then killed the run at the seam.

    Applicability is now declared on the registry, and this equality is what keeps the two
    honest. It is checked at import in ``group_apply`` as well; asserting it here is what
    makes a future ``register_mode`` without a branch fail in the suite rather than only in
    whoever imports the seam next.
    """
    from selfevo.integration.group_apply import _APPLIED
    from selfevo.routing.base import applicable_modes

    assert set(_APPLIED) == set(applicable_modes()), (
        f"seam implements {sorted(_APPLIED)} but the registry declares "
        f"{sorted(applicable_modes())} applicable"
    )


def test_registering_an_applicable_mode_with_no_branch_here_fails_at_import():
    """The drift check itself, in a fresh interpreter.

    Asserting ``set(_APPLIED) == set(applicable_modes())`` in this process only says the two
    agree today; it does not say anything notices when they stop. This registers a mode the
    seam has never heard of and imports the seam, which must refuse rather than leave a
    routable mode that cannot become an update.
    """
    repo = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "from selfevo.routing.base import register_mode\n"
        "register_mode('teleport', needs_teacher=False)\n"
        "import selfevo.integration.group_apply\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo)
    )
    assert r.returncode != 0, "the seam imported cleanly with an unimplemented mode declared"
    assert "teleport" in r.stderr and "no branch for them" in r.stderr, r.stderr[-800:]


def test_distill_is_registered_but_declared_inapplicable():
    """The one registered mode nothing applies. Both halves matter: it stays registered
    (routers still gate on ``needs_teacher``, and it documents the axis) and it is declared
    unappliable, which is what stops a default-configured router from selecting it."""
    from selfevo.routing.base import applicable_modes

    assert known_modes()[TrainingMode.DISTILL] is True
    assert TrainingMode.DISTILL not in applicable_modes()
    for mode in (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP):
        assert mode in applicable_modes()


def test_distill_raises_rather_than_being_silently_skipped():
    """``distill`` is REGISTERED, so this is not the unknown-name check.

    Treating a teacher-requiring mode as SKIP would let a distillation arm report results for
    a run in which no distillation ever happened.
    """
    assert known_modes()[TrainingMode.DISTILL] is True
    with pytest.raises(ValueError, match="teacher-requiring"):
        apply_decisions(
            torch.zeros(4, 3), torch.ones(4, 3), [4], [TrainingMode.DISTILL], sft_weight=0.5
        )


def test_sft_is_allowed_even_though_it_needs_a_teacher():
    """The mode registry says SFT needs a target; this seam accepts it anyway, because the
    target is the group's own correct sample and no tensor is needed to express it."""
    assert known_modes()[TrainingMode.SFT] is True
    out, _ = apply_decisions(*silent(), [4], [TrainingMode.SFT], sft_weight=0.5)
    assert out.abs().max() > 0


def test_a_negative_group_size_cannot_reach_another_groups_rows():
    """The demonstration behind the ``>= 1`` guard.

    ``[-1, 5]`` sums to 4 and so passes the partition check, and ``slice(0, -1)`` is three
    rows of a four-row batch: the SKIP group, which owns no rows at all, would zero three
    rows belonging to the RL group.
    """
    adv = torch.ones(4, 3)
    with pytest.raises(ValueError, match="must be >= 1"):
        apply_decisions(adv, torch.ones(4, 3), [-1, 5], ["skip", "rl"], sft_weight=0.0)


# ------------------------------------------------------------------------- dtypes ------


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", MODES)
def test_the_advantage_dtype_survives(dt, mode):
    """A dtype change here reaches the loss as a silent upcast of the whole batch."""
    adv, lm = informative()
    out, _ = apply_decisions(adv.to(dt), lm, [4], [mode], sft_weight=0.5)
    assert out.dtype == dt


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("w", [0.1, 0.25, 0.5, 1.0])
def test_the_written_value_is_the_weight_rounded_once(dt, w):
    """The mask multiplication must add no error of its own.

    ``sft_weight=0.1`` is not representable in bfloat16 (it becomes 0.10009765625), and that
    rounding is unavoidable. What would NOT be unavoidable is a second rounding from the mask
    multiply, or a value that differs row to row: both would make an SFT group's gradient
    depend on where in the batch it landed.
    """
    adv, lm = informative()
    out, _ = apply_decisions(adv.to(dt), lm, [4], [TrainingMode.SFT], sft_weight=w)
    written = out[lm.bool()]
    assert torch.equal(written, torch.full_like(written, torch.tensor(w, dtype=dt).item()))
    assert len(torch.unique(written)) == 1


@pytest.mark.parametrize(
    "mask_dtype", [torch.bool, torch.int64, torch.uint8, torch.float64, torch.float32]
)
def test_the_mask_dtype_does_not_change_the_result(mask_dtype):
    """``loss_mask`` arrives as float from the actor and as bool from ``gen_mask``."""
    adv, lm = informative()
    ref, _ = apply_decisions(adv, lm, [4], [TrainingMode.SFT], sft_weight=0.5)
    out, _ = apply_decisions(adv, lm.to(mask_dtype), [4], [TrainingMode.SFT], sft_weight=0.5)
    assert torch.equal(out, ref)


# ------------------------------------------------------------------------ metrics ------


def test_changed_row_fraction_is_a_fraction_of_rows():
    """Eight rows in two groups, every row changed, is 1.0 -- not 4.0.

    The denominator is the batch size. ``sum(counts.values())`` counts GROUPS, and dividing
    rows by groups yields the mean number of changed rows per group: at the live group size
    of 8 it read eight times too high, and a 'fraction' above 1.0 in the run log.
    """
    adv = torch.randn(8, 4)
    out, stats = apply_decisions(
        adv, torch.ones(8, 4), [4, 4], [TrainingMode.SFT, TrainingMode.SKIP], sft_weight=0.5
    )
    assert stats.changed_rows == 8
    assert stats.as_metrics()["route/changed_row_fraction"] == 1.0


@pytest.mark.parametrize("sizes", [[8], [4, 4], [2, 2, 2, 2], [1] * 8, [5, 3], [6, 1, 1]])
def test_changed_row_fraction_matches_the_rows_that_actually_moved(sizes):
    """Swept over partitions: with the group count as denominator only ``[1] * 8`` agrees,
    which is why a single spot-check missed this."""
    adv, lm = informative(b=8, t=4)
    modes = [TrainingMode.SFT] * len(sizes)
    out, stats = apply_decisions(adv, lm, sizes, modes, sft_weight=0.5)
    moved = int((out != adv).any(dim=-1).sum())
    assert stats.as_metrics()["route/changed_row_fraction"] == moved / 8
    assert 0.0 <= stats.as_metrics()["route/changed_row_fraction"] <= 1.0


def test_metrics_report_group_counts_per_mode():
    adv, lm = informative(b=6, t=4)
    _, stats = apply_decisions(
        adv, lm, [2, 2, 2], [TrainingMode.RL, TrainingMode.SFT, TrainingMode.SFT], sft_weight=0.5
    )
    m = stats.as_metrics()
    assert m["route/rl_groups"] == 1.0
    assert m["route/sft_groups"] == 2.0
    assert m["route/skip_groups"] == 0.0
    assert m["route/n_groups"] == 3.0
    # Every key is present even at zero, so a dashboard panel does not disappear when a mode
    # stops being chosen.
    assert set(m) == {
        "route/rl_groups", "route/sft_groups", "route/skip_groups",
        "route/changed_row_fraction", "route/n_groups",
    }


def test_stats_carry_both_denominators():
    """Rows and groups are different numbers and both are reported."""
    adv, lm = informative(b=6, t=4)
    _, stats = apply_decisions(adv, lm, [3, 3], [TrainingMode.SFT] * 2, sft_weight=0.5)
    assert isinstance(stats, ApplyStats)
    assert (stats.n_rows, stats.n_groups) == (6, 2)


# ------------------------------------------------------------ empty and single group ---


def test_an_empty_batch_is_not_an_error():
    """A rollout can produce nothing routable; the metric must be 0.0, not a ZeroDivision."""
    out, stats = apply_decisions(torch.zeros(0, 5), torch.zeros(0, 5), [], [], sft_weight=0.5)
    assert tuple(out.shape) == (0, 5)
    assert (stats.n_rows, stats.n_groups, stats.changed_rows) == (0, 0, 0)
    assert stats.as_metrics()["route/changed_row_fraction"] == 0.0


def test_zero_length_sequences_are_not_an_error():
    out, stats = apply_decisions(
        torch.zeros(4, 0), torch.zeros(4, 0), [4], [TrainingMode.SFT], sft_weight=0.5
    )
    assert tuple(out.shape) == (4, 0)
    assert stats.changed_rows == 0


@pytest.mark.parametrize("mode", MODES)
def test_a_single_group_covers_the_whole_batch(mode):
    adv, lm = informative(b=8, t=4)
    out, stats = apply_decisions(adv, lm, [8], [mode], sft_weight=0.5)
    assert stats.n_groups == 1 and stats.n_rows == 8
    expected = 0 if mode == TrainingMode.RL else 8
    assert stats.changed_rows == expected


# ---------------------------------------------------------------------- the sweep ------


@pytest.mark.parametrize("seed", range(25))
def test_random_batches_hold_every_invariant(seed):
    """Random shapes, partitions, modes and weights against the four load-bearing claims.

    ``changed_rows`` is checked against the returned tensor, which is the DEFINITION of the
    field rather than a copy of how it is computed.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    b, t = rng.randint(1, 9), rng.randint(1, 7)
    adv = torch.randn(b, t)
    before = adv.clone()
    lm = (torch.rand(b, t) < 0.6).float()
    sizes = random_partition(rng, b)
    modes = [rng.choice(MODES) for _ in sizes]
    w = rng.choice(WEIGHTS)

    out, stats = apply_decisions(adv, lm, sizes, modes, sft_weight=w)

    assert torch.equal(adv, before), "the caller's tensor was modified in place"
    off = lm == 0
    assert torch.equal(out[off], before[off]), "a write escaped the loss mask"
    start = 0
    for g, mode in zip(sizes, modes):
        rows, start = slice(start, start + g), start + g
        on = lm[rows].bool()
        block = out[rows]
        if mode == TrainingMode.RL:
            assert torch.equal(block, before[rows]), (seed, mode)
        elif mode == TrainingMode.SFT:
            assert torch.equal(block[on], torch.full_like(block[on], w)), (seed, mode)
        else:
            assert torch.equal(block[on], torch.zeros_like(block[on])), (seed, mode)
    assert stats.changed_rows == int((out != before).any(dim=-1).sum()), (seed, stats)
    assert (stats.n_rows, stats.n_groups) == (b, len(sizes))
    assert stats.counts[TrainingMode.RL] == modes.count(TrainingMode.RL)
    assert stats.counts[TrainingMode.SFT] == modes.count(TrainingMode.SFT)
    assert stats.counts[TrainingMode.SKIP] == modes.count(TrainingMode.SKIP)
    assert stats.as_metrics()["route/changed_row_fraction"] == stats.changed_rows / b


# -------------------------------------------------------- against the REAL actor path --


def loss_mask_like_the_actor() -> torch.Tensor:
    """The mask ``make_batch`` builds, so the seam is fed exactly what the actor was fed."""
    lm = torch.zeros(B, T)
    lm[:, PROMPT:] = 1.0
    return lm


def test_sft_on_a_silent_group_reproduces_the_actors_hardcoded_constant():
    """M22's refactor claim, checked against the running code rather than a re-derivation.

    ``MIXED`` makes group 0 unanimously correct, which under group-level reward norm makes
    its advantages identically zero -- the silent group the actor's rule fires on. Routing
    that group to SFT through this seam must produce the actor's tensor bit for bit; group 1
    is informative and stays RL in both.
    """
    w = 0.5
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    torch.manual_seed(0)
    actor_routed = advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=w)), MIXED
    )
    assert not torch.equal(base, actor_routed), "the actor's own routing did not fire"

    seam, stats = apply_decisions(
        base, loss_mask_like_the_actor(), [G, G], [TrainingMode.SFT, TrainingMode.RL], sft_weight=w
    )
    assert torch.equal(seam, actor_routed), (seam - actor_routed).abs().max()
    assert stats.changed_rows == G


def test_the_seam_and_the_actor_diverge_on_an_informative_group_by_design():
    """The actor ADDS a constant; the seam REPLACES. On a silent group that is the same
    tensor, which is why the test above can be a refactor check -- and on an informative one
    it is not, which is the whole reason the seam exists."""
    w = 0.5
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    lm = loss_mask_like_the_actor()
    assert base[G:].abs().max() > 1e-6, "group 1 is supposed to be informative"
    seam, _ = apply_decisions(base, lm, [G, G], [TrainingMode.RL, TrainingMode.SFT], sft_weight=w)
    assert not torch.equal(seam[G:], base[G:] + w * lm[G:])
    assert torch.equal(seam[G:][lm[G:].bool()], torch.full((G * (T - PROMPT),), w))


def test_the_actors_prompt_advantages_are_real_and_the_seam_preserves_them():
    """Why the mask has to bound the write and not merely its non-zero part.

    The actor's GAE carries ``lastgaelam`` backwards through the masked prefix, so an
    informative group arrives here with NON-ZERO advantages on its prompt columns. A write
    that covered the whole row would erase them in a tensor the caller still holds.
    """
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    assert base[G:, :PROMPT].abs().max() > 1e-6, base[G:]
    lm = loss_mask_like_the_actor()
    for mode in (TrainingMode.SFT, TrainingMode.SKIP):
        seam, _ = apply_decisions(base, lm, [G, G], [TrainingMode.RL, mode], sft_weight=0.5)
        assert torch.equal(seam[G:, :PROMPT], base[G:, :PROMPT]), (mode, seam[G:])
