"""The two stabilisers on ``GroupRoutingConfig``: ``zero_mean`` and
``exclude_truncated_from_sft``.

Both exist because of one measurement (2026-08-31). Routing a positive constant onto whole
GRPO groups breaks the zero-mean advantage property GRPO is built on: mean ``advantages/avg``
over a run was -0.0004 and +0.0136 for two UNROUTED arms against +0.1579, +0.1675 and +0.8979
for three routed ones, whose response length knees at roughly step 142, 132 and 120 -- larger
offset, earlier knee, and no knee at all near zero. Nothing bounds the constant: at
``ppo_n_minibatches=1`` with ``recompute_logprob`` and ``use_decoupled_loss`` the importance
ratio is exactly 1.0, so the PPO clip is inert and these are unclipped REINFORCE updates. The
force is carried specifically by TRUNCATED rollouts, which contain no EOS, so every token the
constant pushes on says "keep going" with no counterweight, at ~2.5x the gradient mass of a
terminating row under a token-mean loss.

``zero_mean`` removes the offset; ``exclude_truncated_from_sft`` removes the vehicle. They are
separate axes, and this file tests them separately and together, because a run that switched
both on and improved would otherwise not say which one did it.

Everything that can go through the real ``PPOActor._compute_advantages`` does. A test that
re-derives the arithmetic locally pins a COPY of the code and cannot notice the copy drifting,
which has already cost this project once.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig, NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, _recentre_advantages, _truncated_rows
from areal.utils.data import TrajBatchMeta
from selfevo import compose
from selfevo.integration.group_apply import apply_decisions, apply_mixtures
from selfevo.routing.base import RoutingDecision, TrainingMode

logging.disable(logging.INFO)

RL, SFT, SKIP = TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP

B, T, G = 8, 10, 4          # two groups of four
PROMPT = 2                  # first PROMPT columns are prompt
CAP = 4                     # actor.max_new_tokens for this fixture
W = 0.5                     # solved_advantage

# Rows 0, 2, 4, 6 reach the cap and are therefore TRUNCATED; 1, 3, 5, 7 terminated. The
# alternation is deliberate: it puts a truncated and a terminated row in the SAME group, so a
# defect that keys on the group instead of the row cannot pass.
GEN = [CAP, CAP - 1, CAP, CAP - 1, CAP, CAP - 1, CAP, CAP - 1]
TRUNCATED_ROWS = [0, 2, 4, 6]
TERMINATED_ROWS = [1, 3, 5, 7]

# Group 0 (rows 0-3): every sample correct -> silent because SOLVED, so the fixed rule writes
# the constant there. Group 1 (rows 4-7): two of four correct -> informative.
MIXED = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]

STUB = "_stabiliser_stub"


# ------------------------------------------------------------------------- fixtures ---


def token_mask() -> torch.Tensor:
    """The response mask as it arrives, in TOKEN coordinates: 1 on each response token."""
    lm = torch.zeros(B, T)
    for i, n in enumerate(GEN):
        lm[i, PROMPT : PROMPT + n] = 1.0
    return lm


def emitter_mask() -> torch.Tensor:
    """The mask the LOSS reads: rolled left by one, so index t is the position that predicts
    token t+1.

    ``_compute_advantages`` builds this locally and writes it back to ``data["loss_mask"]``
    only after routing has run -- the coordinate off-by-one documented in that method.
    ``zero_mean`` is defined against THIS mask, because it is the set of positions
    ``grpo_loss_fn`` actually sums.
    """
    return torch.roll(token_mask(), shifts=-1, dims=-1) != 0


def make_actor(group_routing: GroupRoutingConfig | None, cap: int = CAP) -> PPOActor:
    """A CPU actor configured like the live runs, with a small generation cap.

    Args:
        group_routing: Value for ``config.group_routing``.
        cap: ``max_new_tokens``. Small so that a row of ``CAP`` response tokens is genuinely
            at the budget, which is what truncation means.

    Returns:
        A ``PPOActor`` whose ``_compute_advantages`` can be called directly.
    """
    cfg = PPOActorConfig(
        path="unused-for-advantage-computation",
        kl_ctl=0.0,
        adv_norm=None,
        reward_norm=NormConfig(mean_level="group", std_level="group", group_size=G),
    )
    cfg.max_new_tokens = cap
    cfg.group_routing = group_routing
    return PPOActor(cfg, engine=None)


def make_batch(rewards: list[float] | None = None) -> dict[str, torch.Tensor]:
    """A batch whose rows have DIFFERENT response lengths, some at the cap and some not.

    ``attention_mask`` is right-padded to match rather than all-ones: the actor reads
    ``attention_mask.sum(-1)`` to place the outcome reward, and an all-ones mask would put
    every reward in the padding.
    """
    am = torch.zeros(B, T)
    for i, n in enumerate(GEN):
        am[i, : PROMPT + n] = 1.0
    return {
        # Deterministic: these values are read only by the prompt-credit path, and a random
        # tensor would make a failure depend on the seed.
        "input_ids": torch.arange(B * T).reshape(B, T) % 97,
        "loss_mask": token_mask(),
        "rewards": torch.tensor(rewards or MIXED, dtype=torch.float32),
        "logprobs": torch.zeros(B, T),
        "attention_mask": am,
    }


def meta() -> TrajBatchMeta:
    """Group structure matching ``make_batch``: two groups of ``G`` rows."""
    return TrajBatchMeta(
        n_trajs=B, traj_group_sizes=[G, G], traj_seqlens=[PROMPT + n for n in GEN]
    )


def advantages(actor: PPOActor) -> torch.Tensor:
    """Run the real advantage computation and return the advantage tensor."""
    return actor._compute_advantages(make_batch(), meta())["advantages"]


def routed(**kw) -> torch.Tensor:
    """Advantages under the fixed solved/unsolved rule with the given stabiliser flags."""
    return advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=W, **kw))
    )


class AllSFTRouter:
    """Routes every unit to SFT, so the seam is exercised on informative groups too."""

    def route(self, ctx) -> RoutingDecision:
        """Return the fixed SFT decision."""
        return RoutingDecision(weights={SFT: 1.0}, reason="stub")


@pytest.fixture
def stub_router():
    """Register the all-SFT router and restore the registry afterwards.

    The registry is module-level state shared with every other test in the process, so a
    test that mutates it without restoring makes an unrelated test fail later, in a
    different file, with no visible connection.
    """
    previous = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = lambda *a, **kw: AllSFTRouter()
    try:
        yield
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = previous


# --------------------------------------------------------------- the fixture is honest ---


def test_the_fixture_really_contains_both_kinds_of_row():
    """Premise. If every row were truncated, or none, the exclusion tests prove nothing."""
    trunc = _truncated_rows(token_mask(), CAP)
    assert [i for i, v in enumerate(trunc.tolist()) if v] == TRUNCATED_ROWS
    assert [i for i, v in enumerate(trunc.tolist()) if not v] == TERMINATED_ROWS


def test_routing_really_breaks_the_zero_mean_property_on_this_fixture():
    """Premise for every ``zero_mean`` test: the offset it removes has to exist here."""
    m = emitter_mask()
    plain = advantages(make_actor(None))
    with_routing = routed()
    assert abs(float(with_routing[m].mean())) > 0.2
    assert float(with_routing[m].mean()) > float(plain[m].mean()) + 0.1


def test_the_masked_and_unmasked_means_differ_on_this_fixture():
    """Premise that makes "which mean" a testable question.

    The actor deliberately leaves real GAE values on prompt positions, so the mean over the
    whole tensor is not the mean over the tokens the loss reads. If the two coincided, a
    correction that subtracted the wrong one would be indistinguishable from the right one.
    """
    a = routed()
    assert abs(float(a[emitter_mask()].mean()) - float(a.mean())) > 0.1


# ------------------------------------------------------------------- _truncated_rows ---


def test_truncation_is_length_against_the_cap():
    """A response of exactly the cap is truncated; one token short of it is not."""
    lm = torch.zeros(4, 12)
    for i, n in enumerate([5, 4, 6, 0]):
        lm[i, 2 : 2 + n] = 1.0
    assert _truncated_rows(lm, 5).tolist() == [True, False, True, False]


def test_truncation_is_not_measured_against_the_padded_width():
    """The defect this replaces.

    ``no_eos_ratios`` compares a sequence's length to the PADDED BATCH WIDTH, so it flags a
    row only when that row happens to be the longest in its own batch -- it read 0.002-0.005
    at every step of four runs. Every row here is at the cap and none is at the padded
    width, so the two signals must disagree completely.
    """
    lm = torch.zeros(4, 64)
    lm[:, 2:6] = 1.0
    assert _truncated_rows(lm, 4).tolist() == [True] * 4
    # What the broken instrument would have said about the same batch.
    assert (lm.sum(-1) >= lm.shape[-1]).tolist() == [False] * 4


def test_truncation_is_invariant_to_the_emitter_roll():
    """A row's LENGTH is not a coordinate.

    ``torch.roll`` permutes positions within a row, so the per-row sum is unchanged by it.
    That is what lets this signal be read at the routing site -- where ``data["loss_mask"]``
    is still in token coordinates -- without inheriting the off-by-one that afflicts the
    positional writes there.
    """
    lm = token_mask()
    assert torch.equal(
        _truncated_rows(lm, CAP), _truncated_rows(torch.roll(lm, shifts=-1, dims=-1), CAP)
    )


@pytest.mark.parametrize("cap", [0, -1, None])
def test_a_missing_cap_is_refused_rather_than_read_as_no_truncation(cap):
    """Answering "nothing is truncated" would make an exclusion arm that excluded nothing
    indistinguishable from one that was never switched on."""
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        _truncated_rows(token_mask(), cap)


# ---------------------------------------------------------------- _recentre_advantages ---


def test_recentring_zeroes_the_masked_mean_and_reports_both_ends():
    """Both returned means must be MEASURED off the tensors, not asserted.

    ``after`` is checked for exact agreement with the same reduction computed here, not
    merely for being small. A correct implementation returns that value bit for bit -- it is
    literally the same masked float64 mean -- whereas an implementation that reported a
    constant 0.0 would be indistinguishable under a "< 1e-6" check while carrying no
    information at all. The float32 residual on this fixture is ~8.5e-9, which is seven
    orders of magnitude above the tolerance used here, so the two cases are cleanly
    separated.
    """
    a = routed()
    m = emitter_mask()
    out, before, after = _recentre_advantages(a, m.float())
    assert before == pytest.approx(float(a[m].to(torch.float64).mean()), abs=1e-15)
    assert after == pytest.approx(float(out[m].to(torch.float64).mean()), abs=1e-15)
    assert abs(after) < 1e-6


def test_recentring_subtracts_the_MASKED_mean_not_the_whole_tensor_mean():
    """The two differ here by more than 0.1, and only one of them is what the loss sees."""
    a = routed()
    m = emitter_mask()
    out, _, _ = _recentre_advantages(a, m.float())
    assert abs(float(out[m].mean())) < 1e-6
    # Had the whole-tensor mean been subtracted, the masked mean would have survived almost
    # intact.
    assert abs(float((a - a.mean())[m].mean())) > 0.1


def test_recentring_leaves_positions_outside_the_mask_exactly_as_they_were():
    """The loss never reads them and other code still inspects them, so shifting them would
    be an invisible edit to a tensor the caller holds."""
    a = routed()
    m = emitter_mask()
    out, _, _ = _recentre_advantages(a, m.float())
    assert torch.equal(out[~m], a[~m])


def test_recentring_preserves_every_relative_difference():
    """A constant shift is the whole point: the content of a policy gradient is differences
    between advantages, and those must survive untouched."""
    a = routed()
    m = emitter_mask()
    out, _, _ = _recentre_advantages(a, m.float())
    before, after = a[m], out[m]
    assert torch.allclose(before - before[0], after - after[0], atol=1e-6)


def test_recentring_a_batch_with_no_response_tokens_is_a_no_op():
    """There is no mean to subtract, and inventing one would write NaN across the batch."""
    a = torch.randn(3, 5)
    out, before, after = _recentre_advantages(a, torch.zeros(3, 5))
    assert torch.equal(out, a)
    assert (before, after) == (0.0, 0.0)


# ------------------------------------------------------------------- the apply seam ---


def _seam_batch():
    """Four rows in two groups, with response tokens in columns 1-3."""
    adv = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5) / 10.0
    lm = torch.zeros(4, 5)
    lm[:, 1:4] = 1.0
    return adv, lm


def test_sft_rows_withholds_the_constant_from_exactly_those_rows():
    adv, lm = _seam_batch()
    veto = torch.tensor([True, False, True, False])   # rows 1 and 3 are withheld
    out, stats = apply_decisions(adv, lm, [2, 2], [SFT, SFT], sft_weight=W, sft_rows=veto)
    assert torch.equal(out[0][1:4], torch.full((3,), W))
    assert torch.equal(out[2][1:4], torch.full((3,), W))
    assert torch.equal(out[1], adv[1])
    assert torch.equal(out[3], adv[3])
    assert stats.sft_excluded_rows == 2
    # An excluded row is left ALONE, not skipped: it keeps the RL advantage it arrived with.
    assert float(out[1].abs().sum()) > 0


def test_sft_rows_does_not_touch_a_skip_group():
    """SKIP is not the positive constant, and the measured mechanism is specific to that
    constant's sign, so the veto must not reach it."""
    adv, lm = _seam_batch()
    veto = torch.zeros(4, dtype=torch.bool)
    out, stats = apply_decisions(adv, lm, [2, 2], [SKIP, SKIP], sft_weight=W, sft_rows=veto)
    assert torch.equal(out[:, 1:4], torch.zeros(4, 3))
    assert stats.sft_excluded_rows == 0


def test_sft_rows_does_not_touch_an_rl_group():
    """An RL group is not written at all, so nothing there can be withheld or counted."""
    adv, lm = _seam_batch()
    veto = torch.zeros(4, dtype=torch.bool)
    out, stats = apply_decisions(adv, lm, [2, 2], [RL, RL], sft_weight=W, sft_rows=veto)
    assert torch.equal(out, adv)
    assert stats.sft_excluded_rows == 0


def test_sft_rows_none_is_bit_identical_to_not_passing_it():
    adv, lm = _seam_batch()
    a, sa = apply_decisions(adv, lm, [2, 2], [SFT, RL], sft_weight=W)
    b, sb = apply_decisions(adv, lm, [2, 2], [SFT, RL], sft_weight=W, sft_rows=None)
    c, sc = apply_decisions(
        adv, lm, [2, 2], [SFT, RL], sft_weight=W, sft_rows=torch.ones(4, dtype=torch.bool)
    )
    assert torch.equal(a, b) and torch.equal(a, c)
    assert sa.sft_excluded_rows == sb.sft_excluded_rows == sc.sft_excluded_rows == 0


@pytest.mark.parametrize("bad", [(3,), (5,), (4, 1)])
def test_a_mismatched_sft_rows_is_refused(bad):
    """It is indexed by row alongside the advantages; a mismatch vetoes the wrong rows or
    broadcasts over the wrong axis, and both fail silently."""
    adv, lm = _seam_batch()
    with pytest.raises(ValueError, match="sft_rows must have shape"):
        apply_decisions(
            adv, lm, [2, 2], [SFT, SFT], sft_weight=W, sft_rows=torch.ones(bad).bool()
        )


def test_a_one_hot_mixture_with_a_veto_reduces_to_the_hard_path():
    """The reduction claim has to cover the veto, in the tensor AND in the stats."""
    adv, lm = _seam_batch()
    veto = torch.tensor([True, False, False, True])
    hard, hs = apply_decisions(adv, lm, [2, 2], [SFT, SFT], sft_weight=W, sft_rows=veto)
    soft, ss = apply_mixtures(
        adv, lm, [2, 2], [{SFT: 1.0}, {SFT: 1.0}], sft_weight=W, sft_rows=veto
    )
    assert torch.equal(hard, soft)
    assert ss.sft_excluded_rows == hs.sft_excluded_rows == 2


# -------------------------------------------------------------- defaults and rollback ---


def test_both_knobs_default_to_off():
    """A default of True would turn every existing config into a different experiment
    without anyone editing it."""
    gr = GroupRoutingConfig()
    assert gr.zero_mean is False
    assert gr.exclude_truncated_from_sft is False


def test_both_off_reproduces_the_unflagged_configuration_bit_for_bit():
    assert torch.equal(routed(), routed(zero_mean=False, exclude_truncated_from_sft=False))


def test_both_off_is_the_write_this_code_has_always_made():
    """Pins the shipped behaviour numerically rather than only against itself: the constant
    lands on every response token of the silent-and-solved group, in TOKEN coordinates, and
    nowhere else.
    """
    plain = advantages(make_actor(None))
    expected = plain.clone()
    expected[:G] = plain[:G] + W * token_mask()[:G]
    assert torch.equal(routed(), expected)


# ------------------------------------------------------------------------- zero_mean ---


def test_zero_mean_returns_the_batch_to_zero_mean_over_the_tokens_the_loss_reads():
    m = emitter_mask()
    assert abs(float(routed()[m].mean())) > 0.2                     # premise
    assert abs(float(routed(zero_mean=True)[m].mean())) < 1e-6


def test_zero_mean_preserves_relative_differences_between_advantages():
    """It removes the DC offset and nothing else. Every pairwise difference -- the whole
    content of a policy gradient, and everything routing decided -- must survive."""
    m = emitter_mask()
    off, on = routed()[m], routed(zero_mean=True)[m]
    assert torch.allclose(off - off[0], on - on[0], atol=1e-6)
    # And it is a genuine shift, not a no-op dressed up as one.
    assert abs(float((on - off).mean())) > 0.2
    assert float((on - off).std()) < 1e-6


def test_zero_mean_is_a_batch_correction_and_not_a_per_group_one():
    """Per-group centring would also zero the BATCH mean here -- the two groups have equal
    valid-token counts -- so the distinguishing evidence is that each GROUP is still off zero
    afterwards. Centring per group would re-zero every SFT group it had just written and
    erase the between-group separation routing exists to create.
    """
    m = emitter_mask()
    a = routed(zero_mean=True)
    assert abs(float(a[m].mean())) < 1e-6
    assert abs(float(a[:G][m[:G]].mean())) > 1e-3
    assert abs(float(a[G:][m[G:]].mean())) > 1e-3


def test_zero_mean_off_leaves_the_tensor_untouched():
    assert torch.equal(routed(zero_mean=False), routed())


def test_zero_mean_does_not_disturb_positions_the_loss_never_reads():
    m = emitter_mask()
    assert torch.equal(routed(zero_mean=True)[~m], routed()[~m])


# --------------------------------------------------------- exclude_truncated_from_sft ---


def test_exclude_truncated_skips_exactly_the_non_terminating_rows_and_no_others():
    off, on = routed(), routed(exclude_truncated_from_sft=True)
    plain = advantages(make_actor(None))
    for row in TRUNCATED_ROWS:
        if row < G:
            # In the routed (silent, solved) group the constant is withheld, and the row is
            # left with what it arrived with rather than zeroed by a second intervention.
            assert not torch.equal(off[row], plain[row]), row     # premise
            assert torch.equal(on[row], plain[row]), row
        else:
            # Truncated but not in a routed group: unchanged either way.
            assert torch.equal(on[row], off[row]), row
    for row in TERMINATED_ROWS:
        assert torch.equal(on[row], off[row]), row


def test_exclude_truncated_off_leaves_the_tensor_untouched():
    assert torch.equal(routed(exclude_truncated_from_sft=False), routed())


def test_the_veto_reaches_the_router_branch_too(stub_router):
    """The fixed rule only ever writes to silent groups, so testing there cannot show that a
    truncated row inside an INFORMATIVE group is protected. A router that sends every group
    to SFT can."""
    plain = advantages(make_actor(None))
    cfg = dict(enabled=True, solved_advantage=W, router=STUB)
    off = advantages(make_actor(GroupRoutingConfig(**cfg)))
    on = advantages(
        make_actor(GroupRoutingConfig(**cfg, exclude_truncated_from_sft=True))
    )
    for row in TRUNCATED_ROWS:
        assert not torch.equal(off[row], plain[row]), row          # premise
        assert torch.equal(on[row], plain[row]), row
    for row in TERMINATED_ROWS:
        assert torch.equal(on[row], off[row]), row
        assert torch.equal(
            on[row][PROMPT : PROMPT + GEN[row]], torch.full((GEN[row],), W)
        ), row


# ------------------------------------------------------------------------ composition ---


def test_both_on_compose_without_either_undoing_the_other():
    m = emitter_mask()
    excl = routed(exclude_truncated_from_sft=True)
    both = routed(exclude_truncated_from_sft=True, zero_mean=True)
    zm = routed(zero_mean=True)

    # zero_mean acts on top of whatever the exclusion left: a constant shift on the masked
    # positions, of exactly the residual offset, and nothing anywhere else.
    shift = float(excl[m].mean())
    assert torch.allclose(both, torch.where(m, excl - shift, excl), atol=1e-6)
    assert abs(float(both[m].mean())) < 1e-6

    # The offset the correction had to remove is SMALLER once the exclusion has run: the
    # exclusion withheld part of the constant that created it. That is the two knobs acting
    # on the same quantity from different ends, which is why they are separate axes.
    assert 0.0 < shift < float(routed()[m].mean())

    # And the composition is not either knob alone.
    assert not torch.equal(both, zm)
    assert not torch.equal(both, excl)


# --------------------------------------------------------- what the run actually logs ---


def _capture(monkeypatch, gr) -> dict[str, float]:
    """Run one batch and return every scalar the actor handed the tracker."""
    from areal.utils import stats_tracker

    seen: dict[str, float] = {}
    monkeypatch.setattr(stats_tracker, "scalar", lambda **kw: seen.update(kw))
    advantages(make_actor(gr))
    return seen


def test_the_knobs_are_visible_in_the_logs_on_the_rule_branch(monkeypatch):
    """A knob whose effect cannot be seen in the logs is untestable in a real run.

    Asserted by capturing what the actor actually hands the tracker, not by reading the
    source: the branch is the thing under test.
    """
    off = _capture(monkeypatch, GroupRoutingConfig(enabled=True, solved_advantage=W))
    on = _capture(
        monkeypatch,
        GroupRoutingConfig(
            enabled=True,
            solved_advantage=W,
            zero_mean=True,
            exclude_truncated_from_sft=True,
        ),
    )

    # Same keys in both arms, so the ablation stays readable on one panel.
    assert set(off) == set(on), set(off) ^ set(on)

    assert off["route/truncated_row_fraction"] == pytest.approx(0.5)
    assert on["route/truncated_row_fraction"] == pytest.approx(0.5)

    assert off["route/sft_excluded_rows"] == 0.0
    assert on["route/sft_excluded_rows"] == 2.0      # rows 0 and 2: truncated AND routed

    # The offset is reported whether or not it is corrected -- the UNCORRECTED value is the
    # diagnostic that identified this failure -- and "after" says whether it was.
    assert off["route/adv_mean_before"] > 0.2
    assert off["route/adv_mean_after"] == off["route/adv_mean_before"]
    # Smaller here than in the uncorrected arm, because the exclusion withheld part of the
    # constant that created the offset -- but still present, so zero_mean has work to do.
    assert 0.1 < on["route/adv_mean_before"] < off["route/adv_mean_before"]
    assert abs(on["route/adv_mean_after"]) < 1e-6


def test_the_exclusion_count_is_visible_on_the_router_branch_too(monkeypatch, stub_router):
    cfg = dict(enabled=True, solved_advantage=W, router=STUB)
    off = _capture(monkeypatch, GroupRoutingConfig(**cfg))
    on = _capture(
        monkeypatch, GroupRoutingConfig(**cfg, exclude_truncated_from_sft=True)
    )
    assert set(off) == set(on), set(off) ^ set(on)
    assert off["route/sft_excluded_rows"] == 0.0
    assert on["route/sft_excluded_rows"] == 4.0      # every truncated row, both groups
    assert on["route/truncated_row_fraction"] == pytest.approx(0.5)
