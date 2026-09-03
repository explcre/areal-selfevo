"""How one GRPO loss weights an SFT-mode group against an RL-mode group.

arXiv 2604.23747 reports that several published gains from mixing SFT and RL inside one
optimiser step were loss-weighting artefacts: the two modes were normalised differently --
one per token, the other per sequence -- so a length difference between the modes silently
rescaled one mode's gradient against the other. Arms A4 (LSPO-style gold-SFT rows) and A5
(DyME-style gold substitution) put SFT-like rows and RL rows in the SAME microbatch, so
before either arm's number is reported the weighting has to be a checked fact rather than a
reading of the code.

These tests drive the REAL ``grpo_loss_fn`` and read the gradient it puts on ``logprobs``.
They are not a restatement of the implementation's arithmetic: the expected ratios are
derived from the two competing normalisation HYPOTHESES (token-mean and sequence-mean), and
the test's job is to say which hypothesis the measurement matches. Under a sequence-mean the
SFT/RL gradient ratio would not move when only the SFT rows get longer; under a token-mean it
moves proportionally. That single contrast is what pins the answer.

The second half pins what the loss does with a row the policy never sampled -- a gold row,
whose behaviour log-probabilities do not exist. That decides how ``selfevo/gold/substitute.py``
must fill ``logprobs``: there is a path here that silently reports such a row as perfectly
on-policy, and a path that poisons every other row in the batch.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from areal.api.cli_args import NormConfig, RejectionSamplingConfig  # noqa: E402
from areal.trainer.ppo.actor import grpo_loss_fn  # noqa: E402
from areal.utils.data import Normalization  # noqa: E402
from areal.utils.functional.functional import apply_rejection_sampling  # noqa: E402

# The actor fixture is imported rather than rebuilt: it already mirrors the live config, and
# a second copy would drift from the first without either copy failing.
from selfevo.tests.conftest import (  # noqa: E402
    B,
    PROMPT,
    T,
    make_actor,
    make_batch,
    meta,
)

@pytest.fixture(autouse=True)
def _clear_stats_tracker(clear_stats_tracker):
    """Apply the shared conftest fixture to every test in this module.

    These tests deliberately vary the token count, which is the case that fixture exists
    for: without it the second batch of a different width fails inside the stats tracker,
    and the failure looks like a loss bug rather than leaked state.
    """


# The SFT constant the M19 self-target writes (group_routing.solved_advantage) and the
# magnitude a group-normalised RL advantage carries. Kept distinct so a ratio that happened
# to be 1.0 could not hide a length effect.
SFT_C = 0.5
RL_A = 1.0

# The live surrogate is on-policy at the first inner step: ppo_n_minibatches=1, so the
# training forward sees the same weights that produced prox_logp and the ratio is exactly 1.
# Four runs measured importance_weight avg=min=max=1.0 and clip_ratio 0.0 at every step.
EPS_CLIP = 0.2

# examples/math/gsm8k_grpo_lora.yaml:78-80 -- the filter the live 30B run has switched on.
LIVE_REJECTION = RejectionSamplingConfig(metric="ratio", upper=5.0)


def two_group_microbatch(
    len_sft: int, len_rl: int, *, seed: int = 0
) -> tuple[torch.Tensor, dict, int]:
    """One packed microbatch holding an SFT-mode group and an RL-mode group.

    The layout is what ``PPOActor._compute_advantages`` hands the loss after routing: rows
    0-1 are a solved group whose response-token advantages have been REPLACED by the
    constant ``SFT_C`` (selfevo/integration/group_apply.py:251-262, or the fixed-rule
    equivalent at areal/trainer/ppo/actor.py:966-989), and rows 2-3 are an ordinary RL group
    carrying group-normalised advantages that sum to zero over its rows.

    Packed 1-D with ``cu_seqlens``, because that is the shape the FSDP engine delivers -- a
    padded (B, T) rehearsal is what let an earlier routing bug reach a live run.

    Args:
        len_sft: Response tokens in each of the two SFT rows.
        len_rl: Response tokens in each of the two RL rows.
        seed: Seed for the log-probability draw, so a comparison across lengths moves only
            the length.

    Returns:
        ``(logprobs, input_data, n_tokens)``. ``logprobs`` is a fresh leaf; the caller sets
        ``requires_grad`` on the copy it differentiates.
    """
    torch.manual_seed(seed)
    lens = [len_sft, len_sft, len_rl, len_rl]
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32)
    n = int(cu[-1])
    advantages = torch.cat(
        [
            torch.full((len_sft,), SFT_C),
            torch.full((len_sft,), SFT_C),
            torch.full((len_rl,), RL_A),
            torch.full((len_rl,), -RL_A),
        ]
    )
    # prox_logp EQUAL to the sampled logprobs, which is the live configuration
    # (recompute_logprob + ppo_n_minibatches=1): the ratio is then exactly 1 and the
    # measurement is of the NORMALISATION alone, with no clipping or ratio noise in it.
    behaviour = torch.randn(n)
    data = {
        "logprobs": behaviour.clone(),
        "prox_logp": behaviour.clone(),
        "advantages": advantages,
        "loss_mask": torch.ones(n, dtype=torch.long),
        "cu_seqlens": cu,
        "input_ids": torch.randint(0, 100, (n,)),
    }
    return behaviour, data, n


def group_grad_magnitudes(
    len_sft: int, len_rl: int, **loss_kwargs
) -> tuple[float, float, torch.Tensor]:
    """Run the real loss and return each group's total gradient magnitude.

    The gradient is read on ``logprobs``, which is where the surrogate's only differentiable
    dependence sits, so ``|dL/dlogprobs|`` summed over a group's tokens IS that group's
    contribution to the update, in the loss's own units.

    Args:
        len_sft: Response tokens per SFT row.
        len_rl: Response tokens per RL row.
        **loss_kwargs: Extra arguments forwarded to ``grpo_loss_fn``.

    Returns:
        ``(sft_magnitude, rl_magnitude, loss)``.
    """
    behaviour, data, n = two_group_microbatch(len_sft, len_rl)
    logprobs = behaviour.clone().requires_grad_(True)
    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=torch.zeros(n),
        input_data=data,
        eps_clip=EPS_CLIP,
        eps_clip_higher=None,
        c_clip=None,
        **loss_kwargs,
    )
    loss.backward()
    split = 2 * len_sft
    return (
        float(logprobs.grad[:split].abs().sum()),
        float(logprobs.grad[split:].abs().sum()),
        loss.detach(),
    )


# ---------------------------------------------------------------------------------------
# 1. The normalisation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("len_sft,len_rl", [(4, 4), (8, 4), (16, 4), (4, 16)])
def test_loss_is_a_token_mean_over_the_whole_microbatch(len_sft: int, len_rl: int) -> None:
    """Every masked token in the microbatch carries the same denominator.

    ``areal/utils/functional/functional.py:571`` divides the summed per-token surrogate by
    ``loss_mask_count`` (line 506), the microbatch's token count. The engine then rescales
    the microbatch by ``local_tokens / total_tokens``
    (``areal/engine/fsdp_engine.py:2216-2217`` with the weight from
    ``areal/trainer/ppo/actor.py:1230``), so the composition over microbatches is one global
    token mean and this per-microbatch measurement is the whole story.

    Consequence: a group's share of the update is proportional to its TOKEN COUNT times its
    advantage magnitude. Not to its row count, and not to its group count.
    """
    sft, rl, _ = group_grad_magnitudes(len_sft, len_rl)
    n = 2 * len_sft + 2 * len_rl
    # Predicted from the token-mean hypothesis alone: d/dlogprob of -A*exp(lp - prox) at
    # ratio 1 is -A, summed over the group's tokens and divided by the batch token count.
    assert sft == pytest.approx(SFT_C * 2 * len_sft / n, rel=1e-6)
    assert rl == pytest.approx(RL_A * 2 * len_rl / n, rel=1e-6)


def test_a_longer_sft_group_takes_a_proportionally_larger_share() -> None:
    """The 2604.23747 question, answered on this repo's own loss.

    Under a token mean the SFT/RL gradient ratio is ``(c * n_sft_tokens) /
    (a * n_rl_tokens)`` and therefore moves with the SFT rows' length. Under a per-sequence
    mean it would be ``c / a`` -- fixed, because each sequence would carry its own
    denominator. The two hypotheses are computed here and the measurement is asked which one
    it matches, so this test fails if the normalisation is ever changed in either direction.
    """
    len_rl = 4
    seq_mean_prediction = SFT_C / RL_A
    ratios = {}
    for len_sft in (4, 8, 16):
        sft, rl, _ = group_grad_magnitudes(len_sft, len_rl)
        ratios[len_sft] = sft / rl
        token_mean_prediction = (SFT_C * 2 * len_sft) / (RL_A * 2 * len_rl)
        assert ratios[len_sft] == pytest.approx(token_mean_prediction, rel=1e-6)

    # Doubling only the SFT rows' length doubles the SFT group's share. A sequence-averaged
    # loss would have returned the same number three times.
    assert ratios[8] == pytest.approx(2 * ratios[4], rel=1e-6)
    assert ratios[16] == pytest.approx(4 * ratios[4], rel=1e-6)
    assert ratios[8] != pytest.approx(seq_mean_prediction, rel=1e-3)


def test_only_the_token_count_matters_not_how_it_is_split_into_rows() -> None:
    """Two short SFT rows and one twice-as-long SFT row weigh the same.

    This separates "per token" from "per row": the row structure of the SFT group is changed
    while its total token count is held fixed, and the group's gradient mass does not move.
    Stated because A4 and A5 differ in how many gold rows they add, not only in how long
    those rows are, and the two knobs are not independently priced by this loss.
    """
    wide, _, _ = group_grad_magnitudes(8, 4)  # 2 rows x 8 tokens = 16 SFT tokens
    n = 2 * 8 + 2 * 4
    assert wide == pytest.approx(SFT_C * 16 / n, rel=1e-6)
    # ONE row of 16 SFT tokens, with the RL group and the batch token count unchanged.
    torch.manual_seed(0)
    cu = torch.tensor([0, 16, 20, 24], dtype=torch.int32)
    data = {
        "logprobs": torch.zeros(n),
        "prox_logp": torch.zeros(n),
        "advantages": torch.cat(
            [torch.full((16,), SFT_C), torch.full((4,), RL_A), torch.full((4,), -RL_A)]
        ),
        "loss_mask": torch.ones(n, dtype=torch.long),
        "cu_seqlens": cu,
        "input_ids": torch.randint(0, 100, (n,)),
    }
    logprobs = torch.zeros(n, requires_grad=True)
    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=torch.zeros(n),
        input_data=data,
        eps_clip=EPS_CLIP,
        eps_clip_higher=None,
        c_clip=None,
    )
    loss.backward()
    assert float(logprobs.grad[:16].abs().sum()) == pytest.approx(wide, rel=1e-6)


def test_masked_prompt_tokens_do_not_buy_weight() -> None:
    """Weight follows the RESPONSE tokens, because the denominator is the loss mask.

    A gold row is usually longer overall than a sampled one but the prompt is shared. The
    denominator at functional.py:506 counts masked tokens only, so prompt length is neutral
    and the length effect above is entirely a RESPONSE-length effect.
    """
    behaviour, data, n = two_group_microbatch(8, 4)
    mask = torch.ones(n, dtype=torch.long)
    mask[:2] = 0  # first two positions of the first SFT row are prompt
    data["loss_mask"] = mask
    logprobs = behaviour.clone().requires_grad_(True)
    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=torch.zeros(n),
        input_data=data,
        eps_clip=EPS_CLIP,
        eps_clip_higher=None,
        c_clip=None,
    )
    loss.backward()
    assert float(logprobs.grad[:2].abs().sum()) == 0.0
    assert float(logprobs.grad[:16].abs().sum()) == pytest.approx(
        SFT_C * 14 / int(mask.sum()), rel=1e-6
    )


def test_group_level_adv_norm_is_token_weighted_and_breaks_the_zero_sum() -> None:
    """Why GOAL.md 3 requires ``adv_norm=None``: the group mean counts TOKENS, not rows.

    ``Normalization._compute_mean`` (``areal/utils/data.py:1686-1688``) forms
    ``(x * mask).sum() / mask.sum()``. With per-row-constant advantages of ``a_i`` on ``L_i``
    tokens that is ``sum_i L_i a_i / sum_i L_i``, which equals the per-ROW mean only when
    every ``L_i`` is equal. Real generations are not equal length, so subtracting it leaves
    ``sum_i a_i != 0`` -- the precondition the routing rule and the group's own
    interpretation both rest on. The measured residuals on the live pipeline were 0.0 for
    ``adv_norm=None``, 2.139 for ``mean_level=group`` and 0.867 for ``mean_level=batch``.
    """
    norm = Normalization(NormConfig(mean_level="group", std_level=None, group_size=4))
    mask = torch.zeros(4, 10)
    lengths = [2, 4, 6, 8]
    per_row = torch.tensor([1.5, 0.5, -0.5, -1.5])  # sums to zero over ROWS
    advantages = torch.zeros(4, 10)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
        advantages[i, :length] = per_row[i]
    assert float(per_row.sum()) == pytest.approx(0.0, abs=1e-6)

    out = norm(advantages, mask, group_sizes=[4])
    row_sum = float((out * mask).sum(dim=1).div(torch.tensor(lengths).float()).sum())
    assert abs(row_sum) > 0.1, (
        "group-level adv_norm left the per-row advantages summing to zero; if this ever "
        "becomes true the GOAL.md constraint can be relaxed, and the relaxation must be "
        "argued rather than inherited"
    )

    # Equal lengths, same rows: the token mean and the row mean coincide and nothing moves.
    equal_mask = torch.zeros(4, 10)
    equal_adv = torch.zeros(4, 10)
    for i in range(4):
        equal_mask[i, :5] = 1.0
        equal_adv[i, :5] = per_row[i]
    equal_out = norm(equal_adv, equal_mask, group_sizes=[4])
    equal_row_sum = float((equal_out * equal_mask).sum(dim=1).div(5.0).sum())
    assert equal_row_sum == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------------------
# 2. The importance ratio on a row the policy never sampled
# ---------------------------------------------------------------------------------------


def _gold_row_batch(old_logprobs: torch.Tensor | None, *, drop: str | None = None) -> dict:
    """A microbatch whose first sequence stands in for a gold (never-sampled) row.

    Args:
        old_logprobs: Value for ``input_data['logprobs']`` -- the behaviour policy's
            log-probabilities, which a gold row does not have. ``None`` keeps the sampled
            ones, for the control.
        drop: Key to remove entirely, for the absent-key cases.

    Returns:
        The ``input_data`` dict.
    """
    torch.manual_seed(3)
    n = 8
    behaviour = torch.randn(n)
    data = {
        "logprobs": behaviour.clone() if old_logprobs is None else old_logprobs,
        "prox_logp": behaviour.clone(),
        "advantages": torch.full((n,), SFT_C),
        "loss_mask": torch.ones(n, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
        "input_ids": torch.randint(0, 100, (n,)),
    }
    if drop is not None:
        data.pop(drop)
    return data


def _run(data: dict, **kw) -> torch.Tensor:
    """Call the real loss on an 8-token batch."""
    return grpo_loss_fn(
        logprobs=torch.zeros(8, requires_grad=True),
        entropy=torch.zeros(8),
        input_data=data,
        eps_clip=EPS_CLIP,
        eps_clip_higher=None,
        c_clip=None,
        **kw,
    )


def test_nan_proximal_logprobs_raise_rather_than_propagate() -> None:
    """The one guarded slot. ``prox_logp`` is checked for NaN/Inf and refuses.

    ``areal/trainer/ppo/actor.py:1727-1731``. This is the tensor the PPO ratio is actually
    built from -- ``ratio = exp(logprobs - prox_logp)``, functional.py:538 -- so a gold row
    whose prox_logp were NaN would take the whole update down, and it is stopped loudly.
    """
    data = _gold_row_batch(None)
    data["prox_logp"] = torch.full((8,), float("nan"))
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        _run(data)

    data = _gold_row_batch(None)
    data["prox_logp"] = torch.full((8,), float("-inf"))
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        _run(data)


def test_nan_behaviour_logprobs_are_silently_ignored_by_the_default_surrogate() -> None:
    """``input_data['logprobs']`` is NOT the ratio's denominator and is not checked.

    With ``rejection_sampling=None`` and no teacher, ``old_logp`` reaches nothing that feeds
    the loss: the ratio at functional.py:538 uses ``prox_logp``. So a NaN there produces a
    finite loss and a gradient BIT-IDENTICAL to the clean batch. Nothing warns. This is the
    quiet half of the answer for substitute.py -- the loss will not tell you the field was
    wrong.
    """
    clean = _run(_gold_row_batch(None))
    poisoned = _run(_gold_row_batch(torch.full((8,), float("nan"))))
    assert torch.isfinite(poisoned)
    assert torch.equal(clean.detach(), poisoned.detach())


def test_nan_behaviour_logprobs_become_ratio_one_under_the_live_rejection_filter() -> None:
    """The live 30B config DOES read old_logp, and reads NaN as "perfectly on-policy".

    ``examples/math/gsm8k_grpo_lora.yaml:78-80`` switches on ``rejection_sampling``, whose
    ``behave_imp_weight = exp(prox_logp - old_logp)`` multiplies the surrogate
    (functional.py:568). ``functional.py:233`` replaces every non-finite log-ratio with 0.0,
    so a gold row with NaN behaviour logprobs gets weight ``exp(0) = 1`` exactly, is not
    filtered, and is scored as if the policy had sampled it with probability equal to the
    trainer's own. Silent ratio=1 -- the failure mode 2604.23747 is about, arriving through
    the sanitiser rather than through the normalisation.
    """
    result = apply_rejection_sampling(
        proximal_logprobs=torch.randn(8),
        old_logprobs=torch.full((8,), float("nan")),
        loss_mask=torch.ones(8, dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        config=LIVE_REJECTION,
    )
    assert torch.equal(result.behave_imp_weight, torch.ones(8))
    assert bool(result.loss_mask.all()), "the NaN row was not filtered out"
    assert result.filtered_fraction == 0.0
    assert torch.isfinite(
        _run(_gold_row_batch(torch.full((8,), float("nan"))),
             rejection_sampling=LIVE_REJECTION)
    )


def test_zero_placeholder_logprobs_shrink_the_gold_row_instead_of_flagging_it() -> None:
    """The other tempting placeholder is also wrong, and wrong quietly.

    Filling a gold row's behaviour logprobs with 0.0 (log p = 1) makes
    ``behave_imp_weight = exp(prox_logp)``, and prox_logp is negative for any real token, so
    the row's gradient is MULTIPLIED DOWN by that factor. At the modest prox_logp used here
    the row keeps well under half its weight, and nothing is logged as filtered. The only
    placeholder that leaves the surrogate alone is the trainer's own recomputed logp, which
    makes the weight exactly 1.
    """
    prox = torch.full((8,), -1.0)
    zeros = apply_rejection_sampling(
        proximal_logprobs=prox,
        old_logprobs=torch.zeros(8),
        loss_mask=torch.ones(8, dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        config=LIVE_REJECTION,
    )
    assert float(zeros.behave_imp_weight[0]) == pytest.approx(float(torch.exp(prox[0])), rel=1e-6)
    assert float(zeros.behave_imp_weight[0]) < 0.4
    assert zeros.filtered_fraction == 0.0, "the down-weighting is not reported anywhere"

    matched = apply_rejection_sampling(
        proximal_logprobs=prox,
        old_logprobs=prox.clone(),
        loss_mask=torch.ones(8, dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        config=LIVE_REJECTION,
    )
    assert torch.equal(matched.behave_imp_weight, torch.ones(8))


def test_absent_logprob_keys_fail_loudly() -> None:
    """Omitting a field is safe; supplying a wrong one is not. Both are pinned.

    ``prox_logp`` missing hits the configuration check at actor.py:1686-1691, and
    ``logprobs`` missing is an unguarded subscript at actor.py:1300. Neither can reach a
    training step, which is why the NaN paths above are the ones substitute.py has to care
    about.
    """
    with pytest.raises(ValueError, match="prox_logp is None"):
        _run(_gold_row_batch(None, drop="prox_logp"))
    with pytest.raises(KeyError):
        _run(_gold_row_batch(None, drop="logprobs"))


def test_reuse_train_logp_forces_the_ratio_to_exactly_one() -> None:
    """A configuration under which the ratio cannot depend on a gold row's logprobs at all.

    ``actor.py:1307-1308`` overwrites prox_logp with the training forward's own logprobs, so
    ``ratio = exp(0) = 1`` for every token and the gradient is exactly ``-A / N``. Recorded
    because it is the one setting in which a gold row needs no behaviour logprobs to be
    scored correctly -- and because a run that silently ends up here would report an
    importance-weighted result that was never importance weighted.
    """
    data = _gold_row_batch(None)
    data["versions"] = torch.zeros(8, dtype=torch.int32)
    logprobs = torch.randn(8, requires_grad=True)
    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=torch.zeros(8),
        input_data=data,
        eps_clip=EPS_CLIP,
        eps_clip_higher=None,
        c_clip=None,
        prox_logp_method="reuse_train_logp",
    )
    loss.backward()
    assert float(loss.detach()) == pytest.approx(-SFT_C, rel=1e-6)
    assert torch.allclose(logprobs.grad, torch.full((8,), -SFT_C / 8))


def test_nan_logprobs_poison_the_whole_batch_before_the_loss_ever_runs() -> None:
    """The worst path, and the reason substitute.py must write a FINITE logp.

    ``PPOActor._compute_advantages`` reads ``data['logprobs']`` for the KL reward at
    ``areal/trainer/ppo/actor.py:741``. ``kl_ctl`` is 0.0 in the live config, which does not
    help: ``-0.0 * NaN`` is NaN in IEEE754. The NaN then flows through GAE into that row's
    advantages, and batch-level ``adv_norm`` -- ``mean_level: batch`` in
    ``examples/math/gsm8k_grpo_lora.yaml:85-87`` -- takes a mean over the batch and hands
    every OTHER row the NaN too. One gold row destroys the update for all of them.
    """
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    actor = make_actor()
    assert actor.adv_norm is None and actor.kl_ctl == 0.0
    batch = make_batch(rewards)
    batch["logprobs"] = torch.zeros(B, T)
    batch["logprobs"][0, PROMPT:] = float("nan")
    advantages = actor._compute_advantages(batch, meta())["advantages"]
    nan_rows = torch.isnan(advantages).any(dim=1)
    assert bool(nan_rows[0]), "kl_ctl=0 did not stop the NaN: expected it to reach row 0"
    assert not bool(nan_rows[1:].any()), "without adv_norm the damage stays on its own row"

    # The live setting. The normalizer is cached at construction, so both the config and the
    # cached object are set, and the test would fail loudly if only one were honoured.
    actor = make_actor()
    actor.config.adv_norm = NormConfig(mean_level="batch", std_level="batch")
    actor.adv_norm = Normalization(actor.config.adv_norm)
    batch = make_batch(rewards)
    batch["logprobs"] = torch.zeros(B, T)
    batch["logprobs"][0, PROMPT:] = float("nan")
    advantages = actor._compute_advantages(batch, meta())["advantages"]
    assert bool(torch.isnan(advantages).any(dim=1).all()), (
        "batch-level adv_norm was expected to spread one row's NaN across every row"
    )
