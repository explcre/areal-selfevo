"""Per-group observability features: the inputs a learned controller decides from.

``RoutingContext.extra`` was designed as the feature seam and had no producer, so every
router in this project decides from ``solve_rate`` alone. That is enough for a hand-written
threshold and not enough for a policy worth learning: a rule keyed on one scalar is a
heuristic, and a controller that sees the same one scalar cannot be better than the best
threshold on it.

Everything here is computed from tensors the batch **already carries** -- rewards, the loss
mask and the sampled log-probabilities. No extra forward pass, no extra generation, so
turning features on costs nothing measurable. That constraint is deliberate: a feature set
that needs its own inference budget would have to beat spending that budget on more rollouts
instead, which is a much higher bar than beating a threshold.

Features are scale-free wherever a scale exists (dispersions are reported relative to their
own mean), because a linear policy over raw lengths would mostly learn the tokenizer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

__all__ = ["GroupFeatures", "group_features", "FEATURE_NAMES"]


def _safe_div(num: float, den: float, *, default: float = 0.0) -> float:
    """``num / den``, returning ``default`` when the denominator is ~0 or the result is not finite.

    A NaN reaching a policy's feature vector is worse than a wrong number: it poisons every
    downstream weight silently, whereas a 0.0 is merely uninformative.
    """
    if abs(den) < 1e-12:
        return default
    out = num / den
    return out if out == out and abs(out) != float("inf") else default


def _finite(value: float, *, default: float = 0.0) -> float:
    """``value`` when it is finite, else ``default``.

    Every field of :class:`GroupFeatures` is guaranteed finite, and ``_safe_div`` only
    covers the fields that are ratios. This covers the plain reductions: a NaN or an
    infinity in ``loss_mask`` propagates through a mean or a standard deviation without
    ever passing through a division, and a single non-finite feature poisons a linear
    policy's arm permanently while no downstream metric shows anything.
    """
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class GroupFeatures:
    """What one GRPO group looks like, before any decision is taken about it.

    Args:
        solve_rate: Fraction of samples with positive reward. The one feature every existing
            router already uses; kept so a contextual policy strictly extends them.
        reward_std: Population standard deviation of the group's rewards. Zero exactly when
            the group is RL-silent, so it carries the silence flag without a threshold.
        mean_response_len: Mean number of response tokens.
        len_dispersion: Response-length standard deviation divided by the mean. Scale-free.
            High dispersion means the group disagrees about how long the answer should be,
            which distinguishes "hard" from "long".
        mean_logprob: Mean per-token log-probability of the sampled responses. A confidence
            proxy that costs nothing, since the sampler already returned it.
        logprob_dispersion: Standard deviation of per-sample mean log-probability, divided by
            the absolute mean. Behavioural diversity within the group: a group that is
            unanimous in ANSWER may still be diverse in reasoning, and those two cases want
            different treatment.
        truncated_fraction: Fraction of samples whose response reached the token budget.
            Truncation is scored as failure by the grader, so a unit that looks unsolved may
            only be out of budget -- a distinction a solve-rate-only router cannot make.
    """

    solve_rate: float
    reward_std: float
    mean_response_len: float
    len_dispersion: float
    mean_logprob: float
    logprob_dispersion: float
    truncated_fraction: float

    def as_extra(self) -> dict[str, float]:
        """Feature mapping suitable for ``RoutingContext.extra``."""
        return {k: float(v) for k, v in asdict(self).items()}


FEATURE_NAMES: tuple[str, ...] = tuple(GroupFeatures.__dataclass_fields__)


def group_features(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    logprobs: torch.Tensor,
    group_sizes: list[int],
    *,
    max_response_len: int | None = None,
    reward_threshold: float = 0.5,
) -> list[GroupFeatures]:
    """Feature vectors for each group in a batch.

    Args:
        rewards: ``(B,)`` per-sample scalar rewards, RAW -- before bias, scaling, clipping or
            normalisation. Those transforms make an all-correct group indistinguishable from
            an all-wrong one, which is how an earlier metric in this project came to report
            0% solved at an 82% solve rate.
        loss_mask: ``(B, T)``, 1 on response tokens.
        logprobs: ``(B, T)`` per-token log-probabilities, aligned with ``loss_mask``.
        group_sizes: Row counts per group; must sum to ``B``.
        max_response_len: Token budget. When given, a response of exactly this length counts
            as truncated. When ``None`` the truncated fraction is 0.0 rather than guessed.
        reward_threshold: Rewards strictly above this count as correct.

    Returns:
        One :class:`GroupFeatures` per group, in row order.

    Raises:
        ValueError: If the shapes disagree or ``group_sizes`` does not partition the batch.
            A silently wrong grouping would attribute one group's features to another, which
            is unrecoverable downstream. Also if ``rewards`` holds a value that is not
            finite in float32: a NaN reward is counted as *incorrect* by ``solve_rate`` and
            as *unanimous* by any guard on ``reward_std``, so accepting one would report a
            grader bug as data.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    for name, t in (("loss_mask", loss_mask), ("logprobs", logprobs)):
        if t.ndim != 2:
            raise ValueError(f"{name} must be 2-D (B, T), got shape {tuple(t.shape)}")
    b = rewards.shape[0]
    for name, t in (("loss_mask", loss_mask), ("logprobs", logprobs)):
        if t.shape[0] != b:
            raise ValueError(f"{name} has {t.shape[0]} rows, rewards has {b}")
    if loss_mask.shape != logprobs.shape:
        raise ValueError(
            f"loss_mask {tuple(loss_mask.shape)} and logprobs {tuple(logprobs.shape)} "
            "must have the same shape; they are indexed together"
        )
    if sum(group_sizes) != b:
        raise ValueError(f"group_sizes sums to {sum(group_sizes)}, batch has {b} rows")
    if any(g < 1 for g in group_sizes):
        raise ValueError(f"every group size must be >= 1, got {group_sizes}")
    if max_response_len is not None and max_response_len < 1:
        raise ValueError(f"max_response_len must be >= 1, got {max_response_len}")
    # Cast first: every feature is computed in float32, and a float64 reward of 1e300 is
    # finite until it is cast, after which reward_std is NaN and any guard on it reports
    # the group as unanimous -- the exact opposite of what it is.
    if not bool(torch.isfinite(rewards.to(torch.float32)).all()):
        raise ValueError(
            "rewards contains a value that is not finite in float32; solve_rate would "
            "silently score it as incorrect and reward_std would be NaN"
        )

    mask = loss_mask.to(torch.float32)
    lengths = mask.sum(dim=-1)                                    # (B,)
    # Zeroed before the multiply, not by it: a non-finite log-prob at a PROMPT position
    # survives ``nan * 0 == nan`` and would contaminate a sum it is not part of, so the
    # response-only guarantee would hold only for well-behaved padding.
    kept = logprobs.to(torch.float32).masked_fill(mask == 0, 0.0)
    tok_lp = (kept * mask).sum(dim=-1)                            # (B,)
    per_sample_lp = torch.stack(
        [torch.tensor(_safe_div(float(s), float(n))) for s, n in zip(tok_lp, lengths)]
    )

    out: list[GroupFeatures] = []
    start = 0
    for g in group_sizes:
        sl = slice(start, start + g)
        start += g
        r = rewards[sl].to(torch.float32)
        ln = lengths[sl]
        lp = per_sample_lp[sl]

        mean_len = _finite(float(ln.mean()))
        mean_lp = _finite(float(lp.mean()))
        # Population std: a singleton group has dispersion 0, not NaN.
        r_std = _finite(float(r.std(unbiased=False)))
        len_std = _finite(float(ln.std(unbiased=False)))
        lp_std = _finite(float(lp.std(unbiased=False)))

        trunc = (
            float((ln >= max_response_len).to(torch.float32).mean())
            if max_response_len is not None
            else 0.0
        )
        out.append(
            GroupFeatures(
                solve_rate=float((r > reward_threshold).to(torch.float32).mean()),
                reward_std=r_std,
                mean_response_len=mean_len,
                len_dispersion=_safe_div(len_std, mean_len),
                mean_logprob=mean_lp,
                logprob_dispersion=_safe_div(lp_std, abs(mean_lp)),
                truncated_fraction=trunc,
            )
        )
    return out
