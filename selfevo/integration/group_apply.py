"""Turn per-group routing decisions into an advantage tensor.

The actor currently hardcodes the rule -- solved groups get a constant, unsolved groups get
another -- which means the audited routers in :mod:`selfevo.routing` decide nothing during
training. This is the seam that lets a Router actually drive the update, so
``router=contextual`` and ``router=code_policy`` become real arms rather than registry
entries.

Kept as a pure tensor function, outside the actor, for one reason: the actor is imported by
live training processes, so logic that lives there can only be tested by running a job.
Everything here is testable on CPU in milliseconds, and the actor's diff is a call.

Mode semantics, chosen so a decision means the same thing whether or not the group was
already silent:

    RL    leave the advantages alone -- ordinary GRPO for this unit.
    SFT   REPLACE the response-token advantages with ``sft_weight``. For a silent group its
          advantages are already zero so replace and add coincide; for an informative group
          they do not, and replace is what "train this unit by SFT instead of RL" means.
          Adding would leave the RL gradient in place and superimpose a supervised one.
    SKIP  zero the advantages. For a silent group this is a no-op; for an informative one it
          is the decision to spend nothing on this unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from selfevo.routing.base import TrainingMode

__all__ = ["ApplyStats", "apply_decisions"]

_APPLIED = (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP)


@dataclass(frozen=True)
class ApplyStats:
    """What a batch's decisions did, for logging next to the run.

    Args:
        counts: ``{mode: number of groups}``.
        changed_rows: Rows whose advantages the decision actually altered. Distinct from the
            count of non-RL groups: a SKIP on an already-silent group changes nothing, and
            reporting it as an intervention would overstate the method's reach.
        n_groups: Groups seen.
    """

    counts: dict[str, int]
    changed_rows: int
    n_groups: int

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics, prefixed so they do not collide with the actor's own keys."""
        out = {f"route/{m}_groups": float(n) for m, n in self.counts.items()}
        out["route/changed_row_fraction"] = (
            self.changed_rows / max(sum(self.counts.values()), 1) if self.n_groups else 0.0
        )
        out["route/n_groups"] = float(self.n_groups)
        return out


def apply_decisions(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    group_sizes: list[int],
    modes: list[str],
    *,
    sft_weight: float,
) -> tuple[torch.Tensor, ApplyStats]:
    """Apply one mode per group to the advantage tensor.

    Args:
        advantages: ``(B, T)``.
        loss_mask: ``(B, T)``, 1 on response tokens. The mask bounds every write, so a
            decision can never put gradient on a prompt.
        group_sizes: Row counts per group; must sum to ``B``.
        modes: One mode per group, same order.
        sft_weight: Magnitude written for an SFT group. Must be >= 0: SFT is training toward
            a target believed correct, and a negative weight would train away from it.

    Returns:
        ``(advantages, stats)``. The input tensor is not modified in place -- the caller may
        still hold a reference to it, and silently mutating it has already caused one bug in
        this pipeline.

    Raises:
        ValueError: On a shape mismatch, a grouping that does not partition the batch, an
            unknown or unsupported mode, or a negative ``sft_weight``.
    """
    if advantages.shape != loss_mask.shape:
        raise ValueError(
            f"advantages {tuple(advantages.shape)} and loss_mask {tuple(loss_mask.shape)} "
            "must have the same shape"
        )
    b = advantages.shape[0]
    if sum(group_sizes) != b:
        raise ValueError(f"group_sizes sums to {sum(group_sizes)}, batch has {b} rows")
    if len(modes) != len(group_sizes):
        raise ValueError(
            f"{len(modes)} modes for {len(group_sizes)} groups; one decision per group"
        )
    if sft_weight < 0:
        raise ValueError(
            f"sft_weight must be >= 0, got {sft_weight}: SFT trains TOWARD a target believed "
            "correct, so a negative weight trains away from it"
        )
    unknown = sorted(set(modes) - set(_APPLIED))
    if unknown:
        raise ValueError(
            f"cannot apply modes {unknown}; this seam implements {list(_APPLIED)}. A "
            "teacher-requiring mode needs a target tensor that does not exist here, and "
            "silently treating it as SKIP would report a distillation arm that never ran."
        )

    out = advantages.clone()
    mask = loss_mask.to(out.dtype)
    counts = {m: 0 for m in _APPLIED}
    changed = 0
    start = 0
    for g, mode in zip(group_sizes, modes):
        sl = slice(start, start + g)
        start += g
        counts[mode] += 1
        if mode == TrainingMode.RL:
            continue
        before = out[sl]
        new = (
            torch.full_like(before, float(sft_weight)) * mask[sl]
            if mode == TrainingMode.SFT
            else torch.zeros_like(before)
        )
        changed += int((before != new).any(dim=-1).sum())
        out[sl] = new
    return out, ApplyStats(counts=counts, changed_rows=changed, n_groups=len(group_sizes))
