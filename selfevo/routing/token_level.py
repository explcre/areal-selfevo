"""Token-level gating: which positions carry an RL gradient, and which are free.

Companion to :mod:`selfevo.routing.criteria`, which answers the same question at the level
of a whole group. This module answers it per token.

The result it rests on (verified numerically in
``experiments/harness/prefix_cancellation.py``):

    At a position inside a prefix shared by every group member, the context and the emitted
    token are identical across members, so ``grad log pi`` is one common vector ``g`` and
    the group contributes ``(sum_i w_i) * g``.

    * vanilla policy gradient, ``w_i = A_i``: ``sum_i A_i = 0`` by construction of the
      centred advantage, so the contribution is **exactly zero**;
    * PPO with clipping, ``w_i = r_i * A_i``: clipping drops members by the *sign* of
      ``A_i``, an asymmetric subset, so the survivors do **not** sum to zero. Measured net
      weight -1.50 at ratio 1.5 and +0.50 at ratio 0.5.

So the cancellation is a property of *on-policy* updates, not of GRPO in general. Anything
that gates on it must assert the condition rather than assume it -- see
:func:`assert_on_policy`. Both quantities it needs are already logged every step.

What to do with a dead position is a separate decision, and one easy to get wrong:
re-imitating the group's own consensus there (``y*_t = y_t``) merely sharpens a policy that
is already collapsing. The useful signal is an *external* one evaluated at the policy's own
context -- on-policy distillation -- which aligns by construction because the teacher scores
the contexts the policy actually produced.
"""

from __future__ import annotations

import torch

__all__ = [
    "OnPolicyViolation",
    "assert_on_policy",
    "assert_zero_sum_advantage",
    "shared_prefix_lengths",
    "rl_dead_mask",
    "token_gates",
]


class OnPolicyViolation(RuntimeError):
    """Raised when the update is not on-policy, voiding the prefix-cancellation argument."""


def assert_on_policy(
    importance_weight: torch.Tensor,
    clip_fraction: float,
    *,
    ratio_tol: float = 1e-4,
    clip_tol: float = 0.0,
) -> None:
    """Check the condition under which shared-prefix positions are RL-dead.

    Args:
        importance_weight: Per-token ratio ``pi_theta / pi_old``. Every element must be
            within ``ratio_tol`` of 1; checking the mean is not enough, because offsetting
            deviations average to 1 while individually breaking the cancellation.
        clip_fraction: Fraction of tokens whose PPO ratio was clipped.
        ratio_tol: Allowed absolute deviation from 1.
        clip_tol: Allowed clipped fraction. Defaults to 0: any clipping at all removes an
            asymmetric subset of group members and breaks the argument.

    Raises:
        OnPolicyViolation: If the ratio deviates or clipping occurred. Callers should fall
            back to sample-level routing rather than proceed.
    """
    if importance_weight.numel() > 0:
        dev = (importance_weight - 1.0).abs().max().item()
        if dev > ratio_tol:
            raise OnPolicyViolation(
                f"max |ratio - 1| = {dev:.3e} exceeds {ratio_tol:.1e}; the update is not "
                "on-policy, so shared-prefix positions are not RL-dead"
            )
    if clip_fraction > clip_tol:
        raise OnPolicyViolation(
            f"clip fraction {clip_fraction:.3e} exceeds {clip_tol:.1e}; clipping removes "
            "group members asymmetrically, so advantages no longer sum to zero"
        )


def shared_prefix_lengths(
    tokens: torch.Tensor, gen_mask: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    """Length of the longest common prefix of generated tokens within each group.

    Measured in GENERATED-TOKEN RANK, not column index: the members agree on their
    first ``k`` generated tokens, wherever those sit in the padded row. This is the
    coordinate system ``rl_dead_mask`` indexes with, and the two disagreed before.

    Only *generated* positions count. The prompt is shared by construction and is masked
    out of the loss anyway, so including it would inflate every prefix and make the gate
    look far more useful than it is.

    Args:
        tokens: ``(B, T)`` token ids.
        gen_mask: ``(B, T)`` bool/0-1, True where the token was generated (i.e. trained on).
        group_ids: ``(B,)`` integer group label; rows sharing a label are one GRPO group.

    Returns:
        ``(B,)`` int64. Every row of a group gets that group's common-prefix length, so the
        result can be compared elementwise against a position index.

    Raises:
        ValueError: On shape mismatch.
    """
    if tokens.ndim != 2 or gen_mask.shape != tokens.shape:
        raise ValueError(
            f"tokens and gen_mask must be (B, T) and match; got {tuple(tokens.shape)} "
            f"and {tuple(gen_mask.shape)}"
        )
    if group_ids.ndim != 1 or group_ids.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"group_ids must be (B,) matching batch {tokens.shape[0]}; "
            f"got {tuple(group_ids.shape)}"
        )

    out = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    for gid in torch.unique(group_ids):
        rows = (group_ids == gid).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            # A group of one has A = r - rbar = 0 identically, so *every* position is dead
            # rather than none. Reporting a prefix of 0 here would be exactly backwards.
            out[rows] = int(gen_mask[rows].sum().item())
            continue
        member_tokens = tokens[rows]
        member_gen = gen_mask[rows].bool()
        # Compare the k-th GENERATED token of each member, not the k-th column.
        #
        # The previous version counted consecutive columns from the first column where every
        # member generates, while rl_dead_mask indexes by gen_rank (rank within the row).
        # Those coincide only when all members start generating at the same column with a
        # contiguous mask. When they do not -- different prompt lengths, or a multi-turn mask
        # with tool results interleaved -- the count was applied in the wrong coordinate
        # system: an audit case with row 0 generating at columns 2-4 and row 1 at 1-4 routed
        # a token no other member emitted while leaving a genuinely shared one on RL.
        #
        # Rank space is also the correct reading of "shared prefix": the members agree on
        # their first k generated tokens, wherever those sit in the padded row.
        seqs = [member_tokens[i][member_gen[i]] for i in range(rows.numel())]
        shortest = min(int(x.numel()) for x in seqs)
        length = 0
        for k in range(shortest):
            first = seqs[0][k]
            if not all(bool(x[k] == first) for x in seqs[1:]):
                break
            length += 1
        out[rows] = length
    return out


def rl_dead_mask(
    tokens: torch.Tensor, gen_mask: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token mask of positions whose net RL gradient is zero.

    A generated position is dead when it lies inside its group's shared prefix.

    Args:
        tokens: ``(B, T)`` token ids.
        gen_mask: ``(B, T)`` True where the token was generated.
        group_ids: ``(B,)`` group labels.

    Returns:
        ``(B, T)`` bool. True at generated positions inside the shared prefix; always False
        at non-generated positions, which carry no loss to redirect in the first place.
    """
    lengths = shared_prefix_lengths(tokens, gen_mask, group_ids)
    b, t = tokens.shape
    gen = gen_mask.bool()
    # Index generated positions by their rank within the row, so "inside the prefix" means
    # rank < prefix length regardless of where the prompt ends.
    gen_rank = torch.cumsum(gen.long(), dim=-1) - 1
    within = gen_rank < lengths.unsqueeze(1)
    return gen & within & (gen_rank >= 0)


def token_gates(
    tokens: torch.Tensor,
    gen_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    teacher_available: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token weights for the RL term and the teacher term.

    Args:
        tokens: ``(B, T)`` token ids.
        gen_mask: ``(B, T)`` True where generated.
        group_ids: ``(B,)`` group labels.
        teacher_available: Whether an external teacher signal can be evaluated at these
            positions. When False the teacher gate is all zeros -- dead positions are simply
            left unweighted rather than filled with self-imitation, which would reinforce
            the group's own consensus and sharpen an already-collapsing policy.

    Returns:
        ``(alpha_rl, alpha_teacher)``, each ``(B, T)`` float in {0, 1} and disjoint on
        generated positions. Non-generated positions are zero in both.
    """
    dead = rl_dead_mask(tokens, gen_mask, group_ids)
    gen = gen_mask.bool()
    alpha_rl = (gen & ~dead).to(tokens.device, torch.float32)
    alpha_teacher = (
        dead.to(torch.float32) if teacher_available else torch.zeros_like(alpha_rl)
    )
    return alpha_rl, alpha_teacher


def assert_zero_sum_advantage(
    advantages: torch.Tensor, group_ids: torch.Tensor, *, tol: float = 1e-4
) -> None:
    """Check that each group's advantages still sum to zero.

    The prefix-cancellation argument needs ``sum_i A_i = 0`` *per group*. Centring gives
    that for free, but several standard modifications destroy it, and they do so silently:

    * **Entropy-bonus shaping.** MEDS (`The Past Is Not Past`, 2604.11297) does exactly
      this in `verl/workers/actor/dp_actor.py:560`::

          advantages += torch.min(0.4 * entropy.detach(), advantages.abs() / 2)

      Entropy is non-negative, so this adds a non-negative quantity to every element and
      the group sum becomes strictly positive wherever entropy is non-zero.
    * **Per-sequence length normalisation** (``A_i / |y_i|``), where unequal lengths leave
      a non-zero residual -- the same length bias Dr. GRPO and DAPO address.
    * Any reward shaper applied after centring.

    So this is not a redundant assertion: it fails precisely for the shaping schemes we
    are most likely to want to combine with token-level routing.

    Args:
        advantages: ``(B,)`` per-sequence or ``(B, T)`` per-token advantages. Per-token
            input is reduced by taking each row's mean over non-zero entries, matching how
            a sequence-level advantage is broadcast.
        group_ids: ``(B,)`` group labels.
        tol: Allowed absolute deviation of a group's sum from zero.

    Raises:
        OnPolicyViolation: If any group's advantages do not sum to zero within ``tol``.
            Callers must fall back to sample-level routing.
        ValueError: On shape mismatch.
    """
    if advantages.ndim == 2:
        nz = (advantages != 0).sum(dim=-1).clamp(min=1)
        per_seq = advantages.sum(dim=-1) / nz
    elif advantages.ndim == 1:
        per_seq = advantages
    else:
        raise ValueError(f"advantages must be (B,) or (B, T), got {tuple(advantages.shape)}")
    if group_ids.ndim != 1 or group_ids.shape[0] != per_seq.shape[0]:
        raise ValueError(
            f"group_ids must be (B,) matching batch {per_seq.shape[0]}; "
            f"got {tuple(group_ids.shape)}"
        )
    for gid in torch.unique(group_ids):
        rows = group_ids == gid
        total = per_seq[rows].sum().item()
        if abs(total) > tol:
            raise OnPolicyViolation(
                f"group {int(gid)} advantages sum to {total:.3e}, not 0 (tol {tol:.1e}); "
                "centring has been broken by shaping or length normalisation, so "
                "shared-prefix positions are not RL-dead"
            )
