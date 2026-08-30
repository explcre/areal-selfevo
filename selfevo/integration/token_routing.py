"""Per-token routing between the RL and distillation objectives.

AReaL already mixes the two, at ``areal/trainer/ppo/actor.py``::

    loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

with both weights **batch-level scalars**. This module makes them **per-token**, decided by
a router. That is the concrete form of this project's claim: the training signal is chosen
per token rather than per run.

The first routing rule follows from the estimator, but it is CONDITIONAL and the conditions
are frequently violated. In a GRPO group whose advantages sum to zero, and on-policy so every
member's importance ratio is 1, a token shared by every member receives exactly cancelling
gradient contributions and its net RL gradient is zero. Neither precondition is automatic:

* Batch-level ``adv_norm`` (set in this repo's own ``gsm8k_grpo.yaml``) subtracts a
  token-weighted batch mean that is not zero when generations differ in length, which
  destroys per-group centring. Measured on the real pipeline: ``sum_i A_i = 0.98``, 87-115%
  of mean ``|A|``, leaving 9.7% of the live gradient at a supposedly dead prefix.
* ``kl_ctl > 0`` (step0i uses 0.01) makes each member accumulate its own future KL from the
  shared position onward, so advantages differ beyond the centred reward: 14-43% of mean
  ``|A|``.
* PPO clipping breaks it, and so does merely unequal per-member ratios with no clipping at
  all. Under GSPO the ratio is a whole-sequence geometric mean, so it differs across members
  even where the per-token log-ratio at the prefix is identically zero.

So the rule is only valid when the guards below pass, and this module CHECKS rather than
assumes. ``assert_zero_sum_advantage`` and ``assert_on_policy`` already existed in
``routing/token_level.py`` for exactly this and had no callers anywhere.

**Rollback is the default.** With ``enabled=False`` the module returns the caller's own
scalars unchanged, so the loss expression is bit-identical to upstream AReaL. Every feature
here is reachable only through an explicit argument, and
``tests/test_token_routing.py::test_disabled_is_bit_identical_to_upstream`` executes both
paths and compares tensors rather than asserting the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .packed import PackedLayoutError, is_packed, repack, unpack
from ..routing.token_level import (
    assert_on_policy,
    assert_zero_sum_advantage,
    rl_dead_mask,
)

__all__ = ["TokenRoutingSpec", "TokenRoutingResult", "route_token_weights"]


@dataclass(frozen=True)
class TokenRoutingSpec:
    """Configuration for per-token signal routing.

    Attributes:
        enabled: When False every other field is ignored and the caller's scalars are
            returned unchanged. This is the rollback path and the default.
        rule: Which router decides. ``"rl_dead_to_distill"`` sends tokens with provably zero
            RL gradient to the teacher. ``"all_rl"`` and ``"all_distill"`` are degenerate
            controls that must be beaten. ``"random"`` matches the routed proportion but
            shuffles WHICH tokens are routed -- the control that separates "routing helped"
            from "changing the RL/KD ratio helped", which is the confound this whole
            comparison exists to rule out.
        dead_rl_weight: RL weight applied at RL-dead positions. Zero by definition of dead;
            configurable only so the claim can be tested rather than assumed.
        dead_distill_weight: Distillation weight at RL-dead positions. ``None`` means reuse
            the caller's scalar.
        seed: Seed for the ``random`` control, so a run is reproducible.
    """

    enabled: bool = False
    rule: str = "rl_dead_to_distill"
    dead_rl_weight: float = 0.0
    dead_distill_weight: float | None = None
    seed: int = 0
    require_valid_preconditions: bool = True

    RULES = ("rl_dead_to_distill", "all_rl", "all_distill", "random")

    def __post_init__(self) -> None:
        if self.rule not in self.RULES:
            raise ValueError(f"unknown rule {self.rule!r}; expected one of {self.RULES}")
        if not 0.0 <= self.dead_rl_weight:
            raise ValueError(f"dead_rl_weight must be >= 0, got {self.dead_rl_weight}")
        if self.dead_distill_weight is not None and self.dead_distill_weight < 0.0:
            raise ValueError("dead_distill_weight must be >= 0 or None")


@dataclass(frozen=True)
class TokenRoutingResult:
    """Per-token weights plus what the router actually did.

    Attributes:
        rl_weight: Scalar (disabled) or ``(B, T)`` tensor of RL weights.
        distill_weight: Scalar (disabled) or ``(B, T)`` tensor of distillation weights.
        routed_fraction: Fraction of *loss-carrying* tokens routed away from RL. Reported
            because a router that routes nothing and one that routes everything both look
            like "no effect" in the loss but are different failures.
        n_routed: Absolute count, so a zero fraction can be told from an empty batch.
        basis: What the decision rested on.
    """

    rl_weight: Any
    distill_weight: Any
    routed_fraction: float
    n_routed: int
    basis: str


def route_token_weights(
    *,
    spec: TokenRoutingSpec,
    rl_loss_weight: float,
    distill_loss_weight: float,
    loss_mask: torch.Tensor,
    tokens: torch.Tensor | None = None,
    gen_mask: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    importance_weight: torch.Tensor | None = None,
    clip_fraction: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> TokenRoutingResult:
    """Decide per-token RL and distillation weights.

    Args:
        spec: Routing configuration. ``enabled=False`` returns the inputs untouched.
        rl_loss_weight: The caller's batch-level RL weight.
        distill_loss_weight: The caller's batch-level distillation weight.
        loss_mask: ``(B, T)`` mask of positions that carry loss.
        tokens: ``(B, T)`` token ids. Required by ``rl_dead_to_distill``.
        gen_mask: ``(B, T)`` True where generated. Required by ``rl_dead_to_distill``.
        group_ids: ``(B,)`` GRPO group labels. Required by ``rl_dead_to_distill``.

    Returns:
        A :class:`TokenRoutingResult`. When disabled, ``rl_weight`` and ``distill_weight``
        are the caller's own floats, so arithmetic downstream is unchanged.

    Raises:
        ValueError: If an enabled rule is missing an input it needs. Silently falling back
            to the scalar path would make a disabled router indistinguishable from a
            misconfigured one, and the experiment would report a null result for the wrong
            reason.
    """
    if not spec.enabled:
        return TokenRoutingResult(
            rl_weight=rl_loss_weight,
            distill_weight=distill_loss_weight,
            routed_fraction=0.0,
            n_routed=0,
            basis="disabled: upstream scalar weights returned unchanged",
        )

    # AReaL hands the loss a PACKED microbatch: 1-D [total_length] with cu_seqlens marking
    # sequence boundaries. The routing functions want (B, T). Reshaping a packed tensor to
    # (1, total) reads the whole microbatch as ONE group and marks 100% of tokens RL-dead --
    # a confident wrong answer rather than a crash, which is why this is handled explicitly.
    packed = is_packed(loss_mask, cu_seqlens)
    if loss_mask.ndim == 1 and not packed:
        raise ValueError(
            "loss_mask is 1-D but no cu_seqlens was given. Sequence boundaries cannot be "
            "recovered from a packed tensor alone, and assuming one sequence would mark "
            "every token RL-dead. Pass cu_seqlens, or route on an unpacked batch."
        )
    if packed:
        original_shape_1d = True
        loss_mask = unpack(loss_mask, cu_seqlens)
        if tokens is not None and tokens.ndim == 1:
            tokens = unpack(tokens, cu_seqlens)
        if gen_mask is not None and gen_mask.ndim == 1:
            gen_mask = unpack(gen_mask, cu_seqlens)
        if advantages is not None and advantages.ndim == 1:
            advantages = unpack(advantages, cu_seqlens)
        if importance_weight is not None and importance_weight.ndim == 1:
            importance_weight = unpack(importance_weight, cu_seqlens)
    else:
        original_shape_1d = False

    mask = loss_mask.bool()
    n_live = int(mask.sum().item())

    if spec.rule == "all_rl":
        dead = torch.zeros_like(mask)
        basis = "control: no token routed away from RL"
    elif spec.rule == "all_distill":
        dead = mask.clone()
        basis = "control: every loss-carrying token routed to distillation"
    elif spec.rule == "rl_dead_to_distill":
        missing = [n for n, v in (("tokens", tokens), ("gen_mask", gen_mask),
                                  ("group_ids", group_ids)) if v is None]
        if missing:
            raise ValueError(
                f"rule 'rl_dead_to_distill' needs {missing}; refusing to fall back to the "
                "scalar path, which would be indistinguishable from the router being off"
            )
        # The cancellation argument is CONDITIONAL. Check it rather than assume it: batch
        # adv_norm and kl_ctl>0 -- both set in this repo's live configs -- were measured to
        # leave 9.7% of the live gradient at a supposedly dead prefix. These guards already
        # existed with no callers, which is how the violation went unnoticed.
        if spec.require_valid_preconditions:
            if advantages is None:
                raise ValueError(
                    "rl_dead_to_distill needs `advantages` to verify sum_i A_i = 0. Pass "
                    "require_valid_preconditions=False only to deliberately measure the "
                    "rule outside its domain."
                )
            assert_zero_sum_advantage(advantages, group_ids)
            if importance_weight is not None:
                assert_on_policy(importance_weight, clip_fraction if clip_fraction is not None else 0.0)
        dead = rl_dead_mask(tokens, gen_mask, group_ids) & mask
        basis = ("tokens inside the group's shared prefix, preconditions "
                 + ("verified" if spec.require_valid_preconditions else "NOT CHECKED"))
    elif spec.rule == "random":
        if group_ids is None or tokens is None or gen_mask is None:
            raise ValueError("rule 'random' needs the same inputs, to match the real rate")
        real = rl_dead_mask(tokens, gen_mask, group_ids) & mask
        k = int(real.sum().item())
        # Match the COUNT, shuffle the positions: isolates "which tokens" from "how many".
        flat = mask.flatten()
        idx = torch.nonzero(flat, as_tuple=False).flatten()
        g = torch.Generator(device="cpu").manual_seed(spec.seed)
        pick = idx[torch.randperm(idx.numel(), generator=g)[:k]] if k else idx[:0]
        dead = torch.zeros_like(flat)
        dead[pick] = True
        dead = dead.view_as(mask)
        basis = f"control: {k} randomly chosen loss-carrying tokens, matching the real rate"
    else:  # unreachable; __post_init__ validates
        raise ValueError(f"unknown rule {spec.rule!r}")

    # F8: defaulting to the caller's own weight made `distill_weight` a CONSTANT tensor, so
    # the default rule only deleted RL and never routed anything *to* the teacher -- a
    # provable no-op under the module's own theory. A routed token must actually gain
    # teacher weight, so the default is now to take over the budget the RL side gave up.
    dd = (spec.dead_distill_weight if spec.dead_distill_weight is not None
          else distill_loss_weight + max(0.0, rl_loss_weight - spec.dead_rl_weight))
    rl_w = torch.where(dead, torch.full_like(mask, spec.dead_rl_weight, dtype=torch.float32),
                       torch.full_like(mask, float(rl_loss_weight), dtype=torch.float32))
    kd_w = torch.where(dead, torch.full_like(mask, float(dd), dtype=torch.float32),
                       torch.full_like(mask, float(distill_loss_weight), dtype=torch.float32))
    n_routed = int(dead.sum().item())
    if original_shape_1d:
        # Hand back the layout the caller gave us; a padded (n_seq, max_len) weight tensor
        # cannot multiply a packed 1-D advantage.
        rl_w = repack(rl_w, cu_seqlens)
        kd_w = repack(kd_w, cu_seqlens)
    return TokenRoutingResult(
        rl_weight=rl_w,
        distill_weight=kd_w,
        routed_fraction=(n_routed / n_live) if n_live else 0.0,
        n_routed=n_routed,
        basis=f"{spec.rule}: {basis}; routed {n_routed}/{n_live} loss-carrying tokens",
    )
