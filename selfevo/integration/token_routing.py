"""Per-token routing between the RL and distillation objectives.

AReaL already mixes the two, at ``areal/trainer/ppo/actor.py``::

    loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

with both weights **batch-level scalars**. This module makes them **per-token**, decided by
a router. That is the concrete form of this project's claim: the training signal is chosen
per token rather than per run.

The first routing rule follows from the estimator, not from taste. In a GRPO group the
advantages sum to zero, so any token shared by every member of the group receives exactly
cancelling gradient contributions -- its net RL gradient is zero regardless of how the group
scored. Spending the loss budget there buys nothing. A teacher distribution, where one
exists, is strictly more informative than zero. So RL-dead tokens are routed to
distillation and live tokens keep RL.

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

from ..routing.token_level import rl_dead_mask

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
        dead = rl_dead_mask(tokens, gen_mask, group_ids) & mask
        basis = "tokens inside the group's shared prefix carry zero net RL gradient"
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

    dd = spec.dead_distill_weight if spec.dead_distill_weight is not None else distill_loss_weight
    rl_w = torch.where(dead, torch.full_like(mask, spec.dead_rl_weight, dtype=torch.float32),
                       torch.full_like(mask, float(rl_loss_weight), dtype=torch.float32))
    kd_w = torch.where(dead, torch.full_like(mask, float(dd), dtype=torch.float32),
                       torch.full_like(mask, float(distill_loss_weight), dtype=torch.float32))
    n_routed = int(dead.sum().item())
    return TokenRoutingResult(
        rl_weight=rl_w,
        distill_weight=kd_w,
        routed_fraction=(n_routed / n_live) if n_live else 0.0,
        n_routed=n_routed,
        basis=f"{spec.rule}: {basis}; routed {n_routed}/{n_live} loss-carrying tokens",
    )
