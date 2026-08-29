#!/usr/bin/env python3
"""Is the RL gradient on a group's shared prefix exactly zero, or only approximately?

The design doc marks this as a hypothesis. This checks it numerically, because the answer
changes what the token-level router is allowed to claim.

The argument: with a sequence-level centred advantage, member i contributes
``w_i * grad log pi(y_t | ctx_t)`` at position t. On a prefix shared by every member, both
the context and the emitted token are identical across members, so ``grad log pi`` is a
single common vector `g`. The group's contribution at that position is therefore
``(sum_i w_i) * g``.

- Vanilla policy gradient: ``w_i = A_i`` and ``sum_i A_i = 0`` by construction => exactly 0.
- PPO clipped: ``w_i = r_i * A_i`` unclipped, 0 when clipped. On a shared prefix the ratio
  ``r_i = pi_theta(y_t|ctx)/pi_old(y_t|ctx)`` is also identical across members (same token,
  same context), call it r. Clipping therefore triggers identically for all members with
  the same sign of A_i. So the sum is ``r * sum_i A_i`` over the unclipped members --
  which is NOT necessarily zero, because clipping removes an asymmetric subset.

That last point is the interesting one and is why this needs checking rather than asserting.
"""
from __future__ import annotations

import random

EPS = 0.2


def centred(rewards: list[float]) -> list[float]:
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def prefix_weight_vanilla(advs: list[float]) -> float:
    """Net multiplier on the shared-prefix gradient, vanilla policy gradient."""
    return sum(advs)


def prefix_weight_ppo(advs: list[float], ratio: float, eps: float = EPS) -> float:
    """Net multiplier on the shared-prefix gradient under PPO clipping.

    PPO takes min(r*A, clip(r,1-eps,1+eps)*A). The gradient flows through whichever branch
    attains the min. For A>0 the objective is capped above at (1+eps)A, so the gradient is
    zero once r > 1+eps. For A<0 it is capped below, so the gradient is zero once r < 1-eps.
    """
    total = 0.0
    for a in advs:
        if a > 0 and ratio > 1 + eps:
            continue  # clipped: no gradient
        if a < 0 and ratio < 1 - eps:
            continue  # clipped: no gradient
        total += ratio * a
    return total


def main() -> None:
    rng = random.Random(0)
    print(f"{'case':<38} {'vanilla':>10} {'ppo r=1.0':>10} {'ppo r=1.5':>10} {'ppo r=0.5':>10}")
    print("-" * 82)

    cases: list[tuple[str, list[float]]] = [
        ("balanced binary G=4 (p=0.5)", [1, 1, 0, 0]),
        ("one success G=4 (p=0.25)", [1, 0, 0, 0]),
        ("three successes G=4 (p=0.75)", [1, 1, 1, 0]),
        ("unanimous success G=4 (p=1)", [1, 1, 1, 1]),
        ("unanimous failure G=4 (p=0)", [0, 0, 0, 0]),
        ("continuous rewards G=8", [rng.random() for _ in range(8)]),
    ]

    worst_vanilla = 0.0
    ppo_nonzero = []
    for name, rewards in cases:
        advs = centred([float(r) for r in rewards])
        v = prefix_weight_vanilla(advs)
        p10 = prefix_weight_ppo(advs, 1.0)
        p15 = prefix_weight_ppo(advs, 1.5)
        p05 = prefix_weight_ppo(advs, 0.5)
        worst_vanilla = max(worst_vanilla, abs(v))
        for r, val in ((1.5, p15), (0.5, p05)):
            if abs(val) > 1e-12:
                ppo_nonzero.append((name, r, val))
        print(f"{name:<38} {v:10.2e} {p10:10.2e} {p15:10.2e} {p05:10.2e}")

    print()
    print(f"vanilla PG, largest |net prefix weight| over all cases: {worst_vanilla:.2e}")
    print("=> EXACTLY zero: sum_i A_i = 0 by construction, and the shared prefix shares")
    print("   both context and token, so grad log pi is a single common vector.")
    print()
    if ppo_nonzero:
        print("PPO with clipping, cases where the prefix weight is NOT zero:")
        for name, r, val in ppo_nonzero:
            print(f"   ratio={r}: {name:<36} net = {val:+.4f}")
        print()
        print("=> NOT exactly zero once the ratio leaves the trust region. Clipping removes")
        print("   an asymmetric subset of members (only those whose advantage has the sign")
        print("   being capped), so the surviving advantages no longer sum to zero.")
        print("   At r=1 (start of an epoch, on-policy) it IS zero.")
    else:
        print("PPO: zero in every case tested.")


if __name__ == "__main__":
    main()
