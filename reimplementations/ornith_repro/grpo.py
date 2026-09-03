"""GRPO group advantages, with degenerate groups surfaced rather than hidden.

All three Ornith stages are optimised with GRPO. The one property that matters for
correctness here is that a group whose members all received the same reward carries
*exactly zero* reward-directed gradient: the numerator `R_i - mean` is zero for every
member. That is not a bug to be smoothed over with an epsilon; it is a real loss of
signal that must be counted, because a batch made mostly of such groups produces a false
negative when one reports "the gradient is small".

This project has already been bitten by exactly that: 90 of 128 groups were unanimous on
MATH at 32B, so gradient statistics computed over that batch were meaningless.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .types import GroupAdvantages


def grpo_advantages(
    rewards: Sequence[float],
    epsilon: float = 1e-6,
) -> GroupAdvantages:
    """Compute group-normalised advantages `A_i = (R_i - mean) / (std + eps)`.

    Args:
        rewards: The rewards of one GRPO group.
        epsilon: Denominator floor (ambiguity A12; the source does not state one).

    Returns:
        A `GroupAdvantages` carrying the advantages, an explicit `degenerate` flag, and
        the pre-epsilon population standard deviation.

    Raises:
        ValueError: if the group has fewer than two members (guard G3). A one-member or
            empty group has no within-group contrast and must be refused, not scored as
            a vector of zeros that looks like a legitimate update.

    Note:
        Population (not sample) standard deviation, stated because the choice is
        undisclosed and changes the advantage scale by sqrt(G/(G-1)).

        When the group is degenerate every advantage is exactly 0.0. We return that
        vector *and* the flag; callers must not treat the zeros as evidence about the
        policy.
    """
    n = len(rewards)
    if n < 2:
        raise ValueError(
            f"grpo_advantages() needs at least 2 group members, got {n}. An empty or "
            "singleton group must be refused, not scored as zeros (guard G3)."
        )
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    degenerate = std == 0.0
    advantages = [(r - mean) / (std + epsilon) for r in rewards]
    return GroupAdvantages(advantages=advantages, degenerate=degenerate, reward_std=std)


def degenerate_fraction(groups: Sequence[GroupAdvantages]) -> float:
    """Return the fraction of groups in a batch that carry zero reward-directed gradient.

    Args:
        groups: Per-group advantage results.

    Returns:
        Fraction in [0,1].

    Raises:
        ValueError: if `groups` is empty.
    """
    if not groups:
        raise ValueError("degenerate_fraction() received no groups (guard G3).")
    return sum(1 for g in groups if g.degenerate) / len(groups)


def predicted_binary_degeneracy(theta: float, group_size: int) -> float:
    """Predicted degenerate-group rate for a binary reward: `theta^G + (1-theta)^G`.

    A group of `G` i.i.d. Bernoulli(theta) rollouts is constant exactly when it is
    all-success or all-failure. This closed form is what lets the loop's own difficulty
    target be checked against the signal it destroys.

    Args:
        theta: True success probability.
        group_size: GRPO group size `G`.

    Returns:
        The probability that a group is degenerate.

    Raises:
        ValueError: if theta is outside [0,1] or group_size < 2.

    Note:
        Minimised at theta = 0.5. At G=8 this is 0.0078 at theta=0.5 but 0.1678 at
        theta=0.2, so Ornith's published p* = 0.2 targets a region where roughly 21x
        more groups are wasted than at p* = 0.5. That is a prediction about the
        published constant, and `experiments/gate_selection.py` measures it.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0,1], got {theta}")
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    return theta**group_size + (1.0 - theta) ** group_size
