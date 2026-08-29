"""When is a group-based RL gradient informative, and when is it provably zero?

Pure functions, no torch, no config. Everything here is a property of the group size and
the solve rate, so it is cheap enough to evaluate per prompt and testable without a GPU.

The one fact the whole module rests on: for a group of ``G`` samples with binary reward,
GRPO's centred advantage is ``A_i = r_i - rbar``. If every ``r_i`` is equal then every
``A_i`` is exactly zero and the group contributes **no gradient at all**. Silence is not a
small-gradient regime; it is an identically-zero one.
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = [
    "SilenceSide",
    "rl_informativeness",
    "silent_group_probability",
    "silence_side",
    "min_group_size",
    "expected_nonsilent_groups",
]


class SilenceSide(Enum):
    """Which side of the RL silence a unit sits on.

    The two silent regimes need opposite responses, which is why informativeness alone is
    not a sufficient routing key:

    ``UNSOLVED``
        ``p`` near 0. The model cannot solve it, so RL has nothing to push on. Useful only
        if an external target (teacher / gold) exists -- otherwise the unit is unlearnable
        right now and should be deprioritised rather than force-fed.
    ``SOLVED``
        ``p`` near 1. The model already solves it. RL is silent because there is nothing
        left to learn. The right action is to spend less compute here; adding SFT would
        only sharpen an already-correct policy and burn entropy for nothing.
    ``INFORMATIVE``
        ``p`` away from both ends: the group disagrees, so the advantage is non-zero and RL
        carries signal.
    """

    UNSOLVED = "unsolved"
    SOLVED = "solved"
    INFORMATIVE = "informative"


def silent_group_probability(p: float, group_size: int) -> float:
    """Probability that a group of ``group_size`` Bernoulli(``p``) rewards is unanimous.

    A unanimous group has ``A_i = 0`` for every member and contributes zero gradient.

    Args:
        p: Solve rate in [0, 1].
        group_size: Number of samples per prompt (``gconfig.n_samples``). Must be >= 1.

    Returns:
        ``p**G + (1-p)**G``, in [0, 1].

    Raises:
        ValueError: If ``p`` is outside [0, 1] or ``group_size`` < 1.
    """
    _validate(p, group_size)
    return p**group_size + (1.0 - p) ** group_size


def rl_informativeness(p: float, group_size: int) -> float:
    """Fraction of groups that carry a non-zero RL gradient.

    ``I_RL(p, G) = 1 - p**G - (1-p)**G``. Zero at ``p in {0, 1}``, maximal at ``p = 1/2``.
    With ``G = 4``: ``I_RL(0.5) = 0.875``, ``I_RL(0.9) = 0.344``, ``I_RL(0.99) = 0.039``.

    This is the quantity to route on, but never on its own -- see :class:`SilenceSide`,
    because it is symmetric about ``p = 1/2`` while the correct response is not.

    Args:
        p: Solve rate in [0, 1].
        group_size: Samples per prompt, >= 1.

    Returns:
        Informativeness in [0, 1].
    """
    return 1.0 - silent_group_probability(p, group_size)


def silence_side(
    p: float, group_size: int, threshold: float = 0.1
) -> SilenceSide:
    """Classify a unit by whether RL can learn from it, and if not, why not.

    Args:
        p: Solve rate in [0, 1].
        group_size: Samples per prompt, >= 1.
        threshold: Informativeness below which the unit counts as silent. The default of
            0.1 means "fewer than one group in ten produces any gradient".

    Returns:
        ``INFORMATIVE`` when ``rl_informativeness >= threshold``; otherwise ``UNSOLVED``
        or ``SOLVED`` according to which side of ``0.5`` ``p`` falls on.

    Raises:
        ValueError: If ``threshold`` is outside [0, 1], or ``p``/``group_size`` invalid.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if rl_informativeness(p, group_size) >= threshold:
        return SilenceSide.INFORMATIVE
    # Exactly 0.5 cannot be silent for G >= 2, so this branch only sees the tails; ties
    # resolve to SOLVED, which is the conservative choice (spend less, rather than pull in
    # a teacher for something the model may already handle).
    return SilenceSide.UNSOLVED if p < 0.5 else SilenceSide.SOLVED


def min_group_size(p: float, eps: float) -> float:
    """Group size needed to resolve the advantage's sign at tolerance ``eps``.

    ``G >= 1 / (8 * eps * p * (1 - p))``. Returns a float on purpose: it is a bound, and
    rounding it here would hide how far a configuration sits from the requirement.

    At the p=0.76 measured in step0c this gives G >= 6.9 for eps=0.10 and G >= 13.7 for
    eps=0.05, against a configured ``n_samples`` of 4.

    Args:
        p: Solve rate, strictly inside (0, 1).
        eps: Tolerance, strictly positive.

    Returns:
        The minimum group size as a float; ``inf`` when ``p`` is 0 or 1, where the sign is
        never resolvable because there is no variance to resolve.

    Raises:
        ValueError: If ``p`` is outside [0, 1] or ``eps`` <= 0.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    denom = 8.0 * eps * p * (1.0 - p)
    if denom == 0.0:
        return math.inf
    return 1.0 / denom


def expected_nonsilent_groups(p: float, group_size: int, n_groups: int) -> float:
    """Expected number of groups out of ``n_groups`` that produce any gradient.

    Useful for sizing a batch: a batch of 256 prompts at ``p = 0.95`` and ``G = 4`` yields
    only ``256 * I_RL(0.95, 4) = 47`` groups with a non-zero gradient, so the effective
    batch is far smaller than the nominal one.

    Args:
        p: Solve rate in [0, 1].
        group_size: Samples per prompt, >= 1.
        n_groups: Number of prompts in the batch, >= 0.

    Returns:
        ``n_groups * rl_informativeness(p, group_size)``.

    Raises:
        ValueError: If ``n_groups`` is negative, or ``p``/``group_size`` invalid.
    """
    if n_groups < 0:
        raise ValueError(f"n_groups must be non-negative, got {n_groups}")
    return n_groups * rl_informativeness(p, group_size)


def _validate(p: float, group_size: int) -> None:
    """Shared argument validation for the Bernoulli-group functions."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
