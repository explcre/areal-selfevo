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
    "attainable_solve_rates",
    "threshold_is_inert",
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
    # p == 0.5 CAN reach this branch: it is silent whenever threshold > 1 - 2**(1-G),
    # and threshold is validated up to 1.0. Verified: silence_side(0.5, 2, 0.51) and
    # silence_side(0.5, 4, 0.885) both return SOLVED. At the default threshold of 0.1 this
    # is unreachable, but the guarantee is conditional on the threshold, not unconditional.
    # Ties resolve to SOLVED, the conservative choice: spend less, rather than pull in a
    # teacher for something the model may already handle.
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
    if eps >= 0.5:
        # Without this, the bound is satisfiable by group_size 1 -- which this very
        # module proves is identically silent (A = r - rbar = 0 for a singleton). A
        # bound whose feasible region contains a provably degenerate configuration is
        # worse than no bound, because it reads as permission.
        raise ValueError(
            f"eps must be < 0.5, got {eps}: larger tolerances admit group_size < 2, "
            "which has zero informativeness by construction"
        )
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


def attainable_solve_rates(group_size: int) -> list[float]:
    """Solve rates a single group can actually produce: ``k / G`` for k in 0..G.

    Matters because :func:`rl_informativeness` is continuous in ``p`` while an observed
    ``p_hat`` from one group is not -- it takes only ``G + 1`` values.

    Args:
        group_size: Samples per prompt, >= 1.

    Returns:
        The attainable rates in increasing order.

    Raises:
        ValueError: If ``group_size`` < 1.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    return [k / group_size for k in range(group_size + 1)]


def threshold_is_inert(group_size: int, threshold: float) -> bool:
    """Whether ``threshold`` can change any decision at single-group granularity.

    **This is the sharpest limitation of the whole criterion, so it is exposed as code
    rather than left in a docstring.** With ``p_hat = k / G``, ``I_RL(p_hat, G)`` is zero
    exactly when the group was unanimous and is otherwise bounded below by
    ``I_RL(1/G, G)``. So every threshold in ``(0, I_RL(1/G, G)]`` induces the *identical*
    partition, and at that granularity the criterion is not estimating anything -- it is
    a re-encoding of "was this group unanimous?", an outcome already observed.

    At ``G = 4`` the attainable informativeness values are
    ``[0.0, 0.6797, 0.875, 0.6797, 0.0]``, so any threshold in ``(0, 0.6797]`` -- including
    the default 0.1 -- is inert.

    ``I_RL`` earns its keep only at task/cluster granularity, where ``p`` is pooled over
    many prompts and genuinely takes intermediate values. Note also that ``I_RL(p_hat, G)``
    is a *biased* plug-in estimate of ``I_RL(p, G)``, by Jensen, since ``I_RL`` is concave
    in ``p``.

    Args:
        group_size: Samples per prompt, >= 1.
        threshold: Informativeness threshold in [0, 1].

    Returns:
        True when no attainable solve rate straddles ``threshold``, i.e. tuning it cannot
        change any routing decision.

    Raises:
        ValueError: If ``threshold`` is outside [0, 1] or ``group_size`` < 1.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    vals = [rl_informativeness(p, group_size) for p in attainable_solve_rates(group_size)]
    nonzero = [v for v in vals if v > 0.0]
    if not nonzero:
        return True
    # Inert when the threshold separates nothing: every non-zero value lands on the same
    # side of it as every other non-zero value.
    return all(v >= threshold for v in nonzero) or all(v < threshold for v in nonzero)
