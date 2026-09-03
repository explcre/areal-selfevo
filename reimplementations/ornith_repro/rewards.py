"""The three Ornith-1.5 reward functions, exactly as published.

Published on the Ornith-1.5 method page and independently transcribed by a source-level
audit of that page on 2026-08-19:

    R_task        = V(q,s) * D(q,s,{tau_i}) * N(q)
    D             = exp(-(p - p*)^2 / (2 sigma^2)),  p* = 0.2
    N(q)          = 1 - max_{q_j in B} sim(q, q_j)
    R_harness     = C(q,h) * F(h,{tau_i}) * H(h)
    R_rollout     = h(q, tau_i)
    p             = (1/N) sum_i 1[s(q, tau_i) = success]

Only `p* = 0.2` is disclosed. `sigma`, the rollout count, `sim`, and the operational
definitions of `V, C, F, H` are not; every one of those is our choice and is recorded in
AMBIGUITIES.md with an id. Functions below name the ambiguity id they depend on.

Nothing in this module is allowed to invent a factor, a floor, an exponent or a
re-weighting that the source does not state. In particular the elasticity of R_task with
respect to each of V, D, N is exactly 1, and any factor at zero annihilates the product.
That is a strong design choice of Ornith's, not of ours, and it is left intact so that it
can be ablated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .types import Rollout, RolloutOutcome

# Only this one is published.
P_STAR_PUBLISHED = 0.2


def success_rate(
    rollouts: Sequence[Rollout],
    abort_policy: str = "exclude",
) -> tuple[float, int, int]:
    """Compute p_hat, the empirical success rate over rollouts.

    Implements `p = (1/N) sum_i 1[s(q, tau_i) = success]`. The source does not say what
    happens to a rollout that never produced a gradeable answer, because the source does
    not acknowledge that case. We do (guard G1).

    Args:
        rollouts: The rollouts for one task.
        abort_policy: How to treat ABORTED rollouts.
            "exclude" (default): drop them from numerator and denominator, and report the
                count separately. This keeps p_hat an estimate of the policy's success
                rate on gradeable attempts.
            "failure": count them as failures. This is the known-bad setting that inflates
                apparent difficulty; it exists only so the bias can be measured, and
                callers that select it are logged.
            "strict": raise if any rollout aborted.

    Returns:
        A tuple `(p_hat, n_valid, n_aborted)`. `n_valid` is the denominator actually used.

    Raises:
        ValueError: if `rollouts` is empty (guard G3 -- an empty batch must never be
            silently scored as 0.0), if `abort_policy` is unknown, or if
            `abort_policy="strict"` and an abort is present.

    Note:
        Dropping aborted rollouts does not optimise `E[r]`; it optimises `E[A * r]` where
        `A` is the indicator that the rollout was gradeable. Because the policy partly
        controls `A` (it can run away and be cut off), this is the coverage-hacking
        incentive. We keep "exclude" as the default because scoring an abort as a wrong
        answer is the worse of the two biases, but the abort rate is always recorded so
        the correction can be applied later.
    """
    if not rollouts:
        raise ValueError(
            "success_rate() received an empty rollout list. An empty batch must be "
            "refused, never scored as p_hat=0.0 (guard G3)."
        )
    n_aborted = sum(1 for r in rollouts if r.outcome is RolloutOutcome.ABORTED)

    if abort_policy == "strict":
        if n_aborted:
            raise ValueError(
                f"{n_aborted} of {len(rollouts)} rollouts aborted and abort_policy="
                "'strict'."
            )
        graded = list(rollouts)
    elif abort_policy == "exclude":
        graded = [r for r in rollouts if r.outcome is not RolloutOutcome.ABORTED]
    elif abort_policy == "failure":
        graded = list(rollouts)
    else:
        raise ValueError(f"unknown abort_policy {abort_policy!r}")

    if not graded:
        raise ValueError(
            "every rollout aborted; p_hat is undefined. The task must be refused, not "
            "scored 0.0 (guard G1)."
        )

    n_success = sum(1 for r in graded if r.outcome is RolloutOutcome.SUCCESS)
    return n_success / len(graded), len(graded), n_aborted


def difficulty_reward(
    p_hat: float,
    p_star: float = P_STAR_PUBLISHED,
    sigma: float = 0.15,
) -> float:
    """D = exp(-(p_hat - p*)^2 / (2 sigma^2)).

    Args:
        p_hat: Empirical success rate from `success_rate`.
        p_star: Target success rate. Published as 0.2.
        sigma: Kernel width. NOT published (ambiguity A2); ours.

    Returns:
        The difficulty reward in (0, 1].

    Raises:
        ValueError: if sigma is not strictly positive, or p_hat is outside [0, 1].

    Note:
        This is the plug-in estimator: it evaluates the kernel at the noisy `p_hat`, not
        at the true success probability. With k rollouts the standard error of `p_hat` at
        p=0.2 is sqrt(0.16/k), which is 0.141 at k=8 -- comparable to or wider than any
        plausible sigma. The measured consequence is in experiments/gate_selection.py:
        selecting tasks on this quantity produces a real winner's curse.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 <= p_hat <= 1.0):
        raise ValueError(f"p_hat must be in [0,1], got {p_hat}")
    return math.exp(-((p_hat - p_star) ** 2) / (2.0 * sigma**2))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity in [0, 1].

    The default `sim` for novelty (ambiguity A6). The source names no similarity function
    and no encoder, so any cosine-on-a-named-embedder here would be a fabricated detail.
    Jaccard is bounded in [0,1], which keeps N in [0,1]; cosine would permit N in [0,2]
    and an unnormalised dot product would permit N < 0.
    """
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


def novelty_reward(
    task_text: str,
    buffer_texts: Sequence[str],
    sim: Callable[[str, str], float] = jaccard_similarity,
) -> tuple[float, bool]:
    """N(q) = 1 - max_{q_j in B} sim(q, q_j).

    Args:
        task_text: The candidate task `q`.
        buffer_texts: The task buffer `B`.
        sim: Similarity function (ambiguity A6).

    Returns:
        `(N, empty_buffer)`. `empty_buffer` is True when `B` was empty.

    Note:
        The maximum over an empty buffer is undefined in the source (ambiguity A7). We
        return N = 1.0, i.e. maximally novel, and flag it. Returning 0.0 would zero
        R_task for the very first task of a run and silently kill the first group: a
        silent-zero path. The flag is stored on the record so an analysis can always
        exclude the first task rather than having to trust this choice.
    """
    if not buffer_texts:
        return 1.0, True
    return 1.0 - max(sim(task_text, b) for b in buffer_texts), False


def task_reward(V: float, D: float, N: float) -> float:
    """R_task = V * D * N.

    Args:
        V: Validity in [0,1]. May hard-zero the whole reward, by design.
        D: Difficulty reward from `difficulty_reward`.
        N: Novelty from `novelty_reward`.

    Returns:
        The product.

    Raises:
        ValueError: if any factor is outside [0,1].

    Note:
        The multiplicative form is Ornith's, and it is load-bearing: any factor at zero
        annihilates the reward regardless of the other two. It is left exactly as
        published so that `--ablate-product` can replace it with a weighted sum and the
        difference can be measured.
    """
    for name, v in (("V", V), ("D", D), ("N", N)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
    return V * D * N


def harness_reward(C: float, F: float, H: float) -> float:
    """R_harness = C * F * H.

    Args:
        C: Alignment of the scaffold to the task, in [0,1].
        F: Reward fidelity of the scaffold, in [0,1].
        H: Hack resistance of the scaffold, in [0,1].

    Returns:
        The product.

    Raises:
        ValueError: if any factor is outside [0,1].

    Note:
        C, F and H are named on the method page but never defined and never given an
        evaluator (ambiguity A9). Our rubrics are in judges.py. This is the largest
        reconstruction gap in the whole loop: any number this function produces is
        conditional on our rubrics, not on Ornith's.
    """
    for name, v in (("C", C), ("F", F), ("H", H)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
    return C * F * H
