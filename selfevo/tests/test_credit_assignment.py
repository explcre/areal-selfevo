"""Why the learned router does not learn: the credit signal, not the bandit.

The `ctx` run gave the bandit 128 clean feedback updates -- all three modes present in every
batch, `weak_attribution=0.0`, one confounded skip in the whole run -- and its mode mix stayed
indistinguishable from uniform thirds, trending flatter rather than sharper.

The argument is that a single per-batch scalar credited to every decision carries no
information distinguishing the arms: with `b_m += r x` and `A_m += x x^T`, a shared `r` and an
assignment independent of `x` give every arm the same `theta = E[x x^T]^-1 r xbar`, with the
per-arm count cancelling. These tests turn that argument into a measurement, and separate
"the bandit is broken" from "the signal is uninformative" by feeding the SAME bandit a signal
that does distinguish the arms.
"""

from __future__ import annotations

import numpy as np
import pytest

from selfevo.compose import ROUTERS
from selfevo.routing.base import RoutingContext
from selfevo.routing.feedback import DecisionOutcome

FEATURES = ["solve_rate", "reward_std", "mean_response_len", "len_dispersion",
            "mean_logprob", "logprob_dispersion", "truncated_fraction"]
UNITS = 64
BATCHES = 60


def _ctx(rng: np.random.Generator, step: int, i: int) -> RoutingContext:
    """A unit whose features vary, so the arms see genuinely different contexts."""
    solve = float(rng.uniform(0.0, 1.0))
    extra = {
        "solve_rate": solve,
        "reward_std": float(rng.uniform(0, 0.5)),
        "mean_response_len": float(rng.uniform(50, 400)),
        "len_dispersion": float(rng.uniform(0, 1)),
        "mean_logprob": float(rng.uniform(-2, 0)),
        "logprob_dispersion": float(rng.uniform(0, 1)),
        "truncated_fraction": float(rng.uniform(0, 0.3)),
    }
    return RoutingContext(solve_rate=solve, group_size=8, has_teacher=False,
                          unit_id=f"{step}:{i}", extra=extra)


def _thetas(router) -> dict[str, np.ndarray]:
    """Ridge solution per arm, the thing that has to differ for a preference to exist."""
    return {m: np.linalg.inv(router._A[m]) @ router._b[m] for m in router._A}


def _max_pairwise_gap(thetas: dict[str, np.ndarray]) -> float:
    """Largest L2 distance between any two arms' parameter vectors."""
    ms = sorted(thetas)
    return max(float(np.linalg.norm(thetas[a] - thetas[b]))
               for i, a in enumerate(ms) for b in ms[i + 1:])


def _run(reward_fn, seed: int = 0):
    """Drive a registry-built router for BATCHES batches, crediting via reward_fn.

    Args:
        reward_fn: ``(mode, ctx, rng) -> float``, the value credited to that unit.
        seed: RNG seed.

    Returns:
        ``(router, mode_counts)``.
    """
    rng = np.random.default_rng(seed)
    router = ROUTERS["contextual"](cold_start_rounds=UNITS)   # one batch of round-robin
    counts: dict[str, int] = {}
    for step in range(BATCHES):
        chosen = []
        for i in range(UNITS):
            c = _ctx(rng, step, i)
            m = router.route(c).argmax()
            counts[m] = counts.get(m, 0) + 1
            chosen.append((c, m))
        router.observe({
            c.unit_id: DecisionOutcome(mode=m, value=reward_fn(m, c, rng), batch_id=str(step))
            for c, m in chosen
        })
    return router, counts


def test_a_shared_per_batch_scalar_leaves_the_arms_indistinguishable():
    """The measured mechanism: identical credit for every unit in the batch."""
    def shared(mode, ctx, rng, _state={}):
        # One draw per batch, reused for every unit -- exactly what batch_outcomes does.
        key = ctx.unit_id.split(":")[0]
        if key not in _state:
            _state.clear()
            _state[key] = float(rng.normal(0.0, 0.1))
        return _state[key]

    router, counts = _run(shared)
    gap = _max_pairwise_gap(_thetas(router))
    total = sum(counts.values())
    shares = sorted(v / total for v in counts.values())
    assert gap < 0.5, f"arms separated more than expected under shared credit: {gap:.3f}"
    # And the behavioural consequence: no arm dominates.
    assert shares[-1] < 0.55, f"an arm dominated under an uninformative signal: {shares}"


def test_the_same_bandit_separates_when_credit_depends_on_the_mode():
    """Isolates the cause: the bandit works, the signal was uninformative.

    Same router, same contexts, same number of updates -- only the credit changes, to a
    value that actually depends on which mode was applied.
    """
    def mode_dependent(mode, ctx, rng):
        base = {"sft": 1.0, "rl": 0.0, "skip": -1.0}[mode]
        return base + float(rng.normal(0.0, 0.05))

    router, counts = _run(mode_dependent)
    gap = _max_pairwise_gap(_thetas(router))
    total = sum(counts.values())
    best = max(counts, key=counts.get)
    assert gap > 1.0, f"arms failed to separate under an informative signal: {gap:.3f}"
    assert best == "sft", f"expected the rewarded mode to dominate, got {best}: {counts}"
    assert counts[best] / total > 0.5, f"rewarded mode did not dominate: {counts}"


def test_the_contrast_between_the_two_signals_is_large():
    """The comparison is the finding: same bandit, two signals, opposite outcomes."""
    def shared(mode, ctx, rng, _state={}):
        key = ctx.unit_id.split(":")[0]
        if key not in _state:
            _state.clear()
            _state[key] = float(rng.normal(0.0, 0.1))
        return _state[key]

    def mode_dependent(mode, ctx, rng):
        return {"sft": 1.0, "rl": 0.0, "skip": -1.0}[mode] + float(rng.normal(0.0, 0.05))

    flat, _ = _run(shared, seed=1)
    sharp, _ = _run(mode_dependent, seed=1)
    assert _max_pairwise_gap(_thetas(sharp)) > 4 * _max_pairwise_gap(_thetas(flat))


def test_the_real_deadlock_needs_cold_start_because_uniform_batches_are_refused():
    """Reproduces the closed loop measured in the first ctx run.

    The tests above always credit, so LinUCB's own confidence bonus supplies exploration and
    the cold start is optional -- which is why a mutation disabling it survives them. The
    LIVE loop is different: `batch_outcomes` REFUSES a batch whose decisions were all
    identical, so a router that opens uniformly never has `A` updated, its bonus never
    changes, the tie never breaks, and it emits one mode forever.

    Modelled here by skipping `observe` exactly when the batch took a single mode.
    """
    rng = np.random.default_rng(3)

    def run_with_refusal(cold_start_rounds: int) -> set[str]:
        router = ROUTERS["contextual"](cold_start_rounds=cold_start_rounds)
        emitted: set[str] = set()
        for step in range(20):
            chosen = []
            for i in range(UNITS):
                c = _ctx(rng, step, i)
                chosen.append((c, router.route(c).argmax()))
            modes = {m for _, m in chosen}
            emitted |= modes
            if len(modes) < 2:           # batch_outcomes raises ConfoundedUpdate here
                continue
            router.observe({
                c.unit_id: DecisionOutcome(
                    mode=m,
                    value={"sft": 1.0, "rl": 0.0, "skip": -1.0}[m] + float(rng.normal(0, 0.05)),
                    batch_id=str(step),
                )
                for c, m in chosen
            })
        return emitted

    assert run_with_refusal(0) == {"rl"}, "expected the zero-cold-start deadlock"
    assert len(run_with_refusal(UNITS)) > 1, "cold start must break the deadlock"
