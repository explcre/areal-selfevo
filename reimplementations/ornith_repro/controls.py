"""Size-matched random controls at the treatment's own MEASURED proportions.

Every learned component this project has tested has tied with a control of this shape --
the learned router and the MEDS clustering both did, and both were retired for it. So the
control is built before the treatment is believed, not after.

The rule that makes this control honest, and the one that is easy to get wrong: the
control matches the proportions the treatment ACTUALLY produced, measured from the
treatment's own artifacts. Not uniform, and not the proportions the design intended. A
treatment that drifts to 80% of one task family and is compared against a uniform control
will look effective purely because of the drift.

Two axes, matching the two generated stages:

* `random_task`     -- replaces the learned proposer with a draw from a fixed task pool.
* `random_scaffold` -- replaces the learned scaffold generator with a draw from a fixed
                       scaffold pool.

Each is matched independently so that a tie on one axis and an effect on the other is
visible, rather than the two being confounded in a single "random everything" arm.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from .guards import assert_proportions_match
from .types import Scaffold, Task


def measure_proportions(items: Sequence[object], key: str) -> dict[str, float]:
    """Measure the empirical proportions of one covariate over a set of items.

    Args:
        items: Objects exposing a `covariates()` method (Task or Scaffold).
        key: Which covariate to tabulate.

    Returns:
        Mapping from covariate value to proportion, summing to 1.

    Raises:
        ValueError: if `items` is empty, or an item lacks the requested covariate.
    """
    if not items:
        raise ValueError("cannot measure proportions of an empty set")
    values = []
    for it in items:
        cov = it.covariates()  # type: ignore[attr-defined]
        if key not in cov:
            raise ValueError(f"item {it!r} has no covariate {key!r}")
        values.append(cov[key])
    counts = Counter(values)
    total = float(len(values))
    return {k: c / total for k, c in counts.items()}


def _sample_matched(
    pool: Sequence[object],
    key: str,
    target: dict[str, float],
    n: int,
    rng: random.Random,
) -> list[object]:
    """Draw `n` items from `pool` so their `key` proportions match `target`.

    Uses largest-remainder allocation so the realised counts are as close to the target
    proportions as integer counts permit.

    Raises:
        ValueError: if the pool cannot supply a covariate value the target requires.
    """
    by_value: dict[str, list[object]] = {}
    for it in pool:
        by_value.setdefault(it.covariates()[key], []).append(it)  # type: ignore[attr-defined]

    missing = set(target) - set(by_value)
    if missing:
        raise ValueError(
            f"control pool cannot match the treatment: it has no items with {key} in "
            f"{sorted(missing)}. A control that silently drops a stratum is not matched."
        )

    exact = {v: target[v] * n for v in target}
    counts = {v: int(x) for v, x in exact.items()}
    remainder = n - sum(counts.values())
    for v, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if remainder <= 0:
            break
        counts[v] += 1
        remainder -= 1

    out: list[object] = []
    for v, c in counts.items():
        out.extend(rng.choices(by_value[v], k=c))
    rng.shuffle(out)
    return out


def random_task_control(
    treatment_tasks: Sequence[Task],
    pool: Sequence[Task],
    key: str = "family",
    seed: int = 0,
    tol: float = 0.02,
) -> list[Task]:
    """Build the size-matched random-task control arm.

    Args:
        treatment_tasks: The tasks the treatment actually produced and kept.
        pool: Fixed pool of tasks to draw the control from.
        key: Covariate to match on.
        seed: RNG seed.
        tol: Per-stratum tolerance for the post-hoc match check.

    Returns:
        A list of control tasks, the same length as `treatment_tasks`, whose `key`
        proportions match the treatment's measured proportions. Every returned task has
        `source="pool"` so no analysis can silently mix the arms.

    Raises:
        GuardViolation: if the realised proportions do not match within `tol`.
        ValueError: if the pool cannot cover a required stratum.
    """
    target = measure_proportions(treatment_tasks, key)
    rng = random.Random(seed)
    drawn = _sample_matched(pool, key, target, len(treatment_tasks), rng)
    control = [
        Task(task_id=f"ctrl-{i}-{t.task_id}", text=t.text, family=t.family,
             length_bin=t.length_bin, source="pool")
        for i, t in enumerate(drawn)  # type: ignore[arg-type]
    ]
    assert_proportions_match(target, measure_proportions(control, key), tol=tol)
    return control


def random_scaffold_control(
    treatment_scaffolds: Sequence[Scaffold],
    pool: Sequence[Scaffold],
    key: str = "grader_kind",
    seed: int = 0,
    tol: float = 0.02,
) -> list[Scaffold]:
    """Build the size-matched random-scaffold control arm.

    Args and behaviour mirror `random_task_control`; see that docstring.

    Raises:
        GuardViolation: if the realised proportions do not match within `tol`.
        ValueError: if the pool cannot cover a required stratum.
    """
    target = measure_proportions(treatment_scaffolds, key)
    rng = random.Random(seed)
    drawn = _sample_matched(pool, key, target, len(treatment_scaffolds), rng)
    control = [
        Scaffold(scaffold_id=f"ctrl-{i}-{s.scaffold_id}", instructions=s.instructions,
                 tools=s.tools, grader_kind=s.grader_kind, source="pool")
        for i, s in enumerate(drawn)  # type: ignore[arg-type]
    ]
    assert_proportions_match(target, measure_proportions(control, key), tol=tol)
    return control
