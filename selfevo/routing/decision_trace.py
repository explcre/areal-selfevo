"""The per-decision record, and the correspondence control that falsifies it.

WHY A TRACE EXISTS AT ALL. A run's stats stream carries aggregates -- ``route/rl_groups``,
``route/sft_groups`` -- and the number that decides whether a learned router beats a
rate-matched random one is not an aggregate. **Subset contrast** is half the total variation
distance between the mode distribution on one subset of units and on another, so it needs each
unit's mode BESIDE the features that define the subsets. A run logged only as a mix cannot be
re-read for it afterwards: that is exactly why the batch-credited arm's null had to be
re-derived in :mod:`selfevo.routing.credit_sim` instead of measured from its own 129 steps.
The mix alone is also actively misleading here -- an arm whose L1 from uniform rises can be an
arm that learned nothing, since the router picks its own assignment and the arms drift apart
on that feedback alone (``selfevo/FINDINGS_credit_assignment.md`` section 2).

WHY THE SHUFFLE IS IN THE SAME MODULE. The trace is what a claim is computed from and the
shuffle is what makes the claim falsifiable; keeping them together means the control cannot be
forgotten by whoever adds the next reader. Measured in simulation over 8 paired seeds,
shuffling the credits across the prompts that earned them collapses the per-prompt arm from
0.779 to 0.102 -- the batch arm's level -- so an arm that does not beat its own shuffle has
produced a noisier signal, not targeting.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Iterable, Mapping, Sequence, TypeVar

__all__ = ["trace_path", "trace_records", "shuffle_correspondence"]

_T = TypeVar("_T")


def trace_path(base: str) -> pathlib.Path:
    """The file this process appends to, derived from the configured base path.

    One file per process, and deliberately not one shared file: at ``fsdp:d2`` the actor path
    runs in two workers, each holding its own router and each seeing half of every batch.
    Interleaving them would hide that the arm ran two independent routers, which is a property
    of the arm and has to be reportable rather than averaged away.

    Args:
        base: The configured ``group_routing.decision_trace_path``.

    Returns:
        ``<base>.pid<pid>.jsonl``.
    """
    return pathlib.Path(f"{base}.pid{os.getpid()}.jsonl")


def trace_records(gr: object, records: Iterable[Mapping[str, object]]) -> int:
    """Append records to the run's decision trace, if one is configured.

    Args:
        gr: The ``GroupRoutingConfig``. ``decision_trace_path`` of ``None`` -- the default and
            every run before the field existed -- writes nothing and returns 0, so this call
            is inert on an unconfigured arm.
        records: Mappings to serialise, one JSON object per line. Consumed once, so a
            generator may be passed and is not materialised when tracing is off.

    Returns:
        How many records were written.
    """
    base = getattr(gr, "decision_trace_path", None)
    if not base:
        return 0
    path = trace_path(str(base))
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # Append mode, one write() per line: short appends to an O_APPEND handle do not interleave
    # on Linux, so a second worker writing the same directory cannot corrupt a line.
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n += 1
    return n


def shuffle_correspondence(
    pairs: Sequence[tuple[_T, float]], seed: int, step: int
) -> tuple[list[tuple[_T, float]], int]:
    """Permute credits across the decisions that earned them. The correspondence control.

    The multiset of credit values is preserved exactly, as is each arm's count of credited
    decisions, so the treatment and this control differ on ONE thing: whether a credit is
    attached to the prompt that produced it. Anything the treatment gains that survives here
    came from the size or the spread of the credits and not from targeting.

    Args:
        pairs: ``(prior_decision, credit)`` as built by the actor for this step.
        seed: ``group_routing.credit_shuffle_seed``. Combined with ``step`` so consecutive
            steps get different permutations while the whole run stays reproducible from the
            one configured integer.
        step: The batch index.

    Returns:
        ``(shuffled_pairs, inert)``. ``inert`` is 1 when the permutation left every credit
        where it was and 0 otherwise. That covers all three ways this control can do nothing --
        fewer than two pairings, an identity draw, and a step whose credits are all equal --
        and it is counted rather than hidden because a control that could not have failed must
        not be reported as one.
    """
    values = [v for _, v in pairs]
    # Seeded from a STRING rather than a (seed, step) tuple: Python 3.11+ refuses a tuple
    # seed with a TypeError, and a control that raises on its first real step is worse than
    # one that never ran.
    rng = random.Random(f"credit-shuffle:{seed}:{step}")
    order = list(range(len(values)))
    rng.shuffle(order)
    shuffled = [values[i] for i in order]
    inert = int(shuffled == values)
    return [(prior, v) for (prior, _), v in zip(pairs, shuffled)], inert
