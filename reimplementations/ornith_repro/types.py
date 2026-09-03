"""Core record types for the Ornith-1.5 loop reimplementation.

Every stage of the loop emits one of these records. They are plain dataclasses so that
they serialise to JSONL without a schema library, and so that a test can assert on
*observable state* (the field values written to disk) rather than on a function that
returned True.

The `provenance` field on each record exists to defeat the "stage did not run but its
artifact still looks plausible" failure. It is a digest over that stage's actual inputs
and outputs, so a defaulted, copied or hand-written record cannot reproduce it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class RolloutOutcome(str, Enum):
    """Three-valued rollout outcome.

    Deliberately three-valued. A generation that was cut off by a token cap, a server
    refusal, or a timeout is ABORTED, which is *not* the same event as the policy
    producing a wrong answer. Collapsing ABORTED into FAILURE silently inflates the
    apparent difficulty of a task and is the single silent-zero path that has cost this
    project the most (aborted generations graded as wrong answers). See guard G1.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"


def digest(*parts: Any) -> str:
    """Return a short stable digest over the given parts.

    Used for stage provenance. `json.dumps(..., sort_keys=True, default=str)` gives a
    stable encoding for the dataclasses and primitives we pass in.

    Args:
        *parts: Arbitrary JSON-encodable values forming the stage's inputs and outputs.

    Returns:
        A 16-character hex digest.
    """
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Task:
    """A generated task `q`.

    Attributes:
        task_id: Stable identifier.
        text: The task statement handed to the solver.
        family: Observable covariate used by the size-matched random control to match
            proportions. Never used by the reward.
        length_bin: Observable covariate, as above.
        source: "generated" for the treatment, "pool" for the random control. Recorded so
            that no analysis can silently mix the arms.
    """

    task_id: str
    text: str
    family: str = "unknown"
    length_bin: str = "unknown"
    source: str = "generated"

    def covariates(self) -> dict[str, str]:
        """Return the observable covariates used for proportion matching."""
        return {"family": self.family, "length_bin": self.length_bin}


@dataclass
class Scaffold:
    """A generated scaffold / harness `h`.

    In the released description the scaffold carries instructions, tools, decomposition
    and orchestration, and it *is* the reward function for the rollout stage
    (`R_rollout(tau_i) = h(q, tau_i)`).

    Attributes:
        scaffold_id: Stable identifier.
        instructions: Free text handed to the solver.
        tools: Names of tools the scaffold exposes.
        grader_kind: How this scaffold grades a rollout. Recorded because it determines
            `R_rollout` and therefore every downstream reward.
        source: "generated" or "pool" (random control).
    """

    scaffold_id: str
    instructions: str
    tools: tuple[str, ...] = ()
    grader_kind: str = "exact"
    source: str = "generated"

    def covariates(self) -> dict[str, str]:
        """Return the observable covariates used for proportion matching."""
        return {"grader_kind": self.grader_kind, "n_tools": str(len(self.tools))}


@dataclass
class Rollout:
    """A single solver rollout `tau_i` together with how the scaffold graded it."""

    rollout_id: str
    text: str
    outcome: RolloutOutcome
    reward: float
    abort_reason: str | None = None


@dataclass
class TaskRecord:
    """Everything the task stage produced and was scored on.

    `p_hat`, `n_valid` and `n_aborted` are stored separately so that an analysis can
    always recover how many rollouts actually contributed to the difficulty estimate.
    Storing only `p_hat` would make an abort-inflated estimate indistinguishable from a
    genuine one after the fact.
    """

    task: Task
    scaffold: Scaffold
    rollouts: list[Rollout]
    p_hat: float
    n_valid: int
    n_aborted: int
    V: float
    D: float
    N: float
    R_task: float
    empty_buffer: bool
    provenance: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-encodable dict of this record."""
        d = asdict(self)
        d["rollouts"] = [
            {**asdict(r), "outcome": r.outcome.value} for r in self.rollouts
        ]
        return d


@dataclass
class HarnessRecord:
    """Everything the harness/scaffold stage produced and was scored on."""

    task: Task
    scaffold: Scaffold
    C: float
    F: float
    H: float
    R_harness: float
    provenance: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-encodable dict of this record."""
        return asdict(self)


@dataclass
class GroupAdvantages:
    """Result of the GRPO group-advantage computation.

    Attributes:
        advantages: Per-member advantage.
        degenerate: True when every reward in the group was identical, so the group
            carries exactly zero reward-directed gradient. Returned explicitly rather
            than being hidden behind an epsilon, because a batch of such groups produces
            a false negative when one averages "the gradient is small".
        reward_std: Population standard deviation of the group's rewards, before epsilon.
    """

    advantages: list[float]
    degenerate: bool
    reward_std: float


@dataclass
class RunMeta:
    """Run-level provenance written next to the artifacts.

    `base_model` and `served_models` are recorded because an unregistered model id is
    served as the base model silently, with a 200 and no warning. Recording the id we
    asked for next to the ids the backend actually reports is the only way to detect it
    after the fact.
    """

    run_id: str
    base_model: str
    backend: str
    served_models: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
