"""The Ornith-1.5 loop: three sequential stages per iteration, rewards propagated.

Per the method page, one iteration is:

  1. task generation      -- propose a task harder than what the model has solved
  2. scaffold generation  -- produce/refine a task-specific scaffold for that task
  3. rollout              -- solve, conditioned on task and scaffold

with the rollout reward propagating back into the first two stages, and all three
optimised by GRPO on their own rewards.

Reconstruction choices made here, all recorded in AMBIGUITIES.md:

* A10 update order: we score all three stages on the *same* rollout batch and apply
  solver -> harness -> proposer. "Jointly optimised" does not specify an order and the
  stages share rollouts, so the order is ours and is configurable.
* A11 what is frozen: the judges producing V, C, F, H are frozen and never trained. The
  source does not say. Training them on the same rollouts would close the last loop and
  make reward hacking unmeasurable even in principle.
* A8 buffer: append-only, and the scored task is inserted *after* scoring (guard G5).

`apply_update` is a seam, not a trainer. On CPU it records what *would* be updated, so
the whole loop including artifact write/read-back is exercised without a GPU. A real
GRPO trainer is substituted by passing `updater=`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .buffer import TaskBuffer
from .grpo import grpo_advantages
from .guards import assert_batch_not_all_degenerate, assert_task_not_in_buffer
from .judges import Judges
from .rewards import (
    P_STAR_PUBLISHED,
    difficulty_reward,
    harness_reward,
    jaccard_similarity,
    novelty_reward,
    success_rate,
    task_reward,
)
from .types import (
    GroupAdvantages,
    HarnessRecord,
    Rollout,
    RolloutOutcome,
    Scaffold,
    Task,
    TaskRecord,
    digest,
)


@dataclass
class OrnithConfig:
    """Configuration. Every undisclosed quantity is here with its ambiguity id.

    Attributes:
        base_model: The base model id. A PARAMETER, never a constant (A14). Target
            "Qwen/Qwen3.8-27B"; fallback "Qwen/Qwen3-32B"; last resort
            "Qwen2.5-32B-Instruct". Ornith's own bases were Qwen3.5 and Gemma 4, so ours
            is a deliberate substitution and is labelled as one.
        p_star: Target success rate. The ONLY published hyperparameter (0.2).
        sigma: Difficulty kernel width. NOT published (A2).
        k_rollouts: Rollouts per task (A1).
        group_size: GRPO group size (A3).
        min_valid_rollouts: Refuse a task with fewer gradeable rollouts than this (G1).
        abort_policy: See `rewards.success_rate` (G1).
        validity_mode: "soft" (V in [0,1]) or "gate" (V in {0,1}) (A4).
        s_is_h: Whether the success test is the generated scaffold itself (A5).
        stage_order: Order in which the three stages are updated (A10).
        epsilon: GRPO denominator floor (A12).
        max_new_tokens: Generation cap.
    """

    base_model: str = "Qwen/Qwen3.8-27B"
    p_star: float = P_STAR_PUBLISHED
    sigma: float = 0.15
    k_rollouts: int = 8
    group_size: int = 8
    min_valid_rollouts: int = 4
    abort_policy: str = "exclude"
    validity_mode: str = "soft"
    s_is_h: bool = True
    stage_order: tuple[str, ...] = ("solver", "harness", "proposer")
    epsilon: float = 1e-6
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.validity_mode not in ("soft", "gate"):
            raise ValueError(f"validity_mode must be soft|gate, got {self.validity_mode}")
        if set(self.stage_order) != {"solver", "harness", "proposer"}:
            raise ValueError(f"stage_order must cover all three stages, got {self.stage_order}")
        if self.min_valid_rollouts < 2:
            raise ValueError("min_valid_rollouts must be >= 2 (guard G3)")


@dataclass
class IterationResult:
    """Everything one loop iteration produced, for assertion and for artifacts."""

    task_record: TaskRecord
    harness_record: HarnessRecord
    rollout_advantages: GroupAdvantages
    updates_applied: list[str] = field(default_factory=list)
    stages_fired: list[str] = field(default_factory=list)


def grade(scaffold: Scaffold, task: Task, text: str, truncated: bool) -> Rollout:
    """Grade one rollout with the scaffold, which IS the rollout reward function.

    Implements `R_rollout(tau_i) = h(q, tau_i)`.

    Args:
        scaffold: The scaffold `h`, whose `grader_kind` decides how text is graded.
        task: The task `q`.
        text: The generated rollout text.
        truncated: Whether generation was cut off.

    Returns:
        A `Rollout`. A truncated generation becomes `ABORTED` with reward 0.0 and an
        `abort_reason`; it is NOT a `FAILURE`. Keeping those distinct is guard G1.
    """
    rid = digest(scaffold.scaffold_id, task.task_id, text)
    if truncated:
        return Rollout(
            rollout_id=rid,
            text=text,
            outcome=RolloutOutcome.ABORTED,
            reward=0.0,
            abort_reason="truncated_or_no_stop",
        )
    ok = "SOLVED" in text if scaffold.grader_kind == "exact" else text.strip() != ""
    return Rollout(
        rollout_id=rid,
        text=text,
        outcome=RolloutOutcome.SUCCESS if ok else RolloutOutcome.FAILURE,
        reward=1.0 if ok else 0.0,
    )


def run_iteration(
    cfg: OrnithConfig,
    task: Task,
    scaffold: Scaffold,
    rollout_texts: Sequence[tuple[str, bool]],
    buffer: TaskBuffer,
    judges: Judges,
    sim: Callable[[str, str], float] = jaccard_similarity,
    updater: Callable[[str, GroupAdvantages], None] | None = None,
) -> IterationResult:
    """Run one full iteration and return every intermediate quantity.

    Args:
        cfg: Configuration.
        task: The generated (or control) task `q`.
        scaffold: The generated (or control) scaffold `h`.
        rollout_texts: `(text, truncated)` pairs, one per rollout.
        buffer: The task buffer `B`. Read for novelty, written *after* scoring.
        judges: Frozen judges producing V, C, F, H.
        sim: Similarity for novelty (A6).
        updater: Optional real trainer. Called once per stage in `cfg.stage_order`.

    Returns:
        An `IterationResult` with the task record, harness record, rollout advantages,
        the stages that fired, and the updates applied.

    Raises:
        GuardViolation: from any guard that fires (G1/G3/G4/G5).
        ValueError: if fewer than `min_valid_rollouts` rollouts were gradeable.
    """
    stages_fired: list[str] = []

    # ---- guard G5: score novelty before the task enters the buffer -------------
    assert_task_not_in_buffer(task, buffer.texts())

    # ---- stage 3 (rollouts) ----------------------------------------------------
    rollouts = [grade(scaffold, task, t, tr) for t, tr in rollout_texts]
    stages_fired.append("rollout")

    p_hat, n_valid, n_aborted = success_rate(rollouts, abort_policy=cfg.abort_policy)
    if n_valid < cfg.min_valid_rollouts:
        raise ValueError(
            f"only {n_valid} gradeable rollouts (< min_valid_rollouts="
            f"{cfg.min_valid_rollouts}); task refused rather than scored on noise "
            f"({n_aborted} aborted) (guard G1)."
        )

    rollout_adv = grpo_advantages([r.reward for r in rollouts], epsilon=cfg.epsilon)

    # ---- stage 1 scoring (task) ------------------------------------------------
    V = judges.validity(task, scaffold, mode=cfg.validity_mode)
    D = difficulty_reward(p_hat, p_star=cfg.p_star, sigma=cfg.sigma)
    N, empty_buffer = novelty_reward(task.text, buffer.texts(), sim=sim)
    R_task = task_reward(V, D, N)
    stages_fired.append("task")

    task_record = TaskRecord(
        task=task,
        scaffold=scaffold,
        rollouts=rollouts,
        p_hat=p_hat,
        n_valid=n_valid,
        n_aborted=n_aborted,
        V=V,
        D=D,
        N=N,
        R_task=R_task,
        empty_buffer=empty_buffer,
        provenance=digest(task.task_id, scaffold.scaffold_id,
                          [r.rollout_id for r in rollouts], p_hat, V, D, N, R_task),
    )

    # ---- stage 2 scoring (harness/scaffold) ------------------------------------
    C = judges.alignment(task, scaffold)
    F = judges.reward_fidelity(scaffold, rollouts)
    H = judges.hack_resistance(scaffold)
    R_harness = harness_reward(C, F, H)
    stages_fired.append("harness")

    harness_record = HarnessRecord(
        task=task,
        scaffold=scaffold,
        C=C,
        F=F,
        H=H,
        R_harness=R_harness,
        provenance=digest(task.task_id, scaffold.scaffold_id, C, F, H, R_harness),
    )

    # ---- updates, in the configured order --------------------------------------
    updates: list[str] = []
    for stage in cfg.stage_order:
        if updater is not None:
            updater(stage, rollout_adv)
        updates.append(stage)

    # ---- buffer insert AFTER scoring (A8 / guard G5) ---------------------------
    buffer.add(task)

    return IterationResult(
        task_record=task_record,
        harness_record=harness_record,
        rollout_advantages=rollout_adv,
        updates_applied=updates,
        stages_fired=stages_fired,
    )


def write_artifacts(path: Path, result: IterationResult) -> Path:
    """Append one iteration's records to a JSONL artifact file.

    Args:
        path: Destination JSONL path. Parent directories are created.
        result: The iteration to persist.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_record": result.task_record.to_json(),
        "harness_record": result.harness_record.to_json(),
        "rollout_advantages": {
            "advantages": result.rollout_advantages.advantages,
            "degenerate": result.rollout_advantages.degenerate,
            "reward_std": result.rollout_advantages.reward_std,
        },
        "updates_applied": result.updates_applied,
        "stages_fired": result.stages_fired,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def read_artifacts(path: Path) -> list[dict]:
    """Read back the JSONL artifact file written by `write_artifacts`."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify_provenance(row: dict) -> None:
    """Recompute each stage's provenance digest from the artifact and compare.

    Args:
        row: One row as returned by `read_artifacts`.

    Raises:
        GuardViolation: if either digest does not match the record's contents, which
            means the stage did not actually run on the inputs it claims (guard G7).
    """
    from .guards import GuardViolation

    tr = row["task_record"]
    expect_task = digest(
        tr["task"]["task_id"],
        tr["scaffold"]["scaffold_id"],
        [r["rollout_id"] for r in tr["rollouts"]],
        tr["p_hat"], tr["V"], tr["D"], tr["N"], tr["R_task"],
    )
    if expect_task != tr["provenance"]:
        raise GuardViolation(
            f"task-stage provenance mismatch: artifact says {tr['provenance']}, "
            f"recomputation gives {expect_task}. The stage did not run on these inputs "
            f"(guard G7)."
        )
    hr = row["harness_record"]
    expect_harness = digest(
        hr["task"]["task_id"], hr["scaffold"]["scaffold_id"],
        hr["C"], hr["F"], hr["H"], hr["R_harness"],
    )
    if expect_harness != hr["provenance"]:
        raise GuardViolation(
            f"harness-stage provenance mismatch: artifact says {hr['provenance']}, "
            f"recomputation gives {expect_harness} (guard G7)."
        )
