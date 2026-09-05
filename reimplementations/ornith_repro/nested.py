"""Nested sampling: task -> several scaffolds -> several rollouts each.

WHY THIS MODULE EXISTS. Our first implementation (`loop.run_iteration`) was FLAT: one
scaffold per task, one rollout group, and -- worse -- it handed the rollout advantages to
all three stages, so the scaffold and task stages never formed groups of their own. That is
wrong under the nested reading, and GRPO's advantage is group-relative, so the group
structure IS the algorithm rather than a detail of it.

SOURCE STATUS, stated because it is contested. The nesting comes from Ornith's release
FIGURE (`ornith_self_improvement_loop`), which shows each task producing multiple scaffolds
and each scaffold producing several rollouts. The PROSE does not state it: it says the
policy produces "a solution rollout", singular, while the task reward references a set of
rollouts `{tau_i}` for the success rate. The figure is the more specific evidence, so we
treat nested as the faithful reading and keep flat as an ablation. Ambiguity id **A16**.

Neither count is disclosed. `n_scaffolds` and `n_rollouts_per_scaffold` are OUR choices
(ambiguity **A17**) and are named parameters, never constants.

THE TWO COMPARISON LEVELS this creates:

  * within a scaffold: its rollouts compete on `R_rollout = h(q, tau)`;
  * within a task: its scaffolds compete on `R_harness = C*F*H`, where `F` is an aggregate
    over that scaffold's OWN rollouts.

THE HAZARD, and it is the difficulty gate's winner's curse one level up. A scaffold's
reward is an aggregate over the rollouts it happened to draw, so a scaffold that drew lucky
rollouts scores well for reasons that have nothing to do with the scaffold. Selecting or
crediting scaffolds on that number inherits the same optimism. `scaffold_holdout_rewards`
supports the same fresh-block treatment: score the scaffolds on one rollout block, re-score
on an independent one, and report both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .buffer import TaskBuffer
from .grpo import grpo_advantages
from .guards import (GuardViolation, assert_batch_not_all_degenerate,
                     assert_task_not_in_buffer)
from .judges import Judges
from .loop import OrnithConfig, grade
from .rewards import (
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
    Scaffold,
    Task,
    TaskRecord,
    digest,
)


@dataclass
class NestedConfig:
    """Counts for the nested structure. Both are ours; neither is disclosed (A17).

    Attributes:
        n_scaffolds: Scaffolds generated per task (the figure shows several).
        n_rollouts_per_scaffold: Rollouts generated per scaffold.
        sampling: "nested" (faithful to the figure) or "flat" (the ablation, one
            scaffold per task, one rollout group).
    """

    n_scaffolds: int = 3
    n_rollouts_per_scaffold: int = 8
    sampling: str = "nested"

    def __post_init__(self) -> None:
        if self.sampling not in ("nested", "flat"):
            raise ValueError(f"sampling must be nested|flat, got {self.sampling}")
        if self.sampling == "nested" and self.n_scaffolds < 2:
            raise ValueError(
                f"nested sampling needs >= 2 scaffolds to form a scaffold group, got "
                f"{self.n_scaffolds}; use sampling='flat' for the single-scaffold ablation"
            )
        if self.n_rollouts_per_scaffold < 2:
            raise ValueError(
                f"need >= 2 rollouts per scaffold to form a rollout group, got "
                f"{self.n_rollouts_per_scaffold}"
            )

    @property
    def total_rollouts(self) -> int:
        """Rollout budget per task, which is what arms must be matched on."""
        n = self.n_scaffolds if self.sampling == "nested" else 1
        return n * self.n_rollouts_per_scaffold


@dataclass
class NestedIterationResult:
    """One nested iteration, with both comparison levels kept separate."""

    task_record: TaskRecord
    harness_records: list[HarnessRecord]
    rollout_advantages: list[GroupAdvantages]        # one per scaffold
    scaffold_advantages: GroupAdvantages | None      # across scaffolds; None when flat
    scaffold_rewards_discovery: list[float] = field(default_factory=list)
    scaffold_rewards_holdout: list[float] = field(default_factory=list)
    stages_fired: list[str] = field(default_factory=list)


def run_iteration_nested(
    cfg: OrnithConfig,
    ncfg: NestedConfig,
    task: Task,
    scaffolds: Sequence[Scaffold],
    rollout_texts_by_scaffold: Sequence[Sequence[tuple[str, bool]]],
    buffer: TaskBuffer,
    judges: Judges,
    sim: Callable[[str, str], float] = jaccard_similarity,
    holdout_texts_by_scaffold: Sequence[Sequence[tuple[str, bool]]] | None = None,
) -> NestedIterationResult:
    """Run one nested iteration: several scaffolds per task, several rollouts each.

    Args:
        cfg: Loop configuration (p*, sigma, abort policy, epsilon...).
        ncfg: Nested counts and sampling mode.
        task: The task `q`.
        scaffolds: The scaffolds generated for this task.
        rollout_texts_by_scaffold: `(text, truncated)` pairs, per scaffold.
        buffer: Task buffer; read for novelty, written after scoring.
        judges: Frozen judges.
        sim: Similarity for novelty.
        holdout_texts_by_scaffold: Optional independent rollout block. When given, the
            scaffold rewards are recomputed on it, so the scaffold-level winner's curse can
            be measured rather than assumed away.

    Returns:
        A `NestedIterationResult` carrying both comparison levels.

    Raises:
        ValueError: on a shape mismatch between scaffolds and rollout blocks, or when the
            configured counts are not met.
        GuardViolation: from any guard that fires.
    """
    if len(scaffolds) != len(rollout_texts_by_scaffold):
        raise ValueError(
            f"{len(scaffolds)} scaffolds but {len(rollout_texts_by_scaffold)} rollout "
            f"blocks; the nesting is malformed."
        )
    expected = ncfg.n_scaffolds if ncfg.sampling == "nested" else 1
    if len(scaffolds) != expected:
        raise ValueError(
            f"sampling={ncfg.sampling!r} expects {expected} scaffold(s), got "
            f"{len(scaffolds)}."
        )
    for j, block in enumerate(rollout_texts_by_scaffold):
        if len(block) != ncfg.n_rollouts_per_scaffold:
            raise ValueError(
                f"scaffold {j} has {len(block)} rollouts, expected "
                f"{ncfg.n_rollouts_per_scaffold}; arms must be budget-matched."
            )

    assert_task_not_in_buffer(task, buffer.texts())
    stages: list[str] = []

    # ---- level 2: rollouts within each scaffold --------------------------------
    all_rollouts: list[Rollout] = []
    per_scaffold_rollouts: list[list[Rollout]] = []
    rollout_advs: list[GroupAdvantages] = []
    for sc, block in zip(scaffolds, rollout_texts_by_scaffold):
        rs = [grade(sc, task, t, tr) for t, tr in block]
        per_scaffold_rollouts.append(rs)
        all_rollouts.extend(rs)
        rollout_advs.append(
            grpo_advantages([r.reward for r in rs], epsilon=cfg.epsilon)
        )
    stages.append("rollout")

    # ---- difficulty over ALL of the task's rollouts -----------------------------
    p_hat, n_valid, n_aborted = success_rate(all_rollouts, abort_policy=cfg.abort_policy)
    if n_valid < cfg.min_valid_rollouts:
        raise ValueError(
            f"only {n_valid} gradeable rollouts across all scaffolds (< "
            f"min_valid_rollouts={cfg.min_valid_rollouts}); task refused (guard G1)."
        )

    # ---- level 1: scaffolds within the task -------------------------------------
    harness_records: list[HarnessRecord] = []
    disc: list[float] = []
    for sc, rs in zip(scaffolds, per_scaffold_rollouts):
        C = judges.alignment(task, sc)
        F = judges.reward_fidelity(sc, rs)   # aggregate over this scaffold's OWN rollouts
        H = judges.hack_resistance(sc)
        R = harness_reward(C, F, H)
        disc.append(R)
        harness_records.append(
            HarnessRecord(task=task, scaffold=sc, C=C, F=F, H=H, R_harness=R,
                          provenance=digest(task.task_id, sc.scaffold_id, C, F, H, R))
        )
    stages.append("harness")

    # G4, which had no caller anywhere in the package: refuse a batch in which every group
    # carries zero reward-directed gradient. Applied to the ROLLOUT groups, because a task
    # whose every scaffold produced a unanimous block contributes nothing and would
    # otherwise be indistinguishable, after the fact, from one that trained.
    assert_batch_not_all_degenerate(rollout_advs)

    scaffold_adv = (
        grpo_advantages(disc, epsilon=cfg.epsilon) if ncfg.sampling == "nested" else None
    )

    # ---- fresh-block scaffold rewards, for the winner's curse one level up -------
    hold: list[float] = []
    if holdout_texts_by_scaffold is not None:
        if len(holdout_texts_by_scaffold) != len(scaffolds):
            raise ValueError("holdout block count does not match scaffold count.")
        for sc, block in zip(scaffolds, holdout_texts_by_scaffold):
            rs = [grade(sc, task, t, tr) for t, tr in block]
            hold.append(
                harness_reward(
                    judges.alignment(task, sc),
                    judges.reward_fidelity(sc, rs),
                    judges.hack_resistance(sc),
                )
            )

    # ---- task reward -------------------------------------------------------------
    best = max(scaffolds, key=lambda s: disc[scaffolds.index(s)]) if scaffolds else None
    V = judges.validity(task, best, mode=cfg.validity_mode)
    D = difficulty_reward(p_hat, p_star=cfg.p_star, sigma=cfg.sigma)
    N, empty_buffer = novelty_reward(task.text, buffer.texts(), sim=sim)
    R_task = task_reward(V, D, N)
    stages.append("task")

    task_record = TaskRecord(
        task=task, scaffold=best, rollouts=all_rollouts, p_hat=p_hat, n_valid=n_valid,
        n_aborted=n_aborted, V=V, D=D, N=N, R_task=R_task, empty_buffer=empty_buffer,
        provenance=digest(task.task_id, best.scaffold_id,
                          [r.rollout_id for r in all_rollouts], p_hat, V, D, N, R_task),
    )
    buffer.add(task)

    return NestedIterationResult(
        task_record=task_record,
        harness_records=harness_records,
        rollout_advantages=rollout_advs,
        scaffold_advantages=scaffold_adv,
        scaffold_rewards_discovery=disc,
        scaffold_rewards_holdout=hold,
        stages_fired=stages,
    )
