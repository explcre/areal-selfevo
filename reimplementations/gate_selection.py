"""Does Ornith's difficulty gate select difficulty, or does it select noise?

A CPU-only experiment on a SYNTHETIC pool. It is not evidence about Ornith's models; it
is evidence about the published reward design, which is the only thing the release lets
anyone check. Every task in the pool has a KNOWN true success probability `theta`, which
is what makes the question answerable at all -- on real tasks `theta` is never observed,
which is precisely why the defect is invisible in a real run.

The loop's own functions are imported and used; nothing is re-derived here. A re-derived
copy of the reward would test the copy, not the code that ships.

Design:

  1. Draw a pool of tasks with true `theta ~ Uniform(0,1)` and an observable `family`
     label that is INDEPENDENT of theta (so the matched control cannot cheat).
  2. DISCOVERY block: k rollouts each, giving `p_hat`, then `D = exp(-(p_hat-p*)^2/2s^2)`.
  3. TREATMENT: keep the top-n tasks by D, exactly as R_task would rank them with V=N=1.
  4. CONTROL: keep n tasks drawn at random from the pool, size-matched, at the
     TREATMENT'S OWN MEASURED family proportions (not uniform).
  5. HOLDOUT block: a fresh, independent set of k rollouts for the kept tasks.

Reported:

  * winner's curse -- discovery D versus holdout D for the treatment;
  * whether selection actually enriches true theta near p* -- the honest question,
    answered against the matched control rather than against nothing;
  * the degenerate-group fraction the selected tasks then produce, against the closed
    form `theta^G + (1-theta)^G`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from ornith_repro.controls import measure_proportions, random_task_control
from ornith_repro.grpo import grpo_advantages, predicted_binary_degeneracy
from ornith_repro.rewards import difficulty_reward
from ornith_repro.types import Task

FAMILIES = ("algebra", "geometry", "number_theory", "combinatorics")


def draw_pool(n: int, rng: random.Random) -> list[tuple[Task, float]]:
    """Draw `n` (task, true_theta) pairs with family independent of theta."""
    out = []
    for i in range(n):
        theta = rng.random()
        fam = FAMILIES[rng.randrange(len(FAMILIES))]
        out.append(
            (Task(task_id=f"q{i}", text=f"synthetic task {i} in {fam}", family=fam), theta)
        )
    return out


def rollout_block(theta: float, k: int, rng: random.Random) -> float:
    """Return p_hat from k independent Bernoulli(theta) rollouts."""
    return sum(1 for _ in range(k) if rng.random() < theta) / k


def degenerate_rate(theta: float, group_size: int, reps: int, rng: random.Random) -> float:
    """Measured fraction of degenerate GRPO groups at this theta."""
    deg = 0
    for _ in range(reps):
        rewards = [1.0 if rng.random() < theta else 0.0 for _ in range(group_size)]
        if grpo_advantages(rewards).degenerate:
            deg += 1
    return deg / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--keep", type=int, default=400)
    ap.add_argument("--k", type=int, default=8, help="rollouts per task (ambiguity A1)")
    ap.add_argument("--sigma", type=float, default=0.15, help="kernel width (ambiguity A2)")
    ap.add_argument("--p-star", type=float, default=0.2, help="published as 0.2")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", type=Path, default=Path("gate_selection_results.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = draw_pool(args.pool, rng)

    # --- discovery block --------------------------------------------------------
    scored = []
    for task, theta in pool:
        p_hat = rollout_block(theta, args.k, rng)
        D = difficulty_reward(p_hat, p_star=args.p_star, sigma=args.sigma)
        scored.append({"task": task, "theta": theta, "p_hat": p_hat, "D": D})

    # --- treatment: top-n by D (how R_task ranks with V=N=1) --------------------
    scored.sort(key=lambda r: r["D"], reverse=True)
    treatment = scored[: args.keep]
    treat_tasks = [r["task"] for r in treatment]

    # --- control: size-matched, at the treatment's MEASURED family proportions --
    target_props = measure_proportions(treat_tasks, "family")
    control_tasks = random_task_control(
        treat_tasks, [t for t, _ in pool], key="family", seed=args.seed
    )
    theta_by_text = {t.text: th for t, th in pool}
    control = [{"task": t, "theta": theta_by_text[t.text]} for t in control_tasks]

    # --- holdout block: fresh independent rollouts ------------------------------
    for r in treatment:
        r["p_hat_holdout"] = rollout_block(r["theta"], args.k, rng)
        r["D_holdout"] = difficulty_reward(
            r["p_hat_holdout"], p_star=args.p_star, sigma=args.sigma
        )
    for r in control:
        r["p_hat_holdout"] = rollout_block(r["theta"], args.k, rng)
        r["D_holdout"] = difficulty_reward(
            r["p_hat_holdout"], p_star=args.p_star, sigma=args.sigma
        )

    def mean(xs):
        return statistics.fmean(xs)

    def sem(xs):
        return statistics.stdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0

    t_disc = mean([r["D"] for r in treatment])
    t_hold = mean([r["D_holdout"] for r in treatment])
    c_hold = mean([r["D_holdout"] for r in control])

    # The honest question: does the gate enrich tasks whose TRUE theta is near p*?
    t_gap = [abs(r["theta"] - args.p_star) for r in treatment]
    c_gap = [abs(r["theta"] - args.p_star) for r in control]

    # Degeneracy the selected tasks then produce.
    t_deg = mean([degenerate_rate(r["theta"], args.group_size, 200, rng) for r in treatment])
    c_deg = mean([degenerate_rate(r["theta"], args.group_size, 200, rng) for r in control])

    diff = mean(c_gap) - mean(t_gap)
    # SE on the DIFFERENCE of two independent arms.
    diff_se = (sem(t_gap) ** 2 + sem(c_gap) ** 2) ** 0.5

    res = {
        "config": vars(args) | {"out": str(args.out)},
        "matched_family_proportions": target_props,
        "control_family_proportions": measure_proportions(control_tasks, "family"),
        "winners_curse": {
            "treatment_D_discovery": t_disc,
            "treatment_D_holdout": t_hold,
            "shrinkage": t_disc - t_hold,
            "control_D_holdout": c_hold,
        },
        "true_difficulty_enrichment": {
            "treatment_mean_abs_theta_minus_pstar": mean(t_gap),
            "treatment_sem": sem(t_gap),
            "control_mean_abs_theta_minus_pstar": mean(c_gap),
            "control_sem": sem(c_gap),
            "difference_control_minus_treatment": diff,
            "se_on_difference": diff_se,
            "z": diff / diff_se if diff_se else float("nan"),
        },
        "degeneracy": {
            "treatment_measured": t_deg,
            "control_measured": c_deg,
            "closed_form_at_p_star_0.2": predicted_binary_degeneracy(0.2, args.group_size),
            "closed_form_at_0.5": predicted_binary_degeneracy(0.5, args.group_size),
        },
    }
    args.out.write_text(json.dumps(res, indent=2))

    print(f"pool={args.pool} keep={args.keep} k={args.k} sigma={args.sigma} p*={args.p_star}")
    print(f"\nWINNER'S CURSE (treatment, selected on D):")
    print(f"  discovery D          {t_disc:.4f}")
    print(f"  holdout   D          {t_hold:.4f}   shrinkage {t_disc - t_hold:+.4f}")
    print(f"  matched control D    {c_hold:.4f}")
    print(f"\nDOES THE GATE ENRICH TRUE DIFFICULTY?  (|theta - p*|, lower = better)")
    print(f"  treatment            {mean(t_gap):.4f} +/- {sem(t_gap):.4f}")
    print(f"  matched control      {mean(c_gap):.4f} +/- {sem(c_gap):.4f}")
    print(f"  control - treatment  {diff:+.4f} +/- {diff_se:.4f}  (z = {diff/diff_se:.2f})")
    print(f"\nDEGENERATE GROUPS at G={args.group_size} produced by the kept tasks:")
    print(f"  treatment            {t_deg:.4f}")
    print(f"  matched control      {c_deg:.4f}")
    print(f"  closed form @0.2     {predicted_binary_degeneracy(0.2, args.group_size):.4f}")
    print(f"  closed form @0.5     {predicted_binary_degeneracy(0.5, args.group_size):.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
