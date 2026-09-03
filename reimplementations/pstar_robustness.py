"""How robust is the p*=0.2 degeneracy criticism? Assumptions, stress tests, and limits.

The claim under scrutiny: Ornith's published target p* = 0.2 makes a large fraction of the
solver stage's GRPO groups carry exactly zero reward-directed gradient, and the better the
difficulty gate works the more budget it wastes.

This script exists because that is a criticism of a published method, so it has to be held
to a higher standard than a result of our own. It does four things:

  A. States the assumption set the claim needs, and tests each one that can be tested.
  B. Checks the claim is not an artefact of the synthetic pool's theta distribution.
  C. Checks whether the multiplicative structure V*D*N rescues it.
  D. Reports the strongest defence of p* < 1/2 we can construct, so the criticism is not
     stated more strongly than it deserves.

Everything is CPU-only and imports the shipped loop functions rather than re-deriving them.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from ornith_repro.grpo import grpo_advantages, predicted_binary_degeneracy
from ornith_repro.rewards import difficulty_reward


# ----------------------------------------------------------------- A. assumptions
ASSUMPTIONS = {
    "A-binary-rollout-reward": (
        "The solver-stage reward R_rollout = h(q,tau) is BINARY. This is the load-bearing "
        "assumption. The release defines p via a binary success test s, but never states "
        "that h itself is binary. If h returns a graded score, constant groups become rare "
        "and the criticism largely dissolves for that stage. Tested in check_continuous_h()."
    ),
    "B-iid-rollouts-within-group": (
        "Rollouts in a group are independent given the task. Real rollouts share a prompt "
        "and a policy and are positively correlated, which makes degeneracy HIGHER, so the "
        "closed form is a lower bound and the criticism is conservative here. "
        "Tested in check_correlated_rollouts()."
    ),
    "C-group-is-one-task": (
        "A GRPO group is k rollouts of a single task. If groups mix tasks, the reward "
        "spread comes from task difficulty as well and degeneracy falls. The release does "
        "not state group construction."
    ),
    "D-gate-actually-hits-its-target": (
        "The realised theta of trained-on tasks concentrates near p*. This is exactly what "
        "our gate experiment measured, and it is why the tension is real rather than "
        "hypothetical: accuracy of the gate and waste move together."
    ),
    "E-no-resampling": (
        "Degenerate groups are not detected and resampled. DAPO-style dynamic sampling "
        "would convert the waste from lost gradient into extra compute. The release states "
        "no such mechanism, but absence of a statement is not absence of the mechanism."
    ),
}


def check_continuous_h(rng: random.Random, group_size: int, reps: int) -> dict:
    """A: does the criticism survive if the harness returns a graded score, not a bit?

    Models h as binary success plus a continuous quality term with weight `w`. At w=0 the
    reward is purely binary; at w>0 groups almost never tie exactly.
    """
    out = {}
    for w in (0.0, 0.01, 0.05, 0.25):
        for theta in (0.2, 0.5):
            deg = 0
            for _ in range(reps):
                rewards = [
                    (1.0 if rng.random() < theta else 0.0) + w * rng.random()
                    for _ in range(group_size)
                ]
                if grpo_advantages(rewards).degenerate:
                    deg += 1
            out[f"w={w},theta={theta}"] = deg / reps
    return out


def check_correlated_rollouts(rng: random.Random, group_size: int, reps: int) -> dict:
    """B: positive within-group correlation via a shared latent shift.

    Each group draws a shared offset that moves every member's success probability
    together, which is what sharing a prompt and a policy actually does.
    """
    out = {}
    for rho_scale in (0.0, 0.1, 0.2):
        for theta in (0.2, 0.5):
            deg = 0
            for _ in range(reps):
                shift = (rng.random() - 0.5) * 2 * rho_scale
                p = min(1.0, max(0.0, theta + shift))
                rewards = [1.0 if rng.random() < p else 0.0 for _ in range(group_size)]
                if grpo_advantages(rewards).degenerate:
                    deg += 1
            out[f"spread={rho_scale},theta={theta}"] = deg / reps
    return out


def check_pool_shape(rng: random.Random, group_size: int, k: int, sigma: float,
                     keep_frac: float, reps: int) -> dict:
    """C1: is the result an artefact of theta ~ Uniform(0,1)?

    Re-runs the selection under three very different theta priors. If the conclusion only
    holds for the uniform pool it is an artefact of our fixture, not of the design.
    """
    priors = {
        "uniform": lambda: rng.random(),
        "beta(2,5)-easy-skewed": lambda: rng.betavariate(2, 5),
        "beta(5,2)-hard-skewed": lambda: rng.betavariate(5, 2),
        "bimodal-0.05/0.95": lambda: 0.05 if rng.random() < 0.5 else 0.95,
    }
    out = {}
    for name, draw in priors.items():
        pool = [draw() for _ in range(4000)]
        scored = []
        for theta in pool:
            p_hat = sum(1 for _ in range(k) if rng.random() < theta) / k
            scored.append((difficulty_reward(p_hat, 0.2, sigma), theta))
        scored.sort(key=lambda t: t[0], reverse=True)
        kept = [th for _, th in scored[: int(len(scored) * keep_frac)]]
        deg = statistics.fmean(
            [predicted_binary_degeneracy(th, group_size) for th in kept]
        )
        base = statistics.fmean(
            [predicted_binary_degeneracy(th, group_size) for th in pool]
        )
        out[name] = {
            "kept_mean_theta": statistics.fmean(kept),
            "kept_degeneracy": deg,
            "unselected_pool_degeneracy": base,
            "ratio_kept_over_pool": deg / base if base else float("nan"),
        }
    return out


def check_multiplicative(rng: random.Random, group_size: int, k: int, sigma: float,
                         keep_frac: float) -> dict:
    """C2: does the full product V*D*N change the conclusion versus D alone?

    If V and N vary across candidates, selecting on the product no longer concentrates
    theta as tightly at p*, which would blunt the degeneracy penalty. This is the specific
    rescue the coordinator asked to be checked.
    """
    out = {}
    for label, vspread, nspread in (
        ("D only (V=N=1)", 0.0, 0.0),
        ("mild V,N variation", 0.2, 0.2),
        ("strong V,N variation", 0.6, 0.6),
    ):
        scored = []
        for _ in range(4000):
            theta = rng.random()
            p_hat = sum(1 for _ in range(k) if rng.random() < theta) / k
            D = difficulty_reward(p_hat, 0.2, sigma)
            V = 1.0 - vspread * rng.random()
            N = 1.0 - nspread * rng.random()
            scored.append((V * D * N, theta))
        scored.sort(key=lambda t: t[0], reverse=True)
        kept = [th for _, th in scored[: int(len(scored) * keep_frac)]]
        out[label] = {
            "kept_mean_theta": statistics.fmean(kept),
            "kept_mean_abs_theta_minus_pstar": statistics.fmean(
                [abs(th - 0.2) for th in kept]
            ),
            "kept_degeneracy": statistics.fmean(
                [predicted_binary_degeneracy(th, group_size) for th in kept]
            ),
        }
    return out


def pstar_curve(group_size: int) -> dict:
    """D: the exact degeneracy as a function of the target, so the claim is stated right.

    theta^G + (1-theta)^G is monotone DECREASING on [0, 1/2]. So p*=0.2 is not a maximum
    of waste in any global sense -- the maximum is at theta -> 0 or 1. The correct
    statement is that waste is monotone in how far below 1/2 the target sits, and 0.2 pays
    a specific, large multiple relative to 1/2.
    """
    return {
        f"{t:.2f}": predicted_binary_degeneracy(t, group_size)
        for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    }


def task_lifetime_defence(group_size: int) -> dict:
    """The strongest defence of p* < 1/2 we can construct, stated honestly.

    A task at theta=0.5 yields the most informative groups NOW but is closer to being
    solved, so it leaves the useful band sooner. A task at theta=0.2 yields fewer
    informative groups per step but stays useful for longer. Model the useful lifetime as
    proportional to the distance the policy must travel before the task saturates
    (theta -> 1), and score total informative groups over the lifetime.

    This is a model we invented to steelman their choice, not something Ornith published.
    """
    out = {}
    for theta in (0.2, 0.3, 0.5):
        informative_per_step = 1.0 - predicted_binary_degeneracy(theta, group_size)
        lifetime = 1.0 - theta  # distance to saturation, in units of theta
        out[f"theta={theta}"] = {
            "informative_fraction_per_step": informative_per_step,
            "relative_lifetime": lifetime,
            "lifetime_informative_product": informative_per_step * lifetime,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--keep-frac", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=Path("pstar_robustness.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    res = {
        "config": vars(args) | {"out": str(args.out)},
        "assumptions": ASSUMPTIONS,
        "A_continuous_harness": check_continuous_h(rng, args.group_size, args.reps),
        "B_correlated_rollouts": check_correlated_rollouts(rng, args.group_size, args.reps),
        "C1_pool_shape": check_pool_shape(rng, args.group_size, args.k, args.sigma,
                                          args.keep_frac, args.reps),
        "C2_multiplicative": check_multiplicative(rng, args.group_size, args.k,
                                                  args.sigma, args.keep_frac),
        "D_pstar_curve": pstar_curve(args.group_size),
        "E_lifetime_defence": task_lifetime_defence(args.group_size),
    }
    args.out.write_text(json.dumps(res, indent=2))

    print(f"G={args.group_size} k={args.k} sigma={args.sigma} keep={args.keep_frac}\n")
    print("A. IF THE HARNESS RETURNS A GRADED SCORE, DOES THE CRITICISM SURVIVE?")
    for kk, v in res["A_continuous_harness"].items():
        print(f"   {kk:<22} degenerate {v:.4f}")
    print("\nB. POSITIVELY CORRELATED ROLLOUTS (shared prompt/policy)")
    for kk, v in res["B_correlated_rollouts"].items():
        print(f"   {kk:<22} degenerate {v:.4f}")
    print("\nC1. IS IT AN ARTEFACT OF THE UNIFORM theta POOL?")
    for kk, v in res["C1_pool_shape"].items():
        print(f"   {kk:<24} kept_theta {v['kept_mean_theta']:.3f}  "
              f"kept_deg {v['kept_degeneracy']:.4f}  pool_deg "
              f"{v['unselected_pool_degeneracy']:.4f}  ratio {v['ratio_kept_over_pool']:.2f}")
    print("\nC2. DOES THE PRODUCT V*D*N RESCUE IT?")
    for kk, v in res["C2_multiplicative"].items():
        print(f"   {kk:<24} kept_theta {v['kept_mean_theta']:.3f}  "
              f"|theta-p*| {v['kept_mean_abs_theta_minus_pstar']:.3f}  "
              f"kept_deg {v['kept_degeneracy']:.4f}")
    print("\nD. DEGENERACY AS A FUNCTION OF THE TARGET (exact)")
    for kk, v in res["D_pstar_curve"].items():
        print(f"   p*={kk}  {v:.4f}")
    print("\nE. STEELMAN: lifetime-weighted informative groups (our model, not theirs)")
    for kk, v in res["E_lifetime_defence"].items():
        print(f"   {kk:<12} informative/step {v['informative_fraction_per_step']:.4f}  "
              f"lifetime {v['relative_lifetime']:.2f}  product "
              f"{v['lifetime_informative_product']:.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
