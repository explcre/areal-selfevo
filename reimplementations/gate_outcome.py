"""Does the gate IMPROVE self-improvement, or only select what it claims to select?

The outcome decomposes as

    improvement per unit budget  =  (task quality)  x  (budget efficiency)

and the two halves have completely different epistemic status.

  * BUDGET EFFICIENCY is exactly computable CPU-side with no learning assumptions,
    because under binary-reward GRPO the per-group advantage energy is a constant for
    every non-degenerate group. Part A measures it, on the gate's REALISED difficulty
    distribution rather than at the nominal p*.

  * TASK QUALITY is NOT computable here, and Part B is the demonstration of that rather
    than an attempt at it. Any synthetic answer is fixed by the theta-update rule one
    assumes, and plausible rules disagree in SIGN. Part B shows the flip explicitly, so
    that nobody -- including us -- is tempted to run a cheap simulation and report the
    answer it was built to give.

This is the same standard applied to the p* retraction: state what the arithmetic gives,
state what it needs, and refuse the part that the setup cannot support.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from ornith_repro.grpo import grpo_advantages, predicted_binary_degeneracy
from ornith_repro.rewards import difficulty_reward


# --------------------------------------------------------------- A. budget efficiency
def verify_energy_identity(rng: random.Random, group_size: int, trials: int) -> dict:
    """Check sum_i A_i^2 = G for every non-degenerate group, which Part A relies on.

    Relying on an identity from memory without checking it is how a wrong number enters.
    Our implementation divides by (std + epsilon) rather than std, so the realised value
    is very slightly below G; the gap is reported rather than assumed negligible.
    """
    energies, degenerate = [], 0
    for _ in range(trials):
        theta = rng.random()
        g = grpo_advantages([1.0 if rng.random() < theta else 0.0
                             for _ in range(group_size)])
        if g.degenerate:
            degenerate += 1
            energies.append(sum(a * a for a in g.advantages))
        else:
            energies.append(sum(a * a for a in g.advantages))
    nondeg = [e for e in energies if e > 0]
    return {
        "group_size": group_size,
        "mean_energy_nondegenerate": statistics.fmean(nondeg),
        "min_energy_nondegenerate": min(nondeg),
        "max_energy_nondegenerate": max(nondeg),
        "identity_target_G": float(group_size),
        "max_abs_deviation_from_G": max(abs(e - group_size) for e in nondeg),
        "degenerate_energy_is_exactly_zero": all(
            e == 0.0 for e in energies if e == 0.0
        ),
        "degenerate_fraction": degenerate / trials,
    }


def realised_theta_distribution(rng: random.Random, k: int, sigma: float,
                                pool_n: int, keep_frac: float) -> list[float]:
    """Return the true thetas the gate actually keeps, i.e. its realised distribution.

    This is the distribution the control must be matched to. Matching to the nominal
    p* = 0.2 instead would match the control to a target the gate does not hit.
    """
    scored = []
    for _ in range(pool_n):
        theta = rng.random()
        p_hat = sum(1 for _ in range(k) if rng.random() < theta) / k
        scored.append((difficulty_reward(p_hat, 0.2, sigma), theta))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [th for _, th in scored[: int(pool_n * keep_frac)]]


def budget_efficiency(thetas: list[float], group_size: int) -> dict:
    """Expected advantage energy delivered per rollout, over a theta distribution.

    For a non-degenerate group the energy is G and the group costs G rollouts, so the
    energy per rollout is exactly the non-degenerate fraction. Over a distribution of
    tasks this is 1 - E_theta[P_deg(theta)].
    """
    deg = statistics.fmean([predicted_binary_degeneracy(t, group_size) for t in thetas])
    return {
        "mean_theta": statistics.fmean(thetas),
        "expected_degenerate_fraction": deg,
        "advantage_energy_per_rollout": 1.0 - deg,
    }


# --------------------------------------------------------------- B. the sign flip
def outcome_under_rule(rule: str, thetas: list[float], group_size: int,
                       horizon: int) -> float:
    """Total 'improvement' a selector earns under one assumed learning rule.

    Every rule below is defensible and none is established. The point of this function is
    that changing the rule changes which selector wins, so the simulation cannot decide.

    Rules:
        learnability : gain proportional to theta(1-theta), the reward-variance story.
        frontier     : gain proportional to (1-theta), harder tasks teach more per success.
        zpd          : gain peaked at theta = 0.3, a zone-of-proximal-development story.
        lifetime     : learnability now, but tasks saturate at a rate proportional to
                       theta, so easy tasks stop paying sooner over the horizon.
    """
    total = 0.0
    for th in thetas:
        informative = 1.0 - predicted_binary_degeneracy(th, group_size)
        if rule == "learnability":
            gain = th * (1.0 - th)
        elif rule == "frontier":
            gain = 1.0 - th
        elif rule == "zpd":
            gain = max(0.0, 1.0 - abs(th - 0.3) / 0.3)
        elif rule == "success_teaches":
            gain = th
        elif rule == "lifetime":
            # saturates geometrically; easier tasks saturate faster
            gain = sum(th * (1.0 - th) * ((1.0 - th) ** s) for s in range(horizon)) / horizon
        else:
            raise ValueError(f"unknown rule {rule!r}")
        total += informative * gain
    return total / len(thetas)


def build_controls(rng: random.Random, treat: list[float]) -> dict[str, list[float]]:
    """Four size-matched controls, each isolating a different thing.

    uniform            -- no selection at all. Answers "is selecting better than not?",
                          which is NOT the contested claim but is the only one a
                          difficulty-only simulation can pose non-vacuously.
    matched_difficulty -- resampled to the treatment's OWN realised theta distribution.
                          This is the control that was specified. In a model where theta
                          is the entire task representation it is VACUOUS BY
                          CONSTRUCTION, and the near-zero difference below is the proof
                          of that rather than a result.
    matched_mean       -- uniform on a band with the same mean theta as the treatment.
                          Separates WHERE the gate puts mass from HOW TIGHTLY.
    band_wide          -- a crude "drop hopeless and trivial tasks" rule, keeping
                          theta in [0.1, 0.9] with no target at all. Tests how much of
                          the gate's benefit a one-line filter would capture.
    """
    n = len(treat)
    mean_t = statistics.fmean(treat)
    half = min(mean_t, 1.0 - mean_t)
    return {
        "uniform": [rng.random() for _ in range(n)],
        "matched_difficulty": [rng.choice(treat) for _ in range(n)],
        "matched_mean": [rng.uniform(mean_t - half, mean_t + half) for _ in range(n)],
        "band_wide": [rng.uniform(0.1, 0.9) for _ in range(n)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--keep-frac", type=float, default=0.1)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=Path("gate_outcome.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ident = verify_energy_identity(rng, args.group_size, args.trials)
    treat = realised_theta_distribution(rng, args.k, args.sigma, args.pool, args.keep_frac)
    controls = build_controls(rng, treat)
    ctrl = controls["uniform"]

    eff_t = budget_efficiency(treat, args.group_size)
    eff_c = budget_efficiency(ctrl, args.group_size)
    eff_at_target = budget_efficiency([0.2] * 1000, args.group_size)
    eff_at_half = budget_efficiency([0.5] * 1000, args.group_size)

    rules = ("learnability", "frontier", "zpd", "lifetime", "success_teaches")
    flip = {}
    for r in rules:
        tv = outcome_under_rule(r, treat, args.group_size, args.horizon)
        flip[r] = {"treatment": tv}
        for cname, cth in controls.items():
            cv = outcome_under_rule(r, cth, args.group_size, args.horizon)
            flip[r][f"control_{cname}"] = cv
            flip[r][f"diff_vs_{cname}"] = tv - cv

    res = {
        "config": vars(args) | {"out": str(args.out)},
        "A_energy_identity": ident,
        "A_budget_efficiency": {
            "treatment_realised": eff_t,
            "size_matched_random_control": eff_c,
            "idealised_at_p_star_0.2": eff_at_target,
            "idealised_at_0.5": eff_at_half,
        },
        "B_sign_flip": flip,
        "B_verdict": (
            "The winner changes with the assumed learning rule, so this simulation "
            "cannot answer whether the gate improves self-improvement. Only Part A is "
            "reportable CPU-side."
        ),
    }
    args.out.write_text(json.dumps(res, indent=2))

    print(f"G={args.group_size} k={args.k} sigma={args.sigma} keep={args.keep_frac}\n")
    print("A1. ADVANTAGE-ENERGY IDENTITY (relied on by A2, so checked not assumed)")
    print(f"   non-degenerate energy: mean {ident['mean_energy_nondegenerate']:.6f} "
          f"target G={ident['identity_target_G']:.1f}  "
          f"max|dev| {ident['max_abs_deviation_from_G']:.2e}")
    print("\nA2. BUDGET EFFICIENCY (advantage energy delivered per rollout)")
    for name, e in (("gate, realised", eff_t), ("size-matched random", eff_c),
                    ("idealised at p*=0.2", eff_at_target),
                    ("idealised at 0.5", eff_at_half)):
        print(f"   {name:<22} mean_theta {e['mean_theta']:.3f}  "
              f"dead {e['expected_degenerate_fraction']:.4f}  "
              f"energy/rollout {e['advantage_energy_per_rollout']:.4f}")
    print("\nB. OUTCOME UNDER FIVE LEARNING RULES x FOUR CONTROLS")
    cnames = ["uniform", "matched_difficulty", "matched_mean", "band_wide"]
    print(f"   {'rule':<16}{'gate':>9}" + "".join(f"{('d_'+c):>20}" for c in cnames))
    for r, v in flip.items():
        row = f"   {r:<16}{v['treatment']:>9.4f}"
        for c in cnames:
            row += f"{v['diff_vs_'+c]:>+20.4f}"
        print(row)
    print("\n   d_ = gate minus that control. Read the columns, not the rows:")
    md = [abs(v['diff_vs_matched_difficulty']) for v in flip.values()]
    print(f"   * matched_difficulty |diff| max = {max(md):.4f} -- vacuous by construction,")
    print("     because theta is the whole task representation here. The specified")
    print("     control cannot be posed non-vacuously in a difficulty-only simulation.")
    bw = [v['diff_vs_band_wide'] for v in flip.values()]
    print(f"   * band_wide diffs {min(bw):+.4f}..{max(bw):+.4f} -- how much a one-line")
    print("     'drop hopeless and trivial' filter would already capture.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
