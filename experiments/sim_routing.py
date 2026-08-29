#!/usr/bin/env python3
"""Can criterion routing beat a matched shuffle of its own decisions?

This is the cheapest experiment that can kill the idea, so it runs before any GPU time.

**How circularity is avoided.** The simulator does not implement the routing criterion. It
implements GRPO's update rule, and the property the criterion exploits falls out of that
rule rather than being asserted: rewards are drawn Bernoulli(p), the centred advantage
``A_i = r_i - rbar`` is formed explicitly, and when every ``r_i`` is equal every ``A_i`` is
zero, so the update is a no-op. That is arithmetic, not a modelling choice. The criterion
is then judged on whether it allocates a fixed compute budget better than a control that
spends the same budget on the same mode mixture.

**How it can fail.** Three ways, all reported:

1. If the criterion does not beat :class:`MatchedPermutationControl`, the gain came from
   the mode mixture rather than from choosing which unit gets which mode. The whole design
   is then dead, and this is the number that says so.
2. ``all_informative`` is a regime where the criterion should NOT help, because no unit is
   ever silent and there is nothing to detect. A criterion that "wins" there is winning by
   construction, not by information.
3. ``inverted`` should be *worse* than the control. If it merely ties, the criterion
   carries no signal.

Reported effects are paired differences across seeds with a standard error on the
*difference*, not two independent means with overlapping error bars.
"""
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, replace

from selfevo.routing.base import Granularity, RoutingContext, TrainingMode
from selfevo.routing.proportions import MatchedPermutationControl
from selfevo.routing.routers import InvertedRouter, SolveRateRouter, StaticRouter


@dataclass
class Unit:
    """One prompt (or cluster) with a latent solve rate."""

    p: float
    teacher_p: float
    has_teacher: bool


def _grpo_update(p: float, group_size: int, lr: float, rng: random.Random) -> tuple[float, bool]:
    """One GRPO step. Returns (new p, whether any gradient flowed).

    The advantage is formed explicitly rather than assumed, so the zero-gradient case is a
    consequence of the arithmetic and not of the criterion under test.
    """
    rewards = [1.0 if rng.random() < p else 0.0 for _ in range(group_size)]
    mean_r = sum(rewards) / group_size
    advs = [r - mean_r for r in rewards]
    if all(a == 0.0 for a in advs):
        return p, False  # unanimous group: every A_i is exactly zero
    # Signal strength scales with the spread of the advantages, which is what a policy
    # gradient actually sees. Improvement is capped so p stays a probability.
    spread = statistics.pstdev(advs)
    return min(1.0, p + lr * spread), True


def _sft_update(u: Unit, lr: float) -> float:
    """Move p toward the teacher. Does nothing if the teacher is not better."""
    if not u.has_teacher or u.teacher_p <= u.p:
        return u.p
    return min(1.0, u.p + lr * (u.teacher_p - u.p))


COSTS = {TrainingMode.RL: 1.0, TrainingMode.SFT: 1.0, TrainingMode.DISTILL: 1.0,
         TrainingMode.SKIP: 0.0}


def run(
    router,
    units: list[Unit],
    *,
    budget: float,
    group_size: int,
    lr_rl: float,
    lr_sft: float,
    seed: int,
    control_from: list | None = None,
) -> tuple[float, dict[str, int]]:
    """Spend ``budget`` under ``router``; return (mean p, mode counts).

    Every arm gets the *same* budget, so an arm that skips cheap units can afford more
    updates elsewhere. That is the whole point of routing and it must be what is measured.
    """
    rng = random.Random(seed)
    us = [replace(u) for u in units]
    spent = 0.0
    counts: dict[str, int] = {}
    i = 0
    while spent < budget:
        u = us[i % len(us)]
        i += 1
        # p_hat is what a real rollout gives: an observed group mean, not the latent p.
        obs = [1.0 if rng.random() < u.p else 0.0 for _ in range(group_size)]
        ctx = RoutingContext(
            solve_rate=sum(obs) / group_size,
            group_size=group_size,
            granularity=Granularity.CLUSTER,
            has_teacher=u.has_teacher,
        )
        mode = router.route(ctx).argmax()
        counts[mode] = counts.get(mode, 0) + 1
        spent += COSTS[mode]
        if mode == TrainingMode.RL:
            u.p, _ = _grpo_update(u.p, group_size, lr_rl, rng)
        elif mode == TrainingMode.SFT:
            u.p = _sft_update(u, lr_sft)
        if spent >= budget:
            break
    return sum(x.p for x in us) / len(us), counts


def make_units(regime: str, n: int, rng: random.Random) -> list[Unit]:
    """Populations that make the criterion look good, neutral, and useless."""
    if regime == "mixed":
        # Realistic: some unlearnable-by-RL, some informative, some already solved.
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.33:
                p = rng.uniform(0.0, 0.05)       # RL silent, teacher can help
            elif r < 0.66:
                p = rng.uniform(0.3, 0.7)        # informative
            else:
                p = rng.uniform(0.95, 1.0)       # solved, RL silent, nothing to learn
            out.append(Unit(p=p, teacher_p=0.9, has_teacher=True))
        return out
    if regime == "all_informative":
        # The criterion should NOT help here: nothing is ever silent.
        return [Unit(p=rng.uniform(0.4, 0.6), teacher_p=0.9, has_teacher=True) for _ in range(n)]
    if regime == "no_teacher":
        # Unsolved units cannot be rescued; the criterion can only avoid wasting budget.
        out = []
        for _ in range(n):
            p = rng.uniform(0.0, 0.05) if rng.random() < 0.5 else rng.uniform(0.3, 0.7)
            out.append(Unit(p=p, teacher_p=0.9, has_teacher=False))
        return out
    raise ValueError(f"unknown regime {regime!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--units", type=int, default=60)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr-rl", type=float, default=0.05)
    ap.add_argument("--lr-sft", type=float, default=0.05)
    args = ap.parse_args()

    for regime in ("mixed", "all_informative", "no_teacher"):
        print(f"\n=== regime: {regime} ===")
        diffs: dict[str, list[float]] = {}
        props: dict[str, dict[str, int]] = {}
        for seed in range(args.seeds):
            rng = random.Random(10_000 + seed)
            units = make_units(regime, args.units, rng)
            crit = SolveRateRouter()
            ctxs = [
                RoutingContext(
                    solve_rate=u.p, group_size=args.group_size,
                    granularity=Granularity.CLUSTER, has_teacher=u.has_teacher,
                )
                for u in units
            ]
            arms = {
                "criterion": crit,
                "matched_control": MatchedPermutationControl.from_router(
                    # Pool sized to the budget: at most `budget` decisions are
                    # served (SKIP costs 0, so the loop can exceed budget in
                    # iterations), with headroom. Building 200x that was the
                    # bottleneck -- every RoutingDecision validates on construction.
                    crit, ctxs * (int(args.budget // len(ctxs)) + 3), seed=seed
                ),
                "inverted": InvertedRouter(),
                "all_rl": StaticRouter({TrainingMode.RL: 1.0}),
                "all_sft": StaticRouter({TrainingMode.SFT: 1.0}),
            }
            res = {}
            for name, r in arms.items():
                mean_p, counts = run(
                    r, units, budget=args.budget, group_size=args.group_size,
                    lr_rl=args.lr_rl, lr_sft=args.lr_sft, seed=seed,
                )
                res[name] = mean_p
                props.setdefault(name, {})
                for m, c in counts.items():
                    props[name][m] = props[name].get(m, 0) + c
            for name, v in res.items():
                if name != "matched_control":
                    diffs.setdefault(name, []).append(v - res["matched_control"])
            diffs.setdefault("matched_control", []).append(0.0)

        print(f"{'arm':<18} {'mean p - control':>18} {'se(diff)':>10} {'z':>7}  modes")
        for name, ds in diffs.items():
            m = statistics.mean(ds)
            if name == "matched_control":
                print(f"{'matched_control':<18} {'0 (reference)':>18} {'':>10} {'':>7}  "
                      f"{props[name]}")
                continue
            se = statistics.stdev(ds) / (len(ds) ** 0.5) if len(ds) > 1 else float("nan")
            z = m / se if se and se == se and se > 0 else float("nan")
            print(f"{name:<18} {m:+18.4f} {se:10.4f} {z:7.2f}  {props[name]}")


if __name__ == "__main__":
    main()
