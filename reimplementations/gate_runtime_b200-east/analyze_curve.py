#!/usr/bin/env python3
"""The training-side curve for each arm, with intervals, on a scale the arms share.

Raw training reward is NOT comparable across these arms and the pre-registration says so: each
arm draws a different difficulty mixture, so the gate's arm sits lower than the band filter's
from the first step for reasons that have nothing to do with learning.

What IS comparable is the LIFT: the arm's measured success rate on the rollouts it drew, minus
the base model's own measured success rate on those same problems, taken from the independent
fresh block of the discovery run. Under the null "nothing moved" the lift is zero for every arm
at every step, whatever it selects. Its slope over steps is the learning signal.

Reported with a standard error on the slope, and with the k-histogram beside it, because a
batch whose groups are mostly unanimous gives a false negative on any gradient statistic.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics


def base_truncation(graded_path: str) -> dict[int, float]:
    """Per-problem truncation rate of the BASE model, from the discovery samples.

    The arms draw different problems at every step and the hard ones truncate far more, so a
    raw truncation curve mixes "the policy learned to finish" with "this step happened to draw
    easier problems". Subtracting each drawn problem's own base rate removes the second.
    """
    num, den = {}, {}
    for line in open(graded_path):
        r = json.loads(line)
        if r.get("error"):
            continue
        num[r["idx"]] = num.get(r["idx"], 0) + int(bool(r.get("truncated")))
        den[r["idx"]] = den.get(r["idx"], 0) + 1
    return {i: num[i] / den[i] for i in den if den[i]}


def load_arm(path: str, pool: dict, btr: dict) -> list[dict]:
    """Read one arm's step log and attach the base-model references for the tasks it drew."""
    pb = {t["idx"]: t["c_b"] / t["n_b"] for t in pool["tasks"]}
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("rollout_accuracy") is None:
            continue
        ref = statistics.fmean([pb[i] for i in r["tasks"] if i in pb]) if r["tasks"] else None
        if ref is None:
            continue
        r["baseline_p"] = ref
        r["lift"] = r["rollout_accuracy"] - ref
        seen = [btr[i] for i in r["tasks"] if i in btr]
        if seen and r.get("truncation_rate") is not None:
            r["baseline_trunc"] = statistics.fmean(seen)
            r["trunc_lift"] = r["truncation_rate"] - r["baseline_trunc"]
        rows.append(r)
    return rows


def ols_slope(xs, ys):
    """Least-squares slope of y on x with its standard error, per 100 steps."""
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return b * 100, math.sqrt(s2 / sxx) * 100


def main() -> int:
    """Print, per arm, the lift curve in windows plus the fitted slope and its interval."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=steps.jsonl")
    ap.add_argument("--graded", default="/mnt/localssd/gate/out/"
                    "discover_search_k16.jsonl.graded.jsonl.cap8192.jsonl",
                    help="the graded discovery file at the TRAINING cap, for base truncation")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--token-match", action="store_true",
                    help="truncate every arm at the smallest generated-token total any arm "
                         "reached, which is pre-registration matching rule 3: the arms differ "
                         "in generation cost, so equal step counts would hand one arm more "
                         "compute. Applied AFTER the run rather than as a pre-set budget, "
                         "because the per-arm token rate is not predictable in advance.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    btr = base_truncation(a.graded)
    loaded = {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        loaded[name] = load_arm(path, pool, btr)
    budget = None
    if a.token_match:
        budget = min(r[-1]["gen_tokens_cum"] for r in loaded.values() if r)
        loaded = {n: [r for r in rows if r["gen_tokens_cum"] <= budget]
                  for n, rows in loaded.items()}
    out = {"pool_size": len(pool["tasks"]), "token_matched_at": budget, "arms": {}}
    for name, rows in loaded.items():
        if not rows:
            out["arms"][name] = {"steps": 0}
            continue
        wins = []
        for i in range(0, len(rows), a.window):
            chunk = rows[i: i + a.window]
            lifts = [r["lift"] for r in chunk]
            m = statistics.fmean(lifts)
            se = (statistics.stdev(lifts) / math.sqrt(len(lifts))) if len(lifts) > 1 else None
            wins.append({"step_lo": chunk[0]["step"], "step_hi": chunk[-1]["step"],
                         "n": len(chunk), "lift": round(m, 4),
                         "se": round(se, 4) if se else None,
                         "acc": round(statistics.fmean([r["rollout_accuracy"] for r in chunk]), 4),
                         "baseline_p": round(statistics.fmean([r["baseline_p"] for r in chunk]), 4),
                         "informative": round(statistics.fmean(
                             [r["informative_group_fraction"] for r in chunk]), 3),
                         "trunc": round(statistics.fmean(
                             [r["truncation_rate"] for r in chunk
                              if r.get("truncation_rate") is not None]), 3)
                         if any(r.get("truncation_rate") is not None for r in chunk) else None,
                         "trunc_lift": round(statistics.fmean(
                             [r["trunc_lift"] for r in chunk if "trunc_lift" in r]), 3)
                         if any("trunc_lift" in r for r in chunk) else None})
        b, sb = ols_slope([r["step"] for r in rows], [r["lift"] for r in rows])
        tb, tsb = ols_slope([r["step"] for r in rows if "trunc_lift" in r],
                            [r["trunc_lift"] for r in rows if "trunc_lift" in r])
        khist: dict[str, int] = {}
        for r in rows:
            for k, v in (r.get("k_histogram") or {}).items():
                khist[str(k)] = khist.get(str(k), 0) + v
        first = [r["lift"] for r in rows[: a.window]]
        last = [r["lift"] for r in rows[-a.window:]]
        d = statistics.fmean(last) - statistics.fmean(first)
        sed = math.sqrt(
            (statistics.stdev(last) ** 2 if len(last) > 1 else 0) / max(len(last), 1)
            + (statistics.stdev(first) ** 2 if len(first) > 1 else 0) / max(len(first), 1))
        out["arms"][name] = {
            "steps": len(rows),
            "rollouts": sum(r["groups"] * 0 + r["trainable_rollouts"] for r in rows),
            "gen_tokens": rows[-1]["gen_tokens_cum"],
            "mean_baseline_p": round(statistics.fmean([r["baseline_p"] for r in rows]), 4),
            "mean_informative": round(statistics.fmean(
                [r["informative_group_fraction"] for r in rows]), 4),
            "k_histogram": khist,
            "lift_slope_per_100_steps": round(b, 5) if b is not None else None,
            "lift_slope_se": round(sb, 5) if sb is not None else None,
            "trunc_lift_slope_per_100_steps": round(tb, 5) if tb is not None else None,
            "trunc_lift_slope_se": round(tsb, 5) if tsb is not None else None,
            "last_minus_first_window": round(d, 4),
            "se_of_that_difference": round(sed, 4),
            "windows": wins,
        }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
