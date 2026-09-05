#!/usr/bin/env python3
"""Harness-on against harness-off: paired per-iteration differences, with SE on the difference.

The two arms are matched by construction -- same seed, same task draws, same generated-token
budget, same stages running and being scored -- and differ only in whether the harness stage
takes a gradient step. So the comparison is PAIRED by iteration, and the quantity reported is
the mean of the per-iteration differences with the standard error of that difference, not two
independent means with two independent errors. This project has twice mistaken a signal for an
outcome, so the effect is reported with the interval that would contain a null.
"""
from __future__ import annotations

import argparse
import json
import math

#: Process-level outcomes. Each is a per-iteration number; None means the iteration did not
#: form a task group and is excluded pairwise.
METRICS = {
    "informative_rollout_groups": "share of rollout groups carrying gradient",
    "informative_scaffold_groups": "share of scaffold groups carrying gradient",
    "mean_p_hat": "mean success rate of the scored tasks",
    "tasks_scored": "tasks that survived scoring (out of 4)",
}


def load(path):
    """Iteration records keyed by iteration number."""
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["iter"]] = r
    return out


def paired(a, b, key):
    """Per-iteration (on, off) pairs where both arms produced the metric."""
    pairs = []
    for it in sorted(set(a) & set(b)):
        x, y = a[it].get(key), b[it].get(key)
        if x is not None and y is not None:
            pairs.append((it, float(x), float(y)))
    return pairs


def stats(pairs):
    """Mean paired difference (on minus off), its SE, and a 95% interval."""
    d = [x - y for _, x, y in pairs]
    n = len(d)
    if n < 2:
        return {"n": n, "mean_diff": (d[0] if n else None), "se": None, "ci95": None}
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean_diff": round(m, 4), "se": round(se, 4),
            "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
            "mean_on": round(sum(x for _, x, _ in pairs) / n, 4),
            "mean_off": round(sum(y for _, _, y in pairs) / n, 4)}


def main() -> int:
    """Print the paired comparison and the harness stage's own training record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default="/mnt/localssd/gate/runs/HON/iters.jsonl")
    ap.add_argument("--off", default="/mnt/localssd/gate/runs/HOFF/iters.jsonl")
    ap.add_argument("--out", default="/mnt/localssd/gate/out/harness_ab.json")
    a = ap.parse_args()
    A, B = load(a.on), load(a.off)
    common = sorted(set(A) & set(B))
    print("iterations: on %d, off %d, paired %d" % (len(A), len(B), len(common)))

    upd_on = [r for r in A.values() if (r.get("updates") or {}).get("harness", {}).get("rows")]
    avail_off = [r for r in B.values()
                 if (r.get("updates") or {}).get("harness", {}).get("rows_available")]
    print("harness gradient steps actually taken: on %d, off %d (off had rows available in "
          "%d iterations and declined to step)"
          % (len(upd_on),
             len([r for r in B.values()
                  if (r.get("updates") or {}).get("harness", {}).get("rows")]),
             len(avail_off)))

    grouped_on = sum(1 for r in A.values() if (r.get("tasks_scored") or 0) >= 2)
    grouped_off = sum(1 for r in B.values() if (r.get("tasks_scored") or 0) >= 2)
    print("iterations forming a task group: on %d/%d, off %d/%d"
          % (grouped_on, len(A), grouped_off, len(B)))

    res = {"n_paired": len(common), "harness_steps_on": len(upd_on),
           "grouped_on": grouped_on, "grouped_off": grouped_off,
           "n_iters_on": len(A), "n_iters_off": len(B), "metrics": {}}
    print("\n%-32s %5s %9s %9s %9s %s" % ("metric", "n", "on", "off", "diff", "95% CI on diff"))
    for key, desc in METRICS.items():
        st = stats(paired(A, B, key))
        res["metrics"][key] = st
        if st["n"] < 2:
            print("%-32s %5d  (too few paired iterations)" % (key, st["n"]))
            continue
        print("%-32s %5d %9.4f %9.4f %9.4f  [%.4f, %.4f]"
              % (key, st["n"], st["mean_on"], st["mean_off"], st["mean_diff"],
                 st["ci95"][0], st["ci95"][1]))

    print("\nreading: an interval containing 0 is consistent with the harness stage's gradient "
          "changing nothing about the process the loop runs.")
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
