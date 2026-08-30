#!/usr/bin/env python3
"""Restate the demo-vs-scaffold A/B at the FULL 500 problems.

step0d was scored on all 500 in one sweep. step0h was scored in two halves, so its
generations are merged here. The merge is checked, not assumed: the two halves must be
disjoint by source index and must together cover exactly the 500 rows.
"""
from __future__ import annotations
import json, math, pathlib, sys

sys.path.insert(0, "/home/ubuntu/areal-selfevo/experiments/bench")
from math_bench import grade

D_FULL = pathlib.Path("/home/ubuntu/runs/math/sweep_0829_1628")          # step0d, all 500
H_REPORT = pathlib.Path("/home/ubuntu/runs/math/sweep_0830_0557")        # step0h, report
H_SEARCH = pathlib.Path("/home/ubuntu/runs/math/sweep_0830_1018")        # step0h, search
STEPS = [28, 57, 86, 115, 144]


def per_problem(suite: pathlib.Path, tag: str) -> dict[int, bool]:
    f = suite / tag / "generations.jsonl"
    if not f.exists():
        return {}
    return {r["idx"]: grade(r["text"], r["gold"])
            for r in (json.loads(l) for l in f.open() if l.strip())}


def merged(tag: str) -> dict[int, bool]:
    a, b = per_problem(H_REPORT, tag), per_problem(H_SEARCH, tag)
    overlap = set(a) & set(b)
    if overlap:
        sys.exit(f"{tag}: halves overlap on {len(overlap)} problems -- refusing to merge")
    out = {**a, **b}
    if len(out) != 500:
        sys.exit(f"{tag}: merged to {len(out)} problems, expected 500")
    return out


def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar(a, b):
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] and not b[k])
    b10 = sum(1 for k in keys if b[k] and not a[k])
    n = b01 + b10
    if n == 0: return b01, b10, 1.0, len(keys)
    if n < 25:
        p = 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n
        return b01, b10, min(1.0, p), len(keys)
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    return b01, b10, 2 * (1 - norm_cdf(z)), len(keys)


print("FULL MATH-500 (n=500), paired McNemar on the same problems\n")
print(f"{'step':>5} | {'demo 6e-6/0.4':>14} | {'scaffold 1e-6/0.2':>18} | {'diff':>7} | {'p':>10} | {'n':>4}")
print("-" * 78)
db, hb = per_problem(D_FULL, "base"), merged("base")
print(f"{'base':>5} | {sum(db.values())/len(db):14.3f} | {sum(hb.values())/len(hb):18.3f} |"
      f" {sum(hb.values())/len(hb) - sum(db.values())/len(db):+7.3f} |"
      f" {'(same model)':>10} | {len(db):4d}")
rows = []
for s in STEPS:
    tag = f"gs{s:03d}"
    d, h = per_problem(D_FULL, tag), merged(tag)
    if not d or not h:
        print(f"{s:>5} | (missing)"); continue
    pd_, ph = sum(d.values()) / len(d), sum(h.values()) / len(h)
    b01, b10, p, n = mcnemar(d, h)
    print(f"{s:>5} | {pd_:14.3f} | {ph:18.3f} | {ph-pd_:+7.3f} | {p:10.2e} | {n:4d}")
    rows.append((s, pd_, ph, p))
if rows:
    print(f"\nmean over {len(rows)} paired steps: demo {sum(r[1] for r in rows)/len(rows):.3f} "
          f"vs scaffold {sum(r[2] for r in rows)/len(rows):.3f}")
    print(f"scaffold higher at {sum(1 for r in rows if r[2] > r[1])}/{len(rows)} steps; "
          f"p<0.05 at {sum(1 for r in rows if r[3] < 0.05)}/{len(rows)}")
    print(f"\nbase difference (same model, independent sweeps) = "
          f"{abs(sum(hb.values())/len(hb) - sum(db.values())/len(db)):.4f}  <- run-to-run noise")
