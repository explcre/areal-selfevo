#!/usr/bin/env python3
"""Three arms at matched steps on full MATH-500, paired McNemar.

  step0d  demo recipe            lr 6e-6, clip 0.4, kl 0.01, adv_norm batch, routing off
  step0h  scaffold recipe        lr 1e-6, clip 0.2, kl 0.01, adv_norm batch, routing off
  step0j  scaffold + routing     lr 1e-6, clip 0.2, kl 0,    adv_norm off,   routing ON

CONFOUND, stated up front: step0j differs from step0h in THREE ways, not one -- routing,
kl_ctl and adv_norm. Those three had to move together because the routing rule's precondition
does not hold otherwise, but it means a step0h-vs-step0j difference cannot be attributed to
routing alone. The clean control is a routing-OFF arm at step0j's exact config, which does
not exist yet.
"""
from __future__ import annotations
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from math_bench import grade

# Suites live under $HOME by default; MATH_RUNS repoints them without editing
# the historical suite names below, which record which run produced which number.
RUNS_ROOT = os.environ.get("MATH_RUNS", os.path.expanduser("~/runs/math"))


ARMS = {
    "demo (0d)":       pathlib.Path(RUNS_ROOT) / "sweep_0829_1628",
    "scaffold (0h)":   None,   # merged below
    "scaf+route (0j)": pathlib.Path(RUNS_ROOT) / "sweep_0830_1905",
}
H_R = pathlib.Path(RUNS_ROOT) / "sweep_0830_0557"
H_S = pathlib.Path(RUNS_ROOT) / "sweep_0830_1018"
STEPS = [28, 57, 86, 115, 144]


def pp(suite, tag):
    f = suite / tag / "generations.jsonl"
    if not f.exists():
        return {}
    return {r["idx"]: grade(r["text"], r["gold"])
            for r in (json.loads(l) for l in f.open() if l.strip())}


def arm(name, tag):
    if name == "scaffold (0h)":
        a, b = pp(H_R, tag), pp(H_S, tag)
        return {**a, **b}
    return pp(ARMS[name], tag)


def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar(a, b):
    k = sorted(set(a) & set(b))
    b01 = sum(1 for i in k if a[i] and not b[i])
    b10 = sum(1 for i in k if b[i] and not a[i])
    n = b01 + b10
    if n == 0: return 1.0, len(k)
    if n < 25:
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n), len(k)
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    return 2 * (1 - norm_cdf(z)), len(k)


names = list(ARMS)
print(f"{'step':>5} | " + " | ".join(f"{n:>15}" for n in names) + " |  0h vs 0j p")
print("-" * 84)
for tag, label in [("base", "base")] + [(f"gs{s:03d}", str(s)) for s in STEPS]:
    vals, res = {}, []
    for n in names:
        d = arm(n, tag)
        vals[n] = d
        res.append(f"{(sum(d.values())/len(d) if d else float('nan')):15.3f}")
    if vals["scaffold (0h)"] and vals["scaf+route (0j)"]:
        p, n_ = mcnemar(vals["scaffold (0h)"], vals["scaf+route (0j)"])
        pstr = f"{p:11.3f}"
    else:
        pstr = f"{'--':>11}"
    print(f"{label:>5} | " + " | ".join(res) + " |" + pstr)
