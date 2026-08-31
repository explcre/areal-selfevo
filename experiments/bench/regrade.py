#!/usr/bin/env python3
"""Regrade the persisted sweep with the fixed extract_boxed, and test it correctly.

Two changes from the first pass:

  - extract_boxed now falls back to the last BALANCED box. The old rule cost the degraded
    checkpoints 4.2-4.6% and the base model 0.0%, so it biased the exact comparison it was
    being used for.
  - Significance is a PAIRED McNemar test on the same 500 problems, not two independent
    Wilson intervals. Wilson intervals ignore the pairing and overlap for base vs the early
    checkpoints, understating a difference that the paired test resolves cleanly.

No GPU is needed: this regrades persisted text, it does not regenerate.
"""
from __future__ import annotations
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from math_bench import grade  # the repo's own grader, not a reimplementation

SUITE = pathlib.Path(sys.argv[1])
TAGS = ["base", "gs028", "gs057", "gs086", "gs115", "gs144", "gs173"]
ENT = {"base": None, "gs028": 0.2533, "gs057": 0.1344, "gs086": 0.1436,
       "gs115": 0.0989, "gs144": 0.0253, "gs173": 0.0182}

def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    z = 1.959963985; p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

def norm_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))

def mcnemar(a: dict, b: dict):
    """Paired test on the same problems. b01 = base right & other wrong, b10 = reverse."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] and not b[k])
    b10 = sum(1 for k in keys if b[k] and not a[k])
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    # Normal approximation with continuity correction; exact binomial for small n.
    if n < 25:
        p = 2*sum(math.comb(n, i) for i in range(0, min(b01, b10)+1)) / (2**n)
        return b01, b10, min(1.0, p)
    z = (abs(b01-b10) - 1) / math.sqrt(n)
    return b01, b10, 2*(1-norm_cdf(z))

series = {}
print(f"{'ckpt':6} {'ent':>7} | {'old':>6} {'NEW':>6} {'delta':>6} {'95% CI':>15} | "
      f"{'flips':>5} {'trunc':>6} {'tr_ok':>6} | {'McNemar p vs base':>18}")
print("-"*108)
for tag in TAGS:
    f = SUITE / tag / "generations.jsonl"
    if not f.exists():
        print(f"{tag:6} (missing)"); continue
    rows = [json.loads(l) for l in f.open() if l.strip()]
    old = sum(1 for r in rows if r.get("correct"))
    new_by_idx, flips, trunc, tr_ok = {}, 0, 0, 0
    for r in rows:
        ok = grade(r["text"], r["gold"])
        new_by_idx[r["idx"]] = ok
        if ok and not r.get("correct"): flips += 1
        if r.get("finish_reason") == "length":
            trunc += 1
            if ok: tr_ok += 1
    series[tag] = new_by_idx
    n = len(rows); k = sum(new_by_idx.values())
    lo, hi = wilson(k, n)
    if tag == "base":
        pstr = "--"
    else:
        b01, b10, p = mcnemar(series["base"], new_by_idx)
        pstr = f"{p:.2e} ({b01}/{b10})"
    e = ENT[tag]
    print(f"{tag:6} {e if e is not None else '-':>7} | {old/n:6.3f} {k/n:6.3f} "
          f"{(k-old)/n:+6.3f} [{lo:.3f},{hi:.3f}] | {flips:5d} {trunc:6d} {tr_ok:6d} | {pstr:>18}")

print("\nflips = items the fixed grader recovers. tr_ok = truncated items graded correct.")
print("McNemar (b01/b10) = base-right-other-wrong / other-right-base-wrong, same 500 problems.")
