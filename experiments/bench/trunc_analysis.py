#!/usr/bin/env python3
"""How much can the AIME score move if the token cap is raised?

A truncated generation is graded as wrong, but a model that ran out of budget has not
answered incorrectly -- it has not answered. This bounds the resulting uncertainty from the
persisted generations, so the bound is measured rather than asserted.
"""
import collections
import json
import sys
import os

# Suites live under $HOME by default; MATH_RUNS repoints them without editing
# the historical suite names below, which record which run produced which number.
RUNS_ROOT = os.environ.get("MATH_RUNS", os.path.expanduser("~/runs/math"))


path = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.join(RUNS_ROOT, "qwen38_27b_v2/generations.jsonl")
)
recs = [json.loads(l) for l in open(path)]

by = collections.defaultdict(list)
for r in recs:
    by[r["benchmark"]].append(r)

print(f"{'bench':<10} {'n':>3} {'correct':>7} {'trunc':>6} {'trunc+box':>9} "
      f"{'trunc_correct':>13} {'acc':>6} {'upper':>6}")
for b, rs in sorted(by.items()):
    n = len(rs)
    corr = sum(1 for r in rs if r["correct"])
    tr = [r for r in rs if r["finish_reason"] == "length"]
    tr_box = [r for r in tr if r["boxed"]]
    tr_corr = sum(1 for r in tr if r["correct"])
    acc = corr / n
    # Upper bound: every truncated-and-not-already-correct item could go either way.
    upper = (corr + len(tr) - tr_corr) / n
    print(f"{b:<10} {n:>3} {corr:>7} {len(tr):>6} {len(tr_box):>9} {tr_corr:>13} "
          f"{acc:>6.3f} {upper:>6.3f}")

print()
print("trunc+box  = truncated generations that nonetheless contained a complete \\boxed{}")
print("upper      = score if EVERY truncated item were actually correct (a hard ceiling,")
print("             not an estimate). The real value lies in [acc, upper].")
print()
# Length distribution of truncated vs completed, to show truncation is not random.
for b, rs in sorted(by.items()):
    done = [len(r["text"]) for r in rs if r["finish_reason"] != "length"]
    cut = [len(r["text"]) for r in rs if r["finish_reason"] == "length"]
    if done and cut:
        print(f"{b}: median chars completed={sorted(done)[len(done)//2]}, "
              f"truncated={sorted(cut)[len(cut)//2]}")
