#!/usr/bin/env python3
"""Reconcile two readings of the same graded files, and test the three reported problems.

The coordinator read `correct` straight off the rows and got numbers ~5 points higher than the
analysis script on every arm but one. That is not a discrepancy to average over: the two
readings differ by exactly one decision -- what to do with a generation that hit the token cap
but had already emitted a \\boxed{} answer -- and the arms differ enormously in how often that
happens (base truncates 41% of the time, C1 0.07%), so the decision is worth up to six points
and is worth MORE to the arms that truncate more. Both are computed here side by side.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict

OUT = "/mnt/localssd/gate/out"


def load(path):
    """Rows of one graded eval file."""
    return [json.loads(l) for l in open(path)]


def score(rows, convention):
    """Per-item rate under a stated convention.

    * ``graded``   -- use the grader's verdict as it stands. A generation that hit the cap but
      had already emitted a balanced box counts as correct. This is what ``math_bench.grade``
      returns and what the rest of this paper uses; results.tex already records the audit that
      put it there, having measured that discarding those flips 0 items for the base model and
      21/23 for the two most degraded checkpoints, i.e. it penalises exactly the checkpoints
      under argument.
    * ``cap_zero`` -- force any capped generation to 0 regardless of what it contains.
    """
    num, den = defaultdict(int), defaultdict(int)
    for r in rows:
        if r.get("error"):
            continue
        ok = int(bool(r["correct"]))
        if convention == "cap_zero" and r.get("truncated"):
            ok = 0
        num[r["idx"]] += ok
        den[r["idx"]] += 1
    return {i: num[i] / den[i] for i in den if den[i]}


def paired(a, b):
    """Mean paired difference a-b with its standard error, over the items both resolved."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    m = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
    return m, se, n


def truncation(rows):
    """Fraction of generations that hit the cap, and of those how many still carried a box."""
    tr = [r for r in rows if r.get("truncated")]
    return (len(tr) / len(rows),
            sum(int(bool(r["correct"])) for r in tr) / max(len(tr), 1))


FILES = {
    ("base", "both"): "eval_final_base.jsonl.graded.jsonl",
    ("T", "matched"): "eval_matched_T.jsonl.graded.jsonl",
    ("T", "final"): "eval_final_T.jsonl.graded.jsonl",
    ("C1", "matched"): "eval_matched_C1.jsonl.graded.jsonl",
    ("C1", "final"): "eval_final_C1.jsonl.graded.jsonl",
    ("C2", "matched"): "eval_matched_C2.jsonl.graded.jsonl",
    ("C2", "final"): "eval_final_C2.jsonl.graded.jsonl",
    ("C3", "matched"): "eval_matched_C3.jsonl.graded.jsonl",
    ("C3", "final"): "eval_final_C3.jsonl.graded.jsonl",
}

rows = {k: load("%s/%s" % (OUT, v)) for k, v in FILES.items()}
base = rows[("base", "both")]

print("=" * 100)
print("1. RECONCILIATION: the same files under the two conventions")
print("=" * 100)
print("%-6s %-8s %8s %9s %9s   %7s %8s" %
      ("arm", "eval", "graded", "cap_zero", "gap", "trunc", "boxed|trunc"))
sg, sz = {}, {}
for (arm, ev), rs in rows.items():
    g, z = score(rs, "graded"), score(rs, "cap_zero")
    sg[(arm, ev)], sz[(arm, ev)] = g, z
    mg, mz = statistics.fmean(g.values()), statistics.fmean(z.values())
    t, bt = truncation(rs)
    print("%-6s %-8s %8.4f %9.4f %9.4f   %7.3f %8.3f"
          % (arm, ev, mg, mz, mg - mz, t, bt))
print("\nThe gap column IS the disagreement: it is the share of generations that were cut off")
print("after committing an answer, and it tracks the truncation column exactly.")

print()
print("=" * 100)
print("2. EVERY ARM AGAINST BASE, both conventions, paired per item")
print("=" * 100)
for conv, sc in (("graded", sg), ("cap_zero", sz)):
    b = sc[("base", "both")]
    print("-- convention: %s" % conv)
    for ev in ("matched", "final"):
        for arm in ("T", "C1", "C2", "C3"):
            m, se, n = paired(sc[(arm, ev)], b)
            star = "  <-- crosses zero" if abs(m) < 1.96 * se else ""
            print("   %-8s %-3s vs base  %+0.4f  se %.4f  95%% CI [%+0.4f,%+0.4f]  n=%d%s"
                  % (ev, arm, m, se, m - 1.96 * se, m + 1.96 * se, n, star))
