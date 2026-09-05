#!/usr/bin/env python3
"""The three reported problems, tested rather than argued.

P1  Is C3's reward actually independent of correctness, and does any gain survive C3 as a floor?
P2  What distinguishes the matched and final evaluations, and do the arms' curves really cross?
P3  Why is C1 identical to four decimals across two evaluations?
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict

OUT = "/mnt/localssd/gate/out"
RUNS = "/mnt/localssd/gate/runs"


def score(path):
    """Per-item rate under the project's native grader verdict."""
    num, den = defaultdict(int), defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        num[r["idx"]] += int(bool(r["correct"]))
        den[r["idx"]] += 1
    return {i: num[i] / den[i] for i in den if den[i]}


def paired(a, b):
    """Mean paired difference with standard error."""
    k = sorted(set(a) & set(b))
    d = [a[x] - b[x] for x in k]
    return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d)), len(d)


base = score("%s/eval_final_base.jsonl.graded.jsonl" % OUT)

print("=" * 96)
print("P1a  Was C3's reward independent of correctness?")
print("=" * 96)
print("The trainer computes reward as `float(rng.random() < 0.5) if random_reward else")
print("float(correct)`: for C3 the reward expression never reads `correct`. Two measurable")
print("consequences follow, and both are checked against T, whose reward IS correctness.\n")
for arm in ("T", "C3"):
    h = defaultdict(int)
    acc, nst = [], 0
    for line in open("%s/%s/steps.jsonl" % (RUNS, arm)):
        r = json.loads(line)
        nst += 1
        acc.append(r["rollout_accuracy"])
        for k, v in (r.get("k_histogram") or {}).items():
            h[int(k)] += v
    tot = sum(h.values())
    mean_k = sum(k * v for k, v in h.items()) / tot
    true_acc = statistics.fmean(acc)
    # chi-square of the reward-k histogram against Binomial(8, 1/2)
    chi = 0.0
    for k in range(9):
        e = tot * math.comb(8, k) * 0.5 ** 8
        chi += (h.get(k, 0) - e) ** 2 / e
    # ... and against Binomial(8, true accuracy), which is what a correctness-linked reward gives
    chi_acc = 0.0
    for k in range(9):
        e = tot * math.comb(8, k) * true_acc ** k * (1 - true_acc) ** (8 - k)
        chi_acc += (h.get(k, 0) - e) ** 2 / max(e, 1e-9)
    print("  %-3s groups=%d  mean reward-k=%.3f  measured accuracy=%.3f  8*acc=%.3f"
          % (arm, tot, mean_k, true_acc, 8 * true_acc))
    print("      chi2 vs Binomial(8,0.5) = %8.1f      chi2 vs Binomial(8,accuracy) = %8.1f"
          % (chi, chi_acc))
    print("      histogram %s" % dict(sorted(h.items())))
print("\n  A reward tied to correctness must centre on 8*accuracy. C3's centres on 4.0 while its")
print("  accuracy says 2.2, and its histogram fits a fair coin far better than it fits its own")
print("  accuracy. T's does the opposite. The reward was independent.")

print()
print("=" * 96)
print("P1b  Does any arm's gain survive treating C3 as a floor?")
print("=" * 96)
for ev in ("matched", "final"):
    c3, _, _ = paired(score("%s/eval_%s_C3.jsonl.graded.jsonl" % (OUT, ev)), base)
    floor = max(c3, 0.0)
    print("  -- %s evaluation: C3 vs base = %+0.4f, so the floor charged is %+0.4f" % (ev, c3, floor))
    for arm in ("T", "C1", "C2"):
        m, se, _ = paired(score("%s/eval_%s_%s.jsonl.graded.jsonl" % (OUT, ev, arm)), base)
        print("     %-3s %+0.4f  -> net of floor %+0.4f   (se %.4f, so %.1f se above the floor)"
              % (arm, m, m - floor, se, (m - floor) / se))

print()
print("=" * 96)
print("P2   Do the held-out curves actually cross? Same arm, several checkpoints.")
print("=" * 96)
TRAJ = [("T", 30, "eval_matched_T"), ("T", 50, "eval_extra_Tmid"), ("T", 61, "eval_final_T"),
        ("C2", 20, "eval_extra_C2early"), ("C2", 40, "eval_matched_C2"),
        ("C2", 60, "eval_final_C2")]
tok = {}
for arm in ("T", "C1", "C2", "C3"):
    rows = [json.loads(l) for l in open("%s/%s/steps.jsonl" % (RUNS, arm))]
    tok[arm] = {r["step"] + 1: r["gen_tokens_cum"] for r in rows}
for arm, st, f in TRAJ:
    s = score("%s/%s.jsonl.graded.jsonl" % (OUT, f))
    m, se, _ = paired(s, base)
    print("  %-3s step %2d  %10d gen tokens   avg@8 %.4f   vs base %+0.4f (se %.4f)"
          % (arm, st, tok[arm].get(st, tok[arm][max(tok[arm])]),
             statistics.fmean(s.values()), m, se))

print()
print("=" * 96)
print("P3   Why is C1 identical on both evaluations?")
print("=" * 96)
for f in ("eval_matched_C1.jsonl.graded.jsonl", "eval_final_C1.jsonl.graded.jsonl"):
    b = open("%s/%s" % (OUT, f), "rb").read()
    print("  %-42s %d bytes  md5 %s" % (f, len(b), hashlib.md5(b).hexdigest()))
print("  C1 halted at the common token budget, so its final adapter IS its matched checkpoint")
print("  and `eval_matched.sh` copies the file rather than re-generating. One measurement,")
print("  reported twice -- not two evaluations that happen to agree.")
