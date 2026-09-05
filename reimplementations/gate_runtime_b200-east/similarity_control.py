#!/usr/bin/env python3
"""Is the contamination measure trustworthy at these statement lengths? A negative control.

The widened run rejected 20% of model-written candidates and 7.5% of retrieved ones against
the held-out half, which looks like self-generation being the bigger contamination risk. But
the highest-scoring "overlaps" are plainly unrelated -- a derivative of -3/(4x^2-2x+1) scored
0.625 against an arctangent identity -- so before that ordering is reported as contamination
the measure needs a control.

This is that control, and it costs no GPU. Problems from a corpus that CANNOT be contaminating
the held-out half in the pairs examined are scored against it, bucketed by statement length. If
short statements score high against unrelated problems, the rejection rate is measuring brevity
rather than overlap, and the ordering between sources follows statement style rather than
memorisation.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
from tasksource.similarity import most_similar  # noqa: E402
from tasksource.sources import RetrievedSource  # noqa: E402

import random

held = [p["problem"] for p in gate_lib.math_bench.load("olympiadbench", "report")]
src = RetrievedSource("/mnt/localssd/gate/data/azr/evaluation/math_eval/eval/data",
                      corpora=["gsm8k", "college_math", "math500"])
res = src.fetch(400, random.Random(4242))
rows = []
for t in res.tasks:
    s, at = most_similar(t.text, held)
    rows.append((len(t.text), s, t.text, held[at] if at >= 0 else ""))

BUCKETS = [(0, 60), (60, 100), (100, 160), (160, 260), (260, 10000)]
print("Negative control: %d curated problems scored against the %d held-out problems."
      % (len(rows), len(held)))
print("These are DIFFERENT corpora; a high score here is a false positive by construction\n"
      "unless the two corpora genuinely share a problem.\n")
print("%-14s %5s %8s %8s %8s %9s %9s" %
      ("statement len", "n", "mean", "median", "max", "rej@0.45", "rej@0.55"))
out = {}
for lo, hi in BUCKETS:
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        continue
    sims = sorted(r[1] for r in sel)
    rec = {"n": len(sel), "mean": round(statistics.fmean(sims), 4),
           "median": round(sims[len(sims) // 2], 4), "max": round(sims[-1], 4),
           "rej_045": round(sum(1 for s in sims if s >= 0.45) / len(sims), 4),
           "rej_055": round(sum(1 for s in sims if s >= 0.55) / len(sims), 4)}
    out["%d-%d" % (lo, hi)] = rec
    print("%-14s %5d %8.4f %8.4f %8.4f %9.4f %9.4f"
          % ("%d-%d" % (lo, hi), rec["n"], rec["mean"], rec["median"], rec["max"],
             rec["rej_045"], rec["rej_055"]))

allsims = [r[1] for r in rows]
overall = {"n": len(rows), "mean": round(statistics.fmean(allsims), 4),
           "false_positive_rate_at_045": round(sum(1 for s in allsims if s >= 0.45) / len(rows), 4),
           "false_positive_rate_at_055": round(sum(1 for s in allsims if s >= 0.55) / len(rows), 4)}
print("\nOVERALL false-positive rate of the 0.45 cut on unrelated curated problems: %.4f"
      % overall["false_positive_rate_at_045"])
print("             ... and of a 0.55 cut: %.4f" % overall["false_positive_rate_at_055"])

rows.sort(key=lambda r: -r[1])
print("\nWorst false positives (unrelated by inspection):")
for ln, s, a, b in rows[:4]:
    print("  sim=%.3f  len=%d" % (s, ln))
    print("    corpus  : %s" % a[:150].replace("\n", " "))
    print("    held-out: %s" % b[:150].replace("\n", " "))
json.dump({"by_length": out, "overall": overall,
           "worst": [{"sim": s, "len": ln, "a": a[:300], "b": b[:300]}
                     for ln, s, a, b in rows[:15]]},
          open("/mnt/localssd/gate/out/tasksource/similarity_control.json", "w"),
          indent=1, default=str)
