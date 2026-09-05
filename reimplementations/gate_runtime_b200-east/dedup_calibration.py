#!/usr/bin/env python3
"""Floors for all three reference sets, in matched length bands, before any rate is reported.

The lesson from two withdrawn findings: a similarity rate quoted alone is uninterpretable,
because the measure has a length-dependent false-positive rate. And the floor is NOT one
number -- it scales with the size of the reference set, because the statistic is a maximum over
it. The held-out half is 337 problems, the training pool 59, the running buffer grows from
zero. Each therefore gets its own floor, measured against problems known to be unrelated.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
from tasksource.similarity import most_similar  # noqa: E402
from tasksource.sources import RetrievedSource  # noqa: E402

BANDS = [(0, 60), (60, 100), (100, 160), (160, 260), (260, 10**9)]
OUT = "/mnt/localssd/gate/out/tasksource"


def floor_for(name, reference, probes, thresholds):
    """False-positive rate of each threshold against problems known to be unrelated."""
    rows = []
    for text in probes:
        s, _ = most_similar(text, reference)
        rows.append((len(text), s))
    rec = {"reference": name, "reference_n": len(reference), "probes": len(rows),
           "overall": {}, "by_length": {}}
    sims = [s for _, s in rows]
    for th in thresholds:
        rec["overall"][str(th)] = round(sum(1 for s in sims if s >= th) / len(sims), 4)
    rec["mean_sim"] = round(statistics.fmean(sims), 4)
    for lo, hi in BANDS:
        sel = [s for ln, s in rows if lo <= ln < hi]
        if len(sel) < 10:
            continue
        rec["by_length"]["%d-%d" % (lo, hi if hi < 10**9 else 0)] = {
            "n": len(sel), "mean": round(statistics.fmean(sel), 4),
            **{str(th): round(sum(1 for s in sel if s >= th) / len(sel), 4)
               for th in thresholds}}
    return rec


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(8080)
    held = [p["problem"] for p in gate_lib.math_bench.load("olympiadbench", "report")]
    pool = [t["problem"] for t in
            json.load(open("/mnt/localssd/gate/out/pool_cap8192.json"))["tasks"]]
    # Probes: curated problems from OTHER corpora, unrelated to both by construction.
    src = RetrievedSource("/mnt/localssd/gate/data/azr/evaluation/math_eval/eval/data",
                          corpora=["gsm8k", "college_math", "math500", "minerva_math"])
    probes = [t.text for t in src.fetch(500, rng).tasks]
    # A stand-in running buffer: 30 further unrelated problems, the size one reaches mid-run.
    buffer_like = [t.text for t in src.fetch(30, random.Random(999)).tasks]

    ths = (0.45, 0.55, 0.60, 0.70)
    results = [
        floor_for("held_out (contamination, REJECT)", held, probes, ths),
        floor_for("training_pool (redundancy, FLAG)", pool, probes, ths),
        floor_for("running_buffer@30 (repetition, REJECT)", buffer_like, probes, ths),
    ]
    json.dump(results, open(os.path.join(OUT, "dedup_floors.json"), "w"), indent=2)

    print("FALSE-POSITIVE FLOORS on problems known to be unrelated (%d probes)\n" % len(probes))
    print("%-42s %6s %7s %s" % ("reference set", "n_ref", "mean", "  ".join(
        "@%.2f" % t for t in ths)))
    for r in results:
        print("%-42s %6d %7.4f %s" % (r["reference"], r["reference_n"], r["mean_sim"],
                                      "  ".join("%.4f" % r["overall"][str(t)] for t in ths)))
    print("\nThe floor scales with reference-set size: a maximum over 337 problems exceeds a\n"
          "maximum over 59, so the same threshold means different things against each set.\n")
    print("BY STATEMENT LENGTH, at each set's operating threshold")
    for r, th in zip(results, (0.45, 0.60, 0.60)):
        print("  %s  (threshold %.2f)" % (r["reference"], th))
        for band, rec in r["by_length"].items():
            print("     %-10s n=%-4d mean=%.4f  false positive=%.4f"
                  % (band, rec["n"], rec["mean"], rec[str(th)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
