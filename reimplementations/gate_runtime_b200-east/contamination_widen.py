#!/usr/bin/env python3
"""Widen the contamination measurement, and show the actual overlaps.

The first run rejected 7 of 20 self-generated candidates for overlapping the held-out half,
against 3 of 20 retrieved and 2 of 20 distilled -- the opposite of the expectation that
retrieval is where contamination comes from. That is worth more n and, more importantly, worth
showing: a rejection rate at a conservative threshold is an upper bound, and a reader cannot
tell a true duplicate from an artefact of the cut without seeing the pairs.

So this reports the whole similarity DISTRIBUTION per source, the rejection rate at several
thresholds rather than one, and the nearest held-out problem for the top overlaps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
from tasksource.backends import SGLangBackend  # noqa: E402
from tasksource.similarity import most_similar  # noqa: E402
from tasksource.sources import (DISTIL_PROMPT, GENERATE_PROMPT, ModelWrittenSource,  # noqa: E402
                                RetrievedSource)

THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--n-model", type=int, default=40)
    ap.add_argument("--n-retrieved", type=int, default=200)
    ap.add_argument("--gen-cap", type=int, default=16384)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/tasksource")
    ap.add_argument("--seed", type=int, default=5150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    held = [p["problem"] for p in gate_lib.math_bench.load("olympiadbench", "report")]
    print("held-out report half: %d problems" % len(held), flush=True)

    backend = SGLangBackend(a.url, tok, cap=a.gen_cap, concurrency=80, name="qwen38-27b-local")
    sources = [
        ("generated", ModelWrittenSource("generated", backend, GENERATE_PROMPT, oversample=4),
         a.n_model),
        ("distilled", ModelWrittenSource("distilled", backend, DISTIL_PROMPT, oversample=4),
         a.n_model),
        ("retrieved", RetrievedSource(a.data_root,
                                      corpora=["math500", "gsm8k", "amc23", "aime24",
                                               "aime25", "minerva_math", "college_math"]),
         a.n_retrieved),
    ]

    report, pairs = {}, []
    for name, src, n in sources:
        res = src.fetch(n, rng)
        if not res.ok:
            report[name] = {"ok": False, "reason": res.reason}
            print("SOURCE FAILURE %s: %s" % (name, res.reason), flush=True)
            continue
        sims = []
        for t in res.tasks:
            s, at = most_similar(t.text, held)
            sims.append(s)
            pairs.append({"source": name, "sim": s, "candidate": t.text[:300],
                          "nearest_held_out": held[at][:300] if at >= 0 else None,
                          "origin": t.provenance.origin})
        sims.sort()
        report[name] = {
            "ok": True, "n": len(sims),
            "mean_sim": round(statistics.fmean(sims), 4),
            "median_sim": round(sims[len(sims) // 2], 4),
            "p90_sim": round(sims[int(0.9 * len(sims))], 4),
            "max_sim": round(sims[-1], 4),
            "rejection_rate": {str(th): round(sum(1 for s in sims if s >= th) / len(sims), 4)
                               for th in THRESHOLDS},
            "cost_tokens": res.cost_tokens,
        }
        print("%-10s n=%-4d mean=%.3f median=%.3f p90=%.3f max=%.3f  reject@0.45=%.3f"
              % (name, len(sims), report[name]["mean_sim"], report[name]["median_sim"],
                 report[name]["p90_sim"], report[name]["max_sim"],
                 report[name]["rejection_rate"]["0.45"]), flush=True)

    pairs.sort(key=lambda p: -p["sim"])
    json.dump({"held_out_n": len(held), "thresholds": list(THRESHOLDS),
               "per_source": report, "top_overlaps": pairs[:25]},
              open(os.path.join(a.out, "contamination.json"), "w"), indent=1, default=str)

    print()
    print("REJECTION RATE BY THRESHOLD (the reported filter uses 0.45)")
    print("%-10s %s" % ("source", "  ".join("%.2f" % t for t in THRESHOLDS)))
    for name in report:
        if report[name].get("ok"):
            print("%-10s %s" % (name, "  ".join(
                "%.3f" % report[name]["rejection_rate"][str(t)] for t in THRESHOLDS)))
    print()
    print("TOP OVERLAPS -- judge for yourself whether these are duplicates:")
    for p in pairs[:6]:
        print("  sim=%.3f  [%s %s]" % (p["sim"], p["source"], p["origin"]))
        print("    candidate : %s" % p["candidate"][:190].replace("\n", " "))
        print("    held-out  : %s" % (p["nearest_held_out"] or "")[:190].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
