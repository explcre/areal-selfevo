#!/usr/bin/env python3
"""Calibrate the verifier's FALSE-REFUTATION rate on professionally curated keys.

Why this exists. The three-source table put the refuted-key rate at 0.400 for RETRIEVED
tasks, on five decided cases -- higher than for generated ones and an order of magnitude above
the 2.55% published for curated problems. Curated keys from MATH, AIME and AMC are not wrong
two times in five. So that number is almost certainly measuring the VERIFIER, not the corpora,
and reporting it as a property of retrieval would be reporting an instrument as a finding.

Retrieval is free of generation cost, so the curated corpora can double as a calibration set:
their keys are right by assumption, and every REFUTED verdict on them is therefore a false
refutation. This measures that rate and dumps every case so it can be read individually, which
is the same method that established the generated-key figure was real.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import ornith_train as ot  # noqa: E402
from ornith_repro import verify as V  # noqa: E402
from tasksource.sources import RetrievedSource  # noqa: E402
from run_tasksource import ReplayClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/tasksource")
    ap.add_argument("--seed", type=int, default=771)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rng = random.Random(a.seed)
    src = RetrievedSource(a.data_root,
                          corpora=["math500", "gsm8k", "amc23", "aime24", "aime25",
                                   "minerva_math", "college_math"])
    res = src.fetch(a.n, rng)
    if not res.ok:
        print("FAILED: %s" % res.reason)
        return 1
    tasks = res.tasks
    print("curated tasks: %d" % len(tasks), flush=True)

    prompts = [V.SOLUTION_PROMPT.format(problem=t.text) for t in tasks for _ in range(a.k)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, prompts, a.cap, 96, ""))
    tot_tokens = sum(len(r["output_ids"]) for r in recs)

    counts = {"verified": 0, "refuted": 0, "unverifiable_mechanical": 0,
              "unverifiable_substantive": 0}
    refuted, by_corpus = [], {}
    for i, t in enumerate(tasks):
        mine = recs[i * a.k:(i + 1) * a.k]
        source = V.SolverConsensus(
            ReplayClient([(r["text"], ot.unanswered(r)) for r in mine]),
            k=a.k, threshold=a.threshold, max_new_tokens=a.cap)
        r = V.verify_answer(t.text, t.answer, source)
        c = t.provenance.detail.get("corpus", "?")
        d = by_corpus.setdefault(c, {"decided": 0, "refuted": 0})
        if r.verdict is V.Verdict.VERIFIED:
            counts["verified"] += 1
            d["decided"] += 1
        elif r.verdict is V.Verdict.REFUTED:
            counts["refuted"] += 1
            d["decided"] += 1
            d["refuted"] += 1
            refuted.append({"corpus": c, "origin": t.provenance.origin,
                            "problem": t.text[:400], "asserted_key": t.answer,
                            "verifier_computed": r.computed, "detail": r.detail})
        elif r.abstain is V.Abstain.MECHANICAL:
            counts["unverifiable_mechanical"] += 1
        else:
            counts["unverifiable_substantive"] += 1

    decided = counts["verified"] + counts["refuted"]
    rate = counts["refuted"] / decided if decided else None
    lo = hi = None
    if decided:
        # Wilson interval, the same one this paper uses for small counts.
        lo, hi = ot.gate_lib.math_bench.wilson(counts["refuted"], decided)
    out = {"n_tasks": len(tasks), "k": a.k, "consensus_threshold": a.threshold,
           "counts": counts, "decided": decided,
           "false_refutation_rate_on_curated_keys": rate,
           "wilson95": [lo, hi], "verify_tokens": tot_tokens,
           "by_corpus": by_corpus, "refuted_cases": refuted}
    json.dump(out, open(os.path.join(a.out, "verifier_calibration.json"), "w"),
              indent=1, default=str)

    print(json.dumps({k: v for k, v in out.items() if k != "refuted_cases"},
                     indent=1, default=str))
    print("\nREFUTED CURATED KEYS (each one is a candidate FALSE refutation; read them):")
    for r in refuted[:8]:
        print("  [%s %s] key=%r  verifier=%r  (%s)"
              % (r["corpus"], r["origin"], r["asserted_key"], r["verifier_computed"],
                 r["detail"]))
        print("      %s" % r["problem"][:220].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
