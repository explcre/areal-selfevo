#!/usr/bin/env python3
"""Re-run the curated-key calibration with the EXACT-ARITHMETIC comparator, paired.

The first calibration used `answers_match`, the string-normalising comparison the live grader
uses, and reported a 0.447 false-refutation rate on professionally curated keys. Reading the
refutations showed every inspected one was a formatting difference, which is exactly what
`symbolic.symbolic_compare` was built for: LaTeX stripped, answers split as unordered sets,
exact rationals, and inexact decimals REFUSED rather than coerced.

Both comparators are run over the SAME 60 problems and the SAME solutions, so the difference
is the comparator and nothing else. A third outcome matters: `symbolic_compare` may return
`indeterminate`, which becomes UNVERIFIABLE rather than a refutation -- declining to decide is
the correct behaviour for an unparseable pair and is not a false refutation.
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
from ornith_repro import symbolic as S  # noqa: E402
from ornith_repro import verify as V  # noqa: E402
from ornith_repro.loop import answers_match  # noqa: E402
from run_tasksource import ReplayClient  # noqa: E402
from tasksource.sources import RetrievedSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/tasksource")
    ap.add_argument("--seed", type=int, default=771)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    src = RetrievedSource(a.data_root, corpora=["math500", "gsm8k", "amc23", "aime24",
                                                "aime25", "minerva_math", "college_math"])
    res = src.fetch(a.n, random.Random(a.seed))
    tasks = res.tasks
    print("curated tasks: %d (same seed as the first calibration)" % len(tasks), flush=True)

    prompts = [V.SOLUTION_PROMPT.format(problem=t.text) for t in tasks for _ in range(a.k)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, prompts, a.cap, 96, ""))

    rows, agree = [], {"both_verified": 0, "both_refuted": 0,
                       "string_refuted_symbolic_equal": 0,
                       "string_refuted_symbolic_indeterminate": 0,
                       "string_verified_symbolic_different": 0}
    for i, t in enumerate(tasks):
        mine = recs[i * a.k:(i + 1) * a.k]
        source = V.SolverConsensus(ReplayClient([(r["text"], ot.unanswered(r)) for r in mine]),
                                   k=a.k, max_new_tokens=a.cap)
        computed, detail, _ = source.solve(t.text)
        if computed is None:
            rows.append({"origin": t.provenance.origin, "computed": None,
                         "string": "unverifiable", "symbolic": "unverifiable",
                         "detail": detail})
            continue
        s_ok = answers_match(computed, t.answer)
        sym, why = S.symbolic_compare(computed, t.answer)
        string_v = "verified" if s_ok else "refuted"
        symbolic_v = {"equal": "verified", "different": "refuted",
                      "indeterminate": "unverifiable"}[sym]
        if string_v == "verified" and symbolic_v == "verified":
            agree["both_verified"] += 1
        elif string_v == "refuted" and symbolic_v == "refuted":
            agree["both_refuted"] += 1
        elif string_v == "refuted" and symbolic_v == "verified":
            agree["string_refuted_symbolic_equal"] += 1
        elif string_v == "refuted" and symbolic_v == "unverifiable":
            agree["string_refuted_symbolic_indeterminate"] += 1
        elif string_v == "verified" and symbolic_v == "different":
            agree["string_verified_symbolic_different"] += 1
        rows.append({"origin": t.provenance.origin,
                     "corpus": t.provenance.detail.get("corpus"),
                     "problem": t.text[:300], "asserted": t.answer, "computed": computed,
                     "string": string_v, "symbolic": symbolic_v, "symbolic_reason": why})

    def rate(key):
        dec = sum(1 for r in rows if r[key] in ("verified", "refuted"))
        ref = sum(1 for r in rows if r[key] == "refuted")
        lo, hi = ot.gate_lib.math_bench.wilson(ref, dec) if dec else (None, None)
        return {"decided": dec, "refuted": ref,
                "false_refutation_rate": (ref / dec) if dec else None,
                "wilson95": [lo, hi]}

    out = {"n_tasks": len(tasks), "k": a.k,
           "string_comparator_answers_match": rate("string"),
           "symbolic_comparator": rate("symbolic"),
           "paired_agreement": agree,
           "still_refuted_under_symbolic": [r for r in rows if r["symbolic"] == "refuted"]}
    json.dump({**out, "all_rows": rows},
              open(os.path.join(a.out, "recalibration.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "still_refuted_under_symbolic"}, indent=1, default=str))
    print("\nSTILL REFUTED under the exact comparator -- these are the candidate REAL wrong "
          "keys, read them:")
    for r in out["still_refuted_under_symbolic"][:10]:
        print("  [%s] asserted=%r computed=%r (%s)"
              % (r["origin"], r["asserted"], r["computed"], r["symbolic_reason"]))
        print("      %s" % r["problem"][:200].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
