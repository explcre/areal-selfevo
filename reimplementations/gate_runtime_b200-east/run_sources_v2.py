#!/usr/bin/env python3
"""The source table, run honestly: exact comparison, coverage reported, three dedup rates.

Three things this fixes relative to the first attempt:

1. **The comparator.** `symbolic_compare` (LaTeX stripped, unordered answer sets, exact
   rationals, inexact decimals refused) instead of string normalisation, which refuted 43% of
   curated keys that were not wrong and fell hardest on the source whose answer format differs
   most from the solver's.
2. **Coverage travels with every rate.** Symbolic decided 22 of 60 where string decided 37, so
   a source whose keys are mostly UNPARSEABLE is not a source whose keys are mostly right, and
   a refuted rate without its coverage cannot tell the two apart.
3. **Three reference sets, three rates, three meanings**, each against its own measured floor.

And it runs the control that would catch it being wrong: a set of keys known CORRECT and a set
known WRONG, put through the identical pipeline. Specificity without sensitivity is worthless
-- a verifier that refutes nothing scores perfectly on correct keys.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
import ornith_train as ot  # noqa: E402
from ornith_repro import symbolic as S  # noqa: E402
from ornith_repro import verify as V  # noqa: E402
from run_tasksource import ReplayClient  # noqa: E402
from tasksource.backends import SGLangBackend  # noqa: E402
from tasksource.references import RedundancyIndex, ReferenceSet  # noqa: E402
from tasksource.registry import SharedNoveltyBuffer  # noqa: E402
from tasksource.sources import (DISTIL_PROMPT, GENERATE_PROMPT, ModelWrittenSource,  # noqa: E402
                                RetrievedSource)


def corrupt(ans: str) -> str | None:
    """Make a key that is definitely WRONG, by changing the first integer in it.

    Used only for the sensitivity control. Returns None when no integer can be changed, so a
    key that cannot be corrupted is dropped rather than silently left correct -- otherwise the
    'wrong' set would be contaminated with right answers and sensitivity would look worse than
    it is for the wrong reason.
    """
    m = re.search(r"\d+", ans)
    if not m:
        return None
    n = int(m.group(0))
    return ans[:m.start()] + str(n + 3) + ans[m.end():]


def verdicts_for(url, tok, items, k, cap):
    """Consensus-solve every item and compare with symbolic AND string comparators."""
    prompts = [V.SOLUTION_PROMPT.format(problem=t["problem"]) for t in items for _ in range(k)]
    if not prompts:
        return [], 0
    recs = asyncio.run(ot.gen_batch(url, tok, prompts, cap, 96, ""))
    out = []
    for i, t in enumerate(items):
        mine = recs[i * k:(i + 1) * k]
        src = V.SolverConsensus(ReplayClient([(r["text"], ot.unanswered(r)) for r in mine]),
                                k=k, max_new_tokens=cap)
        computed, detail, _ = src.solve(t["problem"])
        if computed is None:
            out.append({"v": "unverifiable", "reason": detail, "computed": None})
            continue
        sym, why = S.symbolic_compare(computed, t["answer"])
        out.append({"v": {"equal": "verified", "different": "refuted",
                          "indeterminate": "unverifiable"}[sym],
                    "reason": why, "computed": computed})
    return out, sum(len(r["output_ids"]) for r in recs)


def summarise(name, vs):
    """Rate plus the coverage it rests on; neither means anything without the other."""
    dec = sum(1 for v in vs if v["v"] in ("verified", "refuted"))
    ref = sum(1 for v in vs if v["v"] == "refuted")
    lo, hi = gate_lib.math_bench.wilson(ref, dec) if dec else (None, None)
    return {"source": name, "n": len(vs), "decided": dec,
            "coverage": round(dec / len(vs), 4) if vs else None,
            "refuted": ref, "refuted_rate": round(ref / dec, 4) if dec else None,
            "wilson95": [round(lo, 4) if lo is not None else None,
                         round(hi, 4) if hi is not None else None]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--n-control", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--gen-cap", type=int, default=16384)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/tasksource")
    ap.add_argument("--seed", type=int, default=31337)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    held = [p["problem"] for p in gate_lib.math_bench.load("olympiadbench", "report")]
    pool = [t["problem"] for t in
            json.load(open("/mnt/localssd/gate/out/pool_cap8192.json"))["tasks"]]

    # ---------- CONTROL FIRST: does the pipeline separate right keys from wrong ones? ------
    ctl_src = RetrievedSource(a.data_root, corpora=["math500", "gsm8k", "amc23"])
    ctl = ctl_src.fetch(a.n_control, random.Random(a.seed + 1)).tasks
    right = [{"problem": t.text, "answer": t.answer} for t in ctl]
    wrong = []
    for t in ctl:
        c = corrupt(t.answer)
        if c is not None and c != t.answer:
            wrong.append({"problem": t.text, "answer": c})
    print("control: %d keys known correct, %d deliberately corrupted" % (len(right), len(wrong)),
          flush=True)
    v_right, tok_r = verdicts_for(a.url, tok, right, a.k, a.cap)
    v_wrong, tok_w = verdicts_for(a.url, tok, wrong, a.k, a.cap)
    spec = summarise("keys known CORRECT", v_right)
    sens = summarise("keys known WRONG", v_wrong)
    print("  specificity: %s" % spec, flush=True)
    print("  sensitivity: %s" % sens, flush=True)

    # ---------- the three sources through the identical path -------------------------------
    backend = SGLangBackend(a.url, tok, cap=a.gen_cap, concurrency=80, name="qwen38-27b-local")
    srcs = [
        ModelWrittenSource("generated", backend, GENERATE_PROMPT, oversample=4),
        RetrievedSource(a.data_root, corpora=["math500", "gsm8k", "amc23", "aime24",
                                              "aime25", "minerva_math", "college_math"]),
        ModelWrittenSource("distilled", backend, DISTIL_PROMPT, oversample=4),
    ]
    held_ref = ReferenceSet("held_out", held, 0.45, "reject", "contamination: fatal")
    buf = SharedNoveltyBuffer(threshold=0.60)
    red = RedundancyIndex(pool, threshold=0.60)

    rows, per_source_items, gen_cost = [], {}, {}
    for src in srcs:
        res = src.fetch(a.n, rng)
        gen_cost[src.name] = res.cost_tokens
        if not res.ok:
            rows.append({"source": src.name, "ok": False, "reason": res.reason})
            per_source_items[src.name] = []
            print("SOURCE FAILURE %s: %s" % (src.name, res.reason), flush=True)
            continue
        n_contam = n_dupe = n_redundant = 0
        kept = []
        for t in res.tasks:
            sim_h, _ = held_ref.check(t.text)
            if sim_h >= held_ref.threshold:
                n_contam += 1
                continue
            ok_b, sim_b, owner = buf.check(t.text)
            if not ok_b:
                n_dupe += 1
                continue
            is_red, sim_r = red.check(t.text)          # FLAG only, never rejects
            if is_red:
                n_redundant += 1
            buf.add(t.text, src.name)
            kept.append({"problem": t.text, "answer": t.answer, "redundant": is_red,
                         "sim_pool": round(sim_r, 4)})
        per_source_items[src.name] = kept
        rows.append({"source": src.name, "ok": True, "candidates": len(res.tasks),
                     "rejected_contamination": n_contam, "rejected_repetition": n_dupe,
                     "flagged_redundant": n_redundant, "kept": len(kept)})

    verify_tokens = 0
    for r in rows:
        if not r.get("ok"):
            continue
        vs, tk = verdicts_for(a.url, tok, per_source_items[r["source"]], a.k, a.cap)
        verify_tokens += tk
        r.update({k: v for k, v in summarise(r["source"], vs).items() if k != "source"})
        r["cost_tokens"] = gen_cost[r["source"]] + tk

    out = {"control": {"specificity_on_correct_keys": spec,
                       "sensitivity_on_corrupted_keys": sens,
                       "control_tokens": tok_r + tok_w},
           "sources": rows, "verify_tokens": verify_tokens,
           "floors": "see dedup_floors.json: held_out@0.45=0.050, pool@0.60=0.000, "
                     "buffer@0.60=0.128 (0.292 on statements under 60 chars)"}
    json.dump(out, open(os.path.join(a.out, "sources_v2.json"), "w"), indent=1, default=str)

    print()
    print("%-10s %5s %7s %7s %7s %6s %8s %8s %9s" %
          ("source", "cand", "contam", "repeat", "redund", "kept", "coverage", "refuted",
           "ref_rate"))
    for r in rows:
        if not r.get("ok"):
            print("%-10s SOURCE FAILURE: %s" % (r["source"], r["reason"]))
            continue
        print("%-10s %5d %7d %7d %7d %6d %8s %8s %9s" %
              (r["source"], r["candidates"], r["rejected_contamination"],
               r["rejected_repetition"], r["flagged_redundant"], r["kept"],
               ("%.3f" % r["coverage"]) if r.get("coverage") is not None else "n/a",
               r.get("refuted"), ("%.4f" % r["refuted_rate"])
               if r.get("refuted_rate") is not None else "n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
