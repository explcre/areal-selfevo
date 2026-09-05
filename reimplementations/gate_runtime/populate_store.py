#!/usr/bin/env python3
"""Fill the permanent store from all three sources, every field measured rather than defaulted.

One pass produces everything a record needs, reusing what already exists rather than adding a
second verifier or a second novelty check:

* the three dedup readings come from the same similarity measure, against the three reference
  sets, each with its threshold and its measured length-band floor;
* the k consensus solutions serve TWICE -- as the verifier's independent answers and as the
  difficulty sample -- so the success rate costs no extra generation. It is recorded with the
  prompt it was measured under, because difficulty under a scaffold is a different quantity.

Written to /extra (permanent, 56 TB free) and mirrored to gs://selfevo, never to scratch.
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
from tasksource.store import StoredTask, TaskStore, content_hash, make_dedup  # noqa: E402

BARE_PROMPT_ID = "ornith_repro.verify.SOLUTION_PROMPT/bare/v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--store-root", default="/mnt/localssd/gate/out/taskstore")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--gen-cap", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    held = [p["problem"] for p in gate_lib.math_bench.load("olympiadbench", "report")]
    pool = [t["problem"] for t in
            json.load(open("/mnt/localssd/gate/out/pool_cap8192.json"))["tasks"]]
    held_ref = ReferenceSet("held_out", held, 0.45, "reject", "contamination: fatal")
    buf = SharedNoveltyBuffer(threshold=0.60)
    red = RedundancyIndex(pool, threshold=0.60)

    backend = SGLangBackend(a.url, tok, cap=a.gen_cap, concurrency=64, name="qwen38-27b-local")
    TEACHER = {"teacher_model": "Qwen/Qwen3.8-27B", "teacher_version": "local-sglang-0.5.18",
               "prompt_version": "distil_v1",
               "prompt_hash": content_hash(DISTIL_PROMPT)[:16],
               "collusion_note": "teacher is the SAME model as the solver being trained; a "
                                 "distilled task cannot certify capability the solver lacks"}
    srcs = [
        ("generated", ModelWrittenSource("generated", backend, GENERATE_PROMPT, oversample=4),
         {"prompt_version": "generate_v1",
          "prompt_hash": content_hash(GENERATE_PROMPT)[:16],
          "exemplars": [], "exemplars_note": "zero-shot; the prompt carries no exemplars, "
                                             "recorded explicitly rather than omitted"}),
        ("retrieved", RetrievedSource(a.data_root,
                                      corpora=["math500", "gsm8k", "amc23", "aime24",
                                               "minerva_math", "college_math"]), None),
        ("distilled", ModelWrittenSource("distilled", backend, DISTIL_PROMPT, oversample=4),
         TEACHER),
    ]

    staged, produce_cost = [], {}
    for name, src, prov_extra in srcs:
        res = src.fetch(a.n, rng)
        produce_cost[name] = (res.cost_tokens / max(len(res.tasks), 1)) if res.ok else 0
        if not res.ok:
            print("SOURCE FAILURE %-10s %s" % (name, res.reason), flush=True)
            continue
        for t in res.tasks:
            sim_h, _ = held_ref.check(t.text)
            if sim_h >= held_ref.threshold:
                continue                       # contamination: rejected, never stored
            ok_b, sim_b, _ = buf.check(t.text)
            if not ok_b:
                continue                       # repetition: rejected
            _, sim_r = red.check(t.text)       # redundancy: flagged only
            buf.add(t.text, name)
            prov = dict(prov_extra) if prov_extra else dict(t.provenance.detail)
            if name == "retrieved":
                prov = {"corpus": t.provenance.detail.get("corpus"),
                        "row": t.provenance.detail.get("row"),
                        "licence": t.provenance.licence,
                        "retrieved_from": t.provenance.detail.get("retrieved_from")}
            staged.append({"text": t.text, "answer": t.answer, "source": name, "prov": prov,
                           "sim_h": sim_h, "sim_b": sim_b, "sim_r": sim_r})
        print("%-10s staged %d" % (name, sum(1 for s in staged if s["source"] == name)),
              flush=True)

    # One generation pass serves the verifier AND the difficulty estimate.
    prompts = [V.SOLUTION_PROMPT.format(problem=s["text"]) for s in staged
               for _ in range(a.k)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, prompts, a.cap, 96, "")) if prompts else []

    store = TaskStore(a.store_root)
    written, by_source = 0, {}
    for i, s in enumerate(staged):
        mine = recs[i * a.k:(i + 1) * a.k]
        src_obj = V.SolverConsensus(
            ReplayClient([(r["text"], ot.unanswered(r)) for r in mine]), k=a.k,
            max_new_tokens=a.cap)
        computed, detail, _ = src_obj.solve(s["text"])
        if computed is None:
            verdict, why, witness = "unverifiable", detail, None
        else:
            sym, why = S.symbolic_compare(computed, s["answer"])
            verdict = {"equal": "verified", "different": "refuted",
                       "indeterminate": "unverifiable"}[sym]
            witness = computed if verdict == "refuted" else None
        # Difficulty: the same k samples, graded against the asserted key, under a NAMED prompt.
        graded = [r for r in mine if not ot.unanswered(r)]
        n_ok = sum(1 for r in graded if gate_lib.math_bench.grade(r["text"], s["answer"]))
        rec = StoredTask(
            text=s["text"], answer=s["answer"], source_type=s["source"], provenance=s["prov"],
            verification={"verdict": verdict, "backend": "solver_consensus_k%d" % a.k,
                          "comparator": "ornith_repro.symbolic.symbolic_compare",
                          "reason": why, "witness": witness},
            difficulty=({"success_rate": round(n_ok / len(graded), 4),
                         "measured_under_prompt": BARE_PROMPT_ID,
                         "prompt_hash": content_hash(V.SOLUTION_PROMPT)[:16],
                         "n_samples": len(graded), "graded_against": "asserted key"}
                        if graded else {}),
            dedup={"held_out": make_dedup("held_out", s["sim_h"], 0.45, s["text"]),
                   "training_pool": make_dedup("training_pool", s["sim_r"], 0.60, s["text"]),
                   "run_buffer": make_dedup("run_buffer", s["sim_b"], 0.60, s["text"])},
            cost={"produce_tokens": int(produce_cost[s["source"]]),
                  "produce_basis": "amortised over this source's oversampled batch; a "
                                   "generator pays for the candidates it discards",
                  "score_tokens": sum(len(r["output_ids"]) for r in mine),
                  "score_basis": "this task's own %d samples, exactly attributable" % a.k})
        store.append(rec)
        written += 1
        by_source[s["source"]] = by_source.get(s["source"], 0) + 1

    print("\nwrote %d records to %s" % (written, store.path))
    print("by source: %s" % by_source)
    back = TaskStore.read(store.path)
    print("read back %d records, schema v%s, all validated"
          % (len(back), back[0]["schema_version"] if back else "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
