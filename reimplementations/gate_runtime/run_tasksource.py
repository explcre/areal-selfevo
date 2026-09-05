#!/usr/bin/env python3
"""The deliverable table: three sources, one path, per-source outcomes.

The falsifiable prediction under test. Self-generated answer keys are refuted 6.56% of the
time among decided cases against 2.55% for professionally curated ones, and reading all
sixteen refutations individually established that this is the generator rather than the
instrument. Retrieved problems carry keys written by someone other than the model being
trained, so they should NOT show that failure: their refuted rate should look like the
curated figure, not the generated one.

Every candidate takes the identical path (contamination, then ONE shared novelty buffer, then
the verifier), so a difference between sources is a difference in what survives it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
import ornith_train as ot  # noqa: E402
from ornith_repro import verify as V  # noqa: E402
from ornith_repro.llm import StubClient  # noqa: E402  (protocol reference)
from tasksource.backends import HostedTeacher, SGLangBackend  # noqa: E402
from tasksource.pipeline import TaskPipeline  # noqa: E402
from tasksource.registry import ContaminationFilter, SharedNoveltyBuffer  # noqa: E402
from tasksource.sources import (DISTIL_PROMPT, GENERATE_PROMPT, ModelWrittenSource,  # noqa: E402
                                RetrievedSource)


class ReplayClient:
    """Serves pre-generated completions to the package's own consensus backend.

    `SolverConsensus` calls `client.generate` k times in sequence. Generating those k attempts
    one at a time would serialise the whole verification; they are batched in advance and
    replayed here, so THEIR consensus logic and THEIR key comparison stay authoritative and
    only the transport changes. `verify_answer` still hands the backend the problem alone, so
    the safeguard that no backend sees the key it is checking is untouched.
    """

    def __init__(self, items):
        self._items, self._i = list(items), 0

    def generate(self, prompt, max_new_tokens, seed):
        if self._i >= len(self._items):
            return "", True
        out = self._items[self._i]
        self._i += 1
        return out


def main() -> int:
    """Run the three sources and print the per-source table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--data-root", default="/mnt/localssd/gate/data/azr/evaluation/"
                                           "math_eval/eval/data")
    ap.add_argument("--n-per-source", type=int, default=20)
    ap.add_argument("--gen-cap", type=int, default=16384)
    ap.add_argument("--verify-k", type=int, default=5)
    ap.add_argument("--verify-cap", type=int, default=8192)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/tasksource")
    ap.add_argument("--seed", type=int, default=90501)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    t0 = time.time()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    # The held-out half: the ONLY thing contamination is checked against.
    problems = gate_lib.math_bench.load("olympiadbench", "report")
    held_out = [p["problem"] for p in problems]
    print("held-out report half: %d problems" % len(held_out), flush=True)

    backend = SGLangBackend(a.url, tok, cap=a.gen_cap, concurrency=64, name="qwen38-27b-local")
    sources = [
        ModelWrittenSource("generated", backend, GENERATE_PROMPT,
                           context="", licence="model-generated (no external content)",
                           oversample=4, detail={"writer": "solver's own policy"}),
        RetrievedSource(a.data_root,
                        corpora=["math500", "gsm8k", "amc23", "aime24", "aime25",
                                 "minerva_math", "college_math"],
                        exclude_corpora=("olympiadbench",)),
        # Provider-agnostic teacher, TESTED LOCALLY. The collusion risk is that this teacher
        # is the same model as the solver, which is the hazard already logged for the judges;
        # it is recorded in provenance rather than assumed away.
        ModelWrittenSource("distilled", backend, DISTIL_PROMPT,
                           context="", licence="model-generated (no external content)",
                           oversample=4,
                           detail={"teacher": "qwen38-27b-local",
                                   "collusion_risk": "teacher is the SAME model as the "
                                                     "solver; a hosted teacher of a "
                                                     "different family is not authorised"}),
    ]

    buffer = SharedNoveltyBuffer(threshold=0.60)
    contamination = ContaminationFilter(held_out=held_out, threshold=0.45)

    # ---- verification, batched but using the package's own consensus and comparison -------
    pending: list = []

    def collect(task):
        pending.append(task)
        return ("unverifiable_substantive", 0)

    pipe = TaskPipeline(buffer, contamination, verifier=collect)
    stats = pipe.run(sources, a.n_per_source, rng)
    print("fetched and filtered in %.0fs; %d candidates reached verification"
          % (time.time() - t0, len(pending)), flush=True)

    prompts = [V.SOLUTION_PROMPT.format(problem=t.text) for t in pending
               for _ in range(a.verify_k)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, prompts, a.verify_cap,
                                    min(len(prompts), 96), "")) if prompts else []
    verify_tokens = sum(len(r["output_ids"]) for r in recs)
    verdicts = {}
    for i, t in enumerate(pending):
        mine = recs[i * a.verify_k:(i + 1) * a.verify_k]
        src = V.SolverConsensus(ReplayClient([(r["text"], ot.unanswered(r)) for r in mine]),
                                k=a.verify_k, max_new_tokens=a.verify_cap)
        res = V.verify_answer(t.text, t.answer, src)
        verdicts[t.task_id] = res

    # Re-run the pipeline with the real verdicts, from a clean buffer, so the accounting is
    # exact rather than patched after the fact.
    buffer2 = SharedNoveltyBuffer(threshold=0.60)
    tok_by_src = {s.name: stats[s.name].cost_tokens for s in sources}
    fetched = {s.name: [t for t in pending] for s in sources}

    def verifier(task):
        r = verdicts.get(task.task_id)
        if r is None:
            return ("unverifiable_mechanical", 0)
        if r.verdict is V.Verdict.VERIFIED:
            return ("verified", 0)
        if r.verdict is V.Verdict.REFUTED:
            return ("refuted", 0)
        return (("unverifiable_mechanical" if r.abstain is V.Abstain.MECHANICAL
                 else "unverifiable_substantive"), 0)

    class Replay:
        def __init__(self, name, tasks, cost, detail, ok, reason):
            self.name, self._t, self._c = name, tasks, cost
            self._d, self._ok, self._r = detail, ok, reason

        def fetch(self, n, rng):
            from tasksource.base import SourceResult
            if not self._ok:
                return SourceResult.failure(self.name, n, self._r, cost_tokens=self._c)
            return SourceResult(self.name, self._t, attempted=n, ok=True,
                                cost_tokens=self._c, detail=self._d)

    by_source: dict = {s.name: [] for s in sources}
    for t in pending:
        by_source[t.provenance.source].append(t)
    replays = [Replay(s.name, by_source[s.name], stats[s.name].cost_tokens,
                      stats[s.name].detail, stats[s.name].ok or bool(by_source[s.name]),
                      stats[s.name].failure_reason) for s in sources]
    pipe2 = TaskPipeline(buffer2, contamination, verifier=verifier)
    final = pipe2.run(replays, a.n_per_source, rng)
    for s in sources:
        final[s.name].rejected_contaminated = stats[s.name].rejected_contaminated
        final[s.name].rejected_duplicate += stats[s.name].rejected_duplicate
        final[s.name].candidates = stats[s.name].candidates
        final[s.name].max_contamination_sim = stats[s.name].max_contamination_sim
        final[s.name].verify_tokens = int(verify_tokens * len(by_source[s.name])
                                          / max(len(pending), 1))

    n_written = pipe2.write_artifacts(os.path.join(a.out, "accepted_tasks.jsonl"))
    rows = [final[s.name].as_row() for s in sources]
    json.dump({"rows": rows, "held_out_n": len(held_out),
               "verify_tokens_total": verify_tokens,
               "elapsed_s": round(time.time() - t0, 1),
               "accepted_written": n_written},
              open(os.path.join(a.out, "table.json"), "w"), indent=1, default=str)

    print()
    print("%-10s %4s %6s %6s %7s %7s %8s %9s %10s" %
          ("source", "cand", "contam", "dupe", "decided", "refuted", "ref_rate",
           "accepted", "tok/acc"))
    for r in rows:
        acc = r["accepted"] or 0
        print("%-10s %4d %6d %6d %7d %7d %8s %9d %10s" %
              (r["source"], r["candidates"], r["rejected_contaminated"],
               r["rejected_duplicate"], r["decided"], r["refuted"],
               ("%.4f" % r["refuted_key_rate"]) if r["refuted_key_rate"] is not None else "n/a",
               acc, ("%.0fk" % (r["cost_tokens"] / max(acc, 1) / 1000)) if acc else "n/a"))
    for r in rows:
        if not r["ok"]:
            print("  SOURCE FAILURE %-10s: %s" % (r["source"], r["failure_reason"]))
        if r["duplicate_of"]:
            print("  %-10s cross-source duplicates lost to: %s" % (r["source"], r["duplicate_of"]))
    print()
    print("held-out contamination: max similarity seen per source: %s"
          % {r["source"]: r["max_contamination_sim"] for r in rows})

    est = HostedTeacher("a-hosted-frontier-teacher", price_in_per_mtok=3.0,
                        price_out_per_mtok=15.0)
    gen_tok = stats["distilled"].cost_tokens
    n_calls = a.n_per_source * 4
    print()
    print("HOSTED-TEACHER COST ESTIMATE (nothing was sent; priced from THIS run's measured "
          "token counts, at illustrative $3/$15 per Mtok):")
    print("  ", est.estimate_cost(n_prompts=n_calls, prompt_tokens=200,
                                  output_tokens=int(gen_tok / max(n_calls, 1))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
