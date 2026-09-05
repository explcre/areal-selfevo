#!/usr/bin/env python3
"""Sample k completions per problem for one half of OlympiadBench, and grade them.

Used three times with different arguments and no other difference: to measure the completion
LENGTH distribution before a cap is chosen, to establish each problem's success rate p-hat
(from which the mixed subset is derived), and to score a checkpoint on the held-out half.

The split is the committed one; its dataset md5 is verified on load by ``math_bench.load``'s
split path, so a different 675-row file cannot silently address different problems.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer  # noqa: E402

import gate_lib  # noqa: E402
from gate_lib import GenSpec, build_prompt, generate_many  # noqa: E402


def main() -> int:
    """Generate, then write one JSONL row per (problem, repetition)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30020")
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--split", default="search", choices=["search", "report", "all"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lora", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (probes)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth problem; with --limit gives a spread sample rather "
                         "than the easy end, since OlympiadBench is ordered by difficulty")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    probs = gate_lib.math_bench.load("olympiadbench", a.split)
    if a.stride > 1:
        probs = probs[:: a.stride]
    if a.limit:
        probs = probs[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)

    jobs = []
    for p in probs:
        prompt = build_prompt(tok, p["problem"], a.effort)
        for rep in range(a.k):
            jobs.append({"idx": p["idx"], "rep": rep, "answer": p["answer"],
                         "prompt": prompt})
    spec = GenSpec(max_new_tokens=a.cap, temperature=a.temperature, top_p=a.top_p,
                   effort=a.effort, lora_path=a.lora)
    print("problems=%d k=%d jobs=%d split=%s cap=%d effort=%s lora=%r"
          % (len(probs), a.k, len(jobs), a.split, a.cap, a.effort, a.lora), flush=True)
    asyncio.run(generate_many(a.url, jobs, spec, a.concurrency, a.out))

    # Grade in a second pass over the completed file, so a resumed run grades everything
    # exactly once and a partial file is never graded at all.
    n = 0
    rows = []
    with open(a.out) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(r)
    graded = a.out + ".graded.jsonl"
    with open(graded, "w") as fh:
        for r in rows:
            truncated = r.get("finish") == "length"
            r["truncated"] = truncated
            r["error"] = r.get("finish") == "error"
            # A truncated generation is a budget artefact, not a wrong answer. It is graded
            # anyway (a completion can carry a box before the cap) but the flag travels with
            # it so p-hat can be computed on RESOLVED samples only.
            r["correct"] = bool(gate_lib.math_bench.grade(r.get("text", ""), r["answer"]))
            r.pop("text", None)
            fh.write(json.dumps(r) + "\n")
            n += 1
    print("graded %d rows -> %s" % (n, graded), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
