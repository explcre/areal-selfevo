#!/usr/bin/env python3
"""Does the proposer terminate at any affordable budget, and what do generated tasks cost?

The loop's first live iteration refused 22 of 24 proposals as truncated at a 16384 cap and
then refused a task for having 1 gradeable rollout of 18 at a 4096 rollout cap. Both are
budget questions and both are measured here before either cap is fixed.
"""
import asyncio
import json
import random
import sys
from collections import Counter

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")
from transformers import AutoTokenizer  # noqa: E402
import ornith_train as ot  # noqa: E402
from ornith_repro import live as ol  # noqa: E402

URL = "http://127.0.0.1:30050"
tok = AutoTokenizer.from_pretrained("/mnt/localssd/gate/models/Qwen3.8-27B")
pool = json.load(open("/mnt/localssd/gate/out/pool_cap8192.json"))["tasks"]
solved = [t["problem"] for t in pool if (t["c_a"] + t["c_b"]) >= (t["n_a"] + t["n_b"]) - 1]
unsolved = [t["problem"] for t in pool if (t["c_a"] + t["c_b"]) <= 1]
rng = random.Random(23)

ps = [ol.PROPOSER_PROMPT.format(
        solved="\n".join("- " + s[:400] for s in rng.sample(solved, 3)),
        unsolved="\n".join("- " + s[:400] for s in rng.sample(unsolved, 3)),
        novelty="") for _ in range(12)]
r = asyncio.run(ot.gen_batch(URL, tok, ps, 22528, 12, "ORN"))
L = sorted(len(x["output_ids"]) for x in r)
res = [ol.parse_proposal(x["text"])[2] for x in r]
print("PROPOSER @22528: hit_cap=%d/%d parse_ok=%d" % (
    sum(x["hit_cap"] for x in r), len(r), sum(1 for v in res if v == "ok")), flush=True)
print("  lengths: %s" % L, flush=True)
print("  reasons: %s" % Counter(res), flush=True)

tasks = [ol.parse_proposal(x["text"])[0] for x in r if ol.parse_proposal(x["text"])[2] == "ok"]
print("  accepted: %d" % len(tasks), flush=True)
if tasks:
    sp = [ol.SCAFFOLD_PROMPT.format(task=t) for t in tasks[:2]]
    sc = asyncio.run(ot.gen_batch(URL, tok, sp, 16384, 2, "ORN"))
    inst = [(x["text"].split("</think>")[-1].strip() or "Solve the problem.") for x in sc]
    rp = []
    for t, i in zip(tasks[:2], inst):
        rp += [ol.SOLVER_PROMPT.format(instructions=i[:2000], task=t)] * 8
    ro = asyncio.run(ot.gen_batch(URL, tok, rp, 16384, 16, "ORN"))
    RL = sorted(len(x["output_ids"]) for x in ro)
    boxed = sum(1 for x in ro if ot.gate_lib.math_bench.extract_boxed(x["text"]) is not None)
    print("SOLVER on GENERATED tasks: n=%d median=%d p90=%d max=%d hit_cap@16384=%d boxed=%d"
          % (len(RL), RL[len(RL) // 2], RL[int(.9 * len(RL))], RL[-1],
             sum(x["hit_cap"] for x in ro), boxed), flush=True)
    for cap in (4096, 8192, 12288, 16384):
        lost = sum(1 for x in ro if len(x["output_ids"]) >= cap)
        print("   at cap %5d: %d/%d rollouts would be cut off" % (cap, lost, len(ro)), flush=True)
