#!/usr/bin/env python3
"""Does a generated scaffold preserve the difficulty the seed was chosen for?

Seeding the loop from the mixed subset is supposed to hand it tasks at a KNOWN success rate.
But the loop does not solve a task bare: the harness stage writes a scaffold first, and the
solver is conditioned on it. If the scaffold moves the success rate, the seed's guarantee does
not survive to the rollout stage and seeding cannot do what it was introduced to do.

Observed in the seeded arm's first iteration and the reason this exists: one seeded task
scored k = [0, 0, 4] across its three scaffolds -- two scaffolds never solved it and one solved
it every time -- and another scored [4, 4, 4] though it was drawn from the mixed subset
precisely because the base does NOT always solve it.

Measured here on the BASE model, so the comparison is against the same policy the pool's
p-hat was measured with.
"""
import asyncio
import json
import random
import statistics
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")
from transformers import AutoTokenizer  # noqa: E402
import ornith_train as ot  # noqa: E402
from ornith_repro import live as ol  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30062"
N_TASKS, N_SCAF, N_ROLL, CAP = 8, 3, 4, 8192
tok = AutoTokenizer.from_pretrained("/mnt/localssd/gate/models/Qwen3.8-27B")
pool = json.load(open("/mnt/localssd/gate/out/pool_cap8192.json"))["tasks"]
rng = random.Random(31)
picked = rng.sample(pool, N_TASKS)

sp = [ol.SCAFFOLD_PROMPT.format(task=t["problem"]) for t in picked for _ in range(N_SCAF)]
sc = asyncio.run(ot.gen_batch(URL, tok, sp, 8192, len(sp), ""))
inst = [(x["text"].split("</think>")[-1].strip() or "Solve the problem.")[:2000] for x in sc]

rp, owner = [], []
for i, t in enumerate(picked):
    for j in range(N_SCAF):
        p = ol.SOLVER_PROMPT.format(instructions=inst[i * N_SCAF + j], task=t["problem"])
        for _ in range(N_ROLL):
            rp.append(p)
            owner.append((i, j))
ro = asyncio.run(ot.gen_batch(URL, tok, rp, CAP, min(len(rp), 96), ""))

print("%-6s %8s %10s   %s" % ("task", "pool_p", "scaffolded", "k per scaffold (of %d)" % N_ROLL))
drift, k_all = [], []
for i, t in enumerate(picked):
    pool_p = (t["c_a"] + t["c_b"]) / (t["n_a"] + t["n_b"])
    ks = []
    for j in range(N_SCAF):
        mine = [ro[m] for m in range(len(ro)) if owner[m] == (i, j)]
        k = sum(1 for x in mine
                if ot.gate_lib.math_bench.grade(x["text"], t["answer"]))
        ks.append(k)
    got = sum(ks) / (N_SCAF * N_ROLL)
    drift.append(got - pool_p)
    k_all += ks
    print("%-6d %8.3f %10.3f   %s" % (t["idx"], pool_p, got, ks))

print()
print("mean |drift| from the pool's measured p-hat: %.3f  (signed mean %+.3f)"
      % (statistics.fmean(abs(d) for d in drift), statistics.fmean(drift)))
deg = sum(1 for k in k_all if k in (0, N_ROLL))
print("scaffold groups unanimous: %d of %d (%.3f)" % (deg, len(k_all), deg / len(k_all)))
spread = [max(ks) - min(ks) for ks in
          [k_all[i * N_SCAF:(i + 1) * N_SCAF] for i in range(N_TASKS)]]
print("within-task spread across scaffolds (max k - min k): %s, mean %.2f of %d"
      % (spread, statistics.fmean(spread), N_ROLL))
tasks_all_unanimous = sum(1 for i in range(N_TASKS)
                          if all(k in (0, N_ROLL) for k in k_all[i * N_SCAF:(i + 1) * N_SCAF]))
print("tasks whose EVERY rollout group was unanimous (guard G4 would refuse): %d of %d"
      % (tasks_all_unanimous, N_TASKS))
