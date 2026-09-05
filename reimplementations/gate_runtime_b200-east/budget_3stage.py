#!/usr/bin/env python3
"""Measure the token budget of all three stages ON THIS CONFIGURATION before fixing any cap.

Inheriting a cap has overturned this configuration four times: a 1024 proposer cap rejected
every proposal as truncated, a 6144 server context capped generations at ~5.4k while reporting
itself as the generation cap, a rollout cap turned a difficulty estimate into a budget
measurement, and a context guard read the KV pool size instead of the context length. So every
cap the loop will use is measured here first, at a ceiling high enough not to bind, and the
distribution is reported rather than a single number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import ornith_train as ot  # noqa: E402
from ornith_repro import live as ol  # noqa: E402


def dist(lengths, cap):
    """Length distribution and the share that ran into the ceiling."""
    if not lengths:
        return {}
    s = sorted(lengths)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": q(0.5), "p90": q(0.9), "max": s[-1],
            "at_ceiling": round(sum(1 for x in s if x >= cap - 8) / len(s), 4),
            "mean": round(sum(s) / len(s))}


def main() -> int:
    """Measure proposer, scaffold and rollout budgets and print recommended caps."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--pool", default="/mnt/localssd/gate/out/pool_cap8192.json")
    ap.add_argument("--ceiling", type=int, default=16384,
                    help="deliberately generous: a cap must be chosen from a distribution "
                         "measured where the cap is not the binding constraint")
    ap.add_argument("--rollout-ceiling", type=int, default=8192)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--only", default="",
                    help="measure only this stage (e.g. proposer)")
    ap.add_argument("--out", default="/mnt/localssd/gate/out/budget_3stage.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ctx = ot.assert_context_fits(a.url, a.ceiling)
    print("context: %s" % ctx, flush=True)

    pool = json.load(open(a.pool))["tasks"]
    solved = [t["problem"] for t in pool if (t["c_a"] + t["c_b"]) >= (t["n_a"] + t["n_b"]) - 1]
    unsolved = [t["problem"] for t in pool if (t["c_a"] + t["c_b"]) <= 1]
    print("exemplars: %d solved, %d unsolved" % (len(solved), len(unsolved)), flush=True)

    out = {"context": ctx, "ceiling": a.ceiling, "rollout_ceiling": a.rollout_ceiling}

    # ---- stage 1: proposer ---------------------------------------------------------
    pp = [ol.PROPOSER_PROMPT.format(
        solved="\n".join("- " + s[:400] for s in rng.sample(solved, min(3, len(solved)))),
        unsolved="\n".join("- " + s[:400] for s in rng.sample(unsolved, min(3, len(unsolved)))),
        novelty="") for _ in range(a.n)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, pp, a.ceiling, a.n, ""))
    reasons = Counter()
    ok = 0
    for r in recs:
        if r["hit_cap"]:
            reasons["truncated"] += 1
            continue
        _, _, why = ol.parse_proposal(r["text"])
        reasons[why] += 1
        ok += why == "ok"
    out["proposer"] = dist([len(r["output_ids"]) for r in recs], a.ceiling)
    out["proposer"]["parse_yield"] = round(ok / len(recs), 4)
    out["proposer"]["reasons"] = dict(reasons)
    print("proposer: %s" % out["proposer"], flush=True)

    if a.only == "proposer":
        json.dump(out, open(a.out, "w"), indent=1)
        return 0

    # ---- stage 2: harness ----------------------------------------------------------
    tasks = rng.sample(pool, min(a.n, len(pool)))
    sp = [ol.SCAFFOLD_PROMPT.format(task=t["problem"]) for t in tasks]
    recs = asyncio.run(ot.gen_batch(a.url, tok, sp, a.ceiling, a.n, ""))
    out["harness"] = dist([len(r["output_ids"]) for r in recs], a.ceiling)
    print("harness: %s" % out["harness"], flush=True)

    # ---- stage 3: solver -----------------------------------------------------------
    rp = [ol.SOLVER_PROMPT.format(instructions="Work carefully. Put the final answer in "
                                               "\\boxed{}.", task=t["problem"])
          for t in tasks for _ in range(2)]
    recs = asyncio.run(ot.gen_batch(a.url, tok, rp, a.rollout_ceiling, len(rp), ""))
    out["solver"] = dist([len(r["output_ids"]) for r in recs], a.rollout_ceiling)
    out["solver"]["unanswered"] = round(
        sum(1 for r in recs if ot.unanswered(r)) / len(recs), 4)
    print("solver: %s" % out["solver"], flush=True)

    # A cap is chosen at the p90 rounded up, so the common case is never truncated and the
    # tail is; the alternative -- the max -- pays for the worst case on every generation.
    def pick(d, ceiling):
        p90 = d.get("p90", 0)
        return min(ceiling, 1024 * ((p90 + 1023) // 1024))
    out["recommended"] = {"gen_cap": max(pick(out["proposer"], a.ceiling),
                                         pick(out["harness"], a.ceiling)),
                          "rollout_cap": pick(out["solver"], a.rollout_ceiling)}
    print("\nrecommended: %s" % out["recommended"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
