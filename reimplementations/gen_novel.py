#!/usr/bin/env python3
"""Generate a task set with the novelty term LIVE, concurrently.

The earlier generated sample repeated itself -- three of eight were digit-sum/product
variants -- and that was OUR defect, not Ornith's: `N(q) = 1 - max_j sim(q, q_j)` exists
precisely to stop the proposer circling, and it had no production caller. This runs the
proposer with the buffer both CONDITIONING it (the prompt shows recent tasks) and SCORING it
(a candidate too close to the buffer is rejected), which is what their design specifies.

Concurrency without breaking novelty: novelty is defined against a buffer that grows, so a
fully parallel run would score every candidate against an empty buffer and defeat the term.
Candidates are therefore admitted IN ARRIVAL ORDER against the buffer as it stands, which
preserves the sequential semantics of `N` while allowing generation to overlap freely.

A first version issued fixed WAVES and gathered each one. That drained the machine at every
wave boundary -- a wave is only finished when its slowest member is, so utilisation decayed
through each tail and fell to two cards of eight. This version keeps a fixed number of
requests permanently in flight instead, which is the difference between a busy accelerator and
an idle one. A restart reseeds the buffer from the admitted tasks already on disk, so stopping
and resuming does not silently reset novelty and re-admit duplicates.

Every candidate is recorded with its realised `N(q)` whether admitted or rejected, so
"did wiring novelty actually remove the repetition?" is answerable from the artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp

from ornith_repro.live import PROPOSER_PROMPT, parse_proposal
from ornith_repro.rewards import jaccard_similarity, novelty_reward


async def propose(session, url, model, prompt, effort, max_tokens, timeout):
    """One proposal from the served model, retrying transient failures.

    Args:
        session: aiohttp session.
        url: Chat-completions endpoint.
        model: Served model id, checked by the caller.
        prompt: Filled proposer prompt.
        effort: reasoning_effort, held identical to the rollout arms.
        max_tokens: Generation cap.
        timeout: Per-request timeout.

    Returns:
        dict with text, finish_reason, status.
    """
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 1.0, "top_p": 0.95, "max_tokens": max_tokens,
               "chat_template_kwargs": {"reasoning_effort": effort}}
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=timeout) as r:
                if r.status != 200:
                    await asyncio.sleep(2 + attempt)
                    continue
                d = await r.json()
                ch = (d.get("choices") or [{}])[0]
                return {"text": (ch.get("message") or {}).get("content") or "",
                        "finish_reason": ch.get("finish_reason"),
                        "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
                        "status": "ok"}
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2 + attempt)
    return {"text": "", "finish_reason": None, "completion_tokens": None, "status": "failed"}


def load_competence(blocks, problems, min_samples=8):
    """Always-solved and never-solved exemplars from a measured pool.

    Args:
        blocks: Rollout JSONL with per-problem outcomes.
        problems: The problem file those outcomes index.
        min_samples: Minimum resolved samples to classify a problem.

    Returns:
        (always_solved, never_solved) statements.
    """
    from collections import defaultdict
    got = defaultdict(list)
    for line in open(blocks):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") == "ok" and r.get("finish_reason") != "length":
            got[r["idx"]].append(1 if r.get("correct") else 0)
    rows = [json.loads(l) for l in open(problems) if l.strip()]
    always, never = [], []
    for i, row in enumerate(rows):
        v = got.get(i, [])
        if len(v) < min_samples:
            continue
        if sum(v) == len(v):
            always.append(row["question"])
        elif sum(v) == 0:
            never.append(row["question"])
    return always, never


async def run(a):
    """Generate until `target` novel tasks are admitted, or the attempt budget runs out."""
    solved, unsolved = load_competence(a.blocks, a.problems)
    print("exemplars: %d always-solved, %d never-solved" % (len(solved), len(unsolved)),
          flush=True)
    url = a.base_url.rstrip("/") + "/chat/completions"
    fh = open(a.out, "a")
    buffer: list[str] = []
    admitted = rejected = 0
    reasons: dict = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("admitted") and rec.get("problem"):
                buffer.append(rec["problem"])
                admitted += 1
            else:
                rejected += 1
        print("resumed: %d admitted already on disk, buffer seeded" % admitted, flush=True)
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.get(a.base_url.rstrip("/") + "/models",
                               timeout=aiohttp.ClientTimeout(total=60)) as r:
            served = [m["id"] for m in (await r.json()).get("data", [])]
        if a.model not in served:
            print("FATAL: %r not served; served=%s" % (a.model, served), flush=True)
            return 2
        print("served-model check OK: %s" % a.model, flush=True)

        state = {"issued": 0, "admitted": admitted, "rejected": rejected}
        lock = asyncio.Lock()

        def build_prompt():
            """Prompt conditioned on the buffer AS IT STANDS when the request is issued."""
            recent = buffer[-5:]
            ctx = ("Do NOT repeat or lightly reword any of these already-used tasks:\n"
                   + "\n".join("- " + t[:200] for t in recent)) if recent else ""
            return PROPOSER_PROMPT.format(
                solved="\n".join("- " + x[:400] for x in solved[:3]),
                unsolved="\n".join("- " + x[:400] for x in unsolved[:2]),
                novelty=ctx)

        async def worker():
            """Keep one request in flight until the target or the attempt budget is met."""
            while True:
                async with lock:
                    if state["admitted"] >= a.target or state["issued"] >= a.max_attempts:
                        return
                    state["issued"] += 1
                    prompt = build_prompt()
                res = await propose(session, url, a.model, prompt, a.effort, a.max_tokens,
                                    aiohttp.ClientTimeout(total=a.timeout))
                problem, answer, why = parse_proposal(res["text"])
                async with lock:
                    if why != "ok":
                        reasons[why] = reasons.get(why, 0) + 1
                        state["rejected"] += 1
                        fh.write(json.dumps({"admitted": False, "reason": why, "N": None,
                                             "problem": None, "answer": None,
                                             "finish_reason": res["finish_reason"]}) + "\n")
                    else:
                        # N(q) against the buffer as it stands, before this task joins it.
                        N, empty = novelty_reward(problem, buffer, sim=jaccard_similarity)
                        keep = empty or N >= a.min_novelty
                        fh.write(json.dumps(
                            {"admitted": bool(keep),
                             "reason": "ok" if keep else "below_novelty",
                             "N": N, "problem": problem, "answer": answer,
                             "finish_reason": res["finish_reason"],
                             "completion_tokens": res["completion_tokens"]}) + "\n")
                        if keep:
                            buffer.append(problem)
                            state["admitted"] += 1
                        else:
                            reasons["below_novelty"] = reasons.get("below_novelty", 0) + 1
                            state["rejected"] += 1
                    fh.flush()
                    if state["issued"] % 50 == 0:
                        print("  issued %d  admitted %d  rejected %d  %.2f/s"
                              % (state["issued"], state["admitted"], state["rejected"],
                                 state["issued"] / max(time.time() - t0, 1e-9)), flush=True)
                        subprocess.run(["gsutil", "-q", "cp", a.out, a.gs], timeout=600,
                                       check=False)

        await asyncio.gather(*[worker() for _ in range(a.inflight)])
        admitted, rejected = state["admitted"], state["rejected"]

    fh.close()
    subprocess.run(["gsutil", "-q", "cp", a.out, a.gs], timeout=600, check=False)
    print("DONE admitted %d, rejected %d; rejections %s" % (admitted, rejected, reasons),
          flush=True)
    return 0


def main():
    """Parse arguments and generate a novelty-filtered task set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--blocks", default="/mnt/localssd/gate/out/blocks_low.jsonl")
    ap.add_argument("--problems",
                    default="/mnt/localssd/gate/searchhalf/olympiadbench/test.jsonl")
    ap.add_argument("--target", type=int, default=338)
    ap.add_argument("--max-attempts", type=int, default=900)
    ap.add_argument("--inflight", type=int, default=96,
                    help="requests kept permanently in flight")
    ap.add_argument("--min-novelty", type=float, default=0.60)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/gen_novel.jsonl")
    ap.add_argument("--gs", default="gs://selfevo/runs/h200/gate/")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
