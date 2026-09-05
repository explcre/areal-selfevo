#!/usr/bin/env python3
"""Run the wired verifier over the generated task set and report the verdict breakdown.

This produces the first honest measurement of gold quality on self-generated tasks: what
fraction of the proposer's own answer keys survive independent checking. It also decides
whether a Lean toolchain is worth installing, by showing WHY the unverifiable ones are
unverifiable. If executable enumeration with a justified bound settles most of them, Lean is
a small marginal gain; if a large share is out of enumeration's reach, that is the
quantitative argument for it.

Concurrency is a thread pool rather than asyncio, because `SoundVerifier` and its backends are
synchronous by design -- the sandboxed subprocess in the middle of each check is not
awaitable, and rewriting the verifier to be async would have meant maintaining two copies of
the logic that decides verdicts.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from ornith_repro.llm import OpenAICompatClient
from ornith_repro.verify import Verdict
from ornith_repro.verify_sound import SoundVerifier


def load_tasks(paths, limit):
    """Collect admitted generated tasks with their asserted keys.

    Args:
        paths: JSONL files from the generators. Rows are accepted when they carry a problem
            and an answer and were not rejected.
        limit: Maximum tasks to return (0 = all).

    Returns:
        List of {problem, answer, source} dicts, de-duplicated by problem text.
    """
    seen, out = set(), []
    for path in paths:
        try:
            fh = open(path)
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                prob, ans = r.get("problem"), r.get("answer")
                if not prob or not ans:
                    continue
                if r.get("admitted") is False or r.get("validity") not in (None, "ok"):
                    continue
                key = " ".join(prob.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({"problem": prob, "answer": ans, "source": path.split("/")[-1]})
    return out[:limit] if limit else out


def classify_reason(detail: str) -> str:
    """Bucket an UNVERIFIABLE detail into why it could not be settled.

    The buckets are what decide whether a proof assistant would help: a problem enumeration
    cannot express is a Lean argument, whereas a program that merely crashed is not.

    Args:
        detail: The verifier's detail string.

    Returns:
        A short reason label.
    """
    d = detail.lower()
    if "bound unjustified" in d:
        return "bound_unjustified"
    if "bound falsified" in d:
        return "bound_falsified"
    if "no usable artifact" in d and "primary" in d:
        return "no_program_produced"
    if "second independent program" in d:
        return "no_second_program"
    if "programs disagree" in d:
        return "programs_disagree"
    if "round-trip different" in d:
        return "round_trip_mismatch"
    if "round-trip" in d or "comparison inconclusive" in d:
        return "round_trip_unclear"
    if "not substantiated" in d:
        return "witness_rejected"
    if "no witness" in d:
        return "no_witness"
    if "bound-stability" in d:
        return "bound_rerun_failed"
    return "other"


def main():
    """Verify every generated task and print the breakdown."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tasks", nargs="+",
                    default=["/mnt/localssd/gate/out/gen_novel.jsonl",
                             "/mnt/localssd/gate/out/gen_tasks_main.jsonl"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/verify_generated.jsonl")
    a = ap.parse_args()

    r = httpx.get(a.base_url.rstrip("/") + "/models", timeout=60.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = [m["id"] for m in data]
    ctx = max((m.get("max_model_len") or 0) for m in data) if data else 0
    if a.model not in ids:
        print("FATAL: %r not served; served=%s" % (a.model, ids))
        return 2
    print("served-model check OK: %s (context %d)" % (a.model, ctx), flush=True)

    tasks = load_tasks(a.tasks, a.limit)
    print("tasks to verify: %d" % len(tasks), flush=True)
    if not tasks:
        return 1

    def work(t):
        """Verify one task with its own client."""
        client = OpenAICompatClient(a.base_url, a.model, ids, ctx or 65536,
                                    reasoning_effort=a.effort)
        v = SoundVerifier(client, primary_name=a.model, secondary_name=a.model,
                          timeout=a.timeout)
        try:
            res = v.verify(t["problem"], t["answer"])
        except Exception as exc:  # noqa: BLE001
            return {**t, "verdict": "ERROR", "detail": repr(exc)[:200], "computed": None}
        return {**t, "verdict": res.verdict.value, "detail": res.detail,
                "computed": res.computed}

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(work, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0 or i == len(futs):
                print("  %d/%d  %.2f/s" % (i, len(futs), i / (time.time() - t0)), flush=True)
                with open(a.out, "w") as fh:
                    for x in rows:
                        fh.write(json.dumps(x) + "\n")

    with open(a.out, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")

    counts = Counter(x["verdict"] for x in rows)
    n = len(rows)
    print("")
    print("=== verdicts over %d generated tasks ===" % n)
    for k in ("verified", "refuted", "unverifiable", "ERROR"):
        if counts.get(k):
            print("  %-14s %4d  %.4f" % (k, counts[k], counts[k] / n))
    dec = counts.get("verified", 0) + counts.get("refuted", 0)
    print("  coverage (decisive) : %.4f" % (dec / n))
    if dec:
        print("  refuted among decided: %.4f"
              % (counts.get("refuted", 0) / dec))
    print("")
    print("  why unverifiable:")
    for reason, c in Counter(classify_reason(x["detail"]) for x in rows
                             if x["verdict"] == "unverifiable").most_common():
        print("    %-22s %4d" % (reason, c))
    print("")
    print("  elapsed %.0fs, wrote %s" % (time.time() - t0, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
