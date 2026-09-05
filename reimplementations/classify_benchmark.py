#!/usr/bin/env python3
"""Classify every OlympiadBench problem and report how much is in executable reach.

The histogram is the cheap measurement that decides whether a full verification pass over the
benchmark is worth running: it says what fraction of competition mathematics is even
addressable by enumeration or a computer algebra system before a single answer is checked.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from ornith_repro.classify import ProblemClass, backend_for, classify
from ornith_repro.llm import OpenAICompatClient


def main():
    """Classify the benchmark and print the class and backend histograms."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--problems", default="/home/ubuntu/reach/data/olympiadbench/test.jsonl")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/classify_olympiadbench.jsonl")
    a = ap.parse_args()

    r = httpx.get(a.base_url.rstrip("/") + "/models", timeout=60.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = [m["id"] for m in data]
    ctx = max((m.get("max_model_len") or 0) for m in data) if data else 0
    if a.model not in ids:
        print("FATAL: %r not served; served=%s" % (a.model, ids))
        return 2
    print("served-model check OK: %s" % a.model, flush=True)

    rows = [json.loads(l) for l in open(a.problems) if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print("problems: %d   samples per problem: %d" % (len(rows), a.samples), flush=True)

    def work(i, row):
        """Classify one problem with its own client."""
        client = OpenAICompatClient(a.base_url, a.model, ids, ctx or 65536,
                                    reasoning_effort=a.effort)
        try:
            cls, detail = classify(client, row["question"], samples=a.samples)
        except Exception as exc:  # noqa: BLE001
            return {"idx": i, "cls": ProblemClass.UNCLASSIFIABLE.value,
                    "detail": "error: %r" % (exc,), "subfield": row.get("subfield")}
        return {"idx": i, "cls": cls.value, "detail": detail,
                "backend": backend_for(cls), "subfield": row.get("subfield")}

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(work, i, r) for i, r in enumerate(rows)]
        for k, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if k % 50 == 0 or k == len(futs):
                print("  %d/%d  %.2f/s" % (k, len(futs), k / (time.time() - t0)), flush=True)
                with open(a.out, "w") as fh:
                    for x in out:
                        fh.write(json.dumps(x) + "\n")

    with open(a.out, "w") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")

    n = len(out)
    counts = Counter(x["cls"] for x in out)
    print("")
    print("=== OlympiadBench problem classes (n=%d, unanimity over %d samples) ===" % (n, a.samples))
    for c in ProblemClass:
        k = counts.get(c.value, 0)
        if k:
            print("  %-22s %4d  %.4f  -> %s"
                  % (c.value, k, k / n, backend_for(c) or "no executable backend"))
    reach = sum(counts.get(c.value, 0) for c in ProblemClass if backend_for(c))
    print("")
    print("  in executable reach      : %4d  %.4f" % (reach, reach / n))
    print("  proof/construction only  : %4d  %.4f"
          % (counts.get("proof_or_construction", 0),
             counts.get("proof_or_construction", 0) / n))
    print("  classifier declined      : %4d  %.4f"
          % (counts.get("unclassifiable", 0), counts.get("unclassifiable", 0) / n))
    nocons = sum(1 for x in out if "no consensus" in x.get("detail", ""))
    trunc = sum(1 for x in out if "truncated" in x.get("detail", ""))
    print("  of which no consensus    : %4d" % nocons)
    print("  of which truncated       : %4d  (mechanical, not a real refusal)" % trunc)
    print("")
    print("  elapsed %.0fs, wrote %s" % (time.time() - t0, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
