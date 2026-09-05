#!/usr/bin/env python3
"""Assign a fine-grained topic to every problem, refusing to guess when samples disagree.

The coarse four subfields cannot separate weakness: three of the four sit inside each other's
intervals. Finer topics give the ranking something to discriminate, and a LABEL is preferred
over a cluster index for a safety reason rather than a cosmetic one -- a label is auditable
against the exemplar constraint, because "weak at pigeonhole" can be checked against the
exemplars actually shown, whereas "weak at cluster seven" gives an auditor nothing.

A controlled vocabulary is used rather than free-form topics so that groups are comparable
across problems and so that a label cannot drift into a paraphrase of the problem itself.
Unanimity across samples is required; disagreement yields `unlabelled`, which is a first-class
outcome and is pooled rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from ornith_repro.llm import OpenAICompatClient

TOPICS = [
    "divisibility", "modular_arithmetic", "primes_and_factorisation", "diophantine",
    "digits_and_bases", "polynomials", "equations_and_systems", "inequalities",
    "sequences_and_series", "functional_equations", "logs_and_exponents",
    "counting", "pigeonhole", "graph_theory", "recursion", "probability",
    "generating_functions", "triangles", "circles", "coordinate_geometry",
    "solid_geometry", "trigonometry",
]

PROMPT = """Classify this competition mathematics problem by its MAIN technique.

PROBLEM: {problem}

Choose exactly one label from this list:
{topics}

If none fits well, reply `unlabelled`. Do not solve the problem.
Reply with exactly one label on the first line and nothing else.
"""


def parse_label(text: str) -> str:
    """Read a reply, accepting only an exact known label.

    Args:
        text: Raw completion.

    Returns:
        A known topic, or `unlabelled`.
    """
    if not text:
        return "unlabelled"
    body = text.split("</think>")[-1].strip()
    first = body.splitlines()[0].strip().lower().strip(".,`\"'` ") if body else ""
    return first if first in TOPICS else "unlabelled"


def main():
    """Label every problem and print the topic histogram."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--problems", default="/home/ubuntu/reach/data/olympiadbench/test.jsonl")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/fine_labels.jsonl")
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
    print("problems %d, %d samples each, unanimity required" % (len(rows), a.samples),
          flush=True)
    topics_block = "\n".join("  " + t for t in TOPICS)

    def work(i, row):
        """Label one problem, requiring every sample to agree."""
        client = OpenAICompatClient(a.base_url, a.model, ids, ctx or 65536,
                                    reasoning_effort=a.effort)
        labels = []
        for s in range(a.samples):
            try:
                text, trunc = client.generate(
                    PROMPT.format(problem=row["question"], topics=topics_block),
                    a.max_tokens, 6100 + s)
            except Exception:  # noqa: BLE001
                return {"idx": i, "label": "unlabelled", "detail": "request failed"}
            if trunc:
                return {"idx": i, "label": "unlabelled", "detail": "truncated"}
            labels.append(parse_label(text))
        top, n = Counter(labels).most_common(1)[0]
        if n == len(labels) and top != "unlabelled":
            return {"idx": i, "label": top, "detail": "unanimous"}
        return {"idx": i, "label": "unlabelled",
                "detail": "no consensus %s" % labels, "majority": top}

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

    c = Counter(x["label"] for x in out)
    print("")
    print("=== fine topic histogram (n=%d) ===" % len(out))
    for k, v in c.most_common():
        print("  %-26s %4d  %.4f" % (k, v, v / len(out)))
    print("  distinct labels used: %d of %d" % (len([k for k in c if k != "unlabelled"]),
                                                len(TOPICS)))
    print("  elapsed %.0fs, wrote %s" % (time.time() - t0, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
