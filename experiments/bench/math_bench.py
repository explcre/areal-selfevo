#!/usr/bin/env python3
"""Score a served model on the frozen math suite.

Generation goes through our sglang OpenAI-compatible endpoint; grading uses `math_verify`,
which is what AZR's own `custom_evaluate.py` imports (`from math_verify import parse,
verify`). So this reuses their grading approach rather than substituting a different one --
the alternative, their `grader.py`, needs a `latex2sympy` that ships only as an unextracted
tarball.

Why not their runner: `math_eval/eval/math_eval.py` does `from vllm import LLM`, and vLLM
is not installed here. Installing it risks the torch/sglang environment, and the runner is
the easy half -- the grader is the part worth reusing.

Datasets come from the pinned AZR clone (MIT). Only benchmarks with a uniform
`problem`/`answer` schema are included; olympiadbench (`question`/`final_answer`) and
minerva_math (answer embedded in `solution`) need their own extraction and are deliberately
left out rather than half-supported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

DATA = Path(
    "/home/ubuntu/baselines/Absolute-Zero-Reasoner/evaluation/math_eval/eval/data"
)
# Uniform problem/answer schema, verified by inspection.
SUITE = ["aime24", "aime25", "amc23", "math500", "hmmt_2024", "hmmt_2025", "livemathbench"]

PROMPT = (
    "Solve the following math problem. Reason step by step, and put your final answer "
    "within \\boxed{{}}.\n\n{problem}"
)


def load(bench: str) -> list[dict]:
    """Load a benchmark's problems.

    Raises:
        FileNotFoundError: If the benchmark is absent, rather than silently scoring zero
            problems -- an empty suite would otherwise report a confident 0.0.
    """
    f = DATA / bench / "test.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{bench}: {f} not found")
    rows = [json.loads(l) for l in f.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        if "problem" not in r or "answer" not in r:
            raise ValueError(f"{bench} row {i}: expected problem/answer, got {sorted(r)}")
        out.append({"problem": r["problem"], "answer": str(r["answer"])})
    if not out:
        raise ValueError(f"{bench}: no problems loaded")
    return out


_BOXED = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...}, brace-balanced.

    A regex like `\\\\boxed\\{([^}]*)\\}` truncates at the first inner brace, which silently
    mangles answers such as `\\boxed{\\frac{1}{2}}` into `\\frac{1`. Balanced scanning
    matters more here than it looks: a mangled extraction grades as wrong, so the bug
    would show up as a plausible-looking lower score rather than an error.
    """
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, buf = 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
        buf.append(c)
        i += 1
    return None  # unbalanced


def grade(pred_text: str, gold: str) -> bool:
    """True when the prediction matches the gold answer.

    Uses math_verify (symbolic), falling back to a normalised string compare only when
    math_verify cannot parse either side.
    """
    boxed = extract_boxed(pred_text)
    if boxed is None:
        return False
    try:
        from math_verify import parse, verify

        g, p = parse(f"${gold}$"), parse(f"${boxed}$")
        if g and p:
            return bool(verify(g, p))
    except Exception:
        pass
    norm = lambda s: re.sub(r"[\s,$]|\\left|\\right|\\!|\\,", "", s).rstrip(".")
    return norm(boxed) == norm(gold)


async def generate(session, url: str, model: str, prompt: str, args) -> str:
    """One completion. Returns '' on failure rather than raising, but failures are counted
    by the caller so they cannot be mistaken for wrong answers."""
    import aiohttp  # noqa: F401  (imported by caller's session)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=args.timeout) as r:
                if r.status != 200:
                    await asyncio.sleep(1 + attempt)
                    continue
                d = await r.json()
                return d["choices"][0]["message"]["content"] or ""
        except Exception:
            await asyncio.sleep(1 + attempt)
    return ""


async def run_bench(bench: str, args) -> dict:
    import aiohttp

    probs = load(bench)
    if args.limit:
        probs = probs[: args.limit]
    url = args.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    fails = 0

    async def one(p, k):
        nonlocal fails
        async with sem:
            async with aiohttp.ClientSession() as s:
                txt = await generate(s, url, args.model, PROMPT.format(problem=p["problem"]), args)
        if not txt:
            fails += 1
            return None
        return grade(txt, p["answer"])

    tasks = [one(p, k) for p in probs for k in range(args.n)]
    res = await asyncio.gather(*tasks)

    # Per-problem mean over n samples, then mean over problems = avg@n (pass@1 when n=1).
    per: list[float] = []
    for i in range(len(probs)):
        chunk = [r for r in res[i * args.n : (i + 1) * args.n] if r is not None]
        if chunk:
            per.append(sum(chunk) / len(chunk))
    acc = statistics.mean(per) if per else float("nan")
    se = (statistics.stdev(per) / len(per) ** 0.5) if len(per) > 1 else float("nan")
    return {
        "benchmark": bench,
        "n_problems": len(probs),
        "n_graded": len(per),
        "generation_failures": fails,
        "accuracy": acc,
        "se": se,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8404/v1")
    ap.add_argument("--model", default="evalmodel")
    ap.add_argument("--benchmarks", default=",".join(SUITE))
    ap.add_argument("--limit", type=int, default=0, help="problems per benchmark, 0 = all")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rows = []
    for b in args.benchmarks.split(","):
        b = b.strip()
        if not b:
            continue
        t0 = time.time()
        r = asyncio.run(run_bench(b, args))
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(
            f"{r['benchmark']:<14} acc={r['accuracy']:.4f} se={r['se']:.4f} "
            f"n={r['n_graded']}/{r['n_problems']} fail={r['generation_failures']} "
            f"({r['seconds']}s)",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    # A benchmark whose generations all failed must not read as 0% accuracy.
    bad = [r for r in rows if r["n_graded"] == 0]
    if bad:
        print(f"WARNING: {len(bad)} benchmark(s) graded nothing: "
              f"{[r['benchmark'] for r in bad]}", file=sys.stderr)


if __name__ == "__main__":
    main()
