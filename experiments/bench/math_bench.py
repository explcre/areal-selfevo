#!/usr/bin/env python3
"""Score a served model on the frozen math suite.

Generation goes through our sglang OpenAI-compatible endpoint; grading uses `math_verify`,
which is what AZR's own `custom_evaluate.py` imports (`from math_verify import parse,
verify`). An independent audit cross-checked this grader against AZR's own cascade over 680
generations and found **zero disagreements**, 1 misgrade in 600 (understatement), and no
false positives -- so grading is not the uncertainty that matters here.

The uncertainties that DO matter, all surfaced by that audit and handled below:

* **Run-to-run nondeterminism dominates problem-sampling error.** The same command twice
  gave MATH-500 74.80% and 76.40%; AMC23 spanned 47.5-60.0% over five runs. A standard
  error computed from problem sampling alone (+/-7.9pp on AMC23) understates the real
  spread. Every number should be reported with the seed and, where it matters, repeated.
* **A binomial SE is the wrong interval at these counts.** At 1/30 the normal interval runs
  negative; at 0/30 or 30/30 the SE is exactly 0, claiming certainty. Wilson is reported
  instead, and it is asymmetric on purpose.
* **Silent-zero paths.** Truncated generations were graded as wrong answers with nothing
  reported; empty 200-responses were dropped from the denominator, which *inflates*
  accuracy; a partial outage shrank the denominator while still printing a normal-looking
  score. Each is now counted and surfaced separately.
* **Nothing was persisted**, so no reported number could be re-audited without regenerating
  it. Every completion is now written to disk.

Why not AZR's runner: `math_eval/eval/math_eval.py` does `from vllm import LLM`, and vLLM
is not installed here. Installing it risks the torch/sglang environment. The runner is the
easy half; the grader is the part worth reusing.

Datasets come from the pinned AZR clone (MIT). Only benchmarks with a uniform
`problem`/`answer` schema are included; olympiadbench (`question`/`final_answer`) and
minerva_math (answer embedded in `solution`) need their own extraction and are deliberately
left out rather than half-supported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

DATA = Path(
    "/home/ubuntu/baselines/Absolute-Zero-Reasoner/evaluation/math_eval/eval/data"
)
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
        ValueError: If a row lacks the uniform problem/answer schema, or none load.
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

    A regex like ``\\\\boxed\\{([^}]*)\\}`` truncates at the first inner brace, silently
    mangling ``\\boxed{\\frac{1}{2}}`` into ``\\frac{1``. A mangled extraction grades as
    wrong, so that bug surfaces as a plausible lower score rather than an error.

    An unbalanced final box (the token cap landing mid-box) returns None rather than a
    truncated string. Note this discards any earlier complete box in the same completion --
    deliberate, because a completion cut off mid-answer has not actually answered.
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
    return None


def grade(pred_text: str, gold: str) -> bool:
    """True when the prediction matches the gold answer.

    Argument order into ``verify(gold, pred)`` matters and is deliberate: the audit found a
    real case where swapping them yields a false positive (gold ``5x-7y+11z+4=0``, pred
    ``0``).
    """
    boxed = extract_boxed(pred_text)
    if boxed is None:
        return False
    cands = [boxed]
    # math_verify parses "10\%" as 0.1, so a percent answer against a bare numeric gold
    # reads as wrong. Only tried when the gold itself carries no percent sign.
    if "%" not in gold and boxed.rstrip().endswith(("\\%", "%")):
        cands.append(re.sub(r"\\?%\s*$", "", boxed).strip())
    try:
        from math_verify import parse, verify

        g = parse(f"${gold}$")
        for c in cands:
            p = parse(f"${c}$")
            if g and p and verify(g, p):
                return True
    except Exception:
        pass
    norm = lambda s: re.sub(r"[\s,$]|\\left|\\right|\\!|\\,|\\?%", "", s).rstrip(".")
    return any(norm(c) == norm(gold) for c in cands)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Used instead of a normal binomial SE because at these counts the SE misleads: at 1/30
    the normal interval runs negative, and at 0/30 or 30/30 it is exactly 0, asserting
    certainty from a single unanimous sample.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


async def generate(session, url: str, model: str, prompt: str, args) -> dict:
    """One completion.

    Returns:
        ``{"text", "finish_reason", "status"}`` where ``status`` is one of ``ok`` (the
        endpoint answered), or ``failed`` (no usable response after retries). An HTTP 200
        carrying empty content is ``ok`` with empty text, NOT ``failed``: an empty
        completion is a wrong answer, and excluding it from the denominator would inflate
        accuracy.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=args.timeout) as r:
                if r.status != 200:
                    await asyncio.sleep(1 + attempt)
                    continue
                d = await r.json()
                ch = (d.get("choices") or [{}])[0]
                return {
                    "text": (ch.get("message") or {}).get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "status": "ok",
                }
        except Exception:
            await asyncio.sleep(1 + attempt)
    return {"text": "", "finish_reason": None, "status": "failed"}


async def run_bench(bench: str, args, gen_fh=None) -> dict:
    import aiohttp

    probs = load(bench)
    if args.limit:
        probs = probs[: args.limit]
    url = args.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    conn = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(connector=conn) as session:
        async def one(idx: int, p: dict, k: int) -> dict:
            async with sem:
                r = await generate(session, url, args.model, PROMPT.format(problem=p["problem"]), args)
            boxed = extract_boxed(r["text"])
            correct = grade(r["text"], p["answer"]) if r["status"] == "ok" else None
            return {
                "benchmark": bench, "idx": idx, "sample": k,
                "gold": p["answer"], "boxed": boxed,
                "finish_reason": r["finish_reason"], "status": r["status"],
                "correct": correct, "text": r["text"],
            }

        recs = await asyncio.gather(
            *[one(i, p, k) for i, p in enumerate(probs) for k in range(args.n)]
        )

    if gen_fh is not None:
        for rec in recs:
            gen_fh.write(json.dumps(rec) + "\n")
        gen_fh.flush()

    n_failed = sum(1 for r in recs if r["status"] == "failed")
    n_trunc = sum(1 for r in recs if r["finish_reason"] == "length")
    n_nobox = sum(1 for r in recs if r["status"] == "ok" and r["boxed"] is None)

    # Per-problem mean over its OK samples, then mean over problems (avg@n; pass@1 at n=1).
    per: list[float] = []
    for i in range(len(probs)):
        chunk = [r["correct"] for r in recs[i * args.n : (i + 1) * args.n] if r["correct"] is not None]
        if chunk:
            per.append(sum(chunk) / len(chunk))
    acc = statistics.mean(per) if per else float("nan")
    lo, hi = wilson(round(acc * len(per)), len(per)) if per else (float("nan"),) * 2
    return {
        "benchmark": bench,
        "n_problems": len(probs),
        "n_graded": len(per),
        "n_failed": n_failed,
        "n_truncated": n_trunc,
        "n_no_box": n_nobox,
        "accuracy": acc,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "seed": args.seed,
        "temperature": args.temperature,
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
    ap.add_argument("--seed", type=int, default=0, help="passed to the server; None to omit")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--gen-out", type=str, default="", help="JSONL of every completion")
    args = ap.parse_args()

    # avg@n at temperature 0 measures batching nondeterminism, not model uncertainty:
    # measured on amc23 n=4, only 11/40 problems produced 4 identical samples.
    if args.n > 1 and args.temperature == 0.0:
        print("ERROR: --n > 1 at temperature 0 measures nondeterminism, not uncertainty. "
              "Raise --temperature or set --n 1.", file=sys.stderr)
        raise SystemExit(2)

    gen_fh = open(args.gen_out, "w") if args.gen_out else None
    rows = []
    try:
        for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
            t0 = time.time()
            r = asyncio.run(run_bench(b, args, gen_fh))
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(
                f"{r['benchmark']:<14} acc={r['accuracy']:.4f} "
                f"wilson=[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
                f"n={r['n_graded']}/{r['n_problems']} "
                f"fail={r['n_failed']} trunc={r['n_truncated']} nobox={r['n_no_box']} "
                f"({r['seconds']}s)",
                flush=True,
            )
    finally:
        if gen_fh:
            gen_fh.close()

    if args.out:
        # allow_nan=False: NaN is not valid JSON and silently breaks downstream readers.
        Path(args.out).write_text(
            json.dumps(
                [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                  for k, v in r.items()} for r in rows],
                indent=2, allow_nan=False,
            )
        )
        print(f"wrote {args.out}")

    for r in rows:
        if r["n_graded"] == 0:
            print(f"WARNING {r['benchmark']}: graded nothing; accuracy is meaningless",
                  file=sys.stderr)
        elif r["n_graded"] < r["n_problems"]:
            # A partial outage shrinks the denominator while still printing a
            # normal-looking score, and timeouts correlate with long/hard generations.
            print(f"WARNING {r['benchmark']}: only {r['n_graded']}/{r['n_problems']} "
                  f"graded ({r['n_failed']} failed); accuracy is over survivors and is "
                  "biased upward", file=sys.stderr)
        if r["n_truncated"]:
            print(f"NOTE {r['benchmark']}: {r['n_truncated']} generation(s) hit the token "
                  "cap and were graded as wrong; raise --max-tokens to test sensitivity",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
