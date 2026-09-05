#!/usr/bin/env python3
"""Can a real model tell whether a Lean statement asks the problem's question?

This measures the load-bearing component of the Lean loop. The compiler only shows a
formalisation is internally consistent; the second agent, seeing the problem text and the
STATEMENT alone, is the only thing standing between a wrong-but-compiling theorem and a
confident false VERIFIED. If this agent is weak, a Lean toolchain buys nothing, because the
compiler would cheerfully certify the wrong theorem.

The set is adversarial by construction. Each problem has one FAITHFUL formalisation and
several MUTANTS carrying the specific errors that matter: a maximum standing in for a sum, a
quantifier flipped, a bound made strict, a condition negated, a count swapped for a sum, the
wrong domain. Every mutant would compile; none asks the stated question.

BOTH error rates are reported. A checker that rejects everything scores a perfect rejection
rate and is useless, and it is easier to build by accident than a good one, so the
false-rejection rate on faithful statements is reported beside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

from ornith_repro.verify_lean import FAITHFUL_PROMPT

# (problem, statement, is_faithful, mutation_kind)
CASES: list[tuple[str, str, bool, str]] = [
    # ---- P1: the error class that started this ----
    ("Find the sum of all positive integers n for which n^2 - 3n + 1 divides n^3 - 2n + 5.",
     "theorem t : (Finset.filter (fun n => (n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5))\n"
     "    (Finset.Icc 1 100)).sum id = 12", True, "faithful"),
    ("Find the sum of all positive integers n for which n^2 - 3n + 1 divides n^3 - 2n + 5.",
     "theorem t : (Finset.filter (fun n => (n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5))\n"
     "    (Finset.Icc 1 100)).max' (by decide) = 6", False, "max_for_sum"),
    ("Find the sum of all positive integers n for which n^2 - 3n + 1 divides n^3 - 2n + 5.",
     "theorem t : (Finset.filter (fun n => ¬ ((n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5)))\n"
     "    (Finset.Icc 1 100)).sum id = 4938", False, "condition_negated"),
    ("Find the sum of all positive integers n for which n^2 - 3n + 1 divides n^3 - 2n + 5.",
     "theorem t : (Finset.filter (fun n => (n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5))\n"
     "    (Finset.Icc 1 100)).card = 4", False, "count_for_sum"),

    # ---- P2: or/and, count/sum ----
    ("How many positive integers less than 100 are divisible by 3 or by 5?",
     "theorem t : (Finset.filter (fun n => 3 ∣ n ∨ 5 ∣ n) (Finset.Ico 1 100)).card = 45",
     True, "faithful"),
    ("How many positive integers less than 100 are divisible by 3 or by 5?",
     "theorem t : (Finset.filter (fun n => 3 ∣ n ∧ 5 ∣ n) (Finset.Ico 1 100)).card = 6",
     False, "or_became_and"),
    ("How many positive integers less than 100 are divisible by 3 or by 5?",
     "theorem t : (Finset.filter (fun n => 3 ∣ n ∨ 5 ∣ n) (Finset.Ico 1 100)).sum id = 2418",
     False, "sum_for_count"),

    # ---- P3: least/greatest ----
    ("Find the smallest positive integer n such that n! is divisible by 1000.",
     "theorem t : IsLeast {n : ℕ | 0 < n ∧ 1000 ∣ Nat.factorial n} 15", True, "faithful"),
    ("Find the smallest positive integer n such that n! is divisible by 1000.",
     "theorem t : IsGreatest {n : ℕ | 0 < n ∧ 1000 ∣ Nat.factorial n} 15",
     False, "least_became_greatest"),
    ("Find the smallest positive integer n such that n! is divisible by 1000.",
     "theorem t : ∃ n : ℕ, 0 < n ∧ 1000 ∣ Nat.factorial n", False, "least_became_exists"),

    # ---- P4: quantifiers and strictness ----
    ("Show that for every real number x, x^2 + 1 is at least 2x.",
     "theorem t : ∀ x : ℝ, x^2 + 1 ≥ 2*x", True, "faithful"),
    ("Show that for every real number x, x^2 + 1 is at least 2x.",
     "theorem t : ∃ x : ℝ, x^2 + 1 ≥ 2*x", False, "forall_became_exists"),
    ("Show that for every real number x, x^2 + 1 is at least 2x.",
     "theorem t : ∀ x : ℝ, x^2 + 1 > 2*x", False, "bound_made_strict"),

    # ---- P5: domain ----
    ("Compute the sum of the first 10 positive even integers.",
     "theorem t : (Finset.range 10).sum (fun k => 2*(k+1)) = 110", True, "faithful"),
    ("Compute the sum of the first 10 positive even integers.",
     "theorem t : (Finset.range 10).sum (fun k => 2*k + 1) = 100", False, "even_became_odd"),
    ("Compute the sum of the first 10 positive even integers.",
     "theorem t : (Finset.range 20).sum (fun k => 2*(k+1)) = 420", False, "wrong_domain"),

    # ---- P6: which variable ----
    ("Let S be the set of integers n with 1 <= n <= 20 such that n^2 + n + 41 is prime. "
     "How many elements does S have?",
     "theorem t : (Finset.filter (fun n => Nat.Prime (n^2 + n + 41))\n"
     "    (Finset.Icc 1 20)).card = 20", True, "faithful"),
    ("Let S be the set of integers n with 1 <= n <= 20 such that n^2 + n + 41 is prime. "
     "How many elements does S have?",
     "theorem t : (Finset.filter (fun n => Nat.Prime (n^2 + n + 41))\n"
     "    (Finset.Icc 1 20)).sum id = 210", False, "sum_for_count"),

    # ---- P7: two more faithful, to keep the balance from flattering rejection ----
    ("Find the number of positive divisors of 360.",
     "theorem t : (Nat.divisors 360).card = 24", True, "faithful"),
    ("Find the remainder when 7^100 is divided by 13.",
     "theorem t : 7^100 % 13 = 9", True, "faithful"),
    ("Find the number of positive divisors of 360.",
     "theorem t : (Nat.divisors 360).sum id = 1170", False, "sum_for_count"),
    ("Find the remainder when 7^100 is divided by 13.",
     "theorem t : 7^100 / 13 = 9", False, "mod_became_div"),
]


async def ask(session, url, model, problem, statement, effort, timeout):
    """Run one faithfulness check against the served model.

    Args:
        session: aiohttp session.
        url: Chat-completions endpoint.
        model: Served model id, checked by the caller.
        problem: Natural-language problem.
        statement: The Lean statement, proof stripped.
        effort: reasoning_effort.
        timeout: Request timeout.

    Returns:
        dict with the parsed verdict, raw head, and token cost.
    """
    payload = {"model": model,
               "messages": [{"role": "user", "content": FAITHFUL_PROMPT.format(
                   problem=problem, statement=statement)}],
               "temperature": 0.0, "max_tokens": 2048,
               "chat_template_kwargs": {"reasoning_effort": effort}}
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=timeout) as r:
                if r.status != 200:
                    await asyncio.sleep(2 + attempt)
                    continue
                d = await r.json()
                ch = (d.get("choices") or [{}])[0]
                text = (ch.get("message") or {}).get("content") or ""
                body = text.split("</think>")[-1].strip()
                head = body.splitlines()[0].strip().upper() if body else ""
                if "MISMATCH" in head:
                    verdict = "MISMATCH"
                elif "FAITHFUL" in head:
                    verdict = "FAITHFUL"
                else:
                    verdict = "UNPARSEABLE"
                return {"verdict": verdict, "head": head[:120],
                        "tokens": (d.get("usage") or {}).get("completion_tokens") or 0,
                        "finish": ch.get("finish_reason")}
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2 + attempt)
    return {"verdict": "ERROR", "head": "", "tokens": 0, "finish": None}


async def run(a):
    """Run the whole adversarial set and print both error rates."""
    url = a.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.get(a.base_url.rstrip("/") + "/models",
                               timeout=aiohttp.ClientTimeout(total=60)) as r:
            served = [m["id"] for m in (await r.json()).get("data", [])]
        if a.model not in served:
            print("FATAL: %r not served; served=%s" % (a.model, served))
            return 2
        print("served-model check OK: %s   cases: %d   repeats: %d"
              % (a.model, len(CASES), a.repeats), flush=True)

        async def one(i, case, rep):
            problem, statement, faithful, kind = case
            async with sem:
                res = await ask(session, url, a.model, problem, statement, a.effort,
                                aiohttp.ClientTimeout(total=a.timeout))
            res.update(case_id=i, kind=kind, is_faithful=faithful, repeat=rep,
                       problem=problem[:80], statement=statement[:120])
            return res

        tasks = [one(i, c, r) for r in range(a.repeats) for i, c in enumerate(CASES)]
        rows = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    faithful = [r for r in rows if r["is_faithful"]]
    mutant = [r for r in rows if not r["is_faithful"]]
    caught = [r for r in mutant if r["verdict"] == "MISMATCH"]
    missed = [r for r in mutant if r["verdict"] == "FAITHFUL"]
    kept = [r for r in faithful if r["verdict"] == "FAITHFUL"]
    false_rej = [r for r in faithful if r["verdict"] == "MISMATCH"]
    unparse = [r for r in rows if r["verdict"] in ("UNPARSEABLE", "ERROR")]

    print("")
    print("=== faithfulness checker, real model ===")
    print("  mutants caught (MISMATCH)   : %3d / %3d = %.4f   <-- the headline"
          % (len(caught), len(mutant), len(caught) / max(len(mutant), 1)))
    print("  mutants MISSED (FAITHFUL)   : %3d / %3d = %.4f   <-- false certifications"
          % (len(missed), len(mutant), len(missed) / max(len(mutant), 1)))
    print("  faithful kept (FAITHFUL)    : %3d / %3d = %.4f"
          % (len(kept), len(faithful), len(kept) / max(len(faithful), 1)))
    print("  faithful REJECTED (MISMATCH): %3d / %3d = %.4f   <-- false rejections"
          % (len(false_rej), len(faithful), len(false_rej) / max(len(faithful), 1)))
    print("  unparseable/error           : %3d" % len(unparse))
    print("")
    print("  by mutation kind:")
    kinds = {}
    for r in mutant:
        k = kinds.setdefault(r["kind"], [0, 0])
        k[1] += 1
        if r["verdict"] == "MISMATCH":
            k[0] += 1
    for k in sorted(kinds):
        c, n = kinds[k]
        print("    %-24s %2d/%2d caught" % (k, c, n))
    if missed:
        print("")
        print("  MISSED mutants (would have been certified):")
        for r in missed[:10]:
            print("    %-24s %s" % (r["kind"], r["statement"][:80]))
    total_tokens = sum(r["tokens"] for r in rows)
    print("")
    print("  cost: %d generations, %d completion tokens, %.1fs wall"
          % (len(rows), total_tokens, elapsed))
    print("  wrote %s" % a.out)
    return 0


def main():
    """Parse arguments and run the probe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/faithfulness_probe.jsonl")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
