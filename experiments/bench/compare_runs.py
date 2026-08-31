#!/usr/bin/env python3
"""Paired comparison of two checkpoint series on the held-out MATH-500 half.

Compares the demo recipe (lr 6e-6, eps_clip 0.4) against AReaL's scaffolding values
(lr 1e-6, eps_clip 0.2) at global steps present in BOTH runs, so every point is paired.

Only the `report` half is scored, per arXiv 2607.12227: searching and reporting on the same
task set overstates the gain. The older sweep scored all 500 problems, so its report-half
score is recomputed here from its persisted generations rather than re-run -- the same
grader, the same problems, no GPU.

Significance is paired McNemar on the same problems, not two independent intervals.

Usage: compare_runs.py <suiteA> <suiteB> [--steps 28,57,86,115,144]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from math_bench import grade

SPLIT = pathlib.Path(__file__).resolve().parent / "math500_split.json"


def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] and not b[k])
    b10 = sum(1 for k in keys if b[k] and not a[k])
    n = b01 + b10
    if n == 0: return b01, b10, 1.0, len(keys)
    if n < 25:
        p = 2 * sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1)) / (2 ** n)
        return b01, b10, min(1.0, p), len(keys)
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    return b01, b10, 2 * (1 - norm_cdf(z)), len(keys)


def score(suite: pathlib.Path, tag: str, keep: set[int]) -> dict[int, bool] | None:
    """Per-problem correctness on the kept indices, regraded from persisted text.

    Returns None when the artifact is absent, so a missing arm is reported as missing
    rather than silently scoring zero.
    """
    f = suite / tag / "generations.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["idx"] not in keep:
            continue
        out[r["idx"]] = grade(r["text"], r["gold"])
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_a"); ap.add_argument("suite_b")
    ap.add_argument("--name-a", default="demo(lr6e-6,clip0.4)")
    ap.add_argument("--name-b", default="scaffold(lr1e-6,clip0.2)")
    ap.add_argument("--steps", default="28,57,86,115,144")
    a = ap.parse_args()

    keep = set(json.loads(SPLIT.read_text())["report"])
    A, B = pathlib.Path(a.suite_a), pathlib.Path(a.suite_b)
    steps = [int(s) for s in a.steps.split(",") if s]

    print(f"held-out half: {len(keep)} problems (report)\n")
    print(f"{'step':>5} | {a.name_a:>24} | {a.name_b:>26} | {'diff':>7} | {'McNemar p':>10}")
    print("-" * 92)
    base_a = score(A, "base", keep)
    base_b = score(B, "base", keep)
    if base_a: print(f"{'base':>5} | {sum(base_a.values())/len(base_a):>24.3f} | "
                     f"{(sum(base_b.values())/len(base_b) if base_b else float('nan')):>26.3f} | "
                     f"{'':>7} | {'':>10}")
    rows = []
    for s in steps:
        tag = f"gs{s:03d}"
        ra, rb = score(A, tag, keep), score(B, tag, keep)
        if ra is None or rb is None:
            miss = a.name_a if ra is None else a.name_b
            print(f"{s:>5} | {'(missing: ' + miss + ')':>60}")
            continue
        pa, pb = sum(ra.values()) / len(ra), sum(rb.values()) / len(rb)
        b01, b10, p, n = mcnemar(ra, rb)
        print(f"{s:>5} | {pa:>24.3f} | {pb:>26.3f} | {pb-pa:>+7.3f} | {p:>10.2e}")
        rows.append((s, pa, pb, p))
    if rows:
        print(f"\nmean over {len(rows)} paired steps: "
              f"{a.name_a} {sum(r[1] for r in rows)/len(rows):.3f}  vs  "
              f"{a.name_b} {sum(r[2] for r in rows)/len(rows):.3f}")
        wins = sum(1 for r in rows if r[2] > r[1])
        print(f"{a.name_b} higher at {wins}/{len(rows)} steps; "
              f"significant (p<0.05) at {sum(1 for r in rows if r[3] < 0.05)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
