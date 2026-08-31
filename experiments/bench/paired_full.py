#!/usr/bin/env python3
"""Paired McNemar on ALL 500 problems, plus the noise floor from the shared base checkpoint.

compare_runs.py restricts to the held-out `report` half, which is right when a comparison was
searched on the other half. This A/B searched nothing -- the arms differ in one predeclared
flag -- so the full 500 is legitimate and doubles the power. Both are reported: the 250 is the
protocol number, the 500 is the best-powered read of the same data.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from math_bench import grade  # noqa: E402


def load(suite: pathlib.Path, name: str) -> dict[str, bool] | None:
    """Per-problem correctness for one checkpoint, keyed by problem index."""
    f = suite / name / "generations.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.open():
        r = json.loads(line)
        if r.get("bench") not in (None, "math500"):
            continue
        key = str(r.get("idx", r.get("index", r.get("problem_id"))))
        out[key] = bool(r.get("correct", grade(r.get("completion", ""), r.get("answer", ""))))
    return out or None


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar over the problems both arms answered."""
    shared = sorted(set(a) & set(b))
    n01 = sum(1 for k in shared if a[k] and not b[k])
    n10 = sum(1 for k in shared if b[k] and not a[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    lo = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def main() -> int:
    """Print the paired table and the noise floor the effects must clear."""
    A, B = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
    names = ["base"] + [f"gs{int(s):03d}" for s in sys.argv[5].split(",")] if len(sys.argv) > 5 \
        else ["base", "gs028", "gs057", "gs086", "gs115", "gs144"]

    floor = None
    print(f"{'ckpt':>6} {na:>22} {nb:>22} {'diff':>8} {'n01/n10':>10} {'McNemar p':>11}")
    for nm in names:
        a, b = load(A, nm), load(B, nm)
        if not a or not b:
            print(f"{nm:>6} {'missing':>22}"); continue
        shared = sorted(set(a) & set(b))
        pa = sum(a[k] for k in shared) / len(shared)
        pb = sum(b[k] for k in shared) / len(shared)
        n01, n10, p = mcnemar(a, b)
        tag = "  <- NOISE FLOOR (same weights)" if nm == "base" else ""
        if nm == "base":
            floor = abs(pb - pa)
        print(f"{nm:>6} {pa:>22.4f} {pb:>22.4f} {pb-pa:>+8.4f} {f'{n01}/{n10}':>10} {p:>11.3g}{tag}")
        if nm != "base" and floor is not None and abs(pb - pa) <= floor:
            print(f"{'':>6} {'':>22} {'':>22} {'':>8} {'':>10} {'within the noise floor':>11}")
    print(f"\nn per comparison: {len(shared)} problems, paired")
    if floor is not None:
        print(f"noise floor from the shared base checkpoint: {floor:.4f}")
        print("An effect at or below this is not distinguishable from scoring the same model twice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
