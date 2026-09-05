#!/usr/bin/env python3
"""Turn the graded 16-sample discovery run into the shared pool file the four arms read.

Written once and read by every arm, so the pool is provably identical across arms rather than
recomputed per arm from the same inputs. The census that comes with it is the number the
experiment turns on: how many problems of the search half are genuinely mixed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms  # noqa: E402
import gate_lib  # noqa: E402


def hypergeom_informative(c: int, n: int, G: int) -> float:
    """Probability that a group of ``G`` drawn without replacement from ``n`` samples disagrees.

    Args:
        c: Correct samples among ``n``.
        n: Resolved samples for the problem.
        G: Group size.

    Returns:
        ``1 - [C(c,G) + C(n-c,G)] / C(n,G)``, or 0 when ``G > n``.
    """
    if G > n:
        return 0.0
    tot = math.comb(n, G)
    same = (math.comb(c, G) if c >= G else 0) + (math.comb(n - c, G) if n - c >= G else 0)
    return 1.0 - same / tot


def main() -> int:
    """Write ``pool.json`` and print the census and the scarcity table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True)
    ap.add_argument("--split", default="search")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truncated", default="wrong", choices=["wrong", "unknown"])
    ap.add_argument("--cap", type=int, default=0,
                    help="re-mark any sample longer than this as truncated, so the pool is "
                         "measured at the cap the arms will actually train at; must be <= the "
                         "cap the file was generated at")
    a = ap.parse_args()

    graded = a.graded
    if a.cap:
        # p-hat has to be measured at the SAME token budget the arms roll out at, or the
        # difficulty a selector reads is not the difficulty it will encounter. Because the
        # discovery run recorded each generation's length, a smaller cap is derivable from it
        # exactly -- a generation that ran past `cap` would have been cut at `cap` -- so no
        # regeneration is needed and the two are the same samples by construction.
        graded = a.graded + ".cap%d.jsonl" % a.cap
        n_re = 0
        with open(graded, "w") as fh:
            for line in open(a.graded):
                r = json.loads(line)
                if (r.get("completion_tokens") or 0) >= a.cap and not r.get("truncated"):
                    r["truncated"] = True
                    r["correct"] = False
                    n_re += 1
                fh.write(json.dumps(r) + "\n")
        print("re-marked %d samples as truncated at cap %d -> %s" % (n_re, a.cap, graded))

    problems = gate_lib.math_bench.load("olympiadbench", a.split)
    pool, census = arms.build_pool(graded, problems, truncated=a.truncated)

    # The scarcity table, computed on every problem that kept all sixteen samples -- including
    # the ones excluded from the pool, since "an always-solved problem cannot disagree at any
    # group size" is the whole point and dropping them would hide it.
    full: list[tuple[int, int]] = {}
    counts: dict[int, tuple[int, int]] = {}
    for line in open(graded):
        r = json.loads(line)
        if r.get("error"):
            counts[r["idx"]] = counts.get(r["idx"], (0, 0))
            continue
        if r.get("truncated") and a.truncated == "unknown":
            counts[r["idx"]] = counts.get(r["idx"], (0, 0))
            continue
        ok = 0 if r.get("truncated") else int(r["correct"])
        c, n = counts.get(r["idx"], (0, 0))
        counts[r["idx"]] = (c + ok, n + 1)
    full16 = {i: (c, n) for i, (c, n) in counts.items() if n == 16}
    scarcity = {}
    for G in (2, 4, 8, 16):
        vals = [hypergeom_informative(c, n, G) for c, n in full16.values()]
        scarcity[G] = {"informative": round(sum(vals) / len(vals), 4),
                       "dead": round(1 - sum(vals) / len(vals), 4)}
    census["problems_with_16_resolved"] = len(full16)
    census["always_solved_of_16"] = sum(1 for c, n in full16.values() if c == n)
    census["never_solved_of_16"] = sum(1 for c, n in full16.values() if c == 0)
    census["always_solved_fraction_of_16"] = round(
        census["always_solved_of_16"] / max(1, len(full16)), 4)

    blob = {"census": census, "scarcity": scarcity, "split": a.split, "cap": a.cap,
            "truncated_convention": a.truncated,
            "graded_source": os.path.abspath(graded),
            "tasks": [asdict(t) for t in pool]}
    json.dump(blob, open(a.out, "w"), indent=1)
    print(json.dumps({"census": census, "scarcity": scarcity}, indent=1))
    print("wrote %s with %d tasks" % (a.out, len(pool)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
