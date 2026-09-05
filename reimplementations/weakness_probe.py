#!/usr/bin/env python3
"""Rank categories by weakness and by headroom, on a probe disjoint from the evaluation set.

Two rankings are reported because they disagree, and the disagreement is the point. Raw
weakness ranks a category the model never solves as the weakest, but every group there is
unanimous-fail and carries exactly zero gradient, so generating into it produces no training
signal at all. Headroom ranks by the supply of MIXED problems, which is the informative
region. Steering by raw weakness would aim the generator at the deadest region available.

Intervals are Wilson and are reported per category, because a category measured on a handful
of problems is mostly noise and must not be chased.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from ornith_repro.probe_split import load_committed, make_probe_split, write_split


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial proportion.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile.

    Returns:
        `(lo, hi)`, or `(nan, nan)` when n is zero.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def main():
    """Build the probe split and report per-category weakness and headroom."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/home/ubuntu/reach/data/olympiadbench/test.jsonl")
    ap.add_argument("--split", default="/home/ubuntu/reach/bench/olympiadbench_split.json")
    ap.add_argument("--blocks", default="/mnt/localssd/gate/out/blocks_low.jsonl")
    ap.add_argument("--searchhalf",
                    default="/mnt/localssd/gate/searchhalf/index_map.json")
    ap.add_argument("--probe-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/probe_split.json")
    a = ap.parse_args()

    search, report, md5 = load_committed(a.split, a.source)
    rows = [json.loads(l) for l in open(a.source) if l.strip()]
    fields = {i: (r.get("subfield") or "unknown") for i, r in enumerate(rows)}

    probe, train = make_probe_split(search, report, fields, a.probe_frac, a.seed)
    write_split(a.out, probe, train, report, md5, a.seed, a.probe_frac)
    print("probe %d, train %d, report %d (md5 %s)" % (len(probe), len(train), len(report), md5[:8]))
    print("probe INTERSECT report = %d  (must be 0)" % len(set(probe) & set(report)))

    # p_hat per SEARCH-HALF POSITION -> original index, via the committed index map.
    orig = json.load(open(a.searchhalf))["original_index"]
    outcomes = defaultdict(list)
    for line in open(a.blocks):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("finish_reason") != "length":
            outcomes[orig[r["idx"]]].append(1 if r.get("correct") else 0)

    per = defaultdict(lambda: {"n": 0, "solved": 0, "always": 0, "never": 0, "mixed": 0})
    for idx in probe:
        v = outcomes.get(idx, [])
        if len(v) < 8:
            continue
        p = sum(v) / len(v)
        d = per[fields[idx]]
        d["n"] += 1
        d["solved"] += p
        d["always"] += (p == 1.0)
        d["never"] += (p == 0.0)
        d["mixed"] += (0.0 < p < 1.0)

    print("")
    print("%-16s %4s %8s %-18s %8s %8s %8s" %
          ("subfield", "n", "acc", "acc 95% CI", "mixed", "always", "never"))
    rowsout = []
    for f in sorted(per):
        d = per[f]
        acc = d["solved"] / d["n"]
        lo, hi = wilson(int(round(d["solved"])), d["n"])
        mixed = d["mixed"] / d["n"]
        rowsout.append((f, d["n"], acc, lo, hi, mixed, d["always"] / d["n"], d["never"] / d["n"]))
        print("%-16s %4d %8.4f [%.3f,%.3f]   %8.4f %8.4f %8.4f"
              % (f, d["n"], acc, lo, hi, mixed, d["always"] / d["n"], d["never"] / d["n"]))

    print("")
    print("ranked by RAW WEAKNESS (lowest accuracy first):")
    for f, n, acc, lo, hi, mx, al, nv in sorted(rowsout, key=lambda x: x[2]):
        print("   %-16s acc %.4f  mixed %.4f  n=%d" % (f, acc, mx, n))
    print("")
    print("ranked by HEADROOM (most mixed problems first) -- the informative supply:")
    for f, n, acc, lo, hi, mx, al, nv in sorted(rowsout, key=lambda x: -x[5]):
        print("   %-16s mixed %.4f  acc %.4f  n=%d" % (f, mx, acc, n))
    print("")
    wide = [f for f, n, acc, lo, hi, mx, al, nv in rowsout if hi - lo > 0.35]
    if wide:
        print("intervals too wide to rank on (do not chase these): %s" % wide)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
