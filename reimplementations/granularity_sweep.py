#!/usr/bin/env python3
"""Sweep grouping granularity through the pre-registered gate to find where it peaks.

Two points do not make a curve. Four groups gave a spread ratio of 2.13 at p = 0.148 and ten
gave 2.70 at p = 0.009, which says the structure is invisible at four and real at ten but not
where it stops improving.

There is a genuine optimum rather than a monotone trend. Too coarse and weakness is invisible.
Too fine and each group holds a handful of problems, so "weakness per group" degenerates
toward per-problem noise and a curriculum cannot act on it, because generating more problems
like one specific problem is not a category. The permutation null keeps the TEST valid at any
granularity -- it is size-matched by construction -- but validity is not usefulness, so the
effective problems per group is reported beside every point.

GRANULARITY IS VARIED WITHOUT USING ACCURACY. Groups are built from the topic label, the
answer type and the pooling threshold, never from the measured success rate. Partitioning on
the very quantity whose spread is being tested would manufacture the result.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

from ornith_repro.probe_split import load_committed, make_probe_split
from ornith_repro.weakness import load_outcomes
from spread_gate import permutation_test, pool_small


def main():
    """Run the gate at a range of granularities and print the curve."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--min-samples", type=int, default=8)
    a = ap.parse_args()

    src = "/home/ubuntu/reach/data/olympiadbench/test.jsonl"
    problems = [json.loads(l) for l in open(src) if l.strip()]
    coarse = {i: (r.get("subfield") or "unknown") for i, r in enumerate(problems)}
    atype = {i: (r.get("answer_type") or "unknown") for i, r in enumerate(problems)}
    search, report, _ = load_committed(
        "/home/ubuntu/reach/bench/olympiadbench_split.json", src)
    probe, _ = make_probe_split(search, report, coarse, 0.35, 20260904)

    imap = json.load(open("/mnt/localssd/gate/searchhalf/index_map.json"))["original_index"]
    merged = defaultdict(list)
    for path in ("/mnt/localssd/gate/out/blocks_low.jsonl",
                 "/mnt/localssd/gate/out/blocks_low_more.jsonl"):
        if os.path.exists(path):
            for k, v in load_outcomes(path, problems, imap).items():
                merged[k].extend(v)

    fine = {}
    for line in open("/mnt/localssd/gate/out/fine_labels.jsonl"):
        if line.strip():
            r = json.loads(line)
            fine[r["idx"]] = r["label"]

    idxs, vals = [], []
    for i in probe:
        resolved = [c for c, t in merged.get(i, []) if not t]
        if len(resolved) >= a.min_samples:
            idxs.append(i)
            vals.append(sum(1 for c in resolved if c) / len(resolved))
    print("probe problems usable: %d" % len(idxs))

    # Label schemes, none derived from accuracy.
    schemes = [
        ("coarse subfield", [coarse[i] for i in idxs], 5),
        ("fine topic", [fine.get(i, "unlabelled") for i in idxs], 20),
        ("fine topic", [fine.get(i, "unlabelled") for i in idxs], 10),
        ("fine topic", [fine.get(i, "unlabelled") for i in idxs], 5),
        ("fine topic", [fine.get(i, "unlabelled") for i in idxs], 3),
        ("fine topic", [fine.get(i, "unlabelled") for i in idxs], 1),
        ("topic x answer-type",
         ["%s|%s" % (fine.get(i, "unlabelled"), atype[i]) for i in idxs], 3),
        ("topic x answer-type",
         ["%s|%s" % (fine.get(i, "unlabelled"), atype[i]) for i in idxs], 1),
    ]

    print("")
    print("%-20s %5s %7s %9s %10s %8s %8s"
          % ("scheme", "pool", "groups", "probs/grp", "V_obs", "ratio", "p"))
    curve = []
    for name, labels, min_size in schemes:
        pooled = pool_small(labels, min_size=min_size)
        res = permutation_test(vals, pooled, draws=a.draws)
        sizes = Counter(pooled)
        per = statistics.median(sizes.values())
        curve.append((name, min_size, res["n_groups"], per, res["ratio"], res["p"]))
        print("%-20s %5d %7d %9.1f %10.6f %8.2f %8.4f"
              % (name, min_size, res["n_groups"], per, res["observed"],
                 res["ratio"], res["p"]))

    print("")
    # A partition with fewer than two surviving groups has zero between-group variance and
    # a zero null, so its ratio is inf. That is a degenerate arithmetic artefact, not a peak,
    # and selecting on it would have reported "one group of 105" as the best granularity.
    real = [c for c in curve if c[2] >= 2 and c[4] != float("inf")]
    usable = [c for c in real if c[3] >= 5]
    if real:
        best = max(real, key=lambda c: c[4])
        print("peak spread ratio among non-degenerate partitions: %s (pool>=%d), %d groups, "
              "median %.1f problems/group, ratio %.2f, p=%.4f"
              % (best[0], best[1], best[2], best[3], best[4], best[5]))
    if usable:
        bu = max(usable, key=lambda c: c[4])
        print("peak among partitions a curriculum could act on (>=5 problems/group): "
              "%s (pool>=%d), %d groups, %.1f problems/group, ratio %.2f, p=%.4f"
              % (bu[0], bu[1], bu[2], bu[3], bu[4], bu[5]))
    print("")
    print("ratio by granularity (non-degenerate only):")
    for name, ms, ng, per, ratio, pv in sorted(real, key=lambda c: c[2]):
        print("   %2d groups  %5.1f probs/grp  ratio %5.2f  p=%.4f   %s"
              % (ng, per, ratio, pv, name))
    print("")
    print("A group holding only a handful of problems is a valid test and an unusable")
    print("curriculum target: 'generate more problems like this one problem' is not a")
    print("category. Read the ratio and the problems-per-group column together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
