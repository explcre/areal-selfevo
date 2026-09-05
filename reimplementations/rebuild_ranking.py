#!/usr/bin/env python3
"""Recompute the weakness ranking, after first proving the machinery recovers a known answer.

WHY THE CONTROL RUNS FIRST. Two wrong numbers on the sister machine came from apparatus that
had never been run against a case whose answer was known in advance: a comparator never tested
on keys known to be RIGHT, and a similarity measure never tested on problems known to be
UNRELATED. The ranking machinery has the same exposure, so before it is pointed at real data it
is pointed at a fabricated pool with a planted weakest category, through the FULL path --
blocks on disk, an index map, the content check, the drop filter, the exclusions and the sort.
If it cannot recover a planted answer, nothing it says about real data means anything.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict

from ornith_repro.probe_split import load_committed, make_probe_split
from ornith_repro.weakness import WeaknessError, category_stats, load_outcomes, rank


def known_answer_control() -> bool:
    """Plant a weakest category in fabricated data and check the pipeline recovers it.

    Returns:
        True when both rankings recover the planted category.
    """
    n_per, n_samp = 12, 12
    cats = {"Planted_Weak": 0.30, "Middle": 0.65, "Strong": 0.95}
    problems, fields, rows, index_map = [], {}, [], []
    idx = 0
    for cat, target in cats.items():
        for j in range(n_per):
            problems.append({"question": "%s-%d" % (cat, j),
                             "final_answer": ["A%d" % idx]})
            fields[idx] = cat
            index_map.append(idx)
            n_correct = int(round(target * n_samp))
            for s in range(n_samp):
                rows.append({"idx": idx, "sample": s, "gold": "A%d" % idx,
                             "correct": s < n_correct, "status": "ok",
                             "finish_reason": "stop"})
            idx += 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "blocks.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        out = load_outcomes(path, problems, index_map)
        st = category_stats(out, fields, probe=list(range(idx)), min_samples=8,
                            max_drop_rate=0.30, max_interval_width=1.0)

    weak = [f for f, _ in rank(st, by="weakness")]
    head = [f for f, _ in rank(st, by="headroom")]
    ok = bool(weak) and weak[0] == "Planted_Weak" and bool(head) and head[0] == "Planted_Weak"
    print("KNOWN-ANSWER CONTROL")
    for f, d in sorted(st.items()):
        print("   %-14s acc=%.3f mixed=%.3f used=%d ranked=%s"
              % (f, d["acc"], d["mixed"], d["used"], d["ranked"]))
    print("   weakness order: %s" % weak)
    print("   headroom order: %s" % head)
    print("   RECOVERED PLANTED CATEGORY: %s" % ok)

    # The corruption the audit reproduced must still be caught on this same path.
    caught = False
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps({"idx": 0, "sample": 0, "gold": "A0", "correct": True,
                                 "status": "ok", "finish_reason": "stop"}) + "\n")
        try:
            load_outcomes(path, problems, [5])          # map says 5, row was graded as 0
        except WeaknessError:
            caught = True
    print("   STALE-MAP CORRUPTION CAUGHT: %s" % caught)
    return ok and caught


def main() -> int:
    """Run the control, then recompute the real ranking."""
    if not known_answer_control():
        print("\nCONTROL FAILED -- refusing to report a ranking from machinery that cannot "
              "recover a planted answer.")
        return 2

    src = "/home/ubuntu/reach/data/olympiadbench/test.jsonl"
    problems = [json.loads(l) for l in open(src) if l.strip()]
    fields = {i: (r.get("subfield") or "unknown") for i, r in enumerate(problems)}
    search, report, md5 = load_committed(
        "/home/ubuntu/reach/bench/olympiadbench_split.json", src)
    probe, train = make_probe_split(search, report, fields, 0.35, 20260904)

    imap = json.load(open("/mnt/localssd/gate/searchhalf/index_map.json"))["original_index"]
    merged = defaultdict(list)
    for path in ("/mnt/localssd/gate/out/blocks_low.jsonl",
                 "/mnt/localssd/gate/out/blocks_low_more.jsonl"):
        if not os.path.exists(path):
            continue
        got = load_outcomes(path, problems, imap)
        for k, v in got.items():
            merged[k].extend(v)
        print("merged %s: %d problems" % (os.path.basename(path), len(got)))

    print("\nprobe %d, train %d, report %d; probe INTERSECT report = %d"
          % (len(probe), len(train), len(report), len(set(probe) & set(report))))

    for min_s in (8, 12, 16):
        st = category_stats(merged, fields, probe, min_samples=min_s)
        print("\n=== min_samples=%d ===" % min_s)
        print("%-16s %5s %5s %6s %8s %-16s %7s %s"
              % ("subfield", "seen", "used", "drop", "acc", "95% CI", "mixed", "ranked"))
        for f in sorted(st):
            d = st[f]
            print("%-16s %5d %5d %6.2f %8.4f [%.3f,%.3f] %7.4f %s%s"
                  % (f, d["seen"], d["used"], d["drop_rate"], d["acc"], d["lo"], d["hi"],
                     d["mixed"], d["ranked"], "" if d["ranked"] else "  <- " + d["exclusion"][:60]))
        w = [f for f, _ in rank(st, by="weakness")]
        h = [f for f, _ in rank(st, by="headroom")]
        print("   weakness: %s" % (w or "NOTHING RANKABLE"))
        print("   headroom: %s" % (h or "NOTHING RANKABLE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
