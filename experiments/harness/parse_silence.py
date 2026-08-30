#!/usr/bin/env python3
"""Extract the group-silence metrics from an AReaL train.log.

The metrics are printed in a multi-column box table, so a name and its value are adjacent in
the TOKEN stream but not on the same physical line -- pairing line-wise silently misaligns
names with values from a different column, which is how an early read of this log reported a
solved fraction that was really something else.
"""
from __future__ import annotations

import re
import statistics as st
import subprocess
import sys

KEYS = (
    "silent_group_fraction",
    "solved_group_fraction",
    "unsolved_group_fraction",
    "routed_group_fraction",
    "n_groups",
)
NUM = re.compile(r"^[0-9]\.[0-9]{4}e[+-][0-9]{2}$")


def parse(path: str) -> dict[str, list[float]]:
    """Pair each metric name with the numeric token that immediately follows it."""
    raw = subprocess.run(["sed", "s/│/\\n/g", path], capture_output=True, text=True).stdout
    toks = [t.strip() for t in raw.split("\n") if t.strip()]
    # Keep only names and numbers so an intervening label cannot break adjacency.
    toks = [t for t in toks if t in KEYS or NUM.match(t)]
    out: dict[str, list[float]] = {k: [] for k in KEYS}
    for i, t in enumerate(toks):
        if t in KEYS and i + 1 < len(toks) and NUM.match(toks[i + 1]):
            out[t].append(float(toks[i + 1]))
    return out


def main() -> int:
    d = parse(sys.argv[1])
    for k in KEYS:
        v = d[k]
        if not v:
            print(f"{k:26s} ABSENT")
            continue
        print(f"{k:26s} n={len(v):3d}  mean {st.mean(v):.4f}  last {v[-1]:.4f}  "
              f"range [{min(v):.4f}, {max(v):.4f}]")
    sil, sol, uns = d["silent_group_fraction"], d["solved_group_fraction"], d["unsolved_group_fraction"]
    m = min(len(sil), len(sol), len(uns))
    if m:
        resid = max(abs(sol[i] + uns[i] - sil[i]) for i in range(m))
        print(f"\nidentity solved+unsolved==silent: max residual {resid:.2e} over {m} batches")
    rt, so = d["routed_group_fraction"], d["solved_group_fraction"]
    if rt:
        m2 = min(len(rt), len(so))
        gap = max(abs(rt[i] - so[i]) for i in range(m2)) if m2 else float("nan")
        print(f"routed vs solved fraction:        max |diff| {gap:.2e} over {m2} batches")
        print(f"ROUTING IS {'FIRING' if max(rt) > 0 else 'INERT (all zero)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
