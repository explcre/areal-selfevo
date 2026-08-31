#!/usr/bin/env python3
"""Seconds per training step, from the timestamps AReaL already writes.

The generation-count instrumentation does not reach the trainer in this controller layout, so
the resampling cost is measured the way a practitioner pays it: wall-clock per accepted step,
on identical hardware. An arm that regenerates to refill its batch spends that time in
rollout, and it shows up here without needing any counter to be exported.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from datetime import datetime

STAMP = re.compile(r"(\d{8})-(\d{2}:\d{2}:\d{2})\.\d+")
STEP = re.compile(r"step (\d+)/\d+")


def step_times(path: str) -> list[tuple[int, datetime]]:
    """(step number, timestamp) for each progress line that carries both."""
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m_s, m_t = STEP.search(line), STAMP.search(line)
            if m_s and m_t:
                d, t = m_t.group(1), m_t.group(2)
                out.append((int(m_s.group(1)), datetime.strptime(d + t, "%Y%m%d%H:%M:%S")))
    return out


def main() -> int:
    """Print seconds/step for each log given as ``label=path``."""
    rows = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        pts = step_times(path)
        if len(pts) < 3:
            print(f"{label:12s} only {len(pts)} timestamped steps; too few"); continue
        deltas = [
            (b[1] - a[1]).total_seconds()
            for a, b in zip(pts, pts[1:])
            if b[0] > a[0] and 0 < (b[1] - a[1]).total_seconds() < 3600
        ]
        if not deltas:
            print(f"{label:12s} no usable intervals"); continue
        rows[label] = deltas
        print(f"{label:12s} steps {pts[0][0]}-{pts[-1][0]}  n={len(deltas):3d}  "
              f"median {st.median(deltas):6.1f}s  mean {st.mean(deltas):6.1f}s")
    if len(rows) == 2:
        (la, a), (lb, b) = rows.items()
        # Median, not mean: a single stall would dominate a mean and overstate the ratio.
        print(f"\nratio {lb}/{la} on medians: {st.median(b)/st.median(a):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
