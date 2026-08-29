#!/usr/bin/env python3
"""Parse AReaL's boxed metric tables into aligned per-step rows.

Independently grepping each metric and pasting the streams together silently misaligns
as soon as two metrics have different occurrence counts -- it produced a table where
'entropy' read 823. This walks the log once and keys every value by the step it belongs
to, so a missing metric leaves a gap instead of shifting the column.

Usage: parse_metrics.py <log> [metric ...]
"""
from __future__ import annotations

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Train step (\d+)/(\d+) done")
# "  key   │  1.2345e+00 " inside the box-drawing table
ROW = re.compile(r"([A-Za-z_][\w/]*)\s*│\s*(-?\d+\.\d+e[+-]\d+)")

DEFAULT = [
    "update/entropy/avg",
    "task_reward/avg",
    "advantages/max",
    "correct_seq_len/avg",
    "n_seqs",
]


def main() -> None:
    path = sys.argv[1]
    keys = sys.argv[2:] or DEFAULT
    pending: dict[str, float] = {}
    rows: list[tuple[int, dict[str, float]]] = []
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = ANSI.sub("", raw)
            for k, v in ROW.findall(line):
                for want in keys:
                    if k == want or k.endswith("/" + want) or want.endswith("/" + k):
                        pending[want] = float(v)
            m = STEP.search(line)
            if m:                       # a step boundary flushes whatever was collected
                rows.append((int(m.group(1)), dict(pending)))
                pending = {}
    def label(k: str) -> str:
        # "update/entropy/avg" -> "entropy/avg"; keep enough to disambiguate the many
        # metrics whose last segment is just "avg".
        parts = k.split("/")
        return ("/".join(parts[-2:]) if len(parts) > 1 else k)[:11]

    hdr = "step | " + " | ".join(f"{label(k):>11}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for step, vals in rows:
        cells = []
        for k in keys:
            cells.append(f"{vals[k]:11.4g}" if k in vals else f"{'-':>11}")
        print(f"{step:4d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
