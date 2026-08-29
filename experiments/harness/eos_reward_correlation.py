#!/usr/bin/env python3
"""Test whether all-EOS failures are injected into the reward as zeros, or dropped.

I have repeatedly claimed that AReaL "launders infrastructure failures into the training
signal as reward 0.0". The evidence for that was reading `_set_last_reward(..., 0.0)` in
the exception handler. But the same handler then does `return None`, which may mean the
trajectory is dropped and replaced instead -- in which case the reward is untouched and my
claim is wrong.

This distinguishes them from the logs, without needing to trace the code further.

If N failures out of a 1024-sequence batch were injected as zeros, then
    observed_reward ~= true_reward * (1024 - N) / 1024
so reward should fall measurably as the per-step failure count rises. The pre-fix run went
from 2 failures at one step to 120 cumulative a few steps later; 120/1024 = 11.7%, which
would drag a 0.80 reward down to ~0.71 -- easily visible.

If instead reward is uncorrelated with the per-step failure count, the failures are being
dropped and replaced, and the claim should be retracted.

Usage: eos_reward_correlation.py <train.log>
"""
from __future__ import annotations

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Train step (\d+)/(\d+) done")
ROW = re.compile(r"(task_reward/avg|n_seqs)\s*│\s*(-?\d+\.\d+e[+-]\d+)")
EOS = "All output_tokens are EOS or PAD"

rows: list[tuple[int, float, float, int]] = []
pending: dict[str, float] = {}
eos_in_window = 0

with open(sys.argv[1], errors="replace") as fh:
    for raw in fh:
        line = ANSI.sub("", raw)
        if EOS in line:
            eos_in_window += 1
        for k, v in ROW.findall(line):
            pending[k] = float(v)
        m = STEP.search(line)
        if m:
            rows.append((int(m.group(1)),
                         pending.get("task_reward/avg", float("nan")),
                         pending.get("n_seqs", float("nan")),
                         eos_in_window))
            pending = {}
            eos_in_window = 0

print(f"{'step':>5} {'reward':>8} {'n_seqs':>8} {'eos_errs':>9}")
for s, r, n, e in rows:
    print(f"{s:5d} {r:8.4f} {n:8.0f} {e:9d}")

usable = [(r, e) for _, r, n, e in rows if r == r and e >= 0]
if len(usable) >= 6:
    ys = [r for r, _ in usable]
    xs = [float(e) for _, e in usable]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    print()
    if sxx == 0:
        print("no variation in failure counts; inconclusive")
    else:
        r_pearson = sxy / (sxx * syy) ** 0.5 if syy > 0 else float("nan")
        slope = sxy / sxx
        print(f"steps with failures: {sum(1 for x in xs if x > 0)}/{n}, max {max(xs):.0f}")
        print(f"corr(failures, reward) = {r_pearson:+.3f}, slope = {slope:+.5f} reward per failure")
        print(f"predicted slope if each failure were a zero in a 1024 batch: "
              f"{-my / 1024:+.5f}")
