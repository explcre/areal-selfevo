#!/usr/bin/env python3
"""Convert training steps into GENERATION BUDGET, so arms that resample are compared fairly.

DAPO's dynamic sampling discards unanimous groups and regenerates, so at any given training
step it has produced more rollouts than an arm that keeps everything. Comparing at matched
steps would credit DAPO with accuracy it bought with extra inference. The honest axis is
total generations, and this converts one to the other.

Two sources, in order of preference:

  measured   ``rollout/accepted__count`` + ``rollout/rejected__count``. These are the real
             numbers, but they only reach the trainer in runs started after the
             RolloutController.export_stats fix -- before it the counts were used as a
             denominator and dropped, and the trainer saw a constant 1.0.
  derived    ``steps * batch_size * n_samples``. Exact for an arm with no filter, because
             nothing is rejected; wrong for a filtered arm, so it is refused there.

The source used is always printed. A number whose provenance is unclear is worse than no
number, and this is exactly the quantity a reviewer will push on.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

NUM = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:e[+-][0-9]{2})?$")
ACC, REJ = "rollout/accepted__count", "rollout/rejected__count"


def _tokens(path: str) -> list[str]:
    """Flatten an AReaL box-table log into a token stream of names and numbers."""
    raw = subprocess.run(["sed", "s/│/\\n/g", path], capture_output=True, text=True).stdout
    return [t.strip() for t in raw.split("\n") if t.strip()]


def counts(path: str) -> tuple[list[float], list[float]]:
    """Per-logged-batch accepted and rejected counts, empty when the run predates the fix."""
    toks = [t for t in _tokens(path) if t in (ACC, REJ) or NUM.match(t)]
    acc: list[float] = []
    rej: list[float] = []
    for i, t in enumerate(toks):
        if i + 1 < len(toks) and NUM.match(toks[i + 1]):
            if t == ACC:
                acc.append(float(toks[i + 1]))
            elif t == REJ:
                rej.append(float(toks[i + 1]))
    return acc, rej


def last_step(path: str) -> int:
    """Highest ``step N/M`` seen in the log."""
    seen = re.findall(r"step (\d+)/\d+", pathlib.Path(path).read_text(errors="ignore"))
    return max((int(x) for x in seen), default=0)


def report(path: str, label: str, batch_size: int, n_samples: int, filtered: bool) -> None:
    """Print an arm's generation budget and, when measurable, its resampling multiplier."""
    acc, rej = counts(path)
    steps = last_step(path)
    print(f"\n--- {label} ---")
    print(f"steps completed: {steps}")

    if acc and rej:
        # Cumulative or per-batch? Detect rather than assume: a cumulative series never
        # decreases, and mistaking one for the other changes the answer by orders of
        # magnitude.
        cumulative = all(b >= a for a, b in zip(acc, acc[1:]))
        kind = "cumulative" if cumulative else "per-batch"
        total_acc = acc[-1] if cumulative else sum(acc)
        total_rej = rej[-1] if cumulative else sum(rej)
        gens = (total_acc + total_rej) * n_samples
        mult = (total_acc + total_rej) / total_acc if total_acc else float("nan")
        print(f"source: MEASURED counts ({kind})")
        print(f"groups accepted {total_acc:.0f}  rejected {total_rej:.0f}")
        print(f"generations: {gens:,.0f}")
        print(f"resampling multiplier: {mult:.3f}x")
        return

    if filtered:
        print("source: UNAVAILABLE. This arm filters groups, so the derived estimate "
              "(steps x batch x n_samples) would undercount every rejected group and "
              "silently flatter it. Re-run with the export_stats fix, or compare on steps "
              "and say so.")
        return
    gens = steps * batch_size * n_samples
    print(f"source: DERIVED (run predates the export_stats fix; no filter, so nothing was "
          f"rejected and this is exact)")
    print(f"generations: {gens:,.0f}   multiplier: 1.000x by construction")


def main() -> int:
    """Report the generation budget of each arm passed as ``label=path``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="label=path/to/train.log")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--filtered", default="", help="comma-separated labels that filter groups")
    args = ap.parse_args()

    filtered = {s for s in args.filtered.split(",") if s}
    for arm in args.arms:
        if "=" not in arm:
            print(f"expected label=path, got {arm!r}", file=sys.stderr)
            return 2
        label, path = arm.split("=", 1)
        report(path, label, args.batch_size, args.n_samples, label in filtered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
