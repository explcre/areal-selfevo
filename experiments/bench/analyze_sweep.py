#!/usr/bin/env python3
"""Separate capability loss from format collapse across the step0d checkpoint series.

Headline accuracy counts an unparseable or truncated generation as wrong, so a model that
still knows the maths but has stopped emitting \\boxed{} is indistinguishable from one that
has forgotten it. The paper's claim depends on telling those apart, so this reports:

  acc          overall, the number a benchmark table would print
  acc|boxed    accuracy among generations that produced a parseable answer
  nobox        fraction with no parseable answer at all
  trunc        fraction that ran into the token cap
  len          mean completion length -- the mechanism behind trunc

A decline in `acc` with `acc|boxed` flat means the model lost the format, not the skill.
"""
from __future__ import annotations
import json, pathlib, sys, math

SUITE = pathlib.Path(sys.argv[1])
TAGS = ["base", "gs028", "gs057", "gs086", "gs115", "gs144", "gs173"]
# Entropy and train reward at each checkpoint, read from the training log earlier.
META = {"base": (None, None), "gs028": (0.2533, 0.5469), "gs057": (0.1344, 0.2090),
        "gs086": (0.1436, 0.7207), "gs115": (0.0989, 0.4932), "gs144": (0.0253, 0.5459),
        "gs173": (0.0182, 0.4814)}

# Kept byte-identical to math_bench.wilson and regrade.wilson: the three copies disagreed
# at n=0 (this one printed a confident [0.000, 0.000] for an empty benchmark) and in the
# fifth decimal everywhere else. selfevo/ARCHITECTURE.md 4.2 consolidates them; until then
# any edit here belongs in all three.
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Used instead of a normal binomial SE because at these counts the SE misleads: at 1/30
    the normal interval runs negative, and at 0/30 or 30/30 it is exactly 0, asserting
    certainty from a single unanimous sample.

    ``n <= 0`` is NOT a measurement and returns NaN. Two copies of this function used to
    return ``(0.0, 0.0)`` there, which prints ``[0.000, 0.000]`` for an EMPTY benchmark --
    a confident interval around zero, indistinguishable in the table from a real result of
    zero. A negative n is refused for the same reason: the clamp inside the square root
    hides it and the interval comes back with ``lo > hi``, outside [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))

print(f"{'ckpt':6} {'entropy':>8} {'train_r':>8} | {'acc':>6} {'95% CI':>15} | "
      f"{'acc|boxed':>10} {'n_boxed':>8} | {'nobox':>6} {'trunc':>6} {'len_ch':>7}")
print("-" * 104)
rows = []
for tag in TAGS:
    f = SUITE / tag / "generations.jsonl"
    if not f.exists():
        print(f"{tag:6} {'(not finished)':>60}")
        continue
    g = [json.loads(l) for l in f.open() if l.strip()]
    n = len(g)
    if n == 0:
        print(f"{tag:6} EMPTY ARTIFACT -- refusing to score"); continue
    correct = sum(1 for r in g if r.get("correct"))
    boxed = [r for r in g if r.get("boxed")]
    cb = sum(1 for r in boxed if r.get("correct"))
    nbox = n - len(boxed)
    trunc = sum(1 for r in g if r.get("finish_reason") == "length")
    mlen = sum(len(r.get("text", "")) for r in g) / n  # characters, not tokens
    # Cross-check against the authoritative score. A key-name mismatch silently reported
    # 0.000 for every checkpoint once; a zero must be a measurement, never a typo.
    rj = SUITE / tag / "results.json"
    if rj.exists():
        try:
            recs = json.loads(rj.read_text())
            auth = next(r["accuracy"] for r in recs if r["benchmark"] == "math500")
            if abs(auth - correct / n) > 1e-6:
                sys.exit(f"{tag}: recomputed acc {correct/n:.4f} != results.json {auth:.4f} "
                         f"-- the artifact schema changed; fix the key names, do not report this")
        except (KeyError, TypeError, ValueError, StopIteration) as e:
            sys.exit(f"{tag}: results.json unreadable ({e!r}) -- refusing to report an unverified score")
    lo, hi = wilson(correct, n)
    ent, tr = META.get(tag, (None, None))
    accb = cb / len(boxed) if boxed else float("nan")
    print(f"{tag:6} {ent if ent is not None else '-':>8} {tr if tr is not None else '-':>8} | "
          f"{correct/n:6.3f} [{lo:.3f},{hi:.3f}] | {accb:10.3f} {len(boxed):8d} | "
          f"{nbox/n:6.3f} {trunc/n:6.3f} {mlen:7.0f}")
    rows.append((tag, correct/n, accb, nbox/n, trunc/n, mlen))

if len(rows) >= 2:
    b, l = rows[0], rows[-1]
    print(f"\n{b[0]} -> {l[0]}:  acc {b[1]:.3f} -> {l[1]:.3f} ({l[1]-b[1]:+.3f})   "
          f"acc|boxed {b[2]:.3f} -> {l[2]:.3f} ({l[2]-b[2]:+.3f})   "
          f"nobox {b[3]:.3f} -> {l[3]:.3f}   len {b[5]:.0f} -> {l[5]:.0f}")
    print("\nIf acc falls while acc|boxed holds, the loss is format, not capability.")
