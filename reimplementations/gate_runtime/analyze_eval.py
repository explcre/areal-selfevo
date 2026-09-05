#!/usr/bin/env python3
"""Paired comparison of arms on the held-out half, with the interval on the DIFFERENCE.

Per-arm error bars are the wrong instrument for this question: the arms are scored on the SAME
items, so the item-to-item variance -- which dominates -- cancels in the difference and does
not cancel in two separate intervals. The pre-registration says to report the standard error
on the difference, and this computes exactly that, from the per-item paired differences.

An item's score is its success RATE over k samples, not a single Bernoulli draw, so the
paired-difference t interval is used rather than McNemar; McNemar on avg@k would throw away
the within-item resolution that k samples bought. The discordant-pair counts are printed
alongside for readers who want the McNemar view of the same data.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict


def per_item(path: str, truncated: str = "wrong") -> dict[int, float]:
    """Per-item success rate over one graded eval file, under a stated truncation convention.

    The convention is not a detail and the two answers differ by tens of points here, so both
    are reported rather than one being chosen quietly:

    * ``wrong`` -- a generation that hit the token cap did not produce an answer, so it scores
      zero. This is the BENCHMARK convention and the one the outcome claim uses: learning to
      finish inside the budget is a real capability gain and this is the only convention that
      can see it.
    * ``unknown`` -- truncated samples are dropped. This is the convention p-hat uses, because
      a gate that reads success rate must not be handed the token budget as if it were
      difficulty. It is reported alongside so a reader can see how much of any difference is
      termination rather than mathematics.

    An errored request (no response after retries) is dropped under both, because it is a
    property of the harness rather than of the model.
    """
    num, den = defaultdict(int), defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        if r.get("truncated"):
            if truncated == "unknown":
                continue
            num[r["idx"]] += 0
            den[r["idx"]] += 1
            continue
        num[r["idx"]] += int(r["correct"])
        den[r["idx"]] += 1
    return {i: num[i] / den[i] for i in den if den[i] > 0}


def compare(a: dict[int, float], b: dict[int, float], label: str) -> dict:
    """Paired difference ``a - b`` over the items both arms resolved."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    mean = statistics.fmean(d) if n else float("nan")
    sd = statistics.stdev(d) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    win = sum(1 for x in d if x > 0)
    loss = sum(1 for x in d if x < 0)
    mcnemar = None
    if win + loss > 0:
        # continuity-corrected McNemar on the discordant items
        chi = (abs(win - loss) - 1) ** 2 / (win + loss)
        mcnemar = {"discordant_a_better": win, "discordant_b_better": loss,
                   "chi2_cc": round(chi, 3)}
    return {"comparison": label, "n_items": n,
            "mean_a": round(statistics.fmean([a[k] for k in keys]), 4) if n else None,
            "mean_b": round(statistics.fmean([b[k] for k in keys]), 4) if n else None,
            "diff": round(mean, 4), "se_diff": round(se, 4) if se == se else None,
            "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)]
                    if se == se else None,
            "resolvable_at_1se": round(1.96 * se, 4) if se == se else None,
            "mcnemar": mcnemar}


def main() -> int:
    """Print every requested pairing plus each arm's own mean."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=path/to/graded.jsonl, repeatable")
    ap.add_argument("--against", default="", help="baseline arm name for every comparison")
    ap.add_argument("--truncated", default="wrong", choices=["wrong", "unknown"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = dict(spec.split("=", 1) for spec in a.arm)
    scores = {n: per_item(p, a.truncated) for n, p in paths.items()}
    alt = {n: per_item(p, "unknown" if a.truncated == "wrong" else "wrong")
           for n, p in paths.items()}
    out = {"truncation_convention": a.truncated,
           "per_arm": {n: {"n_items": len(s),
                           "mean": round(statistics.fmean(s.values()), 4),
                           "mean_other_convention": round(statistics.fmean(alt[n].values()), 4)}
                       for n, s in scores.items()}, "comparisons": []}
    names = list(scores)
    if a.against:
        pairs = [(n, a.against) for n in names if n != a.against]
    else:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    for x, y in pairs:
        out["comparisons"].append(compare(scores[x], scores[y], "%s - %s" % (x, y)))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
