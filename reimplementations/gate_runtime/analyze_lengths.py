#!/usr/bin/env python3
"""Measure the completion-length distribution, then choose a cap from it.

Four separate budget traps on this project have the same shape: a cap was chosen for the
length of the visible ANSWER while the model emits reasoning first, so the answer never
arrived and the failure read as a refusal. The cap is therefore set from a measured
distribution taken at a deliberately over-generous budget, and the quantile it is set at is
reported next to the truncation it implies.
"""
from __future__ import annotations

import argparse
import json
import statistics


def main() -> int:
    """Print the length distribution and the truncation each candidate cap would cause."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True)
    ap.add_argument("--probe-cap", type=int, required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.graded)]
    n = len(rows)
    errs = [r for r in rows if r.get("error")]
    tr = [r for r in rows if r.get("truncated")]
    # RESOLVED lengths only. A truncated generation's length is the cap, not its length, so
    # mixing them in makes every upper quantile equal the cap and hides the real tail.
    lens = sorted(r["completion_tokens"] for r in rows
                  if r.get("completion_tokens") is not None and not r.get("error")
                  and not r.get("truncated"))
    if not lens:
        raise SystemExit("no completion lengths recorded")

    def q(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]

    res = {
        "n_rows": n, "n_error": len(errs),
        "truncated_at_probe_cap": len(tr) / n, "probe_cap": a.probe_cap,
        "min": lens[0], "median": statistics.median(lens),
        "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "p995": q(0.995), "max": lens[-1],
        "mean": round(statistics.fmean(lens), 1),
        "n_resolved": len(lens),
        "accuracy_resolved": (sum(r["correct"] for r in rows if not r.get("truncated")
                                  and not r.get("error"))
                              / max(1, sum(1 for r in rows if not r.get("truncated")
                                           and not r.get("error")))),
    }
    # What each candidate cap would cost. A cap only truncates what exceeds it, and rows that
    # already truncated at the probe cap stay truncated at any smaller one.
    res["cap_table"] = {}
    n_ok = n - len(errs)
    for cap in (2048, 4096, 6144, 8192, 12288, 16384, 24576, a.probe_cap):
        # A generation that already ran past `cap` before finishing would have been cut at
        # `cap`; one that never finished at the probe cap never finishes at a smaller one.
        over = sum(1 for L in lens if L >= cap) + len(tr)
        res["cap_table"][cap] = {
            "truncation": round(over / n_ok, 4),
            "mean_tokens_per_rollout": round(
                (sum(min(L, cap) for L in lens) + len(tr) * cap) / n_ok, 1)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
