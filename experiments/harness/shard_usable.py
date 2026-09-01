"""Is a scoring shard's results.json a real measurement, or a run that measured nothing?

Exit 0 when usable, nonzero otherwise. Existence and non-emptiness are not enough: a run whose
every request was rejected -- which is what a cap larger than the model's context does --
completes normally and writes a valid file full of nulls.
"""
import json
import sys

try:
    rows = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
rows = rows if isinstance(rows, list) else [rows]
if not rows:
    sys.exit(1)
for r in rows:
    acc = r.get("accuracy")
    if acc is None or acc != acc:          # None, or a NaN that survived the round-trip
        sys.exit(1)
    n_prob = r.get("n_problems") or 0
    n_fail = r.get("n_failed") or 0
    # Above this share of failed requests the value is an average over survivors, which is
    # biased upward -- measured at about +0.08 on one benchmark -- not a conservative estimate.
    if n_prob and n_fail / n_prob > 0.10:
        sys.exit(1)
sys.exit(0)
