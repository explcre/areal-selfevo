"""Per-category weakness, with the index space verified rather than assumed.

WHY THIS MODULE REPLACES THE SCRIPT IT CAME FROM. An audit reproduced three defects that could
each flip the ranking, and all three were invisible:

1. **Report-half measurements entered probe statistics while `probe INTERSECT report = 0`
   still printed zero.** The overlap was in the MEASUREMENT index space. Rollout rows carry a
   position in whatever problem file produced them; that position was mapped to an original
   index through a list that was assumed, never checked, to correspond to the current search
   half. With stale blocks and a correctly regenerated map every set-level check passed while
   11 of 21 probe indices drew their numbers from a report problem. Four different corrupt
   maps completed silently and produced four different rankings.

2. **The disjointness guard could not fire.** Probe is built only from search, and loading
   already refuses a search that intersects report, so `probe & report` is empty by
   construction. Twenty thousand fuzzed inputs fired one of five error paths. The tests made
   the rest fail by calling the assertion directly with hand-built lists, which proves the
   statements execute, not that the pipeline can reach them.

3. **The minimum-rollout filter conditioned on resolution.** Truncation is not random -- hard
   problems truncate -- so the filter dropped the hardest problems first and the surviving
   accuracy was conditioned on the model finishing. On identical problems with only truncation
   added, one category inflated by 16 points and moved from rank one to rank three, and
   nothing reported how many problems had been dropped.

THE FIX FOR (1) IS NOT A BETTER MAP. Pinning the map does not help when the map is right and
the BLOCKS are stale. So the index space is verified against content: every rollout row records
the gold answer it was graded against, and that gold must match the answer of the problem the
row is mapped to. A stale map or stale blocks fails immediately and loudly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict


class WeaknessError(RuntimeError):
    """Raised when a measurement cannot be trusted. Never downgraded to a warning."""


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial proportion, correct at the boundaries.

    Args:
        k: Successes, possibly fractional when averaging per-problem rates.
        n: Trials.
        z: Normal quantile.

    Returns:
        `(lo, hi)`, or `(nan, nan)` when n is zero.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))
    # Clamped: at k=0 the algebraic form returns -2e-17, which is not a probability and
    # makes a boundary test fail for a reason that has nothing to do with the statistics.
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def load_outcomes(blocks_path: str, problems: list, index_map: list) -> dict:
    """Map rollout rows to ORIGINAL problem indices, verifying the mapping against content.

    The verification is the point. `index_map[position]` claims which original problem a
    rollout row refers to; that claim is checked by comparing the gold answer the row was
    graded against with the answer of the problem it maps to. A mismatch means the blocks, the
    map, or the problem file disagree, and any ranking computed from them is meaningless.

    Args:
        blocks_path: Rollout JSONL.
        problems: The full original problem list, indexed by original index.
        index_map: Position -> original index.

    Returns:
        Mapping original index -> list of (correct, truncated) pairs.

    Raises:
        WeaknessError: on any gold mismatch, naming the position and both answers.
    """
    out = defaultdict(list)
    checked = mismatched = 0
    examples = []
    with open(blocks_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pos = r.get("idx")
            if pos is None or pos >= len(index_map):
                raise WeaknessError(
                    "rollout row has position %r but the index map has %d entries; the "
                    "blocks and the map describe different problem files"
                    % (pos, len(index_map)))
            orig = index_map[pos]
            row_gold = str(r.get("gold", "")).strip()
            if row_gold:
                checked += 1
                fa = problems[orig].get("final_answer")
                want = str(fa[0] if isinstance(fa, list) and fa else fa).strip()
                if row_gold != want:
                    mismatched += 1
                    if len(examples) < 3:
                        examples.append((pos, orig, row_gold[:30], want[:30]))
            if r.get("status") != "ok":
                continue
            out[orig].append((bool(r.get("correct")),
                              r.get("finish_reason") == "length"))
    if mismatched:
        raise WeaknessError(
            "%d of %d rollout rows were graded against a different answer than the problem "
            "they map to (e.g. %s). The blocks, the index map or the problem file are out of "
            "step, and a ranking computed from them would attribute one problem's difficulty "
            "to another. Refusing." % (mismatched, checked, examples))
    if not checked:
        raise WeaknessError(
            "no rollout row carried a gold answer, so the position-to-problem mapping could "
            "not be verified against content. Refusing to rank on an unverified index space.")
    return dict(out)


def category_stats(outcomes: dict, fields: dict, probe: list, min_samples: int = 8,
                   max_drop_rate: float = 0.30, max_interval_width: float = 0.35) -> dict:
    """Per-category accuracy, with drops counted and unreliable categories EXCLUDED.

    Two exclusions, both refusals rather than annotations:

    * a category whose drop rate exceeds `max_drop_rate` is not ranked, because its surviving
      problems are conditioned on the model finishing and truncation is not random;
    * a category whose accuracy interval is wider than `max_interval_width` is not ranked,
      because the earlier version printed the warning AFTER the ranking and the leading
      category was flagged and ranked anyway.

    Args:
        outcomes: Original index -> list of (correct, truncated).
        fields: Original index -> category label.
        probe: Original indices forming the probe.
        min_samples: Resolved samples a problem needs to contribute.
        max_drop_rate: Above this share dropped, the category is not ranked.
        max_interval_width: Above this width, the category is not ranked.

    Returns:
        Mapping category -> stats dict, including `ranked` and `exclusion` for every category.
    """
    per = defaultdict(lambda: {"seen": 0, "used": 0, "dropped_few": 0, "dropped_none": 0,
                               "acc_sum": 0.0, "mixed": 0, "always": 0, "never": 0,
                               "trunc_rate_sum": 0.0})
    for idx in probe:
        f = fields.get(idx, "unknown")
        d = per[f]
        d["seen"] += 1
        rows = outcomes.get(idx, [])
        if not rows:
            d["dropped_none"] += 1
            continue
        resolved = [c for c, t in rows if not t]
        d["trunc_rate_sum"] += sum(1 for _, t in rows if t) / len(rows)
        if len(resolved) < min_samples:
            d["dropped_few"] += 1
            continue
        p = sum(1 for c in resolved if c) / len(resolved)
        d["used"] += 1
        d["acc_sum"] += p
        d["always"] += (p == 1.0)
        d["never"] += (p == 0.0)
        d["mixed"] += (0.0 < p < 1.0)

    out = {}
    for f, d in per.items():
        used, seen = d["used"], d["seen"]
        dropped = d["dropped_few"] + d["dropped_none"]
        drop_rate = dropped / seen if seen else 1.0
        acc = d["acc_sum"] / used if used else float("nan")
        lo, hi = wilson(d["acc_sum"], used) if used else (float("nan"), float("nan"))
        width = hi - lo if used else float("inf")
        reasons = []
        if used == 0:
            reasons.append("no problem survived the sample filter")
        if drop_rate > max_drop_rate:
            reasons.append("drop rate %.3f exceeds %.2f; surviving accuracy is conditioned "
                           "on the model finishing" % (drop_rate, max_drop_rate))
        if used and width > max_interval_width:
            reasons.append("interval width %.3f exceeds %.2f" % (width, max_interval_width))
        out[f] = {"seen": seen, "used": used, "dropped": dropped,
                  "dropped_few": d["dropped_few"], "dropped_none": d["dropped_none"],
                  "drop_rate": drop_rate, "acc": acc, "lo": lo, "hi": hi, "width": width,
                  "mixed": d["mixed"] / used if used else float("nan"),
                  "always": d["always"] / used if used else float("nan"),
                  "never": d["never"] / used if used else float("nan"),
                  "mean_trunc": d["trunc_rate_sum"] / seen if seen else float("nan"),
                  "ranked": not reasons, "exclusion": "; ".join(reasons)}
    return out


def rank(stats: dict, by: str = "headroom") -> list:
    """Order the RANKABLE categories only.

    Args:
        stats: Output of `category_stats`.
        by: "headroom" ranks by the supply of mixed problems, descending; "weakness" ranks by
            accuracy, ascending.

    Returns:
        List of (category, stats) for categories that passed every exclusion.

    Raises:
        WeaknessError: on an unknown key, rather than silently defaulting to one of them.
    """
    if by not in ("headroom", "weakness"):
        raise WeaknessError("rank key must be headroom|weakness, got %r" % by)
    ok = [(f, d) for f, d in stats.items() if d["ranked"]]
    if by == "headroom":
        return sorted(ok, key=lambda kv: -kv[1]["mixed"])
    return sorted(ok, key=lambda kv: kv[1]["acc"])
