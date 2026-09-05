#!/usr/bin/env python3
"""The pre-registered spread gate: does a grouping identify weakness better than chance?

Runs exactly what `PREREG_grouping_gate.md` fixed before any statistic existed: weighted
between-group variance of per-group accuracy, against a size-matched permutation null of 2000
draws per arm, groups under five problems pooled into `other` before computing, each arm
against its own null because the arms have different group-size multisets.

The null permutes the ASSIGNMENT of problems to groups while holding each problem's measured
accuracy fixed. That is what makes it a fair reference: small groups inflate observed variance
through sampling noise, and the permutation null carries exactly the same inflation, so the
comparison is against noise of the right shape rather than against zero.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from ornith_repro.probe_split import load_committed, make_probe_split
from ornith_repro.weakness import load_outcomes


def weighted_between_variance(values: list, groups: list) -> float:
    """V = sum_g (n_g/N) (acc_g - acc_pool)^2.

    Args:
        values: Per-problem accuracy.
        groups: Per-problem group label, same order.

    Returns:
        The weighted between-group variance.
    """
    n = len(values)
    if n == 0:
        return 0.0
    pool = sum(values) / n
    agg = defaultdict(list)
    for v, g in zip(values, groups):
        agg[g].append(v)
    return sum((len(vs) / n) * ((sum(vs) / len(vs)) - pool) ** 2 for vs in agg.values())


def pool_small(groups: list, min_size: int = 5) -> list:
    """Pool groups below `min_size` into a single `other`, as pre-registered.

    Args:
        groups: Per-problem group labels.
        min_size: Minimum group size to survive.

    Returns:
        Relabelled groups.
    """
    counts = Counter(groups)
    return [g if counts[g] >= min_size else "other" for g in groups]


def permutation_test(values: list, groups: list, draws: int = 2000,
                     seed: int = 20260905) -> dict:
    """Compare observed spread against size-matched random partitions.

    Args:
        values: Per-problem accuracy.
        groups: Per-problem group label.
        draws: Null draws.
        seed: RNG seed.

    Returns:
        Dict with observed V, null median, p-value and effect ratio.
    """
    rng = random.Random(seed)
    obs = weighted_between_variance(values, groups)
    shuffled = list(groups)
    null = []
    for _ in range(draws):
        rng.shuffle(shuffled)
        null.append(weighted_between_variance(values, shuffled))
    null_sorted = sorted(null)
    med = null_sorted[len(null_sorted) // 2]
    ge = sum(1 for v in null if v >= obs)
    return {"observed": obs, "null_median": med,
            "null_p95": null_sorted[int(0.95 * len(null_sorted))],
            "p": (ge + 1) / (draws + 1),
            "ratio": (obs / med) if med > 0 else float("inf"),
            "n_groups": len(set(groups)), "n": len(values)}


def main():
    """Run every available arm through the pre-registered gate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--min-samples", type=int, default=8)
    a = ap.parse_args()

    src = "/home/ubuntu/reach/data/olympiadbench/test.jsonl"
    problems = [json.loads(l) for l in open(src) if l.strip()]
    coarse = {i: (r.get("subfield") or "unknown") for i, r in enumerate(problems)}
    search, report, md5 = load_committed(
        "/home/ubuntu/reach/bench/olympiadbench_split.json", src)
    probe, _train = make_probe_split(search, report, coarse, 0.35, 20260904)

    imap = json.load(open("/mnt/localssd/gate/searchhalf/index_map.json"))["original_index"]
    merged = defaultdict(list)
    import os
    for path in ("/mnt/localssd/gate/out/blocks_low.jsonl",
                 "/mnt/localssd/gate/out/blocks_low_more.jsonl"):
        if os.path.exists(path):
            for k, v in load_outcomes(path, problems, imap).items():
                merged[k].extend(v)

    fine = {}
    for line in open("/mnt/localssd/gate/out/fine_labels.jsonl"):
        if line.strip():
            r = json.loads(line)
            fine[r["idx"]] = r["label"]

    idxs, vals = [], []
    for i in probe:
        rows = merged.get(i, [])
        resolved = [c for c, t in rows if not t]
        if len(resolved) >= a.min_samples:
            idxs.append(i)
            vals.append(sum(1 for c in resolved if c) / len(resolved))
    print("probe problems with >= %d resolved samples: %d" % (a.min_samples, len(idxs)))

    arms = {"coarse_subfield": [coarse[i] for i in idxs],
            "fine_topic": [fine.get(i, "unlabelled") for i in idxs]}
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        k = len(set(pool_small(arms["fine_topic"])))
        X = TfidfVectorizer(stop_words="english", max_features=4000).fit_transform(
            [problems[i]["question"] for i in idxs])
        km = KMeans(n_clusters=max(2, k), n_init=10, random_state=0).fit(X)
        arms["tfidf_cluster"] = ["c%d" % c for c in km.labels_]
    except Exception as exc:  # noqa: BLE001
        print("cluster arm unavailable: %r" % (exc,))

    print("")
    print("%-18s %6s %6s %11s %11s %11s %8s %7s"
          % ("arm", "n", "groups", "V_obs", "null_med", "null_p95", "ratio", "p"))
    results = {}
    for name, labels in arms.items():
        pooled = pool_small(labels)
        res = permutation_test(vals, pooled, draws=a.draws)
        results[name] = res
        print("%-18s %6d %6d %11.6f %11.6f %11.6f %8.2f %7.4f"
              % (name, res["n"], res["n_groups"], res["observed"], res["null_median"],
                 res["null_p95"], res["ratio"], res["p"]))

    print("")
    passed = {k: v for k, v in results.items() if v["p"] < 0.05}
    for k, v in results.items():
        print("  %-18s %s (p=%.4f)" % (k, "PASSES" if v["p"] < 0.05 else "fails", v["p"]))
    print("")
    if not passed:
        print("REGISTERED OUTCOME: neither fine nor coarse beats a size-matched random")
        print("partition. Fine granularity identifies nothing on this pool. Do NOT build")
        print("weakness-targeted generation here; report as a negative.")
    elif "fine_topic" in passed and results["fine_topic"]["observed"] > results["coarse_subfield"]["observed"]:
        print("REGISTERED OUTCOME: fine passes and exceeds coarse -> proceed on fine labels.")
    elif "fine_topic" in passed:
        print("REGISTERED OUTCOME: fine passes but does not exceed coarse -> the finer")
        print("machinery is unnecessary complexity; proceed on coarse labels.")
    else:
        print("REGISTERED OUTCOME: coarse passes, fine does not -> proceed on coarse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
