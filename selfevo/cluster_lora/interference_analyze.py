#!/usr/bin/env python3
"""Turn one gradient dump into the four-partition interference comparison.

This is the CPU half of the probe and it is where the method's central claim either survives
or does not. The claim is that behavioural subpopulations inside ONE task want conflicting
LoRA updates. Three separate objections have to be answered on the same batch, the same
checkpoint and the same gradients, or the number means nothing:

``meds`` vs ``random_matched``
    Is it the clustering, or just having more adapters? The control has the same number of
    clusters at the same sizes with feature-blind labels, so anything it reproduces is not
    a discovery.

``meds`` vs ``elrea``
    ELREA (arXiv 2502.00089) already clusters, trains a LoRA per cluster and merges -- in
    SFT, on PROMPT-token gradient features. If prompt-gradient clusters conflict as much as
    behavioural ones then rollouts buy nothing and this method is ELREA in RL clothing.

``meds`` vs ``task``
    arXiv 2608.03573 measures cross-TASK RL update cosine at about 1e-5 and publishes
    Parallel-RL (per-group LoRA under GRPO, then merge); arXiv 2602.12566 finds mixing beats
    merging in multi-domain RLVR. A reviewer will say the interference we target is
    negligible. Reproducing the cross-task figure on our own batch is the scale bar that
    makes the within-task figure a comparison instead of an assertion.

Per partition it reports pairwise cosine between per-cluster gradients, the conflict rate
(fraction of pairs with negative cosine), the cancellation ``||sum_c g_c|| / sum_c ||g_c||``,
the cluster sizes, and a bootstrap CI on the mean pairwise cosine resampled over GROUPS -- so
"MEDS conflicts more than random" is a difference with an error bar rather than two numbers.

**The instrument reports its own floor.** The gradients are CountSketches, and a sketched
cosine has standard error about ``1/sqrt(dim)``; every block carries ``resolution_floor`` and
a ``resolved`` flag, and a cross-task cosine of 1e-5 is reported as BELOW THE FLOOR rather
than as a measurement. The dump also stores a few full gradients, and ``sketch_validation``
compares exact cosines against sketched ones on those pairs, so the sketch is checked rather
than trusted.

Runs under ``~/venv_probe`` (numpy, scikit-learn, hdbscan). The training venv deliberately
has none of those and must not acquire them.

Usage::

    ~/venv_probe/bin/python -m selfevo.cluster_lora.interference_analyze \
        --dump dump.npz --out interference.json [--bootstrap 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

from .partition import (
    Partition,
    PartitionUnavailable,
    feature_partition,
    meds_partition,
    random_matched_partition,
    task_partition,
)
from .sketch import sketch_dim_resolution

__all__ = ["analyse_dump", "cluster_gradients", "partition_block", "pairwise_stats"]


def cluster_gradients(sketches: np.ndarray, partition: Partition) -> dict[str, np.ndarray]:
    """Per-cluster gradient as the SUM of its member groups' sketches.

    Exact, not an approximation, and it is the reason the probe can be split in two. The
    dump gives every group's loss the same denominator, so the batch loss is the sum of the
    group losses and the batch gradient is the sum of the group gradients; the sketch is
    linear, so summing sketches is sketching the sum. Any partition of the same dump is
    therefore free.

    Args:
        sketches: ``(n_groups, dim)``.
        partition: The partition to aggregate under.

    Returns:
        ``{adapter_name: (dim,) gradient}``.

    Raises:
        ValueError: If the shapes disagree.
    """
    if sketches.shape[0] != partition.n_groups:
        raise ValueError(
            f"{sketches.shape[0]} sketches but {partition.n_groups} labels; the partition "
            "does not describe this dump"
        )
    out: dict[str, np.ndarray] = {}
    for key, vec in zip(partition.keys, sketches):
        out[key] = out[key] + vec if key in out else vec.astype(np.float64).copy()
    return out


def pairwise_stats(grads: dict[str, np.ndarray]) -> dict[str, Any]:
    """Cosines, conflict rate and cancellation for one set of per-cluster gradients.

    Args:
        grads: ``{name: gradient}``.

    Returns:
        A dict of statistics. With fewer than two clusters the pairwise fields are ``None``
        rather than 0.0 -- an empty mean is not "no conflict", and reporting it as a number
        is how a degenerate partition comes to look like a favourable result.
    """
    names = sorted(grads)
    norms = {n: float(np.linalg.norm(grads[n])) for n in names}
    live = [n for n in names if norms[n] > 0.0]
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            cos = float(np.dot(grads[a], grads[b]) / (norms[a] * norms[b]))
            pairs.append({"a": a, "b": b, "cosine": cos})
    total = np.zeros_like(next(iter(grads.values())), dtype=np.float64)
    for n in names:
        total = total + grads[n]
    denom = sum(norms[n] for n in names)
    cosines = [p["cosine"] for p in pairs]
    return {
        "n_clusters": len(names),
        "n_clusters_with_gradient": len(live),
        "n_pairs": len(pairs),
        "mean_cosine": float(np.mean(cosines)) if cosines else None,
        "median_cosine": float(np.median(cosines)) if cosines else None,
        "min_cosine": float(np.min(cosines)) if cosines else None,
        "max_cosine": float(np.max(cosines)) if cosines else None,
        "conflict_rate": (
            float(np.mean([c < 0 for c in cosines])) if cosines else None
        ),
        "cancellation": float(np.linalg.norm(total) / denom) if denom > 0 else None,
        "cluster_norms": norms,
        "pairs": pairs,
    }


def _bootstrap_mean_cosine(
    sketches: np.ndarray, labels: Sequence[int], n_boot: int, seed: int
) -> dict[str, Any]:
    """Bootstrap CI on the mean pairwise cosine, resampling GROUPS.

    Groups are the independent unit here, not pairs: the pairwise cosines share cluster
    sums and are strongly dependent, so a CI computed over pairs would be far too narrow and
    would make every difference significant.

    A resample can empty a cluster; those replicates keep the clusters that survive, and the
    count of usable replicates is reported so a CI computed from a handful of them cannot be
    read as if it came from a thousand.

    Args:
        sketches: ``(n_groups, dim)``.
        labels: Cluster label per group.
        n_boot: Replicates.
        seed: Seed for a private generator.

    Returns:
        ``{"lo", "hi", "std", "n_effective"}``, all ``None`` when too few replicates
        produced two live clusters.
    """
    rng = np.random.default_rng(seed)
    labs = np.asarray(labels)
    n = len(labs)
    means: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        grads: dict[int, np.ndarray] = {}
        for j in idx:
            k = int(labs[j])
            grads[k] = grads[k] + sketches[j] if k in grads else sketches[j].astype(np.float64).copy()
        ks = [k for k in grads if np.linalg.norm(grads[k]) > 0]
        if len(ks) < 2:
            continue
        cs = []
        for i, a in enumerate(ks):
            for b in ks[i + 1 :]:
                cs.append(
                    float(
                        np.dot(grads[a], grads[b])
                        / (np.linalg.norm(grads[a]) * np.linalg.norm(grads[b]))
                    )
                )
        means.append(float(np.mean(cs)))
    if len(means) < 20:
        return {"lo": None, "hi": None, "std": None, "n_effective": len(means)}
    arr = np.array(means)
    return {
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "std": float(arr.std(ddof=1)),
        "n_effective": len(means),
    }


def partition_block(
    name: str,
    partition: Partition,
    sketches: np.ndarray,
    *,
    floor: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """One partition's full result block.

    Args:
        name: Block name, as it appears in the output JSON.
        partition: The partition.
        sketches: ``(n_groups, dim)`` per-group gradient sketches.
        floor: Sketch resolution floor, for the ``resolved`` verdict.
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        The block.
    """
    grads = cluster_gradients(sketches, partition)
    stats = pairwise_stats(grads)
    stats.pop("cluster_norms", None)
    mean = stats["mean_cosine"]
    return {
        "partition": name,
        "basis": partition.basis,
        "sizes": dict(partition.sizes),
        "size_multiset": list(partition.size_multiset()),
        "n_groups": partition.n_groups,
        "n_noise": partition.n_noise,
        **stats,
        "bootstrap_mean_cosine": _bootstrap_mean_cosine(
            sketches, partition.labels, n_boot, seed
        ),
        "resolution_floor": floor,
        # The verdict this instrument is entitled to give. A mean cosine under the floor is
        # not "a small conflict", it is "no conflict this sketch can see", and the two read
        # very differently in a paper.
        "resolved": (mean is not None and abs(mean) > floor),
    }


def _sketch_validation(dump, sketches: np.ndarray) -> dict[str, Any]:
    """Compare sketched cosines against exact ones on the groups whose full gradient was kept.

    The whole comparison rests on the sketch preserving angles. That is a theorem about
    expectations, not a guarantee about this batch, so it is measured: the same pairs are
    scored both ways and the disagreement is reported next to the results it qualifies.
    """
    full = dump.get("full_grad")
    if full is None or getattr(full, "size", 0) == 0 or full.shape[0] < 2:
        return {
            "n_groups": 0,
            "note": "no full gradients in the dump, so the sketch is unvalidated on this run",
        }
    m = full.shape[0]
    exact, approx = [], []
    for i in range(m):
        for j in range(i + 1, m):
            na, nb = np.linalg.norm(full[i]), np.linalg.norm(full[j])
            sa, sb = np.linalg.norm(sketches[i]), np.linalg.norm(sketches[j])
            if min(na, nb, sa, sb) == 0:
                continue
            exact.append(float(np.dot(full[i], full[j]) / (na * nb)))
            approx.append(float(np.dot(sketches[i], sketches[j]) / (sa * sb)))
    if not exact:
        return {"n_groups": m, "note": "every stored full gradient is zero"}
    err = np.abs(np.array(exact) - np.array(approx))
    return {
        "n_groups": m,
        "n_pairs": len(exact),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "exact_mean_cosine": float(np.mean(exact)),
        "sketched_mean_cosine": float(np.mean(approx)),
    }


def analyse_dump(
    path: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    control_seed: int = 0,
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    """Read a dump and produce all four partition blocks in one result.

    Args:
        path: The ``.npz`` written by ``interference_dump``.
        n_boot: Bootstrap replicates per block.
        seed: Bootstrap seed.
        control_seed: Seed for the size-matched random control.
        min_cluster_size: HDBSCAN's minimum cluster size. Defaults to MEDS' own 2, which is
            faithful but OVER-FRAGMENTS when the clusters are used to allocate adapters:
            measured 2026-09-02 on 24 points in four perfectly separated directional blobs
            of six, ``min_cluster_size=2`` returned six clusters and two noise points, while
            4 and 6 both recovered exactly four clusters of six with no noise. MEDS uses the
            clusters to shape a reward, where over-splitting is harmless; here every extra
            cluster is another expert trained on fewer groups, so the value is exposed and a
            run should sweep it rather than inherit it.

    Returns:
        The full result, ready to serialise.

    Raises:
        PartitionUnavailable: Only from the MEDS partition, which every other block depends
            on for its N and its sizes. A missing task label or a failed feature clustering
            is recorded as a SKIPPED block with its reason, not raised: three answered
            objections and one skipped is a usable result, and losing the other three to an
            exception is not.
    """
    dump = np.load(path, allow_pickle=True)
    sketches = np.asarray(dump["sketch"], dtype=np.float64)
    prompt_sketches = np.asarray(dump["prompt_sketch"], dtype=np.float64)
    feats = np.asarray(dump["meds_feature"], dtype=np.float64)
    gids = [str(x) for x in dump["group_id"]]
    tasks = [str(x) for x in dump["task"]]
    meta = json.loads(str(dump["meta"].item())) if "meta" in dump else {}
    floor = sketch_dim_resolution(sketches.shape[1])

    from selfevo.clustering.meds import MEDSClusterer

    meds = meds_partition(
        feats, clusterer=MEDSClusterer(min_cluster_size=min_cluster_size), group_ids=gids
    )
    blocks = [
        partition_block("meds", meds, sketches, floor=floor, n_boot=n_boot, seed=seed),
        partition_block(
            "random_matched",
            random_matched_partition(meds, seed=control_seed),
            sketches, floor=floor, n_boot=n_boot, seed=seed,
        ),
    ]

    # ELREA: same N and the same sizes as MEDS, so the only thing that differs is WHAT the
    # clustering looked at. Matching sizes matters -- a more lopsided partition has less
    # within-cluster averaging to do and would conflict more for a reason unrelated to the
    # feature.
    try:
        sizes = list(meds.size_multiset())
        elrea = feature_partition(
            prompt_sketches, n_clusters=len(sizes), match_sizes=sizes,
            seed=control_seed, tag="elrea (prompt-token gradient)", group_ids=gids,
        )
        blocks.append(
            partition_block("elrea", elrea, sketches, floor=floor, n_boot=n_boot, seed=seed)
        )
    except (PartitionUnavailable, ValueError) as exc:
        blocks.append({"partition": "elrea", "skipped": str(exc)})

    try:
        task = task_partition(tasks, group_ids=gids)
        blocks.append(
            partition_block("task", task, sketches, floor=floor, n_boot=n_boot, seed=seed)
        )
    except PartitionUnavailable as exc:
        blocks.append({"partition": "task", "skipped": str(exc)})

    by_name = {b["partition"]: b for b in blocks}
    contrasts = {}
    base = by_name.get("meds", {}).get("mean_cosine")
    for other in ("random_matched", "elrea", "task"):
        got = by_name.get(other, {}).get("mean_cosine")
        contrasts[f"meds_minus_{other}"] = (
            None if base is None or got is None else float(base - got)
        )
    return {
        "dump": path,
        "dump_meta": meta,
        "n_groups": len(gids),
        "sketch_dim": int(sketches.shape[1]),
        "min_cluster_size": int(min_cluster_size),
        "resolution_floor": floor,
        "sketch_validation": _sketch_validation(dump, sketches),
        "zero_block_fraction_mean": (
            float(np.mean(dump["zero_block_fraction"]))
            if "zero_block_fraction" in dump else None
        ),
        "partitions": blocks,
        # The headline the paper needs, stated as a DIFFERENCE. A bare "MEDS conflict rate
        # is 0.4" answers none of the three objections; the differences do.
        "contrasts": contrasts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-seed", type=int, default=0)
    p.add_argument("--min-cluster-size", type=int, default=2)
    a = p.parse_args(argv)
    result = analyse_dump(
        a.dump, n_boot=a.bootstrap, seed=a.seed, control_seed=a.control_seed,
        min_cluster_size=a.min_cluster_size,
    )
    text = json.dumps(result, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
