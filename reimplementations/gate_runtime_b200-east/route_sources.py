#!/usr/bin/env python3
"""Audit the permanent store and route across all three sources from what it records.

The profiles are derived from the STORE rather than from the run's summary table, because the
summary reported one `cost_tokens` column that was production and verification added together.
Those are different purchases: production is a property of the source, while verification cost
follows from how long the resulting problems are, so a corpus that is free to read can still be
the most expensive source to trust. The store keeps them apart per record, so the store is the
only honest place to derive a cost ranking from.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, "/mnt/localssd/gate/code")

from tasksource.routing import SourceProfile, plan  # noqa: E402
from tasksource.store import TaskStore  # noqa: E402

STORE = "/mnt/localssd/gate/out/taskstore/tasks.v3.jsonl"

STRUCTURE = {
    "generated": dict(supply="unbounded", targetable=True, collusion=True,
                      licence_constrained=False,
                      notes="the solver writes its own tasks; collusion is total"),
    "retrieved": dict(supply="bounded", targetable=False, collusion=False,
                      licence_constrained=True,
                      notes="fixed pool; keys human-curated, but pretraining contamination "
                            "is not measurable from here"),
    "distilled": dict(supply="unbounded", targetable=True, collusion=True,
                      licence_constrained=False,
                      notes="the teacher was the SAME checkpoint as the solver, so this row "
                            "is not evidence about a stronger teacher; no paid call was made"),
}


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval, so a rate over few decided cases is not read as a point."""
    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def main() -> int:
    """Print the store audit, the per-source table and the routing plan."""
    recs = TaskStore.read(STORE)
    by = defaultdict(list)
    for r in recs:
        by[r["source_type"]].append(r)

    print("store: %s   %d records, schema v%d" % (STORE, len(recs), recs[0]["schema_version"]))
    print("\nAUDIT: every record carries every required field")
    need = {"task_id", "content_hash", "source_type", "provenance", "verification",
            "difficulty", "dedup", "cost", "schema_version", "created_utc"}
    assert all(need <= set(r) for r in recs), "a record is missing a top-level field"
    ids = [r["task_id"] for r in recs]
    print("  unique task ids           %d/%d" % (len(set(ids)), len(ids)))
    print("  content hash == id prefix %s"
          % all(r["content_hash"].startswith(r["task_id"]) for r in recs))
    print("  three dedup sets w/ floor %s"
          % all(all(k in r["dedup"] and "false_positive_floor" in r["dedup"][k]
                    for k in ("held_out", "training_pool", "run_buffer")) for r in recs))
    print("  every refutation w/ witness %s"
          % all(r["verification"].get("witness")
                for r in recs if r["verification"]["verdict"] == "refuted"))
    print("  every success rate named its prompt %s"
          % all(r["difficulty"].get("measured_under_prompt")
                for r in recs if r["difficulty"].get("success_rate") is not None))

    print("\nper-source, from the store")
    hdr = ("source", "n", "verified", "refuted", "unverif", "cov", "ref_rate", "wilson95",
           "prod_tok", "ver_tok", "p_solve")
    print("%-10s %3s %8s %7s %7s %6s %8s %-16s %8s %8s %7s" % hdr)
    profiles = []
    for name in ("generated", "retrieved", "distilled"):
        rs = by.get(name, [])
        if not rs:
            continue
        v = [r["verification"]["verdict"] for r in rs]
        dec = v.count("verified") + v.count("refuted")
        ref = v.count("refuted")
        cov = dec / len(rs)
        rate = ref / dec if dec else float("nan")
        lo, hi = wilson(ref, dec)
        prod = sum(r["cost"]["produce_tokens"] for r in rs) / len(rs)
        ver = sum(r["cost"]["score_tokens"] for r in rs) / len(rs)
        sr = [r["difficulty"]["success_rate"] for r in rs
              if r["difficulty"].get("success_rate") is not None]
        print("%-10s %3d %8d %7d %7d %6.3f %8.4f [%.3f, %.3f]   %8d %8d %7.3f"
              % (name, len(rs), v.count("verified"), ref, v.count("unverifiable"),
                 cov, rate, lo, hi, prod, ver, sum(sr) / len(sr) if sr else float("nan")))
        profiles.append(SourceProfile(
            name=name, key_refuted_rate=round(rate, 4), key_coverage=round(cov, 4),
            tokens_per_accepted=int(prod), verify_tokens_per_accepted=int(ver),
            usd_per_accepted=None,    # every arm ran on local hardware; money is unmeasured
            **STRUCTURE[name]))

    print("\ndedup, mean max-similarity per reference set (all kept records are below "
          "their thresholds by construction)")
    print("%-10s %14s %14s %14s" % ("source", "held_out", "training_pool", "run_buffer"))
    for name in ("generated", "retrieved", "distilled"):
        rs = by.get(name, [])
        if not rs:
            continue
        cells = []
        for k in ("held_out", "training_pool", "run_buffer"):
            sc = [r["dedup"][k]["score"] for r in rs]
            above = sum(1 for r in rs if r["dedup"][k]["above_floor"])
            cells.append("%.3f (%d>fl)" % (sum(sc) / len(sc), above))
        print("%-10s %14s %14s %14s" % (name, *cells))

    print("\nrouting plan (all three used; no winner declared)")
    pl = plan(profiles)
    for need_name, ranked in pl.items():
        print("  %s" % need_name)
        for i, (nm, why) in enumerate(ranked):
            print("    %d. %-10s %s" % (i + 1, nm, why))

    json.dump({"store": STORE, "n": len(recs),
               "profiles": [p.__dict__ for p in profiles], "plan": pl},
              open("/mnt/localssd/gate/out/tasksource/routing_plan.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
