#!/usr/bin/env python3
"""COLD vs SEED, against the quantities fixed in PREREG_seeded_loop.md."""
import json
import statistics
from collections import Counter

RUNS = {"COLD": "/mnt/localssd/gate/runs/COLD/iters.jsonl",
        "SEED": "/mnt/localssd/gate/runs/SEED/iters.jsonl"}
data = {}
for arm, f in RUNS.items():
    try:
        data[arm] = [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        data[arm] = []

print("=" * 94)
print("PRIMARY: iterations that formed a task group and took a gradient step")
print("=" * 94)
for arm, rows in data.items():
    formed = [r for r in rows if r.get("tasks_scored", 0) >= 2]
    seeded = [r for r in rows if r.get("task_source") == "seed"]
    prop = [r for r in rows if r.get("task_source") != "seed"]
    print("%-5s %d iterations: %d formed a task group   (seeded %d/%d, proposer-driven %d/%d)"
          % (arm, len(rows), len(formed),
             sum(1 for r in seeded if r.get("tasks_scored", 0) >= 2), len(seeded),
             sum(1 for r in prop if r.get("tasks_scored", 0) >= 2), len(prop)))

print()
print("%-5s %-5s %-9s %8s %8s %10s %8s" %
      ("arm", "iter", "source", "accepted", "scored", "gen_tok", "sec"))
for arm, rows in data.items():
    for r in rows:
        print("%-5s %-5d %-9s %8s %8s %10s %8s"
              % (arm, r["iter"], r.get("task_source", "?"), r.get("tasks_accepted"),
                 r.get("tasks_scored", 0), r.get("gen_tokens") or "-",
                 int(r["iter_s"]) if r.get("iter_s") else "-"))

print()
print("WHY TASKS WERE REFUSED, and the k-vectors that say which degeneration")
for arm, rows in data.items():
    ref = Counter()
    for r in rows:
        ref.update(r.get("refusals", {}))
    ks = [kv for r in rows for kv in r.get("refused_k_vectors", [])]
    allk = [k for kv in ks for k in kv["k_per_scaffold"] if k is not None]
    G = ks[0]["G"] if ks else 4
    print("  %-5s refusals=%s" % (arm, {k.strip(): v for k, v in ref.items()}))
    if allk:
        print("        %d refused tasks, %d scaffold groups: all-correct %d, all-wrong %d, "
              "mixed %d" % (len(ks), len(allk), sum(1 for k in allk if k == G),
                            sum(1 for k in allk if k == 0),
                            sum(1 for k in allk if 0 < k < G)))
        scaf_informative = sum(1 for kv in ks
                               if len({k for k in kv["k_per_scaffold"] if k is not None}) > 1)
        print("        of those refused tasks, %d of %d had DIFFERENT k across their "
              "scaffolds, i.e. carried scaffold-level signal that guard G4 discarded"
              % (scaf_informative, len(ks)))

print()
print("SECONDARY: proposer behaviour on proposer-driven iterations")
for arm, rows in data.items():
    prop = [r for r in rows if r.get("task_source") != "seed" and r.get("proposer_calls")]
    if not prop:
        print("  %-5s no proposer-driven iteration completed" % arm)
        continue
    calls = sum(r["proposer_calls"] for r in prop)
    cap = sum(r["proposer_hit_cap"] for r in prop)
    tok = sum(r["proposer_tokens"] for r in prop)
    acc = sum(r["tasks_accepted"] for r in prop)
    print("  %-5s %d attempts, never terminated %d (%.3f), accepted %d (yield %.3f), "
          "%d proposer tokens" % (arm, calls, cap, cap / calls, acc, acc / calls, tok))

print()
print("ITERATIONS THAT TRAINED, in full")
for arm, rows in data.items():
    for r in rows:
        if r.get("tasks_scored", 0) < 2:
            continue
        print("  %-5s iter %d (%s): p_hat %.3f  N %.3f  info(roll) %.2f of %d  info(scaf) %.2f "
              "of %d  R_task %.3g" % (arm, r["iter"], r.get("task_source"), r["mean_p_hat"],
                                      r["mean_novelty_N"], r["informative_rollout_groups"],
                                      r["n_rollout_groups"], r["informative_scaffold_groups"],
                                      r["n_scaffold_groups"], r["mean_R_task"]))
        for st, u in r["updates"].items():
            print("        %-9s rows=%-3s tokens=%-7s grad_norm=%s"
                  % (st, u.get("rows"), u.get("tokens"), round(u.get("grad_norm") or 0, 5)))

print()
print("COST")
for arm, rows in data.items():
    tot = sum(r.get("gen_tokens") or 0 for r in rows)
    scored = sum(r.get("tasks_scored", 0) for r in rows)
    sec = sum(r.get("iter_s") or 0 for r in rows)
    print("  %-5s %d generated tokens over %d iterations, %d tasks reached scoring"
          % (arm, tot, len(rows), scored)
          + ("  -> %.0fk per scored task" % (tot / scored / 1000) if scored else ""))
