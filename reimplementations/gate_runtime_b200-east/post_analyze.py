#!/usr/bin/env python3
"""PRE vs POST, against the quantities fixed in PREREG_postscaffold.md."""
import json
import statistics
from collections import Counter

RUNS = {"PRE": "/mnt/localssd/gate/runs/PRE/iters.jsonl",
        "POST": "/mnt/localssd/gate/runs/POST/iters.jsonl"}
data = {a: [json.loads(l) for l in open(f)] for a, f in RUNS.items()}

print("=" * 96)
print("PRIMARY: task groups formed, and informative rollout groups")
print("=" * 96)
for a, rows in data.items():
    formed = [r for r in rows if r.get("formed_task_group")]
    inf = [r["informative_rollout_groups"] for r in formed]
    print("%-5s %d iterations, %d formed a task group (%.2f)   informative rollout groups in "
          "those: %s" % (a, len(rows), len(formed), len(formed) / len(rows),
                         ("%.3f" % statistics.fmean(inf)) if inf else "n/a"))

print()
print("%-5s %-4s %7s %6s %10s %9s %9s %9s %7s" %
      ("arm", "iter", "scored", "group", "tok_total", "tok_scaf", "tok_probe", "tok_roll", "sec"))
for a, rows in data.items():
    for r in rows:
        print("%-5s %-4d %7s %6s %10d %9d %9d %9d %7d"
              % (a, r["iter"], r.get("tasks_scored", 0), r.get("formed_task_group"),
                 r.get("tokens_total", 0), r.get("tokens_scaffold", 0),
                 r.get("tokens_probe", 0), r.get("tokens_rollout", 0), r.get("iter_s", 0)))

print()
print("COST -- the figure to beat is ~350k generated tokens per task reaching scoring")
for a, rows in data.items():
    tot = sum(r.get("tokens_total", 0) for r in rows)
    scored = sum(r.get("tasks_scored", 0) for r in rows)
    probe = sum(r.get("tokens_probe", 0) for r in rows)
    print("  %-5s %d tokens over %d iterations, %d tasks scored -> %.0fk per scored task"
          % (a, tot, len(rows), scored, tot / max(scored, 1) / 1000)
          + ("   (probe alone %.0f%% of spend)" % (100 * probe / tot) if probe else ""))
pre_t = sum(r.get("tokens_total", 0) for r in data["PRE"])
post_t = sum(r.get("tokens_total", 0) for r in data["POST"])
print("  POST costs %.2fx PRE per iteration; at EQUAL tokens PRE would get %.1fx the iterations"
      % (post_t / pre_t, post_t / pre_t))

print()
print("DOES THE SCAFFOLDED ESTIMATE PREDICT THE SCORED BLOCK? (POST only)")
drift, spread, pool_sel, post_sel = [], [], [], []
for r in data["POST"]:
    for d in (r.get("probe_drift") or []):
        if d is not None:
            drift.append(d)
    for ks in (r.get("probe_k_vectors") or []):
        ks = [k for k in ks if k is not None]
        if ks:
            spread.append(max(ks) - min(ks))
    pool_sel += r.get("pool_phat_selected") or []
    post_sel += [x for x in (r.get("selected_on") or []) if x is not None]
if drift:
    print("  candidate drift (scaffolded - bare): mean %+0.3f, mean |drift| %.3f, n=%d"
          % (statistics.fmean(drift), statistics.fmean(abs(d) for d in drift), len(drift)))
    print("  within-task spread of k across a task's own scaffolds: mean %.2f of 4, n=%d"
          % (statistics.fmean(spread), len(spread)))
print("  POST selected tasks whose SCAFFOLDED p_hat averaged %.3f (their bare p_hat: %.3f)"
      % (statistics.fmean(post_sel) if post_sel else float("nan"),
         statistics.fmean(pool_sel) if pool_sel else float("nan")))
pre_pool = [x for r in data["PRE"] for x in (r.get("selected_on") or [])]
print("  PRE  selected tasks whose BARE p_hat averaged %.3f" % statistics.fmean(pre_pool))
for a, rows in data.items():
    ph = [r["mean_p_hat"] for r in rows if r.get("formed_task_group")]
    if ph:
        print("  %-5s realised p_hat on the SCORED block: %s (mean %.3f)"
              % (a, [round(x, 3) for x in ph], statistics.fmean(ph)))

print()
print("REFUSALS, and the scaffold-aware counterfactual (their guard is left as published)")
for a, rows in data.items():
    ref = Counter()
    for r in rows:
        ref.update(r.get("refusals", {}))
    kvs = [ks for r in rows for ks in (r.get("refused_k_vectors") or [])]
    scaf_sig = sum(1 for ks in kvs if len({k for k in ks if k is not None}) > 1)
    print("  %-5s refusals=%s" % (a, {k.strip(): v for k, v in ref.items()}))
    print("        %d refused tasks; %d of them had DIFFERENT k across their scaffolds, i.e. "
          "carried scaffold-level signal that guard G4 discarded" % (len(kvs), scaf_sig))
    extra = 0
    for r in rows:
        if not r.get("formed_task_group"):
            n = r.get("tasks_scored", 0) + sum(
                1 for ks in (r.get("refused_k_vectors") or [])
                if len({k for k in ks if k is not None}) > 1)
            if n >= 2:
                extra += 1
    print("        iterations that WOULD have formed a task group under a scaffold-aware "
          "guard: %d more (%d of %d total)" % (extra,
          sum(1 for r in rows if r.get("formed_task_group")) + extra, len(rows)))

print()
print("TRUNCATION, measured on this configuration rather than inherited")
for a, rows in data.items():
    rt = [r["rollout_truncation"] for r in rows if r.get("rollout_truncation") is not None]
    pt = [r["probe_truncation"] for r in rows if r.get("probe_truncation") is not None]
    print("  %-5s scored-block truncation mean %.3f%s"
          % (a, statistics.fmean(rt) if rt else float("nan"),
             ("   probe-block %.3f" % statistics.fmean(pt)) if pt else ""))

print()
print("ITERATIONS THAT TRAINED")
for a, rows in data.items():
    for r in rows:
        if not r.get("formed_task_group"):
            continue
        print("  %-5s iter %d: p_hat %.3f R_task %.3g info(roll) %.2f of %d info(scaf) %.2f"
              % (a, r["iter"], r["mean_p_hat"], r["mean_R_task"],
                 r["informative_rollout_groups"], r["n_rollout_groups"],
                 r["informative_scaffold_groups"]))
        for st, u in r.get("updates", {}).items():
            print("        %-8s rows=%-3s tokens=%-7s grad_norm=%s"
                  % (st, u.get("rows"), u.get("tokens"), round(u.get("grad_norm") or 0, 5)))
