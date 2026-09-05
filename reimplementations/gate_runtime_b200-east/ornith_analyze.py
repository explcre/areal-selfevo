#!/usr/bin/env python3
"""What the three-stage loop did, against the conditions declared before it ran."""
import json
import sys
from collections import Counter

sys.path.insert(0, "/mnt/localssd/gate/ornith")
from ornith_repro.grpo import predicted_binary_degeneracy  # noqa: E402

FILES = [("v1", "/mnt/localssd/gate/out/ornith_run_v1_iters.jsonl"),
         ("v2", "/mnt/localssd/gate/runs/ornith_v2/iters.jsonl"),
         ("smoke", "/mnt/localssd/gate/runs/ornith_smoke/iters.jsonl")]
rows = []
for tag, f in FILES:
    try:
        for l in open(f):
            r = json.loads(l)
            r["_run"] = tag
            rows.append(r)
    except FileNotFoundError:
        pass

print("=" * 92)
print("THE LOOP, %d iterations across %d runs (no update ever changed the policy between"
      % (len(rows), len({r["_run"] for r in rows})))
print("iterations that produced none, so the iterations are independent draws)")
print("=" * 92)
formed = [r for r in rows if r.get("tasks_scored", 0) >= 2]
print("iterations that formed a TASK GROUP and took a gradient step: %d of %d"
      % (len(formed), len(rows)))
print()
print("%-6s %-5s %8s %8s %8s %10s %9s" %
      ("run", "iter", "accepted", "scored", "yield", "prop_tok", "capped"))
for r in rows:
    print("%-6s %-5d %8s %8s %8.3f %10d %6d/%d"
          % (r["_run"], r["iter"], r.get("tasks_accepted"), r.get("tasks_scored", 0),
             r.get("proposer_yield") or 0.0, r.get("proposer_tokens", 0),
             r.get("proposer_hit_cap", 0), r.get("proposer_calls", 0)))

tot_calls = sum(r.get("proposer_calls", 0) for r in rows)
tot_cap = sum(r.get("proposer_hit_cap", 0) for r in rows)
tot_tok = sum(r.get("proposer_tokens", 0) for r in rows)
tot_acc = sum(r.get("tasks_accepted", 0) for r in rows)
tot_scored = sum(r.get("tasks_scored", 0) for r in rows)
print()
print("PROPOSER, pooled: %d attempts, %d (%.3f) never terminated, %d accepted as valid tasks "
      "(yield %.3f)" % (tot_calls, tot_cap, tot_cap / tot_calls, tot_acc, tot_acc / tot_calls))
print("  %d generated tokens, i.e. %.0fk tokens per accepted task and %.0fk per SCORED task"
      % (tot_tok, tot_tok / max(tot_acc, 1) / 1000, tot_tok / max(tot_scored, 1) / 1000))
reasons = Counter()
for r in rows:
    reasons.update(r.get("proposer_reasons", {}))
print("  rejection reasons:", dict(reasons))

print()
print("WHY ACCEPTED TASKS WERE REFUSED AT SCORING")
ref = Counter()
for r in rows:
    ref.update(r.get("refusals", {}))
for k, v in ref.items():
    print("   %-62s %d" % (k.strip(), v))

print()
print("THE DEGENERATION CONDITION, with the k-vectors that identify WHICH one")
ks = [kv for r in rows for kv in r.get("refused_k_vectors", [])]
if ks:
    allk = [k for kv in ks for k in kv["k_per_scaffold"] if k is not None]
    G = ks[0]["G"]
    print("   refused tasks with k recorded: %d, scaffold groups: %d, G=%d"
          % (len(ks), len(allk), G))
    print("   k histogram (successes per scaffold group):", dict(sorted(Counter(allk).items())))
    print("   all-correct groups: %d of %d;  all-wrong groups: %d of %d"
          % (sum(1 for k in allk if k == G), len(allk),
             sum(1 for k in allk if k == 0), len(allk)))
else:
    print("   (none recorded: only the instrumented run carries them)")
ph = [p for r in rows for p in r.get("p_hats", [])]
if ph:
    print("   p-hat of the tasks that WERE scored: %s (mean %.4f)"
          % ([round(x, 3) for x in ph], sum(ph) / len(ph)))
    m = sum(ph) / len(ph)
    print("   predicted degenerate-group rate at that difficulty: %.3f at G=4, %.3f at G=8"
          % (predicted_binary_degeneracy(m, 4), predicted_binary_degeneracy(m, 8)))
    print("   (Ornith's published target is p*=0.2, which would give %.3f at G=4)"
          % predicted_binary_degeneracy(0.2, 4))

print()
print("WHAT THE ONE ITERATION THAT DID TRAIN LOOKED LIKE")
for r in formed:
    print("   run=%s iter=%d  informative rollout groups %.2f of %d, scaffold groups %.2f of %d,"
          " task group informative=%s"
          % (r["_run"], r["iter"], r["informative_rollout_groups"], r["n_rollout_groups"],
             r["informative_scaffold_groups"], r["n_scaffold_groups"],
             r["task_group_informative"]))
    print("      mean p_hat %.4f  N %.4f  V %.4f  D %.3g  R_task %.3g  R_harness %.4f"
          % (r["mean_p_hat"], r["mean_novelty_N"], r["mean_validity_V"],
             r["mean_difficulty_D"], r["mean_R_task"], r["mean_R_harness"]))
    for st, u in r["updates"].items():
        print("      %-9s rows=%-3s tokens=%-7s grad_norm=%s"
              % (st, u.get("rows"), u.get("tokens"), round(u.get("grad_norm") or 0, 5)))
    print("      iteration wall time %.0fs, %d generated tokens" % (r["iter_s"], r["gen_tokens"]))
