#!/usr/bin/env python3
"""Read a three-stage run and answer the process questions it was run to answer.

Reports, in this order: whether each stage ever trained (on the three observables, not on the
loop having iterated), whether informative groups are sustained across iterations, what guard
G4 discarded, and the cost per scored task.
"""
from __future__ import annotations

import argparse
import json

STAGES = ("proposer", "harness", "solver")


def main() -> int:
    """Print the per-stage evidence, the group rates, the G4 counterfactual and the cost."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.run_dir + "/iters.jsonl") if l.strip()]
    print("iterations logged: %d" % len(rows))

    print("\nPER-STAGE EVIDENCE  (a stage trained only if rows>0 AND grad>0 AND params moved)")
    print("%-5s %-9s %-7s %s" % ("iter", "scored", "source", "  ".join(
        "%-28s" % s for s in STAGES)))
    ever = {s: False for s in STAGES}
    for r in rows:
        u = r.get("updates") or {}
        cells = []
        for s in STAGES:
            d = u.get(s)
            if not d:
                cells.append("%-28s" % "-")
                continue
            moved = abs(d.get("fp_delta") or 0) > 0
            gn = d.get("grad_norm") or 0
            ok = d.get("rows", 0) > 0 and gn > 0 and moved
            ever[s] = ever[s] or ok
            cells.append("%-28s" % ("rows=%-3d g=%.4f d=%.1f%s"
                                    % (d.get("rows", 0), gn, abs(d.get("fp_delta") or 0),
                                       "" if ok else " X")))
        print("%-5d %-9s %-7s %s" % (r["iter"], r.get("tasks_scored"),
                                     (r.get("task_source") or "-")[:6], "  ".join(cells)))
    print("\nstage trained at least once:")
    for s in STAGES:
        print("  %-9s %s" % (s, "YES" if ever[s] else "NO  <-- never received data"))

    print("\nINFORMATIVE GROUPS ACROSS ITERATIONS")
    print("%-5s %10s %10s %8s %10s %8s" % ("iter", "rollout", "scaffold", "task", "mean_phat",
                                           "trunc"))
    for r in rows:
        if r.get("tasks_scored"):
            print("%-5d %10s %10s %8s %10s %8s" % (
                r["iter"],
                "%.3f (%d)" % (r.get("informative_rollout_groups", 0),
                               r.get("n_rollout_groups", 0)),
                "%.3f (%d)" % (r.get("informative_scaffold_groups", 0),
                               r.get("n_scaffold_groups", 0)),
                r.get("task_group_informative"),
                ("%.3f" % r["mean_p_hat"]) if r.get("mean_p_hat") is not None else "-",
                ("%.3f" % r["rollout_truncation"]) if r.get("rollout_truncation") is not None
                else "-"))

    print("\nGUARD G4: what refusing on rollout groups discarded")
    n_ref = n_cf = n_cf_inf = 0
    for r in rows:
        for k, v in (r.get("refusals") or {}).items():
            if "degenerate" in k:
                n_ref += v
        for c in r.get("g4_counterfactual", []) or []:
            if "error" in c:
                continue
            n_cf += 1
            n_cf_inf += bool(c.get("scaffold_group_informative"))
    print("  tasks refused for all-degenerate rollout groups : %d" % n_ref)
    print("  of those, scaffold group WOULD have been informative: %d/%d%s"
          % (n_cf_inf, n_cf, "" if not n_cf else " = %.3f" % (n_cf_inf / n_cf)))

    print("\nCOST")
    # `gen_tokens` is this trainer's own total (proposer + scaffold + rollout); the other
    # trainer names the same quantities differently, so both spellings are accepted rather
    # than one silently summing to zero.
    def toks(r):
        if r.get("gen_tokens") is not None:
            return r["gen_tokens"]
        return ((r.get("proposer_tokens") or 0) + (r.get("scaffold_tokens") or 0)
                + (r.get("rollout_tokens") or 0) + (r.get("tokens_total") or 0))
    tot = sum(toks(r) for r in rows)
    scored = sum(r.get("tasks_scored") or 0 for r in rows)
    prop = sum(r.get("proposer_tokens") or 0 for r in rows)
    print("  total tokens %d, tasks scored %d" % (tot, scored))
    print("  cost per scored task: %s"
          % ("%d tokens" % (tot / scored) if scored else "UNDEFINED (nothing was scored)"))
    print("  proposer share: %d tokens (%.1f%%)"
          % (prop, 100.0 * prop / tot if tot else 0.0))
    print("  by stage: proposer %d, scaffold %d, rollout %d"
          % (prop, sum(r.get("scaffold_tokens") or 0 for r in rows),
             sum(r.get("rollout_tokens") or 0 for r in rows)))
    seeded = [r for r in rows if r.get("task_source") == "seed"]
    live = [r for r in rows if r.get("task_source") == "proposer"]
    for name, grp in (("seeded", seeded), ("proposer-live", live)):
        sc = sum(r.get("tasks_scored") or 0 for r in grp)
        tk = sum(toks(r) for r in grp)
        print("  %-14s %2d iters, %2d scored, %9d tokens, %s per scored task"
              % (name, len(grp), sc, tk,
                 ("%d" % (tk / sc)) if sc else "UNDEFINED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
