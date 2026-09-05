#!/usr/bin/env python3
"""Run the wired Ornith loop against a served model and report what actually fired.

This is the caller that `llm.py` never had. It constructs an `OpenAICompatClient`, verifies
the served id against the endpoint's own `/v1/models` before generating (an unregistered id
is answered by the BASE model with a 200 and no warning), and drives `live.run_live_iteration`
which uses the NESTED entry point.

It reports, per run and never as an assumption:
  * the validity gate's rejection rate and its reasons;
  * the fraction of rollout groups that are non-degenerate, i.e. that carry any gradient;
  * whether the scaffold-level group is degenerate, which decides whether nesting does
    anything at all;
  * the gold-agreement rate ONLY when it comes from independent verification; the naive
    version was algebraically identical to p_hat and is no longer reported as a second
    opinion;
  * which stages fired, asserted rather than assumed.

Generation is serial, which is fine for the one-sample proof this exists to provide and is
NOT how a scaled run should be done; concurrency belongs in the trainer, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ornith_repro.buffer import TaskBuffer
from ornith_repro.judges import Judges
from ornith_repro.live import load_competence, run_live_iteration
from ornith_repro.llm import OpenAICompatClient
from ornith_repro.loop import OrnithConfig, read_artifacts, verify_provenance
from ornith_repro.nested import NestedConfig
from ornith_repro.verify_sound import SoundVerifier


def served_models(base_url: str) -> tuple[list[str], int]:
    """Ask the endpoint what it serves, so the check uses its answer and not our config.

    Args:
        base_url: Endpoint ending in /v1.

    Returns:
        (served ids, max context length reported).
    """
    import httpx
    r = httpx.get(base_url.rstrip("/") + "/models", timeout=60.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = [m["id"] for m in data]
    ctx = max((m.get("max_model_len") or 0) for m in data) if data else 0
    return ids, ctx


def main() -> int:
    """Drive the loop and print the accounting."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--n-scaffolds", type=int, default=3)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "xhigh"])
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--blocks", default="/mnt/localssd/gate/out/blocks_low.jsonl")
    ap.add_argument("--problems",
                    default="/mnt/localssd/gate/searchhalf/olympiadbench/test.jsonl")
    ap.add_argument("--artifacts", default="/mnt/localssd/gate/out/live_iters.jsonl")
    ap.add_argument("--summary", default="/mnt/localssd/gate/out/live_summary.json")
    ap.add_argument("--key-policy", default="reject_refuted",
                    choices=["off", "reject_refuted", "require_verified"],
                    help="what to do with a task whose asserted key fails independent "
                         "verification; 'off' reproduces the confounded measurement")
    a = ap.parse_args()

    ids, ctx = served_models(a.base_url)
    print("served: %s (context %d)" % (ids, ctx), flush=True)
    client = OpenAICompatClient(a.base_url, a.model, ids, ctx or 65536,
                                reasoning_effort=a.effort)
    print("client built, model=%s effort=%s" % (client.model_id, a.effort), flush=True)

    solved, unsolved = load_competence(a.blocks, a.problems)
    print("competence exemplars: %d always-solved, %d never-solved"
          % (len(solved), len(unsolved)), flush=True)
    if len(solved) < 3 or not unsolved:
        print("FATAL: too few exemplars to condition the proposer", flush=True)
        return 2

    cfg = OrnithConfig(base_model=a.model, sigma=a.sigma,
                       max_new_tokens=a.max_new_tokens, min_valid_rollouts=4)
    ncfg = NestedConfig(n_scaffolds=a.n_scaffolds, n_rollouts_per_scaffold=a.n_rollouts)
    print("budget per task: %d rollouts (%d scaffolds x %d), doubled by the holdout block"
          % (ncfg.total_rollouts, ncfg.n_scaffolds, ncfg.n_rollouts_per_scaffold), flush=True)

    buffer = TaskBuffer()
    judges = Judges()
    # The key check runs before any rollout budget is committed, so a refuted task costs one
    # verification instead of a full nested rollout block.
    verifier = (None if a.key_policy == "off"
                else SoundVerifier(client, primary_name=a.model, secondary_name=a.model,
                                   timeout=30.0))
    print("key policy: %s (verifier: %s)"
          % (a.key_policy, "SoundVerifier" if verifier else "none"), flush=True)
    art = Path(a.artifacts)
    pool: list = []
    rows = []

    for it in range(a.iterations):
        live = run_live_iteration(client, cfg, ncfg, solved[:3], unsolved[:2], buffer,
                                  judges, seed=1000 + it, artifacts=art,
                                  scaffold_pool=pool or None,
                                  verifier=verifier, key_policy=a.key_policy)
        if live is None:
            print("iteration %d: proposer produced nothing valid" % it, flush=True)
            continue
        if live.result is None:
            rows.append({"iteration": it, "task_id": live.task.task_id,
                         "task_preview": live.task.text[:160],
                         "asserted_answer": live.task.answer,
                         "key_verdict": live.key_verdict, "key_detail": live.key_detail,
                         "rejected_reason": live.rejected_reason, "rollouts_spent": 0})
            print("iteration %d: task DROPPED (%s) -- 0 rollouts spent"
                  % (it, live.rejected_reason), flush=True)
            continue
        pool.extend(live.scaffolds)
        r = live.result
        nondeg = [not g.degenerate for g in r.rollout_advantages]
        row = {
            "iteration": it,
            "task_id": live.task.task_id,
            "task_preview": live.task.text[:160],
            "asserted_answer": live.task.answer,
            "stages_fired": live.stages_fired,
            "p_hat": r.task_record.p_hat,
            "n_valid": r.task_record.n_valid,
            "n_aborted": r.task_record.n_aborted,
            "V": r.task_record.V, "D": r.task_record.D, "N": r.task_record.N,
            "R_task": r.task_record.R_task,
            "rollout_groups": len(r.rollout_advantages),
            "rollout_groups_informative": sum(nondeg),
            "scaffold_group_degenerate": (r.scaffold_advantages.degenerate
                                          if r.scaffold_advantages else None),
            "scaffold_rewards_discovery": r.scaffold_rewards_discovery,
            "scaffold_rewards_holdout": r.scaffold_rewards_holdout,
            # Not reported as a separate number: it was identical to p_hat. The
            # informative version needs independent verification (verify.py), not yet wired.
            "gold_agreement_independent": live.gold_agreement,
            "proposals_used": live.n_proposals,
            "validity_rejections": live.rejection_reasons,
            "control_scaffolds": len(live.control_scaffolds),
            "key_verdict": live.key_verdict,
            "key_detail": live.key_detail,
            "rollouts_spent": live.rollouts_spent,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    if not rows:
        print("no iteration completed", flush=True)
        return 1

    verified = 0
    for r in read_artifacts(art):
        verify_provenance(r)
        verified += 1
    print("artifact rows written and provenance-verified: %d" % verified, flush=True)

    dropped = [r for r in rows if r.get("rejected_reason")]
    scored = [r for r in rows if not r.get("rejected_reason")]
    print("tasks scored %d, dropped on key verification %d, rollouts saved %d"
          % (len(scored), len(dropped),
             len(dropped) * ncfg.n_rollouts_per_scaffold * ncfg.n_scaffolds * 2), flush=True)
    if not scored:
        print("every task was dropped; nothing was scored", flush=True)
        Path(a.summary).write_text(json.dumps(rows, indent=2))
        return 0

    stages = set()
    for r in scored:
        stages |= set(r["stages_fired"])
    print("stages fired across the run: %s" % sorted(stages), flush=True)
    missing = {"rollout", "harness", "task"} - stages
    if missing:
        print("FATAL: stages never fired: %s" % sorted(missing), flush=True)
        return 3

    Path(a.summary).write_text(json.dumps(rows, indent=2))
    print("wrote %s" % a.summary, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
