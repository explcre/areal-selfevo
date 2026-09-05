#!/usr/bin/env python3
"""Does measuring difficulty AFTER the scaffold let the loop form task groups?

THE FINDING THIS TESTS. The difficulty gate selects on a success rate measured by prompting
the model with the task alone, but the loop never solves a task alone: the harness writes a
scaffold and the solver is conditioned on it. Measured on this box, that shifts success upward
by 0.112 on average, sends six of eight mixed-subset tasks to a rate of exactly 1.000, and
leaves six of eight refusable by the package's own degeneracy guard. So the gate selects on a
statistic a later stage of the same loop overwrites.

THE FIX THIS TESTS. Estimate difficulty on the prompt the rollouts actually receive --- the
scaffolded one. That is a change in WHERE the statistic is computed, not in the reward, the
kernel or the target, and the two arms here differ in nothing else.

    PRE   select by D(p_hat) with p_hat measured bare, as published.
    POST  generate the scaffolds first, probe them, and select by D(p_hat) with p_hat measured
          under those scaffolds. The probe block is INDEPENDENT of the block the loop then
          scores, so the selection is not scored on its own draw.

WHAT IT COSTS, which is half the result. POST cannot filter before paying for scaffold
generation, and it needs a probe block the loop does not consume, so it pays scaffolds and
rollouts for every CANDIDATE rather than for every selected task. Both arms are given the same
number of candidates so the selection is equally wide, and the token cost of each is reported.

The proposer stage does not appear: both arms draw tasks from the mixed subset, so only the
solver and harness stages take gradient. That is deliberate -- it isolates where difficulty is
measured from the separate question of whether a prompted proposer can hit a target at all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time

import torch

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
import ornith_train as ot  # noqa: E402
import train_arm  # noqa: E402
from ornith_repro import live as ol  # noqa: E402
from ornith_repro.buffer import TaskBuffer  # noqa: E402
from ornith_repro.grpo import grpo_advantages  # noqa: E402
from ornith_repro.guards import GuardViolation  # noqa: E402
from ornith_repro.judges import Judges  # noqa: E402
from ornith_repro.loop import OrnithConfig, grade as olgrade  # noqa: E402
from ornith_repro.nested import NestedConfig, run_iteration_nested  # noqa: E402
from ornith_repro.rewards import difficulty_reward  # noqa: E402
from ornith_repro.types import Task, digest  # noqa: E402


def scaffolds_for(url, tok, tasks, cfg, ncfg, cap, seed):
    """Generate `ncfg.n_scaffolds` scaffolds per task, built by the package's constructor."""
    prompts, owner = [], []
    for i, t in enumerate(tasks):
        for _ in range(ncfg.n_scaffolds):
            prompts.append(ol.SCAFFOLD_PROMPT.format(task=t.text))
            owner.append(i)
    recs = asyncio.run(ot.gen_batch(url, tok, prompts, cap, len(prompts), ""))
    out, recs_by_task = [], []
    for i, t in enumerate(tasks):
        mine = [recs[k] for k in range(len(recs)) if owner[k] == i]
        rc = ot.ReplayClient([(r["text"], ot.unanswered(r)) for r in mine])
        out.append(ol.propose_scaffolds(rc, cfg, ncfg, t, seed + i))
        recs_by_task.append(mine)
    return out, recs_by_task, sum(len(r["output_ids"]) for r in recs)


def rollouts_for(url, tok, tasks, scaffolds_by_task, n, cap, conc, lora):
    """Generate `n` rollouts for every (task, scaffold) pair."""
    prompts, owner = [], []
    for i, t in enumerate(tasks):
        for j, sc in enumerate(scaffolds_by_task[i]):
            p = ol.SOLVER_PROMPT.format(instructions=sc.instructions, task=t.text)
            for _ in range(n):
                prompts.append(p)
                owner.append((i, j))
    recs = asyncio.run(ot.gen_batch(url, tok, prompts, cap, min(len(prompts), conc), lora))
    by_pair = {}
    for k in range(len(recs)):
        by_pair.setdefault(owner[k], []).append(recs[k])
    # Truncation is measured on THIS configuration every iteration rather than inherited: a
    # cap that silently cuts rollouts turns a difficulty estimate into a budget measurement,
    # which is the failure this whole line of work started from.
    trunc = sum(1 for r in recs if ot.unanswered(r)) / max(len(recs), 1)
    return by_pair, sum(len(r["output_ids"]) for r in recs), trunc


def post_scaffold_phat(task, scaffolds, by_pair, i):
    """Success rate of a task under ITS OWN scaffolds, and the per-scaffold counts.

    Truncated-and-unanswered rollouts are excluded rather than scored wrong, matching
    `abort_policy='exclude'` and this paper's scoring convention.
    """
    ks, n_ok, n_res = [], 0, 0
    for j, sc in enumerate(scaffolds):
        rs = [olgrade(sc, task, r["text"], ot.unanswered(r)) for r in by_pair.get((i, j), [])]
        live = [r for r in rs if r.abort_reason is None]
        k = sum(1 for r in live if r.reward > 0)
        ks.append(k)
        n_ok += k
        n_res += len(live)
    return (n_ok / n_res if n_res else None), ks, n_res


def smoke_verdict(recs, mode: str) -> list:
    """Problems with a finished run, or an empty list if it is sound.

    Asserted over the WHOLE run rather than its last iteration, and per stage rather than
    across stages. Both of those were real defects: reading only the last record let a run
    whose final iteration formed no task group pass without testing anything, and an
    across-stage test let the harness stage be handed an empty list every iteration while the
    solver stage carried the check.

    Args:
        recs: The iteration records, as written to ``iters.jsonl``.
        mode: ``"pre"`` or ``"post"``.

    Returns:
        A list of human-readable problems; empty means the run is sound.
    """
    probs = []
    if not recs:
        return ["no iteration completed"]
    grouped = [r for r in recs if r.get("formed_task_group")]
    if not grouped:
        probs.append("no iteration formed a task group, so no stage could be checked; "
                     "this is the vacuous pass the previous check gave")
    if mode == "post" and not any(r.get("probe_post_phat") for r in recs):
        probs.append("post mode recorded no scaffolded difficulty estimate")
    if not any(r.get("tokens_rollout") for r in recs):
        probs.append("no scored rollout block was generated")
    for st_name in ("solver", "harness"):
        ok = False
        for r in grouped:
            u = (r.get("updates") or {}).get(st_name, {})
            if (u.get("rows") and (u.get("grad_norm") or 0) > 0
                    and abs(u.get("fp_delta") or 0) > 0):
                ok = True
                break
        if not ok:
            probs.append("stage %r never trained in any iteration that formed a group "
                         "(rows/grad/parameter movement)" % st_name)
    return probs


def main() -> int:
    """Run one arm to a budget fixed in advance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--mode", required=True, choices=["pre", "post"])
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--adapter-name", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-candidates", type=int, default=6)
    ap.add_argument("--tasks-per-iter", type=int, default=3)
    ap.add_argument("--n-scaffolds", type=int, default=3)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--n-probe", type=int, default=4)
    ap.add_argument("--scaffold-cap", type=int, default=4096)
    ap.add_argument("--rollout-cap", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--mb-tokens", type=int, default=16384)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=1.2)
    ap.add_argument("--ckpt-every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    a.run_dir = os.path.abspath(a.run_dir)
    os.makedirs(a.run_dir, exist_ok=True)
    log_path = os.path.join(a.run_dir, "iters.jsonl")
    adapter_dir = os.path.join(a.run_dir, "adapter")

    cfg = OrnithConfig(max_new_tokens=a.scaffold_cap)
    ncfg = NestedConfig(n_scaffolds=a.n_scaffolds, n_rollouts_per_scaffold=a.n_rollouts,
                        sampling="nested")
    judges, buffer = Judges(), TaskBuffer()
    rng = random.Random(a.seed)
    pool = json.load(open(a.pool))["tasks"]
    used: set = set()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = "cuda:0"
    print("[%s] loading base model" % a.mode, flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=train_arm.TARGET_MODULES, exclude_modules=train_arm.EXCLUDE))
    cov = train_arm.assert_coverage(model)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    mem = train_arm.calibrate_microbatch(model, opt, device, pad_id, a.mb_tokens, 1)
    ctxinfo = ot.assert_context_fits(a.url, max(a.scaffold_cap, a.rollout_cap))
    print("[%s] coverage %s; microbatch %s; context %s" % (a.mode, cov, mem, ctxinfo), flush=True)
    train_arm.save_adapter(model, adapter_dir)
    train_arm.push_adapter(a.url, a.adapter_name, adapter_dir)
    json.dump({"mode": a.mode, "coverage": cov, "microbatch": mem, "context": ctxinfo,
               "args": vars(a)}, open(os.path.join(a.run_dir, "meta.json"), "w"),
              indent=1, default=str)

    logf = open(log_path, "a")
    t0 = time.time()
    it = 0
    while it < a.max_iters and (time.time() - t0) < a.max_hours * 3600:
        tic = time.time()
        st = {"iter": it, "mode": a.mode, "refusals": {}}
        avail = [t for t in pool if t["idx"] not in used]
        if len(avail) < a.n_candidates:
            st["halted"] = "pool exhausted"
            logf.write(json.dumps(st) + "\n"); logf.flush(); break
        cand_rows = rng.sample(avail, a.n_candidates)
        cands = [Task(task_id=digest("seed", r["idx"])[:16], text=r["problem"],
                      answer=r["answer"], source="seed") for r in cand_rows]
        pool_p = [(r["c_a"] + r["c_b"]) / (r["n_a"] + r["n_b"]) for r in cand_rows]
        tok_scaf = tok_probe = tok_roll = 0

        if a.mode == "pre":
            # Select on the BARE statistic, as published: no scaffold needed to choose.
            order = sorted(range(len(cands)),
                           key=lambda i: -difficulty_reward(pool_p[i], cfg.p_star, cfg.sigma))
            keep = order[: a.tasks_per_iter]
            sel = [cands[i] for i in keep]
            scafs, screcs, tok_scaf = scaffolds_for(a.url, tok, sel, cfg, ncfg,
                                                    a.scaffold_cap, a.seed + it)
            st["selected_on"] = [round(pool_p[i], 4) for i in keep]
        else:
            # Scaffold EVERY candidate, probe it, and select on the scaffolded statistic.
            scafs_all, screcs_all, tok_scaf = scaffolds_for(a.url, tok, cands, cfg, ncfg,
                                                            a.scaffold_cap, a.seed + it)
            probe, tok_probe, st["probe_truncation"] = rollouts_for(
                a.url, tok, cands, scafs_all, a.n_probe, a.rollout_cap, a.concurrency,
                a.adapter_name)
            post_p, ks_all = [], []
            for i, t in enumerate(cands):
                p, ks, _ = post_scaffold_phat(t, scafs_all[i], probe, i)
                post_p.append(p)
                ks_all.append(ks)
            scored = [(difficulty_reward(p, cfg.p_star, cfg.sigma) if p is not None else -1.0)
                      for p in post_p]
            keep = sorted(range(len(cands)), key=lambda i: -scored[i])[: a.tasks_per_iter]
            sel = [cands[i] for i in keep]
            scafs = [scafs_all[i] for i in keep]
            # The records must be subset by the SAME index list as the scaffolds, or the
            # harness stage would attribute one scaffold's advantage to another's tokens.
            screcs = [screcs_all[i] for i in keep]
            st["selected_on"] = [None if post_p[i] is None else round(post_p[i], 4)
                                 for i in keep]
            st["probe_post_phat"] = [None if p is None else round(p, 4) for p in post_p]
            st["probe_pool_phat"] = [round(p, 4) for p in pool_p]
            st["probe_k_vectors"] = ks_all
            st["probe_drift"] = [None if post_p[i] is None else round(post_p[i] - pool_p[i], 4)
                                 for i in range(len(cands))]
        for t in sel:
            used.add(next(r["idx"] for r in cand_rows if r["problem"] == t.text))
        st["pool_phat_selected"] = [round(pool_p[i], 4) for i in keep]

        # The scored block is FRESH in both arms, so POST is never scored on its probe.
        by_pair, tok_roll, st["rollout_truncation"] = rollouts_for(
            a.url, tok, sel, scafs, a.n_rollouts, a.rollout_cap, a.concurrency,
            a.adapter_name)

        results, kept = [], []
        for i, t in enumerate(sel):
            blocks = [[(r["text"], ot.unanswered(r)) for r in by_pair.get((i, j), [])]
                      for j in range(len(scafs[i]))]
            try:
                res = run_iteration_nested(cfg, ncfg, t, scafs[i], blocks, buffer, judges)
            except (GuardViolation, ValueError) as exc:
                key = type(exc).__name__ + ":" + str(exc).split("(")[0][:56]
                st["refusals"][key] = st["refusals"].get(key, 0) + 1
                ks = []
                for j, sc in enumerate(scafs[i]):
                    rs = [olgrade(sc, t, r["text"], ot.unanswered(r))
                          for r in by_pair.get((i, j), [])]
                    ks.append(sum(1 for r in rs if r.reward > 0 and r.abort_reason is None))
                st.setdefault("refused_k_vectors", []).append(ks)
                continue
            results.append(res)
            kept.append((i, t))
        st["tasks_scored"] = len(results)
        st.update({"tokens_scaffold": tok_scaf, "tokens_probe": tok_probe,
                   "tokens_rollout": tok_roll,
                   "tokens_total": tok_scaf + tok_probe + tok_roll})

        if len(results) >= 2:
            task_adv = grpo_advantages([r.task_record.R_task for r in results],
                                       epsilon=cfg.epsilon)
            upd = {}
            for stage in ("solver", "harness"):
                rows = []
                if stage == "solver":
                    for res, (i, t) in zip(results, kept):
                        for j, ga in enumerate(res.rollout_advantages):
                            if ga.degenerate:
                                continue
                            for m, adv in enumerate(ga.advantages):
                                r = by_pair[(i, j)][m]
                                if r["output_ids"]:
                                    rows.append({"prompt_ids": r["prompt_ids"],
                                                 "output_ids": r["output_ids"], "adv": adv})
                else:
                    # The defect this repairs: the loop iterated over both stages but only
                    # ever built rows for the solver, so the harness optimiser was handed an
                    # empty list every iteration and stepped on nothing. The scaffold that
                    # earned each advantage is the generation in `screc_by_task`, which was
                    # already being carried and never used.
                    for res, (i, t) in zip(results, kept):
                        ga = res.scaffold_advantages
                        if ga is None or ga.degenerate:
                            continue
                        for j, adv in enumerate(ga.advantages):
                            r = screcs[i][j]
                            if r["output_ids"]:
                                rows.append({"prompt_ids": r["prompt_ids"],
                                             "output_ids": r["output_ids"], "adv": adv})
                fp0 = train_arm.adapter_fingerprint(model)
                if rows:
                    u = train_arm.policy_step(model, rows, opt, device, a.rollout_cap, 1.0,
                                              a.mb_tokens, pad_id, "sequence")
                else:
                    u = {"loss": None, "tokens": 0, "grad_norm": 0.0}
                upd[stage] = {"rows": len(rows),
                              "fp_delta": train_arm.adapter_fingerprint(model) - fp0,
                              **{k: u[k] for k in ("loss", "tokens", "grad_norm") if k in u}}
            rg = [g for res in results for g in res.rollout_advantages]
            sg = [res.scaffold_advantages for res in results
                  if res.scaffold_advantages is not None]
            tr = [res.task_record for res in results]
            st.update({
                "formed_task_group": True,
                "task_group_informative": (not task_adv.degenerate),
                "informative_rollout_groups": sum(1 for g in rg if not g.degenerate) / len(rg),
                "n_rollout_groups": len(rg),
                "informative_scaffold_groups": sum(1 for g in sg if not g.degenerate) / max(len(sg), 1),
                "mean_p_hat": sum(t.p_hat for t in tr) / len(tr),
                "mean_R_task": sum(t.R_task for t in tr) / len(tr),
                "updates": upd,
                "adapter_absB": train_arm.adapter_fingerprint(model),
            })
            if any(u.get("tokens") for u in upd.values()):
                train_arm.save_adapter(model, adapter_dir)
                train_arm.push_adapter(a.url, a.adapter_name, adapter_dir)
        else:
            st["formed_task_group"] = False

        st.update({"iter_s": round(time.time() - tic, 1),
                   "elapsed_s": round(time.time() - t0, 1)})
        logf.write(json.dumps(st) + "\n")
        logf.flush()
        print("[%s] iter %d scored=%d group=%s tok=%d (scaf %d probe %d roll %d) %.0fs"
              % (a.mode, it, st["tasks_scored"], st["formed_task_group"], st["tokens_total"],
                 tok_scaf, tok_probe, tok_roll, st["iter_s"]), flush=True)
        it += 1
        if a.ckpt_every and it % a.ckpt_every == 0:
            snap = os.path.join(a.run_dir, "ckpt", "iter%05d" % it)
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            shutil.copytree(adapter_dir, snap, dirs_exist_ok=True)
        if a.smoke and it >= a.smoke:
            break

    logf.close()
    if a.smoke:
        recs = [json.loads(l) for l in open(log_path)]
        probs = smoke_verdict(recs, a.mode)
        if probs:
            for pb in probs:
                print("SMOKE FAIL: " + pb, file=sys.stderr)
            return 1
        grouped = sum(1 for r in recs if r.get("formed_task_group"))
        print("SMOKE PASS: %s mode, %d/%d iterations formed a task group, and both the solver "
              "and the harness stage trained with a non-zero gradient and a parameter change"
              % (a.mode, grouped, len(recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
