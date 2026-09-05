#!/usr/bin/env python3
"""Ornith's three-stage self-improvement loop, with the optimiser it has never had.

WHAT WAS MISSING. `ornith_repro` has sixteen modules and nothing trains: there is no
optimiser, no backward pass and no parameter update anywhere in the package, and
`loop.apply_update` is a seam that records what WOULD be updated. So the central claim --
that a model improves itself by co-evolving tasks, scaffolds and weights -- had never been
executed. Everything measured before this file is their task-selection criterion.

WHAT THIS ADDS, and only this. Group-relative optimisation on all three stages, each against
its own published reward, with the reward propagation and the nested group structure taken
from the package rather than reimplemented:

    R_task    = V(q,s) * D(q,s,{tau}) * N(q),   D = exp(-(p-p*)^2 / 2 sigma^2), p* = 0.2
    R_harness = C(q,h) * F(h,{tau}) * H(h)
    R_rollout = h(q, tau)                        the scaffold IS the rollout reward function

`nested.run_iteration_nested` computes every one of those and both within-task comparison
levels (rollouts within a scaffold, scaffolds within a task); it is imported, not rewritten,
and the smoke check asserts it fired and that both levels carry non-zero advantage. The one
level it cannot form is the TASK group, because it scores a single task per call -- so this
file forms it across the tasks of one iteration, which is the only new GRPO group here.

STAGE ORDER is `cfg.stage_order`, solver -> harness -> proposer, three separate optimiser
steps on one shared adapter: one policy improving itself, not three policies.

TRUNCATION. `loop.grade` marks a capped generation ABORTED with reward 0 regardless of what
it contains. This programme measured that forcing capped-but-answered generations to zero
charges most to whichever arm truncates most and inflated every effect it was applied to, so
the flag passed in here is `hit_cap AND no boxed answer` -- their function is untouched, the
decision is made at the call boundary where it belongs, and it is recorded per rollout.
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

import aiohttp
import torch

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import gate_lib  # noqa: E402
import train_arm  # noqa: E402
from ornith_repro import live as ol  # noqa: E402
from ornith_repro.buffer import TaskBuffer  # noqa: E402
from ornith_repro.grpo import grpo_advantages  # noqa: E402
from ornith_repro.guards import GuardViolation  # noqa: E402
from ornith_repro.judges import Judges  # noqa: E402
from ornith_repro.loop import OrnithConfig, grade as olgrade  # noqa: E402
from ornith_repro.nested import NestedConfig, run_iteration_nested  # noqa: E402
from ornith_repro.types import Task, digest  # noqa: E402


class ReplayClient:
    """An `LLMClient` that returns already-generated completions, in order.

    Generation here is asynchronous and batched so that a whole iteration's calls are in
    flight at once; the package's constructors (`propose_scaffolds`, which fixes
    `grader_kind='boxed_exact'` and the scaffold id digest) are synchronous. Replaying the
    texts through them keeps their construction code authoritative while the round trips
    stay parallel.
    """

    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def generate(self, prompt: str, max_new_tokens: int, seed: int):
        """Return the next pre-generated ``(text, truncated)`` pair."""
        if self._i >= len(self._items):
            raise RuntimeError("ReplayClient exhausted: the caller made more generation "
                               "calls than were pre-generated for it")
        out = self._items[self._i]
        self._i += 1
        return out

    def exhausted(self) -> bool:
        """True when every pre-generated item was consumed."""
        return self._i == len(self._items)


def chat_wrap(tok, text: str, effort: str = "low") -> str:
    """Render one stage prompt through the model's chat template at the loop's thinking budget.

    NOT optional, and getting it wrong is silent. Sent as a raw string the base model does
    free-form CONTINUATION rather than instruction-following: measured on the proposer, every
    completion ran past a 1024-token cap deliberating aloud and 7 of 12 were rejected as
    truncated with the remaining 4 rejected for trailing text after the answer. The rest of
    this programme runs at `reasoning_effort=low`; so does the loop.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": effort})


async def gen_batch(url, tok, prompts, cap, concurrency, lora, temperature=1.0):
    """Generate completions for many prompts at once, keeping exact token ids.

    ``input_ids`` are sent and the produced ids are read back from the server's logprob
    record, so the sequence the gradient is taken over is the sequence the server produced.
    Re-tokenising the returned text would drift.

    Returns:
        One dict per prompt: ``prompt_ids``, ``output_ids``, ``text``, ``hit_cap``.
    """
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(prompts)

    async def one(session, i, prompt):
        ids = tok(chat_wrap(tok, prompt), add_special_tokens=False)["input_ids"]
        payload = {"input_ids": ids, "return_logprob": True,
                   "sampling_params": {"max_new_tokens": cap, "temperature": temperature,
                                       "top_p": 0.95, "skip_special_tokens": True}}
        if lora:
            payload["lora_path"] = lora
        async with sem:
            for attempt in range(3):
                try:
                    async with session.post(url + "/generate", json=payload,
                                            timeout=aiohttp.ClientTimeout(total=3600)) as r:
                        body = await r.text()
                        if r.status != 200:
                            raise RuntimeError("HTTP %d %s" % (r.status, body[:200]))
                        d = json.loads(body)
                    break
                except Exception as exc:
                    if attempt == 2:
                        out[i] = {"prompt_ids": ids, "output_ids": [], "text": "",
                                  "hit_cap": False, "error": repr(exc)[:200]}
                        return
                    await asyncio.sleep(2.0 * (attempt + 1))
        mi = d["meta_info"]
        fin = mi.get("finish_reason") or {}
        out[i] = {"prompt_ids": ids,
                  "output_ids": [t for _, t, _ in (mi.get("output_token_logprobs") or [])],
                  "text": d.get("text", ""),
                  "hit_cap": (fin.get("type") if isinstance(fin, dict) else str(fin)) == "length"}

    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*(one(session, i, p) for i, p in enumerate(prompts)))
    return out


def unanswered(rec) -> bool:
    """Whether a generation genuinely failed to answer, as opposed to merely hitting the cap.

    This is the scoring convention this programme corrected to: a generation cut off after
    committing a boxed answer is graded on that answer. Passing THIS as `truncated` applies
    the correction without touching `loop.grade`.
    """
    return bool(rec["hit_cap"]) and gate_lib.math_bench.extract_boxed(rec["text"]) is None


def propose_round(recs, cfg, slots_needed):
    """Vectorised form of `live.propose_task`'s retry loop, one round over all open slots.

    Their loop is sequential per slot, which would serialise one round trip per attempt per
    slot. The accept/reject logic is theirs -- `live.parse_proposal` and the same `Task`
    construction -- only the iteration order changes, so rejections are counted the same way.

    Returns:
        ``(accepted, reasons)`` where accepted is a list of ``(Task, record)``.
    """
    accepted, reasons = [], {}
    for rec in recs:
        if rec.get("error"):
            reasons["error"] = reasons.get("error", 0) + 1
            continue
        if rec["hit_cap"]:
            # Their loop discards a truncated proposal outright; a cut-off proposal has no
            # ANSWER field to parse anyway, so this is the same rejection, counted the same.
            reasons["truncated"] = reasons.get("truncated", 0) + 1
            continue
        problem, answer, reason = ol.parse_proposal(rec["text"])
        if reason != "ok":
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        accepted.append((Task(task_id=digest("task", problem)[:16], text=problem,
                              answer=answer, source="generated"), rec))
        if len(accepted) >= slots_needed:
            break
    return accepted, reasons


def main() -> int:
    """Run the three-stage loop for a budget fixed in advance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--adapter-name", default="ORN")
    ap.add_argument("--seed-pool", required=True, help="pool.json, for competence exemplars")
    ap.add_argument("--tasks-per-iter", type=int, default=4)
    ap.add_argument("--n-scaffolds", type=int, default=3)
    ap.add_argument("--n-rollouts", type=int, default=6)
    ap.add_argument("--gen-cap", type=int, default=16384,
                    help="proposer/scaffold cap. MEASURED before it was fixed, at a context "
                         "window large enough not to be the binding constraint: the proposer "
                         "runs 7.1k-16.4k tokens per attempt (median at the cap) and the "
                         "scaffold stage 3.6k median, 11.9k p90. An earlier 1024 guess "
                         "rejected every proposal as truncated, and a 6144 server context "
                         "capped them at ~5.4k while reporting it as the generation cap.")
    ap.add_argument("--rollout-cap", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--mb-tokens", type=int, default=16384)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--proposer-oversample", type=int, default=3,
                    help="proposer candidates per task slot, in ONE batch. 3 from the "
                         "measured validity yield of 4/12.")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    a.run_dir = os.path.abspath(a.run_dir)
    os.makedirs(a.run_dir, exist_ok=True)
    log_path = os.path.join(a.run_dir, "iters.jsonl")
    adapter_dir = os.path.join(a.run_dir, "adapter")

    cfg = OrnithConfig(max_new_tokens=a.gen_cap)
    ncfg = NestedConfig(n_scaffolds=a.n_scaffolds, n_rollouts_per_scaffold=a.n_rollouts,
                        sampling="nested")
    judges = Judges()
    buffer = TaskBuffer()
    rng = random.Random(a.seed)

    # Competence exemplars for the proposer, taken from the mixed-subset measurement: the
    # problems this base always solves and never solves. The proposer is asked to aim
    # between them, which is where this programme measured the gradient to be.
    pool = json.load(open(a.seed_pool))
    tasks_pool = pool["tasks"]
    solved_ex = [t["problem"] for t in tasks_pool if (t["c_a"] + t["c_b"]) >= (t["n_a"] + t["n_b"]) - 1]
    unsolved_ex = [t["problem"] for t in tasks_pool if (t["c_a"] + t["c_b"]) <= 1]
    if len(solved_ex) < 2 or len(unsolved_ex) < 2:
        solved_ex = solved_ex or [t["problem"] for t in tasks_pool[:3]]
        unsolved_ex = unsolved_ex or [t["problem"] for t in tasks_pool[-3:]]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = "cuda:0"
    print("[orn] loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=train_arm.TARGET_MODULES, exclude_modules=train_arm.EXCLUDE))
    cov = train_arm.assert_coverage(model)
    print("[orn] adapter coverage: %s" % cov, flush=True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    mem = train_arm.calibrate_microbatch(model, opt, device, pad_id, a.mb_tokens, 1)
    print("[orn] microbatch calibration: %s" % mem, flush=True)

    train_arm.save_adapter(model, adapter_dir)
    train_arm.push_adapter(a.url, a.adapter_name, adapter_dir)
    json.dump({"coverage": cov, "microbatch": mem, "args": vars(a),
               "config": {"p_star": cfg.p_star, "sigma": cfg.sigma,
                          "stage_order": list(cfg.stage_order),
                          "n_scaffolds": ncfg.n_scaffolds,
                          "n_rollouts_per_scaffold": ncfg.n_rollouts_per_scaffold,
                          "total_rollouts_per_task": ncfg.total_rollouts},
               "n_solved_exemplars": len(solved_ex), "n_unsolved_exemplars": len(unsolved_ex)},
              open(os.path.join(a.run_dir, "meta.json"), "w"), indent=1, default=str)

    logf = open(log_path, "a")
    t0 = time.time()
    it = 0
    route_checked = False
    while it < a.max_iters and (time.time() - t0) < a.max_hours * 3600:
        tic = time.time()
        stats = {"iter": it, "refusals": {}, "proposer_reasons": {}}

        # ---------- stage 1: propose a group of tasks -----------------------------
        accepted, attempts = [], 0
        prop_tokens = prop_calls = prop_capped = 0
        # One OVER-SAMPLED batch rather than one round trip per attempt. The accept/reject
        # logic is `live.parse_proposal`'s, unchanged, and every rejection is still counted;
        # only the batching changes. It has to: a proposer attempt runs to ~14k tokens, so
        # issuing `tasks_per_iter` at a time left four requests decoding alone and one
        # iteration's proposer stage took over twenty minutes. The over-sample factor is set
        # from the MEASURED validity yield (4 of 12 parsed), and the realised yield is logged
        # every iteration because it is one of the things this run is watching.
        while len(accepted) < 2 and attempts < 2:
            attempts += 1
            recent = buffer.texts()[-5:]
            novelty = ("Do NOT repeat or lightly reword any of these already-used tasks:\n"
                       + "\n".join("- " + t[:200] for t in recent)) if recent else ""
            prompts = []
            for _ in range(a.tasks_per_iter * a.proposer_oversample):
                prompts.append(ol.PROPOSER_PROMPT.format(
                    solved="\n".join("- " + s[:400] for s in
                                     rng.sample(solved_ex, min(3, len(solved_ex)))),
                    unsolved="\n".join("- " + s[:400] for s in
                                       rng.sample(unsolved_ex, min(3, len(unsolved_ex)))),
                    novelty=novelty))
            recs = asyncio.run(gen_batch(a.url, tok, prompts, a.gen_cap,
                                         len(prompts), a.adapter_name))
            prop_calls += len(recs)
            prop_tokens += sum(len(r["output_ids"]) for r in recs)
            prop_capped += sum(1 for r in recs if r["hit_cap"])
            got, reasons = propose_round(recs, cfg, a.tasks_per_iter)
            for k, v in reasons.items():
                stats["proposer_reasons"][k] = stats["proposer_reasons"].get(k, 0) + v
            for t, r in got:
                if len(accepted) >= a.tasks_per_iter:
                    break
                if t.text not in buffer.texts() and all(t.text != x.text for x, _ in accepted):
                    accepted.append((t, r))
        stats["proposer_rounds"] = attempts
        stats["proposer_yield"] = (len(accepted) / prop_calls) if prop_calls else None
        stats["tasks_accepted"] = len(accepted)
        stats["proposer_tokens"] = prop_tokens
        stats["proposer_calls"] = prop_calls
        stats["proposer_hit_cap"] = prop_capped
        if len(accepted) < 2:
            stats["refusals"]["task_group_too_small"] = 1
            logf.write(json.dumps(stats) + "\n"); logf.flush()
            print("[orn] iter %d: only %d task(s) accepted, no task group; skipping"
                  % (it, len(accepted)), flush=True)
            it += 1
            continue

        # ---------- stage 2: scaffolds, built by THEIR constructor ----------------
        sc_prompts, owner = [], []
        for ti, (task, _) in enumerate(accepted):
            for _ in range(ncfg.n_scaffolds):
                sc_prompts.append(ol.SCAFFOLD_PROMPT.format(task=task.text))
                owner.append(ti)
        sc_recs = asyncio.run(gen_batch(a.url, tok, sc_prompts, a.gen_cap,
                                        len(sc_prompts), a.adapter_name))
        scaffolds_by_task, screc_by_task = [], []
        for ti, (task, _) in enumerate(accepted):
            mine = [sc_recs[k] for k in range(len(sc_recs)) if owner[k] == ti]
            rc = ReplayClient([(r["text"], unanswered(r)) for r in mine])
            scaffolds_by_task.append(ol.propose_scaffolds(rc, cfg, ncfg, task, a.seed + it))
            screc_by_task.append(mine)

        # ---------- stage 3: rollouts -------------------------------------------
        ro_prompts, ro_owner = [], []
        for ti, (task, _) in enumerate(accepted):
            for sj, sc in enumerate(scaffolds_by_task[ti]):
                p = ol.SOLVER_PROMPT.format(instructions=sc.instructions, task=task.text)
                for _ in range(ncfg.n_rollouts_per_scaffold):
                    ro_prompts.append(p)
                    ro_owner.append((ti, sj))
        ro_recs = asyncio.run(gen_batch(a.url, tok, ro_prompts, a.rollout_cap,
                                        min(len(ro_prompts), 96), a.adapter_name))

        # ---------- score: THEIR nested entry point, both comparison levels -------
        results, kept = [], []
        for ti, (task, prec) in enumerate(accepted):
            blocks, recs_by_sc = [], []
            for sj in range(len(scaffolds_by_task[ti])):
                mine = [ro_recs[k] for k in range(len(ro_recs)) if ro_owner[k] == (ti, sj)]
                blocks.append([(r["text"], unanswered(r)) for r in mine])
                recs_by_sc.append(mine)
            try:
                res = run_iteration_nested(cfg, ncfg, task, scaffolds_by_task[ti], blocks,
                                           buffer, judges)
            except (GuardViolation, ValueError) as exc:
                key = type(exc).__name__ + ":" + str(exc).split("(")[0][:60]
                stats["refusals"][key] = stats["refusals"].get(key, 0) + 1
                # A refusal without its k-vector is unreadable: "all groups degenerate" can
                # mean the proposer made a task the solver always solves or one it never
                # solves, and those are opposite failures with opposite repairs. Grade the
                # blocks we already hold, with their grader, and record the counts.
                ks = []
                for sj, blk in enumerate(blocks):
                    try:
                        rs = [olgrade(scaffolds_by_task[ti][sj], task, t, tr) for t, tr in blk]
                        ks.append(sum(1 for r in rs
                                      if r.outcome.value == "success"))
                    except Exception:
                        ks.append(None)
                stats.setdefault("refused_k_vectors", []).append(
                    {"k_per_scaffold": ks, "G": ncfg.n_rollouts_per_scaffold})
                continue
            results.append(res)
            kept.append((ti, task, prec, recs_by_sc))
        stats["tasks_scored"] = len(results)
        if len(results) < 2:
            stats["refusals"]["scored_group_too_small"] = 1
            logf.write(json.dumps(stats) + "\n"); logf.flush()
            print("[orn] iter %d: %d task(s) survived scoring; no task group" % (it, len(results)),
                  flush=True)
            it += 1
            continue

        # ---------- the one group nested.py cannot form: across tasks -------------
        task_adv = grpo_advantages([r.task_record.R_task for r in results],
                                   epsilon=cfg.epsilon)

        # ---------- three GRPO updates, in cfg.stage_order ------------------------
        upd = {}
        for stage in cfg.stage_order:
            rows = []
            if stage == "solver":
                for res, (ti, task, prec, recs_by_sc) in zip(results, kept):
                    for sj, ga in enumerate(res.rollout_advantages):
                        if ga.degenerate:
                            continue
                        for i, adv in enumerate(ga.advantages):
                            r = recs_by_sc[sj][i]
                            if r["output_ids"]:
                                rows.append({"prompt_ids": r["prompt_ids"],
                                             "output_ids": r["output_ids"], "adv": adv})
            elif stage == "harness":
                for res, (ti, task, prec, _) in zip(results, kept):
                    ga = res.scaffold_advantages
                    if ga is None or ga.degenerate:
                        continue
                    for sj, adv in enumerate(ga.advantages):
                        r = screc_by_task[ti][sj]
                        if r["output_ids"]:
                            rows.append({"prompt_ids": r["prompt_ids"],
                                         "output_ids": r["output_ids"], "adv": adv})
            else:  # proposer
                if not task_adv.degenerate:
                    for adv, (ti, task, prec, _) in zip(task_adv.advantages, kept):
                        if prec["output_ids"]:
                            rows.append({"prompt_ids": prec["prompt_ids"],
                                         "output_ids": prec["output_ids"], "adv": adv})
            if rows:
                u = train_arm.policy_step(model, rows, opt, device, a.rollout_cap, 1.0,
                                          a.mb_tokens, pad_id, "sequence")
            else:
                u = {"loss": None, "tokens": 0, "grad_norm": 0.0}
            upd[stage] = {"rows": len(rows), **{k: u[k] for k in ("loss", "tokens", "grad_norm")
                                                if k in u}}

        # ---------- the process-level statistics this run is FOR ------------------
        rgroups = [g for res in results for g in res.rollout_advantages]
        sgroups = [res.scaffold_advantages for res in results
                   if res.scaffold_advantages is not None]
        tr = [res.task_record for res in results]
        golds = [ol.gold_agreement(res, res.task_record.task) for res in results]
        golds = [g for g in golds if g is not None]
        stats.update({
            "informative_rollout_groups": sum(1 for g in rgroups if not g.degenerate) / max(len(rgroups), 1),
            "n_rollout_groups": len(rgroups),
            "informative_scaffold_groups": sum(1 for g in sgroups if not g.degenerate) / max(len(sgroups), 1),
            "n_scaffold_groups": len(sgroups),
            "task_group_informative": (not task_adv.degenerate),
            "mean_p_hat": sum(t.p_hat for t in tr) / len(tr),
            "p_hats": [round(t.p_hat, 4) for t in tr],
            "mean_novelty_N": sum(t.N for t in tr) / len(tr),
            "mean_validity_V": sum(t.V for t in tr) / len(tr),
            "mean_difficulty_D": sum(t.D for t in tr) / len(tr),
            "mean_R_task": sum(t.R_task for t in tr) / len(tr),
            "mean_R_harness": sum(h.R_harness for res in results for h in res.harness_records)
                              / max(sum(len(res.harness_records) for res in results), 1),
            "mean_gold_agreement": (sum(golds) / len(golds)) if golds else None,
            "n_aborted": sum(t.n_aborted for t in tr),
            "n_rollouts": sum(len(t.rollouts) for t in tr),
            "buffer_size": len(buffer.texts()),
            "stages_fired": results[0].stages_fired,
            "updates": upd,
            "rollout_tokens": sum(len(r["output_ids"]) for r in ro_recs),
            "scaffold_tokens": sum(len(r["output_ids"]) for r in sc_recs),
            "gen_tokens": (sum(len(r["output_ids"]) for r in ro_recs)
                           + sum(len(r["output_ids"]) for r in sc_recs) + prop_tokens),
            "elapsed_s": round(time.time() - t0, 1),
            "iter_s": round(time.time() - tic, 1),
            "adapter_absB": train_arm.adapter_fingerprint(model),
        })
        logf.write(json.dumps(stats) + "\n")
        logf.flush()
        print("[orn] iter %d  tasks %d/%d  p_hat %.3f  N %.3f  info(roll) %.2f info(scaf) %.2f "
              "task_grp %s  gold %s  %.0fs"
              % (it, stats["tasks_scored"], a.tasks_per_iter, stats["mean_p_hat"],
                 stats["mean_novelty_N"], stats["informative_rollout_groups"],
                 stats["informative_scaffold_groups"], stats["task_group_informative"],
                 ("%.2f" % stats["mean_gold_agreement"]) if stats["mean_gold_agreement"] is not None else "n/a",
                 stats["iter_s"]), flush=True)

        if any(u.get("tokens") for u in upd.values()):
            train_arm.save_adapter(model, adapter_dir)
            train_arm.push_adapter(a.url, a.adapter_name, adapter_dir)
            if not route_checked:
                pr = asyncio.run(gate_lib.assert_adapter_routes(
                    a.url, gate_lib.build_prompt(tok, "What is 17 times 23?", "low"),
                    a.adapter_name))
                json.dump(pr, open(os.path.join(a.run_dir, "route_probe.json"), "w"), indent=1)
                route_checked = True
                print("[orn] ROUTE VERIFIED after first update", flush=True)
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
        scored = [r for r in recs if r.get("tasks_scored", 0) >= 2]
        problems = []
        if not scored:
            problems.append("no iteration produced a scorable task group")
        else:
            r = scored[-1]
            if set(r["stages_fired"]) != {"rollout", "harness", "task"}:
                problems.append("nested.run_iteration_nested did not fire all three stages: %s"
                                % r["stages_fired"])
            if r["n_scaffold_groups"] == 0:
                problems.append("no scaffold group formed: the nesting collapsed to flat")
            if r["informative_rollout_groups"] == 0:
                problems.append("every rollout group was degenerate: nothing to learn from")
            for st in ("solver", "harness", "proposer"):
                u = r["updates"].get(st, {})
                if not u.get("tokens"):
                    problems.append("stage %r took no gradient step (rows=%s)"
                                    % (st, u.get("rows")))
                elif not (u.get("grad_norm") or 0) > 0:
                    problems.append("stage %r stepped with grad_norm 0" % st)
            if not (r.get("adapter_absB") or 0) > 0:
                problems.append("adapter LoRA-B is still exactly zero: nothing was updated")
            if not route_checked:
                problems.append("the served adapter was never route-verified")
        if problems:
            for p in problems:
                print("SMOKE FAIL: " + p, file=sys.stderr)
            return 1
        print("SMOKE PASS: three stages fired, both nested comparison levels formed, all "
              "three stages took a non-zero gradient step, adapter updated and route verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
