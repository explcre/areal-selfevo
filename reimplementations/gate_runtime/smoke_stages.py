#!/usr/bin/env python3
"""Prove each of the three stages trains, separately, on observable state.

The check this replaces asserted that SOME stage stepped, which is why it passed a run in
which the harness stage was handed an empty list every iteration. It is the seventh guard in
this project to pass exactly the condition it was written to refuse, so this one is built the
other way round: it asserts per stage, on three observables, and it is itself tested against a
stage deliberately starved of data, which it must FAIL.

Per stage the three observables are:

* rows populated -- the optimiser was handed this stage's data, not an empty list;
* gradient norm non-zero -- a loss was actually differentiated;
* adapter fingerprint moved -- summed |LoRA-B| over the whole adapter, exactly zero at init,
  so a change proves parameters moved rather than that a step function returned.

Advantages here are assigned directly rather than earned, because this tests the MACHINERY of
each stage -- whether data of that stage's shape reaches the optimiser -- not whether the loop
produces such data. Whether it does is what the run measures, and the two must not be
conflated: that conflation is how the harness defect survived.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import torch

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

import ornith_train as ot  # noqa: E402
import train_arm  # noqa: E402
from ornith_repro import live as ol  # noqa: E402


def rows_from(recs, advs):
    """Turn generations into policy-gradient rows, dropping any that produced no tokens."""
    out = []
    for r, adv in zip(recs, advs):
        if r["output_ids"]:
            out.append({"prompt_ids": r["prompt_ids"], "output_ids": r["output_ids"],
                        "adv": adv})
    return out


def check_stage(name, model, rows, opt, device, cap, mb_tokens, pad_id):
    """Run one stage's step and return the three observables plus a pass/fail verdict."""
    fp0 = train_arm.adapter_fingerprint(model)
    if rows:
        u = train_arm.policy_step(model, rows, opt, device, cap, 1.0, mb_tokens, pad_id,
                                  "sequence")
    else:
        u = {"loss": None, "tokens": 0, "grad_norm": 0.0}
    fp1 = train_arm.adapter_fingerprint(model)
    gn = u.get("grad_norm") or 0.0
    d = abs(fp1 - fp0)
    ok = bool(rows) and gn > 0 and d > 0
    return {"stage": name, "rows": len(rows), "grad_norm": round(float(gn), 6),
            "fp_delta": float(d), "tokens": u.get("tokens", 0), "PASS": ok}


def main() -> int:
    """Exercise proposer, harness and solver separately; fail loudly on any that did not."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--seed-pool", required=True)
    ap.add_argument("--gen-cap", type=int, default=4096)
    ap.add_argument("--rollout-cap", type=int, default=2048)
    ap.add_argument("--mb-tokens", type=int, default=16384)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/smoke_stages.json")
    a = ap.parse_args()

    pool = json.load(open(a.seed_pool))["tasks"]
    task_text = pool[0]["problem"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = "cuda:0"
    print("[smoke] loading base model", flush=True)
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
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    print("[smoke] coverage %s" % cov, flush=True)
    assert train_arm.adapter_fingerprint(model) == 0.0, "LoRA-B must start at exactly zero"

    # One real generation per stage, in that stage's own prompt shape.
    solved = [t["problem"] for t in pool[:3]]
    prop_prompts = [ol.PROPOSER_PROMPT.format(
        solved="\n".join("- " + s[:400] for s in solved),
        unsolved="\n".join("- " + s[:400] for s in solved), novelty="")] * 2
    sc_prompts = [ol.SCAFFOLD_PROMPT.format(task=task_text)] * 2
    print("[smoke] generating proposer and scaffold blocks", flush=True)
    prop = asyncio.run(ot.gen_batch(a.url, tok, prop_prompts, a.gen_cap, 2, ""))
    scaf = asyncio.run(ot.gen_batch(a.url, tok, sc_prompts, a.gen_cap, 2, ""))
    ro_prompts = [ol.SOLVER_PROMPT.format(instructions="Solve it. Show the final answer.",
                                          task=task_text)] * 2
    roll = asyncio.run(ot.gen_batch(a.url, tok, ro_prompts, a.rollout_cap, 2, ""))
    print("[smoke] tokens: proposer %d, harness %d, solver %d"
          % (sum(len(r["output_ids"]) for r in prop),
             sum(len(r["output_ids"]) for r in scaf),
             sum(len(r["output_ids"]) for r in roll)), flush=True)

    cap = max(a.gen_cap, a.rollout_cap)
    checks = [
        check_stage("proposer", model, rows_from(prop, [+1.0, -1.0]), opt, device, cap,
                    a.mb_tokens, pad_id),
        check_stage("harness", model, rows_from(scaf, [+1.0, -1.0]), opt, device, cap,
                    a.mb_tokens, pad_id),
        check_stage("solver", model, rows_from(roll, [+1.0, -1.0]), opt, device, cap,
                    a.mb_tokens, pad_id),
    ]
    # The check must have teeth: a stage handed nothing has to FAIL it. This is the exact
    # condition the previous smoke test passed.
    starved = check_stage("STARVED CONTROL (must fail)", model, [], opt, device, cap,
                          a.mb_tokens, pad_id)

    print("\n%-30s %6s %11s %12s %8s %s"
          % ("stage", "rows", "grad_norm", "fp_delta", "tokens", "verdict"))
    for c in checks + [starved]:
        print("%-30s %6d %11.6f %12.6f %8d %s"
              % (c["stage"], c["rows"], c["grad_norm"], c["fp_delta"], c["tokens"],
                 "PASS" if c["PASS"] else "FAIL"))

    json.dump({"coverage": cov, "checks": checks, "starved_control": starved},
              open(a.out, "w"), indent=1, default=str)

    bad = [c["stage"] for c in checks if not c["PASS"]]
    if starved["PASS"]:
        print("\nFAILED: the starved control PASSED, so this check cannot detect the defect "
              "it exists to detect.")
        return 2
    if bad:
        print("\nFAILED: these stages did not train: %s" % bad)
        return 1
    print("\nAll three stages trained separately, and the starved control failed as it must.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
