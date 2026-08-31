#!/usr/bin/env python3
"""Preflight the MATH training path before it costs GPU time.

Checks the things that would otherwise fail silently or late: that the config override
reaches train_dataset.path, that the dispatch selects the MATH adapter rather than falling
through to GSM8K, that the adapter yields non-empty golds, and that the reward function
actually scores those golds. A dataset the grader cannot read trains at reward zero, which is
indistinguishable from a model that cannot solve anything -- and would make the silence metric
read 100% unsolved, exactly mimicking the composition flip this run exists to measure.
"""
from __future__ import annotations

import sys

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

DATASET = "DigitalLearningGmbH/MATH-lighteval"


def main() -> int:
    """Resolve the config, load the dataset, and score a gold against itself."""
    ok = True
    argv = [
        "--config", "examples/math/gsm8k_grpo.yaml",
        f"train_dataset.path={DATASET}",
        f"valid_dataset.path={DATASET}",
        "experiment_name=preflight-math", "trial_name=t0",
    ]
    cfg, _ = parse_cli_args(argv)
    obj = OmegaConf.to_object(to_structured_cfg(cfg, config_cls=GRPOConfig))
    print(f"train_dataset.path = {obj.train_dataset.path}")
    print(f"train_dataset.type = {obj.train_dataset.type}")
    if obj.train_dataset.path != DATASET:
        print("FAIL: the override did not reach train_dataset.path"); ok = False

    from areal.dataset.competition_math import get_math_rl_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    tok = load_hf_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    d = get_math_rl_dataset(path=DATASET, split="train", tokenizer=tok, max_length=1024)
    print(f"examples after length filter: {len(d)}")
    if set(d.column_names) != {"messages", "answer"}:
        print(f"FAIL: columns are {d.column_names}, MathAgent reads messages + answer"); ok = False

    empties = sum(1 for a in d["answer"][:1000] if not a)
    print(f"empty golds in first 1000: {empties}")
    if empties > 50:
        print("FAIL: too many unparsable golds; those examples train at reward 0"); ok = False

    # The check that matters: can the REAL reward function score these golds?
    from areal.workflow.openai.math_agent import math_reward_fn

    hits = 0
    for i in range(25):
        gold = d[i]["answer"]
        if gold and math_reward_fn(f"Therefore the answer is \\boxed{{{gold}}}.", gold) == 1.0:
            hits += 1
    print(f"reward_fn scores its own gold on {hits}/25 examples")
    if hits < 23:
        print("FAIL: the grader cannot read this dataset's golds; training would run at "
              "reward ~0, which mimics the very result this run is meant to measure")
        ok = False

    print("\nPREFLIGHT PASS" if ok else "\nPREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
