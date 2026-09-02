#!/usr/bin/env python3
"""Preflight the DeepMath-103K training path before it costs GPU time.

Mirrors ``preflight_math.py`` and adds the checks that are specific to THIS corpus, all of
which fail silently or late if they fail at all:

* that the config override reaches ``train_dataset.path``;
* that the dispatch selects the DeepMath adapter and NOT the MATH one -- the id
  ``zwhe99/DeepMath-103K`` satisfies the MATH branch's own ``"math" in path.lower()``
  predicate, so this is a live ordering hazard rather than a formality;
* that the adapter yields non-empty golds and that the REAL reward function can score them --
  a corpus the grader cannot read trains at reward zero, which is indistinguishable from a
  model that cannot solve anything, and would make the silence metric read 100% unsolved,
  mimicking exactly the composition this switch exists to produce;
* that the corpus is the harder operating point it was chosen for, reported as its own
  difficulty distribution rather than assumed from its name;
* that the gold derivations FIT the generation cap they will be padded against, because on
  this corpus they mostly do not, and ``groups_no_fit`` rather than group composition is what
  bounds the gold path's reach.

Usage:
    python3 experiments/harness/preflight_deepmath.py [--max-length 1024] [--gen-cap 8192]
"""
from __future__ import annotations

import argparse
import statistics
import sys

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

DATASET = "zwhe99/DeepMath-103K"

#: Measured on this box over all 103,022 rows, so a drift in the corpus is visible here rather
#: than at step 700. See areal/dataset/deepmath.py for the full table.
EXPECTED_ROWS = 103022


def main() -> int:
    """Resolve the config, load the corpus, characterise it, and score a gold against itself."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=1024,
                    help="prompt length filter, matching train_dataset.max_length")
    ap.add_argument("--gen-cap", type=int, default=8192,
                    help="the generation cap gold rows will be padded against")
    ap.add_argument("--min-difficulty", type=float, default=None)
    args = ap.parse_args()

    ok = True
    argv = [
        "--config", "examples/math/gsm8k_grpo.yaml",
        f"train_dataset.path={DATASET}",
        f"valid_dataset.path={DATASET}",
        "experiment_name=preflight-deepmath", "trial_name=t0",
    ]
    cfg, _ = parse_cli_args(argv)
    obj = OmegaConf.to_object(to_structured_cfg(cfg, config_cls=GRPOConfig))
    print(f"train_dataset.path = {obj.train_dataset.path}")
    print(f"train_dataset.type = {obj.train_dataset.type}")
    if obj.train_dataset.path != DATASET:
        print("FAIL: the override did not reach train_dataset.path"); ok = False

    # Through the DISPATCHER, not the adapter, because the ordering hazard lives in dispatch.
    import areal.dataset as ds
    from areal.utils.hf_utils import load_hf_tokenizer

    tok = load_hf_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

    kwargs = {}
    if args.min_difficulty is not None:
        kwargs["min_difficulty"] = args.min_difficulty
    d = ds._get_custom_dataset(
        path=DATASET, type="rl", split="train", tokenizer=tok,
        max_length=args.max_length, **kwargs,
    )
    print(f"examples after length filter: {len(d)}")
    if set(d.column_names) != {"messages", "answer"}:
        print(f"FAIL: columns are {d.column_names}, MathAgent reads messages + answer"); ok = False

    empties = sum(1 for a in d.select(range(min(1000, len(d))))["answer"] if not a)
    print(f"empty golds in first 1000: {empties}")
    if empties > 0:
        print("FAIL: unparsable golds train at reward 0 and look like an unsolvable problem")
        ok = False

    # The check that matters: can the REAL reward function score these golds?
    from areal.workflow.openai.math_agent import math_reward_fn

    hits = 0
    for i in range(25):
        gold = d[i]["answer"]
        if gold and math_reward_fn(f"Therefore the answer is \\boxed{{{gold}}}.", gold) == 1.0:
            hits += 1
    print(f"reward_fn scores its own gold on {hits}/25 examples")
    if hits < 23:
        print("FAIL: the grader cannot read this corpus's golds; training would run at "
              "reward ~0, which mimics the very result this run is meant to measure")
        ok = False

    # ---- the corpus is the harder operating point, reported not assumed --------------------
    import datasets

    raw = datasets.load_dataset(path=DATASET, split="train")
    if len(raw) != EXPECTED_ROWS:
        print(f"WARN: corpus has {len(raw)} rows, expected {EXPECTED_ROWS}; the measurements "
              f"recorded in areal/dataset/deepmath.py were taken on {EXPECTED_ROWS}")
    diff = list(raw["difficulty"])
    hard = sum(1 for x in diff if x >= 5.0) / len(diff)
    print(f"difficulty: median {statistics.median(diff):.1f}, "
          f"mean {statistics.mean(diff):.2f}, >=5.0 {100 * hard:.1f}%")
    if hard < 0.5:
        print("FAIL: this corpus is not the harder operating point it was selected for")
        ok = False

    # ---- the gold-fit check, which is what actually bounds the gold path here ---------------
    gold_ds = ds._get_custom_dataset(
        path=DATASET, type="rl", split="train[:256]", tokenizer=tok,
        max_length=args.max_length, keep_solution=True, **kwargs,
    )
    lens = sorted(len(g) for g in gold_ds["gold_ids"])
    fits = sum(1 for x in lens if x <= args.gen_cap) / len(lens)
    print(f"gold tokens over 256 rows: median {lens[len(lens) // 2]}, "
          f"p90 {lens[int(0.9 * len(lens))]}, max {lens[-1]}")
    print(f"gold rows fitting a {args.gen_cap}-token cap: {100 * fits:.1f}%")
    if fits < 0.5:
        print(f"WARN: most golds do NOT fit a {args.gen_cap}-token width. selfevo/gold pads the "
              f"gold to the trajectory's width and counts the rest as groups_no_fit, so on this "
              f"corpus the gold path's reach is bounded by the CAP and not by how many groups "
              f"come back unsolved. Raise the cap or truncate deliberately via gold_template.")

    print("\nPREFLIGHT PASS" if ok else "\nPREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
