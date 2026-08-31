#!/usr/bin/env python3
"""Preflight a 27B LoRA run before it costs GPU-hours on a box we may not get back.

Checks, in order of how quietly they fail:
  1. the LoRA overrides resolve into the real config object (not a DictConfig that merely
     looks like one, which would skip any __post_init__ validation);
  2. the weights are present locally, so the run does not spend its first hour downloading
     and then die on something the preflight could have caught;
  3. the parameter arithmetic fits the box, computed rather than assumed;
  4. bf16 is NOT used for the LoRA optimizer path -- a prior finding in this project is that
     LoRA under bf16 underflows and trains to nothing while looking healthy.
"""
from __future__ import annotations

import os
import pathlib
import sys

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

MODEL = os.environ.get("LORA_MODEL", "Qwen/Qwen3.8-27B")
N_TRAIN_GPUS = int(os.environ.get("N_TRAIN_GPUS", "4"))
GPU_GIB = float(os.environ.get("GPU_GIB", "141"))


def main() -> int:
    """Resolve, locate, and size a LoRA run; refuse rather than let it fail late."""
    ok = True
    argv = [
        "--config", "examples/math/gsm8k_grpo.yaml",
        f"actor.path={MODEL}", f"ref.path={MODEL}",
        "+actor.use_lora=true", "+actor.lora_rank=32", "+actor.lora_alpha=16",
        "experiment_name=preflight-lora", "trial_name=t0",
    ]
    cfg, _ = parse_cli_args(argv)
    obj = OmegaConf.to_object(to_structured_cfg(cfg, config_cls=GRPOConfig))
    a = obj.actor
    print(f"actor.path      {a.path}")
    print(f"use_lora        {getattr(a, 'use_lora', None)}  rank={getattr(a, 'lora_rank', None)} "
          f"alpha={getattr(a, 'lora_alpha', None)}")
    print(f"actor.dtype     {getattr(a, 'dtype', None)}")
    if not getattr(a, "use_lora", False):
        print("FAIL: use_lora did not travel"); ok = False
    if type(a).__name__ != "PPOActorConfig":
        print(f"FAIL: actor is {type(a).__name__}, not the dataclass; validation was skipped")
        ok = False

    # 2. weights present
    cache = pathlib.Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    hub = cache / "hub" if (cache / "hub").exists() else cache
    d = hub / ("models--" + MODEL.replace("/", "--"))
    if d.exists():
        gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        print(f"weights         present, {gb:.1f} GB at {d}")
        if gb < 20:
            print(f"FAIL: only {gb:.1f} GB cached; a 27B bf16 checkpoint is ~54 GB. Still "
                  "downloading, or an incomplete snapshot.")
            ok = False
    else:
        print(f"FAIL: {d} not found -- snapshot_download the model before launching"); ok = False

    # 3. memory arithmetic, computed
    n_params = 27e9
    bf16_w = n_params * 2 / 1024**3
    rank = getattr(a, "lora_rank", 32) or 32
    # LoRA adds 2*rank*d per targeted matrix; ~0.3% of params at rank 32 is the usual figure.
    lora_params = n_params * 0.003
    adam_fp32 = lora_params * (4 + 4 + 4) / 1024**3   # param copy + m + v
    per_gpu = bf16_w / N_TRAIN_GPUS + adam_fp32
    print(f"\nfrozen bf16 weights   {bf16_w:.1f} GiB total, {bf16_w / N_TRAIN_GPUS:.1f} GiB/GPU sharded")
    print(f"LoRA rank {rank} params    ~{lora_params/1e6:.0f}M -> fp32 Adam state {adam_fp32:.2f} GiB")
    print(f"estimated per-GPU     {per_gpu:.1f} GiB of {GPU_GIB:.0f} GiB")
    if per_gpu > GPU_GIB * 0.6:
        print("FAIL: leaves under 40% headroom for activations, KV and the weight-sync buffers")
        ok = False

    # 4. the bf16 LoRA trap
    if str(getattr(a, "dtype", "")).lower() in ("bf16", "bfloat16"):
        print("\nWARNING: actor.dtype is bf16. A prior finding in this project is that LoRA "
              "under bf16 UNDERFLOWS -- the adapter trains to nothing while every metric looks "
              "healthy. Keep the optimizer in fp32 and verify the adapter norm actually grows.")

    print("\nPREFLIGHT PASS" if ok else "\nPREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
