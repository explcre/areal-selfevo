#!/usr/bin/env python3
"""Build a deliberately NON-NULL LoRA adapter, straight from the checkpoint's tensor shapes.

Why this exists. Two independent failure modes on this stack are silent, and a freshly
initialised adapter (``B = 0``) cannot distinguish either of them from success, because a
zero-initialised adapter is mathematically identical to the base model:

  1. an adapter reaches the server only through a request's adapter-path field; naming it in
     the ``model`` field returns base-model output with HTTP 200 and no warning;
  2. ``target_modules`` inherited from an older Qwen matches ``self_attn.{q,k,v,o}_proj``,
     which on this checkpoint exists on **16 of 64** decoder layers -- the other 48 are
     ``linear_attn`` -- and the shortfall is invisible in forward loss.

So the probe adapter is built with both factors non-zero. Any request routed through it MUST
produce different text from the base model; if it does not, the route is dead.

Shapes are read from the safetensors headers rather than by instantiating a 27B model, which
keeps this runnable on a CPU while the GPUs are busy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

# The three module families of the language model, and the layer count each one reaches on
# Qwen3.8-27B. Recorded here so the coverage assertion below has something to compare with
# that was not derived from the same walk it is checking.
EXPECTED_LAYER_COVERAGE = {
    "self_attn": 16,    # full_attention layers only (full_attention_interval = 4)
    "linear_attn": 48,  # gated-delta-net layers
    "mlp": 64,          # every layer
}


def read_shapes(model_dir: str) -> dict[str, list[int]]:
    """Map every tensor name in a sharded safetensors checkpoint to its shape.

    Reads only each shard's JSON header (the first 8 bytes give its length), so the weights
    themselves are never touched.

    Args:
        model_dir: Directory holding ``model.safetensors.index.json`` and its shards.

    Returns:
        Tensor name -> shape.
    """
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    shapes: dict[str, list[int]] = {}
    for shard in shards:
        with open(os.path.join(model_dir, shard), "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = meta["shape"]
    return shapes


def select_targets(shapes: dict[str, list[int]], pattern: str) -> list[str]:
    """Weight names of the linear modules a target pattern reaches.

    Args:
        shapes: Tensor name -> shape, from :func:`read_shapes`.
        pattern: Regex matched against the MODULE name (the weight name minus ``.weight``).

    Returns:
        Sorted module names (no ``.weight`` suffix), 2-D tensors only -- a 1-D match would be
        a norm or a bias and LoRA does not apply to it.
    """
    rx = re.compile(pattern)
    out = []
    for name, shape in shapes.items():
        if not name.endswith(".weight") or len(shape) != 2:
            continue
        module = name[: -len(".weight")]
        if rx.fullmatch(module):
            out.append(module)
    return sorted(out)


def layer_coverage(modules: list[str]) -> dict[str, set[int]]:
    """Which decoder layers each module family is reached on.

    This is the assertion that catches the 16-of-64 trap. It counts DISTINCT LAYER INDICES
    per family rather than module hits, because a config can match many modules and still
    leave three quarters of the depth untouched.

    Args:
        modules: Module names as returned by :func:`select_targets`.

    Returns:
        Family name -> set of layer indices reached.
    """
    cov: dict[str, set[int]] = {k: set() for k in EXPECTED_LAYER_COVERAGE}
    for m in modules:
        hit = re.search(r"layers\.(\d+)\.(self_attn|linear_attn|mlp)\.", m)
        if hit:
            cov[hit.group(2)].add(int(hit.group(1)))
    return cov


def main() -> int:
    """Write a non-null PEFT adapter and print its measured layer coverage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--pattern", required=True, help="regex over module names")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="std of B; 0.0 gives the ordinary null adapter, >0 a live probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-list", default="",
                    help="comma-separated module SUFFIXES to record in adapter_config.json; "
                         "sglang refuses a regex there (it accepts only a list, 'all' or "
                         "'all-linear'), while the regex above still decides which tensors "
                         "are written")
    a = ap.parse_args()

    shapes = read_shapes(a.model)
    modules = select_targets(shapes, a.pattern)
    if not modules:
        print("FATAL: pattern matched no 2-D module", file=sys.stderr)
        return 2
    cov = layer_coverage(modules)
    print("modules matched: %d" % len(modules))
    for fam, want in EXPECTED_LAYER_COVERAGE.items():
        print("  %-12s layers reached %2d / %2d" % (fam, len(cov[fam]), want))
    depth = sorted(set().union(*cov.values()))
    print("  distinct decoder layers reached: %d / 64" % len(depth))

    rng = np.random.default_rng(a.seed)
    tensors: dict[str, np.ndarray] = {}
    for m in modules:
        out_f, in_f = shapes[m + ".weight"]
        # PEFT's own initialisation: A ~ Kaiming-uniform, B zeros. `scale` > 0 replaces the
        # zeros so the adapter is observably live.
        bound = (1.0 / in_f) ** 0.5
        A = rng.uniform(-bound, bound, size=(a.rank, in_f)).astype(np.float32)
        B = (rng.normal(0.0, a.scale, size=(out_f, a.rank)).astype(np.float32)
             if a.scale > 0 else np.zeros((out_f, a.rank), dtype=np.float32))
        key = "base_model.model." + m
        tensors[key + ".lora_A.weight"] = A
        tensors[key + ".lora_B.weight"] = B

    os.makedirs(a.out, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(tensors, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.alpha,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": False,
        "target_modules": (sorted(set(a.target_list.split(","))) if a.target_list
                           else a.pattern),
        "base_model_name_or_path": a.model, "modules_to_save": None,
    }
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print("wrote %s (%d tensors, scale=%g)" % (a.out, len(tensors), a.scale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
