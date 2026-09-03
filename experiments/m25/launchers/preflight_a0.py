"""Refuse a launch that would OOM, sample the wrong policy, or share cards.

Resolves the launcher's overrides through AReaL's own Hydra + dataclass path, so what is
checked is what the trainer will build. Trimmed from lora30b.sh's preflight to the checks
that still bind at 4 GPUs.
"""
from __future__ import annotations

import dataclasses
import json
import hashlib
import os
import sys

import yaml
from omegaconf import OmegaConf

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

GIB = 1024.0**3
N_GPUS = int(os.environ["SMOKE_NGPUS"])
GPU_GIB = float(os.environ["SMOKE_GPU_GIB"])
MIN_KV_GIB = float(os.environ["SMOKE_MIN_KV_GIB"])
ACTOR_SHARD_FRAC = float(os.environ["SMOKE_ACTOR_SHARD_FRAC"])
DUMP = os.environ["SMOKE_DUMP"]

_rows = []


def check(ok, label, detail):
    """Record one preflight verdict and return it."""
    _rows.append((bool(ok), label, detail))
    return bool(ok)


def note(label, detail):
    """Record an informational line that can never fail the preflight."""
    _rows.append((None, label, detail))


def model_params(path):
    """Return (parameter count, note) read from the checkpoint's safetensors index."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx))
        total = float(j["metadata"]["total_size"])
        files = sorted(set(j["weight_map"].values()))
        on_disk = sum(
            os.path.getsize(os.path.join(path, f))
            for f in files
            if os.path.exists(os.path.join(path, f))
        )
        missing = [f for f in files if not os.path.exists(os.path.join(path, f))]
        return total / 2.0, (
            f"{len(files)} shards, index total_size={total:.0f} B, on disk {on_disk:.0f} B, "
            f"missing={missing}"
        )
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return os.path.getsize(single) / 2.0, "single-file safetensors"
    raise FileNotFoundError(f"no safetensors index or model.safetensors under {path}")


cfg, _ = parse_cli_args(sys.argv[1:])
obj = OmegaConf.to_object(to_structured_cfg(cfg, config_cls=GRPOConfig))
with open(DUMP, "w") as fh:
    yaml.dump(dataclasses.asdict(obj), fh, default_flow_style=False, sort_keys=False)
note("resolved config", f"written to {DUMP}")
check(
    type(obj.actor).__name__ == "PPOActorConfig",
    "config is the real dataclass",
    f"actor is {type(obj.actor).__name__}",
)

a, r, sg, g = obj.actor, obj.rollout, obj.sglang, obj.gconfig

note("actor.path", a.path)
note("HF_HOME as the preflight sees it", repr(os.environ.get("HF_HOME")))

ckpt_ok = check(
    os.path.isdir(a.path),
    "checkpoint path exists",
    a.path if os.path.isdir(a.path) else f"{a.path} IS NOT A DIRECTORY ON THIS HOST",
)
n_params = 0.0
if ckpt_ok:
    try:
        n_params, detail = model_params(a.path)
        note("checkpoint", f"{n_params / 1e9:.2f}B params; {detail}")
    except Exception as exc:  # noqa: BLE001
        check(False, "checkpoint is readable", f"{type(exc).__name__}: {exc}")

check(
    bool(a.use_lora) and bool(r.use_lora),
    "LoRA consistent across actor and rollout",
    f"actor.use_lora={a.use_lora} rollout.use_lora={r.use_lora}",
)
check(
    bool(sg.enable_lora) is bool(a.use_lora),
    "sglang.enable_lora tracks actor.use_lora",
    f"sglang.enable_lora={sg.enable_lora} actor.use_lora={a.use_lora}",
)
check(
    sg.max_lora_rank == a.lora_rank,
    "sglang.max_lora_rank == actor.lora_rank",
    f"max_lora_rank={sg.max_lora_rank} lora_rank={a.lora_rank}",
)
check(bool(g.lora_name), "gconfig.lora_name is set", f"lora_name={g.lora_name!r}")
check(
    a.weight_update_mode == "disk",
    "actor.weight_update_mode == disk",
    f"weight_update_mode={a.weight_update_mode!r}",
)
for role, engine in (("actor", a), ("rollout", r)):
    st = getattr(engine.scheduling_strategy, "type", None)
    st = getattr(st, "value", st)
    check(st != "colocation", f"{role} is not colocated", f"{role}.scheduling_strategy.type={st!r}")
check(
    float(a.kl_ctl) == 0.0,
    "actor.kl_ctl == 0.0, so no ref model is built",
    f"kl_ctl={a.kl_ctl}",
)
check(
    bool(a.fsdp.memory_efficient_load),
    "actor.fsdp.memory_efficient_load is on",
    f"memory_efficient_load={a.fsdp.memory_efficient_load}",
)

actor_alloc = ModelAllocation.from_str(a.backend, name="actor")
rollout_alloc = ModelAllocation.from_str(r.backend, name="rollout")
aw = actor_alloc.parallel.world_size
rw = rollout_alloc.parallel.world_size
check(
    aw + rw <= N_GPUS,
    "actor + rollout GPUs fit the node",
    f"{a.backend} needs {aw} + {r.backend} needs {rw} = {aw + rw} of {N_GPUS}",
)

check(
    obj.total_train_steps is not None and obj.total_train_steps > 0,
    "total_train_steps is bounded",
    f"total_train_steps={obj.total_train_steps} (an unbounded smoke run is not a smoke run)",
)
check(
    obj.saver.freq_steps is not None and obj.saver.freq_steps > 0,
    "saver.freq_steps is set",
    f"saver.freq_steps={obj.saver.freq_steps} (no step saves means no checkpoint to verify)",
)
check(
    obj.evaluator.freq_epochs is None
    and obj.evaluator.freq_secs is None
    and obj.evaluator.freq_steps is None,
    "in-training evaluator disabled",
    f"freq_epochs={obj.evaluator.freq_epochs} freq_secs={obj.evaluator.freq_secs} "
    f"freq_steps={obj.evaluator.freq_steps} (it deadlocks this stack)",
)

if n_params:
    store = 4.0 if str(a.optimizer_dtype).endswith("32") else 2.0
    train_total = n_params * store / GIB
    per_actor = train_total / max(actor_alloc.parallel.data_parallel_size, 1)
    note(
        "actor per-GPU base params",
        f"{n_params / 1e9:.2f}B x {store:.0f} B ({a.optimizer_dtype}) = {train_total:.1f} GiB "
        f"/ dp{actor_alloc.parallel.data_parallel_size} = {per_actor:.1f} GiB of {GPU_GIB:.2f}",
    )
    check(
        per_actor < GPU_GIB * ACTOR_SHARD_FRAC,
        "actor shard leaves room for activations",
        f"{per_actor:.1f} GiB = {per_actor / GPU_GIB:.2f} of {GPU_GIB:.2f} GiB per actor GPU "
        f"(ceiling {ACTOR_SHARD_FRAC:.2f}); headroom {GPU_GIB - per_actor:.1f} GiB for the FSDP "
        f"all-gather, activations and fragmentation",
    )
    tp = rollout_alloc.parallel.tensor_parallel_size
    infer_per_gpu = (n_params * 2 / GIB) / max(tp, 1)
    pool = float(sg.mem_fraction_static) * GPU_GIB
    kv = pool - infer_per_gpu
    note(
        "sglang per-GPU",
        f"weights {n_params * 2 / GIB:.1f} GiB / tp{tp} = {infer_per_gpu:.1f} GiB; "
        f"static pool {sg.mem_fraction_static} x {GPU_GIB:.2f} = {pool:.1f} GiB; KV = {kv:.1f} GiB",
    )
    check(kv >= MIN_KV_GIB, "sglang KV cache above the floor", f"{kv:.1f} >= {MIN_KV_GIB:.1f} GiB")


# ---------------------------------------------------------------------------------------
# A0-specific refusals, from experiments/m25/PLAN.md. A0 is the baseline every other arm is
# measured against, so a config that quietly carried a router, a harness ladder or a
# normaliser would move the reference point everything else is compared to.
# ---------------------------------------------------------------------------------------
gr = getattr(a, "group_routing", None)
check(
    gr is None or (not getattr(gr, "enabled", False)
                   and not getattr(gr, "router", None)
                   and not getattr(gr, "harness_variants", None)
                   and not getattr(gr, "harness_selector", None)),
    "A0 carries no partition and no harness ladder",
    f"group_routing={'null' if gr is None else 'set'}"
    + ("" if gr is None else
       f" enabled={getattr(gr, 'enabled', None)} router={getattr(gr, 'router', None)} "
       f"variants={getattr(gr, 'harness_variants', None)} "
       f"selector={getattr(gr, 'harness_selector', None)}"),
)
check(
    a.adv_norm is None,
    "actor.adv_norm is null",
    f"adv_norm={a.adv_norm} (the plan forbids mean_level=group, and A0 runs with none)",
)
check(
    "deepmath" in obj.train_dataset.path.lower(),
    "training on DeepMath, not MATH or GSM8K",
    f"train_dataset.path={obj.train_dataset.path!r} (measured on this box: A0's own MATH "
    f"batches were 76.7% unanimous, so only 23.3% of groups carried any gradient)",
)
check(
    "decontam" in obj.train_dataset.path.lower(),
    "training on the DECONTAMINATED copy",
    f"{obj.train_dataset.path!r} -- rows containing OlympiadBench problems by word-5-gram "
    f"or LaTeX-normalised char-25-gram containment >= 0.3 were dropped; see "
    f"dropped_rows.json beside the parquet",
)
note(
    "valid_dataset",
    f"{obj.valid_dataset.path!r} split={obj.valid_dataset.split!r} -- DeepMath ships no test "
    f"split, and this is never sampled because the in-training evaluator is disabled below",
)
note(
    "fixed generation cap",
    f"gconfig.max_new_tokens={g.max_new_tokens}, max_tokens={g.max_tokens}, "
    f"actor.max_new_tokens={a.max_new_tokens} -- constant for the whole run, no ladder",
)
check(
    a.max_new_tokens == g.max_new_tokens,
    "actor and rollout agree on the cap",
    f"actor.max_new_tokens={a.max_new_tokens} gconfig.max_new_tokens={g.max_new_tokens} "
    f"(they must match or truncation is measured against the wrong number)",
)
check(
    g.max_tokens >= g.max_new_tokens + obj.train_dataset.max_length,
    "total token budget covers the longest prompt plus a full response",
    f"max_tokens={g.max_tokens} >= max_new_tokens={g.max_new_tokens} + "
    f"train_dataset.max_length={obj.train_dataset.max_length}",
)
# ---------------------------------------------------------------- launcher provenance ---
# ~/harness4 is NOT a git repository, and this file and run_a0.sh are what decide which arm
# runs. An unversioned edit to either is an unrecorded change to the experiment. The
# committed copies live in experiments/m25/launchers/; a divergence refuses the launch
# rather than being noted, because a note is exactly what nobody reads at 3am.
_MIRROR = "/home/ubuntu/areal-selfevo/experiments/m25/launchers"


def _sha(path):
    """sha256 of a file, or None if it is not there."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


_here = os.path.dirname(os.path.abspath(__file__))
for _name in ("run_a0.sh", "preflight_a0.py"):
    _live_h = _sha(os.path.join(_here, _name))
    _mirr_h = _sha(os.path.join(_MIRROR, _name))
    check(
        _live_h is not None and _live_h == _mirr_h,
        f"{_name} matches its committed copy",
        f"live={_live_h[:12] if _live_h else 'MISSING'} "
        f"committed={_mirr_h[:12] if _mirr_h else 'MISSING'}"
        + (
            ""
            if _live_h == _mirr_h
            else f" -- copy {os.path.join(_here, _name)} to {_MIRROR}/ and commit it, "
            "or the arm that runs is not the arm that is recorded"
        ),
    )

# Arm identity, checked against the launcher rather than read out of the config it
# produced. An arm that names the fix and resolves to the baseline is the failure this
# refuses: nothing else in the run would disagree, because 'keep' IS what every prior run
# did and every metric would look normal.
_want_trunc = os.environ.get("SMOKE_TRUNC_ADV", "keep")
check(
    a.truncated_advantage == _want_trunc,
    "actor.truncated_advantage is the arm the launcher asked for",
    f"resolved={a.truncated_advantage!r} launcher asked for {_want_trunc!r}"
    + (
        ""
        if a.truncated_advantage == _want_trunc
        else " -- the override did not reach the config"
    ),
)
note(
    "batch shape",
    f"{obj.train_dataset.batch_size} prompts x {g.n_samples} samples = "
    f"{obj.train_dataset.batch_size * g.n_samples} rollouts per step",
)
# The check the old note asserted by assumption. max_tokens_per_mb packs sequences into a
# microbatch; it cannot SPLIT one. A single sequence longer than the cap therefore forms its
# own oversized microbatch and peak activation memory is set by that sequence, not by the cap.
_worst_seq = int(obj.train_dataset.max_length) + int(g.max_new_tokens)
_mb_cap = int(a.mb_spec.max_tokens_per_mb)
check(
    _worst_seq <= _mb_cap,
    "no single sequence can exceed max_tokens_per_mb",
    f"worst case train_dataset.max_length={obj.train_dataset.max_length} + "
    f"gconfig.max_new_tokens={g.max_new_tokens} = {_worst_seq} tokens vs "
    f"actor.mb_spec.max_tokens_per_mb={_mb_cap}. "
    + (
        f"Fits, so max_tokens_per_mb genuinely bounds peak activation memory."
        if _worst_seq <= _mb_cap
        else f"A {_worst_seq}-token sequence CANNOT be split and forms its own microbatch at "
        f"{_worst_seq / _mb_cap:.2f}x the cap; measured on this box the transient pool is "
        f"9.95 GB at {_mb_cap} tokens against a 3.37 GB margin, so this overruns the card. "
        f"Lower gconfig.max_new_tokens or train_dataset.max_length, or raise "
        f"actor.mb_spec.max_tokens_per_mb only if the margin allows."
    ),
)
note(
    "checkpoint cadence",
    f"saver.freq_steps={obj.saver.freq_steps}",
)

width = max(len(label) for _, label, _ in _rows)
failed = 0
for ok, label, detail in _rows:
    tag = "note" if ok is None else ("ok  " if ok else "FAIL")
    failed += 1 if ok is False else 0
    print(f"[{tag}] {label.ljust(width)}  {detail}")
print(f"\nPREFLIGHT {'FAIL' if failed else 'PASS'} ({failed} failed)")
sys.exit(1 if failed else 0)
