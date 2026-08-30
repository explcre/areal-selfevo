#!/usr/bin/env python3
"""Preflight: does the CLI override actually reach actor.group_routing?

Answers the question a syntax check cannot: a Hydra override can parse cleanly, land on a
key nothing reads, and leave the run silently identical to the control. That failure is
invisible in the log and would make the whole A/B a null by construction.

Uses the SAME loader the trainer uses (parse_cli_args + to_structured_cfg), so a divergence
between preflight and run is not possible.
"""
from __future__ import annotations

import sys

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

BASE = [
    "--config", "examples/math/gsm8k_grpo.yaml",
    "actor.path=Qwen/Qwen2.5-1.5B-Instruct",
    "~actor.adv_norm",
    "experiment_name=preflight", "trial_name=t0",
]
ON = ["+actor.group_routing.enabled=true", "+actor.group_routing.solved_advantage=0.5"]


def load(argv: list[str]):
    """Resolve a config exactly as the trainer does, minus the side effects."""
    cfg, _ = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=GRPOConfig)
    return OmegaConf.to_object(cfg)


def main() -> int:
    off = load(BASE)
    on = load(BASE + ON)

    print(f"ARM off -> actor.group_routing = {off.actor.group_routing!r}")
    print(f"ARM on  -> actor.group_routing = {on.actor.group_routing!r}")

    ok = True
    if off.actor.group_routing is not None:
        print("FAIL: the control arm is not the vanilla default"); ok = False
    gr = on.actor.group_routing
    if gr is None:
        print("FAIL: the override did not create a group_routing object at all"); ok = False
    else:
        if not getattr(gr, "enabled", False):
            print("FAIL: enabled did not travel"); ok = False
        if float(getattr(gr, "solved_advantage", 0.0)) != 0.5:
            print(f"FAIL: solved_advantage did not travel (got {gr.solved_advantage!r})"); ok = False
        # The actor reads it via getattr on the dataclass, so the object must BE the
        # dataclass and not a DictConfig that merely looks like one.
        if type(gr).__name__ != "GroupRoutingConfig":
            print(f"FAIL: wrong type {type(gr).__name__}; the actor's getattr would still "
                  f"work but the sign guards in __post_init__ would not have run"); ok = False

    # The sign guard must survive the CLI path too, or a typo becomes a silent bad run.
    try:
        load(BASE + ["+actor.group_routing.enabled=true",
                     "+actor.group_routing.unsolved_advantage=0.5"])
        print("FAIL: a positive unsolved_advantage was accepted through the CLI"); ok = False
    except Exception as exc:
        print(f"guard held through the CLI: {type(exc).__name__}: {str(exc)[:90]}")

    print("\nPREFLIGHT PASS" if ok else "\nPREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
