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
DAPO = ["+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling"]


def load(argv: list[str]):
    """Resolve a config exactly as the trainer does, minus the side effects."""
    cfg, _ = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=GRPOConfig)
    return OmegaConf.to_object(cfg)




def _check_router_modes(name: str) -> bool:
    """Route synthetic units through a router and confirm the seam can apply every mode.

    ``_route_groups`` passes ``decision.argmax()` straight to ``apply_decisions``, which
    implements RL/SFT/SKIP and REFUSES a teacher-requiring mode -- deliberately, because
    treating DISTILL as SKIP would report a distillation arm that never ran. That refusal is
    a ValueError raised inside ``_compute_advantages``, i.e. it kills the run. A router that
    can emit such a mode should therefore be rejected here, before any GPU is allocated,
    rather than at whatever batch first happens to trigger it.

    Args:
        name: A key in ``selfevo.compose.ROUTERS``.

    Returns:
        True if every mode the router produced over the probe grid is applicable.
    """
    from selfevo.compose import ROUTERS
    from selfevo.integration.group_apply import apply_decisions
    from selfevo.observability import FEATURE_NAMES
    from selfevo.routing.base import RoutingContext
    import torch

    factory = ROUTERS.get(name)
    if factory is None:
        print(f"FAIL: router {name!r} is not registered"); return False
    router = factory()

    seen, ok = set(), True
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        for teacher in (False, True):
            extra = {k: 0.5 for k in FEATURE_NAMES}
            extra["solve_rate"] = rate
            ctx = RoutingContext(
                solve_rate=rate, group_size=8, has_teacher=teacher,
                unit_id=f"probe-{rate}-{teacher}", extra=extra,
            )
            seen.add(router.route(ctx).argmax())
    for mode in sorted(seen):
        try:
            apply_decisions(
                torch.zeros(2, 3), torch.ones(2, 3), [2], [mode], sft_weight=0.5
            )
        except ValueError as exc:
            print(f"FAIL: router {name!r} can emit {mode!r}, which the seam cannot apply "
                  f"-- this would raise inside _compute_advantages and kill the run: "
                  f"{str(exc)[:90]}")
            ok = False
    if ok:
        print(f"router {name!r} emits {sorted(seen)}; all applicable")
    return ok


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

    # --- the DAPO baseline arm -------------------------------------------------------
    dapo = load(BASE + DAPO)
    print(f"ARM dapo -> dynamic_filter_fn = {dapo.dynamic_filter_fn!r}")
    print(f"ARM dapo -> dynamic_bs        = {dapo.dynamic_bs!r}")
    if off.dynamic_filter_fn is not None:
        print("FAIL: the control arm has a filter"); ok = False
    if dapo.dynamic_filter_fn is None:
        print("FAIL: the DAPO override did not land"); ok = False
    else:
        from areal.utils.dynamic_import import import_from_string
        import torch

        fn = import_from_string(dapo.dynamic_filter_fn)
        # Decisions, not just importability: a filter that accepts everything is vanilla GRPO
        # wearing DAPO's name, and nothing in the log would say so.
        checks = [
            ("all-correct group", torch.tensor([1.0, 1.0, 1.0, 1.0]), False),
            ("all-wrong group", torch.tensor([0.0, 0.0, 0.0, 0.0]), False),
            ("mixed group", torch.tensor([0.0, 1.0, 0.0, 1.0]), True),
        ]
        for label, r, want in checks:
            got = fn({"rewards": r})
            if got != want:
                print(f"FAIL: {label} -> accept={got}, expected {want}"); ok = False
        print(f"DAPO filter decisions correct on {len(checks)} shapes")
    if dapo.dynamic_bs:
        print("FAIL: dynamic_bs is true for the DAPO arm. It reads as the oversampling "
              "switch and is the OPPOSITE: true stops after batch_size ATTEMPTS and returns "
              "a shrunken batch. DAPO needs it FALSE so collection continues until "
              "batch_size are ACCEPTED."); ok = False

    # The sign guard must survive the CLI path too, or a typo becomes a silent bad run.
    try:
        load(BASE + ["+actor.group_routing.enabled=true",
                     "+actor.group_routing.unsolved_advantage=0.5"])
        print("FAIL: a positive unsolved_advantage was accepted through the CLI"); ok = False
    except Exception as exc:
        print(f"guard held through the CLI: {type(exc).__name__}: {str(exc)[:90]}")

    # Every registered router that a run could name, checked against what the seam applies.
    for name in ("solve_rate", "coharness", "static", "cluster", "random", "contextual"):
        ok = _check_router_modes(name) and ok

    print("\nPREFLIGHT PASS" if ok else "\nPREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
