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
# The mixture arm needs a router as well as the flag: the fixed solved/unsolved rule emits
# no weights, so decision=mixture without one is refused rather than silently ignored.
MIXTURE = ["+actor.group_routing.router=solve_rate",
           "+actor.group_routing.decision=mixture"]
DAPO = ["+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling"]


def load(argv: list[str]):
    """Resolve a config exactly as the trainer does, minus the side effects."""
    cfg, _ = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=GRPOConfig)
    return OmegaConf.to_object(cfg)




def _check_router_modes(name: str) -> bool:
    """Route synthetic units through a router and confirm the seam can apply every mode.

Under ``decision=argmax`` ``_route_groups`` passes ``decision.argmax()`` to
    ``apply_decisions``; under ``decision=mixture`` it passes ``decision.normalised()`` to
    ``apply_mixtures``. Both implement RL/SFT/SKIP and REFUSE a teacher-requiring mode --
    deliberately, because treating DISTILL as SKIP would report a distillation arm that
    never ran. That refusal is a ValueError raised inside ``_compute_advantages``, i.e. it
    kills the run. A router that can emit such a mode should therefore be rejected here,
    before any GPU is allocated, rather than at whatever batch first happens to trigger it.

    BOTH paths are probed, because they are not the same test. A decision like
    ``{rl: 0.6, distill: 0.4}`` has an applicable ARGMAX and an inapplicable COMPONENT, so a
    router that is safe for the argmax arm can still kill a mixture run on its first batch.

    Args:
        name: A key in ``selfevo.compose.ROUTERS``.

    Returns:
        True if every mode AND every whole weight mapping the router produced over the probe
        grid is applicable.
    """
    from selfevo.compose import ROUTERS
    from selfevo.integration.group_apply import apply_decisions, apply_mixtures
    from selfevo.observability import FEATURE_NAMES
    from selfevo.routing.base import RoutingContext
    import torch

    factory = ROUTERS.get(name)
    if factory is None:
        print(f"FAIL: router {name!r} is not registered"); return False
    router = factory()

    seen, mixes, ok = set(), [], True
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        for teacher in (False, True):
            extra = {k: 0.5 for k in FEATURE_NAMES}
            extra["solve_rate"] = rate
            ctx = RoutingContext(
                solve_rate=rate, group_size=8, has_teacher=teacher,
                unit_id=f"probe-{rate}-{teacher}", extra=extra,
            )
            # ONE call per probe: a learned router carries state, and asking twice to read
            # the label and then the weights would probe a router that has already moved.
            decision = router.route(ctx)
            seen.add(decision.argmax())
            mixes.append((ctx.unit_id, decision.normalised()))
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
    for unit_id, mix in mixes:
        try:
            apply_mixtures(
                torch.zeros(2, 3), torch.ones(2, 3), [2], [mix], sft_weight=0.5
            )
        except ValueError as exc:
            print(f"FAIL: router {name!r} emitted mixture {mix} at {unit_id}, which the "
                  f"seam cannot apply under decision=mixture -- this would raise inside "
                  f"_compute_advantages and kill the run: {str(exc)[:90]}")
            ok = False
            break
    if ok:
        n_mixed = sum(1 for _, m in mixes if max(m.values()) < 1.0)
        print(f"router {name!r} emits {sorted(seen)}; all applicable "
              f"({n_mixed}/{len(mixes)} probes were genuine mixtures)")
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

    # --- the mixture arm --------------------------------------------------------------
    #
    # Same question as the ON arm, for the axis that decides whether the router's WEIGHTS
    # reach the tensor: an override that parses but lands on an unread key would run the
    # argmax path and report itself as a mixture arm.
    mixture = load(BASE + ON + MIXTURE)
    mgr = mixture.actor.group_routing
    print(f"ARM mixture -> decision = {getattr(mgr, 'decision', None)!r}, "
          f"router = {getattr(mgr, 'router', None)!r}")
    if getattr(mgr, "decision", None) != "mixture":
        print("FAIL: decision=mixture did not travel"); ok = False
    if getattr(gr, "decision", None) != "argmax":
        print(f"FAIL: the routed arm's default decision is "
              f"{getattr(gr, 'decision', None)!r}, not 'argmax'; the ablation pair would "
              f"differ on more than one axis"); ok = False
    try:
        load(BASE + ON + ["+actor.group_routing.decision=mixture"])
        print("FAIL: decision=mixture with no router was accepted through the CLI; that arm "
              "would run the hardcoded solved/unsolved constants and report as a mixture")
        ok = False
    except Exception as exc:
        print(f"mixture guard held through the CLI: {type(exc).__name__}: {str(exc)[:90]}")
    try:
        load(BASE + ON + ["+actor.group_routing.router=solve_rate",
                          "+actor.group_routing.decision=mixtures"])
        print("FAIL: an unknown decision value was accepted through the CLI"); ok = False
    except Exception as exc:
        print(f"decision guard held through the CLI: {type(exc).__name__}: {str(exc)[:90]}")

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
