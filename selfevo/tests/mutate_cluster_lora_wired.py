#!/usr/bin/env python3
"""Mutation-test the two cluster-LoRA call sites against a COPY of the repo.

A copy, never the live checkout: the 8xA100 job imports this tree through a ``.pth`` in the
venv and relaunches its workers on exit, so a mutated ``actor.py`` or ``fsdp_engine.py``
sitting on disk for even a few seconds could be imported by a real run. Every target's
sha256 is taken before the first mutation, re-checked after every restore, and checked again
at the end.

Every mutation below is a single-line defect a careless edit to one of the two seams could
produce, and each one is aimed at a GUARD rather than at arithmetic: the failure mode this
work has to avoid is not a wrong number, it is a run that trains, logs and reports as the
method while being the baseline. A mutation whose anchor is not unique is reported as a SKIP
and counted as NOT killed, because an anchor that matched twice mutated something other than
what it names.

Usage: mutate_cluster_lora_wired.py <path to a copy of the repo>
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
ACTOR = REPO / "areal/trainer/ppo/actor.py"
ENGINE = REPO / "areal/engine/fsdp_engine.py"
WIRING = REPO / "selfevo/cluster_lora/wiring.py"
TARGETS = {"actor": ACTOR, "engine": ENGINE, "wiring": WIRING}
TESTS = [
    "selfevo/tests/test_cluster_lora_wired.py",
    "selfevo/tests/test_cluster_lora_engine.py",
]

# (target, label, find, replace)
MUTATIONS = [
    # --------------------------------------------------------------- the actor gate ------
    ("actor", "the seam is entered whatever the configuration says",
     'if os.environ.get("SELFEVO_CLUSTER_LORA", "").strip():',
     "if True:"),
    ("actor", "the gate reads the FEATURES variable, so the arm never switches on",
     'if os.environ.get("SELFEVO_CLUSTER_LORA", "").strip():',
     'if os.environ.get("SELFEVO_CLUSTER_LORA_FEATURES", "").strip():'),
    ("actor", "the reach report is formed and then dropped instead of emitted",
     "            stats_tracker.scalar(\n"
     "                **begin_cluster_batch(self, router, data, contexts, list(sizes))\n"
     "            )",
     "            begin_cluster_batch(self, router, data, contexts, list(sizes))"),
    ("actor", "the group sizes are not passed, so features pair with the wrong groups",
     "**begin_cluster_batch(self, router, data, contexts, list(sizes))",
     "**begin_cluster_batch(self, router, data, contexts, [1] * len(contexts))"),

    # -------------------------------------------------------------- the engine dispatch --
    ("engine", "the cluster branch is taken exactly when it should not be",
     "        if cluster_plan is not None:",
     "        if cluster_plan is None:"),
    ("engine", "the cluster branch never runs, so the arm is the baseline",
     "        if cluster_plan is not None:",
     "        if False:"),
    ("engine", "an armed plan with no experts trains the active adapter silently",
     '        if adapters is None:\n            raise ClusterWiringError(',
     '        if False:\n            raise ClusterWiringError('),
    ("engine", "a batch with no group identity is routed by position instead of refused",
     '        if "group_ids" not in input_batched:\n            raise ClusterWiringError(',
     '        if False:\n            raise ClusterWiringError('),
    ("engine", "rows are routed by batch position, which microbatching reorders",
     '            input_batched["group_ids"][:, 0].tolist(), cluster_plan.key_of_group',
     "            list(range(input_batched[\"attention_mask\"].shape[0])), cluster_plan.key_of_group"),
    ("engine", "the denominator becomes one cluster's tokens rather than the batch's",
     "            self._prepare_mb_list(input_batched), loss_weight_fn, self.dp_group",
     "            self._prepare_mb_list(select_rows(input_batched, next(iter(rows.values())))), loss_weight_fn, self.dp_group"),
    ("engine", "the last microbatch is backwarded twice",
     "                if remaining == 0:\n                    held.append(loss)\n                    return None\n                return loss",
     "                if remaining == 0:\n                    held.append(loss)\n                    return loss\n                return loss"),
    ("engine", "the microbatch countdown never reaches the last one",
     "                remaining -= 1",
     "                remaining -= 0"),
    ("engine", "every microbatch but the first is dropped from the backward",
     "                if remaining == 0:",
     "                if remaining >= 0:"),
    ("engine", "the per-cluster record is not reported, so the arm logs nothing about itself",
     "        stats.update(record.as_metrics())",
     "        stats.update({})"),
    ("engine", "the expert roster is built whatever the configuration says",
     '        if os.environ.get("SELFEVO_CLUSTER_LORA_ADAPTERS", "").strip():',
     "        if True:"),
    ("engine", "the roster is never built, so the experts do not exist",
     '        if os.environ.get("SELFEVO_CLUSTER_LORA_ADAPTERS", "").strip():',
     "        if False:"),
    ("engine", "the wrapped model is discarded, leaving the base model unwrapped",
     "            self.model = self._selfevo_adapters.model",
     "            self.model = self.model"),

    # -------------------------------------------------------------------- the refusals ---
    ("wiring", "meds without features silently becomes the none arm",
     '    elif cfg.partition != "none":',
     "    elif False:"),
    ("wiring", "the features refusal is inverted, so only the baseline refuses",
     '    elif cfg.partition != "none":',
     '    elif cfg.partition == "none":'),
    ("wiring", "the extra forward is on by default",
     '            features=(env.get("SELFEVO_CLUSTER_LORA_FEATURES") or "0").strip().lower()',
     '            features=(env.get("SELFEVO_CLUSTER_LORA_FEATURES") or "1").strip().lower()'),
    ("wiring", "an unknown partition name is accepted and falls through",
     "        if self.partition not in TRAINING_PARTITIONS:",
     "        if False:"),
    ("wiring", "a non-numeric sweep value silently restores the shipped default",
     "                raise ValueError(f\"{name}={raw!r} is not an integer\") from exc",
     "                return default"),
    ("wiring", "a repeated adapter name is accepted as a two-expert roster",
     "    if len(set(names)) != len(names):",
     "    if False:"),
    ("wiring", "a partition outside the roster is accepted",
     "        missing = [n for n in named if n not in roster]",
     "        missing = []"),
    ("wiring", "the roster is only checked when it is empty",
     "    if roster:\n        missing =",
     "    if not roster:\n        missing ="),
    ("wiring", "a router that cannot carry a partition is accepted",
     "    if not isinstance(router, ClusterRouter):",
     "    if False:"),
    ("wiring", "a row whose group the plan does not name gets the shared expert",
     '        if key is None:\n            raise ClusterWiringError(',
     '        if False:\n            raise ClusterWiringError('),
    ("wiring", "gradients may be zeroed into tensors, so idle experts decay",
     "        if not set_to_none:",
     "        if False:"),
    ("wiring", "features asked for with no model are silently skipped",
     "        if model is None:\n            raise ClusterWiringError(",
     "        if False:\n            raise ClusterWiringError("),

    # ------------------------------------------------------------------- the mechanism ---
    ("wiring", "every cluster is routed to SKIP, zeroing the batch it partitioned",
     "    router.policy = {key: TrainingMode.RL for key in sorted(set(partition.keys))}",
     "    router.policy = {key: TrainingMode.SKIP for key in sorted(set(partition.keys))}"),
    ("wiring", "the policy is left alone, so every cluster takes the SKIP fallback",
     "    router.policy = {key: TrainingMode.RL for key in sorted(set(partition.keys))}",
     "    router.policy = dict(router.policy)"),
    ("wiring", "a fresh key function per batch, so every group is relabelled every step",
     '    keyfn = getattr(actor, "_selfevo_cluster_keyfn", None)',
     "    keyfn = None"),
    ("wiring", "the key function is never attached, so the router partitions by silence",
     "    if router.key_fn is not keyfn:\n        router.key_fn = keyfn",
     "    if False:\n        router.key_fn = keyfn"),
    ("wiring", "the engine is never armed, so train_batch keeps one adapter for the batch",
     "    if engine is not None:\n        engine._selfevo_cluster_plan = ClusterPlan(",
     "    if False:\n        engine._selfevo_cluster_plan = ClusterPlan("),
    ("wiring", "churn is keyed on batch position, which is reshuffled every step",
     "        unit_ids, features, group_ids=_prompt_ids(data, sizes, unit_ids)",
     "        unit_ids, features, group_ids=tuple(unit_ids)"),
    ("wiring", "a group's vector is one rollout's rather than the group's mean",
     "        out.append(np.mean(np.stack(per_row[start : start + size], 0), axis=0))",
     "        out.append(per_row[start])"),
    ("wiring", "the behavioural forward runs with dropout active",
     "    model.eval()",
     "    pass"),
    ("wiring", "the model is left in eval mode after the behavioural forward",
     "        if was_training:\n            model.train()",
     "        if False:\n            model.train()"),
    ("wiring", "fallbacks are not counted, so a batch read at the wrong position looks clean",
     "                fallbacks += 1",
     "                fallbacks += 0"),
    ("wiring", "group sizes that do not partition the batch are accepted",
     "    if sum(sizes) != n_rows:",
     "    if False:"),
    ("wiring", "row selection returns the whole batch for every cluster",
     "        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n:",
     "        if False:"),
    ("wiring", "an empty plan is accepted and routes nothing",
     "        if not self.key_of_group:",
     "        if False:"),
    ("wiring", "the feature cost is reported as zero whatever it was",
     '    metrics["cluster_lora/feature_seconds"] = float(seconds)',
     '    metrics["cluster_lora/feature_seconds"] = 0.0'),
    # ------------------------------------------- the two seams that never checked each other --
    #
    # The arm is switched on by two environment variables read at two seams. Out of step, the
    # run created every expert, armed no partition, trained one expert on the whole batch and
    # emitted no cluster_lora key at all -- the baseline reporting itself as the method. Each
    # mutation here restores one half of that.
    ("actor", "the unrouted seam never runs, so the arm is the baseline again",
     '                cluster_arm = os.environ.get("SELFEVO_CLUSTER_LORA", "").strip()',
     '                cluster_arm = ""'),
    ("actor", "the unrouted seam runs even when routing IS carrying the arm",
     '                if cluster_arm and not (routing_on and getattr(gr, "router", None)):',
     "                if cluster_arm:"),
    ("actor", "an arm needing features is armed as all-shared instead of refused",
     '        if cfg.partition != "none":', "        if False:"),
    ("actor", "a roster with no shared adapter is accepted for the none arm",
     "        if roster and SHARED_CLUSTER not in roster:", "        if False:"),
    ("actor", "the unrouted plan step never advances, so every batch reports step 0",
     '        step = int(getattr(self, "_selfevo_batch", 0))\n        self._selfevo_batch = step + 1',
     '        step = int(getattr(self, "_selfevo_batch", 0))\n        self._selfevo_batch = step'),
    ("actor", "the unrouted plan is formed and then not handed to the engine",
     "        if engine is not None:\n            engine._selfevo_cluster_plan = ClusterPlan(",
     "        if False:\n            engine._selfevo_cluster_plan = ClusterPlan("),
    ("actor", "the unrouted plan names one group, so the rest of the batch is unrouted",
     "        partition = no_partition(n_groups)", "        partition = no_partition(1)"),
    ("actor", "the unrouted arm always reports itself as armed, engine or no engine",
     '            "cluster_lora/plan_armed": float(engine is not None),',
     '            "cluster_lora/plan_armed": 1.0,'),
    ("actor", "the routed half of the gate is dropped, so a routed arm is armed twice",
     '                if cluster_arm and not (routing_on and getattr(gr, "router", None)):',
     "                if cluster_arm and not routing_on:"),
    ("actor", "the unrouted arm does not stamp the batch it armed",
     '        tag_cluster_batch(data, step)\n        engine = getattr(self, "engine", None)',
     '        engine = getattr(self, "engine", None)'),
    ("engine", "a roster of experts with no plan armed runs the unrouted step again",
     '        adapters = getattr(self, "_selfevo_adapters", None)\n        if adapters is not None:',
     '        adapters = getattr(self, "_selfevo_adapters", None)\n        if False:'),
    ("engine", "that refusal is inverted, so the rollback path is the one that dies",
     '        adapters = getattr(self, "_selfevo_adapters", None)\n        if adapters is not None:',
     '        adapters = getattr(self, "_selfevo_adapters", None)\n        if adapters is None:'),
    # ------------------------------------- the plan describes ONE batch, and which one -----
    ("engine", "the plan is never checked against the batch it was armed for",
     "        assert_plan_describes_batch(cluster_plan, input_batched)", "        pass"),
    ("wiring", "a batch with no identity is routed by whatever plan is lying there",
     "    if tag is None:", "    if False and tag is None:"),
    ("wiring", "the batch identity is read and compared to nothing",
     "    if seen != [int(plan.step)]:", "    if False:"),
    ("wiring", "a batch carrying two identities passes as one of them",
     "    seen = sorted({int(v) for v in tag.reshape(int(tag.shape[0]), -1)[:, 0].tolist()})",
     "    seen = [int(tag.reshape(int(tag.shape[0]), -1)[0, 0])]"),
    ("wiring", "the batch is never stamped, so nothing armed can be identified",
     '    tag_cluster_batch(data, step)\n    engine = getattr(actor, "engine", None)',
     '    engine = getattr(actor, "engine", None)'),
    ("wiring", "the stamp is a constant, so every batch looks like batch zero",
     '    data[CLUSTER_BATCH_KEY] = torch.full_like(\n        data["loss_mask"], int(batch_id), dtype=torch.long\n    )',
     '    data[CLUSTER_BATCH_KEY] = torch.full_like(\n        data["loss_mask"], 0, dtype=torch.long\n    )'),
    ("wiring", "the partitioner is built unbounded, so the ids outgrow the roster again",
     "                max_experts=bound,", "                max_experts=None,"),
    ("wiring", "the none arm is bounded too, so a roster it can use is refused",
     '                if cfg.partition == "none"', "                if False"),
]


def run_tests() -> bool:
    """True if both seam test files pass inside the copy."""
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    env.pop("SELFEVO_CLUSTER_LORA", None)
    env.pop("SELFEVO_CLUSTER_LORA_ADAPTERS", None)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-c",
         "import areal.trainer.ppo.actor as a, areal.engine.fsdp_engine as e, "
         "selfevo.cluster_lora.wiring as w; print(a.__file__); print(e.__file__); "
         "print(w.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=600,
    )
    # The AReaL logger writes a banner to stdout at import, so the paths are picked out
    # rather than read positionally -- an isolation check that mis-parses is worse than none.
    got = [pathlib.Path(x).resolve() for x in r.stdout.split()
           if x.endswith(".py") and "/" in x]
    want = [ACTOR, ENGINE, WIRING]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {want}")
    print(f"isolated: imports resolve inside {REPO}")


def main() -> int:
    _assert_isolated()
    original = {k: v.read_text() for k, v in TARGETS.items()}
    digests = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in original.items()}

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    for target, label, find, repl in MUTATIONS:
        src = original[target]
        name = f"[{target}] {label}"
        if src.count(find) != 1:
            print(f"SKIP  {name}: anchor appears {src.count(find)}x")
            survivors.append((name, "anchor not unique"))
            continue
        mutated = src.replace(find, repl, 1)
        if mutated == src:
            print(f"SKIP  {name}: replacement is a no-op")
            survivors.append((name, "replacement changed nothing"))
            continue
        TARGETS[target].write_text(mutated)
        # A mutation that does not compile is a typo in this file, not a defect the tests
        # should be credited with catching.
        syntax = subprocess.run(
            [sys.executable, "-m", "py_compile", str(TARGETS[target])],
            capture_output=True, text=True, timeout=300,
        )
        if syntax.returncode != 0:
            TARGETS[target].write_text(src)
            print(f"SKIP  {name}: mutant does not compile")
            survivors.append((name, "mutant does not compile"))
            continue
        passed = run_tests()
        TARGETS[target].write_text(src)
        assert hashlib.sha256(TARGETS[target].read_text().encode()).hexdigest() == \
            digests[target], f"restore failed for {target}"
        if passed:
            print(f"SURVIVED  {name}")
            survivors.append((name, "tests still passed"))
        else:
            print(f"killed    {name}")

    for k, v in TARGETS.items():
        assert hashlib.sha256(v.read_text().encode()).hexdigest() == digests[k], k
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed; "
          "all targets restored and re-hashed")
    if survivors:
        print("\nNOT KILLED (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
