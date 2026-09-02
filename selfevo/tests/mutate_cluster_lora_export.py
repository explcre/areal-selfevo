#!/usr/bin/env python3
"""Mutation-test the expert-merge exit points against a COPY of the repo.

A copy, never the live checkout: the 8xA100 job imports this tree through a ``.pth`` in the
venv, so a mutated ``fsdp_engine.py`` sitting on disk for even a few seconds could be picked
up by a worker that restarts. Every target's sha256 is taken before the first mutation,
re-checked after every restore, and checked again at the end.

Every mutation is a single-line defect a careless edit to the save/sync path could produce,
and every one is aimed at the property the exit points exist to hold: that what leaves the
engine is the SUM of the experts, at the summed rank, and that a selection which cannot be
that refuses instead of shipping. The failure being hunted is not a crash -- it is an
artifact that loads, serves and evaluates to a plausible number describing a model no arm
produced.

REPORTING. A mutation whose anchor is not unique, whose replacement changes nothing, or whose
mutant does not compile is reported SKIP and is NOT counted as a survivor: it was never
exercised, so the tests were never given the chance to catch it. A SKIP is a defect in this
file, not evidence about the tests, and ``test_every_export_mutation_anchor_occurs_once``
turns a decayed anchor into a failing test rather than a line someone has to read.

Usage: mutate_cluster_lora_export.py <path to a copy of the repo>
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
ENGINE_REL = "areal/engine/fsdp_engine.py"
MERGE_REL = "selfevo/cluster_lora/merge.py"
#: The shared engine harness. Mutated too, because the invariant that no module leaves a
#: device mesh behind a destroyed process group lives there and is worth exactly as much as
#: the tests that hold it -- and its failure mode is a red suite in a different file, which
#: is the kind nobody attributes to the file that caused it.
HARNESS_REL = "selfevo/tests/test_cluster_lora_engine.py"
#: The call sites that decide which experts a forward sees. The export defect ships a wrong
#: artifact; this one changes the gradient, because compute_logp is the importance-ratio
#: denominator for every group in the batch.
WIRING_REL = "selfevo/cluster_lora/wiring.py"
TESTS = [
    "selfevo/tests/test_cluster_lora_export.py",
    "selfevo/tests/test_cluster_lora_merge.py",
    "selfevo/tests/test_cluster_lora_engine.py",
    "selfevo/tests/test_cluster_lora_wired.py",
]

# (target, label, find, replace)
MUTATIONS = [
    # ------------------------------------------------------- the defect itself, restored --
    ("engine", "no export ever sees a roster, so every exit point ships one expert",
     "        return () if adapters is None else tuple(adapters.names)",
     "        return ()"),
    ("engine", "the checkpoint takes the unmerged branch",
     "        experts = self._lora_export_experts()\n        adapter_state = {}\n        if experts:",
     "        experts = self._lora_export_experts()\n        adapter_state = {}\n        if False:"),
    ("engine", "the weight sync takes the unmerged branch",
     "            experts = self._lora_export_experts()\n            if experts:",
     "            experts = self._lora_export_experts()\n            if False:"),
    ("engine", "the advertised shapes are one expert's",
     "        experts = self._lora_export_experts()\n        if experts:\n            # EXIT POINT: these shapes",
     "        experts = self._lora_export_experts()\n        if False:\n            # EXIT POINT: these shapes"),

    # ------------------------------------------------------------- the metadata a server --
    ("engine", "the server is told one expert's rank for a merged adapter",
     '                "r": merged.r if merged is not None else self.config.lora_rank,',
     '                "r": self.config.lora_rank,'),
    ("engine", "the server is told one expert's alpha, so alpha/r rescales the delta",
     "                    merged.lora_alpha if merged is not None else self.config.lora_alpha",
     "                    self.config.lora_alpha"),
    ("engine", "the saved adapter_config describes one expert",
     "        return merged_lora_config(self.model.peft_config, experts)",
     "        return self.model.peft_config[experts[0]]"),

    # -------------------------------------------------------------------- the refusals ----
    ("engine", "a multi-adapter model exports its active expert without complaint",
     "        if len(configs) > 1:",
     "        if False:"),
    ("engine", "the guard only fires on a model with no adapters at all",
     "        if len(configs) > 1:",
     "        if len(configs) > 99:"),
    ("engine", "a trainable non-LoRA parameter is dropped from the export in silence",
     "                if param.requires_grad:\n                    raise MergeSelectionError(",
     "                if False:\n                    raise MergeSelectionError("),

    # ---------------------------------------------------------------- the merge itself ----
    ("merge", "the A blocks are not scaled, so the deployed delta is r/alpha times wrong",
     "                [t * (weight_of[n] * float(scalings[n])) for n, t in zip(names, ordered)],",
     "                [t for n, t in zip(names, ordered)],"),
    ("merge", "the A and B branches are swapped, each concatenated on the other's axis",
     "        if layer_name.endswith(\"_A\"):",
     "        if layer_name.endswith(\"_B\"):"),
    ("merge", "the A blocks are concatenated in the reverse order to the B blocks, so each "
              "expert's B multiplies another expert's A",
     "                [t * (weight_of[n] * float(scalings[n])) for n, t in zip(names, ordered)],",
     "                [t * (weight_of[n] * float(scalings[n])) for n, t in zip(names, ordered)][::-1],"),
    ("merge", "the merge is the MEAN, which every training metric would report identically",
     "    w = [1.0] * len(names) if weights is None else [float(x) for x in weights]\n"
     "    if len(w) != len(names):\n"
     "        raise ValueError(f\"{len(w)} weights for {len(names)} adapters\")\n"
     "    weight_of = dict(zip(names, w))",
     "    w = [1.0 / len(names)] * len(names) if weights is None else [float(x) for x in weights]\n"
     "    if len(w) != len(names):\n"
     "        raise ValueError(f\"{len(w)} weights for {len(names)} adapters\")\n"
     "    weight_of = dict(zip(names, w))"),
    ("merge", "only the first expert reaches the merged adapter",
     "        ordered = [per_adapter[n] for n in names]",
     "        ordered = [per_adapter[names[0]] for _n in names]"),
    ("merge", "the documented operator and the code disagree",
     'MERGE_OPERATOR = "sum"',
     'MERGE_OPERATOR = "mean"'),

    # ------------------------------------------------------------------ the merged rank ---
    ("merge", "the merged config keeps one expert's rank",
     "        r=int(sum(ranks)),",
     "        r=int(ranks[0]),"),
    ("merge", "the merged config keeps the original alpha, so its own scaling is not 1.0",
     "        lora_alpha=int(sum(ranks)),",
     "        lora_alpha=int(peft_configs[names[0]].lora_alpha),"),
    ("merge", "the advertised shapes do not sum the rank axis",
     "        merged[axis] = int(sum(s[axis] for s in shapes))",
     "        merged[axis] = int(shapes[0][axis])"),

    # ------------------------------------------------------------- the selection guards ---
    ("merge", "a ragged module set merges whatever experts happen to be present",
     "        missing = [n for n in names if n not in per_adapter]\n        if missing:",
     "        missing = [n for n in names if n not in per_adapter]\n        if False:"),
    ("merge", "a tensor from an adapter outside the roster is folded in",
     "        if adapter not in wanted:",
     "        if False:"),
    ("merge", "a DoRA magnitude vector is skipped and left zeroed, as PEFT's own cat does",
     '        if layer_name == "lora_magnitude_vector":',
     "        if False:"),
    ("merge", "an empty selection writes a valid adapter that changes nothing",
     "    if not blocks:",
     "    if False:"),
    ("merge", "an unparseable parameter name is filed under a guessed key",
     '    raise MergeSelectionError(\n        f"{name!r} carries no adapter layer',
     '    return (".".join(parts[:-1]), "lora_A", "default", "weight")\n'
     '    raise MergeSelectionError(\n        f"{name!r} carries no adapter layer'),
    ("merge", "an expert whose scaling differs between modules is merged anyway",
     "            if name in out and out[name] != value:",
     "            if False:"),
    ("merge", "an expert on no module at all contributes nothing and is not noticed",
     "    missing = [n for n in names if n not in out]",
     "    missing = []"),
    ("merge", "shapes that disagree off the rank axis are concatenated regardless",
     "        if len(others) != 1:",
     "        if False:"),

    # ------------------------------------------------------------- the mesh invariant ----
    ("harness", "the mesh cache is not cleared, so it outlives the group it was built on",
     "    global _MESH, _MESH_GROUP\n    _MESH = None\n    _MESH_GROUP = None",
     "    global _MESH, _MESH_GROUP\n    return"),
    ("harness", "destroying the group no longer clears the mesh built on it",
     "    reset_mesh_cache()\n    dist.destroy_process_group()",
     "    dist.destroy_process_group()"),
    ("harness", "the cache is memoised rather than keyed on the live process group",
     "    if _MESH is None or _MESH_GROUP is not group:",
     "    if _MESH is None:"),

    # ------------------------------------------- forwards taken outside step() -----------
    ("engine", "a read-only forward runs on whichever expert only() last restored",
     "        if not forward_only:\n            return nullcontext()",
     "        if True:\n            return nullcontext()"),
    ("engine", "the seam reaches the TRAINING forward too, so every cluster trains them all",
     "        if not forward_only:\n            return nullcontext()",
     "        if False:\n            return nullcontext()"),
    ("engine", "an armed engine still takes its read-only forwards on one expert",
     '        adapters = getattr(self, "_selfevo_adapters", None)\n'
     "        if adapters is None:\n"
     "            return nullcontext()\n"
     "        from selfevo.cluster_lora.wiring import every_expert",
     "        adapters = None\n"
     "        if adapters is None:\n"
     "            return nullcontext()\n"
     "        from selfevo.cluster_lora.wiring import every_expert"),

    ("wiring", "the behavioural forward runs on one expert, so the next partition clusters it",
     "    activation = every_expert(model, experts) if experts else nullcontext()",
     "    activation = nullcontext()"),
    ("wiring", "the roster never reaches the behavioural forward",
     "            experts=() if adapters is None else tuple(adapters.names),",
     "            experts=(),"),
    ("wiring", "an activation that does not take is accepted, leaving one expert active",
     "        if active != set(names):",
     "        if False:"),
    ("wiring", "the multi-expert block is not read-only, so a backward reaches every expert",
     "        with torch.no_grad():\n            yield model",
     "        with nullcontext():\n            yield model"),
    ("wiring", "the previous activation is not restored, so the next step routes everywhere",
     "        if previous:\n"
     "            tuner.set_adapter(previous[0] if len(previous) == 1 else list(previous))",
     "        if False:\n"
     "            tuner.set_adapter(previous[0] if len(previous) == 1 else list(previous))"),
    ("wiring", "an expert the model does not carry is activated silently",
     "    missing = [n for n in names if n not in present]\n    if missing:",
     "    missing = [n for n in names if n not in present]\n    if False:"),
]


def targets():
    """``{name: path}`` for the files this harness mutates."""
    return {
        "engine": REPO / ENGINE_REL,
        "merge": REPO / MERGE_REL,
        "harness": REPO / HARNESS_REL,
        "wiring": REPO / WIRING_REL,
    }


def run_tests() -> bool:
    """True if every test file passes inside the copy."""
    env = dict(
        os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""
    )
    env.pop("SELFEVO_CLUSTER_LORA", None)
    env.pop("SELFEVO_CLUSTER_LORA_ADAPTERS", None)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider", "-p", "no:randomly"],
        cwd=REPO, capture_output=True, text=True, timeout=3600, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-c",
         "import areal.engine.fsdp_engine as e, selfevo.cluster_lora.merge as m, "
         "selfevo.tests.test_cluster_lora_engine as h, "
         "selfevo.cluster_lora.wiring as w; "
         "print(e.__file__); print(m.__file__); print(h.__file__); print(w.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=600,
    )
    # The AReaL logger writes a banner at import, so the paths are picked out rather than
    # read positionally -- an isolation check that mis-parses is worse than none.
    got = [pathlib.Path(x).resolve() for x in r.stdout.split()
           if x.endswith(".py") and "/" in x]
    want = [REPO / ENGINE_REL, REPO / MERGE_REL, REPO / HARNESS_REL, REPO / WIRING_REL]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {want}")
    print(f"isolated: imports resolve inside {REPO}")


def main() -> int:
    _assert_isolated()
    files = targets()
    original = {k: v.read_text() for k, v in files.items()}
    digests = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in original.items()}

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors, skips, killed = [], [], 0
    for target, label, find, repl in MUTATIONS:
        src = original[target]
        name = f"[{target}] {label}"
        if src.count(find) != 1:
            print(f"SKIP      {name}: anchor appears {src.count(find)}x")
            skips.append((name, "anchor not unique"))
            continue
        mutated = src.replace(find, repl, 1)
        if mutated == src:
            print(f"SKIP      {name}: replacement is a no-op")
            skips.append((name, "replacement changed nothing"))
            continue
        files[target].write_text(mutated)
        syntax = subprocess.run(
            [sys.executable, "-m", "py_compile", str(files[target])],
            capture_output=True, text=True, timeout=300,
        )
        if syntax.returncode != 0:
            files[target].write_text(src)
            print(f"SKIP      {name}: mutant does not compile")
            skips.append((name, "mutant does not compile"))
            continue
        passed = run_tests()
        files[target].write_text(src)
        assert (
            hashlib.sha256(files[target].read_text().encode()).hexdigest()
            == digests[target]
        ), f"restore failed for {target}"
        if passed:
            print(f"SURVIVED  {name}")
            survivors.append((name, "tests still passed"))
        else:
            killed += 1
            print(f"killed    {name}")

    for key, path in files.items():
        assert (
            hashlib.sha256(path.read_text().encode()).hexdigest() == digests[key]
        ), key
    print(
        f"\n{killed} killed, {len(skips)} skipped, {len(survivors)} survived "
        f"of {len(MUTATIONS)}"
    )
    for name, why in skips:
        print(f"  SKIP     {name}: {why}")
    for name, why in survivors:
        print(f"  SURVIVED {name}: {why}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
