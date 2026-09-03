"""EVERY EXIT POINT: what leaves a cluster-routed engine must be the SUM of the experts.

Training routes each behavioural cluster's loss into its own LoRA expert. Inference has no
such label -- the cluster is read off the response and at inference time there is no response
yet -- so the deployed model is the experts SUMMED into one adapter (LSPO-style;
``selfevo/cluster_lora/merge.py`` documents why the sum and not a mean or a size weighting).
Between those two facts sits every exit point of the engine: the HF checkpoint an eval reads,
the weight sync that feeds the rollout engine mid-training, and the adapter metadata a server
sizes its LoRA slots from. Each of them selected LoRA tensors by ``requires_grad``.

``requires_grad`` is the wrong selector here, and mechanically so: ``ClusterAdapterSet.only()``
activates one expert at a time and PEFT's ``set_adapter`` freezes the rest, so after a routed
step exactly ONE expert is trainable. Every exit point therefore carried one arbitrary expert.
That artifact loads, serves, and evaluates to a number that looks real and describes a model
no arm produced, and nothing else in the run says so -- which is why the defect is asserted
here as a LIVE COUNTERFACTUAL against the committed code rather than described in prose.

The engine harness is ``test_cluster_lora_engine``'s, imported rather than re-derived: a
second copy of an engine builder is a second thing to drift, and the version that trains the
experts is the version whose exports have to be right.

FLOAT32 THROUGHOUT, deliberately. ``_cast_to_compute_dtype`` casts an export to
``config.dtype``, so under the shipped bfloat16 a reloaded checkpoint could only be compared
against the in-memory model to within bf16. The round trip below is asserted exactly, which
needs the export dtype to be the storage dtype; the cast is a shipped behaviour that applies
equally to a merged and an unmerged export and is not what these tests are about.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import Future

import numpy as np

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import torch.distributed as dist  # noqa: E402

import areal.engine.fsdp_engine as fe  # noqa: E402
from areal.api.io_struct import WeightUpdateMeta  # noqa: E402
from selfevo.cluster_lora.wiring import ClusterWiringError, every_expert  # noqa: E402
from selfevo.cluster_lora.merge import (  # noqa: E402
    MERGE_OPERATOR,
    MergeSelectionError,
    adapter_delta,
    expert_scalings,
    merge_expert_tensors,
    merge_sum,
    merged_lora_config,
    split_lora_param_name,
)
from selfevo.tests.test_cluster_lora_engine import (  # noqa: E402
    REPO,
    linear_loss,
    make_batch,
    make_engine,
    plan_of,
    run,
    start_world,
    stop_world,
    tiny_hf_config,
    weight_fn,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """This module's own process group, created and destroyed here.

    Its own, deliberately. Importing another module's ``world`` fixture made this file the
    only one in ``selfevo/tests`` that created a process group with no
    ``destroy_process_group`` anywhere in it -- a cross-module dependency that is invisible
    until the ordering changes. ``start_world`` and ``stop_world`` are the engine harness's,
    because one lifecycle implementation is what keeps the mesh invariant in one place; what
    is not shared is the OWNERSHIP of the group.
    """
    store = tmp_path_factory.mktemp("pg") / "store"
    if not start_world(store):
        yield
        return
    try:
        yield
    finally:
        stop_world()

@pytest.fixture(autouse=True)
def _clear_stats_tracker():
    """Drop the process-global stats between tests, as the engine harness's own file does."""
    from areal.utils.stats_tracker import DEFAULT_TRACKER

    DEFAULT_TRACKER.stats.clear()
    yield
    DEFAULT_TRACKER.stats.clear()


#: The tree pytest actually imported, which is NOT always ``REPO``: the mutation harness
#: runs these tests against a copy, and a source-level assertion that read the live checkout
#: instead would pass on every mutant and pin nothing. ``REPO`` stays the git repository,
#: because the committed reference must come from real history.
IMPORTED_TREE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(fe.__file__)))
)

#: A roster whose FIRST name is PEFT's default. Used for the counterfactual against the
#: committed save path: with any other first name that path ALSO crashes, on
#: ``peft_config["default"]``, which is loud -- but it crashes after already writing the wrong
#: ``adapter_model.safetensors``. With "default" first it writes a complete, loadable, wrong
#: adapter and says nothing at all, which is the form the method would actually have met.
SILENT_ROSTER = ("default", "cluster_1")

#: The realistic roster: experts plus the shared adapter HDBSCAN noise routes to.
ROSTER = ("cluster_0", "cluster_1", "shared")


# --------------------------------------------------------------------------- harness -----


def export_engine(tmp_path, names=SILENT_ROSTER, *, seed=0, train=True):
    """A real ``FSDPEngine`` whose experts have all been trained, ready to export.

    Args:
        tmp_path: Handed to the shared builder.
        names: The adapter roster.
        seed: Seeds the base model and the adapters, so two engines built alike start
            bit-identical.
        train: Run one routed step per expert first. LoRA initialises ``B = 0``, so at init
            EVERY expert's delta is exactly zero and any merge whatsoever reproduces the sum
            -- a test that skipped this would pass on an export that dropped all but one
            expert. Every test that depends on it also asserts the deltas differ.

    Returns:
        ``(engine, adapter_set)``.
    """
    engine, adapters = make_engine(tmp_path, names=names, seed=seed)
    engine._initialized = True
    engine._cpu_group = dist.group.WORLD
    engine.is_vision_model = False
    engine.config.use_lora = True
    engine.config.dtype = "float32"  # see the module docstring
    if train:
        for step, name in enumerate(names):
            run(
                engine,
                make_batch(seed=step),
                plan_of({i: name for i in range(4)}, step=step),
            )
    return engine, adapters


def fresh_base(seed=0):
    """A base model bit-identical to the one ``make_engine`` wrapped at this seed."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(tiny_hf_config())


def load_saved_adapter(path, seed=0):
    """The saved artifact, loaded onto a freshly built base model -- the eval's own path."""
    from peft import PeftModel

    return PeftModel.from_pretrained(fresh_base(seed), path)


def logits_of(model, ids):
    """One deterministic forward pass."""
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits.clone()


def merged_reference(engine, names):
    """The in-memory merged model, built by PEFT's own ``add_weighted_adapter``.

    ``merge_sum`` is the tested path: it calls PEFT and then VERIFIES the result against an
    independently computed sum of per-adapter deltas. Using it as the reference means an
    export is compared against the officially merged model rather than against a second
    implementation written here.
    """
    target = merge_sum(engine.model, names, target="__ref__")
    engine.model.set_adapter(target)
    return engine.model


def total_delta(model, names):
    """``sum_c scaling_c B_c A_c`` per module, in float64, from the individual experts."""
    total = {}
    for name in names:
        for mod_name, dW in adapter_delta(model, name).items():
            total[mod_name] = total.get(mod_name, torch.zeros_like(dW)) + dW
    return total


def saved_delta(path, module_suffix):
    """The weight increment one module of a SAVED adapter contributes, in float64.

    Reads the rank and alpha out of ``adapter_config.json`` and applies ``alpha / r`` exactly
    as a server does, so a merge that got the tensors right and the config wrong fails here.
    """
    from safetensors.torch import load_file

    state = load_file(os.path.join(path, "adapter_model.safetensors"))
    with open(os.path.join(path, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    scaling = cfg["lora_alpha"] / cfg["r"]
    a = next(v for k, v in state.items() if k.endswith(f"{module_suffix}.lora_A.weight"))
    b = next(v for k, v in state.items() if k.endswith(f"{module_suffix}.lora_B.weight"))
    return scaling * (b.to(torch.float64) @ a.to(torch.float64))


def probe_module(adapters):
    """One module every expert is on, and the suffix a saved key ends with."""
    module = sorted(adapter_delta(adapters.model, adapters.names[0]))[0]
    return module, module.split("base_model.model.")[-1]


class _CPUPlatform:
    """The platform stub every drive of the sync path installs.

    Module level rather than local, because the committed reference compiled by
    :func:`_committed_method` resolves ``current_platform`` out of a NAMESPACE SNAPSHOT and
    must be handed the same stub explicitly. This box's GPUs are running a live job:
    unstubbed, ``current_platform.synchronize()`` initialises a CUDA context and
    ``_get_full_tensor`` copies every host tensor onto a card in use.
    """

    device_type = "cpu"

    @staticmethod
    def synchronize():
        return None


class _StubRollout:
    """The rollout engine's side of a weight sync, reduced to what the path calls."""

    def pause_generation(self):
        return None

    def continue_generation(self):
        return None

    def update_weights_from_distributed(self, meta, param_specs):
        fut: Future = Future()
        fut.set_result(None)
        return fut


def _committed_method(method: str, absent_marker: str, overrides=None):
    """The newest committed version of ``FSDPEngine.<method>`` that predates this work.

    Compiled from the git blob rather than transcribed here: a transcription pins someone's
    reading of the old code and cannot notice the reading being wrong. The search walks the
    file's history for the newest version whose source does NOT contain ``absent_marker``, so
    it keeps finding the pre-merge version after this work is itself committed.

    Args:
        method: Method name on ``FSDPEngine``.
        absent_marker: A string this work introduced, absent from every earlier version.
        overrides: Names to install in the compiled function's globals. The namespace is a
            SNAPSHOT of the engine module, so a monkeypatch applied to the module afterwards
            is invisible to the reference -- and a reference that quietly kept the real
            ``current_platform`` would touch the live job's cards.

    Returns:
        The unbound function, or ``None`` if the history is unavailable.
    """
    rel = "areal/engine/fsdp_engine.py"
    try:
        revs = subprocess.run(
            ["git", "log", "--format=%H", "--", rel],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        ).stdout.split()
    except Exception:
        return None
    for rev in revs:
        blob = subprocess.run(
            ["git", "show", f"{rev}:{rel}"],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        ).stdout
        if not blob or absent_marker in blob:
            continue
        tree = ast.parse(blob)
        for cls in tree.body:
            if not (isinstance(cls, ast.ClassDef) and cls.name == "FSDPEngine"):
                continue
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == method:
                    # Decorators are instrumentation (``trace_perf``) and would drag a tracer
                    # dependency into a comparison that is about tensors and bytes.
                    fn.decorator_list = []
                    ns = dict(fe.__dict__)
                    ns.update(overrides or {})
                    exec(compile(ast.Module([fn], []), rel, "exec"), ns)
                    return ns[method]
    return None


def digest(path):
    """``{filename: sha256}`` for a saved adapter directory."""
    out = {}
    for name in sorted(os.listdir(path)):
        with open(os.path.join(path, name), "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def drive_weight_sync(engine, monkeypatch):
    """Run the REAL ``_update_weights_from_distributed`` and record what it broadcast.

    Only the rollout engine and the platform are stubbed. The parameter selection, the
    gathers, the dtype cast, the bucketing, the ``ParamSpec`` list and the LoRA metadata are
    the shipped code, and ``dist.broadcast`` really runs -- over the world-of-one gloo group,
    where it is a self-broadcast.

    ``current_platform`` and ``torch.cuda.is_available`` are neutered because this box's GPUs
    are running a live job: unstubbed, the shipped path would allocate a CUDA stream and
    ``_get_full_tensor`` would copy every host tensor onto a card in use.

    Returns:
        ``(sent, meta)`` where ``sent`` is ``[(name, tensor)]`` in broadcast order.
    """
    sent: list[tuple[str, torch.Tensor]] = []
    original = fe.FSDPEngine._update_bucket_weights_from_distributed_async

    def spy(self, meta, named_tensors, **kwargs):
        # Cloned because ``_wait_pending_weight_update_bucket`` clears the list it was given.
        sent.extend((name, tensor.clone()) for name, tensor in named_tensors)
        return original(self, meta, named_tensors, **kwargs)

    monkeypatch.setattr(fe, "current_platform", _CPUPlatform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        fe.FSDPEngine, "_update_bucket_weights_from_distributed_async", spy
    )
    engine.rollout_engine = _StubRollout()
    engine.weight_update_master_addr = ""
    engine.weight_update_master_port = 0
    engine.weight_update_group_names = ["update_weight_group"]
    engine.weight_update_groups = [dist.group.WORLD]
    meta = WeightUpdateMeta(type="xccl", use_lora=True, version=0)
    engine._update_weights_from_distributed(meta)
    return sent, meta


# ============================================================== THE DEFECT, AS A FACT =====


def test_after_a_routed_step_exactly_one_expert_requires_grad(world, tmp_path):
    """The mechanism the whole failure rests on, asserted rather than argued.

    Without this, "``requires_grad`` selects one expert" is a claim about PEFT's activation
    semantics that this project does not own.
    """
    _engine, adapters = export_engine(tmp_path, names=ROSTER)
    trainable = {
        name
        for name in ROSTER
        if any(p.requires_grad for _k, p in adapters.parameters(name))
    }
    assert len(trainable) == 1, (
        f"{sorted(trainable)} require grad; the selector every exit point used is only a "
        "defect because exactly one does"
    )


def test_the_committed_save_path_wrote_one_expert_and_this_one_writes_the_sum(
    world, tmp_path
):
    """The live counterfactual: run the OLD save, run the new one, compare against the sum.

    This is the failing test that proved the defect, kept afterwards. It does not remember a
    number -- it executes the last committed ``_save_lora_to_hf`` against the same trained
    model and shows what that produced. Without the counterfactual the final assertion could
    be satisfied by a merge that happened to be right for a reason unrelated to the fix.
    """
    old = _committed_method("_save_lora_to_hf", "_lora_export_experts")
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history to compare with")
    engine, adapters = export_engine(tmp_path, names=SILENT_ROSTER)
    module, suffix = probe_module(adapters)
    reference = total_delta(engine.model, SILENT_ROSTER)[module]
    one_expert = adapter_delta(engine.model, "default")[module]
    assert not torch.allclose(one_expert, reference, atol=1e-6), (
        "the experts' deltas are indistinguishable on this batch, so nothing here could "
        "tell a one-expert export from the sum"
    )

    old_dir, new_dir = str(tmp_path / "old"), str(tmp_path / "new")
    old(engine, old_dir)
    engine._save_lora_to_hf(new_dir)

    got_old = saved_delta(old_dir, suffix)
    got_new = saved_delta(new_dir, suffix)
    assert torch.allclose(got_old, one_expert, atol=1e-6), (
        "the committed save path was expected to carry exactly the one active expert"
    )
    assert torch.allclose(got_new, reference, atol=1e-6), (
        f"the saved checkpoint is not the sum of the experts: worst disagreement "
        f"{float((got_new - reference).abs().max()):.3e}"
    )


def test_the_one_expert_checkpoint_was_a_complete_valid_adapter(world, tmp_path):
    """Why this is the silent class: the wrong artifact is not a broken one.

    It has every key a correct one has, it loads into a fresh model, and its names carry no
    adapter segment for vLLM to reject. Nothing downstream can tell it from the export of a
    run that trained a single adapter.
    """
    old = _committed_method("_save_lora_to_hf", "_lora_export_experts")
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=SILENT_ROSTER)
    old_dir, new_dir = str(tmp_path / "old"), str(tmp_path / "new")
    old(engine, old_dir)
    engine._save_lora_to_hf(new_dir)

    from safetensors.torch import load_file

    old_state = load_file(os.path.join(old_dir, "adapter_model.safetensors"))
    new_state = load_file(os.path.join(new_dir, "adapter_model.safetensors"))
    assert set(old_state) == set(new_state), "the wrong artifact had the same key set"
    assert all(f".{n}." not in k for k in old_state for n in SILENT_ROSTER)
    assert load_saved_adapter(old_dir) is not None, "it loads cleanly, which is the problem"


# ================================================================== THE ROUND TRIP ========


def test_the_reloaded_checkpoint_reproduces_the_merged_model_exactly(world, tmp_path):
    """The only test that proves an evaluation of this checkpoint would mean anything.

    Train every expert, save, reload the SAVED ARTIFACT into a fresh model, and require its
    logits to equal the in-memory merged model's. Both halves matter: the merge can be right
    and the serialisation wrong -- a dropped key, an adapter segment vLLM rejects, a rank the
    config does not advertise -- and every one of those produces a model that runs.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    path = str(tmp_path / "ckpt")
    engine._save_lora_to_hf(path)

    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(99))
    reloaded = logits_of(load_saved_adapter(path), ids)
    reference = logits_of(merged_reference(engine, ROSTER), ids)
    base_only = logits_of(fresh_base(0), ids)

    assert not torch.allclose(reference, base_only, atol=1e-5), (
        "the merged model is indistinguishable from the base model, so this proves nothing"
    )
    assert torch.allclose(reloaded, reference, atol=1e-5, rtol=0), (
        f"the reloaded checkpoint is not the merged model: worst difference "
        f"{float((reloaded - reference).abs().max()):.3e}"
    )


def test_a_one_expert_checkpoint_would_NOT_have_passed_the_round_trip(world, tmp_path):
    """The round trip's own control: it must FAIL on the artifact the defect produced.

    A round trip that passed on both the merged and the one-expert checkpoint would be
    measuring serialisation and would not constrain the merge at all.
    """
    old = _committed_method("_save_lora_to_hf", "_lora_export_experts")
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=SILENT_ROSTER)
    path = str(tmp_path / "one_expert")
    old(engine, path)

    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(99))
    reloaded = logits_of(load_saved_adapter(path), ids)
    reference = logits_of(merged_reference(engine, SILENT_ROSTER), ids)
    assert not torch.allclose(reloaded, reference, atol=1e-5), (
        "the one-expert checkpoint reproduced the merged model, so the round trip cannot "
        "tell them apart and does not constrain the merge"
    )


def test_the_saved_config_advertises_the_summed_rank(world, tmp_path):
    """A server applies ``alpha / r`` itself, so the config is part of the deployed weight."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    path = str(tmp_path / "ckpt")
    engine._save_lora_to_hf(path)
    with open(os.path.join(path, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    per_expert = engine.model.peft_config[ROSTER[0]].r
    assert cfg["r"] == per_expert * len(ROSTER)
    assert cfg["lora_alpha"] == cfg["r"], (
        "the merged A blocks are pre-scaled, so the merged adapter's own scaling must be 1.0"
    )

    from safetensors.torch import load_file

    state = load_file(os.path.join(path, "adapter_model.safetensors"))
    a = next(v for k, v in state.items() if k.endswith("lora_A.weight"))
    assert a.shape[0] == cfg["r"], "the advertised rank is not the rank of the tensors"


# ============================================================ EVERY OTHER EXIT POINT ======


def test_the_weight_sync_broadcasts_the_sum_and_not_one_expert(world, tmp_path, monkeypatch):
    """The least visible exit point: a run that syncs one expert trains right, rolls wrong.

    A bad checkpoint is read by a human eventually. This one feeds every rollout from the
    moment it runs, and nothing in the training metrics moves.
    """
    engine, adapters = export_engine(tmp_path, names=ROSTER)
    module, suffix = probe_module(adapters)
    reference = total_delta(engine.model, ROSTER)[module]
    one_expert = adapter_delta(engine.model, ROSTER[0])[module]
    assert not torch.allclose(one_expert, reference, atol=1e-6), "the batch cannot show it"

    sent, _meta = drive_weight_sync(engine, monkeypatch)
    by_name = dict(sent)
    assert by_name, "nothing was broadcast at all"
    assert all(f".{n}." not in k for k in by_name for n in ROSTER), (
        f"a broadcast name still carries an adapter segment: {sorted(by_name)[:3]}"
    )

    a = next(v for k, v in by_name.items() if k.endswith(f"{suffix}.lora_A.weight"))
    b = next(v for k, v in by_name.items() if k.endswith(f"{suffix}.lora_B.weight"))
    got = b.to(torch.float64) @ a.to(torch.float64)  # the merged adapter's scaling is 1.0
    assert torch.allclose(got, reference, atol=1e-6), (
        f"the weight sync is not the sum of the experts: worst disagreement "
        f"{float((got - reference).abs().max()):.3e}"
    )


def test_the_lora_metadata_the_server_receives_carries_the_merged_rank(
    world, tmp_path, monkeypatch
):
    """vLLM and SGLang apply ``alpha / r``, so one expert's ``r`` deploys 1/K of the delta."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    _sent, meta = drive_weight_sync(engine, monkeypatch)
    per_expert = engine.model.peft_config[ROSTER[0]].r
    assert meta.peft_config["r"] == per_expert * len(ROSTER)
    assert meta.peft_config["lora_alpha"] == meta.peft_config["r"]


def test_get_lora_adapter_info_reports_the_merged_shapes(world, tmp_path):
    """The shapes a server sizes its LoRA slots from, found without a single collective."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    info = engine.get_lora_adapter_info()
    assert info, "no adapter info at all"
    assert all(f".{n}." not in k for k in info for n in ROSTER)
    per_expert = engine.model.peft_config[ROSTER[0]].r
    a_shapes = [v for k, v in info.items() if k.endswith("lora_A.weight")]
    assert a_shapes and all(s[0] == per_expert * len(ROSTER) for s in a_shapes)


def test_the_shape_only_path_agrees_with_the_tensor_path(world, tmp_path):
    """Two selections would be two chances to disagree, and a shape that disagreed with its
    tensor is what a rollout engine reports as a load failure at the worst moment."""
    import re

    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    tensors = dict(
        engine._merged_lora_named_tensors(
            ROSTER, engine.model.named_parameters(), for_checkpoint=True
        )
    )
    by_tensor = {
        re.sub(r"^base_model\.model\.", "", k): list(v.shape) for k, v in tensors.items()
    }
    assert engine.get_lora_adapter_info() == by_tensor


def test_the_dcp_checkpoint_keeps_every_expert_separate(world, tmp_path):
    """The resume path must NOT be merged, and must not be filtered by ``requires_grad``.

    A merged DCP checkpoint could not resume a run: the experts would be gone and the
    optimizer state would refer to parameters that no longer exist. This exit point is
    correct as it stands, and the assertion is here so that a later change applying the merge
    everywhere fails rather than quietly destroying the ability to restart.
    """
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    saved = get_model_state_dict(engine.model)
    for name in ROSTER:
        assert any(f".{name}." in key for key in saved), (
            f"expert {name!r} is absent from the DCP state dict, so a resume would restart "
            "it from its initialisation"
        )
    path = str(tmp_path / "dcp")
    engine._save_to_dcp(path, with_optim=False)
    assert os.listdir(path), "the DCP save wrote nothing"


# ================================================== THE SELECTION REFUSES, LOUDLY =========


def test_a_multi_adapter_model_with_no_roster_refuses_to_save(world, tmp_path):
    """The guard that catches this defect from any FUTURE caller, on the default path.

    A model carrying several adapters without an armed ``ClusterAdapterSet`` has no correct
    ``requires_grad`` answer, and the artifact it would write is indistinguishable from a
    correct one. This is the only place that difference is observable.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    del engine._selfevo_adapters
    with pytest.raises(MergeSelectionError, match="requires_grad"):
        engine._save_lora_to_hf(str(tmp_path / "nope"))


def test_a_multi_adapter_model_with_no_roster_refuses_to_sync(world, tmp_path, monkeypatch):
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    del engine._selfevo_adapters
    with pytest.raises(MergeSelectionError, match="requires_grad"):
        drive_weight_sync(engine, monkeypatch)


def test_a_multi_adapter_model_with_no_roster_refuses_to_report_its_shapes(world, tmp_path):
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    del engine._selfevo_adapters
    with pytest.raises(MergeSelectionError, match="requires_grad"):
        engine.get_lora_adapter_info()


def test_a_trainable_parameter_that_is_not_a_lora_tensor_is_refused(world, tmp_path):
    """It would be dropped from a merged export, and nothing downstream would say so."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    victim = next(p for n, p in engine.model.named_parameters() if "lora_" not in n)
    victim.requires_grad_(True)
    with pytest.raises(MergeSelectionError, match="trainable but is not a LoRA tensor"):
        engine._merged_lora_named_tensors(
            ROSTER, engine.model.named_parameters(), for_checkpoint=True
        )


def test_a_module_carrying_only_some_experts_is_refused():
    """A merge over a ragged module set applies some experts on some layers only."""
    t = torch.ones(2, 3)
    named = {
        "m.lora_A.a.weight": t,
        "m.lora_B.a.weight": t.T,
        "m.lora_A.b.weight": t,  # and no lora_B for b
    }
    with pytest.raises(MergeSelectionError, match="ragged|but not for"):
        merge_expert_tensors(named, ("a", "b"), {"a": 1.0, "b": 1.0})


def test_a_tensor_from_an_adapter_outside_the_roster_is_refused():
    t = torch.ones(2, 3)
    named = {"m.lora_A.a.weight": t, "m.lora_B.a.weight": t.T, "m.lora_A.z.weight": t}
    with pytest.raises(MergeSelectionError, match="not one of the experts"):
        merge_expert_tensors(named, ("a",), {"a": 1.0})


def test_a_dora_magnitude_vector_is_refused_rather_than_silently_zeroed():
    """PEFT's own ``cat`` branch skips it and leaves it zeroed, which is not a merge."""
    t = torch.ones(2, 3)
    named = {
        "m.lora_A.a.weight": t,
        "m.lora_B.a.weight": t.T,
        "m.lora_magnitude_vector.a": torch.ones(3),
    }
    with pytest.raises(MergeSelectionError, match="DoRA"):
        merge_expert_tensors(named, ("a",), {"a": 1.0})


def test_a_name_with_no_adapter_layer_is_refused_rather_than_guessed():
    with pytest.raises(MergeSelectionError, match="carries no adapter layer"):
        split_lora_param_name("model.layers.0.mlp.down_proj.weight")


def test_an_empty_selection_is_refused():
    """An empty export writes a valid adapter that changes nothing, and every eval of it
    would report the base model as the method."""
    with pytest.raises(MergeSelectionError, match="empty export"):
        merge_expert_tensors({}, ("a",), {"a": 1.0})


def test_an_expert_whose_scaling_varies_by_module_is_refused(world, tmp_path):
    """One scaling per expert is only meaningful if every module agrees on it."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    touched = False
    for _n, mod in engine.model.named_modules():
        if isinstance(mod, BaseTunerLayer) and ROSTER[0] in getattr(mod, "scaling", {}):
            mod.scaling[ROSTER[0]] *= 2.0
            touched = True
            break
    assert touched, "no tuner layer to perturb, so this test cannot fire"
    with pytest.raises(MergeSelectionError, match="scaling"):
        expert_scalings(engine.model, ROSTER)


# ================================================= THE ARITHMETIC IS PEFT'S, PINNED =======


def test_the_merged_tensors_are_what_peft_add_weighted_adapter_cat_produces(world, tmp_path):
    """PEFT stays the reference implementation; this module is transport, and that is a TEST.

    The export cannot call ``add_weighted_adapter``: at a save point the experts are FSDP2
    DTensors and PEFT would build new unsharded modules on the live model, outside the
    optimizer and outside FSDP, in the middle of training. So the tensor-level merge is
    pinned against PEFT here, on a model where PEFT can run.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    named = {
        n: p.detach().clone() for n, p in engine.model.named_parameters() if "lora_" in n
    }
    mine = merge_expert_tensors(named, ROSTER, expert_scalings(engine.model, ROSTER))

    merge_sum(engine.model, ROSTER, target="__peft__")
    theirs = {
        n.replace(".__peft__.", "."): p.detach()
        for n, p in engine.model.named_parameters()
        if ".__peft__." in n
    }
    assert set(mine) == set(theirs), f"key sets differ: {sorted(set(mine) ^ set(theirs))[:4]}"
    for key in mine:
        assert torch.equal(mine[key], theirs[key]), (
            f"{key} differs from PEFT's own cat by "
            f"{float((mine[key] - theirs[key]).abs().max())}"
        )


def test_the_merged_config_is_what_peft_add_weighted_adapter_produces(world, tmp_path):
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    mine = merged_lora_config(engine.model.peft_config, ROSTER)
    merge_sum(engine.model, ROSTER, target="__peft__")
    theirs = engine.model.peft_config["__peft__"]
    assert (mine.r, mine.lora_alpha) == (theirs.r, theirs.lora_alpha)
    assert set(mine.target_modules) == set(theirs.target_modules)


def test_the_operator_is_the_sum_and_not_the_mean(world, tmp_path):
    """Held FIXED across arms, so it is asserted rather than left to a docstring.

    A mean would divide the deployed update by K while every training metric stayed
    identical, and A1 - A0 would then be reading a K-fold difference in effective learning
    rate rather than the clustering.
    """
    assert MERGE_OPERATOR == "sum"
    engine, adapters = export_engine(tmp_path, names=ROSTER)
    module, suffix = probe_module(adapters)
    reference = total_delta(engine.model, ROSTER)[module]
    path = str(tmp_path / "ckpt")
    engine._save_lora_to_hf(path)
    got = saved_delta(path, suffix)
    assert torch.allclose(got, reference, atol=1e-6)
    assert not torch.allclose(got, reference / len(ROSTER), atol=1e-6), (
        "the export is the mean of the experts, not their sum"
    )


def test_the_reloaded_checkpoint_is_not_any_single_expert(world, tmp_path):
    """The input-level form of the defect assertion, on the whole model rather than a weight.

    Weights can agree while the deployed model does not: rank, alpha and key format all sit
    between a correct tensor and a correct forward pass.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    path = str(tmp_path / "ckpt")
    engine._save_lora_to_hf(path)
    ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(5))
    reloaded = logits_of(load_saved_adapter(path), ids)

    for name in ROSTER:
        engine.model.set_adapter(name)
        single = logits_of(engine.model, ids)
        assert not torch.allclose(reloaded, single, atol=1e-5), (
            f"the reloaded checkpoint is exactly expert {name!r}"
        )


# ======================================================== DEFAULT OFF, BIT FOR BIT ========


def test_the_default_save_path_is_the_committed_one_byte_for_byte(world, tmp_path):
    """The rollback check for the checkpoint: an unconfigured run writes the same BYTES.

    Compares the files, not the tensors. A tensor comparison would pass an export whose
    config, key order or dtype had shifted, and every one of those changes what a server
    loads.
    """
    old = _committed_method("_save_lora_to_hf", "_lora_export_experts")
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=("default",), seed=13)
    del engine._selfevo_adapters  # an ordinary single-adapter run arms nothing
    old_dir, new_dir = str(tmp_path / "old"), str(tmp_path / "new")
    old(engine, old_dir)
    engine._save_lora_to_hf(new_dir)
    before, after = digest(old_dir), digest(new_dir)
    assert set(before) == {"adapter_config.json", "adapter_model.safetensors"}
    assert before == after, (
        "the default save path drifted from the committed one: "
        f"{[k for k in before if before[k] != after.get(k)]}"
    )


def test_the_default_weight_sync_is_the_committed_one_bit_for_bit(
    world, tmp_path, monkeypatch
):
    """The rollback check for the sync: the same names in the same order, the same tensors,
    and the same LoRA metadata handed to the server."""
    old = _committed_method(
        "_update_weights_from_distributed",
        "_lora_export_experts",
        overrides={"current_platform": _CPUPlatform},
    )
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=("default",), seed=13)
    del engine._selfevo_adapters

    sent_new, meta_new = drive_weight_sync(engine, monkeypatch)
    # Copied: the recorder ``drive_weight_sync`` installed is still in place and the
    # committed reference below goes through it too, so the live list would grow under us
    # and the comparison would be against both runs concatenated.
    sent_new = list(sent_new)

    captured: list[tuple[str, torch.Tensor]] = []
    current = fe.FSDPEngine._update_bucket_weights_from_distributed_async

    def spy(self, meta, named_tensors, **kwargs):
        captured.extend((name, tensor.clone()) for name, tensor in named_tensors)
        return current(self, meta, named_tensors, **kwargs)

    monkeypatch.setattr(
        fe.FSDPEngine, "_update_bucket_weights_from_distributed_async", spy
    )
    meta_old = WeightUpdateMeta(type="xccl", use_lora=True, version=0)
    old(engine, meta_old)

    assert captured, "the committed sync broadcast nothing, so this compares two empties"
    assert [n for n, _ in captured] == [n for n, _ in sent_new]
    for (_n, before), (_m, after) in zip(captured, sent_new):
        assert torch.equal(before, after)
    assert meta_old.peft_config == meta_new.peft_config


def test_the_default_adapter_info_is_the_committed_one(world, tmp_path):
    old = _committed_method("get_lora_adapter_info", "_lora_export_experts")
    if old is None:
        pytest.skip("no pre-merge fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=("default",), seed=13)
    del engine._selfevo_adapters
    assert old(engine) == engine.get_lora_adapter_info()


def test_an_unconfigured_run_does_not_import_selfevo_to_export(world):
    """The hard constraint, checked in a FRESH interpreter.

    ``selfevo`` is already imported inside this test process, so an in-process check would
    pass on any implementation at all.
    """
    script = (
        "import sys;"
        "import areal.engine.fsdp_engine;"
        "print(sorted(k for k in sys.modules if k.startswith('selfevo')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=IMPORTED_TREE,
        capture_output=True,
        text=True,
        timeout=600,
        env=dict(os.environ, PYTHONPATH=IMPORTED_TREE),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    # Last line only: importing the engine emits banner logging on stdout, and a test that
    # compared the whole stream would fail on a logging change rather than on an import.
    printed = out.stdout.strip().splitlines()[-1].strip()
    assert printed == "[]", (
        f"importing the engine pulled in {printed}; the default path must not touch selfevo"
    )


def test_every_selfevo_import_in_the_engine_is_function_local(world):
    """The mechanism behind the test above, so a top-level import fails at source level."""
    with open(os.path.join(IMPORTED_TREE, "areal/engine/fsdp_engine.py")) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        assert not (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and "selfevo" in (getattr(node, "module", "") or "")
        ), "a module-level selfevo import would load it for every run"
    found = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and "selfevo" in (node.module or "")
    )
    assert found >= 4, f"only {found} selfevo imports found; the export seams are not wired"


def test_every_export_mutation_anchor_occurs_exactly_once(world):
    """The mutation harness decays silently as the code it points at is rewritten.

    An anchor that matches zero lines is reported SKIP and is not a kill, which is the right
    discipline -- but only if somebody reads the output. This makes a decayed anchor a
    failing test instead. Source-only, so it costs nothing and runs anywhere.
    """
    import importlib.util

    path = os.path.join(IMPORTED_TREE, "selfevo/tests/mutate_cluster_lora_export.py")
    if not os.path.exists(path):
        pytest.skip("the export mutation harness is not in this tree")
    spec = importlib.util.spec_from_file_location("_mutate_export", path)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [path, IMPORTED_TREE]
    spec.loader.exec_module(module)

    # Taken from the harness's own target table rather than listed again here: a second
    # list is a second thing to forget when a target is added, and forgetting it would make
    # every mutation on the new target invisible to this check.
    sources = {
        name: open(path).read() for name, path in module.targets().items()
    }
    for target, label, find, replace in module.MUTATIONS:
        found = sources[target].count(find)
        assert found == 1, f"[{target}] {label}: anchor appears {found}x, not once"
        assert find != replace, f"[{target}] {label}: the replacement is a no-op"


# ============================ THE OTHER HALF: FORWARDS TAKEN OUTSIDE step() ==============
#
# The export defect ships a wrong artifact. This one changes the gradient. ``only()``
# restores ``names[0]``, so between training steps the model IS one arbitrary expert, and
# every forward taken outside ``ClusterAdapterSet.step`` ran on it: ``compute_logp``, which
# is the importance-ratio DENOMINATOR for every group in the batch and not only the routed
# one; ``eval_batch``, which is where reported numbers come from; and the behavioural
# forward, whose output decides the next partition. All three are silent.


def forward_logprobs(engine, batch):
    """One real ``forward_batch``, the call ``PPOActor._compute_logp`` makes."""
    return engine.forward_batch(
        dict(batch), aggregate_fn=lambda xs: torch.cat(xs, dim=-1)
    )


def test_between_steps_the_model_is_one_arbitrary_expert(world, tmp_path):
    """The mechanism, asserted before anything is claimed to depend on it."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    assert list(engine.model.active_adapters) == [ROSTER[0]], (
        "only() was expected to leave exactly the first expert active; if it does not, the "
        "account below of what every read-only forward was seeing is wrong"
    )


def test_a_read_only_forward_sees_every_expert_and_not_one(world, tmp_path):
    """``compute_logp`` runs through ``forward_batch``; it must see the DEPLOYED model.

    The importance ratio divides by this. If the denominator comes from ``cluster_0`` while
    the update came from the sum, every group's ratio is wrong on every step, and nothing
    downstream reports it.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    batch = make_batch(seed=21)

    got = forward_logprobs(engine, batch)

    merged_reference(engine, ROSTER)                  # engine.model is now the merged model
    del engine._selfevo_adapters                      # so the seam does not re-activate
    want = forward_logprobs(engine, batch)
    engine.model.set_adapter(ROSTER[0])
    single = forward_logprobs(engine, batch)

    assert not torch.allclose(single, want, atol=1e-5), (
        "one expert is indistinguishable from the sum on this batch, so nothing here could "
        "detect the defect"
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=0), (
        f"the read-only forward is not the merged model: worst difference "
        f"{float((got - want).abs().max()):.3e}"
    )


def test_eval_batch_also_sees_every_expert(world, tmp_path):
    """Any number we report comes through here."""
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    batch = make_batch(seed=22)

    got = engine.eval_batch(dict(batch), loss_fn=linear_loss, loss_weight_fn=weight_fn)

    merged_reference(engine, ROSTER)
    del engine._selfevo_adapters
    want = engine.eval_batch(dict(batch), loss_fn=linear_loss, loss_weight_fn=weight_fn)
    engine.model.set_adapter(ROSTER[0])
    single = engine.eval_batch(dict(batch), loss_fn=linear_loss, loss_weight_fn=weight_fn)

    assert not torch.allclose(single, want, atol=1e-6), "the batch cannot show it"
    assert torch.allclose(got, want, atol=1e-6), (
        f"eval_batch ran on a single expert: {float(got)} vs merged {float(want)}"
    )


def test_a_training_forward_still_sees_exactly_one_expert(world, tmp_path):
    """The seam must NOT reach the training pass, or the method becomes its own baseline.

    Routing each cluster's gradient into its own expert is the whole method. If the training
    forward activated every expert, the run would train them all on every batch, report the
    same per-cluster metrics, and BE the shared-adapter arm wearing the method's name.
    """
    engine, adapters = export_engine(tmp_path, names=ROSTER, train=False)
    seen = []
    original = type(engine.model).forward

    def spy(self, *args, **kwargs):
        seen.append(tuple(self.active_adapters))
        return original(self, *args, **kwargs)

    before = {n: adapters.snapshot(n) for n in ROSTER}
    type(engine.model).forward = spy
    try:
        run(engine, make_batch(seed=1), plan_of(dict.fromkeys(range(4), "cluster_1")))
    finally:
        type(engine.model).forward = original

    assert seen, "no forward was taken at all"
    assert all(active == ("cluster_1",) for active in seen), (
        f"a training forward saw {sorted({a for s in seen for a in s})}; the cluster's "
        "gradient would land in every expert"
    )
    assert adapters.unchanged("cluster_0", before["cluster_0"])
    assert adapters.unchanged("shared", before["shared"])


def test_the_read_only_seam_restores_the_active_adapter_exactly(world, tmp_path):
    """A leaked activation sends the NEXT cluster's gradient to every expert."""
    engine, adapters = export_engine(tmp_path, names=ROSTER)
    before_active = list(engine.model.active_adapters)
    before_grad = {
        n: {k: bool(p.requires_grad) for k, p in adapters.parameters(n)} for n in ROSTER
    }
    forward_logprobs(engine, make_batch(seed=23))
    assert list(engine.model.active_adapters) == before_active
    after_grad = {
        n: {k: bool(p.requires_grad) for k, p in adapters.parameters(n)} for n in ROSTER
    }
    assert after_grad == before_grad, (
        "the read-only forward left experts trainable that were frozen before it"
    )


def test_the_read_only_forward_leaves_no_gradient_anywhere(world, tmp_path):
    """``set_adapter`` marks every active adapter trainable, so this is not free.

    ``every_expert`` runs under ``torch.no_grad()`` precisely so a backward cannot be taken
    inside it. Checked on the parameters rather than on the context manager, because what
    matters is that no expert acquired a gradient.
    """
    engine, adapters = export_engine(tmp_path, names=ROSTER)
    engine.optimizer_zero_grad()
    forward_logprobs(engine, make_batch(seed=24))
    for name in ROSTER:
        for key, param in adapters.parameters(name):
            assert param.grad is None, f"{key} acquired a gradient during a read-only pass"


def test_the_behaviour_feature_forward_sees_every_expert(world, tmp_path):
    """The feature decides the NEXT partition; read from one expert it clusters that adapter.

    Asserted on the vectors themselves: the roster-aware call must differ from the
    single-expert one, and must equal what the merged model produces.
    """
    from selfevo.cluster_lora.wiring import ClusterLoRAConfig, behaviour_features

    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    cfg = ClusterLoRAConfig(partition="meds", features=True)
    batch = make_batch(seed=25)
    sizes = [2, 2, 2, 2]

    with_all, _ = behaviour_features(engine.model, batch, sizes, cfg, experts=ROSTER)
    single, _ = behaviour_features(engine.model, batch, sizes, cfg)

    merged_reference(engine, ROSTER)
    merged, _ = behaviour_features(engine.model, batch, sizes, cfg)

    assert not np.allclose(single, merged, atol=1e-5), (
        "one expert and the merged model give the same behavioural vectors here, so this "
        "cannot tell them apart"
    )
    assert np.allclose(with_all, merged, atol=1e-5), (
        f"the behavioural forward did not see the deployed model: worst difference "
        f"{float(np.abs(with_all - merged).max()):.3e}"
    )


def test_every_expert_is_exactly_the_merged_adapter_at_the_output(world, tmp_path):
    """The one place the in-process model and the exported artifact are pinned together.

    Activating N adapters and deploying their concatenated sum are two different mechanisms.
    They have to agree, or a run's own logprobs describe a different model from its
    checkpoint -- and the difference would appear only as an unexplained train/eval gap.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(31))
    with every_expert(engine.model, ROSTER):
        multi = logits_of(engine.model, ids)
    merged = logits_of(merged_reference(engine, ROSTER), ids)
    assert torch.allclose(multi, merged, atol=1e-5, rtol=0), (
        f"activating every expert is not the merged adapter: worst difference "
        f"{float((multi - merged).abs().max()):.3e}"
    )


def test_every_expert_refuses_a_name_the_model_does_not_carry(world, tmp_path):
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    with pytest.raises(ClusterWiringError, match="not on this model"):
        with every_expert(engine.model, (*ROSTER, "ghost")):
            pass


def test_every_expert_refuses_an_empty_roster(world, tmp_path):
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    with pytest.raises(ClusterWiringError, match="at least one expert"):
        with every_expert(engine.model, ()):
            pass


def test_every_expert_refuses_an_activation_that_does_not_take(world, tmp_path, monkeypatch):
    """A silently-ignored activation would leave every forward on one expert.

    ``PeftModel.set_adapter`` raises on a list, so the seam goes through ``base_model``. An
    API that accepted the call and did nothing is the failure this checks: the context
    manager would be apparently in place while the model stayed on one expert.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    monkeypatch.setattr(engine.model.base_model, "set_adapter", lambda names: None)
    with pytest.raises(ClusterWiringError, match="subset of the experts"):
        with every_expert(engine.model, ROSTER):
            pass


def test_peft_still_rejects_a_list_from_PeftModel_set_adapter(world, tmp_path):
    """The constraint the seam is designed against, pinned so an upgrade reports it.

    If a future peft accepts a list on ``PeftModel.set_adapter``, the ``base_model``
    indirection in ``every_expert`` stops being necessary and this test says so, rather than
    the indirection quietly becoming folklore.
    """
    engine, _adapters = export_engine(tmp_path, names=ROSTER, train=False)
    with pytest.raises(TypeError):
        engine.model.set_adapter(list(ROSTER))


def test_an_unarmed_read_only_forward_is_the_committed_one_bit_for_bit(world, tmp_path):
    """Default off: with no roster the read-only forward is exactly the pre-seam one.

    Compares the LOGITS every microbatch produced, in order, with ``torch.equal``. A
    tolerance would pass a leaked activation, which is the only difference this can produce.
    """
    old = _committed_method("forward_backward_batch", "_read_only_forward_adapters")
    if old is None:
        pytest.skip("no pre-seam fsdp_engine.py in this checkout's history")
    engine, _adapters = export_engine(tmp_path, names=("default",), seed=17)
    del engine._selfevo_adapters

    def capture(fn):
        seen = []
        mb_list = engine._prepare_mb_list(make_batch(seed=8)).to(engine.device)
        fn(engine, mb_list, lambda logits, _ctx: seen.append(logits.detach().clone()),
           forward_only=True)
        return seen

    before = capture(old)
    after = capture(type(engine).forward_backward_batch)
    assert before, "the committed forward produced no microbatch at all"
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert torch.equal(a, b), f"the default forward drifted by {(a - b).abs().max()}"
