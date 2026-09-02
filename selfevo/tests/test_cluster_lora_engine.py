"""THE GUARD, through the REAL engine: a cluster's step moves its own expert and no other.

``selfevo/tests/test_cluster_lora_routing.py`` already proves that property of
``ClusterAdapterSet.step`` on a real PEFT model with a real optimizer. It cannot prove that
TRAINING has it, because until this seam existed ``FSDPEngine.train_batch`` ran one adapter
for the whole batch and never called that class. So everything here drives the actual
``FSDPEngine.train_batch`` -- the method ``PPOActor._ppo_update`` calls -- with its real
microbatch packing, its real ``_compute_logprobs_and_loss``, and its real
``optimizer_step`` including gradient clipping and the scheduler.

The engine is constructed rather than ``initialize()``d: initialize() shards with FSDP2 and
needs accelerators, so the distributed attributes are supplied for a world of one and
everything downstream of them is the shipped code. What is exercised is therefore the whole
train_batch path minus sharding, on CPU, against a real Qwen2 of four layers.

``AdamW`` with a NON-ZERO weight decay is load-bearing, exactly as in the unit-level guard:
decoupled decay moves a parameter on every step it is stepped on, including one whose
gradient is exactly zero, so ``set_to_none=True`` is what makes an idle expert bit-identical
rather than merely close. Under a zero-decay optimizer a correct and an incorrect
implementation both pass and these tests constrain nothing.
"""

from __future__ import annotations

import ast
import subprocess

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch.distributed as dist  # noqa: E402

from areal.api.alloc_mode import FSDPParallelStrategy  # noqa: E402
from areal.api.cli_args import (  # noqa: E402
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
)
from areal.engine.fsdp_engine import FSDPEngine  # noqa: E402
from areal.engine.fsdp_utils.parallel import ParallelHelper  # noqa: E402
from areal.utils import logging as areal_logging  # noqa: E402
from areal.utils.stats_tracker import DEFAULT_TRACKER  # noqa: E402
from selfevo.cluster_lora.adapters import ClusterAdapterSet  # noqa: E402
from selfevo.cluster_lora.wiring import ClusterPlan, ClusterWiringError  # noqa: E402

REPO = "/home/ubuntu/areal-selfevo"
NAMES = ("cluster_0", "cluster_1", "shared")
B, T, G = 8, 12, 2          # four groups of two
PROMPT = 4


@pytest.fixture(autouse=True)
def _clear_stats_tracker():
    """Drop the process-global stats between tests, as the loss-weighting audit does."""
    DEFAULT_TRACKER.stats.clear()
    yield
    DEFAULT_TRACKER.stats.clear()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A world of one, over a FILE store so no port is taken from the live job.

    Torn down at the end of the module: the default process group is process-global state
    and leaving it initialised would change how unrelated test modules behave.
    """
    if dist.is_initialized():
        yield
        return
    store = tmp_path_factory.mktemp("pg") / "store"
    dist.init_process_group(
        backend="gloo", init_method=f"file://{store}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def tiny_hf_config():
    """A four-layer Qwen2 small enough to train on CPU in a test."""
    from transformers import AutoConfig

    return AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=128, tie_word_embeddings=False,
    )


def make_engine(tmp_path, names=NAMES, *, seed=0, n_mbs=1, weight_decay=0.01):
    """A real ``FSDPEngine`` on CPU, wrapping a real PEFT model with one expert per cluster.

    Args:
        tmp_path: Where the model config is written; ``FSDPEngine.__init__`` reads it.
        names: Adapter roster. Created through ``ClusterAdapterSet.build``, which is what
            the engine's own ``_apply_peft_wrapper`` calls when the roster is configured.
        seed: Seeds the model and adapter initialisation, so two engines built with the same
            seed start bit-identical -- asserted where that matters rather than assumed.
        n_mbs: Microbatches per cluster stream.
        weight_decay: NON-ZERO by default. See the module docstring.

    Returns:
        ``(engine, adapter_set)``.
    """
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM

    hf = tiny_hf_config()
    path = tmp_path / f"model_{seed}_{'_'.join(names)}"
    path.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(path)

    cfg = PPOActorConfig(
        path=str(path),
        mb_spec=MicroBatchSpec(n_mbs=n_mbs),
        optimizer=OptimizerConfig(lr=1e-2, weight_decay=weight_decay),
    )
    engine = FSDPEngine(cfg)

    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_config(tiny_hf_config())
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    adapters = ClusterAdapterSet.build(base, names, lora)
    engine.model = adapters.model
    engine._selfevo_adapters = adapters
    # Every expert, exactly as _create_optimizer does it (model.parameters()), so an expert
    # that exists but is not optimised cannot hide behind a hand-picked parameter list.
    engine.optimizer = torch.optim.AdamW(
        engine.model.parameters(), lr=1e-2, weight_decay=weight_decay
    )
    engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        engine.optimizer, factor=1.0, total_iters=1
    )
    engine.device = torch.device("cpu")
    engine.parallel_helper = ParallelHelper.from_parallel_strategy(
        FSDPParallelStrategy(
            data_parallel_size=1, tensor_parallel_size=1,
            context_parallel_size=1, pipeline_parallel_size=1,
        )
    )
    engine.world_mesh = _mesh()
    engine.dp_group = dist.group.WORLD
    engine.logger = areal_logging.getLogger("cluster-lora-test")
    engine.is_offload = False
    engine.model.train()
    return engine, adapters


_MESH = None


def _mesh():
    """One shared device mesh: building a new one per engine creates new subgroups."""
    global _MESH
    if _MESH is None:
        from torch.distributed.device_mesh import init_device_mesh

        _MESH = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp_sp", "tp"))
    return _MESH


def make_batch(seed=0, group_of_row=None):
    """A batch shaped like the one ``_ppo_update`` hands the engine.

    ``group_ids`` is carried PER TOKEN because that is the only shape that survives
    microbatch splitting, and the plan is keyed on it because splitting reorders rows.

    Args:
        seed: Seeds the tokens and advantages.
        group_of_row: Group index per row. Defaults to contiguous groups of ``G``; pass a
            permuted one to check that the routing follows group identity and not position.

    Returns:
        The batch dict.
    """
    g = torch.Generator().manual_seed(seed)
    loss_mask = torch.zeros(B, T, dtype=torch.long)
    loss_mask[:, PROMPT:] = 1
    if group_of_row is None:
        group_of_row = [i // G for i in range(B)]
    gid = torch.tensor(group_of_row, dtype=torch.long)
    return {
        "input_ids": torch.randint(0, 128, (B, T), generator=g),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "loss_mask": loss_mask,
        "advantages": torch.randn(B, T, generator=g),
        "logprobs": torch.zeros(B, T),
        "old_logp": torch.zeros(B, T),
        "prox_logp": torch.zeros(B, T),
        "group_ids": gid.unsqueeze(1).expand(B, T).contiguous(),
    }


def linear_loss(logprobs, entropy, input_data, **_kw):
    """A token-MEAN loss that is otherwise linear in the rows, so cluster losses must sum.

    The mean is the engine's convention, not a choice: ``_compute_logprobs_and_loss``
    multiplies whatever a loss returns by ``local_weight / total_loss_weight``, so a loss
    that returned a SUM would be counted once per microbatch and scaled down again, and
    splitting a batch in two would halve its gradient. Verified by
    ``test_a_cluster_spanning_two_microbatches_accumulates_both_of_them``, which is the
    check that would fail if this were a sum.

    Deliberately not ``grpo_loss_fn``: the clipped surrogate is linear in the rows too, but
    it carries its own normalisation and its own statistics tracker, and a denominator test
    should fail on the denominator rather than on either of those. One test below does run
    the real GRPO loss, to show the seam works with what training actually uses.
    """
    mask = input_data["loss_mask"]
    return (logprobs * input_data["advantages"] * mask).sum() / mask.count_nonzero()


def weight_fn(x):
    """The loss weight the live actor passes."""
    return x["loss_mask"].count_nonzero()


def run(engine, data, plan=None, loss_fn=linear_loss):
    """One real ``train_batch``, optionally with a cluster plan armed."""
    if plan is not None:
        engine._selfevo_cluster_plan = plan
    return engine.train_batch(data, loss_fn=loss_fn, loss_weight_fn=weight_fn)


def plan_of(mapping, step=0):
    """A ``ClusterPlan`` from ``{group index: adapter}``."""
    return ClusterPlan(key_of_group=dict(mapping), step=step, basis="test")


ALL_C0 = {0: "cluster_0", 1: "cluster_0", 2: "cluster_0", 3: "cluster_0"}
ALL_C1 = {0: "cluster_1", 1: "cluster_1", 2: "cluster_1", 3: "cluster_1"}
SPLIT = {0: "cluster_0", 1: "cluster_0", 2: "cluster_1", 3: "cluster_1"}


# ------------------------------------------------------- the guard, through the engine ---


def test_a_step_for_one_cluster_leaves_every_other_expert_bit_identical(world, tmp_path):
    """The property the method rests on, asserted after real engine steps.

    Two steps, not one: Adam state and decoupled decay only diverge from the correct
    behaviour once state exists, so a single step passes on an implementation that leaks.
    """
    engine, adapters = make_engine(tmp_path)
    before = {n: adapters.snapshot(n) for n in NAMES}
    for step in range(2):
        run(engine, make_batch(seed=step), plan_of(ALL_C1, step=step))
    assert adapters.unchanged("cluster_0", before["cluster_0"])
    assert adapters.unchanged("shared", before["shared"])


def test_the_cluster_that_did_receive_the_batch_actually_changed(world, tmp_path):
    """Without this the assertion above passes on a step that did nothing at all."""
    engine, adapters = make_engine(tmp_path)
    before = adapters.snapshot("cluster_1")
    run(engine, make_batch(), plan_of(ALL_C1))
    assert not adapters.unchanged("cluster_1", before)


def test_the_isolation_holds_in_the_other_direction_too(world, tmp_path):
    """Symmetry, so an implementation that happens to freeze one expert fails."""
    engine, adapters = make_engine(tmp_path)
    before = {n: adapters.snapshot(n) for n in NAMES}
    for step in range(2):
        run(engine, make_batch(seed=step), plan_of(ALL_C0, step=step))
    assert adapters.unchanged("cluster_1", before["cluster_1"])
    assert adapters.unchanged("shared", before["shared"])
    assert not adapters.unchanged("cluster_0", before["cluster_0"])


def test_two_clusters_in_one_batch_both_move_and_the_absent_one_does_not(world, tmp_path):
    """The realistic step: several clusters present, one absent."""
    engine, adapters = make_engine(tmp_path)
    before = {n: adapters.snapshot(n) for n in NAMES}
    stats = run(engine, make_batch(), plan_of(SPLIT))
    assert not adapters.unchanged("cluster_0", before["cluster_0"])
    assert not adapters.unchanged("cluster_1", before["cluster_1"])
    assert adapters.unchanged("shared", before["shared"])
    assert stats["cluster_lora/clusters_stepped"] == 2.0
    assert stats["cluster_lora/rows/cluster_0"] == 4.0
    assert stats["cluster_lora/rows/cluster_1"] == 4.0


def test_an_expert_that_is_idle_this_step_does_not_decay(world, tmp_path):
    """The defect that survived the unit-level guard, now checked through the engine.

    An expert that was never trained has no ``.grad`` and is skipped by the optimizer
    whatever the implementation does. The leak needs an expert that WAS trained and is then
    idle -- the normal case, since most clusters are absent from most batches -- because
    only then does it have Adam state and a decay to apply.
    """
    engine, adapters = make_engine(tmp_path)
    run(engine, make_batch(seed=0), plan_of(SPLIT))
    after_first = {n: adapters.snapshot(n) for n in NAMES}
    for step in range(1, 4):
        run(engine, make_batch(seed=step), plan_of(ALL_C0, step=step))
    assert adapters.unchanged("cluster_1", after_first["cluster_1"]), (
        "a trained-then-idle expert moved; weight decay reached it, which means its "
        "gradient was a zero tensor rather than None"
    )
    assert not adapters.unchanged("cluster_0", after_first["cluster_0"])


def test_the_real_grpo_loss_drives_the_same_isolation(world, tmp_path):
    """The seam with the loss training actually uses, not only the linear stand-in."""
    import functools

    from areal.trainer.ppo.actor import grpo_loss_fn

    engine, adapters = make_engine(tmp_path)
    loss_fn = functools.partial(
        grpo_loss_fn, eps_clip=0.2, eps_clip_higher=None, c_clip=None
    )
    before = {n: adapters.snapshot(n) for n in NAMES}
    for step in range(2):
        run(engine, make_batch(seed=step), plan_of(ALL_C1, step=step), loss_fn=loss_fn)
    assert not adapters.unchanged("cluster_1", before["cluster_1"])
    assert adapters.unchanged("cluster_0", before["cluster_0"])
    assert adapters.unchanged("shared", before["shared"])


# ----------------------------------------------------------------- default is unchanged --


@pytest.mark.parametrize("n_mbs", [1, 2, 3], ids=["one-mb", "two-mbs", "three-mbs"])
def test_the_none_arm_reproduces_the_unrouted_step_bit_for_bit(world, tmp_path, n_mbs):
    """``partition=none`` must be the vanilla arm running the method's code path.

    Every group on one adapter is the degenerate partition, so the cluster path reduces to
    exactly one microbatch stream over exactly the rows the unrouted path uses. Any
    difference -- a per-cluster denominator, a dropped or doubled microbatch, a second
    optimizer step -- shows up as a parameter that is not bit-identical.

    Run at several microbatch counts on purpose. The cluster path hands the LAST microbatch
    of each cluster back to ``ClusterAdapterSet.step`` to be backwarded there instead of
    inside the engine loop; at ``n_mbs=1`` that is the only microbatch and the arrangement
    is untested, while at 2 and 3 a dropped or doubled final microbatch changes the
    gradient and breaks bit-equality against the engine's own accumulation.
    """
    a, ad_a = make_engine(tmp_path, names=("shared",), seed=7, n_mbs=n_mbs)
    b, ad_b = make_engine(tmp_path, names=("shared",), seed=7, n_mbs=n_mbs)
    start = ad_a.snapshot("shared")
    assert ad_b.unchanged("shared", start), "the two engines did not start identical"

    data = make_batch(seed=3)
    run(a, data)                                                  # today's path
    run(b, make_batch(seed=3), plan_of({i: "shared" for i in range(4)}))
    now_a, now_b = dict(ad_a.parameters("shared")), dict(ad_b.parameters("shared"))
    assert not ad_a.unchanged("shared", start), "neither step trained anything"
    assert set(now_a) == set(now_b)
    for k in now_a:
        assert torch.equal(now_a[k], now_b[k]), (
            f"{k} differs between the unrouted step and the degenerate cluster step by "
            f"{(now_a[k] - now_b[k]).abs().max()}"
        )


def _committed_train_batch():
    """The last committed ``train_batch`` that predates this seam, as a callable.

    Compiled from the git blob rather than from a copy in this file: a copy pins a
    transcription of the old code and cannot notice the transcription being wrong. The
    search walks the file's history for the newest version that does NOT dispatch to
    ``_train_batch_by_cluster``, so it keeps finding the pre-seam version after this work is
    itself committed.

    Returns:
        The unbound function, or ``None`` if the history is unavailable.
    """
    rel = "areal/engine/fsdp_engine.py"
    try:
        revs = subprocess.run(
            ["git", "log", "--format=%H", "--", rel],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        ).stdout.split()
    except Exception:
        return None
    import areal.engine.fsdp_engine as fe

    for rev in revs:
        blob = subprocess.run(
            ["git", "show", f"{rev}:{rel}"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        ).stdout
        if not blob or "_train_batch_by_cluster" in blob:
            continue
        tree = ast.parse(blob)
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == "FSDPEngine":
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "train_batch":
                        ns = dict(fe.__dict__)
                        exec(compile(ast.Module([fn], []), rel, "exec"), ns)
                        return ns["train_batch"]
    return None


def test_the_default_path_is_the_committed_one_bit_for_bit(world, tmp_path):
    """The rollback check: with no cluster config the step is the pre-seam step exactly.

    Runs the last committed pre-seam ``train_batch`` and the current one on two engines
    seeded identically, and requires every parameter to match with ``torch.equal``. A
    tolerance would pass a leaked difference, which is the only kind this can produce.
    """
    original = _committed_train_batch()
    if original is None:
        pytest.skip("no pre-seam fsdp_engine.py in this checkout's history to compare with")
    a, ad_a = make_engine(tmp_path, names=("shared",), seed=11)
    b, ad_b = make_engine(tmp_path, names=("shared",), seed=11)
    start = ad_a.snapshot("shared")
    assert ad_b.unchanged("shared", start), "the two engines did not start identical"

    original(a, make_batch(seed=5), loss_fn=linear_loss, loss_weight_fn=weight_fn)
    b.train_batch(make_batch(seed=5), loss_fn=linear_loss, loss_weight_fn=weight_fn)
    assert not ad_a.unchanged("shared", start), "the reference step trained nothing"
    now_a, now_b = dict(ad_a.parameters("shared")), dict(ad_b.parameters("shared"))
    for k in now_a:
        assert torch.equal(now_a[k], now_b[k]), f"{k} drifted from the committed step"


def test_an_unarmed_engine_never_looks_for_a_plan(world, tmp_path):
    """With no plan the engine must not require a roster, a group_ids column, or anything."""
    engine, adapters = make_engine(tmp_path, names=("shared",))
    del engine._selfevo_adapters
    data = make_batch()
    del data["group_ids"]
    before = adapters.snapshot("shared")
    run(engine, data)
    assert not adapters.unchanged("shared", before)


# ------------------------------------------------------------------ the denominator ------


def test_the_loss_denominator_is_the_WHOLE_batch_and_is_taken_once(world, tmp_path,
                                                                    monkeypatch):
    """Splitting by cluster must change where the gradient lands, not how big it is.

    ``_compute_logprobs_and_loss`` scales each microbatch by ``local_weight /
    total_loss_weight``. If ``total_loss_weight`` were recomputed per cluster, each cluster
    would be divided by its own token count instead of the batch's and a small cluster would
    silently receive a large learning rate -- the difference between a partition that
    reassigns gradient and one that reweights it.

    Asserted on the quantity itself rather than on the losses it scales. The losses cannot
    settle this on CPU: microbatches are PACKED, and without flash-attention's
    ``cu_seq_lens`` the attention runs causally across the whole packed row, so regrouping
    the same rows changes every logprob slightly. That artefact is confined to this harness
    -- it is why every other comparison here holds the packing fixed -- and it does not
    touch the denominator, which is arithmetic over token counts.
    """
    import areal.engine.fsdp_engine as fe

    seen = []
    original = fe.compute_total_loss_weight

    def spy(mb_list, loss_weight_fn, dp_group, device=None):
        """Record the token span and the value every call reduces over."""
        value = original(mb_list, loss_weight_fn, dp_group, device=device)
        seen.append((sum(mb_list.group_lens), float(value)))
        return value

    monkeypatch.setattr(fe, "compute_total_loss_weight", spy)
    plain, _ = make_engine(tmp_path, seed=13)
    run(plain, make_batch(seed=2))
    clustered, _ = make_engine(tmp_path, seed=13)
    run(clustered, make_batch(seed=2), plan_of({0: "cluster_0", 1: "cluster_0",
                                                2: "cluster_1", 3: "shared"}))
    assert len(seen) == 2, (
        f"the denominator was computed {len(seen)} times for two steps; a per-cluster "
        "denominator would be computed once per cluster"
    )
    assert seen[1] == seen[0], (
        f"the cluster step normalised by {seen[1]} against the unrouted step's {seen[0]}"
    )
    assert seen[0][0] == B * T, "the reference did not span the whole batch"


# ---------------------------------------------------------------------- plan handling ----


def test_the_plan_follows_group_identity_not_row_position(world, tmp_path):
    """Microbatch splitting reorders rows, so the plan cannot be keyed on position.

    The rows here are interleaved -- group 0 owns rows 0 and 4 -- so an implementation that
    read the plan by slicing the batch into contiguous blocks would route half the rows to
    the wrong expert and still produce a plausible step.
    """
    engine, adapters = make_engine(tmp_path)
    interleaved = [0, 1, 2, 3, 0, 1, 2, 3]
    stats = run(
        engine,
        make_batch(seed=8, group_of_row=interleaved),
        plan_of({0: "cluster_0", 1: "cluster_0", 2: "cluster_1", 3: "cluster_1"}),
    )
    assert stats["cluster_lora/rows/cluster_0"] == 4.0
    assert stats["cluster_lora/rows/cluster_1"] == 4.0


def test_a_batch_whose_group_is_not_in_the_plan_is_refused(world, tmp_path):
    """A row the plan does not name has no expert; routing it anywhere is a guess."""
    engine, _ = make_engine(tmp_path)
    with pytest.raises(ClusterWiringError, match="does not name"):
        run(engine, make_batch(), plan_of({0: "cluster_0"}))


def test_a_plan_with_no_adapter_set_is_refused(world, tmp_path):
    """An armed plan and no experts is a run that would train nothing and report a step."""
    engine, _ = make_engine(tmp_path)
    del engine._selfevo_adapters
    with pytest.raises(ClusterWiringError, match="ClusterAdapterSet"):
        run(engine, make_batch(), plan_of(ALL_C0))


def test_a_batch_without_group_ids_is_refused_rather_than_positioned(world, tmp_path):
    """Without the per-token carrier there is no identity to route on."""
    engine, _ = make_engine(tmp_path)
    data = make_batch()
    del data["group_ids"]
    with pytest.raises(ClusterWiringError, match="group_ids"):
        run(engine, data, plan_of(ALL_C0))


def test_the_engine_step_still_reports_grad_norm_and_the_learning_rate(world, tmp_path):
    """The cluster path takes the ENGINE's optimizer step, not a second one written here.

    Clipping, the scheduler and ``update_successful`` are what every run's dashboard reads;
    a per-cluster step that bypassed them would train while making those keys disappear.
    """
    engine, _ = make_engine(tmp_path)
    stats = run(engine, make_batch(), plan_of(SPLIT))
    assert set(stats) >= {"update_successful", "grad_norm", "lr", "num_micro_batches"}
    assert stats["update_successful"] == 1.0
    assert stats["grad_norm"] > 0.0
    assert stats["cluster_lora/clusters_skipped"] == 1.0
