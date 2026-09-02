"""SEAM 1, driven through the REAL ``PPOActor._compute_advantages``.

``FINDINGS_cluster_lora.md`` section 9 records the gap this file closes: the key function
reached ``ClusterRouter`` but the actor never supplied behavioural features, so
``cluster_lora.partition=meds`` in a config could only REFUSE. Everything here goes through
the actual advantage computation training calls, not through ``_route_groups`` or
``begin_cluster_batch`` directly, because a test that calls the helper cannot catch the
helper being unreachable -- which is precisely the state this work was fixing.

The fixtures are imported from :mod:`selfevo.tests.test_group_routing` rather than copied,
for the reason that file already gives: two definitions of "an actor configured like the
live runs" drift, and the drift is silent.

**The environment is the config surface**, as it is for ``router=random``'s
``SELFEVO_RANDOM_PROPORTIONS``: ``_route_groups`` builds routers with ``factory()`` and no
kwargs, so nothing can be passed from ``GroupRoutingConfig``. Every test sets the variables
through ``monkeypatch`` so the process environment is restored, and the default -- every
variable unset -- is asserted to be a no-op rather than assumed to be one.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig  # noqa: E402
from areal.utils import stats_tracker  # noqa: E402
from selfevo.cluster_lora import wiring  # noqa: E402
from selfevo.cluster_lora.partition import PartitionUnavailable  # noqa: E402
from selfevo.cluster_lora.wiring import (  # noqa: E402
    ClusterLoRAConfig,
    ClusterPlan,
    ClusterWiringError,
    EngineOptimizer,
    adapter_roster,
    rows_by_adapter,
    select_rows,
)
from selfevo.routing.base import TrainingMode  # noqa: E402

from selfevo.tests.test_group_routing import (  # noqa: E402
    B,
    G,
    MIXED,
    PROMPT,
    T,
    advantages,
    make_actor,
    make_batch,
    meta,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CLUSTER_ROUTED = GroupRoutingConfig(enabled=True, router="cluster")


class FakeEngine:
    """Somewhere for the actor to arm a plan, and optionally a model to read features from.

    A real ``FSDPEngine`` is exercised by ``test_cluster_lora_engine.py``; here the engine is
    only the other end of the handoff, and standing one up would make a failure ambiguous
    between the two seams.
    """

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer


def routed_actor(monkeypatch, partition, *, engine=None, **env):
    """An actor configured like the live runs, with a cluster-LoRA arm switched on.

    Args:
        monkeypatch: pytest's, so the process environment is restored afterwards.
        partition: Value for the master switch.
        engine: What to hang on ``actor.engine``.
        **env: Further ``SELFEVO_CLUSTER_LORA_*`` suffixes, e.g. ``FEATURES="1"``.

    Returns:
        The actor.
    """
    monkeypatch.setenv(wiring.CLUSTER_LORA_ENV, partition)
    for key, value in env.items():
        monkeypatch.setenv(f"SELFEVO_CLUSTER_LORA_{key}", value)
    actor = make_actor(CLUSTER_ROUTED)
    actor.engine = engine
    return actor


def tiny_lm(seed=0):
    """A four-layer Qwen2 the layer-logit extractor can walk. No PEFT: features need none."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.train()
    return model


@pytest.fixture
def emitted(monkeypatch):
    """Every key the actor puts on the stats stream during a test.

    The requirement is that the reach report lands in the SAME stream as the ``route/*``
    keys, so it is captured at ``stats_tracker.scalar`` -- the one call site both use --
    rather than by reading the report object, which would prove only that the report exists.
    """
    seen: dict[str, float] = {}
    original = stats_tracker.scalar

    def spy(**kw):
        seen.update(kw)
        return original(**kw)

    monkeypatch.setattr(stats_tracker, "scalar", spy)
    return seen


# ------------------------------------------------------------------------- default off ---


def test_the_actor_gate_and_the_wiring_constant_are_the_same_variable():
    """The gate is a literal in ``actor.py`` so the default path imports no selfevo at all.

    That duplication is deliberate and is therefore pinned here: a rename on one side would
    otherwise leave a run whose config asks for the method and whose actor never notices.
    """
    src = (REPO / "areal/trainer/ppo/actor.py").read_text()
    assert f'os.environ.get("{wiring.CLUSTER_LORA_ENV}", "")' in src
    engine_src = (REPO / "areal/engine/fsdp_engine.py").read_text()
    assert f'os.environ.get("{wiring.ROSTER_ENV}", "")' in engine_src


def test_with_no_cluster_configuration_nothing_is_armed_and_nothing_moves(monkeypatch):
    """The rollback: an unconfigured run arms nothing at all.

    The advantage tensor is NOT compared against an unrouted run here, because a
    ``router=cluster`` arm without cluster-LoRA is the derived silence split and is supposed
    to change it. What must be true is that no key function was built, no partition was
    formed and no plan was armed, so the engine takes its unmodified path.
    """
    monkeypatch.delenv(wiring.CLUSTER_LORA_ENV, raising=False)
    actor = make_actor(CLUSTER_ROUTED)
    actor.engine = FakeEngine()
    advantages(actor, MIXED)
    assert not hasattr(actor, "_selfevo_cluster_keyfn")
    assert not hasattr(actor.engine, "_selfevo_cluster_plan")
    assert actor._selfevo_router.key_fn is None


def test_the_config_is_absent_by_default_and_the_extra_forward_is_off(monkeypatch):
    """Two separate defaults, and the second is the one that costs a forward pass."""
    monkeypatch.delenv(wiring.CLUSTER_LORA_ENV, raising=False)
    assert ClusterLoRAConfig.from_env() is None
    assert ClusterLoRAConfig.from_env({}) is None
    assert ClusterLoRAConfig.from_env({wiring.CLUSTER_LORA_ENV: "none"}).features is False
    assert adapter_roster({}) == ()


# ------------------------------------------------------------------- the seam itself -----


def test_the_actor_supplies_features_and_arms_the_engine(monkeypatch):
    """The seam: a partition is formed during the real advantage computation and handed on."""
    engine = FakeEngine()
    actor = routed_actor(monkeypatch, "none", engine=engine)
    advantages(actor, MIXED)
    keyfn = actor._selfevo_cluster_keyfn
    assert keyfn.batches == 1
    assert actor._selfevo_router.key_fn is keyfn
    plan = engine._selfevo_cluster_plan
    assert isinstance(plan, ClusterPlan)
    assert plan.key_of_group == {0: "shared", 1: "shared"}


def test_the_key_function_survives_across_batches(monkeypatch):
    """MEDS labels are stabilised against the buffered history.

    A fresh key function per batch would refit from nothing and relabel every group every
    step -- churn 1.0, the measured failure in findings 5.1 -- so the instance has to be the
    same object on batch two, and it has to have seen both.
    """
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine())
    advantages(actor, MIXED)
    first = actor._selfevo_cluster_keyfn
    advantages(actor, MIXED)
    assert actor._selfevo_cluster_keyfn is first
    assert first.batches == 2


def test_the_cluster_arm_does_not_touch_the_advantage_tensor(monkeypatch):
    """Which EXPERT gets the gradient is the independent variable; the loss is not.

    ClusterRouter's default policy names the three silence clusters and an adapter name
    matches none of them, so without the explicit RL policy every group would take the SKIP
    fallback and ``apply_decisions`` would zero the batch -- a run that trains nothing while
    reporting a healthy partition. This is the assertion that says it does not.
    """
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    torch.manual_seed(0)
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine())
    got = advantages(actor, MIXED)
    assert torch.equal(base, got), (base - got).abs().max()
    assert actor._selfevo_router.policy == {"shared": TrainingMode.RL}


def test_the_reach_report_reaches_the_same_stream_as_the_route_keys(monkeypatch, emitted):
    """An arm's log must say what the partition did, on the panel the routing keys are on."""
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine())
    advantages(actor, MIXED)
    assert "route/mixed_groups" in emitted, "the routing keys did not come through either"
    for key in (
        "cluster_lora/n_groups",
        "cluster_lora/n_clusters",
        "cluster_lora/noise_fraction",
        "cluster_lora/largest_cluster_fraction",
        "cluster_lora/churn",
        "cluster_lora/churn_overlap",
        "cluster_lora/refusals",
        "cluster_lora/size/shared",
        "cluster_lora/feature_seconds",
        "cluster_lora/feature_fallbacks",
        "cluster_lora/adapters_available",
    ):
        assert key in emitted, key
    assert emitted["cluster_lora/size/shared"] == 2.0
    assert emitted["cluster_lora/noise_fraction"] == 1.0


def test_churn_is_measurable_because_the_groups_keep_their_prompt_identity(monkeypatch,
                                                                          emitted):
    """Churn keyed on batch position is noise; keyed on the prompt it is a measurement.

    The same two prompts are routed twice, so the second batch must report a non-zero
    OVERLAP. A run whose overlap is always zero cannot tell "nothing moved" from "nothing
    was comparable", which is the distinction ``ReachReport`` keeps two fields for.
    """
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine())
    torch.manual_seed(0)
    batch = make_batch(MIXED)
    actor._compute_advantages(dict(batch), meta())
    actor._compute_advantages(dict(batch), meta())
    assert emitted["cluster_lora/churn_overlap"] == 2.0
    assert emitted["cluster_lora/churn"] == 0.0


# ---------------------------------------------------------------------- the refusals -----


def test_partition_meds_without_the_extra_forward_still_refuses(monkeypatch):
    """The deliberate refusal, and it must survive being made satisfiable.

    A mode that needs behavioural features and is given none has exactly one thing it could
    return -- one adapter for everything -- and that is the ``none`` arm wearing the
    method's label. It raises instead, and it raised before this seam existed for the same
    reason.
    """
    actor = routed_actor(monkeypatch, "meds", engine=FakeEngine())
    with pytest.raises(PartitionUnavailable, match="FEATURES"):
        advantages(actor, MIXED)


def test_random_matched_without_features_refuses_too(monkeypatch):
    """The control needs the method's own sizes, so it needs the same features."""
    actor = routed_actor(monkeypatch, "random_matched", engine=FakeEngine())
    with pytest.raises(PartitionUnavailable):
        advantages(actor, MIXED)


def test_a_router_that_cannot_carry_a_partition_is_refused(monkeypatch):
    """``router=solve_rate`` decides per unit; a partition handed to it would go unused."""
    monkeypatch.setenv(wiring.CLUSTER_LORA_ENV, "none")
    actor = make_actor(GroupRoutingConfig(enabled=True, router="solve_rate"))
    actor.engine = FakeEngine()
    with pytest.raises(ClusterWiringError, match="router='cluster'"):
        advantages(actor, MIXED)


def test_a_partition_naming_an_adapter_outside_the_roster_is_refused(monkeypatch):
    """Every expert is created before the optimizer, so one discovered later has none.

    Refusing here rather than at the engine is deliberate: this is the earliest point the
    mismatch is knowable, and the message names both fixes (a bigger roster, or a bigger
    ``min_cluster_size``) rather than leaving the run to discover it after the GPUs are
    allocated.
    """
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(), ADAPTERS="cluster_0")
    with pytest.raises(ClusterWiringError, match="not in SELFEVO_CLUSTER_LORA_ADAPTERS"):
        advantages(actor, MIXED)


def test_a_roster_naming_the_partition_is_accepted(monkeypatch):
    """The counterpart, so the refusal above is not passing on every roster."""
    actor = routed_actor(
        monkeypatch, "none", engine=FakeEngine(), ADAPTERS="cluster_0,cluster_1,shared"
    )
    advantages(actor, MIXED)
    assert actor.engine._selfevo_cluster_plan.key_of_group == {0: "shared", 1: "shared"}


def test_an_unknown_partition_is_refused_by_name(monkeypatch):
    """A misspelled mode silently becoming the baseline is the failure being prevented."""
    with pytest.raises(ValueError, match="not a training partition"):
        ClusterLoRAConfig.from_env({wiring.CLUSTER_LORA_ENV: "med"})


def test_a_repeated_adapter_in_the_roster_is_refused():
    """Two names that are one expert would look like a two-expert arm."""
    with pytest.raises(ValueError, match="repeats an adapter name"):
        adapter_roster({wiring.ROSTER_ENV: "cluster_0,cluster_0"})


def test_a_non_numeric_sweep_value_is_refused_rather_than_defaulted():
    """A typo in a swept knob must not silently restore MEDS' shipped min_cluster_size."""
    with pytest.raises(ValueError, match="is not an integer"):
        ClusterLoRAConfig.from_env(
            {wiring.CLUSTER_LORA_ENV: "none",
             "SELFEVO_CLUSTER_LORA_MIN_CLUSTER_SIZE": "five"}
        )


# ------------------------------------------------------------------- the extra forward ---


def test_the_extra_forward_produces_one_vector_per_group(monkeypatch):
    """Features on: the behavioural vector reaches the key function through the real actor.

    The vector is the per-layer logit trace at the answer token, reduced by ``meds_feature``
    and averaged over a group's rollouts -- the same quantity ``interference_dump.py``
    stores, so the training arm and the probe cluster the same thing.
    """
    model = tiny_lm()
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(model), FEATURES="1")
    advantages(actor, MIXED)
    assert actor._selfevo_cluster_keyfn.batches == 1


def test_the_feature_forward_leaves_the_model_exactly_as_it_found_it(monkeypatch):
    """It is a measurement, not a step: no parameter moves and train mode comes back.

    Eval mode during the pass is not cosmetic. With dropout active the same rollout would
    give a different behavioural vector every batch, and the clustering would be measuring
    dropout.
    """
    model = tiny_lm()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(model), FEATURES="1")
    advantages(actor, MIXED)
    assert model.training, "the model was left in eval mode"
    for k, v in model.named_parameters():
        assert torch.equal(v, before[k]), k


def test_the_cost_of_the_extra_forward_is_reported_every_batch(monkeypatch, emitted):
    """The matched-budget claim in experiments/m25/PLAN.md is counted, not estimated."""
    model = tiny_lm()
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(model), FEATURES="1")
    advantages(actor, MIXED)
    assert emitted["cluster_lora/feature_seconds"] > 0.0
    off = routed_actor(monkeypatch, "none", engine=FakeEngine(model), FEATURES="0")
    advantages(off, MIXED)
    assert emitted["cluster_lora/feature_seconds"] == 0.0


def test_features_asked_for_with_no_model_are_refused(monkeypatch):
    """An arm configured to spend a forward pass and unable to must not quietly not."""
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(None), FEATURES="1")
    with pytest.raises(ClusterWiringError, match="no engine model"):
        advantages(actor, MIXED)


def test_meds_with_features_reaches_the_clusterer_rather_than_the_refusal(monkeypatch):
    """With features supplied, ``meds`` gets past ``PartitionUnavailable`` -- the whole seam.

    The training venv deliberately carries neither hdbscan nor scikit-learn, so on it the
    call lands on ``ClusteringUnavailable``. Reaching THAT exception is itself the proof
    that the features arrived: ``PartitionUnavailable`` is raised first and is what an
    unsupplied one gives. Under a venv that has the dependencies the same test asserts a
    partition came back, so neither environment turns it into a test of nothing.
    """
    from selfevo.clustering.meds import ClusteringUnavailable

    model = tiny_lm()
    actor = routed_actor(monkeypatch, "meds", engine=FakeEngine(model), FEATURES="1")
    try:
        advantages(actor, MIXED)
    except ClusteringUnavailable as exc:
        assert "scikit-learn" in str(exc) or "hdbscan" in str(exc)
        return
    assert actor._selfevo_cluster_keyfn.partition is not None
    assert actor.engine._selfevo_cluster_plan.key_of_group


# ---------------------------------------------------------------------- the plan units ---


def test_rows_by_adapter_follows_group_identity():
    """Rows are grouped by the plan's key, in whatever order the batch presents them."""
    got = rows_by_adapter([0, 1, 0, 1, 2], {0: "a", 1: "b", 2: "a"})
    assert got == {"a": [0, 2, 4], "b": [1, 3]}


def test_rows_by_adapter_refuses_a_group_the_plan_does_not_name():
    """Dropping the row trains on less than the batch; defaulting it is a guess."""
    with pytest.raises(ClusterWiringError, match="does not name"):
        rows_by_adapter([0, 5], {0: "a"})


def test_select_rows_over_every_row_is_the_identity():
    """The degenerate partition must reproduce the unrouted batch, value for value."""
    batch = make_batch(MIXED)
    got = select_rows(batch, range(B))
    assert set(got) == set(batch)
    for k, v in batch.items():
        assert torch.equal(got[k], v), k


def test_select_rows_carries_entries_that_are_not_shaped_like_the_batch():
    """Mirrors the microbatch splitter, which carries such entries rather than dropping."""
    batch = dict(make_batch(MIXED))
    batch["cu_seqlens"] = torch.arange(3)
    batch["note"] = "not a tensor"
    got = select_rows(batch, [0, 1])
    assert got["note"] == "not a tensor"
    assert torch.equal(got["cu_seqlens"], batch["cu_seqlens"])
    assert got["input_ids"].shape[0] == 2


def test_an_empty_plan_is_refused():
    """A plan with no groups routes nothing and would report a step that trained nothing."""
    with pytest.raises(ValueError, match="routes nothing"):
        ClusterPlan(key_of_group={}, step=0, basis="test")


def test_zeroing_gradients_into_tensors_is_refused():
    """``set_to_none`` is the isolation guarantee, not a micro-optimisation.

    A parameter whose ``.grad`` is ``None`` is skipped by ``torch.optim`` entirely; one
    holding a zero TENSOR acquires Adam state and is decayed on every subsequent step. An
    idle expert would move, and the bit-equality guard would be measuring an implementation
    that leaks.
    """
    with pytest.raises(ClusterWiringError, match="set_to_none=False"):
        EngineOptimizer(object()).zero_grad(set_to_none=False)


def test_an_unconfigured_run_does_not_even_import_the_wiring(monkeypatch):
    """The gate is what keeps the live job's import graph unchanged, so it is pinned.

    ``begin_cluster_batch`` would return an empty dict with no configuration anyway, which
    makes the environment check look redundant -- it is not. The check is what stops the
    default path importing :mod:`selfevo.cluster_lora` at all, and this asserts it by making
    that import fail and requiring an unconfigured run to be unaffected.
    """
    import sys

    monkeypatch.delenv(wiring.CLUSTER_LORA_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "selfevo.cluster_lora.wiring", None)
    actor = make_actor(CLUSTER_ROUTED)
    actor.engine = FakeEngine()
    advantages(actor, MIXED)


def test_the_feature_forward_runs_with_the_model_in_eval_mode(monkeypatch):
    """Recorded DURING the pass, not after it.

    Checking ``model.training`` afterwards cannot see this: an implementation that never
    called ``eval()`` leaves training True, which is also what a correct one restores.
    """
    model = tiny_lm()
    seen = []
    original = model.forward

    def watched(*a, **kw):
        seen.append(model.training)
        return original(*a, **kw)

    monkeypatch.setattr(model, "forward", watched)
    actor = routed_actor(monkeypatch, "none", engine=FakeEngine(model), FEATURES="1")
    advantages(actor, MIXED)
    assert seen and not any(seen), "the behavioural forward ran with dropout active"


def test_a_group_vector_is_the_MEAN_over_its_rollouts():
    """A group's behaviour is its rollouts' average, computed here independently.

    Taking one member instead would produce a plausible vector of the right shape from a
    single rollout, and the clustering would then be over rollouts wearing group labels.
    """
    from selfevo.cluster_lora.wiring import behaviour_features

    model = tiny_lm()
    batch = make_batch(MIXED)
    cfg = ClusterLoRAConfig(partition="none", features=True)
    grouped, fallbacks = behaviour_features(model, batch, [G, G], cfg)
    per_row, _ = behaviour_features(model, batch, [1] * B, cfg)
    assert grouped.shape == (2, per_row.shape[1])
    assert fallbacks == 0
    for i in range(2):
        assert grouped[i] == pytest.approx(
            per_row[i * G : (i + 1) * G].mean(axis=0), rel=1e-9, abs=1e-9
        )


def test_group_sizes_that_do_not_partition_the_batch_are_refused():
    """A mismatch would pair a group with another group's behaviour, silently."""
    from selfevo.cluster_lora.wiring import behaviour_features

    cfg = ClusterLoRAConfig(partition="none", features=True)
    with pytest.raises(ValueError, match="group sizes sum to"):
        behaviour_features(tiny_lm(), make_batch(MIXED), [G], cfg)


def test_a_sequence_whose_answer_token_cannot_be_found_is_counted(monkeypatch, emitted):
    """A feature read at a different position is not comparable with its peers.

    ``answer_strategy=boxed`` with no tokenizer on the engine cannot locate ``\\boxed{``, so
    every row falls back to its final position. That is allowed -- refusing mid-batch would
    lose the step -- but it is COUNTED, because a run where it happened on every row is
    measuring something else from one where it never did.
    """
    model = tiny_lm()
    actor = routed_actor(
        monkeypatch, "none", engine=FakeEngine(model), FEATURES="1", ANSWER="boxed"
    )
    advantages(actor, MIXED)
    assert emitted["cluster_lora/feature_fallbacks"] == float(B)
