"""ADVERSARIAL AUDIT of the cluster-LoRA and gold paths, 2026-09-02.

Read ``selfevo/AUDIT_2026_09_02.md`` alongside this file; it carries the measurements and
the recommended fix for each defect, and this file is the evidence for it.

EIGHT CONFIRMED DEFECTS, each an ``xfail(strict=True)``. Every test named ``test_defect_*``
asserts the behaviour the code SHOULD have and does NOT have. They are marked rather than
left red so that the suite stays green and so a reader who did not commission the audit can
tell a finding from breakage -- and ``strict=True`` is the point of the marker: the moment
someone lands the fix, the test XPASSes, which pytest reports as a FAILURE, so the fix
cannot be made without noticing that the marker is now wrong. Remove the marker in the same
commit as the fix. Do NOT remove a test: deleting one removes the evidence.

The audit did not apply any of those fixes. The tree is imported by a live 8xA100 job
through a ``.pth``, so no existing source file was modified.

SEVEN REFUTED HYPOTHESES, each an ordinary passing test named ``test_refuted_*``. Those are
things this audit suspected and disproved. They are kept because a refuted hypothesis with a
test still constrains future change, and because two of them -- the gold coordinate roll and
the shared loss denominator -- are the properties a plausible-looking edit would break first.

The engine fixtures are imported from :mod:`selfevo.tests.test_cluster_lora_engine` rather
than copied. Two definitions of "an engine configured like the live runs" drift, and the
drift is silent, which is the whole subject of this file.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch.distributed as dist  # noqa: E402

from areal.api.cli_args import GroupRoutingConfig  # noqa: E402
from selfevo.cluster_lora.partition import (  # noqa: E402
    Partition,
    label_churn,
    random_matched_partition,
)
from selfevo.gold.substitute import (  # noqa: E402
    GOLD_LOGP_SENTINEL,
    GoldOrderingError,
    assert_gold_logprobs_filled,
    reconcile_gold_logprobs,
    substitute_gold_rows,
    substitute_in_place,
)
from selfevo.cluster_lora.wiring import (  # noqa: E402
    ClusterWiringError,
    tag_cluster_batch,
)
from selfevo.tests.test_cluster_lora_engine import (  # noqa: E402
    NAMES,
    SPLIT,
    linear_loss,
    make_batch,
    make_engine,
    plan_of,
    weight_fn,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A world of one over a FILE store, so no port is taken from the live job.

    Torn down at the end of the module, exactly as ``test_cluster_lora_engine.py`` does it
    -- and, unlike that module, the cached device mesh is dropped with it. ``make_engine``
    is imported from there, so the FIRST module to call it builds that module's ``_MESH``
    global against whichever process group is live at the time. Leaving a stale mesh behind
    a destroyed group makes every later engine test fail inside a collective, which is how
    this fixture was found to need the extra line rather than by reasoning about it.
    """
    import selfevo.tests.test_cluster_lora_engine as engine_tests

    if dist.is_initialized():
        yield
        return
    store = tmp_path_factory.mktemp("audit_pg") / "store"
    dist.init_process_group(
        backend="gloo", init_method=f"file://{store}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        engine_tests._MESH = None
        dist.destroy_process_group()


# ===================================================================== gold fixtures =====

T, G, PROMPT, N_GOLD = 12, 4, 4, 5


def gold_traj(reward: float, gold_len: int, seed: int = 0) -> dict:
    """One GRPO group as ``prepare_batch`` hands it over, with or without a usable gold.

    Args:
        reward: The raw reward every rollout in the group scored. 0.0 makes the group
            all-wrong and therefore a DyME candidate; 1.0 makes it solved.
        gold_len: Gold tokens the dataset row supplied. ZERO is not an error state -- it is
            what ``attach_gold_from_data`` writes for a row with no solution AND for a gold
            longer than the realised width, which ``FINDINGS_gold_path.md`` section 4
            measures at between 1% and 62% of MATH rows depending on that width.
        seed: Seeds the token ids.

    Returns:
        The trajectory dict, shaped as one element of the trainer's ``rollout_batch`` list.
    """
    g = torch.Generator().manual_seed(seed)
    loss_mask = torch.zeros(G, T, dtype=torch.long)
    loss_mask[:, PROMPT:] = 1
    gold_mask = torch.zeros(G, T, dtype=torch.int32)
    gold_ids = torch.zeros(G, T, dtype=torch.long)
    if gold_len:
        gold_mask[:, :gold_len] = 1
        gold_ids[:, :gold_len] = torch.arange(1, gold_len + 1)
    return {
        "input_ids": torch.randint(1, 50, (G, T), generator=g),
        "attention_mask": torch.ones(G, T, dtype=torch.long),
        "loss_mask": loss_mask,
        "logprobs": torch.full((G, T), -0.5),
        "rewards": torch.full((G,), float(reward)),
        "gold_ids": gold_ids,
        "gold_mask": gold_mask,
    }


# ============================================================ CONFIRMED DEFECTS ==========


def test_defect_a_roster_of_experts_with_no_plan_armed_trains_one_and_says_nothing(
    world, tmp_path
):
    """A configured cluster-LoRA arm that never arms a partition IS the ``none`` arm.

    ``SELFEVO_CLUSTER_LORA_ADAPTERS`` creates the experts inside ``_apply_peft_wrapper``.
    ``SELFEVO_CLUSTER_LORA`` arms the partition -- but only from inside
    ``PPOActor._route_groups``, which is itself reached only when ``group_routing.enabled``
    is true AND ``group_routing.router`` is set. Nothing cross-checks the two. So a run with
    both variables set and ``group_routing`` left alone creates every expert, trains exactly
    the first one for the whole batch, emits NO ``cluster_lora/*`` key at all, and merges
    N-1 experts still at their LoRA init -- reproducing the shared-LoRA baseline bit for bit
    while the config says ``partition=meds``.

    That is the failure ``PartitionUnavailable``, ``ClusterWiringError`` and
    ``AdapterIsolationError`` all exist to refuse one seam earlier, arrived at through the
    one seam that has no guard. ``test_an_unarmed_engine_never_looks_for_a_plan`` deletes
    ``_selfevo_adapters`` before it runs, so this state is not covered by it.

    FIXED 2026-09-02, and the marker is gone with the fix. ``FSDPEngine.train_batch`` now
    refuses when ``self._selfevo_adapters`` exists and no plan is armed, exactly as the
    mirrored case (a plan armed with no adapter set) already did. The assertion below is the
    refusal rather than the two measurements above, because a step that refuses returns no
    stats to inspect -- and a run that cannot start is the outcome asked for: there is no
    partition for a dashboard to report, only a configuration that has to be fixed before
    the accelerators are spent. What survives from the measurement is the second assertion:
    no expert may have moved, since a refusal taken after the optimizer step would train
    ``cluster_0`` alone and still raise.

    ``test_an_unarmed_engine_never_looks_for_a_plan`` covers the OTHER state -- no roster and
    no plan, the rollback path -- and ``test_a_roster_of_experts_with_no_plan_armed_is_refused``
    in that same module now covers this one, so the gap between them is closed at the source
    as well as here.
    """
    from selfevo.cluster_lora.wiring import ClusterWiringError

    engine, adapters = make_engine(tmp_path)
    before = {n: adapters.snapshot(n) for n in NAMES}
    with pytest.raises(ClusterWiringError, match="no partition is armed"):
        engine.train_batch(
            make_batch(seed=0), loss_fn=linear_loss, loss_weight_fn=weight_fn
        )
    moved = [n for n in NAMES if not adapters.unchanged(n, before[n])]
    assert not moved, (
        f"the engine refused and {moved} moved anyway, so the refusal is downstream of the "
        "optimizer step and the arm still trained one expert on the whole batch"
    )


def test_defect_the_actor_arms_nothing_when_group_routing_is_off(monkeypatch):
    """The reachability of the defect above, at the seam that is supposed to refuse.

    ``begin_cluster_batch`` refuses a router that cannot carry a partition. It never runs,
    because the branch that calls it lives inside ``_route_groups`` and ``_route_groups`` is
    called only under ``group_routing.enabled and group_routing.router``. With the master
    switch set and group routing left at its default the actor computes advantages, arms
    nothing, and returns without a word.

    FIXED 2026-09-02, and the marker is gone with the fix. The switch is now read outside
    the routing branch, into ``PPOActor._arm_unrouted_cluster_batch``. ``partition=none`` is
    the vanilla shared-LoRA arm expressed in the method's type and needs neither features nor
    a router, so it is ARMED there -- every group on the shared adapter, which is what that
    arm means -- and the baseline runs the identical engine path as the method. ``meds`` and
    ``random_matched`` need vectors that reach the partitioner only through the routing seam,
    so those are REFUSED; both cases are asserted in ``test_cluster_lora_wired.py``.

    The mirrored guard in the engine, for a roster armed with no plan at all, is the subject
    of the test above and is still open.
    """
    from selfevo.cluster_lora import wiring
    from selfevo.tests.test_cluster_lora_wired import FakeEngine
    from selfevo.tests.test_group_routing import MIXED, make_actor, make_batch, meta

    monkeypatch.setenv(wiring.CLUSTER_LORA_ENV, "none")
    monkeypatch.setenv("SELFEVO_CLUSTER_LORA_ADAPTERS", "cluster_0,cluster_1,shared")
    actor = make_actor(GroupRoutingConfig(enabled=False))
    engine = FakeEngine()
    actor.engine = engine
    actor._compute_advantages(make_batch(MIXED), meta())
    assert getattr(engine, "_selfevo_cluster_plan", None) is not None, (
        "SELFEVO_CLUSTER_LORA and the expert roster are both set and the actor armed no "
        "partition, because group_routing.enabled is false. Neither seam refused, so the "
        "run is the none arm under the method name"
    )


def test_defect_a_plan_armed_for_one_batch_is_silently_reused_by_the_next(world, tmp_path):
    """``ClusterPlan.step`` exists so a stale plan is identifiable. Nothing identifies it.

    The engine reads ``_selfevo_cluster_plan`` and never clears it, and never compares its
    ``step`` to anything. A step on which the actor does not re-arm therefore trains the new
    batch under the PREVIOUS batch group-to-expert assignment. ``rows_by_adapter`` cannot
    catch it: group ids are ``0..G-1`` in every batch, so a stale plan names every group of
    the new batch and looks complete. The only observable is ``cluster_lora/plan_step``,
    which is emitted and read by nobody.

    FIXED 2026-09-02, and the marker is gone with the fix. The batch identity is carried,
    not the plan consumed: ``begin_cluster_batch`` stamps every row of the batch it armed
    for with the same number the plan records as its ``step``, and
    ``assert_plan_describes_batch`` refuses a mismatch, which finally gives ``plan_step`` the
    consumer its docstring promised.

    POPPING WOULD HAVE BEEN WRONG, which is why the assertion below is no longer that the
    plan is ``None``. ``PPOActor._ppo_update`` calls ``train_batch`` once per PPO MINIBATCH,
    so one arming is legitimately used ``ppo_n_minibatches`` times; a pop would leave every
    minibatch after the first unrouted. Both halves are asserted here -- the same plan serves
    a second minibatch of ITS batch, and refuses the next batch -- because a fix that only
    refused would break the live update loop and a fix that only accepted would be the
    defect.
    """
    engine, _adapters = make_engine(tmp_path)
    engine._selfevo_cluster_plan = plan_of(SPLIT, step=0)
    for _ in range(2):
        armed = make_batch(seed=0)
        tag_cluster_batch(armed, 0)
        engine.train_batch(armed, loss_fn=linear_loss, loss_weight_fn=weight_fn)
    assert getattr(engine, "_selfevo_cluster_plan", None) is not None, (
        "the plan was consumed, so every PPO minibatch after the first is unrouted"
    )
    nxt = make_batch(seed=1)
    tag_cluster_batch(nxt, 1)
    with pytest.raises(ClusterWiringError, match="armed for batch 0"):
        engine.train_batch(nxt, loss_fn=linear_loss, loss_weight_fn=weight_fn)


@pytest.mark.xfail(
    strict=True,
    reason=(
    "CONFIRMED DEFECT 3: the empty-gold guard fires for groups that never qualified, so an "
    "all-solved batch is refused. Fix: scope it to qualifying groups and raise the "
    "no-gold-anywhere refusal from substitute_in_place "
    ),
)
def test_defect_an_all_solved_batch_with_one_goldless_trajectory_is_refused():
    """A batch that needed no gold must not fail. It does, if one row has no gold text.

    ``FINDINGS_gold_path.md`` section 4 pins this exactly: no group qualified means NO
    refusal, counts returned as zero, a batch that needed no gold. But the empty-gold guard
    inside ``substitute_gold_rows`` fires on ``gold_mask.sum() == 0`` for the trajectory
    REGARDLESS of whether that group qualified, ``substitute_in_place`` defers the refusal,
    and the batch-level ``if not stats.rows_substituted`` then re-raises it -- which is
    precisely the state an all-solved batch is in.

    An empty ``gold_mask`` is not an error state. ``attach_gold_from_data`` writes one
    deliberately for a gold longer than the realised row, which the same document measures
    at up to 62 percent of MATH rows at narrow widths. So this refusal is the common case,
    and it kills the training step. The message is wrong as well: it says every gold_mask in
    this batch is empty when one of three was.

    FAILS ON CURRENT CODE. The fix: scope the empty-gold guard to groups that QUALIFIED, and
    raise the no-gold-anywhere refusal from ``substitute_in_place``, which is the only level
    at which anywhere is knowable.
    """
    batch = [gold_traj(1.0, 5, 0), gold_traj(1.0, 0, 1), gold_traj(1.0, 5, 2)]
    _out, stats = substitute_in_place(batch, "dyme")
    assert stats.groups_qualifying == 0
    assert stats.rows_substituted == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
    "CONFIRMED DEFECT 4: the deferred-refusal path returns the original trajectory without "
    "is_gold, so concat_padded_tensors refuses the batch. Fix: attach the all-zero is_gold on "
    "every non-none return path "
    ),
)
def test_defect_a_deferred_refusal_drops_is_gold_and_the_batch_stops_collating():
    """The recovery path returns a trajectory with no ``is_gold``, so the key sets disagree.

    ``substitute_in_place`` catches a per-trajectory ``GoldMissingError`` and substitutes
    ``dict(traj)`` -- the ORIGINAL, which never acquired ``is_gold``. Every other element
    has it. ``concat_padded_tensors`` refuses a list whose dicts disagree on keys, so the
    step dies at collation with a message naming neither gold nor the substitution, which is
    the exact failure shape ``selfevo/gold/attach.py`` own docstring says the padding exists
    to avoid.

    Reached by one solved group whose dataset row carried no usable gold, beside one
    all-wrong group that was served -- an ordinary MATH batch.

    FAILS ON CURRENT CODE. The fix: attach the all-zero ``is_gold`` on every non-``none``
    return path, including the refusal path. ``substitute_gold_rows`` already builds the
    tensor before it raises, and ``substitute_in_place`` can take it off the refusal.
    """
    from areal.utils.data import concat_batch

    out, _stats = substitute_in_place([gold_traj(0.0, 5, 0), gold_traj(1.0, 0, 1)], "dyme")
    keys = [sorted(d) for d in out]
    assert keys[0] == keys[1], (
        f"element 0 carries {set(keys[0]) - set(keys[1])} and element 1 does not; "
        "concat_padded_tensors refuses a batch whose dicts disagree on keys"
    )
    concat_batch(out)


@pytest.mark.xfail(
    strict=True,
    reason=(
    "CONFIRMED DEFECT 5: reconcile_gold_logprobs must run before _compute_advantages and does "
    "not guard it, so every gold token silently adopts its predecessor's logprob. Fix: refuse "
    "a batch carrying advantages/returns/gen_mask/group_ids "
    ),
)
def test_defect_reconciling_after_the_advantage_computation_is_not_refused():
    """The second gold seam has an ordering requirement and no ordering guard.

    ``substitute_gold_rows`` refuses a batch that has been through ``compute_logp``.
    ``reconcile_gold_logprobs`` has the symmetric requirement -- it must run BEFORE
    ``_compute_advantages``, because that method rolls ``loss_mask`` and ``logprobs`` LEFT
    into emitter coordinates and this function reads ``loss_mask`` to locate the gold. Run
    it afterwards and every gold token adopts its PREDECESSOR log-probability: the
    importance ratio becomes ``exp(prox[t] - prox[t-1])`` instead of exactly 1, over exactly
    the rows the arm exists to add.

    Nothing raises. ``assert_gold_logprobs_filled`` passes, because the shifted values are
    still finite and still negative -- it checks the SIGN of the number, not the position it
    was written to. The reconciliation even reports the row as successfully filled.

    FAILS ON CURRENT CODE. The fix is one guard of the same kind the first seam already has:
    refuse a batch carrying ``advantages``/``returns``/``gen_mask``/``group_ids``, all of
    which ``_compute_advantages`` writes and none of which exists before it.
    """
    batch = gold_traj(0.0, N_GOLD, 3)
    out, _st = substitute_gold_rows(batch, "dyme")
    out["prox_logp"] = torch.arange(G * T, dtype=torch.float32).reshape(G, T) * -0.01

    inverted = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in out.items()}
    inverted["loss_mask"] = torch.roll(out["loss_mask"].float(), shifts=-1, dims=-1)
    inverted["logprobs"] = (
        torch.roll(out["logprobs"], shifts=-1, dims=-1) * inverted["loss_mask"]
    )
    inverted["advantages"] = torch.zeros(G, T)
    with pytest.raises(GoldOrderingError):
        reconcile_gold_logprobs(inverted)


def test_defect_the_size_matched_control_also_destroys_expert_identity():
    """The mandatory control differs from the method on TWO axes, and one of them is fatal.

    ``random_matched_partition`` re-permutes the labels every batch under a fresh seed
    (``ClusterLoRAKeyFn.begin_batch`` passes ``seed + self.batches``). On a partition whose
    membership never changes, the method churn is 0.0 and the control churn is 0.67 to 0.88.

    ``FINDINGS_cluster_lora.md`` section 5.1 establishes that churn is not a nuisance
    parameter here: at churn 1.0 every expert receives a different subpopulation each time
    and learns the average anyway, the exact thing the method claims to avoid, which is why
    ``MEDSPartitioner._resync`` exists at all. So a control built this way is not the method
    without clustering, it is the method without clustering AND without stable expert
    identity, and a gain over it cannot be attributed to the clustering.

    The existing tests assert the control sizes match exactly and that it is feature-blind
    over 50 seeds. Both pass. Neither looks at stability, so both pass for the wrong reason.

    FIXED 2026-09-02, and the marker is gone with the fix. The control assignment is now
    drawn ONCE per group IDENTITY and carried forward on
    ``partition.MatchedControlMemory``, the way ``_resync`` carries the method, and the
    groups the control has not seen are filled from the labels the seeded draw has left --
    so the size match is preserved by REARRANGING the draw rather than by a second one.
    """
    from selfevo.cluster_lora.partition import DEFAULT_CONTROL_MEMORY

    # The default memory is process-global, so the measurement is taken against a known
    # empty one rather than against whatever an earlier test in this session left in it.
    DEFAULT_CONTROL_MEMORY.reset()
    ids = tuple(f"p{i}" for i in range(24))
    labels = tuple(i // 6 for i in range(24))
    reference = [
        Partition(labels=labels, basis="stable fixture", group_ids=ids) for _ in range(3)
    ]
    control = [random_matched_partition(reference[k], seed=k) for k in range(3)]

    method_churn = [label_churn(reference[k - 1], reference[k])[0] for k in (1, 2)]
    control_churn = [label_churn(control[k - 1], control[k])[0] for k in (1, 2)]
    assert max(method_churn) == 0.0, "the fixture is not stable, so it proves nothing"
    assert max(control_churn) <= 0.25, (
        f"the control churns {control_churn} per step against the method {method_churn} on "
        "a partition whose membership never changed. The control differs from the arm in "
        "expert STABILITY as well as in feature-blindness"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
    "CONFIRMED DEFECT 7: only() restores names[0], so compute_logp, eval_batch, "
    "behaviour_features and the weight sync all run on one expert rather than the sum. Fix: "
    "activate a merged deployment adapter before any forward outside step "
    ),
)
def test_defect_a_forward_outside_the_step_sees_exactly_one_expert(world, tmp_path):
    """After a routed step the model carries one active adapter, not the sum of the experts.

    ``ClusterAdapterSet.only`` restores whatever was active when it was entered, which is
    ``names[0]``. Every forward that is NOT inside ``ClusterAdapterSet.step`` therefore runs
    on ONE expert: ``PPOActor.compute_logp`` -- whose ``prox_logp`` is the denominator of
    the importance ratio for every group, including the groups trained on a different expert
    -- ``eval_batch``, the ``behaviour_features`` extra forward that decides the clustering,
    and whatever reads weights for the inference engine. ``merge.merge_sum`` builds the
    summed adapter that is supposed to be deployed, and nothing in the training loop calls
    it between steps.

    So the ratio the loss computes compares two different models, the behavioural feature
    that decides the clustering is read from a model no group but cluster_0 is trained
    under, and the rollouts that generate the next batch come from that same one expert.

    FAILS ON CURRENT CODE. The fix: make the non-training state explicit -- either activate
    a summed deployment adapter (``merge_sum`` refreshed each step) before any forward
    outside ``step``, or refuse; but do not leave it to whichever adapter happened to be
    active when the last cluster finished.

    NOTE peft 0.18.1 ``PeftModel.set_adapter`` takes a single name and raises ``TypeError``
    on a list, so ``only()`` own multi-adapter restore branch cannot run either; that is
    pinned by ``test_refuted_peft_set_adapter_takes_one_name`` below.
    """
    engine, _adapters = make_engine(tmp_path)
    for step in range(3):
        engine._selfevo_cluster_plan = plan_of(SPLIT, step=step)
        data = make_batch(seed=step)
        tag_cluster_batch(data, step)
        engine.train_batch(data, loss_fn=linear_loss, loss_weight_fn=weight_fn)
    active = list(engine.model.active_adapters)
    assert set(active) == set(NAMES), (
        f"after a routed step the active adapters are {active}; every forward outside "
        "ClusterAdapterSet.step -- compute_logp, eval_batch, the behavioural feature pass, "
        "the weight upload -- therefore runs on that one expert rather than on the sum the "
        "method deploys"
    )


# ============================================================ REFUTED HYPOTHESES =========


def test_refuted_the_gold_coordinate_roll_survives_the_actors_own_left_roll():
    """HYPOTHESIS: ``reconcile_gold_logprobs`` rolls the wrong way. REFUTED.

    Written independently of the authors own round-trip test and driven from the arithmetic
    ``_compute_advantages`` actually performs on the branch the live config takes
    (``use_decoupled_loss`` gives ``old_logp = roll(data['logprobs'], -1)``). Writing
    ``roll(prox, +1)`` at TOKEN positions is exactly what makes ``old_logp`` equal ``prox``
    at the gold EMITTER positions, so the importance ratio is exactly 1 there.

    Kept because the unshifted write is the edit a reader would make, and it changes no
    shape and raises nothing.
    """
    batch = gold_traj(0.0, N_GOLD, 7)
    out, _st = substitute_gold_rows(batch, "dyme")
    # Strictly negative and all distinct, so a one-position slip cannot coincide with the
    # right answer, and so the value is a possible log-probability -- a positive prox_logp
    # is correctly refused by assert_gold_logprobs_filled and would make this a test of the
    # fixture rather than of the roll.
    prox = -torch.arange(1, G * T + 1, dtype=torch.float32).reshape(G, T) * 0.01
    out["prox_logp"] = prox
    filled, n_rows = reconcile_gold_logprobs(out)
    assert n_rows == 1

    old_logp = torch.roll(filled["logprobs"], shifts=-1, dims=-1)
    emitter = torch.roll(out["loss_mask"].float(), shifts=-1, dims=-1).bool()[0].clone()
    emitter[-1] = False
    gold_emitter = emitter & torch.roll(out["is_gold"].bool()[0], shifts=-1, dims=-1)
    assert bool(gold_emitter.any())
    assert torch.allclose(old_logp[0][gold_emitter], prox[0][gold_emitter]), (
        "after the actor own left roll the gold behaviour log-probability must equal "
        "prox_logp exactly, or the importance ratio on the gold row is not 1"
    )
    unshifted = torch.where(
        out["is_gold"].bool() & out["loss_mask"].bool(), prox, out["logprobs"]
    )
    assert not torch.allclose(
        torch.roll(unshifted, shifts=-1, dims=-1)[0][gold_emitter], prox[0][gold_emitter]
    ), "the anti-vacuity clause: an unshifted write must NOT satisfy the assertion above"


def test_refuted_the_cluster_denominator_is_not_rescaled_by_cluster_size(world, tmp_path):
    """HYPOTHESIS: a small cluster gets a large effective learning rate. REFUTED.

    ``_train_batch_by_cluster`` computes ``total_loss_weight`` over the WHOLE batch once,
    before splitting, and hands the same scalar to every cluster microbatches. Checked here
    on a lopsided partition -- six rows on one expert and two on another -- by comparing the
    scalar against the unrouted step.
    """
    import areal.engine.fsdp_engine as fe

    seen: list[float] = []
    original = fe.compute_total_loss_weight

    def spy(mb_list, loss_weight_fn, dp_group, device=None):
        """Record the value every reduction returns."""
        value = original(mb_list, loss_weight_fn, dp_group, device=device)
        seen.append(float(value))
        return value

    fe.compute_total_loss_weight = spy
    try:
        plain, _ = make_engine(tmp_path, seed=21)
        # The unrouted step is now refused on an engine that carries a roster, which is the
        # misconfigured arm; the comparison here is against a run with no cluster-LoRA at
        # all, so the adapter set goes with the plan.
        del plain._selfevo_adapters
        plain.train_batch(
            make_batch(seed=4), loss_fn=linear_loss, loss_weight_fn=weight_fn
        )
        lopsided, _ = make_engine(tmp_path, seed=21)
        lopsided._selfevo_cluster_plan = plan_of(
            {0: "cluster_0", 1: "cluster_0", 2: "cluster_0", 3: "cluster_1"}
        )
        routed = make_batch(seed=4)
        tag_cluster_batch(routed, 0)
        lopsided.train_batch(routed, loss_fn=linear_loss, loss_weight_fn=weight_fn)
    finally:
        fe.compute_total_loss_weight = original
    assert len(seen) == 2, f"the denominator was taken {len(seen)} times, not once per step"
    assert seen[0] == seen[1]


def test_refuted_a_stale_plan_that_names_too_few_groups_still_refuses(world, tmp_path):
    """HYPOTHESIS: a stale plan silently drops rows. REFUTED, but only in one direction.

    ``rows_by_adapter`` raises when a row group is not named, so a plan from a SMALLER batch
    is caught. It is the other direction -- a plan from a batch with as many groups or more
    -- that passes silently, which is what
    ``test_defect_a_plan_armed_for_one_batch_is_silently_reused_by_the_next`` is about. Kept
    so that a future default-the-missing-group-to-shared cannot be added quietly.
    """
    from selfevo.cluster_lora.wiring import ClusterWiringError

    engine, _ = make_engine(tmp_path)
    engine._selfevo_cluster_plan = plan_of({0: "cluster_0", 1: "cluster_0"})
    short = make_batch(seed=6)
    tag_cluster_batch(short, 0)
    with pytest.raises(ClusterWiringError, match="does not name"):
        engine.train_batch(short, loss_fn=linear_loss, loss_weight_fn=weight_fn)


def test_refuted_peft_set_adapter_takes_one_name(world, tmp_path):
    """HYPOTHESIS: ``only()`` can restore a multi-adapter state. REFUTED -- it would raise.

    ``ClusterAdapterSet.only`` ends with ``set_adapter(previous[0] if len(previous) == 1
    else previous)``. On peft 0.18.1 ``PeftModel.set_adapter`` takes a single name and does
    ``if adapter_name not in self.peft_config``, which raises ``TypeError`` on a list. The
    branch is unreachable today only because nothing ever activates more than one adapter --
    which is itself the subject of
    ``test_defect_a_forward_outside_the_step_sees_exactly_one_expert``. Pinned so that
    switching the deployment forward to a multi-active state fails here rather than inside a
    finally block on a live run.
    """
    engine, _adapters = make_engine(tmp_path)
    with pytest.raises(TypeError):
        engine.model.set_adapter(list(NAMES))


def test_refuted_the_off_arm_of_the_gold_path_is_still_inert():
    """HYPOTHESIS: the empty-gold refusal reaches the ``none`` arm too. REFUTED.

    ``rule='none'`` returns before every guard, so a batch of gold-less trajectories is
    returned untouched with zero counts and no ``is_gold`` key. The defect above is confined
    to a CONFIGURED gold arm, which is what makes it a defect rather than a rollback failure.
    """
    batch = [gold_traj(1.0, 0, 0), gold_traj(0.0, 0, 1)]
    out, stats = substitute_in_place(batch, "none")
    assert stats.rows_substituted == 0
    assert all("is_gold" not in d for d in out)
    for original, returned in zip(batch, out):
        for key, value in original.items():
            assert returned[key] is value, f"the off arm copied {key}"


def test_refuted_the_sentinel_guard_is_keyed_on_the_same_tensor_as_the_repair():
    """HYPOTHESIS: the last-line guard can be defeated. TRUE, but no path was found to it.

    ``assert_gold_logprobs_filled`` returns clean on a batch that still carries
    ``GOLD_LOGP_SENTINEL`` in ``logprobs`` as soon as ``is_gold`` is absent, and
    ``reconcile_gold_logprobs`` returns ``(batch, 0)`` on the same batch. Both the repair and
    the check are keyed on ``is_gold``, so one lost key disables both at once and the audit
    measured consequence -- the row silently multiplied by ``exp(prox_logp - 1)`` -- returns
    with no metric moving.

    Recorded as a latent weakness rather than a defect because no stage between the two
    seams was found that drops the key: the deferred-refusal path drops it only on rows that
    were never rewritten and therefore carry no sentinel. A sentinel is a VALUE, and a guard
    on a value does not need a second tensor to be present in order to fire; making
    ``assert_gold_logprobs_filled`` check ``logprobs > 0`` over the whole batch would cost
    nothing and would not be defeatable this way.
    """
    batch = gold_traj(0.0, N_GOLD, 11)
    out, _st = substitute_gold_rows(batch, "dyme")
    assert bool((out["logprobs"] == GOLD_LOGP_SENTINEL).any())
    with pytest.raises(GoldOrderingError):
        assert_gold_logprobs_filled(out)

    stripped = {k: v for k, v in out.items() if k != "is_gold"}
    assert_gold_logprobs_filled(stripped)
    again, n_rows = reconcile_gold_logprobs(stripped)
    assert n_rows == 0
    assert bool((again["logprobs"] == GOLD_LOGP_SENTINEL).any()), (
        "the sentinel survived both the repair and the guard, silently"
    )


def test_refuted_the_dyme_predicate_reads_the_raw_reward_not_the_normalised_one():
    """HYPOTHESIS: the DyME predicate can fire on a group containing a correct rollout.

    REFUTED: ``_qualifies`` counts ``(rewards > 0.5).sum() == 0`` on the RAW rewards, and
    ``original_rewards`` is preferred over ``rewards`` when present, so a reward-normalised
    batch does not change the predicate. Checked with a normalisation applied to ``rewards``
    that would flip a naive threshold in both directions.
    """
    solved = gold_traj(0.0, N_GOLD, 13)
    solved["rewards"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    solved["original_rewards"] = solved["rewards"].clone()
    _out, stats = substitute_gold_rows(solved, "dyme")
    assert stats.groups_qualifying == 0

    normalised = gold_traj(0.0, N_GOLD, 13)
    normalised["original_rewards"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    normalised["rewards"] = torch.tensor([1.5, -0.5, -0.5, -0.5])
    _out2, stats2 = substitute_gold_rows(normalised, "dyme")
    assert stats2.groups_qualifying == 0, (
        "the predicate read the normalised rewards rather than the raw ones"
    )

    all_wrong = gold_traj(0.0, N_GOLD, 14)
    all_wrong["original_rewards"] = torch.zeros(G)
    all_wrong["rewards"] = torch.tensor([0.9, -0.3, -0.3, -0.3])
    _out3, stats3 = substitute_gold_rows(all_wrong, "dyme")
    assert stats3.groups_qualifying == 1, (
        "an all-wrong group whose normalised rewards contain a positive entry must still "
        "qualify, or reward normalisation silently switches the arm off"
    )
