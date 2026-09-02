"""The two call sites that turn the cluster-LoRA mechanism into a training arm.

Everything this module drives was built and tested elsewhere in
:mod:`selfevo.cluster_lora`; nothing here re-derives any of it. What was missing was reach:
``FINDINGS_cluster_lora.md`` section 9 records two seams that were satisfied and tested but
had no caller, so ``cluster_lora.partition=meds`` could only ever REFUSE.

seam 1 -- ``PPOActor._route_groups`` builds ``RoutingContext.extra`` from
``group_features`` only, and ``extra`` is ``Mapping[str, float]`` and cannot carry a vector.
:func:`begin_cluster_batch` is the one call that supplies the behavioural vectors, through
``ClusterLoRAKeyFn.begin_batch``, and arms the engine with the partition that comes back.

seam 2 -- ``FSDPEngine.train_batch`` ran one adapter for the whole batch.
:class:`ClusterPlan` plus :func:`rows_by_adapter` turn a per-GROUP partition into the
per-cluster row sets the engine microbatches over, and :class:`EngineOptimizer` lets
``ClusterAdapterSet.step`` -- the tested isolation guard -- drive the engine's own clipping,
scheduler and stats rather than a second optimizer step written here.

**Default off, and the gate is one environment variable.** ``SELFEVO_CLUSTER_LORA`` unset or
empty means every call site short-circuits before importing this module, so a run that does
not ask for cluster-LoRA is bit-identical to one from before it existed. The variable is read
rather than added to ``GroupRoutingConfig`` for the same reason ``router=random`` reads
``SELFEVO_RANDOM_PROPORTIONS``: ``_route_groups`` builds routers with ``factory()`` and no
kwargs, and ``areal/api/cli_args.py`` is not this agent's file.

WHAT THE EXTRA FORWARD COSTS. The MEDS feature is not reachable from anything the training
forward already produces (section 1 of the findings), so ``features=1`` buys it with one
extra no-grad forward per SAMPLE, truncated at the answer token. That is off by default and
must be counted against the matched-budget baseline in ``experiments/m25/PLAN.md``; this
module emits ``cluster_lora/feature_seconds`` every batch so the count comes from the run
itself rather than from an estimate.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "CLUSTER_LORA_ENV",
    "ClusterLoRAConfig",
    "ClusterPlan",
    "EngineOptimizer",
    "adapter_roster",
    "begin_cluster_batch",
    "behaviour_features",
    "every_expert",
    "rows_by_adapter",
    "select_rows",
]

#: Master switch AND partition selector: ``meds | random_matched | none``. Unset or empty is
#: OFF. Inlined as a literal at both call sites so that the default path imports nothing from
#: selfevo at all; ``test_cluster_lora_wired.py`` asserts the literal and this constant agree.
CLUSTER_LORA_ENV = "SELFEVO_CLUSTER_LORA"

#: Adapter names the engine creates up front, comma separated. Required by the engine seam.
ROSTER_ENV = "SELFEVO_CLUSTER_LORA_ADAPTERS"


class ClusterWiringError(RuntimeError):
    """A cluster-LoRA arm was configured but cannot run as configured.

    Raised, never downgraded to a warning or a fallback. Every case this covers -- a router
    that cannot carry a partition, a partition naming an adapter the model does not have,
    features asked for with no model to compute them on -- has the same silent form if it is
    absorbed: the run trains, logs, and reports as the method while being the ``none`` arm
    wearing its label. That is the failure ``PartitionUnavailable`` already refuses upstream,
    and this is the same refusal at the call sites.
    """


@dataclass(frozen=True)
class ClusterLoRAConfig:
    """How a run's cluster-LoRA arm is configured, read from the environment.

    Args:
        partition: ``meds``, ``random_matched`` or ``none``, passed to
            ``partition_from_config`` unchanged. ``none`` is the vanilla shared-LoRA arm
            expressed in the same type, so the baseline runs the identical code path.
        features: Whether to spend the extra forward that produces the behavioural vectors.
            False is the default and makes ``meds`` and ``random_matched`` REFUSE, which is
            the intended behaviour of a mode whose input is absent -- see
            ``partition_from_config``.
        seed: Seed for the size-matched control.
        min_cluster_size: HDBSCAN's, which MEDS ships at 2 and which was measured
            over-fragmenting (findings 5.2). Swept, not inherited.
        warmup_batches: Batches buffered before the first fit.
        answer_strategy: ``boxed`` (MEDS' own) or ``last``. ``boxed`` needs the token ids of
            ``"\\boxed{"``, which are tokenizer-specific and are read from the engine's
            tokenizer when one is present.
        extractor_mode: ``hooks`` (one position per layer) or ``hidden_states`` (the
            MEDS-faithful reference, which retains the whole activation stack).
        use_layer_diff: MEDS' own post-processing switch; shipped off.
        last_n_layers: Trailing layers kept. ``None`` keeps the latter half, which is the
            rule MEDS' constant expresses.

    Raises:
        ValueError: On an unknown partition or a non-numeric numeric field. A misspelled
            ``partition=med`` silently becoming the baseline is exactly what this refuses.
    """

    partition: str
    features: bool = False
    seed: int = 0
    min_cluster_size: int = 2
    warmup_batches: int = 1
    answer_strategy: str = "last"
    extractor_mode: str = "hooks"
    use_layer_diff: bool = False
    last_n_layers: int | None = None

    def __post_init__(self) -> None:
        from .partition import TRAINING_PARTITIONS

        if self.partition not in TRAINING_PARTITIONS:
            raise ValueError(
                f"{CLUSTER_LORA_ENV}={self.partition!r} is not a training partition; "
                f"expected one of {list(TRAINING_PARTITIONS)}"
            )
        if self.answer_strategy not in ("boxed", "last"):
            raise ValueError(
                f"answer_strategy must be 'boxed' or 'last', got {self.answer_strategy!r}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ClusterLoRAConfig | None":
        """Read the arm's configuration, or ``None`` when no arm is configured.

        Args:
            env: Environment to read. Defaults to ``os.environ``; passed explicitly by the
                tests so the process environment is never mutated.

        Returns:
            The configuration, or ``None`` if :data:`CLUSTER_LORA_ENV` is unset or empty --
            the default, under which every call site is a no-op.

        Raises:
            ValueError: If a value is set but unusable.
        """
        env = os.environ if env is None else env
        partition = (env.get(CLUSTER_LORA_ENV) or "").strip()
        if not partition:
            return None

        def _int(name: str, default: int) -> int:
            raw = (env.get(name) or "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name}={raw!r} is not an integer") from exc

        last_n = (env.get("SELFEVO_CLUSTER_LORA_LAST_N_LAYERS") or "").strip()
        return cls(
            partition=partition,
            # Anything but an explicit 0/false/no turns the extra forward ON, but the
            # DEFAULT is off: an unset variable is not a request for a second forward pass.
            features=(env.get("SELFEVO_CLUSTER_LORA_FEATURES") or "0").strip().lower()
            not in ("", "0", "false", "no"),
            seed=_int("SELFEVO_CLUSTER_LORA_SEED", 0),
            min_cluster_size=_int("SELFEVO_CLUSTER_LORA_MIN_CLUSTER_SIZE", 2),
            warmup_batches=_int("SELFEVO_CLUSTER_LORA_WARMUP", 1),
            answer_strategy=(env.get("SELFEVO_CLUSTER_LORA_ANSWER") or "last").strip(),
            extractor_mode=(env.get("SELFEVO_CLUSTER_LORA_EXTRACTOR") or "hooks").strip(),
            use_layer_diff=(env.get("SELFEVO_CLUSTER_LORA_LAYER_DIFF") or "0").strip()
            not in ("", "0", "false", "no"),
            last_n_layers=int(last_n) if last_n else None,
        )


def adapter_roster(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Adapter names the engine must create up front, or ``()`` when none are configured.

    Every expert is created before FSDP sharding and before the optimizer exists, so that
    each one is sharded and optimised identically and each has the same number of steps
    behind it. Creating one later would give a cluster a freshly-initialised expert half way
    through a run -- which trains, but is not the run that was configured.

    Args:
        env: Environment to read. Defaults to ``os.environ``.

    Returns:
        The names, in order, with duplicates refused.

    Raises:
        ValueError: On a duplicate name. ``["cluster_0", "cluster_0"]`` would look like a
            two-expert roster and be one expert.
    """
    env = os.environ if env is None else env
    raw = (env.get(ROSTER_ENV) or "").strip()
    if not raw:
        return ()
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    if len(set(names)) != len(names):
        raise ValueError(
            f"{ROSTER_ENV}={raw!r} repeats an adapter name; a repeated name is one expert "
            "wearing a two-expert roster's label"
        )
    return names


@dataclass(frozen=True)
class ClusterPlan:
    """This batch's group -> adapter assignment, as the engine consumes it.

    Args:
        key_of_group: Adapter name per GROUP INDEX, where the index is the value carried in
            ``data["group_ids"]``. Keyed on that rather than on row position because
            microbatch splitting reorders rows for balanced packing -- ``group_ids`` is a
            per-token tensor precisely so it survives that reordering, and a plan keyed on
            positions would hand each row another row's expert.
        step: The training step that formed it, so a stale plan is identifiable.
        basis: The partition's own record of what it rested on.

    Raises:
        ValueError: If the plan is empty. An empty plan would route nothing and skip every
            expert while reporting a step.
    """

    key_of_group: Mapping[int, str]
    step: int
    basis: str

    def __post_init__(self) -> None:
        if not self.key_of_group:
            raise ValueError("a cluster plan with no groups routes nothing")


def rows_by_adapter(
    group_ids: Sequence[int], key_of_group: Mapping[int, str]
) -> dict[str, list[int]]:
    """Row indices per adapter, from the per-row group ids the batch carries.

    Args:
        group_ids: Group index per ROW of the batch, in the batch's current order.
        key_of_group: Adapter name per group index.

    Returns:
        ``{adapter_name: row indices}`` in first-appearance order. Every row appears once.

    Raises:
        ClusterWiringError: If a row's group is not named by the plan. Dropping that row
            would train on less than the batch while reporting the whole batch's loss
            weight, and assigning it a default expert would be a partition decided by a
            missing lookup.
    """
    out: dict[str, list[int]] = {}
    for row, gid in enumerate(group_ids):
        key = key_of_group.get(int(gid))
        if key is None:
            raise ClusterWiringError(
                f"row {row} belongs to group {int(gid)}, which this batch's partition does "
                f"not name (it names {sorted(key_of_group)}); the plan does not describe "
                "this batch"
            )
        out.setdefault(key, []).append(row)
    return out


def select_rows(batch: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """A row subset of a batched input dict, leaving non-batched entries alone.

    Mirrors what ``split_padded_tensor_dict_into_mb_list`` already does: values shaped like
    the batch are indexed, everything else is carried through untouched rather than being
    dropped or guessed at.

    Args:
        batch: The batched input dict.
        rows: Row indices to keep, in order.

    Returns:
        The subset dict.
    """
    import torch

    n = int(batch["attention_mask"].shape[0])
    idx = list(rows)
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n:
            out[key] = value[idx]
        elif isinstance(value, (list, tuple)) and len(value) == n:
            out[key] = type(value)(value[i] for i in idx)
        else:
            out[key] = value
    return out


class EngineOptimizer:
    """Lets ``ClusterAdapterSet.step`` drive the engine's own optimizer step.

    ``ClusterAdapterSet.step`` owns the loop that keeps each cluster's gradient inside its
    own expert, and it ends with one ``optimizer.step()``. An FSDP engine's step is not a
    bare ``optimizer.step()`` -- it clips across the process group, advances the scheduler
    and reports ``grad_norm``/``update_successful`` -- so the two are joined here rather than
    by writing a second step inside the engine, which would be a second implementation of
    the thing every run's ``grad_norm`` is read from.

    Args:
        engine: The engine whose ``optimizer_zero_grad``/``optimizer_step`` to drive.

    Attributes:
        stats: What ``optimizer_step`` returned, empty until it has run.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self.stats: dict[str, float] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients through the engine.

        ``set_to_none`` is accepted and asserted rather than forwarded: the engine's
        ``optimizer_zero_grad`` is ``optimizer.zero_grad()``, whose torch default is already
        ``set_to_none=True``, and that default is load-bearing -- a parameter holding a ZERO
        gradient tensor acquires Adam state and decays, so an idle expert would move.

        Raises:
            ClusterWiringError: If a caller ever asks for zeroed tensors, which would break
                the isolation guarantee silently.
        """
        if not set_to_none:
            raise ClusterWiringError(
                "set_to_none=False would give every idle expert a zero gradient tensor, "
                "and decoupled weight decay moves such a parameter on every step; the "
                "isolation guarantee is exactly that it does not"
            )
        self.engine.optimizer_zero_grad()

    def step(self) -> None:
        """Take the engine's optimizer step and keep its stats."""
        self.stats = self.engine.optimizer_step()


@contextmanager
def every_expert(model, names: Sequence[str]):
    """Run a READ-ONLY forward with every cluster expert active: the deployed model.

    ``ClusterAdapterSet.only()`` restores ``names[0]`` on the way out, so between training
    steps the model IS one arbitrary expert. Every forward taken outside ``step`` therefore
    ran on that one expert -- including ``compute_logp``, which is the importance-ratio
    denominator for every group in the batch and not only the routed one, and including the
    behavioural forward whose output decides the NEXT partition. Both failures are silent:
    the ratio is merely wrong, and the clustering merely describes one adapter.

    Activating all of them is not an approximation of the merge, it IS the merge.
    ``LoraLayer.forward`` iterates ``self.active_adapters`` and adds each one's
    ``scaling * B A`` to the result, so the output equals the summed adapter's exactly. That
    equality is asserted in ``test_cluster_lora_export.py``, because it is the one place the
    in-process model and the exported artifact are guaranteed to be the same model.

    READ-ONLY BY CONSTRUCTION. ``set_adapter`` marks every ACTIVE adapter trainable, so a
    backward inside this block would accumulate into all of them at once and destroy the
    isolation the whole method rests on. Rather than document that, the block runs under
    ``torch.no_grad()``: every call site is already no-grad, so it costs nothing and makes
    the leak impossible instead of merely forbidden.

    Args:
        model: A ``PeftModel`` carrying every name.
        names: The experts to activate, i.e. the roster.

    Yields:
        The model, with every expert active.

    Raises:
        ClusterWiringError: If ``names`` is empty, if a name is not on the model, if the
            model carries no tuner able to activate several adapters, or if the activation
            did not take. The last is checked rather than assumed: ``PeftModel.set_adapter``
            raises ``TypeError: unhashable type: 'list'`` on a list (verified on peft
            0.18.1), so this must go through ``base_model`` -- and an API that quietly
            accepted a list and activated nothing would leave every forward on one expert
            with this context manager apparently in place.
    """
    import torch

    names = tuple(names)
    if not names:
        raise ClusterWiringError(
            "every_expert needs at least one expert; activating none would leave the "
            "forward on whichever adapter happened to be active"
        )
    present = getattr(model, "peft_config", {}) or {}
    missing = [n for n in names if n not in present]
    if missing:
        raise ClusterWiringError(
            f"adapters {missing} are not on this model (it has {sorted(present)}), so this "
            "forward would see a subset of the deployed model"
        )
    tuner = getattr(model, "base_model", None)
    if tuner is None or not hasattr(tuner, "set_adapter"):
        raise ClusterWiringError(
            "this model has no tuner that can activate several adapters at once, so the "
            "sum of the experts cannot be reached in process"
        )
    previous = list(getattr(model, "active_adapters", []) or [])
    tuner.set_adapter(list(names))
    try:
        active = set(getattr(model, "active_adapters", []) or [])
        if active != set(names):
            raise ClusterWiringError(
                f"activating {list(names)} left {sorted(active)} active; this forward would "
                "run on a subset of the experts while appearing to run on all of them"
            )
        with torch.no_grad():
            yield model
    finally:
        if previous:
            tuner.set_adapter(previous[0] if len(previous) == 1 else list(previous))


def behaviour_features(
    model,
    batch: Mapping[str, Any],
    group_sizes: Sequence[int],
    cfg: ClusterLoRAConfig,
    *,
    boxed_ids: Sequence[int] | None = None,
    experts: Sequence[str] = (),
) -> tuple[np.ndarray, int]:
    """MEDS behavioural vectors, one per GROUP, from one extra no-grad forward per sample.

    This is the cost the arm buys and the reason it is off by default. The vector is the
    per-layer logit trace of the answer token, reduced by ``meds_feature``; a GROUP's vector
    is the mean over its rollouts, which is what ``interference_dump.py`` stores and what
    ``interference_analyze.py`` clusters, so the training arm and the probe cluster the same
    quantity.

    The model is put in eval mode for the duration and restored afterwards. A dropout-active
    forward would give the same rollout a different behavioural vector on every batch, and
    the clustering would then be measuring dropout.

    Args:
        model: The causal LM, wrapped or not.
        batch: Needs ``input_ids``, ``attention_mask`` and ``loss_mask``, all ``(B, T)``.
        group_sizes: Rows per group, summing to ``B``.
        cfg: The arm's configuration.
        boxed_ids: Token ids of ``"\\boxed{"`` under this run's tokenizer, for
            ``answer_strategy='boxed'``.
        experts: The cluster roster. When given, the trace runs with EVERY expert active --
            the deployed model -- instead of on whichever one ``only()`` last restored. A
            feature computed under one expert makes the partition a function of that
            adapter's behaviour rather than the model's, and the next batch is routed by it.
            ``()`` is the single-adapter case and is unchanged.

    Returns:
        ``((n_groups, d) float64, n_fallbacks)`` where a fallback is a sequence whose answer
        token could not be located and which was read at its final position instead. The
        count is returned rather than logged so it reaches the metrics: a group whose feature
        came from a different place than its peers is a group compared against something
        else.

    Raises:
        ValueError: If any group size is below 1, or if the sizes do not partition the batch.
    """
    import torch

    from .features import LayerLogitExtractor, answer_token_index, meds_feature

    sizes = [int(s) for s in group_sizes]
    n_rows = int(batch["input_ids"].shape[0])
    # Before the sum check, in the same order group_apply.py uses and for the reason its
    # comment gives: a non-positive size still passes the sum check, and the
    # per_row[start:start + size] walk below then pools one group's rows into another. With
    # sizes [-1, 5] over four rows this returned two vectors of the right shape where the
    # first was the mean of rows 0-2 and the second was row 3, and raised nothing.
    if any(g < 1 for g in sizes):
        raise ValueError(
            f"every group size must be >= 1, got {sizes}: a size of 0 or less passes the "
            "sum check below and the slice walk then gives one group another group's rows"
        )
    if sum(sizes) != n_rows:
        raise ValueError(
            f"group sizes sum to {sum(sizes)} but the batch has {n_rows} rows; a mismatch "
            "would give one group another group's behaviour"
        )
    from areal.trainer.ppo.actor import _infer_prompt_lens

    prompt_lens = _infer_prompt_lens(batch["attention_mask"], batch["loss_mask"])
    ids_all = batch["input_ids"].detach().cpu()
    lens = batch["attention_mask"].detach().cpu().long().sum(-1)
    extractor = LayerLogitExtractor(mode=cfg.extractor_mode)
    device = ids_all.device
    param = next((p for p in model.parameters()), None)
    if param is not None:
        device = param.device

    was_training = bool(getattr(model, "training", False))
    model.eval()
    fallbacks = 0
    per_row: list[np.ndarray] = []
    # The trace must see the DEPLOYED model. Between steps ``only()`` leaves one arbitrary
    # expert active, and a feature read from it makes the next partition a function of that
    # adapter rather than of the model that produced the rollouts.
    activation = every_expert(model, experts) if experts else nullcontext()
    try:
        with activation:
            for row in range(n_rows):
                seq = ids_all[row, : int(lens[row])].tolist()
                try:
                    pos = answer_token_index(
                        seq,
                        boxed_ids=boxed_ids,
                        strategy=cfg.answer_strategy,
                        response_start=int(prompt_lens[row]),
                    )
                except Exception:
                    # Recorded, not hidden: the same fallback interference_dump.py takes,
                    # and for the same reason -- a feature read at a different position is
                    # not comparable with its peers, and the count is what says how many
                    # there were.
                    pos = len(seq) - 2
                    fallbacks += 1
                ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                trace = extractor.trace(model, ids, pos, int(seq[pos + 1]))
                per_row.append(
                    meds_feature(
                        trace,
                        use_layer_diff=cfg.use_layer_diff,
                        last_n_layers=cfg.last_n_layers,
                    )
                )
    finally:
        if was_training:
            model.train()

    out = []
    start = 0
    for size in sizes:
        out.append(np.mean(np.stack(per_row[start : start + size], 0), axis=0))
        start += size
    return np.stack(out, 0), fallbacks


def begin_cluster_batch(actor, router, data, contexts, group_sizes) -> dict[str, float]:
    """SEAM 1: supply this batch's behavioural features and arm the engine's partition.

    One call, from ``PPOActor._route_groups``, immediately before the routing call. It is
    where the vector that ``RoutingContext.extra`` cannot carry is handed to the key
    function, and where the partition that comes back is put somewhere the engine can read
    it.

    The key function is built once and cached on the actor. That is not an optimisation:
    MEDS labels are stabilised against the buffered history, so a fresh instance per batch
    would refit from nothing and relabel every group every step -- churn 1.0, which is the
    measured failure mode in findings 5.1.

    Args:
        actor: The ``PPOActor``. Supplies ``engine`` and holds the cached key function.
        router: The router ``_route_groups`` built. Must be a ``ClusterRouter``.
        data: The batch dict, for the feature forward and for prompt identities.
        contexts: The routing contexts, in group order; their ``unit_id`` is what the key
            function is keyed on, so they are taken from the call site rather than rebuilt.
        group_sizes: Rows per group, in the same order.

    Returns:
        Flat metrics for the run's stats stream: the reach report plus the wiring's own
        numbers.

    Raises:
        ClusterWiringError: If the router cannot carry a partition, if features were asked
            for with no model to compute them on, or if the partition names an adapter the
            engine was not told to create.
        PartitionUnavailable: If the mode needs features and none were computed. Passed
            through deliberately -- ``partition=meds`` with the extra forward off is a
            configuration that would otherwise run as the ``none`` arm under the method's
            name.
    """
    from selfevo.routing.base import TrainingMode
    from selfevo.routing.cluster import ClusterRouter

    from .features import ClusterLoRAKeyFn
    from .partition import MEDSPartitioner, PartitionUnavailable

    cfg = ClusterLoRAConfig.from_env()
    if cfg is None:  # pragma: no cover - the call site gates on the same variable
        return {}
    if not isinstance(router, ClusterRouter):
        raise ClusterWiringError(
            f"{CLUSTER_LORA_ENV}={cfg.partition!r} needs group_routing.router='cluster'; "
            f"got {type(router).__name__}, which routes per unit and would leave the "
            "partition unused while the config claims the method"
        )

    keyfn = getattr(actor, "_selfevo_cluster_keyfn", None)
    if keyfn is None:
        from selfevo.clustering.meds import MEDSClusterer

        keyfn = ClusterLoRAKeyFn(
            MEDSPartitioner(
                MEDSClusterer(min_cluster_size=cfg.min_cluster_size),
                warmup_batches=cfg.warmup_batches,
            ),
            mode=cfg.partition,
            seed=cfg.seed,
        )
        actor._selfevo_cluster_keyfn = keyfn
    if router.key_fn is not keyfn:
        router.key_fn = keyfn

    sizes = [int(s) for s in group_sizes]
    unit_ids = [c.unit_id for c in contexts]

    seconds = 0.0
    fallbacks = 0
    features: np.ndarray | None = None
    if cfg.features:
        engine = getattr(actor, "engine", None)
        model = getattr(engine, "model", None)
        if model is None:
            raise ClusterWiringError(
                "SELFEVO_CLUSTER_LORA_FEATURES asks for the behavioural forward but the "
                "actor has no engine model to run it on"
            )
        started = time.perf_counter()
        # The roster is taken from the MODEL, not from the environment: these are exactly
        # the adapters this forward would otherwise see one of.
        adapters = getattr(engine, "_selfevo_adapters", None)
        features, fallbacks = behaviour_features(
            model,
            data,
            sizes,
            cfg,
            boxed_ids=_boxed_ids(engine),
            experts=() if adapters is None else tuple(adapters.names),
        )
        seconds = time.perf_counter() - started
    elif cfg.partition != "none":
        raise PartitionUnavailable(
            f"{CLUSTER_LORA_ENV}={cfg.partition!r} needs behavioural features and "
            "SELFEVO_CLUSTER_LORA_FEATURES is off, so no forward computed them. Refusing: "
            "the only thing this could return is one adapter for everything, which is the "
            "'none' arm wearing the method's label"
        )
    if features is None:
        # 'none' ignores its features by construction -- every group shares one adapter --
        # so the placeholder cannot change the partition, and passing it keeps the baseline
        # on the identical code path as the method.
        features = np.zeros((len(unit_ids), 1))

    partition = keyfn.begin_batch(
        unit_ids, features, group_ids=_prompt_ids(data, sizes, unit_ids)
    )

    # Every cluster trains with RL, and that has to be SET rather than assumed. The
    # independent variable of this arm is which EXPERT receives a group's gradient, not
    # which loss produces it, so the mode axis is held fixed. ClusterRouter's default policy
    # names the three silence clusters, and an adapter name matches none of them -- left
    # alone, every group would hit default_mode=SKIP and apply_decisions would zero the
    # whole batch's advantages while the partition metrics looked healthy. That is a run
    # that trains nothing and reports a clustering.
    router.policy = {key: TrainingMode.RL for key in sorted(set(partition.keys))}

    roster = adapter_roster()
    named = sorted(set(partition.keys))
    if roster:
        missing = [n for n in named if n not in roster]
        if missing:
            raise ClusterWiringError(
                f"the partition put groups on adapters {missing}, which are not in "
                f"{ROSTER_ENV}={list(roster)}. Every expert is created before the optimizer "
                "exists, so one discovered mid-run has no parameters to train; raise the "
                "roster or raise SELFEVO_CLUSTER_LORA_MIN_CLUSTER_SIZE"
            )
    engine = getattr(actor, "engine", None)
    if engine is not None:
        engine._selfevo_cluster_plan = ClusterPlan(
            key_of_group=dict(enumerate(partition.keys)),
            step=int(getattr(actor, "_selfevo_batch", 0)),
            basis=partition.basis,
        )

    metrics = dict(keyfn.report().as_metrics())
    metrics["cluster_lora/feature_seconds"] = float(seconds)
    metrics["cluster_lora/feature_fallbacks"] = float(fallbacks)
    metrics["cluster_lora/adapters_available"] = float(len(roster))
    return metrics


def _boxed_ids(engine) -> list[int] | None:
    """Token ids of ``"\\boxed{"`` under the run's tokenizer, or ``None`` if there is none.

    Read from the engine rather than tokenized in :mod:`selfevo.cluster_lora.features`,
    which deliberately takes no tokenizer dependency.

    Args:
        engine: The train engine, possibly ``None``.

    Returns:
        The ids, or ``None`` when no tokenizer is available -- in which case
        ``answer_strategy='boxed'`` refuses per sequence and falls back with a recorded
        count, rather than silently reading a different position.
    """
    tok = getattr(engine, "tokenizer", None)
    if tok is None:
        return None
    try:
        return list(tok("\\boxed{", add_special_tokens=False)["input_ids"])
    except Exception:
        return None


def _prompt_ids(data, sizes: Sequence[int], unit_ids: Sequence[str]) -> tuple[str, ...]:
    """Stable prompt identity per group, so churn is measured on prompts not positions.

    Batch positions are reshuffled every step, so churn keyed on them reads as noise
    whatever the clustering did. Falls back to the unit ids for any group whose first row
    has no prompt region, which ``ReachReport`` then reports as zero OVERLAP rather than as
    zero churn.

    Args:
        data: The batch dict.
        sizes: Rows per group.
        unit_ids: Per-group unit ids, used where a prompt key cannot be formed.

    Returns:
        One identifier per group.
    """
    from selfevo.routing.prompt_credit import prompt_key

    ids = data["input_ids"].detach().cpu().tolist()
    mask = data["loss_mask"].detach().cpu().tolist()
    out: list[str] = []
    row = 0
    for i, size in enumerate(sizes):
        try:
            out.append(prompt_key(ids[row], mask[row]))
        except (ValueError, IndexError):
            out.append(unit_ids[i])
        row += size
    return tuple(out)
