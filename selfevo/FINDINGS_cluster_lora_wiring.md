# Wiring the per-cluster LoRA arm into the trainer: the two call sites, what they cost,
# and the third seam that is still missing

2026-09-02. CPU only. No GPU job started, killed or touched; the 8xA100 `lora30b` run was
live throughout and every change below is import-safe and default-off.

Written in its own file rather than folded into `FINDINGS_cluster_lora.md` because that file
was held by another agent for the whole of this session. It continues section 9 of that
document, which lists the two seams this closes.

## 1. The two call sites

**Seam 1, the actor.** `areal/trainer/ppo/actor.py`, in `_route_groups`, immediately before
`route_all`:

    if os.environ.get("SELFEVO_CLUSTER_LORA", "").strip():
        from selfevo.cluster_lora.wiring import begin_cluster_batch
        stats_tracker.scalar(
            **begin_cluster_batch(self, router, data, contexts, list(sizes))
        )

`begin_cluster_batch` builds the behavioural vectors (when the extra forward is on), calls
`ClusterLoRAKeyFn.begin_batch(unit_ids, features, group_ids=...)`, sets the router's policy,
arms the engine with the resulting partition, and returns the reach report as flat metrics.
`unit_ids` are taken from the contexts the call site already built rather than rebuilt, so
the ids the key function is keyed on cannot drift from the ids the router routes.

**Seam 2, the engine.** `areal/engine/fsdp_engine.py`, in `train_batch`, immediately after
`_normalize_batch_input`:

    cluster_plan = getattr(self, "_selfevo_cluster_plan", None)
    if cluster_plan is not None:
        return self._train_batch_by_cluster(
            input_batched, loss_fn, loss_weight_fn, cluster_plan
        )

`_train_batch_by_cluster` splits the batch into one microbatch stream per cluster and hands
those streams to `ClusterAdapterSet.step`, which is the tested isolation guard, with an
`EngineOptimizer` shim so the one optimizer step is the engine's own -- clipping across the
process group, the scheduler, and `grad_norm`/`update_successful`.

**Seam 2a, the experts.** `_apply_peft_wrapper` creates the roster through
`ClusterAdapterSet.build` when `SELFEVO_CLUSTER_LORA_ADAPTERS` is set. That point is the only
one that is after the base model exists and before BOTH FSDP sharding and
`_create_optimizer`, which takes `self.model.parameters()`. An expert created once a cluster
is discovered mid-run would have neither a shard nor optimizer state, and would train nothing
while reporting a loss.

## 2. Two things the seam had to decide, and both are refusals turned into mechanism

**`ClusterRouter`'s policy does not name adapters, and left alone the arm trains nothing.**
The policy maps the three SILENCE clusters (`informative`, `unsolved`, `solved`) to modes.
An adapter name matches none of them, so every group would take `default_mode=SKIP` and
`apply_decisions` would zero the whole batch's advantages while the partition metrics read
perfectly healthy. `begin_cluster_batch` therefore sets `router.policy` to route every
cluster in the partition to RL. The arm's independent variable is which EXPERT receives a
group's gradient, not which loss produces it, so the mode axis is held fixed -- and
`test_the_cluster_arm_does_not_touch_the_advantage_tensor` asserts the routed tensor is
bit-identical to the unrouted one.

**`ClusterAdapterSet.step` ends each cluster with one `loss.backward()`, and the engine
backwards per microbatch.** Summing a cluster's microbatch losses and backwarding once would
satisfy `step`'s contract and retain every microbatch graph at the same time, which is the
memory the microbatch loop exists to avoid. Instead the last microbatch's `process_output`
returns `None` -- the engine's own supported way to skip a backward -- and that loss is
handed to `step`. One graph is alive at a time and `step`'s contract is unchanged.
`test_the_none_arm_reproduces_the_unrouted_step_bit_for_bit` runs at one, two and three
microbatches, because at one microbatch the arrangement is not exercised at all.

## 3. Default off, and it is checked rather than argued

Both call sites are gated on an environment variable that is unset in every run to date. The
gate is a LITERAL at each call site rather than an import of the constant, so an
unconfigured run does not import `selfevo.cluster_lora` at all; the duplication is pinned by
a test that reads both source files, and a second test makes the import fail and requires an
unconfigured run to be unaffected.

The rollback is asserted three ways:

| check | what it rules out |
|---|---|
| `test_the_default_path_is_the_committed_one_bit_for_bit` compiles the last committed `train_batch` that predates this seam out of the git blob, runs it and the current one on two identically seeded engines, and requires `torch.equal` on every parameter | the default path drifting from the code it replaced |
| `test_the_none_arm_reproduces_the_unrouted_step_bit_for_bit`, at 1/2/3 microbatches | the cluster path differing from the default path on the degenerate partition |
| `test_with_no_cluster_configuration_nothing_is_armed_and_nothing_moves` | a key function, partition or plan being built when nothing asked for one |

The git-blob comparison walks the file's history for the newest version that does NOT contain
`_train_batch_by_cluster`, so it keeps finding the pre-seam version after this work is itself
committed, rather than comparing the new code against itself.

## 4. The isolation guard through the real engine

Everything in `selfevo/tests/test_cluster_lora_engine.py` drives the real
`FSDPEngine.train_batch` -- real packing, real `_compute_logprobs_and_loss`, real
`optimizer_step` -- on a real four-layer Qwen2 with PEFT, on CPU, in a world of one. The
engine is constructed rather than `initialize()`d because initialize() shards with FSDP2 and
needs accelerators; everything downstream of the distributed attributes is the shipped code.

* a step containing only `cluster_1` leaves `cluster_0` and `shared` bit-identical
  (`torch.equal`), over TWO steps;
* the cluster that did receive the batch changed, so the assertion above is not passing on a
  step that did nothing;
* the same holds in the other direction, so an implementation that froze one particular
  expert fails;
* an expert that was TRAINED and is then idle for three steps does not move. That is the
  case the unit-level guard originally missed: an expert that was never trained has no
  `.grad` and is skipped whatever the implementation does, so only a trained-then-idle one
  can show a decay leak.

`AdamW` with **non-zero** weight decay throughout. Under zero decay a correct and an
incorrect implementation both pass and these tests constrain nothing.

## 5. What the CPU harness cannot decide, and why one test is written the way it is

Microbatches are PACKED, and without flash-attention's `cu_seq_lens` the attention runs
causally across the whole packed row. So on CPU, regrouping the same rows into different
microbatches changes every logprob slightly, and no comparison between two different
packings is exact. Two consequences:

* every comparison in the engine tests holds the packing FIXED (same rows, same
  `n_mbs`), which is why the none-arm equivalence is the shape the rollback proof takes;
* the "the denominator is the whole batch's" property is asserted on the denominator itself
  -- one call per step, over the whole batch, equal to the unrouted step's value -- rather
  than on the losses it scales. A per-cluster denominator would divide each cluster by its
  own token count and give a small cluster a large learning rate; the mutation that makes
  exactly that change is killed by that test.

## 6. The cost of the extra forward, MEASURED on a CPU proxy and BOUNDED analytically

The behavioural feature needs one extra no-grad forward per SAMPLE, truncated at the answer
token. `experiments/m25/PLAN.md` requires that cost counted against A0 for the matched-budget
claim, so it is emitted every batch as `cluster_lora/feature_seconds` and the real number
will come from the run. What can be said today:

**Measured, on a CPU proxy, at three model widths.** Real `FSDPEngine.train_batch` against
real `behaviour_features` over the same batch (8 rows x 64 tokens, 4 rows per group), median
of five, `torch.set_num_threads(1)`:

| hidden | layers | step (s) | features (s) | features / step | per sequence (ms) |
|---|---|---|---|---|---|
| 64 | 4 | 0.1074 | 0.0343 | **0.319** | 4.28 |
| 256 | 8 | 0.2947 | 0.1328 | **0.451** | 16.60 |
| 768 | 12 | 2.0509 | 0.9911 | **0.483** | 123.89 |

**The ratio RISES with width and settles near 0.5, and that corrects the arithmetic I
expected.** The bound I first wrote was `f/3`, from one no-grad forward against a step of one
forward plus a backward at twice a forward. That backward cost is for a FULL fine-tune. Under
LoRA the base weights are frozen, so the backward computes activation gradients but almost no
weight gradients, and a step is closer to TWO forward-equivalents than three. The bound is
therefore

    extra / step  ~=  f / 2

where `f` is the answer token's position as a fraction of the sequence. At `f ~= 0.97` -- what
the measured configuration has -- that predicts 0.48, which is what the widest model gives.
The small model's 0.32 is the outlier, not the large one: at hidden=64 the step's fixed
per-microbatch costs are a large share of it.

**Truncation is exact and pays exactly in proportion to `f`.** Measured separately at
hidden=768, 12 layers, one sequence of 512 tokens, median of five:

| answer position | `f` | truncated (s) | untruncated (s) |
|---|---|---|---|
| 64 | 0.12 | 0.1225 | 0.8834 |
| 128 | 0.25 | 0.2176 | 0.8701 |
| 256 | 0.50 | 0.4186 | 0.8768 |
| 510 | 1.00 | 0.8842 | 0.8693 |

So the truncation is real and linear -- and it buys nothing on the runs this method targets,
because MEDS reads the token inside `\boxed{`, which on a math rollout is at the END. Plan
for `f ~= 1` and therefore for **about half a step**, not for the third I first assumed.

**Two reasons the GPU number can be WORSE than 0.5, and one reason it can be better.**

* `behaviour_features` traces ONE SEQUENCE AT A TIME, because `LayerLogitExtractor.trace`
  takes a single sequence. On CPU at one thread that costs nothing -- throughput is roughly
  linear in tokens either way -- but on a GPU it is a batch of one per rollout against a
  packed microbatch of ten thousand tokens, and launch overhead and occupancy will dominate.
  **Batching the traced forward is the obvious optimisation and is not done here.**
* The hooks add `n_layers` Python callbacks per sequence, which is again free on CPU and
  relatively more expensive against fast GPU kernels.
* Against that, the dot product with one unembedding row instead of a full `lm_head` matmul
  saves a factor of `vocab_size` of arithmetic, and at 32B with a 151936-row vocabulary that
  term is a much larger share of a forward than it is at the 1024-row vocabulary measured
  here. The GPU ratio could fall below 0.5 for that reason alone.

These are bounds and a proxy, not the measurement. `cluster_lora/feature_seconds` is emitted
every batch so the real figure comes from the run, and the matched-budget baseline in
`experiments/m25/PLAN.md` should be sized against **half a step per step**, revised down only
once a GPU run says otherwise.

## 7. The third seam, which is NOT wired and is the remaining blocker for a number

**The experts are never merged, so nothing outside training sees more than one of them.**
`merge_sum` exists and is tested, and nothing calls it. Two places assume a single adapter:

* `FSDPEngine.upload_weights` iterates LoRA parameters `if param.requires_grad`. After
  `ClusterAdapterSet.only()` exits, exactly one adapter is active and trainable, so a weight
  sync pushes ONE expert to the rollout engine rather than the sum.
* `_save_lora_to_hf` selects by the same predicate and strips `".default"` for vLLM's
  adapter-name parser, which the roster's names do not match.

So a cluster-routed run can TRAIN today and cannot yet be evaluated or served. That is a
third call site (in the save/sync path), it was not one of the two this work was asked to
close, and it is stated here rather than absorbed.

## 8. Four other things a run needs to know before it is configured

1. **The expert roster is fixed and the partitioner is not bounded by it.**
   `MEDSPartitioner` allocates a fresh stable id for every new raw label, so HDBSCAN can
   discover more clusters than there are experts. `begin_cluster_batch` REFUSES at that point
   -- the earliest it is knowable, before any training -- naming both fixes: a larger roster,
   or a larger `min_cluster_size`. Given findings 5.2 (MEDS' shipped `min_cluster_size=2`
   over-fragments), size the roster with headroom and sweep the parameter.
2. **`~/venv312b` has neither scikit-learn nor hdbscan**, so `partition=meds` raises
   `ClusteringUnavailable` there. Installing them under a live job is refused by policy, so
   a MEDS training arm needs a venv built while no job is running.
   `test_meds_with_features_reaches_the_clusterer_rather_than_the_refusal` asserts the
   feature supply reaches the clusterer in EITHER venv, because `PartitionUnavailable` --
   what an unsupplied one raises -- comes first.
3. **Gradient clipping couples the clusters.** `optimizer_step` clips over all model
   parameters, so one cluster's gradient magnitude scales another's step. Bit-identity for
   an untouched expert still holds, because its `.grad` is `None` and clipping does not
   reach it, but the effective per-cluster learning rate is not independent.
4. **`cluster_lora/loss/<name>` is the LAST microbatch's loss** when a cluster spans several
   microbatches, because `ClusterStepRecord` records what `loss_fn` returned.
   `cluster_lora/grad_norm/<name>` is over the whole cluster and is the number to read.

## 9. Configuration

| variable | default | meaning |
|---|---|---|
| `SELFEVO_CLUSTER_LORA` | unset = OFF | `meds`, `random_matched` or `none` |
| `SELFEVO_CLUSTER_LORA_ADAPTERS` | unset | comma-separated expert roster, e.g. `cluster_0,cluster_1,cluster_2,shared` |
| `SELFEVO_CLUSTER_LORA_FEATURES` | `0` | spend the extra forward |
| `SELFEVO_CLUSTER_LORA_MIN_CLUSTER_SIZE` | `2` | HDBSCAN's; sweep it, do not inherit it |
| `SELFEVO_CLUSTER_LORA_SEED` | `0` | seed for the size-matched control |
| `SELFEVO_CLUSTER_LORA_WARMUP` | `1` | batches buffered before the first fit |
| `SELFEVO_CLUSTER_LORA_ANSWER` | `last` | `boxed` is MEDS' own and needs the engine's tokenizer |
| `SELFEVO_CLUSTER_LORA_EXTRACTOR` | `hooks` | `hidden_states` is the MEDS-faithful reference |
| `SELFEVO_CLUSTER_LORA_LAYER_DIFF` | `0` | MEDS ships it off |
| `SELFEVO_CLUSTER_LORA_LAST_N_LAYERS` | latter half | trailing layers kept |

The environment is the surface because `_route_groups` builds routers with `factory()` and
no kwargs -- the same reason `router=random` reads `SELFEVO_RANDOM_PROPORTIONS` -- and
`areal/api/cli_args.py` was another agent's file. Requires `group_routing.enabled=true` and
`group_routing.router=cluster`; any other router is refused, because a per-unit router would
leave the partition unused while the config claimed the method.

Metrics emitted into the same stream as the `route/*` keys: `cluster_lora/n_groups`,
`n_clusters`, `noise_fraction`, `largest_cluster_fraction`, `churn`, `churn_overlap`,
`refusals`, `size/<adapter>` per adapter, `feature_seconds`, `feature_fallbacks`,
`adapters_available`; and from the engine step `clusters_stepped`, `clusters_skipped`,
`plan_step`, and `loss/`, `grad_norm/`, `rows/` per adapter.

## 10. Discipline

`~/venv312b/bin/python -m pytest selfevo experiments -q`: **1944 passed, 3 skipped** after,
against 1881 passed / 3 skipped / 1 failed measured on the same tree with the two source
edits already in place and none of the tests yet written. The 51 tests added here account for
part of that delta and three other agents' concurrent work for the rest -- the checkout is
shared and was moving throughout, which is why the before-figure is quoted with its own
timestamp rather than as a clean baseline. The one failure in the before-run,
`test_dapo_baseline.py::test_rejected_groups_are_regenerated_under_the_shipped_dynamic_bs`,
passed in isolation at the time and passes in the after-run; it belongs to another agent's
in-flight edits to `areal/infra/workflow_executor.py` and never touched anything here.

**None of the 51 tests skip.** The 3 skips in the suite are the pre-existing
clustering-dependent modules, which run for real under `~/venv_probe`.

Mutation testing is `selfevo/tests/mutate_cluster_lora_wired.py`, run against a COPY at
`/home/ubuntu/mutcopy_wiring` whose three targets were asserted sha256-identical to the
originals before starting, re-checked after every restore and again at the end. A copy and
not the live checkout, because the 8xA100 job imports this tree through a `.pth` in the venv
and relaunches its workers on exit. Mutants that do not compile, and anchors that are not
unique, are reported as SKIP and counted as NOT killed -- a mutation that was never exercised
is not a passing one.

**One operational lesson: never kill the harness mid-run.** Doing so left a mutated
`fsdp_engine.py` on the copy's disk, and the next run reported BASELINE RED rather than a
false kill -- which is the harness's sha256 discipline working, but the copy had to be
restored from the original and re-hashed before it could run.

| target | mutations | killed | survivors | skipped |
|---|---|---|---|---|
| `areal/trainer/ppo/actor.py` | 4 | **4** | none | none |
| `areal/engine/fsdp_engine.py` | 13 | **13** | none | none |
| `selfevo/cluster_lora/wiring.py` | 26 | **26** | none | none |
| total | 43 | **43** | none | none |

Every mutation is aimed at a guard rather than at arithmetic, because the failure mode here
is not a wrong number but a run that trains, logs and reports as the method while being the
baseline. The four that would most plausibly have shipped:

1. *the seam is entered whatever the configuration says* -- `if True:` in place of the
   environment gate. `begin_cluster_batch` returns an empty dict with nothing configured, so
   this looks harmless and is not: the gate is what keeps `selfevo.cluster_lora` out of the
   default import graph, and it is killed only by the test that makes that import fail.
2. *every cluster is routed to SKIP* -- one word in the policy line, and the arm would zero
   every advantage in the batch while its partition metrics stayed healthy.
3. *the denominator becomes one cluster's tokens rather than the batch's* -- a small cluster
   would get a large learning rate, and the losses cannot see it on this harness.
4. *a fresh key function per batch* -- churn 1.0, which is the exact failure findings 5.1
   measured MEDS' own kNN stabilisation walking into.
