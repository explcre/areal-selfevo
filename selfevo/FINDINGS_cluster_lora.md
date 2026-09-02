# Per-cluster LoRA experts routed by discovered behavioural clusters: what is built, what
# reaches what, and what still needs a GPU

2026-09-02. CPU only, no GPU job started or touched. Written here rather than in `GOAL.md` or
`EXPERIMENTS.md` because both were held by other agents for the whole of this session; fold
it in from here.

The method: cluster GRPO groups by the model's own behaviour (MEDS-style HDBSCAN over the
latter-half layer-wise logits at the final answer token), keep one LoRA adapter per cluster
plus a shared adapter for HDBSCAN noise, let each group's loss update only its own cluster's
adapter, and sum the adapters at inference (LSPO-style). The hypothesis is mechanical:
behavioural subpopulations inside one task want conflicting updates, a shared adapter averages
them, and per-cluster adapters remove the averaging.

Everything below that is a number was measured on this box today. Everything that needs a GPU
is listed in section 9 and is not estimated as if it had been measured.

## 1. Where the layer-wise logits come from, and at what cost

**They are not reachable at the `ClusterRouter.key_fn` seam without an extra forward pass.**
This was the first question in the brief and the answer is negative. Precisely, in
`areal/engine/fsdp_engine.py`:

* `forward_backward_batch` calls `outputs = self.model(**inputs)` with no
  `output_hidden_states`, keeps `logits = outputs.logits.squeeze(0)`, and lets `outputs` --
  and every intermediate hidden state with it -- go out of scope at the end of the loop
  iteration.
* The only tensors that cross into selfevo territory are `logprobs`, `entropy` and the
  per-token `vocab_min/max_logits` (`_compute_logprobs_and_loss`). Per-LAYER logits need the
  residual stream at each layer, which is never materialised as an output at all.
* By the time `PPOActor._compute_advantages` runs -- where the `key_fn` seam is reached -- the
  forward is over and the batch holds only `advantages`, `loss_mask`, `logprobs` and rewards.

MEDS gets the feature free because verl's `dp_actor._forward_micro_batch` sets
`extra_args["output_hidden_states"] = True` in the log-prob pass it was already running.
AReaL does not expose that seam, and the one-line upstream change is both outside this agent's
territory and the expensive option: retaining every layer for every token of a packed
microbatch is `n_layers x tokens x hidden`, about **6.3 GB per 10240-token microbatch at 32B
in bf16** -- the per-position-cache OOM this project's notes already warn about.

So the extra-forward version is implemented, behind `LayerLogitExtractor`, and made cheap in
ways the engine change could not be:

| choice | what it saves | verified by |
|---|---|---|
| forward **hooks** at one position per layer instead of `output_hidden_states` | retains `n_layers x hidden` floats per sequence -- **1.3 MB at 32B** (64 x 5120 x 4) instead of the whole activation stack | `test_the_hook_path_reproduces_the_hidden_states_path_exactly` |
| **truncate** at the answer token before the forward | all work after that position; exact, not approximate, because attention is causal | `test_truncating_at_the_answer_token_is_exact_because_attention_is_causal` |
| **dot product** with one unembedding row instead of a full `lm_head` matmul | a factor of `vocab_size` (151936 at Qwen) of arithmetic | `test_the_dot_product_shortcut_is_the_same_number_as_the_full_unembedding` |

The hook path pins a transformers layout assumption, so it is a test and not a comment.
Verified against transformers 5.3 today: `hidden_states` is
`(embeddings, layer_0_out, ..., layer_{N-2}_out, norm(layer_{N-1}_out))`, so reproducing
MEDS' `hidden_states[1:]` means hooking layers `0..N-2` **and the final norm**, never every
layer. An upgrade that changes the convention fails the test rather than shifting the feature
by one layer in silence.

**Cost still outstanding**: the extra forward as a fraction of a real training step. It is one
no-grad truncated forward against a forward+backward over the full sequence, so it should be
well under a third of a step, but that is arithmetic, not a measurement. See section 9.

## 2. The routing guard, which is the load-bearing test

`ClusterAdapterSet.step` runs each cluster's microbatch with only that cluster's adapter
active and applies one optimizer step afterwards. The guard the brief asked for:

* `test_an_untouched_adapter_is_bit_identical_after_a_step_for_another_cluster` -- adapters
  `cluster_0` and `shared` are **bit-identical** (`torch.equal`, not `allclose`) after two
  steps containing only `cluster_1` groups.
* `test_the_adapter_that_did_receive_the_batch_actually_changed` -- and `cluster_1` does
  change, so the assertion above is not passing on a step that did nothing.
* `test_the_isolation_holds_in_the_other_direction_too` -- symmetric, so an implementation
  that happened to freeze one particular adapter fails.

Two design points that the tests, not the prose, are what enforce:

**`AdamW` with non-zero weight decay is load-bearing in the test.** Decoupled weight decay
moves a parameter on every step it is stepped on, including one whose gradient is exactly
zero. So `zero_grad(set_to_none=True)` is not a micro-optimisation: a parameter whose `.grad`
is `None` is skipped by `torch.optim` entirely, while one holding a zero TENSOR acquires Adam
state and decays. Under a zero-decay optimizer both implementations pass and the guard would
constrain nothing. Two steps, not one, for the same reason.

**`unchanged()` is bit-equality on purpose**, pinned by
`test_unchanged_is_bit_equality_and_not_a_tolerance`: a 1e-9 perturbation must be reported as
changed. `allclose` would call a leaked small update "unchanged", which is exactly the defect
being hunted, and the mutation harness confirms no other test in the file catches it.

Refusals rather than silent no-ops: a cluster whose backward leaves its own adapter with no
gradient raises `AdapterIsolationError` ("the loss is not connected to the adapter, so this
cluster would train nothing while reporting a loss"); so does a batch naming an unmanaged
adapter, an adapter missing from the model, and activation that leaves another expert
trainable.

## 3. The merge

`merge_sum` calls PEFT's own `add_weighted_adapter(combination_type="cat")` -- the repo's
policy is to use official code rather than reimplement -- and then **verifies** the result
against an independently computed sum of per-adapter deltas, raising `MergeInexact` if they
disagree. The merged adapter is what gets deployed, so a silent difference would change every
inference number without changing anything a training metric can see.

The headline test is at the input level, as asked:
`test_the_merged_adapter_applied_to_an_input_equals_the_sum_of_the_two_deltas` asserts
`merged(x) - base(x) == (A(x) - base(x)) + (B(x) - base(x))` on a real module output, and
separately asserts both deltas are non-zero. That last clause matters: **LoRA initialises
`B = 0`, so every adapter's delta is exactly zero at init and any merge whatsoever reproduces
the sum.** A test that skipped randomising `B` would pass on an implementation that returned
an empty adapter.

`test_the_verification_fires_when_the_merge_is_actually_wrong` breaks the merge on purpose
(weights doubled) and requires the refusal, because a disabled check is indistinguishable from
a passing one in every other test here.

## 4. The size-matched control, and it is not optional

`random_matched_partition` builds the control by **permuting the method's own labels** rather
than sampling from the observed proportions. The sizes then match exactly rather than in
expectation, for any batch and any size distribution -- the same argument
`selfevo.routing.proportions.MatchedPermutationControl` makes for mode proportions, and it was
added there after a configured control was measured to be mis-matched.

Asserted exactly on five adversarial shapes (even, lopsided, all-noise, singletons, noise plus
uneven clusters), and asserted to be genuinely feature-blind over 50 seeds (a single seed can
permute to the identity by chance). The noise bucket is permuted with the rest, so the control
also carries the same number of groups on the shared adapter -- a control with fewer would
have more per-cluster capacity than the method it controls for.

Selectable by config exactly as specified: `cluster_lora.partition = meds | random_matched |
none`, through `partition_from_config`. `none` is the vanilla shared-LoRA arm expressed in the
same `Partition` type, so the baseline runs the identical code path and a difference between
arms cannot come from the plumbing. **A mode that needs features and is given none REFUSES**
rather than returning one adapter for everything, which would be "the `none` arm wearing the
method's label" -- the silent no-op this project keeps hitting.

## 5. Reach and stability, and the finding that changes the method

### 5.1 MEDS' kNN classify does NOT stabilise adapter identity. Measured churn 1.0.

This is the most important thing found today. MEDS refits HDBSCAN on the growing buffer every
batch, and **HDBSCAN renames its clusters on each refit**. On four perfectly separated blobs
whose membership did not change at all, three consecutive batches produced labels `2`, then
`3`, then `0` for the same six groups. Every group therefore changed adapter every step --
churn **1.0** -- while the clustering was structurally identical.

MEDS is not wrong; it is doing something else. It uses the label to look up a cluster SIZE for
reward shaping, where a permutation of names costs nothing. Selecting an ADAPTER by that name
is a different use, and under it a permutation sends every expert somebody else's gradient --
the method failing through precisely the mechanism meant to prevent it.

The fix (`MEDSPartitioner._resync`) matches new raw labels to existing expert ids by OVERLAP on
the buffered groups, greedily, largest overlap first, so an expert keeps the subpopulation it
was fitted to. Raw labels matching nothing get a fresh expert.

| arm | churn per step |
|---|---|
| naive MEDS (kNN classify against raw labels) | **1.0** at every step |
| with overlap matching | **0.0 - 0.083** |

Both are tests, and the naive number is a live counterfactual
(`test_without_the_label_matching_every_group_changes_expert_every_step`) rather than a
remembered figure -- without it the bound on the fixed version could be satisfied by a
partitioner that never clustered anything.

The residual 0.083 is real and is not hidden: the buffer grows every batch, HDBSCAN finds more
structure in more points, and a blob eventually splits off a fragment that a couple of groups
migrate into. That is the method's own behaviour and is bounded and reported, not asserted
away.

### 5.2 MEDS' shipped `min_cluster_size=2` over-fragments

On 24 points in four perfectly separated directional blobs of six:

| `min_cluster_size` | clusters found | noise |
|---|---|---|
| 2 (MEDS' shipped default) | 6 | 2 |
| 4 | 4 | 0 |
| 6 | 4 | 0 |

Again, harmless for reward shaping and not harmless here: **every extra cluster is another
expert trained on fewer groups.** The value is exposed
(`analyse_dump(min_cluster_size=...)`, `--min-cluster-size`) and a run should sweep it rather
than inherit it. The coordinator's MATH batch has 128 groups, so `min_cluster_size=5` is a
sensible starting point and the sweep is cheap because it is CPU-side.

### 5.3 The clustering sees only DIRECTION

The vendored path L2-normalises before euclidean HDBSCAN. Two features differing only in
magnitude are the same point, a feature at the origin is degenerate, and after normalisation
the whole space is a sphere -- on which there is nowhere far from everything. A 2-D outlier at
45 degrees was measured being absorbed into a neighbouring blob; it only reads as noise when
placed on an axis the blobs do not occupy. Anyone designing a behavioural feature whose
information is in its scale should know this first.

### 5.4 What is reported per batch

`ReachReport.as_metrics()` emits, flat: `n_groups`, `n_clusters`, `noise_fraction`,
`largest_cluster_fraction`, `churn`, `churn_overlap`, `refusals`, and `size/<adapter>` for
every adapter separately. Per-cluster sizes are never summed away -- two clusters of 32 and
eight of eight give the same total and call for opposite readings.

Three degeneracies are recorded rather than raised, because a refusal mid-run loses the
training while a metric pinned at 1.0 for 400 steps is a result: everything-is-noise
(`noise_fraction` 1.0, the run is vanilla LoRA in disguise), one-cluster-swallows-the-batch
(`largest_cluster_fraction`, which counts the shared bucket deliberately), and churn.
`churn_overlap` is reported separately from `churn` so "nothing moved" and "nothing was
comparable" cannot read the same -- a run whose prompts never recur would otherwise report
perfect stability. Churn is keyed on prompt identity, never batch position, which is
reshuffled every step.

## 6. The interference probe, split in two

Per the coordinator's blocker, the probe is two scripts, and the split is forced by a real
dependency constraint: the GPU venv has torch/peft/transformers and must NOT acquire
scikit-learn or hdbscan under a live job.

**`interference_dump.py`** (GPU; torch, peft, transformers only). Per GROUP it computes and
stores: a linear sketch of the gradient of that group's GRPO loss over the LoRA parameters; a
linear sketch of the gradient of the prompt-token NLL (the ELREA feature); the MEDS
behavioural vector; task label, size, mean reward, and the fraction of gradient blocks that
were exactly zero. It also stores the FULL unprojected gradient for the first few groups. It
does not cluster.

**`interference_analyze.py`** (CPU; numpy, scikit-learn, hdbscan -- runs under `~/venv_probe`).
It forms four partitions from that one file and reports, for each: pairwise cosine between
per-cluster gradients, conflict rate, cancellation `||sum g_c|| / sum ||g_c||`, cluster sizes,
and a bootstrap CI on the mean pairwise cosine resampled over GROUPS. Plus the three headline
contrasts as DIFFERENCES, because a bare "MEDS conflict is 0.4" answers none of the objections.

| partition | what objection it answers |
|---|---|
| `meds` | the method |
| `random_matched` | "it is more adapters, not clustering" |
| `elrea` | ELREA (2502.00089) already clusters -> per-cluster LoRA -> merges, on PROMPT-token gradients. If those conflict as much, rollouts buy nothing |
| `task` | 2608.03573 measures cross-task RL cosine at ~1e-5 and publishes Parallel-RL; the calibration is the scale bar |

**Why the split is free.** A cluster's gradient is the SUM of its members' gradients, exactly,
because every group's loss carries the same GLOBAL denominator (batch response tokens for GRPO,
batch prompt tokens for the NLL). The sketch is linear, so summing sketches is sketching the
sum. Any partition is therefore free once the dump exists. Both halves of that are tests:
`test_the_group_losses_sum_to_the_batch_loss` checks the denominator identity against an
independent reference computation, and `test_the_sketch_is_linear` /
`test_summing_group_sketches_equals_sketching_the_summed_gradient` check the sketch.

### 6.1 The instrument reports its own floor, and it matters for the reviewer

The sketch is a CountSketch (a dense Gaussian projection of 1e8 x 8192 is 3 TB and cannot be
formed). A sketched cosine has standard error about `1/sqrt(dim)`, so the resolution floor is
`3/sqrt(dim)` = **0.0331 at dim=8192**.

**A cross-task cosine of ~1e-5 therefore cannot be confirmed from sketches at any dimension
this probe can afford**, and the analysis says so: every block carries `resolution_floor` and a
`resolved` flag, and an unresolvable cosine is reported as below the floor rather than as a
number. This is the honest answer to the reviewer citing 2608.03573: our instrument can say
"below our floor", and the exact cosines for the stored full-gradient pairs are the only
place a 1e-5 claim could be checked at all. Raise `--sketch-dim` to trade memory for floor.

### 6.2 Two properties of the statistics worth knowing before reading a result

* **The numerator of `cancellation` is partition-invariant.** `sum_c g_c` is the batch
  gradient whatever the partition, so cancellation varies across partitions only through
  `sum_c ||g_c||`. It measures how much the per-cluster gradients GROW relative to a fixed
  batch gradient. Measured on the planted structure: `sum ||g_c||` was 25.07 for the true
  partition against 9.6-13.2 across five control seeds -- the true partition's clusters are
  internally coherent, the control's cancel inside each cluster.
* **The bootstrap CI need not contain the point estimate**, and the tests do not require it
  to. Resampling groups with replacement duplicates members and biases a cluster sum toward
  its duplicated directions, so the percentile interval sits slightly above the point
  estimate. Asserting containment would pin a bias rather than a result. The bootstrap SPREAD
  is a second and independent discriminator: std **0.0085** for the true partition against
  **0.064-0.090** for permuted ones.
* **Two clusters is not enough.** With `K=2` there is exactly one pair, and the control's mean
  cosine was measured swinging between -0.12 and -0.22 across seeds. Prefer `K>=4`; the MATH
  batch at 128 groups should give that.

### 6.3 CPU verification, all four partitions

The arithmetic is verified on a planted structure whose answer is known: four clusters whose
gradients lie along four simplex directions, so every pair has cosine exactly `-1/3`.

| partition | mean cosine | conflict rate | cancellation |
|---|---|---|---|
| true (`meds`) | **-0.3025** (planted: -1/3) | 1.000 | 0.153 |
| `random_matched`, 5 seeds | -0.117 to -0.221 | 0.667 - 1.000 | 0.291 - 0.399 |

The contrast is asserted over five control seeds, not one. `elrea` is checked against its NULL
-- the fixture's prompt-gradient features carry no behavioural information, so the ELREA
partition must land near the control, and a probe that reported it as conflicted here would
report it as conflicted on any batch. `task` is checked to SKIP with a reason on a single-task
batch, never to report a zero cross-task cosine, while the other three blocks still come back.

The two halves are also run against each other as two real processes in their two real venvs
(`test_the_dump_and_the_analysis_run_in_their_two_SEPARATE_venvs`), on a randomly-initialised
four-layer model, producing all four blocks from one `.npz`. That test also asserts that the
analysis venv has NO torch and the training venv has NO hdbscan, so the dependency split is
tested rather than assumed. Values are not asserted there: a random tiny model has no
behaviour to cluster, and this proves the arithmetic, not the science.

### 6.4 Costs of the dump

* **Per group stored**: 2 sketches x 8192 float64 = **128 KB**, plus the behavioural vector
  (`n_layers/2` floats). Negligible; 128 groups is ~16 MB.
* **Full gradients**: `n_groups x n_lora_params x 4` bytes. For a 134 MB adapter (~33M params
  in fp32) at the default 8 groups that is **~1.07 GB**. The script REFUSES above
  `--max-full-grad-gb` (default 8) and names the size, so the count is lowered on purpose
  rather than discovered after the GPUs are allocated.
* **Compute per group**: two forward+backward passes over the group's samples (GRPO and prompt
  NLL) plus one truncated no-grad forward per sample for the behavioural feature.
* **Wall clock**: NOT measured, because CPU timings on this box are contaminated -- a 32x32
  `nn.Linear` took ~10 ms and one dump of six tiny groups took 27 s, essentially all of it
  inside `torch._C._nn.linear`, thread-thrashing against the training job's host processes.
  `torch.set_num_threads(1)` brought the same dump to ~2 s. Nothing here predicts GPU cost;
  the dump prints `seconds_gradients` / `seconds_features` / `seconds_total` so the real
  figure comes from the run itself.

### 6.5 The rollout schema

`load_rollouts` accepts both shapes -- one line per SAMPLE (`response`, scalar `reward`) and
one line per GROUP (`responses` list, `rewards` list), which is what the harness writes -- and
token ids in place of text in either. Alternate field names are searched
(`group_id|prompt_id|uid|qid|index`, `task|dataset|source|data_source`,
`responses|completions|outputs|generations`, `rewards|scores|accs|acc`, ...) and can be
overridden with `--group-key` / `--task-key` / `--reward-key`. Extra fields (lengths,
truncation flags) are ignored, not rejected. A missing prompt, response or reward raises
`RolloutSchemaError` **naming the file, the line, the names looked for and the keys the record
actually has**, so a schema mismatch on another box is fixed by reading the error. A reward
list whose length disagrees with the responses is refused -- it would score one rollout with
another rollout's reward.

Two behaviours to know before reading a first result:

* **At a fresh LoRA init `B = 0`, so `dL/dA = 0` exactly and half of every gradient vanishes.**
  Measured `zero_block_fraction` = 0.50 on the CPU fixture. The cosines are then taken over the
  `B` blocks alone -- still a real gradient, but not a mid-training one. **Pass `--adapter` with
  a trained checkpoint** (`globalstep24`, `globalstep49`); the dump records
  `zero_block_fraction` either way so a degenerate run cannot be read as a trained one.
* A group whose samples all score alike has advantages identically zero and contributes NO
  gradient. That is the 29-44% RL-silent share this project already measures, and the probe
  shows it rather than renormalising to give it a direction; `pairwise_stats` excludes
  zero-norm clusters from the pairs instead of counting them as orthogonal, which would drag
  every mean toward zero in proportion to how many groups were silent.

## 7. Environment, and the dependency decision

`~/venv312b` is imported by the live 8xA100 job through `areal.pth`. It has numpy 2.2.6,
scipy 1.15.3, torch 2.9.1+cu128, peft 0.18.1, transformers 5.3.0, and **neither scikit-learn
nor hdbscan**.

**`pip install hdbscan --no-deps` was NOT run, and could not have worked.** The brief allowed
it only if hdbscan's dependencies were already satisfied; `hdbscan` imports `sklearn`, which is
absent, so installing it no-deps would have produced a package that raises on import, and
installing scikit-learn would have added joblib and threadpoolctl underneath a running job.
Per the hard constraint, the install was stopped and reported rather than risked.

`~/venv_probe` (created by the coordinator; numpy 2.5.2, scikit-learn 1.9.0, hdbscan, python
3.12.14, **no torch**) runs `interference_analyze.py` and the two clustering-dependent test
files. `pytest` was installed into `~/venv_probe` only -- the single package added anywhere
today. `~/venv312b` is verified unchanged: torch 2.9.1+cu128, numpy 2.2.6, sklearn absent,
hdbscan absent.

`selfevo/clustering/meds.py` already falls back to `sklearn.cluster.HDBSCAN` when the
standalone package is missing, and `MEDSClusterer.backend` records which ran -- so "the
clustering differed" and "the clustering LIBRARY differed" stay distinguishable in a
comparison. Measured backend under `~/venv_probe`: `hdbscan`.

## 8. Discipline

`~/venv312b/bin/python -m pytest selfevo -q`: **1417 passed** before, **1551 passed, 3 skipped**
after. The three skips are the two clustering-dependent modules and the cross-venv test, all of
which RUN for real under `~/venv_probe` -- they are not tests that never execute.

Mutation testing is `selfevo/tests/mutate_cluster_lora.py`, run against a COPY at
`/home/ubuntu/mutcopy` whose `cluster_lora` modules were asserted sha256-identical to the
originals before starting; every target's sha256 is re-checked after every restore and again
at the end. The harness refuses to start unless imports resolve inside the copy. It runs in two
arms because the two halves live in two venvs, and a mutation whose tests cannot run under the
chosen interpreter is reported NOT APPLICABLE rather than killed -- a mutation that was never
exercised is not a passing one. Kill table in section 10.

Four tests exist only because the mutation harness predicted survivors, and each pins a
property no other test could see: bit-equality in `unchanged()`, the merge verification
actually firing, per-parameter hashing in the sketch, and the control's own size-match
assertion.

`git status --porcelain` shows only files in this agent's territory:
`selfevo/cluster_lora/`, eight `selfevo/tests/test_cluster_lora_*.py`,
`selfevo/tests/mutate_cluster_lora.py` and this file. `selfevo/routing/cluster.py` was NOT
edited: `ClusterRouter.key_fn` is already `Callable[[RoutingContext], str]` and
`ClusterLoRAKeyFn` satisfies it as-is.

## 9. What still needs a GPU, and what does not yet reach the trainer

**Not wired, and this is the honest statement of reach.** Two seams are satisfied and tested
but not connected, both because the files that would connect them are other agents' territory:

1. **`ClusterLoRAKeyFn` reaches `ClusterRouter` but the actor never supplies features.**
   `test_the_key_fn_partitions_a_batch_through_the_real_ClusterRouter` drives a real
   `ClusterRouter.route_batch` and gets this partition back, so the seam works. But
   `PPOActor._route_groups` builds `RoutingContext.extra` from `group_features(...)` only, and
   `extra` is `Mapping[str, float]` and cannot carry a vector. A run needs one call to
   `key_fn.begin_batch(unit_ids, features)` added at that seam in
   `areal/trainer/ppo/actor.py`, plus the extra forward that produces the features. Until then
   `partition=meds` in a config would REFUSE (it raises rather than collapsing to one adapter),
   which is the intended behaviour and not a working arm.
2. **`ClusterAdapterSet` is not called by `FSDPEngine.train_batch`.** The isolation guard is
   proved on a real PEFT model with a real optimizer, but the engine still runs one adapter for
   the whole batch. Wiring it means microbatching by cluster inside the engine, which is
   `areal/engine/fsdp_engine.py`.

Needing a GPU:

* **The interference measurement itself** -- `interference_dump.py` on `globalstep24`,
  `globalstep49` and `initial_lora` with `~/runs/harnessT_trunc/rollouts_math64.jsonl` (128
  groups, k-histogram 21.9% fully wrong / 29.7% partial / 48.4% fully correct -- not saturated,
  which is what HDBSCAN needs), then `interference_analyze.py` under `~/venv_probe`. Suggested:
  `--adapter <ckpt> --full-grad-groups 8 --sketch-dim 8192`, analysed at
  `--min-cluster-size 5` with a sweep over 2/5/8. Prefer the MATH batch over the GSM8K ones:
  their non-saturation is partly caused by harness truncation, so clusters there may separate
  "truncated vs not" rather than anything about reasoning.
* **Whether MEDS features cluster at all on a real 32B checkpoint.** They may come back all
  noise, in which case `noise_fraction` is 1.0 and the method is vanilla LoRA -- which the
  reach metrics report rather than hide, and which is a result.
* **The measured cost of the extra forward** as a fraction of a training step, and the dump's
  own wall clock (it prints both phase timings).
* **Real churn across consecutive training batches.** The 0.0-0.083 above is on a synthetic
  fixture. Churn on real behavioural features, with the buffer growing over hundreds of steps,
  is the number that decides whether the experts get coherent updates.
* **Everything downstream**: the per-cluster GRPO run, the merged-adapter benchmark numbers,
  and the matched-budget baselines (shared LoRA, LSPO 2-expert, random-matched, task-label).

## 10. Kill table

See section 8 for the harness. Results are appended below when both arms have run.

Run against `/home/ubuntu/mutcopy` and `/home/ubuntu/mutcopy2`, each verified sha256-identical
to the originals for all nine `cluster_lora` modules before starting and verified restored
clean afterwards. 53 distinct mutations, two arms, **every applicable mutation killed and no
SKIPs** -- an anchor that stopped being unique is reported as NOT a kill, and one was
(`combination_type="cat"` appeared twice) until the anchor was made specific.

| arm | interpreter | mutations applicable | killed | survivors |
|---|---|---|---|---|
| `torch` | `~/venv312b` | 43 (10 not applicable) | **43** | none |
| `cluster` | `~/venv_probe` | 27 (26 not applicable) | **27** | none |

Mutations by area: adapter isolation 11, merge 5, partition and control 15, sketch 6, dump 9,
analysis 7.

**Five defects survived the first pass and each produced a new test.** They are recorded
because the tests that missed them looked entirely reasonable:

1. *`zero_grad(set_to_none=False)` survived the isolation guard.* The two-step test trained
   only `cluster_1`, so `cluster_0` never had a `.grad` at all and was skipped by the
   optimizer either way. The leak needs an expert that was trained and is then IDLE -- the
   normal case, since most clusters are absent from most batches. `test_an_expert_that_is_
   IDLE_this_step_does_not_decay` trains every expert once and then leaves one idle for three
   steps.
2. *The `requires_grad` leak check inside `only()` survived.* It is defensive, so nothing
   fires it while PEFT behaves. `test_the_leak_check_fires_if_PEFT_stops_freezing_the_other_
   experts` replaces `set_adapter` with one that activates without freezing and requires the
   refusal -- the semantics belong to PEFT, not to this project, and an upgrade could change
   them.
3. *Masking the prompt NLL to the RESPONSE survived.* The test only asserted the two sketches
   were not parallel, and they still were not -- the GRPO loss is advantage-weighted and the
   NLL is not, so a response-masked NLL is a different vector from the RL gradient while being
   useless as the ELREA feature. `test_the_prompt_nll_covers_exactly_the_prompt_positions`
   checks against an independent reference over the prompt positions.
4. *Dropping the CountSketch's random signs survived.* On zero-mean random vectors the
   unsigned estimator is still unbiased, only noisier, so the angle-preservation test could
   not see it. It breaks on non-negative vectors, where every hash collision ADDS:
   `test_the_random_signs_are_what_keep_disjoint_vectors_orthogonal` sketches two exactly
   orthogonal non-negative vectors and requires the cosine to stay near zero.
5. *A bootstrap that stopped resampling survived.* Every replicate identical gives
   `std = 0.0` for one partition and `5.6e-17` for the other, so the ratio assertion compared
   float noise and passed. Now checked on interval WIDTH, which is what a degenerate bootstrap
   actually gets wrong: a zero-width interval claims a precision the estimate does not have.

Two of those five (1 and 4) are defects that would have produced a plausible, publishable
number rather than an error.

## 11. How to run it on the H100 box

The dump needs only torch, peft and transformers, so it runs in the H100's own venv with no
installs. Nothing here starts a training job.

    # one dump per checkpoint; three checkpoints gives the early/mid/late comparison
    for CKPT in initial_lora globalstep24 globalstep49; do
      python -m selfevo.cluster_lora.interference_dump \
        --model  <base model path> \
        --adapter ~/runs/.../$CKPT \
        --rollouts ~/runs/harnessT_trunc/rollouts_math64.jsonl \
        --out    ~/runs/interference_$CKPT.npz \
        --sketch-dim 8192 --full-grad-groups 8 --device cuda --dtype bfloat16
    done

`--adapter` is not optional in practice: without it `B = 0`, `dL/dA` is exactly zero and half
of every gradient vanishes. The dump prints `zero_block_fraction` so a run made without it is
identifiable after the fact rather than mistaken for a trained one; expect ~0.5 if it was
omitted and well under that otherwise.

The dump prints `seconds_gradients`, `seconds_features` and `seconds_total`. Those are the
first real timings for the extra forward and belong in this file when they exist -- no
estimate is recorded here, because CPU timings on the A100 box are contaminated by thread
contention with the live job (section 6.4).

If the rollout jsonl uses field names the loader does not know, it raises naming the file, the
line, the names it looked for and the keys the record has. Pass `--group-key` / `--task-key` /
`--reward-key` rather than converting the file.

Then, on the A100 box, under the venv that has the clustering dependencies:

    for CKPT in initial_lora globalstep24 globalstep49; do
      for MCS in 2 5 8; do
        PYTHONPATH=~/areal-selfevo ~/venv_probe/bin/python \
          -m selfevo.cluster_lora.interference_analyze \
          --dump ~/runs/interference_$CKPT.npz \
          --out  ~/runs/interference_${CKPT}_mcs${MCS}.json \
          --bootstrap 1000 --min-cluster-size $MCS
      done
    done

Read `contrasts` first -- `meds_minus_random_matched`, `meds_minus_elrea`,
`meds_minus_task` -- then check `resolved` and `resolution_floor` before believing any single
`mean_cosine`, and read `sketch_validation` to see how far the sketched cosines sat from the
exact ones on the pairs whose full gradients were kept. A `task` block reporting `skipped` is
the correct output for a single-task batch, not a failure.

The `min_cluster_size` sweep is the point of running three of them: it decides how many
experts the method would allocate, MEDS' shipped 2 was measured over-fragmenting, and the
sweep costs nothing because it is CPU-side on a dump that already exists.
