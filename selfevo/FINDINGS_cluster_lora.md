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

### 6.1b The probe is only as informative as the batch's k-spread

**MEASURED on the first real run, 2026-09-02.** Of the probe batch's 128 groups, **90 are
unanimous** (k=0 or k=8). A unanimous group has advantages identically zero under group-level
normalisation, so its GRPO gradient is exactly zero and it contributes NOTHING to any cosine.
Every MEDS-side number the probe reports therefore rests on **38 groups, not 128**, and a
reader who does not know that will over-read it.

This is a property of the BATCH, not of the code, and it is not something the probe should
paper over -- the 29-44% RL-silent share is a real measurement this project already tracks,
and rescuing those groups with a different normalisation would invent the signal under test.
What the probe owes the reader is the count, so `run_dump` prints it and both the dump
metadata (`n_zero_grpo_sketches`, `n_groups_informative`) and the analysis carry it.

**The damage is asymmetric, and that matters for the ELREA comparison.** The prompt-gradient
sketch is a plain likelihood over prompt tokens and does not depend on the rewards at all, so
0 of 128 were zero. A silent batch therefore weakens the MEDS side of the contrast while
leaving the ELREA side at full strength -- which biases `meds_minus_elrea` in a specific
direction and is exactly the sort of thing that reads as a finding. `experiments/m25/PLAN.md`
now carries a precondition of at least 60% non-unanimous groups before Gate 0 may be read;
this file records why that precondition exists.

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

### 6.4 The dump OOMed at 32B, because the wrong thing was budgeted

**Measured on an 80 GiB H100, 2026-09-02**, failing in the FIRST group about 17 s in, on all
three checkpoints, inside the LM head:

    torch.OutOfMemoryError: Tried to allocate 58.00 MiB. 50.31 MiB free. 78.23 GiB allocated.

The arithmetic at 32B, vocab 152,064, G=8. The old `group_losses` built every response in a
group as ONE padded batch and ran it through the causal-LM wrapper, whose forward ends in a
full-vocabulary `lm_head` over every position at once. The worst group is 8 x 2,089 = **16,712
padded tokens**: bf16 logits 4.73 GB, the `.float()` copy 9.47 GB, and the copy keeps the bf16
node alive, so **14.20 GB resident** -- against **61.02 GB of weights**, i.e. 75.22 GB of 79.18
usable, leaving under 4 GB for 64 layers of backward activations. `expandable_segments:True`
moved the shortfall from 114 MiB to 58 MiB, so it was capacity, not fragmentation.

**The defect was in this document as much as in the code.** Section 6.6 below budgeted the
full-gradient STORE (~1.07 GB, guarded by `--max-full-grad-gb`) and never budgeted the
per-group LOGITS, which at this scale are 13x larger and are the binding constraint. The guard
that existed could not fire, because the failure was inside the forward.

Worse than the diagnosis: the old code materialised a THIRD full-vocabulary tensor. It did
`.float()` and *then* `log_softmax` on the result, so a successful allocation would have held
bf16 + fp32 + fp32 = 23.7 GB for that group.

#### What the batch actually looks like

All 128 groups have n_seq=8, so B is constant and the whole spread is in T. Padded tokens per
group (B*T): min 2,392 / median 7,000 / p90 9,320 / p99 11,712 / **max 16,712**.

**The tail is a long PROMPT, not long responses.** Prompt lengths are min 56 / median 102 /
p90 215 / **max 1,330**, and the max group (id=11) is that 1,330-token prompt -- prepended to
all eight samples, so one outlier prompt multiplies straight through B. p75 to p95 is nearly
flat (8,856 -> 9,608) and then two groups sit far above everything: id=11 at 16,712 and id=94
at 11,712.

That is why **the chunk is a count of TOKENS, never of sequences.** At T=2,089 a single
sequence is already 2,089 positions, so a chunk of one SEQUENCE still allocates 0.59 GB of
bf16 logits for id=11, while a fixed B=2 would be twice the p90 group and 3.6x the median. A
token chunk is uniform across groups; a sequence chunk is not.

#### The fix, and why it does not change what is measured

1. **The trunk is called directly** (`_decoder_and_unembedding`), so the forward yields hidden
   states of B x T x 5120 -- 171 MB for the worst group -- instead of B x T x 152,064. LoRA
   still applies because PEFT replaces the Linear MODULES in place; that is PEFT's design and
   not this project's, so it is asserted by a test rather than assumed.
2. **The unembedding is applied in chunks of positions**, each freed as it goes. The loss is a
   sum over positions and chunking reassociates that sum, so the value and the gradient are
   unchanged -- exact arithmetic, not an approximation.
3. **The reduction is `log_softmax(logits, dtype=torch.float32)`**, which does the same fp32
   reduction WITHOUT first materialising an fp32 copy. Measured on CPU at bf16 over a 19k
   vocabulary: **bit-identical** to the old `.float()` form, while dropping the `dtype`
   argument and reducing in bf16 costs **0.044 in log-probability**.
4. **Per-chunk checkpointing** recomputes a chunk's logits in the backward rather than
   retaining them, so only one chunk's logits are ever alive.
5. **Whole-model gradient checkpointing, on by default.** Not optional at 32B: fixing the
   unembedding leaves the decoder's own retained activations as the binding constraint.
6. **The trunk is sub-batched over SEQUENCES** (`group_backward`), because checkpointing alone
   is not enough. Under it the decoder still retains one hidden-state tensor per layer:
   `layers x tokens x hidden x 2` is **640 KB per token** at 64 x 5120, so the worst group's
   16,712 padded tokens retain **10.95 GB** on top of 61.02 GB of weights -- which would have
   OOMed again after the head was fixed. Sequences in a GRPO group are independent (causal
   attention, right padding) and the loss is a sum over them under a denominator that does not
   depend on the split, so `sum_s L_s` has gradient `sum_s grad L_s`: each sub-batch is
   backwarded immediately and its activations freed, accumulating into the same `.grad`.

   `--activation-budget-gb` (default 6.0) gives 9,830 tokens per forward, so id=11 at T=2,089
   runs as **two forwards of four sequences** retaining 5.48 GB instead of 10.95, while the
   median group (T=875) still runs as ONE forward -- only the tail pays for the split.

   The subtle failure this could have introduced, and it is a test: a sub-batch must NOT
   recompute its own advantages. GRPO advantages are the rewards centred and scaled WITHIN the
   group, so a slice that recomputed them would centre a subset -- for a two-sample slice,
   nearly meaningless -- and the run would complete with plausible numbers. The parent's
   advantages are passed down and the test uses rewards chosen so every slice has a different
   mean from the whole.

Points 4 and 5 both differentiate a RECOMPUTED forward, which is only exact if the forward is
deterministic. So the dump **refuses to run with any dropout active**, naming the modules --
a recomputed dropout mask would give a plausible gradient with no error anywhere, which is the
failure class this project distrusts most and has already recorded once for checkpointing. The
shipped configs set `disable_dropout` and `lora_dropout` 0, so the assertion passes and
train/eval are provably equivalent.

Equivalence is tested, not argued. Loss and every LoRA gradient are compared against the
ORIGINAL unchunked path at chunk sizes 1, 3, 7 and 10,000, for both losses; at sub-batch sizes
1, 2, 3 and 8; checkpointed and unchecked gradients are asserted BIT-equal; and a test records
the size of every unembedding output and requires that none exceeds `chunk x V`, which is the
guarantee the fix exists for rather than a consequence of it.

**Projected peak for the worst group at the defaults**: 61.02 GB weights + 5.48 GB retained
trunk activations + ~1.4 GB one layer's recompute + 4.0 GB head chunk + 0.09 GB hidden states
= **about 72 GB of 79.18**, roughly 7 GB spare. That is a projection from the measured
constants, not a measurement; the run itself will settle it.

#### The budget, as a guard that fires before the forward

`LOGIT_PEAK_BYTES_PER_ELEMENT = 12` -- bf16 matmul output (2), fp32 log-softmax output (4),
and during the recomputed backward the gradients of both (4 + 2). It is **derived, an upper
bound, and not measured**, so it is exposed as `--logit-peak-bytes` for a box that measures
otherwise. The old unchunked path cost 10 bytes per element over the whole group at once.

`--logits-budget-gb` (default 4.0) is a ceiling on the head's transient memory, and the chunk
is DERIVED from it: at vocab 152,064 that is **~2,190 tokens**, inside the 2,048-4,096 band the
length distribution calls for. Deriving from a memory budget rather than fixing a token count
is what makes the same setting mean the same memory at another vocabulary -- a count that is
comfortable at 32k is five times the intended footprint at 152k. Sizing near p90 (9,320) would
leave about the ~10 GB headroom at which the run died.

`assert_logits_fit` runs **before any forward**, twice: once for the whole batch's worst group
before the loop, and once per group. It refuses if the chunk exceeds the budget, or if 1.5x
the estimate exceeds free device memory -- above 1 because the estimate covers the head only
and the decoder's activations share the same pool. The refusal names the group, its token
count, the chunked estimate and what the unchunked one would have been.

**VERIFIED ON THE BOX 2026-09-02.** On all three adapters the guard refused cleanly before the
forward, with a named group and a full estimate, instead of dying in the LM head. Two defects
showed up only by running it, and both are now fixed and tested.

#### The advice was inverted in one branch

`assert_logits_fit` has two raise paths and the second reused the first's closing clause:

    ... but only 4.60 GB is. Lower --chunk-tokens, raise --logits-budget-gb, or give the
    probe a card with more free memory

In the FREE-MEMORY branch, raising `--logits-budget-gb` is strictly counterproductive. The
chunk is DERIVED from the budget, so a larger budget makes a larger chunk, which makes a
larger `peak` -- the very quantity being compared against free memory. Only a smaller chunk,
or more free memory, can satisfy it. The message sent the reader in exactly the wrong
direction at the moment they were stuck.

It now says which way to move, and computes the value:

    ... but only 4.2 GiB is. Unchunked this group would have needed 14.9 GiB. LOWER the
    budget: set --logits-budget-gb 2.8 or less (equivalently --chunk-tokens 1680 or less),
    or free memory on the card. Do NOT raise --logits-budget-gb: the chunk is derived from
    it, so a larger budget means a larger chunk and a larger requirement

Every number needed was already in hand -- free bytes, the headroom factor, the vocabulary,
the per-element cost -- so leaving the arithmetic to the reader was a choice, and the run that
hit it had to derive 2.5 by hand. `satisfying_plan` computes the largest budget and chunk that
pass, and a **round-trip test** feeds that value back into the guard and requires it to pass,
so the suggestion cannot drift from the check it is offered to satisfy. Units were wrong too:
the flag is GiB (`logits_budget_gb * 1024**3`) while the message printed decimal GB, ~7% adrift
at this scale; the guard now prints GiB throughout and floors rather than rounds, because a
suggestion rounded up does not satisfy.

#### The refusal was systematic, not a tail case

The group that was refused is **group 2, with 8,824 padded tokens -- p75 of the batch**
(p75 = 8,856), not the 16,712-token outlier. Because the per-chunk requirement is
near-constant across groups, a fixed `--logits-budget-gb 4.0` on an 80 GB card already holding
61 GB of weights leaves ~4.60 GB free against a 6.44 GB requirement and would have refused
**the majority of the 128 groups**. That is a misconfigured default, not a hard batch.

So the default is no longer a fixed number. `resolve_logits_budget` derives it from MEASURED
free memory -- `free / headroom`, capped at the 4 GiB ceiling -- and records which branch it
took in the dump's metadata and on stdout. An explicit `--logits-budget-gb` is honoured
exactly, including one that will not fit: a flag the caller set is a statement of intent, and
silently overriding it would make the refusal that follows unattributable to anything they
did. On the real card the derivation picks 2.8 GiB; on an emptier one it stays at the ceiling.

There is a second consequence, and it is a better operating point than was planned for: a
smaller chunk fits in the headroom a concurrent TRAINING run leaves, so the probe does not
need the card to itself. The probe now runs at `--logits-budget-gb 2.5` alongside the A0
baseline.

#### The full-gradient store took the wrong groups, and that silently disabled the validation

Two defects, both found only by running the probe on the real batch.

**It stored the FIRST N groups.** With 90 of 128 unanimous, `--full-grad-groups 8` stored
k = [8, 0, 0, 4, 8, 8, 8, 0]: seven exactly-zero gradients and one of norm 1.28e-4.
`_sketch_validation` needs two non-zero gradients to form a single pair, found one, and
correctly reported that it could not validate -- so **the sketch went unvalidated, and that
selection could not have validated it on any rerun**: drawing 8 arbitrary groups from a
70%-silent batch yields two non-zero about 0.5% of the time. The store now selects from the
INFORMATIVE groups (`Group.is_silent`, tested on the reward spread rather than on a k
threshold, since k=0 and k=G are the same case), which makes N=2 sufficient.

The failure is now a named condition rather than a sentence to notice: the dump computes
`sketch_validation_status`, prints it, and writes it to the metadata as either
`ok: N non-zero stored gradients` or `IMPOSSIBLE: ... the sketch is UNVALIDATED for this run`,
so a script can refuse on it. The analysis carries a matching `status` field.

**It was upcast on the card before being moved.** `gr.float().cpu()` builds a full fp32 copy
on the accelerator first -- twice the transfer, and 134 MB of device memory per stored group
at 33.5M LoRA parameters. Free memory was measured falling 4.60 -> 3.44 -> 2.44 GB across
budget attempts, almost exactly 8 x 268 MB, which is what forced the budget down to 1.5 GiB,
the chunk to 882 tokens and the run to 16 minutes per adapter. The store is read only at the
very end, so `gr.cpu().float()` transfers in the gradient's own dtype and upcasts on the host,
and `empty_cache()` is called on the few groups that store so the freed blocks are visible to
the guard's next check. That frees ~2.1 GB.

The two levers used to point in opposite directions -- storing fewer gradients freed memory
but made validation less likely to have anything to validate. With the store off the card and
selected from informative groups, `--full-grad-groups` is a statistical choice again.

The ordering cannot be observed on a CPU box by DEVICE, but it can by DTYPE: under the correct
order `.cpu()` is called on a bfloat16 tensor, under the wrong one it only ever sees float32.
That is the test.

**Not done, deliberately.** All eight sequences in a group share an identical prompt prefix, so
at id=11 the same 1,330 tokens are forwarded eight times; sharing that prefix would cut the
worst group by most of its cost but needs KV-cache reuse across the eight continuations, and it
is not worth taking on without a proof that the resulting gradient is identical. The chunked
path plus gradient checkpointing is sufficient. Sharding the model across cards was likewise
left alone: it adds complexity and does not remove the waste.

### 6.5 Other costs of the dump

* **Per group stored**: 2 sketches x 8192 float64 = **128 KB**, plus the behavioural vector
  (`n_layers/2` floats). Negligible; 128 groups is ~16 MB.
* **Full gradients**: `n_groups x n_lora_params x 4` bytes. This is the store the
  original `--max-full-grad-gb` guard covers, and it was never the binding constraint. For a 134 MB adapter (~33M params
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

### 6.6 The rollout schema

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
  Measured `zero_block_fraction` = 0.50 on the CPU fixture, and **confirmed at exactly 0.500
  on `initial_lora` in the first real run** -- that is the predicted effect, not a defect, and
  the trained adapters sit below it. The cosines are then taken over the
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

Mutation testing is `selfevo/tests/mutate_cluster_lora.py`, run against COPIES at
`/home/ubuntu/mutcopy` and `/home/ubuntu/mutcopy2` whose `cluster_lora` modules were asserted
sha256-identical to the originals before starting; every target's sha256 is re-checked after every restore and again
at the end. The harness refuses to start unless imports resolve inside the copy. It runs in two
arms because the two halves live in two venvs, and a mutation whose tests cannot run under the
chosen interpreter is reported NOT APPLICABLE rather than killed -- a mutation that was never
exercised is not a passing one. Kill table in section 10.

Eighteen tests exist only because the mutation harness found survivors or a run on the box
did, and each pins a
property no other test could see: bit-equality in `unchanged()`, the merge verification
actually firing, per-parameter hashing in the sketch, the control's own size-match assertion,
the prompt loss covering the prompt positions, the trunk actually splitting into sub-batches,
the harness's own anchors still matching, the refusal pointing the reader the right way, the
suggested value actually satisfying the check, the budget following measured free memory, the
store selecting informative groups, the unvalidatable case being named, and the store reaching
the host before it is upcast.

`git status --porcelain` shows only files in this agent's territory:
`selfevo/cluster_lora/`, eight `selfevo/tests/test_cluster_lora_*.py`,
`selfevo/tests/mutate_cluster_lora.py` and this file. `selfevo/routing/cluster.py` was NOT
edited: `ClusterRouter.key_fn` is already `Callable[[RoutingContext], str]` and
`ClusterLoRAKeyFn` satisfies it as-is.

## 9. What still needs a GPU, and what does not yet reach the trainer

**Reach, as of 2026-09-02.** This agent built two seams and deliberately did not connect
them, because the files that connect them belong to other agents. They have since been wired
by the agent that owns those files, in `selfevo/cluster_lora/wiring.py`; what follows is what
this agent established, and the wiring's own reach is that agent's to report.

1. **`ClusterLoRAKeyFn` satisfies the `ClusterRouter.key_fn` seam.**
   `test_the_key_fn_partitions_a_batch_through_the_real_ClusterRouter` drives a real
   `ClusterRouter.route_batch` and gets this partition back. What was missing was the
   features: `PPOActor._route_groups` builds `RoutingContext.extra` from `group_features(...)`
   only, and `extra` is `Mapping[str, float]` and cannot carry a vector, so a run needed one
   `begin_batch(unit_ids, features)` call at that seam. Until that existed `partition=meds`
   REFUSED rather than collapsing to one adapter -- the intended behaviour, and not a working
   arm.
2. **`ClusterAdapterSet` was not called by `FSDPEngine.train_batch`.** The isolation guard is
   proved on a real PEFT model with a real optimizer, but the engine ran one adapter for the
   whole batch; wiring it means microbatching by cluster inside the engine.

Two things this agent did NOT verify and which the wiring must carry on its own evidence: that
the features reaching `begin_batch` in a live run are the ones this module computes, and that
the engine's own gradient clipping and scheduler still see each cluster's update correctly.
The isolation guard here is proved against a plain `AdamW`, not against the engine's optimizer
path.

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
to the originals for every `cluster_lora` module before starting and verified restored clean
afterwards. **90 distinct mutations, two arms, every one killed and no SKIPs.**

| arm | interpreter | applicable | killed | survivors |
|---|---|---|---|---|
| `torch` | `~/venv312b` | 78 | **78** | none |
| `cluster` | `~/venv_probe` | 29 | **29** | none |

17 mutations run on both arms, so the union is 90: every mutation is exercised somewhere. By
area: the dump 44, the partition and control 15, adapter isolation 11, the analysis 9, the
sketch 6, the merge 5. Seventeen of them were added only after the probe was RUN on the real
box -- covering the memory guard's advice, the derived budget, the full-gradient store's
selection and its paging -- and four of those restore a clause the code actually shipped with.

**Ten defects survived a first pass and each produced a new test.** They are recorded
because the tests that missed them all looked entirely reasonable, and because five of the
seven would have produced a plausible number rather than an error:

1. *`zero_grad(set_to_none=False)` survived the isolation guard.* The guard trained only one
   expert, so the others never held a gradient at all and the optimizer skipped them either
   way. The leak needs an expert that was trained and is then IDLE -- the normal case, since
   most clusters are absent from most batches.
2. *The `requires_grad` check inside `only()` survived.* It is defensive, so nothing fires it
   while PEFT behaves; a stand-in that activates without freezing now fires it.
3. *Masking the prompt NLL to the RESPONSE survived* a test that only asked whether the two
   sketches were parallel -- an advantage-weighted loss and a plain likelihood are never
   parallel even over identical tokens. Now checked against an independent reference over the
   prompt positions.
4. *Dropping the CountSketch's random signs survived*, because the unsigned estimator is still
   unbiased on zero-mean vectors. Caught on two non-negative vectors with disjoint supports,
   where every hash collision adds.
5. *A bootstrap that stopped resampling survived*, because the two standard deviations were
   `0.0` and `5.6e-17` and the ratio compared float noise. Now checked on interval WIDTH.
6. *Disabling the trunk sub-batching survived every gradient-equality test*, because one
   forward over the whole group gives exactly the same gradient -- and OOMs at 32B exactly as
   before. Equality tests cannot see it; the number of trunk forwards can, so that is now
   asserted directly at five sub-batch sizes.
7. *Rewriting the loss for the OOM fix left one anchor matching zero lines.* The harness
   reported `anchor appears 0x` and counted it as NOT a kill, which is the correct discipline
   -- but only because someone read the output. A guard that is never exercised looks exactly
   like a guard that passes, so `test_every_mutation_anchor_still_occurs_exactly_once` now
   parses the harness and requires every anchor to occur exactly once in its target. It is
   source-only, so it runs under both interpreters.

8. *The guard's advice was inverted in one branch, and no test looked at advice at all.*
   Every test asserted that the refusal FIRED, which it did, correctly, on the real card.
   None asserted what it then told the reader to do, so reusing the wrong clause was
   invisible until a person followed it. The message is now checked for direction, and a
   round-trip test feeds the value it suggests back into the guard and requires it to pass.

9. *The full-gradient store took the first N groups, and every test of it used a fixture
   whose groups were all informative.* The selection was never wrong on any batch the tests
   built, so nothing failed until a batch that was 70% unanimous reached it -- and then the
   symptom was not an error but a validation that quietly could not run. The fixtures now
   include a mostly-silent batch with its informative groups placed LAST, which is the shape
   that breaks a first-N selection.
10. *The store was upcast on the accelerator before being moved to the host.* No test could
   see it, because on a CPU box the device is the same either way. It is observable by DTYPE
   rather than by device: under the correct order the host transfer sees bfloat16, under the
   wrong one only float32.

Lessons 6, 7, 8 and 9 are the general ones. An optimisation whose whole purpose is to use less
memory cannot be validated by asserting the answer is unchanged, because the answer is
unchanged when the optimisation does not happen -- the mechanism has to be observed, not the
result. And a mutation harness silently decays as the code it points at is rewritten. The third:
a guard is not finished when it refuses correctly -- what it says next is part of it, and
asserting only that it raised leaves the half a human actually reads untested. The fourth:
a fixture built from healthy data cannot test a selection rule, because every selection looks
right on a batch where every choice is a good one -- the fixture has to contain the pathology
the rule exists to handle.

### 10.1 A process failure worth recording

Commit `31ca3056` contains nine files that are not this agent's -- another agent's gold
batch-construction work, which was staged in the same shared checkout at that moment.

The brief said to stage by name, and `git add` WAS given explicit paths. That was not
sufficient, and the reason is worth writing down: **`git commit` commits the INDEX, not the
paths you added.** On a checkout three agents share, another agent's `git add` between your
`status` and your `commit` puts their files into your commit whatever you named. The defence
is a pathspec-limited commit -- `git commit -F msg -- <paths>` -- which commits only those
paths regardless of what else is staged, and which this agent used for every commit after the
incident.

The mistake was not repaired. No content was lost, the worktree matched HEAD, and by the time
it was noticed two other agents had already committed on top; rewriting shared history to fix
an attribution error is how a race becomes lost work.

## 11. How to run it on the H100 box

The dump needs only torch, peft and transformers, so it runs in the H100's own venv with no
installs. Nothing here starts a training job.

    # one dump per checkpoint; three checkpoints gives the early/mid/late comparison
    for CKPT in initial_lora globalstep24 globalstep49; do
      python -m selfevo.cluster_lora.interference_dump \
        --model  <base model path> \
        --adapter ~/runs/.../$CKPT \
        --rollouts ~/runs/harnessT_trunc/rollouts_math64_probe.jsonl \
        --out    ~/runs/interference_$CKPT.npz \
        --sketch-dim 8192 --full-grad-groups 8 --device cuda --dtype bfloat16 \
        --logits-budget-gb 2.5 --activation-budget-gb 6.0
    done

`--logits-budget-gb` may be omitted entirely: it then derives from measured free memory,
which on a card holding 61 GB of weights picks about 2.8 GiB. It is given explicitly above
because the probe is sharing the card with the A0 baseline, and an explicit value is honoured
exactly rather than re-derived under whatever happens to be free at that instant.

The chunked unembedding, the sequence sub-batching and gradient checkpointing are on by
default and are what make this fit at 32B; `--no-checkpoint` and `--no-gradient-checkpointing` exist only to reproduce the
old memory behaviour and will OOM there. If a group is still refused, the message names it and
its estimate: lower `--logits-budget-gb` (the chunk follows) rather than raising it, and
lower `--activation-budget-gb` if the failure is in the trunk rather than the head. Use
`~/runs/harnessT_trunc/rollouts_math64_probe.jsonl`, whose per-response rewards are already
lifted into group-level `rewards` lists.

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
