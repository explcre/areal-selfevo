# M25 experiment plan — MEDS-routed LoRA experts

The paper's method (GOAL.md M25). One spec so every arm is run the same way and no agent
improvises a control. Decision rules are written down BEFORE any number exists.

## Gate 0 — the interference probe (no training, minutes on one H100)

`selfevo/cluster_lora/interference_probe.py` on a 32B checkpoint + one saved rollout batch,
FOUR partitions on the same batch: (1) MEDS behavioural clusters, (2) size-matched random,
(3) ELREA-style prompt-gradient clusters, (4) task-label calibration if the batch spans tasks.

**Amended 2026-09-02 from `FINDINGS_cluster_lora.md`, before any GPU ran:**
- **The task-label calibration cannot reach 1e-5.** A CountSketch cosine has s.e. ~1/sqrt(dim),
  so the floor is `3/sqrt(8192) = 0.0331`; a dense projection at the needed size would be 3 TB.
  Every block carries `resolution_floor` and a `resolved` flag, and an unresolvable cosine is
  reported as BELOW THE FLOOR, never as a number. Against a reviewer citing 2608.03573 the claim
  is "below our floor", plus the stored full gradients for the first 8 groups, which are the only
  place a 1e-5 figure could be checked at all.
- **Require K >= 4 clusters.** At K=2 there is one pair and the control's mean cosine was measured
  swinging -0.12 to -0.22 across seeds; no discrimination is possible.
- **Sweep `--min-cluster-size`, do not inherit it.** MEDS ships 2, which over-fragments (6
  clusters + 2 noise where 4 + 0 was right); every extra cluster is an expert trained on fewer
  groups. Start at 5 for the 128-group MATH batch; the sweep is CPU-side and cheap.
- **Read the bootstrap SPREAD, not containment.** The CI need not contain the point estimate
  (resampling duplicates bias a cluster sum toward its duplicated directions). The discriminator
  is the spread: std 0.0085 for a true partition vs 0.064-0.090 permuted.
- **`cancellation` varies only through `sum_c ||g_c||`**, since `sum_c g_c` is the batch gradient
  for every partition. Read it as internal coherence, not as a conflict measure.
- **Adapter identity must be overlap-matched, not label-matched** — HDBSCAN renames clusters on
  every refit (churn 1.0 naive, 0.0-0.083 fixed). Any arm must report `churn` per step.
- **The MEDS features need an extra forward pass, and its cost is MEASURED at ~half a step, not a third.**
  Under LoRA the base is frozen, so a training step is ~2 forward-equivalents and the bound is `f/2`,
  not the `f/3` a full fine-tune would give. CPU proxy across three widths: 0.319, 0.451, **0.483** of a
  step as the model widens. **A0 must therefore be given half a step of extra budget** (as extra rollouts)
  for the matched-budget arm A0'. Truncating the trace at the answer token is exact but **buys nothing**,
  because MEDS reads the token inside `\boxed{}` which sits at the END of a math rollout. On GPU the
  largest unimplemented saving is batching the trace (it currently runs one sequence at a time against a
  packed microbatch); the largest realised saving is the one-row unembedding dot against a 151,936-row vocab.
Per partition: pairwise cosine of per-cluster LoRA gradients, conflict rate, cancellation
`||Σg_c||/Σ||g_c||`, sizes, bootstrap CI on mean cosine. Three checkpoints (early/mid/late) if
available, since interference plausibly grows as the policy specialises.

**PRECONDITION on the probe batch, added 2026-09-02 after a measured false-negative risk.**
A GRPO group whose rollouts all score alike (k=0 or k=G) has every advantage identically zero and
therefore an EXACTLY zero gradient. Measured on the first real dump: **90 of 128 groups were
unanimous, so 90 sketches were exactly zero** — the k-histogram (k=0: 28, k=8: 62) predicts the
count exactly, so this is arithmetic, not a bug.

Consequences, both of which invalidate the gate rather than merely weakening it:
- **The measurement would rest on 38 groups, not 128**, and every partition's mean cosine would be
  computed over a set where 90 members are zero vectors — pulling MEDS, random-matched and ELREA
  toward the same value for a reason about advantage sparsity rather than about clustering. A
  "MEDS approximately equals random" reading would satisfy the STOP rule below **while meaning
  nothing about clusters.** That is a false negative that kills the method on an artifact.
- **The damage is asymmetric.** The ELREA prompt-gradient sketch is unaffected (0 of 128 zero), so
  the MEDS-vs-ELREA contrast would compare 38 usable groups against 128.

**So Gate 0 may only be read on a batch where the NON-UNANIMOUS fraction is high.** Require at
least 60% of groups with 0 < k < G, report the k-histogram alongside every probe result, and treat
any dump whose zero-sketch fraction exceeds 40% as UNREADABLE rather than as evidence. This is the
same regime error as the retracted M7 teacher demotion and GOAL.md 2c's training-corpus gap: on
data the model has largely solved or wholly failed, the quantity of interest does not exist.

**Related probe defect (author's call, not a plan item):** `--full-grad-groups N` stores the FIRST
N groups in file order, which on a 70%-unanimous batch has ~0.5% chance of yielding the two
non-zero gradients `_sketch_validation` needs — it stored 8 groups of which 7 were unanimous, so
the sketch went unvalidated. Store the first N NON-UNANIMOUS groups instead. Separately, the store
is only read at the end, so paging it to host memory frees ~2.1 GB of card and makes N a
statistical choice rather than a memory one.

**Decision rules, fixed now:**
- Proceed to arms iff MEDS mean cosine is below the random partition's by more than the
  bootstrap CI **and** MEDS conflict rate is materially above the cross-task calibration.
- MEDS ≈ random → the clusters are noise; **stop**, no adapters get built on top.
- MEDS ≈ prompt-gradient → rollouts are unnecessary and the method reduces to ELREA-in-RL;
  report it and reconsider before spending arms.

## Arms — identical everything except the partition

All: Qwen2.5-32B-Instruct + LoRA r=32 on q/k/v/o, 4×H100 as 2 training (fsdp d2) + 2 rollout
(sglang TP=2), same seed, same data order, same steps, same generation cap (FIXED — no harness
ladder in this experiment), same eval samples. Adapters merged (summed) at eval for every
multi-adapter arm; the merge operator is held fixed across arms.

| arm | partition | adapters | purpose |
|---|---|---|---|
| **A0** | none | 1 shared | the baseline everything is relative to (vanilla GRPO+LoRA) |
| **A1** | MEDS clusters, kNN-stabilised | N + 1 shared for noise | **the method** |
| **A2** | size-matched random (A1's realised N and sizes, feature-blind, seeded) | same as A1 | **mandatory control** — "clustering" vs "more adapters" |
| **A3** | ELREA-style prompt-gradient clusters, same N | same as A1 | required ablation — are rollouts needed |
| **A4** | `{k=0, k>0}` with gold-SFT rows into the `k=0` adapter | 2 | **LSPO** (closest published baseline; no public code) |
| **A5** | none; gold row substituted on `k=0` groups | 1 shared | **DyME** rule — separates "gold helps" from "routing helps" |
| A6 (opt.) | task label (math vs code) | 2 | only if a mixed batch is used; reproduces the cross-task calibration |

Order: A0, A1, A2 first (pilot at S steps). Extend to full length only if A1 − A0 clears the
noise floor. A3 next (decides the "rollouts needed" claim). A4/A5 need the gold path
(`selfevo/gold/`) and run once it lands.

## Matched budgets — the AI2 objection, pre-empted

- Same rollout count, same steps, same cap, same eval samples in every arm.
- The clustering cost is COUNTED: if MEDS needs an extra forward, its FLOPs are reported per
  step and A0 is optionally given the same FLOPs as extra rollouts (budget-matched A0').
- Eval at matched inference budget: same n per problem; maj@k at the same k for every arm.
- Feedback budget: verifier calls per arm logged and reported (GOAL.md §4 has this NOT MET —
  this experiment closes it).

## Benchmarks and statistics

- **OlympiadBench** 675 problems, noise floor 0.027; paired per-problem McNemar vs A0; report
  the difference with its CI, never two point estimates. Any decision that READS eval uses the
  committed held-out 250/250 split; the full set is reported once, at the end.
- **LiveCodeBench v6** 175 problems / 7000 tests (`11757e61`); report `accuracy` AND
  `accuracy_all`; the 12 s per-test clock and `max_tokens=16384` are validated on A0's first
  eval before any comparison is made (they have never met a live model).
- Per arm, alongside accuracy: realised cluster sizes per step, cluster-id churn between
  steps, fraction of groups assigned to noise, and the probe's cosine on the arm's own
  checkpoints — so a null can be diagnosed (no interference to remove vs adapters failed to
  remove it).

## The claims and which comparison carries each

| claim | comparison |
|---|---|
| the method helps | A1 − A0 > noise floor |
| it is the clustering, not extra parameters | A1 − A2 |
| behavioural clusters beat prompt-side clusters (rollouts needed) | A1 − A3 |
| beats the published rules at matched budget | A1 − A4, A1 − A5 |
| mechanism | probe: MEDS cosine < random cosine, on the arm's own checkpoints |

## Launch blockers for A1/A2 — established 2026-09-02, must be cleared BEFORE the arms run

1. **`max_lora_rank` must be raised.** A merged adapter has rank `sum_c r_c`, but every shipped
   config sets `sglang.max_lora_rank: ${actor.lora_rank}` and `experiments/harness/lora30b.sh:500`
   **asserts** `sg.max_lora_rank == a.lora_rank`. That assertion must become
   `== actor.lora_rank * len(roster)` or **the rollout server will reject the merged adapter** and
   the arm cannot serve at all.
2. **The roster must be sized with headroom.** `MEDSPartitioner._resync` allocates a fresh stable
   id for any raw label matching nothing and `_next_stable` grows without limit, while
   `adapter_roster()` is fixed at process start. `begin_cluster_batch` raises `ClusterWiringError`
   — loud, but **fatal mid-training**, and FINDINGS 5.1/5.2 say a blob splitting off a fragment as
   the buffer grows is EXPECTED. So either size the roster with headroom or raise
   `min_cluster_size`, or the run stops at an arbitrary step hours in.
3. **`fsdp.per_layer_optim_step` must stay False.** `PerLayerOptimWrapper` selects param groups by
   `p.requires_grad` **at construction**, when only `names[0]` requires grad — every other expert
   would accumulate gradients that are never applied, silently. Latent (defaults False), but it
   must not be turned on for these arms.

## Constraints the arms inherit, which shape what can be claimed

- **Gradient clipping is global.** One `fsdp2_clip_grad_norm` over all parameters, once, after every
  cluster has accumulated — so the clip factor is shared and one cluster's large gradient scales
  down every other expert's update. Worse, `if not math.isfinite(grad_norm): zero_grad()` discards
  **every** expert's update for that step when one cluster goes non-finite. **Per-cluster learning
  rates are not independent and cannot be** while there is one clip and one step. Any claim about
  per-cluster adaptation must be read against this.
- **Read `grad_norm`, never `cluster_lora/loss/<name>`**, whenever a cluster spans more than one
  microbatch: `cluster_loss` returns `held[0]`, so the logged loss is the LAST microbatch's, while
  `norms[name]` accumulates over all of them.
- **The merge operator is `sum` with weights 1.0 and is HELD FIXED across arms.** Each expert
  already carries the whole batch as its denominator, so summing reconstitutes what a single shared
  adapter would have accumulated — exactly A0. A mean would divide the deployed update by K, and
  `A1 - A0` would then read a K-fold learning-rate difference as a method effect. The operator moves
  the deployed scale without moving any training metric, so it must never vary between arms.
- **The training forward deliberately sees ONE expert; only inference forwards see the sum.** In
  process the sum is reached by ACTIVATION rather than merging (`LoraLayer.forward` adds each
  active adapter's contribution; measured equal to the merged adapter at 1.19e-7 against a 0.597
  single-expert difference). Applying it to the training forward would make the arm its own
  baseline.


## AMENDED 2026-09-02 (PI): run the routing comparison that does NOT depend on cluster-LoRA

Gate 0 returned a null — MEDS clusters are indistinguishable from a size-matched random partition
on GRPO-gradient separation (p 0.44-0.74 at every setting, sign flipping, corroborated on exact
gradients at mean cosine +0.0025 with 46% of pairs negative). One confirmatory run is scheduled at
`globalstep199`, but the method is not the only thing this repo can test, and the rest has been
waiting on it unnecessarily.

**The experiment that is runnable today, with everything already built and config-reachable:**

| arm | config | what it is |
|---|---|---|
| **A0** | no `group_routing` | vanilla GRPO + one shared LoRA. RUNNING on decontaminated DeepMath |
| **R-rule** | `router=rule` | the hand-written predicate. Our own audit showed it is behaviourally identical to a solve-rate threshold, i.e. **the published DyME + DAPO rule**. The baseline to beat |
| **R-learned** | `router=contextual`, **`credit=prompt`** | the learned router WITH the credit fix. **Never GPU-tested** |
| **R-control** | `router=random` at R-learned's REALISED proportions | the matched control. Differs only in WHICH unit gets which mode |

**Why this is the paper's live question.** GOAL.md Result 7 records the null this overturns or
confirms: `router=contextual` vs `router=random` at matched proportions, `globalstep149`, MATH-500
**-0.0020**, OlympiadBench **+0.0000**. That was measured with `credit="batch"` — the setting since
proven, as an identity, to make every arm converge to the same parameters. The learned router in
that null **could not** have learned. `credit="prompt"` is the fix, validated on CPU at 0.098 ->
0.779 subset contrast with a shuffle control collapsing it to 0.102, and it is already reachable
from config (`actor.py:558`).

So the claim under test is precise and pre-registered: **with per-prompt credit, does the learned
router beat its rate-matched control at matched budget — where with batch credit it provably could
not?** A null here is also publishable, because the identity explains the mechanism either way.

**Corpus**: decontaminated DeepMath-103K (53.9% informative, versus 23.3% on MATH — the earlier
null ran on the saturated corpus). **Benchmarks**: OlympiadBench (675, gold 675/675) and
LiveCodeBench v6 (175/7000, gold 175/175). **Controls**: rate-matched at realised proportions,
subset contrast rather than L1-from-uniform, paired McNemar with the error bar on the DIFFERENCE.

**PRE-REGISTERED for R-learned / R-control, fixed 2026-09-02 before any number exists.**
`solved_advantage=0.5` — the value Result 7 used, so the credit fix is the SINGLE changed variable.
0.0 is not an option: an SFT decision writing zero makes the two arms byte-identical, a guaranteed
null for a trivial reason.

But 0.5 is recorded as *slow-acting and harmful*, not inert: truncation 4/60 at step 199 -> 59/60 at
224, MATH-500 0.52 -> 0.19 — **and the random control collapses just as completely** (0.3080 with
499/500 truncated), so the constant causes it, not the routing. Mechanism: a positive constant on
solved groups reinforces every token of an already-correct rollout uniformly, including the
tendency to continue, with no term opposing continuation — so termination decays. Same altitude
error as gold: the advantage seam multiplies tokens the model ACTUALLY EMITTED, so a constant
there reinforces the emission pattern rather than the content.

1. **Primary evaluation at `globalstep149`**, matching Result 7's point exactly.
2. **Stop rule**: halt an arm when `route/truncated_row_fraction` > **0.20**; report the crossing step.
3. **Both arms compared at `min(149, first crossing in either arm)`.** Different steps is not a
   comparison. If the crossing precedes 149 the comparison does not exist and **the collapse
   threshold IS the result**, reported as such rather than evaluated at whatever checkpoint survived.
4. R-learned's REALISED mode proportions reported regardless — R-control cannot be configured
   without them.

The earlier collapse data is 1.5B on MATH; nobody has measured the threshold at 32B on DeepMath,
and `route/truncated_row_fraction` is the live instrument those runs lacked. Earlier, later or
absent is a finding either way.

**ADDED 2026-09-02 (PI): `r_covariate` — code-as-policy over covariates ONLY, ahead of the
published baselines.** The feature audit found that of the seven observability features, **two are
k-functions** (`solve_rate`, `reward_std`) and **five are covariates** (`mean_response_len`,
`len_dispersion`, `mean_logprob`, `logprob_dispersion`, `truncated_fraction`) — and that the
102/102 rule-collapse constrains NOTHING about covariates, since four of the five were held constant
across those contexts. **Whether any covariate carries routing information beyond the pass count is
therefore untested**, and under binary rewards k is all the outcome statistics contain, so this is
the question the method rests on.

`router=code_policy` is the direct test and the executor is already hardened (AST allowlist,
subprocess cost-vetting, ~150 adversarial policies, 26/26 mutants, `PolicyRejected` at
construction); `actor.py:610` already passes the seven features as `extra`. The arm's policy must
read **only covariates and never a k-function** — that exclusion IS the experiment — with the source
in a file under `experiments/`, fixed in advance, and archived with the run.

Control and protocol identical to the others: `router=random` at `r_covariate`'s REALISED
proportions, `solved_advantage=0.5`, evaluated at `min(149, first crossing)`, name-pinned.

**Reading either outcome.** Beating its matched control means covariates carry signal beyond k and
the method has a mechanism. Failing to means the pass-count collapse is the whole story at this
scale — a strong negative that would explain every routing null in this repo. Both are more
informative than reproducing DyME or Co-Harness, which is why this arm precedes them.

**The LLM-generator idea sits behind this, not beside it.** The fuller design is an LLM writing that
policy source from observability and MEDS signals. The safe executor exists; what is missing is the
generator, and the MEDS cluster id is not in the feature set so a policy cannot currently read it.
But a generator has nothing to find if no covariate carries signal, so `r_covariate` is its
precondition rather than a detour.

**Sequencing on one box**: A0 continues (it is the baseline for every arm), the confirmatory Gate 0
dump takes ~25 minutes at `globalstep199`, then R-learned, then R-control configured from
R-learned's measured proportions. R-rule last if time allows, since DyME and HPT are also cloned
and their published numbers exist.

## Discipline

- Config asserted from `process_env.json` inside every run, never the launcher line.
- Checkpoint every C steps with C chosen so a lost box costs < 1 h; every eval's
  `results.json` is pulled off the ephemeral box immediately.
- 3.3 GB actor headroom at 32B: no batch increase, no `kl_ctl>0`; `DRY_RUN=1` before any
  config change.
- Orphans reaped before diagnosing any second attempt; artifacts read, never echoes.
- **Loss-weighting: audit DONE (`603230a8`, `selfevo/FINDINGS_loss_weighting.md`).** The
  published bug class cannot occur here (one reduction, one tensor). But the reduction is a
  per-TOKEN mean over the global batch, so **A4 and A5 must match on TOKEN MASS, not row
  counts** — an injected row at 2x the sampled length carries 2x the gradient weight (measured
  0.5 / 1.0 / 2.0 at 4 / 8 / 16 tokens). Every arm logs `sft_tokens / total_tokens` (new key,
  nothing existing reports it); A4/A5 either match token mass to A0 or reweight by
  `mean_len / row_len`, and the choice is stated with the number.
- **Gold rows carry a FINITE logprob** equal to the trainer's recomputed `prox_logp`. NEVER NaN:
  `kl_ctl=0.0` does not neutralise it (`-0.0 * NaN = NaN`), and under the live
  `adv_norm: mean_level=batch` one NaN row spreads to all 8/8. A 0.0 placeholder is silently
  down-weighted instead. Any batch reaching the loss with an unfilled gold row must raise.
- Keep `adv_norm=None` and `kl_ctl=0` for routed arms; `mean_level=group` is forbidden (its
  token-weighted mean destroys `sum(a_i)=0` unless all rows match in length).
