# GOAL — self-evolving LLM agents

**Objective: a publishable top-AI-conference paper on self-evolving LLMs that is innovative
and beats SOTA.** Not a system report. That means (a) a claim prior art does not already
own, (b) a number that beats a named baseline on the benchmark that baseline reports, and
(c) evaluation that survives arXiv 2607.12227's controls. All three, or it is not the paper.

Single reference for what we are claiming, what is built, what is measured, and what is
open. Every status here is checkable; nothing is marked done on intent.

---

## 1. The claim

Route the **training signal** — SFT / RL / distillation / teacher / reverse-KL / KL /
reward-shaped — **per task, per cluster, per sample, and per token**, chosen by a learned
meta-controller, with a **validity condition** saying when each choice is legitimate.

### What prior art already owns

| work | what it owns | what it leaves |
|---|---|---|
| **SIA** (2605.27276) | selects among PPO / GRPO / entropic-advantage / REINFORCE+KL / best-of-N BC / DPO **per task**, via an LLM Feedback-Agent | finer granularity; *learned* rather than prompted; the authors defer "a fuller treatment of algorithm selection" |
| **Co-Harness** (2607.22688) | HarnessCritic reads **failed** trajectories → harness; model trained on **successful** ones; alternating | the partition is a hard success/failure rule. One trajectory can serve **both**, and which signal it feeds is itself a routing decision |
| **Learning Fast & Slow** (2605.12484) | two timescales: context = fast weights, parameters = slow. 3× sample efficiency, 70% less KL drift | a different axis — *what to update*, not *which signal per unit* |
| **AI2 eval** (2607.12227) | shows harness-evolution gains vanish against matched-budget test-time scaling | — this is the bar, not a competitor |

**Therefore claimable:** cluster/sample/token granularity, derived from a validity condition
(`I_RL`), learned rather than prompted, measured against a task-level controller.
**Not claimable:** task-level algorithm selection; success→model / failure→harness.

---

## 2. Requirements, with honest status

Legend: **DONE** built + tested + audited · **PARTIAL** exists but incomplete or unproven ·
**NOT BUILT** declared only.

**Scoreboard (2026-08-31, after the seam was tested).** Method **10 DONE / 5 PARTIAL /
7 NOT BUILT**; Engineering **6 DONE / 2 PARTIAL**; Benchmarks **4 DONE / 1 PARTIAL /
10 NOT BUILT**. So: **no, the goal is not implemented.** What IS complete is one vertical
slice -- measure the silent channel, decompose it, act on it, verify end-to-end -- and that
slice is the paper's spine.

**The "no caller in training" gap is now closed.** `actor.py::_route_groups` builds the
configured Router, feeds it observability features, and applies its decisions through
`group_apply`. As of 2026-08-31 that path has 11 tests driven through the REAL
`_compute_advantages` entry point (not through the helper, which could not catch the helper
being unreachable) and **7/7 mutants killed**. So `router=contextual` and
`router=code_policy` are reachable arms rather than registry entries.

What that does NOT establish, and the reason M8 and M15 stay PARTIAL: reachable is not the
same as effective. Nothing yet shows a learned router DECIDES BETTER than the fixed rule in
training. That needs a GPU arm, and it is now the top item on the critical path.

### Method

| # | requirement | status | evidence / gap |
|---|---|---|---|
| M1 | Per-**token** routing | **DONE** | wired into `grpo_loss_fn`; 3 audits; measured reach **1.7%** |
| M2 | Per-**cluster** routing | **PARTIAL** | `ClusterRouter` built + audited; `route_batch` still has no caller. MEDS layer-logit key not wired |
| M3 | Per-**sample** routing | **DONE** | `SolveRateRouter`, registered |
| M4 | Per-**task** routing | **NOT BUILT** | and it is SIA's, so low value |
| M5 | RL/reverse-KL mix per token | **DONE** | extends AReaL's own `rl_loss_weight`/`distill_loss_weight` |
| M6 | SFT / forward-KL modes | **PARTIAL** | SFT now REACHES the update for solved groups (a constant on their zero advantages IS the supervised step). Forward-KL still a name |
| M7 | **Teacher** supplying routed units | **NOT BUILT** | DEMOTED by measurement: reaches only ~4.5% of groups. The free self-target covers 31.4% and is built |
| M8 | Learned meta-controller | **PARTIAL** | `ContextualBanditRouter` (LinUCB over 7 features) built, audited, 29/29 mutants, demonstrably learns on a clean reward. **Now called in training** via `_route_groups`. Gap: no evidence it out-decides the fixed rule on a real task |
| M9 | Rule evolve-policy (cold start + baseline) | **NOT BUILT** | needed before "learned" is falsifiable |
| M10 | Evolve model / harness / reward | **PARTIAL** | `HarnessAction` axis built, audited, 18/18 mutants. **No consumer**: nothing writes `can_evolve_harness`, nothing reads the action. Reward axis still a name |
| M11 | Cadence: frozen / alternating / simultaneous | **NOT BUILT** | `CADENCES` = bare strings |
| M12 | Evolvable reward formula | **NOT BUILT** | |
| M13 | MEDS clustering | **PARTIAL** | `_cluster_with_hdbscan` + `_classify_with_knn` vendored VERBATIM; verified to recover 2 modes. Deps not installed on training boxes; not yet a controller feature |
| M14 | BigBang two-level critic | **NOT BUILT** | `"two_level": None` |
| M15 | Trajectory observability → policy inputs | **PARTIAL** | `observability.py`, 7 features at zero extra compute, 27/29 mutants (2 equivalent). **Now called in training** -- `_route_groups` computes them from RAW reward each batch. Gap: no ablation showing which features carry the decision |
| M24 | **Multi-teacher on-policy distillation** as the supplier for the UNSOLVED branch | **NOT BUILT** | Now the highest-value model-evolution item: the composition flip made unsolved the majority (60.9% of the channel on MATH) and it has no self-target |
| M23 | **LLM-as-router**: an LLM reads a unit's rollouts (failure modes, solve rate, observability metrics) and decides evolve-model vs evolve-harness | **NOT BUILT** | See the positioning note below -- the idea is a combination of existing lines, so the contribution has to be the comparison, not the idea |
| M21 | Code-as-policy (`learned_code`) | **DONE** | AST allowlist + subprocess cost-vetting; ~150 adversarial policies, 26/26 mutants. 2 escape families documented as unclosable by any AST rule |
| M22 | Decision→advantage seam | **DONE** | `group_apply.py` + `actor._route_groups`; 11 tests through the real `_compute_advantages`, **7/7 mutants**. Router state cached across batches, unit ids batch-prefixed, unregistered name refused |
| M18 | Per-**group** routing (the silent channel) | **DONE** | in `_compute_advantages`; 17 tests on the REAL path, 7/7 mutants; verified firing live (`routed == solved`, diff 0.00e+00) |
| M19 | Free self-target (RFT on solved groups) | **DONE** | `solved_advantage`; needs no teacher; reach 31.4% and RISING to 60% |
| M20 | Unlikelihood on unsolved groups | **DONE** | `unsolved_advantage`, sign-guarded; reach only ~3.7% at this operating point |
| M16 | ScalarCritic | **DONE** | 30/30 mutants |
| M17 | Outcome-calibrated meta-critic | **DONE** | 19/20, survivor equivalent |

### Engineering

| # | requirement | status |
|---|---|---|
| E1 | Clean rollback to vanilla AReaL by argument | **DONE** — 96/96 bit-identical, loss *and* gradients |
| E2 | Every feature configurable for ablation | **PARTIAL** — routers registered; shaper/gate/cadence/evolve axes still empty |
| E3 | Subagent audit per feature | **DONE** — 4 audits, all findings fixed or recorded |
| E4 | Mutation-test every guard | **DONE** — routing 30/30, meta-critic 19/20, split 9/9, packed 3/3, cluster 9 regressions added |
| E5 | Clean, organised, extendable | **PARTIAL** — good in `selfevo/`; `experiments/` has accumulated one-offs |
| E6 | Both boxes busy | **DONE** — supervisor watches LOG GROWTH, not utilization; a dead run held 4 GPUs at 100% for 46 min undetected before this |
| E7 | Save progress promptly | **DONE** — HF verified 288 artifacts + 67.75 GB checkpoints |
| E8 | Per-server branches | **DONE** — `selfevo/a100`, `selfevo/h200` |

### Benchmarks — named, with the paper each belongs to

We must be able to say "we beat X on the benchmark X reports". Each row names the source so
a comparison is like-for-like rather than a benchmark of our choosing.

| # | benchmark | whose | status | note |
|---|---|---|---|---|
| B1 | MATH-500 | ours/common | **DONE** | saturated at 27B (0.966); usable only at 1.5B/7B |
| B2 | AIME 24/25 | common | **DONE** | unusable: 0.000 @1.5B, 0.867 @27B with a 24-pt CI on 30 problems |
| B3 | AMC23, HMMT 24/25 | common | **DONE** | 30-40 problems; indicative only |
| B4 | **OlympiadBench** | common | **DONE** | **the frontier target**: 675 problems, 7-pt CI, 27B = 0.825 @8% trunc |
| B5 | **Terminal-Bench 2.1** (Terminus-2 harness) | **Ornith-1.5** | **NOT BUILT** | Ornith-397B 86.1, 35B-A3B 67.8. Also AI2 2607.12227's benchmark, so it is where the evaluation bar was set |
| B6 | **DeepSWE** (Claude Code harness) | **Ornith-1.5** | **NOT BUILT** | Ornith-397B 56.0 |
| B7 | **Frontier-Bench** | **Ornith-1.5** | **NOT BUILT** | competitors near zero |
| B8 | **SWE-bench Pro** (+ `-os`) | **BigBang-v1** | **NOT BUILT** | `ScaleAI/SWE-bench_Pro` |
| B9 | **HLE** (Humanity's Last Exam) | **BigBang-v1** | **NOT BUILT** | `cais/hle` |
| B10 | **FrontierScience** | **BigBang-v1** | **NOT BUILT** | `openai/frontierscience` |
| B11 | math + competitive programming + repo-level SWE | **EvoTrainer** (2606.03108) | **PARTIAL** | math done; the other two are B8/B12 |
| B12 | **LiveCodeBench v6** | common | **NOT BUILT** | data obtained; generation + sandboxed grading unwired |
| B13 | **Spider2 / BIRD** | enterprise SQL | **NOT BUILT** | both on disk, never run |
| B14 | BioMysteryBench | domain | **NOT BUILT** | 90 gated, 155 GB |
| B15 | GeneBench-Pro | domain | **NOT BUILT** | 10/129 public |

**Ornith-1.5 evaluation protocol, to copy rather than invent:** five independent runs
averaged; Terminal-Bench 2.1 via Harbor/Terminus-2 with `parser=json`, `temperature=1.0`,
`top_p=1.0`, 128K context, 4h timeout / 32 CPU / 48 GB; DeepSWE via the Claude Code harness
at `temperature=1.0`, `top_p=0.95`, 256K context. Five-run averaging matters — our own
measured noise floor is 0.008-0.027 depending on the benchmark.

### Note on M23 (LLM-as-router): what would and would not be novel

Asked directly: is an LLM that analyses rollouts and decides "evolve the harness or the
model" innovative, or already done? Honestly, **the pieces are all published and the
combination is not obviously claimed**, so the novelty cannot rest on the idea.

Already owned by prior work:

* **LLM reads its own failures and proposes a fix** -- Reflexion / Self-Refine, and the
  large self-improvement literature. Reading trajectories to decide *something* is standard.
* **LLM writes the policy as code** -- FunSearch, OPRO, ADAS. This is our `learned_code` arm.
* **Which target to evolve, decided from outcomes** -- Co-Harness (success to model, failure
  to harness) and EvoTrainer. They decide it with a FIXED RULE, not a language model.
* **Clustering failure modes to act on them** -- MEDS, with HDBSCAN over layer logits rather
  than an LLM.

What is not obviously done is the specific composition: routing the **evolve-target axis**
per unit, with an LLM reading the actual trajectories rather than a threshold reading one
scalar. So M23 is worth building as a **fourth controller in the existing ablation** --
rule / learned-weights / learned-code / **LLM** -- where what is defensible is the measured
comparison at matched cost, not the mechanism.

**Two reasons from our own measurements to be sceptical before spending on it:**

1. **The credit-assignment problem does not go away.** We measured that a contextual
   controller learns during cold start and then stops, because a converged policy produces
   single-mode batches and a batch-level scalar over those carries no comparative
   information. An LLM router faces exactly the same channel. It may make better *initial*
   decisions from richer input; it does not get better *feedback*.
2. **The cheap controllers must be beaten first.** The solved branch was measured inert, and
   the noise floor on our benchmark is 0.02 systematic. An arm costing an LLM call per group
   per batch has to clear that floor against a free threshold. If a threshold on
   `truncated_fraction` captures most of the signal, the LLM is an expensive way to
   rediscover it -- and the honest paper reports that.

**Therefore the build order is: LLM router LAST**, after the free controllers have set the
bar, and it is measured against the matched-proportion control like every other arm. Its
best outcome is not "the LLM wins" but "here is what reading the trajectory buys over reading
a scalar, priced".

### The granularity of harness evolution is NOT the granularity of model evolution

A question worth settling explicitly, because the current design quietly conflates two things.

**Model evolution is fine-grained all the way down.** The gradient acts per TOKEN, aggregates
per SAMPLE, and is centred per GROUP. Token-level routing is meaningful there -- we built it,
measured its 1.7% reach, and reported it null. Nothing about a per-token model decision is
incoherent.

**Harness evolution is not, because the harness is a SHARED ARTEFACT.** A scaffold, prompt
template or tool configuration applies to every future rollout. So the two granularities come
apart, and they have to be named separately:

| | evidence granularity | action granularity |
|---|---|---|
| **token** | meaningless -- a token does not identify a failure mode a scaffold could fix | **meaningless** |
| **sample** | natural -- one trajectory exhibits one failure | degenerate -- a per-sample harness is not a harness |
| **cluster** | natural | **natural, and the interesting one** |
| **task** | natural | natural; this is Co-Harness / SIA territory |

**The consequence: CLUSTER is the natural unit for the harness axis**, and that raises the
stakes on the clustering work. A single trajectory failing says little; twenty trajectories
failing the *same way* justify one scaffold change and give it something to be validated
against. This is where MEDS' layer-logit clustering stops being an ablation about the MODEL
axis and becomes load-bearing for the HARNESS axis -- the clusters ARE the units a harness
edit is scoped to.

**A gap in what is built, stated plainly.** `HarnessAction` is currently emitted PER GROUP,
which makes it evidence, not action -- and there is no aggregation step between "these units
proposed" and "here is one harness change". The design as it stands would let a caller
mistake a stream of per-group PROPOSE flags for harness evolution, when what it actually has
is an unaggregated evidence queue. Naming the two granularities separately is the fix:
propose per unit, aggregate per cluster, act once.

**Scope is therefore a config axis** (`harness_scope` in {global, per_task, per_cluster}),
not a constant, and `per_cluster` requires the cluster key to be stable across steps -- which
is exactly why `MEDSClusterer` fits once and assigns thereafter rather than re-fitting per
batch.


#### Correction: a harness DISPATCHER makes fine granularity coherent after all

The table above says per-sample harness action is "degenerate", and that is wrong as stated.
It is only degenerate if each sample gets its own UNIQUE harness. It is entirely coherent for
each sample to be routed to one of a small **harness VARIANT SET** by a predicate on its
features -- a tool-heavy scaffold for units whose failures look like missing information, a
longer-budget scaffold for units that truncate, the plain scaffold otherwise.

That reframes what "evolving the harness" means, into two things that evolve at different
rates:

* **the variant set** -- what scaffolds exist. Coarse, changes rarely, and is what
  Co-Harness and EvoTrainer actually evolve.
* **the dispatch rule** -- which unit gets which variant. Fine, changes every batch, and is
  *the same object as our routing policy*.

**This unifies the two axes into one action space**, which is the cleanest version of the
claim in this document: the controller selects a pair

    pi(u) = (estimator, harness_variant)

per unit, from features. Model routing and harness routing stop being two mechanisms and
become two coordinates of one decision. `truncated_fraction` -- already one of the seven
features, and already the one that separates "failed" from "out of budget" -- is exactly the
signal that should pick a longer-budget variant. That is a concrete, testable prediction the
current design already has the inputs for.

Granularity, restated correctly:

| | dispatch (which variant) | variant-set evolution |
|---|---|---|
| token | still meaningless -- a scaffold applies to a whole rollout | meaningless |
| sample | **coherent** -- pick a variant per prompt | too fine to learn from |
| cluster | **coherent, and where the evidence aggregates** | **natural** |
| task | coherent | natural (Co-Harness / SIA) |

**What this costs to build, and why it is honest to say so:** a variant set needs at least two
genuinely different scaffolds, and a dispatch rule needs the matched-proportion control like
every other arm -- otherwise "we dispatched" cannot be told apart from "one variant is just
better". Neither exists yet. The value of writing it down now is that the action space, the
features, and the control are all already built; what is missing is the variants.


### Feedback for a learned router is only defined when the batch has mode diversity

Found by the seam tests on 2026-08-31, and it constrains every learned-controller design here
including M23 (LLM-as-router).

`batch_outcomes` credits one scalar -- the change in mean raw reward between consecutive
batches -- across the decisions in a batch. When every group took the SAME mode that scalar
cannot be divided among them, so the update is refused as `ConfoundedUpdate` rather than
applied to a vacuous attribution. Correct, and it has a consequence worth designing around:

* A router that CONVERGES to one mode stops receiving feedback entirely. It is not punished
  for converging; it simply goes blind, and any later drift in what the right mode would be
  is invisible to it. Exploration is not merely helpful here, it is the precondition for the
  learning signal existing at all.
* So a converged-but-wrong router and a converged-and-right router are indistinguishable from
  the feedback stream. The `feedback/confounded_skips` counter is the diagnostic: a run whose
  skips rise to 100% has a controller that has stopped learning, however good its metrics
  look.
* This is a real property to report, not a bug to paper over. It also predicts a specific
  failure for M23: an LLM-as-router asked to be decisive will collapse the mode distribution
  faster than a bandit with explicit exploration, and will therefore starve its own feedback.

### Note on M24 (multi-teacher OPD): why it moved up, and three traps already known

**Why it moved up.** The solved branch is abandoned (inert at 0.5, harmful at 2.0), and the
composition flip put the UNSOLVED branch at 60.9% of the silent channel on MATH -- 25.5% of
all groups, against 4.5% on GSM8K. That branch has no self-target by construction: every
sample was wrong, so nothing in the rollout can serve as a target. A teacher is the only thing
that turns it from SKIP into signal. Multi-teacher on-policy distillation is the natural
supplier, and it is on-policy, which matters because the units are drawn from the student's
own rollouts rather than from a fixed corpus.

**Where it sits in the action space.** It does not add an axis. It supplies the target that
makes `SFT_teacher` reachable in \eqref{eq:rule} -- the branch that is currently
`(SKIP, PROPOSE)` because `has_teacher` is False everywhere. So the routing decision is
unchanged; what changes is that one of its arms stops being a no-op.

**Three traps, from measurements already in hand -- do not rediscover them:**

1. **A null teacher is NOT a control.** Comparing "real teacher" against "no teacher" does not
   isolate transferred knowledge; the difference includes the effect of simply having ANY
   target, and the sign can flip. The control is a teacher matched in every respect except
   the knowledge -- e.g. a shuffled or same-capability teacher -- not the absence of one.
2. **Do not score multi-teacher gain by an energy ratio.** That statistic grows with the
   number of teachers by construction, so "more teachers is better" falls out of the
   arithmetic rather than the data. Use a normalised difference against the matched control.
3. **Report the standard error on the DIFFERENCE, not per arm**, and watch for
   validation/test inversion. An earlier multi-teacher analysis was underpowered in exactly
   this way, and the scaling-with-N claim is the part that was fresh -- so it is also the part
   most exposed if the statistics are loose.

**Prior art bounds the claim.** Multi-teacher distillation itself is not ours (ECLARE, VDD and
the wider ensemble-distillation line). What is not obviously claimed is using it as the
*supplier for a routed branch selected by measured RL-silence*, with the teacher spent only
on the units that provably cannot learn from RL. That framing also makes the cost argument
sharp: a teacher is expensive, and routing means paying for it on ~25% of groups rather than
all of them.


### Models

| model | status |
|---|---|
| Qwen2.5-1.5B / 7B-Instruct | **DONE** — both trained and scored |
| Qwen3.8-27B | **DONE** — scored on all math benchmarks |
| **Ornith-1.5-9B**, **Ornith-1.5-35B-A3B** (MoE, 35B total / ~3B active) | **DONE** — downloaded and scored on OlympiadBench |
| Ornith-1.5-397B | **NOT PULLED** — too large for this training loop |
| **BigBang-v1** (`endless-frontier/BigBang-v1`) | **NOT PULLED** |

---

## 2b. Methods to compare against, and code to reuse rather than reimplement

Standing preference: **use the authors' own repo** where one exists.

| method | role for us | code | status |
|---|---|---|---|
| **SIA** (2605.27276) | the **baseline to beat** — task-level algorithm selection is theirs, so our finer granularity must beat it at matched budget | `hexo-ai/sia` | **NOT CLONED** |
| **MEDS** (2604.11297) | a concrete **`key_fn` for `ClusterRouter`**: reuses **layer-wise logits** as lightweight behaviour representations and clusters **error patterns** with **HDBSCAN**. `cluster_method=hdbscan`, `use_layer_diff`; actor-side logic in `verl/workers/actor/dp_actor.py` | on disk at `~/baselines/MEDS` | **PRESENT, unused** |
| **Co-Harness** (2607.22688) | prior art for success→model / failure→harness; our extension is that one trajectory can feed **both**, decided by routing | — | **NOT CLONED** |
| **EvoTrainer** (2606.03108) | co-evolves policy **and training harness**; closest to our evolve-target axis | — | **NOT CLONED** |
| **Learning Fast & Slow** (2605.12484) | bounds the cadence claim; 3x sample efficiency, 70% less KL drift | — | reference only |
| **autoresearch** | an **additional baseline method** to compare against | on disk at `~/baselines/autoresearch` | **PRESENT, unused** |
| **deepseek-harness** (`deepseek-ai/deepseek-harness`) | the **target** for the `evolve_target=harness` arm — a real plugin-architected agent runtime, ~453K lines TS. Far more credible than a toy scaffold | — | **NOT CLONED** — bring in only when the harness arm actually runs |
| **hermes-agent** | second harness candidate | — | **NOT CLONED**; no canonical repo confirmed |
| **R-Zero, RAGEN, Search-R1, Absolute-Zero-Reasoner** | related self-improvement baselines already on disk | `~/baselines/` | **PRESENT, unused** |

**MEDS is the highest-value of these**, because it plugs straight into the `key_fn` seam
`ClusterRouter` already exposes: derived (`SilenceSide`) clustering versus learned
(layer-logit HDBSCAN) clustering becomes a direct ablation, with
`MatchedPermutationControl` separating "clustering helped" from "the mode proportions
changed".

## 3. Measured constraints — do not rediscover

- **GSM8K train reward MIS-ORDERS checkpoints** vs held-out MATH-500. Never select on it.
- **MATH-500 saturated at 27B** (0.966). **AIME unusable at both ends** (0.000 @1.5B; 0.867 @27B with a 24-pt CI on 30 problems).
- **OlympiadBench is the frontier target**: 675 problems, 7-pt CI. 27B = **0.825** at 8% truncation.
- **`evaluator.freq_epochs` AND `freq_steps` AND `freq_secs` must all be null** or the trainer deadlocks in `_evaluate_fn`.
- **The routing rule needs `adv_norm=None` and `kl_ctl=0`.** `mean_level=group` (which I once prescribed) *breaks* it: residual 2.139 vs 0.0.
- **`mb_spec.granularity` must equal `n_samples`**, or microbatches split GRPO groups and the precondition fails.
- **Token-level reach is 1.7%**, growing 3.5× as entropy falls. **Group-level silence is 57.4%.**
- On the motivating data the 57% is bounded **solved ∈ [0.394, 0.574], unsolved ∈ [0, 0.180]** — so the dominant cluster action is SKIP, i.e. gradient deletion.

---

## 4. Evaluation bar (AI2 2607.12227)

| control | status |
|---|---|
| Held-out benchmark for capability | **MET** |
| Fixed-mode baseline | **MET** (step0l, running) |
| Matched permutation control | **MET** (in code) |
| Paired significance testing | **MET** (McNemar) |
| Replicate / noise floor | **MET** — 0.008 @n=250, 0.022 @n=500, 0.027 on OlympiadBench |
| maj@k at matched budget | **PARTIAL** — tooling built + mutation-tested; run deprioritised |
| Matched inference budget | **NOT MET** |
| Matched feedback budget | **NOT MET** — query counts not logged |
| Held-out split when feedback drives a decision | **MET** (committed 250/250 split, content-pinned) |

---

## 5. Results so far

1. **Train reward is anti-informative.** It *mis-orders* checkpoints against held-out MATH-500.
2. **Demo recipe destroys capability; scaffolding preserves it.** 0.528→0.334 vs flat. n=500, 5/5 significant.
3. **Replicates on OlympiadBench** — 0.188→0.107, independent problem set.
4. **Benchmark headroom mapped** across three scales.
5. **NEGATIVE: token routing has no measurable effect** (McNemar 0.53–0.92) — consistent with its 1.7% reach.

6. **G=16 does not halve the silent channel.** Matched GSM8K/1.5B/routing-off pair:
   silent fraction 0.5906 at G=8 vs 0.4553 at G=16, where the homogeneous-binomial account
   calibrated on the G=8 run predicts 0.3488. Doubling G buys a 22.9% relative reduction
   against a predicted 41%, so a large share of the channel is prompts that are silent at ANY
   group size. Between-run comparison, one run per G -- directional, not a coefficient.

**Nothing yet demonstrates the method works.** Results 1–4 are evaluation contributions;
result 5 is a null; result 6 is a property of the problem, not of the method.

> **PROVISIONAL — composition numbers are under review (2026-08-31).** The silent-channel
> decomposition violates the identity it must satisfy by construction:
> `silent == solved + unsolved` fails at 104 of 116 steps in `g16`, with a second-half mean
> residual of +0.277 and an excursion to **-0.109**, which is impossible for a partition into
> subsets. Every "X% of the silent channel is solved" figure below and in Sec. 6 is a ratio of
> the two implicated metrics, so those figures -- including the 87.5% that re-ordered the
> critical path -- are provisional until the violation is explained. `silent_group_fraction`
> is computed directly and is not implicated. See EXPERIMENTS.md for the measurement and for
> the CPU identity test that should have preceded the claims.

---

## 6. Critical path

**Re-ordered on 2026-08-30 by measurement, not by preference.** The silent channel was
measured to be **87.5% solved** (1152 group observations on the routing-off control), so the
external teacher reaches ~4.5% of groups while the free self-target reaches ~31.4% -- a 7x
difference, and the reverse of the previous ordering.

1. **A/B the solved branch.** `SFT` on a unit's own correct sample versus `SKIP`, at matched
   compute. This is the method's main lever and its main risk: sharpening an already-correct
   policy spends entropy, and entropy collapse is the failure mode already measured twice
   here. Until this runs, the solved branch defaults to SKIP.
1b. **Run `router=contextual` as an arm** against the fixed rule at matched compute. The seam
   is now tested end-to-end (M22), so this is a launch, not a build. It is what turns the
   learned controller from "reachable" into a result -- or into an honest null.
2. **Give `route_batch` a caller** -- otherwise M2 and the new harness axis are dead code.
   NOTE this is now the ONLY remaining "no caller" gap; the group-level seam is closed.
3. **Re-measure the split on OlympiadBench.** 0.875 is a property of GSM8K at a high solve
   rate, not a constant; a harder task moves mass to the unsolved branch.
4. **M9 rule evolve-policy** -- cold start *and* the baseline the learned one must beat.
5. **A teacher (M7)** -- now a lever on ~4.5% of groups, so it follows rather than leads.
6. Then M8 learned controller, then the named benchmarks in Sec. 2.

## 7. Standing operating rules learned the hard way

* **Liveness is log growth, never `nvidia-smi`.** A dead run held 4 GPUs at 100% for 46
  minutes after a 1-retry weight-sync disconnect. Every run goes under
  `experiments/harness/supervise.sh`, which watches the log and py-spy-dumps before killing.
* **Checkpoints mirror to HF as they are produced**, newest first, via
  `experiments/harness/hf_mirror.py` -- both boxes are rented and may be reclaimed at short
  notice. Skipping is per-file by size; folder-level checks accept partial uploads.
* **Every capability is flag-gated and rolls back to vanilla AReaL bit-identically.**
