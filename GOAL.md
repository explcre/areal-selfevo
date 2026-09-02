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

**Scoreboard (2026-08-31, after the seam was tested and M9 was built and audited).** Method **10 DONE / 6 PARTIAL /
6 NOT BUILT**; Engineering **6 DONE / 2 PARTIAL**; Benchmarks **4 DONE / 1 PARTIAL /
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
| M8 | Learned meta-controller | **PARTIAL — cause proven as an identity, fix validated on CPU** | Built, audited, called in training, explores, receives clean feedback. The recorded cause now holds as an **identity**, not an argument: with one scalar credited to every decision in a batch, neither ridge update mentions the mode except as the dictionary key the result lands in, so an arm's parameters are a function of which contexts it received and nothing else — presenting the same contexts rotated by one hands each arm what a different arm had, and the fitted parameters rotate with them to floating-point equality. Make the credited value depend on the mode and the identity breaks. **The L1-from-uniform signature this row used to cite is RETRACTED as a validator**: in a world where a right answer exists, the batch-credited router's L1 *rises* 0.067→0.173 by quarter over eight seeds **while its targeting stays at chance** — the router picks its own assignment, so arms drift apart on that feedback alone, and away-from-uniform is also what an arm that learned nothing looks like. A fix validated on L1 alone can be validated by noise. Results are now carried by a **subset contrast** (zero for a router with a favourite mode however lopsided the mix) with direction checked by a **per-subset targeting fraction**, so an inverted preference cannot pass. **Fix measured** (`credit_sim`, half the prompts helped only by SFT and half only by RL, the half written into one of the seven observability features and the other six noise, everything downstream of a decision the shipped code): over eight paired seeds the batch arm ends at **0.098** subset contrast and the per-prompt arm at **0.779**, a paired difference of **0.682 ± 0.021**, per-subset targeting above 0.72 on both halves for every seed. **Correspondence control**: shuffling the credits across the prompts that earned them — same ledger, same pairings, same multiset — collapses the arm to **0.102**, so the effect is the prompt-to-credit correspondence and not a noisier signal; a batch-credited run has nothing to shuffle and the simulator REFUSES to report that as a control rather than emitting a fake. **Negative result**: the already-shipped `credit=prompt_centered` should NOT be the arm that gets a GPU — it subtracts the batch mean delta, one number shared by every arm and so the same class of quantity per-prompt credit exists to escape, and it measures at or below plain per-prompt credit in both regimes tested. `PromptCreditLedger` gains baseline `self_mean` (mean of that prompt's own strictly-earlier deltas, so a credit is never its own control); the first delta is withheld and counted rather than credited against a zero baseline, since a zero baseline hands the first credited mode the whole common trend — the bias that made the live arm abandon RL at the exact step credit began flowing. History is two floats on the prompt's existing record, so nothing grows with run length. Default unchanged; every earlier arm reproduces exactly. Details in `selfevo/FINDINGS_credit_assignment.md`. **Gaps: NOT reachable from config** (the two lines it needs in the actor wiring are both mutation anchors in a tree a live GPU job imports), and **no arm has trained with it** |
| M9 | Rule evolve-policy (cold start + baseline) | **PARTIAL — built, and it does NOT de-confound the comparison** | `selfevo/routing/rule_policy.py`, registered as `router=rule`. **Audited 2026-08-31 and the headline claim retracted**: under this repo's binary graders it is behaviourally IDENTICAL to `SolveRateRouter` (102/102 contexts, both teacher settings), so it does not separate "written vs learned" from "1 feature vs 7". That is not fixable by writing a better rule -- only `reward_std` has a measurement behind it, and inventing thresholds for the other six would violate the standard this repo holds. **The finding IS that**: a defensible hand-written policy here collapses to one predicate, and the confound has to be reported, not engineered away. What the router does add is narrow: branches cited to their measurements (`reward_std` above a MEASURED float32 tolerance -> RL; solved+silent -> SKIP, inert at 0.5 and harmful at 2.0; unsolved+silent -> teacher else SKIP), the learned router's input contract, and `solved_mode` as the seam for critical-path item 1. Its `HarnessAction.PROPOSE` **cannot fire** — nothing writes `can_evolve_harness` and `_route_groups` discards `.harness` (same gap as M10). 57 tests, 3 through the real `_compute_advantages`; 31/31 mutants. Gaps: no arm has trained with it; no subagent audit of the code beyond the adversarial claims audit |
| M10 | Evolve model / harness / reward | **PARTIAL** | `HarnessAction` axis built, audited, 18/18 mutants. **Consumer built 2026-09-01**: `selfevo/harness/dispatch.py` dispatches over a `HarnessVariant` set; `_route_groups` writes `can_evolve_harness` (True iff `group_routing.harness_variants` has 2+ members), passes `harness_consumer=True` only when a dispatcher exists, and feeds `.harness` to it; `route/harness_*` metrics name the active variant. `_refuse_dropped_harness` NOT weakened and still fires with no variant set. 82 tests, 50/50 mutants. **Feature-driven rule + its matched control built 2026-09-01**: `selfevo/harness/selectors.py::TruncationStepLimitSelector` moves the step budget one rung toward a LONGER `step_limit` when the batch-mean `truncated_fraction` is at or above 0.5 and one rung SHORTER at or below 0.05, refusing in between and at the ends through a typed, recorded, counted `HarnessSelectionRefused` rather than silently staying put; `RateMatchedControlSelector` replays the treatment's MEASURED move/stay multiset on a seeded schedule that reads no feature, so the two arms are rate-matched by construction at every deck boundary and the difference between them is the only quantity attributable to targeting. 65 further tests, 56/56 mutants. Gaps: dispatch is over CONFIGURATIONS only -- no adapter runs on this box (Docker absent); no config field selects a selector and `observe()` has no production call site, so both new arms are reachable only through `HarnessDispatcher(..., selector=...)`; the two thresholds are prices for an asymmetric error rather than measurements; the control matches the switch RATE but not the destination mixture; and no arm has trained with any of it. Reward axis still a name |
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
| B4 | **OlympiadBench** | common | **DONE** | **the frontier math target**: 675 problems, 7-pt CI, 27B = 0.825 @8% trunc. Now IN `math_bench.SUITE` (2026-08-31); gold self-verifies **675/675** through our grader, so a low score is the model's. 94/675 carry multiple golds in one string -- exact match marks those wrong, so scores are a LOWER BOUND |
| B5 | **Terminal-Bench 2.1** | **Ornith-1.5** | **HARNESS AVAILABLE** | Ornith-397B 86.1, 35B-A3B 67.8. LongHorizon-Harness (MIT) ships `eval/TB-harness` with a conda env file and scripts; only the TASKS must be fetched separately into `datasets/terminal-bench-2-1/tasks`. LHH reports Qwen3.7-Plus 69.7 -> 77.2 here, so a published harness-swap comparison already exists on this benchmark |
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
| B16 | **WeaveBench** | LongHorizon-Harness | **HARNESS AVAILABLE** | `eval/WeaveBench-harness`. The benchmark with the largest reported harness effect: 51.8 -> 80.7 at fixed model. Long-horizon GUI+CLI |
| B17 | **OSWorld-V2** | LongHorizon-Harness | **HARNESS AVAILABLE** | `eval/OSWorldv2-harness`, with VM providers for vmware/gcp/azure/aliyun/virtualbox/volcengine. Qwen3.7-Plus 2.8 -> 8.3; Claude Opus 4.7 20.0 -> 34.3 on a subset. Low absolute scores make it a poor separator today |
| B18 | **MLE-Bench Lite** | Frontis-MA1 / OpenRSI | **HARNESS AVAILABLE** | `OpenMLE-Evo` adapters. 12h/task on ONE 12GB card. Base 39.39, Evo 60.61, Evo-Max 71.21 (their own note: Evo-Max changes the search system and is not a pure model gain) |
| B19 | **NatureBench Lite** | Frontis-MA1 / OpenRSI | **HARNESS AVAILABLE** | held-out transfer set; Match-SOTA 50 -> 70 |

**Do we need a frontier CODE benchmark? Yes -- but scale gates which one.** Answering this
with measurement rather than preference:

* **Math is not the gap.** B4 OlympiadBench is the frontier target, is wired into the scoring
  suite as of 2026-08-31, and its golds self-verify 675/675. MATH-500 saturates at 27B and
  AIME is unusable at 1.5B (0.000), so OlympiadBench is where a math claim gets made. The
  blocker there is a method checkpoint worth scoring, not a missing benchmark.
* **Code is the gap, and B5/B6/B8 are the wrong first move at our scale.** Terminal-Bench 2.1,
  DeepSWE and SWE-bench Pro are reported by 397B-class systems (Ornith-397B: 86.1, 56.0).
  A 1.5B or 7B policy scores at or near zero on repo-level SWE, and a table of zeros
  distinguishes no arm from any other -- the same trap as AIME at 1.5B, which is already
  recorded as B2's status. Running them now would buy a benchmark row and no signal.
* **B12 LiveCodeBench v6 is the realistic first code benchmark**, because it is
  problem-level rather than repo-level, so a small policy produces a non-degenerate score
  distribution that arms can separate on.
* **The frontier SDE benchmarks are the natural home of the HARNESS axis, not the model
  axis.** On Terminal-Bench and DeepSWE the harness carries much of the work, which is exactly
  what M10's harness-evolution action space is about, and both benchmarks publish a fixed
  harness (Terminus-2, Claude Code) to evolve against. That makes them the right target for
  the harness claim and the wrong target for a 1.5B model claim -- they need a competent base
  policy first. Sequence accordingly: LiveCodeBench for the model claim, SDE for the harness
  claim once a large-enough policy is available.

**Ornith-1.5 evaluation protocol, to copy rather than invent:** five independent runs
averaged; Terminal-Bench 2.1 via Harbor/Terminus-2 with `parser=json`, `temperature=1.0`,
`top_p=1.0`, 128K context, 4h timeout / 32 CPU / 48 GB; DeepSWE via the Claude Code harness
at `temperature=1.0`, `top_p=0.95`, 256K context. Five-run averaging matters — our own
measured noise floor is 0.008-0.027 depending on the benchmark.

### The full granularity x axis x policy grid, and an HONEST feasibility ordering

Asked to run every level (task / cluster / sample / turn / token), every evolve target (model /
harness / reward), every decider (rule / learned weights / code-as-policy / LLM agent), with
MEDS clustering choosing per-cluster treatment, over every named benchmark. Writing the grid
down in full, then ordering it by what the measurements say is worth doing -- because the grid
is much larger than the evidence, and the project has already spent GPU-weeks on cells that
turned out not to separate.

**The grid.** Granularity {task, cluster, sample, turn, token} x evolve-target {model,
harness, reward} x decider {rule, learned-weights, code-as-policy, LLM} x mode-cell
(target x objective x on/off-policy, Sec. above). That is several hundred arms. Nothing about
the project's measured position supports enumerating it.

**What the measurements already say about this grid, before spending anything:**

* **Token level is nearly empty.** Measured reach 1.7%, and the paired test was null
  (McNemar 0.53-0.92). Recorded as result 5.
* **Sample/group level does not separate.** Three credit signals, three very different
  training trajectories, ONE identical capability outcome (MATH-500 spread 0.010 against a
  0.020 noise floor; OlympiadBench spread 0.0015). Adding granularity to a decision that does
  not reach the benchmark multiplies arms, not information.
* **The binding constraint is CREDIT, not granularity.** A per-batch scalar provably collapses
  every LinUCB arm to one parameter vector; a per-prompt delta changed behaviour and moved no
  benchmark. Until a decision's credit is attached to something the decision CAUSED, finer
  granularity produces finer indistinguishability.
* **Turn level is unexplored and is the one gap the evidence does not close.** Every null so
  far is single-turn math. A multi-turn agentic setting is where turn-level routing could
  matter and where the harness axis becomes real, which is exactly what OpenMLE-Gym provides.

**Therefore the ordering, cheapest decisive first:**

1. **Serve a 30B and score it on one frontier benchmark.** Removes the scale confound behind
   every null so far. In progress: Frontis-MA1-30B (`Qwen3MoeForCausalLM`, 48 layers)
   downloaded, 57 GB.
2. **Get ONE execution-grounded credit signal working** in the OpenMLE-Gym loop -- did the
   artefact this decision produced actually run better. This is the thing every previous arm
   lacked, and no amount of granularity substitutes for it.
3. **Model-versus-harness routing at TURN/trajectory level**, the axis their operators do not
   cover and the one gap not closed by our nulls.
4. **MEDS clustering as the granularity step**, and only here: cluster trajectories by failure
   mode, then treat clusters differently. MEDS is vendored and now runs (`sklearn` backend,
   isolated venv) but has no caller. It is worth wiring ONLY once (2) exists, because
   per-cluster treatment without per-cluster credit reproduces the sample-level null one level
   up.
5. **Deciders, in increasing cost**: rule (built, and measured equivalent to `solve_rate` --
   report the confound rather than hide it), learned weights (built, null), code-as-policy
   (built, no caller), LLM agent (not built; OpenRSI now owns "an LLM decides", so only the
   model-vs-harness axis is novel).
6. **Reward evolution (M12)** last: it changes the objective the other arms are measured
   against, so it invalidates comparisons made before it.

**Benchmarks, ordered by what a 30B can actually move, not by prestige.** MLE-Bench Lite
(12h/task on ONE 12 GB card -- affordable, and the pivot's home benchmark) and OlympiadBench
(wired, full 675, frontier math) first. LiveCodeBench v6 next (problem-level, so a mid-size
policy yields a separable distribution). SWE-bench Pro, Terminal-Bench 2.1, DeepSWE, HLE,
FrontierScience, Spider2/BIRD, BioMysteryBench, GeneBench-Pro are all recorded in Sec. 2 and
all remain NOT BUILT -- they are 397B-class or domain-gated targets, and running them at 30B
would produce the same uninformative floor AIME produces at 1.5B. **The rule this project
already learned applies: a benchmark that cannot separate two arms is not evidence, it is a
table row.**

### Build on AReaL or on OpenRSI? Measured from the code, 2026-08-31

Cloned and inspected rather than judged from the README. OpenRSI is 110 MB, ~106k lines of
Python in 485 files:

| component | lines | what it gives us |
|---|---|---|
| `OpenMLE-ERL` | 59,294 | SFT + RL. **Built on `slime`** (Megatron + SGLang), vendored in-tree |
| `OpenMLE-Evo` | 33,608 | search runtime and benchmark adapters (MLE-Bench, NatureBench) |
| `OpenMLE-Gym` | 13,493 | task construction, execution, evaluation |

**Two facts that change the calculus.**

1. **They serve with SGLang and run Python 3.12 -- exactly our stack.** AReaL also rolls out
   through SGLang. The serving layer is shared, so this is not two incompatible worlds.
2. **They have an analogous group-advantage seam.** `OpenMLE-ERL/RL/
   adaptive_reward_advantage_utils.py` (267 lines) exposes `group_samples_by_group_index`,
   `score_to_group_adaptive_reward`, `compute_gspo_group_advantages` and
   `compute_ttt_entropic_advantages`. That is the same structural layer our
   `apply_decisions(advantages, loss_mask, group_sizes, modes)` operates at. Porting the
   routing seam is a MAPPING onto an existing seam, not a rewrite -- which is much cheaper
   than assumed before reading the code.

**What each choice actually costs.**

*Staying on AReaL* keeps 1166 tests, the routing seam, the mixture machinery, the credit
work, and hard-won knowledge of its failure modes (weight-sync deadlock, the stall at step
~150, the watchdog). It costs building an agentic harness, an operator set and an MLE
benchmark adapter from nothing -- roughly the 47k lines of Gym + Evo that OpenRSI already
ships -- and it leaves us at a scale where the intervention is measured NOT to reach the
benchmark.

*Moving the pivot to OpenRSI* buys the environment, the operators, the benchmark adapters and
released 30B/35B weights, which together remove both the harness gap and the scale blocker.
It costs learning slime/Megatron internals rather than FSDP, accepting **CC BY-NC 4.0**, and
patching a codebase with no documented plugin interface.

**Recommendation: build the PIVOT on OpenRSI; keep AReaL for token-level work and as the
source of methodology.** The reasoning is not preference. Our pivot contribution -- routing
the model-versus-harness axis -- requires a harness worth evolving and a benchmark where
harness changes are measurable. OpenRSI has both and AReaL has neither. The asset we would
otherwise be giving up, the group-advantage seam, turns out to have a structural counterpart
on their side, so it ports. And what genuinely transfers regardless of base -- the
matched-proportion controls, the credit-assignment analysis, the measured noise floors, the
mutation discipline -- is methodology, which is base-independent.

**Stated limit on this recommendation:** structure was inspected, nothing was run. The real
cost of patching slime is unknown until a first patch is attempted, and that is the cheapest
next thing that would confirm or overturn this.

### M23 vs OpenRSI: an LLM DOES decide the evolution action there, and the axis differs

Asked directly on 2026-08-31: is "let the LLM itself decide whether data evolves the model or
the harness" the same idea as OpenRSI (2607.28568)? **Related, and importantly not identical.**

| | OpenRSI / Frontis-MA1 | M23 as specified here |
|---|---|---|
| who decides | the 35B agent | an LLM router |
| decides WHAT | which of four operators to apply: Draft, Improve, Debug, Crossover | which TARGET a trajectory improves: the policy, or the harness |
| what each choice changes | the PROGRAM, in all four cases | different artefacts -- weights vs scaffold |
| how the decider is trained | execution-grounded SFT **and** RL over the operators | not built |

Every OpenRSI operator evolves the same artefact. The choice is *how* to improve the program,
never *whether the value of this trajectory belongs in the model or in the scaffold*. Their
operator set therefore does not cover the axis this project designed, and Co-Harness
(2607.22688) covers it only as a fixed partition. That gap is the whole of M23's remaining
novelty -- and it is narrower than "an LLM decides", which OpenRSI now clearly owns at scale.

**The part that should update our plan, not just our citations.** OpenRSI trains its decider
with **execution-grounded** credit: whether the resulting program actually ran better. Compare
what this project measured -- a per-batch scalar reward delta credited to 64 decisions, which
provably collapses every LinUCB arm to one parameter vector, and a per-prompt delta that
changed the router's behaviour without changing any benchmark. Our credit signal was the
binding constraint, and theirs is categorically stronger: per-decision, causally attached to
an execution the decision produced, and available without waiting for a policy update to
propagate.

So the lesson to take from them is **not** "use an LLM to decide". It is: *make the credit
signal something the decision directly causes*. Any M23 built here should be scored on whether
its credit is execution-grounded, and if it is not, the credit-assignment finding predicts it
will behave exactly like the contextual router did -- visibly active, measurably indifferent.

**Consequence for sequencing.** M23 remains a fourth controller in the ablation
(rule / learned-weights / learned-code / LLM), but it is now clearly downstream of two things
that matter more: a credit signal worth learning from, and a consumer for the harness axis.
Building an LLM router on top of the current batch-scalar credit would reproduce the measured
null with a more expensive decider.

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

### The mode vocabulary is UNDER-FACTORED: three axes, not one list (2026-08-31)

`TrainingMode` is a flat list -- `{rl, sft, distill, skip}` -- and that list conflates three
independent choices. Asked directly whether the routed decision should instead pick, per unit,
where the target comes from (teacher rollout, student rollout, oracle, ground truth) and what
objective is applied (KL, reverse KL, CE, reward). It should, and the flat list is why we
cannot currently ask that question.

**Axis A -- where the TARGET tokens come from**

| source | available without a teacher? | note |
|---|---|---|
| student's own rollout, filtered by reward | **yes** | the free self-target: a group with `solve_rate > 0` contains its own correct sample. Measured reach 29-33% of all groups |
| student rollout, scored by a teacher | no | the on-policy distillation setting: student's distribution, teacher's judgement |
| teacher's own rollout | no | off-policy; needs an importance correction the seam does not apply |
| oracle / ground-truth reference | dataset-dependent | GSM8K and MATH ship reference solutions; most RL datasets do not |
| none | yes | RL needs no target; SKIP uses nothing |

**Axis B -- what OBJECTIVE is applied to it**

| objective | what it optimises | where it already exists here |
|---|---|---|
| CE / max-likelihood on target tokens | mode of the target | this is what `sft` does, via a positive constant on a zero-advantage group |
| forward KL to a teacher distribution | mass-covering | **name only** (M6 records forward-KL as unimplemented) |
| reverse KL | mode-seeking | AReaL's own `distill_loss_weight` path (M5, DONE) |
| policy gradient weighted by reward | expected reward | `rl` |
| zero | nothing | `skip` |

**Axis C -- whose distribution the SAMPLES came from**

On-policy (student's current rollouts, importance ratio 1) versus off-policy (teacher rollouts
or replay, needing correction). Everything routed today is on-policy, which is exactly why the
self-target trick is sound: the policy-gradient step maximising `log p(y)` for a sampled `y`
IS the supervised step on `y` when the ratio is 1.

**What the current vocabulary actually spans.** `rl` = (no target, PG, on-policy). `sft` =
(own correct sample, CE, on-policy) -- i.e. rejection-sampling FT. `skip` = null. `distill` is
a registered NAME spanning an unspecified (teacher target, KL-family, on/off-policy) cell, and
`apply_decisions` correctly refuses it rather than guessing which cell it means. So of a
5 x 5 x 2 space we implement three cells and refuse one under-specified label.

**Why factoring is worth doing, and why NOT yet as arms.** Factoring makes the interesting
comparisons expressible: same target, different objective (CE vs reverse KL on the same
self-target) isolates the objective; same objective, different target (CE on own-sample vs CE
on oracle) isolates the target. Neither is askable with a flat list, and both are cheaper and
sharper than adding whole new modes.

But the measured position forbids running them as arms yet: three credit signals produced
three very different training trajectories and ONE identical capability outcome. A finer
factorisation multiplies the arms without touching the reason none of them separated. **The
factorisation is a representational fix and a paper-framing asset; it is not an experiment.**

**Concretely, the cheap first step** is to make `TrainingMode` carry `(target, objective)`
rather than a bare name, keeping the three implemented cells bit-identical -- so the existing
arms are unchanged by construction, and the vocabulary stops lying about how many choices are
being made. That is a refactor with an exact-rollback property, exactly like the mixture seam,
and it is the shape this repo has repeatedly shown it can verify.

### OPD as a ROUTED MODE, chosen on measured performance (asked 2026-08-31)

The proposal: on-policy distillation should not be a separate pipeline but one of the modes
the router can pick per unit, and it should be preferred where it actually performs better.
Recording it because the architecture already almost supports it, and because the parts that
are missing are specific.

**Already there.** `distill` is a REGISTERED `TrainingMode` and `known_modes()` reports it as
target-requiring: `{'rl': False, 'sft': True, 'distill': True, 'skip': False}`. Since the
mixture seam landed, a decision can also express a BLEND -- `{rl: 0.6, distill: 0.4}` is
representable today, and `RoutingDecision` validates it.

**Deliberately not there.** `apply_decisions` implements `_APPLIED = (RL, SFT, SKIP)` and
REFUSES `distill` rather than silently degrading it to SKIP. That refusal is correct and
should stay: a teacher-requiring mode needs a target tensor the seam does not have, and
quietly skipping it would report a distillation arm that never ran -- the exact failure class
that produced the inert contextual router and the inert random control.

**So "OPD as a routed mode" needs three things, in this order:**

1. **A teacher, and a target tensor at the seam.** Without it the mode is unimplementable, not
   merely unimplemented. M24's three recorded traps apply here unchanged.
2. **A distill branch in `apply_decisions`**, blending like the others: `new = a*original +
   d*distill_target + ...`. The mixture machinery generalises; the target does not exist yet.
3. **A credit signal that can tell distill from RL for the SAME unit.** This is the binding
   constraint, not the plumbing.

**Why (3) is the whole difficulty, from our own measurements.** "Prefer OPD where it performs
better" requires the controller to MEASURE which mode performed better per unit. This project
has now measured that a per-batch scalar cannot do that (it provably collapses every LinUCB
arm to one parameter vector), and that a per-prompt delta changes the router's visible
behaviour while moving no benchmark: three credit signals, three very different training
trajectories, one identical capability outcome. Adding a fourth mode to a controller that
cannot yet rank the three it has would produce a fourth indistinguishable arm.

The OpenRSI lesson (Sec. 2c) applies directly: their decider works because credit is
EXECUTION-GROUNDED -- did the artefact the decision produced actually run better. An OPD-vs-RL
choice has a natural execution-grounded analogue (did this prompt's solve rate improve more
under distill than under RL, on matched prompts), and that is the version worth building.
A preference learned from batch-mean reward is not.

**Status: NOT BUILT, and correctly sequenced behind a teacher (M24) and a credit signal that
can rank modes.** Recorded so the idea is not lost, and so it is not started in the wrong
order.

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

## 2d. LongHorizon-Harness (2608.01964) — and THE DECISION about what this paper is

**What it is.** Long-horizon execution reformulated as **task-state management**. The task
state is held explicitly OUTSIDE execution and updated only with facts *independently verified
from the environment*. The **Manage-Execute-Audit** loop: a *manager* maintains the state and
picks the next subtask, a *fresh-context executor* performs it (so errors do not accumulate in
a growing context), and a **read-only auditor verifies the resulting environment state** before
the next round. An `AgentAdapter` makes model and harness backends interchangeable without
modifying their native agent loops.

**Reported.** Qwen3.7-Plus 51.8 -> **80.7** on WeaveBench, 69.7 -> **77.2** on Terminal-Bench
2.1, 2.8 -> **8.3** on OSWorld 2.0; Claude Opus 4.7 20.0 -> **34.3** on an OSWorld 2.0 subset.
Consistent across models, harnesses and domains. It appears to be **inference-time scaffolding**
-- no training reported.

### Why this is the most important paper for this project so far

Read together with 2c, the two halves of our claimed axis are now BOTH occupied by strong,
recent, released work:

* **Model evolution** -- Frontis-MA1 trains a 35B agent over four program operators, 39 -> 71
  on MLE-Bench Lite.
* **Harness evolution** -- LongHorizon-Harness improves a *fixed* model by up to **+29 points**
  with no training at all.

That second number is the one to sit with. **A harness change bought +28.9 points on WeaveBench
without touching a weight.** Every intervention this project has measured -- token routing,
group routing, three credit signals -- moved capability by less than the noise floor. The
harness side of the axis is where the effect sizes are, and we have been spending GPU-weeks on
the model side at 1.5B.

### What is STILL not owned, stated precisely

Neither paper decides, per trajectory, **whether that trajectory's value belongs in the model
or in the harness.** Frontis-MA1's four operators all evolve the program. LongHorizon-Harness
improves the scaffold for every trajectory uniformly. Co-Harness (2607.22688) makes the choice
but with a FIXED rule (success -> model, failure -> harness). Nobody routes it, and nobody has
a credit signal for the routing.

### The connection that makes this a paper rather than a wish

Our own measurement says the binding constraint is credit: a per-batch scalar provably collapses
every LinUCB arm to one parameter vector, and a per-prompt delta changed behaviour while moving
no benchmark. OpenRSI's answer is execution-grounded credit. **LongHorizon-Harness's auditor
already produces exactly that** -- independently verified facts about the environment state,
per round, attached to the action that produced them.

So the auditor's verdicts can BE the credit signal for a model-versus-harness router. That is
not a wish: it is our measured requirement met by a released mechanism.

### STANDING RULE: every per-task config comes from a SOTA source, not from us

Raised 2026-08-31 and adopted. For each task, the configuration should be taken FIRST from the
strongest published result on that task and its repo, and only invented where no source exists.

Why this is a rule and not a preference: we have twice set a value ourselves and been wrong in
a way that invalidated measurements. Token caps calibrated on a 1.5B truncated 50% of MATH-500
on a 30B. A `cap_limited` threshold chosen against 1.5B rates fired on rows where the right
reading was not "edge case" but "measurement invalid". Both would have been avoided by reading
the source first.

Applied so far:

| setting | value | source |
|---|---|---|
| eval max response length | **32768** | OpenRSI `run_openmle_rl_*.sh`, `--eval-max-response-len` default |
| repeats per task | **3** | Terminal-Bench 2.1 leaderboard: pass@1 averaged over 3 repeats |
| TB baseline harness | **Terminus 2** | the leaderboard's own harness; ships in harbor, import verified |
| per-role caps | 16000 / 10000 / 8192 / 4096 | LongHorizon-Harness role configs |

The LHH pattern is the more important import: caps are **per role**, not one global value,
because a manager summarising state and an executor writing code have different length
distributions. Our suite still uses one cap per benchmark, which is the same error one level up.

**How to apply it going forward:** before running any new benchmark, find the strongest
published number on it, open that work's repo, and copy the generation and protocol settings.
Record the source in the config file itself, as
`experiments/harness/tb21_baseline_terminus2.yaml` does. A config with no cited source is a
config we will later discover was wrong.

### Generation caps, taken from THEIR configs rather than guessed (2026-08-31)

Our per-task caps were measured on a 1.5B and are wrong for a reasoning model -- MATH-500
truncated 50% and OlympiadBench 32.9% on the 30B. Rather than keep doubling, read what the
released work configures.

**OpenRSI (2607.28568):** all four RL run scripts carry
`--eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-32768}"`
(`OpenMLE-ERL/RL/scripts/run_openmle_rl_{a,}sync_{single,multi}_node.sh`). Other `max_tokens`
values in that tree are 2000/2048/5000 and belong to utility calls, not to model generation.
So **32768 is their evaluation default**, and the re-run now in flight uses exactly that --
matching a published configuration rather than picking the next power of two.

**LongHorizon-Harness (2608.01964):** role-dependent rather than single-valued --
16000, 10000, 8192 and 4096 for the substantive roles, with 1500 and 512 for utility calls.
This is worth copying as a PATTERN, not just as numbers: a long-horizon system does not use
one cap, it uses a cap per role, because a manager summarising state and an executor writing
code have different length distributions. Our own suite uses one cap per benchmark, which is
the same mistake one level up.

**Consequence for the per-task table.** The right shape is a cap per (benchmark x model class
x role), and we currently have one dimension of that. The immediate fix is only the model-class
dimension, since our scorer has no roles; the role dimension arrives with the harness work.

**Also worth noting:** OpenRSI's training-side `seq_length = 4096` is far below its eval cap of
32768. If that is the training context, a model trained at 4096 and evaluated at 32768 is being
asked to generalise well beyond its training length, which is a plausible contributor to
non-termination at long generations and is worth checking before blaming any routing arm for
it.

### Initialise from the best-known harness per task, then evolve (2026-08-31)

Proposed, and it resolves the constraint recorded below rather than dodging it. If
best-reported-harness-per-benchmark is a strong published static policy that we must beat,
then **start there and evolve from it** instead of evolving from scratch.

Why this is the right shape and not just optimism:

* **The baseline becomes the initialisation.** A policy initialised at the best static harness
  cannot do worse than it by construction, provided the evolution step is gated on measured
  improvement. The comparison "ours vs best static harness" then measures exactly the thing we
  add, with no scaling or base-model confound.
* **It is the cold start M9 was supposed to supply, and did not.** M9 was built to be the
  sensible written rule a learned controller must beat, and its audit showed it collapses to
  one predicate and does not de-confound anything. Best-known-harness-per-task is a genuinely
  informative prior that the literature already computed for us, which is what M9's threshold
  could not be grounded in.
* **It fixes the failure mode we measured.** The contextual router sat at ~93% RL before any
  credit arrived, purely because untouched LinUCB arms tie and `argmax` breaks ties
  alphabetically. Initialising at a known-good policy replaces that arbitrary starting point
  with an informative one, so the early phase is not wasted discovering what a leaderboard
  already knows.
* **Evolution then has a floor.** Every measured null in this project came from an arm that
  could drift anywhere. An arm that starts at a strong prior and only moves on verified
  improvement has a worst case equal to the baseline, which makes a null informative rather
  than merely disappointing.

**What it does not fix.** The credit problem is unchanged: deciding *when* a per-trajectory
deviation from the static best is justified still needs execution-grounded evidence, which is
falsifier (1). And a policy that never deviates is just the baseline wearing a controller --
so the reportable quantity is the deviation rate AND its effect, not the final score alone.

**Sequencing.** This becomes step 0 of the harness work: fix the per-task best harness from
published numbers, reproduce it ourselves at matched protocol (Terminus 2, e2b, pass@1 over 3
repeats), and only then allow per-trajectory deviation.

### The harness is already a measured axis in public leaderboards (2026-08-31)

Raised: different papers/repos ship benchmark-specific harnesses, and each is presumably best
on its own benchmark -- so pick the best harness per task. Checked, and this is stronger than
a hunch: **the field already measures it and reports it separately.**

* The Terminal-Bench 2.1 leaderboard states outright that it **evaluates the model and the
  agent harness TOGETHER**, and that *"the same model can therefore receive different
  Terminal-Bench 2.1 scores when paired with different agent harnesses."*
* A **"Terminal-Bench 2.1 (Best Reported Harness)"** listing is tracked as its own leaderboard
  variant, distinct from the standard-harness numbers. Somebody already found this distinction
  worth maintaining.
* Magnitudes are large: Claude Code with Opus 4.6 gained **+12.1 points** from a 2.1 harness
  update alone. LHH reports Qwen3.7-Plus 69.7 -> 77.2 on the same benchmark. Both are
  fixed-model, harness-only deltas.
* Top of leaderboard: GPT-5.6 Sol 89.5, Claude Opus 5 via Claude Code 89.1.

**Protocol we must match, taken from the leaderboard rather than invented:** Terminus 2 harness
in an e2b sandbox, **pass@1 averaged over 3 repeats per task**. Our own greedy-scoring work
measured a 0.020 noise floor and ~1 point of jitter on a single OlympiadBench score; a
single-repeat harness comparison would be indistinguishable from that jitter. Three repeats is
the published protocol AND the statistically necessary one here.

`harbor` is the open-source runner for this, supports custom agent implementations, and is
**verified installable in a plain venv (0.18.0, `harbor --help` works)** -- so the standard
harness and a custom one can be run through the same executor, which is what makes a fair
swap possible at all.

**What this does and does not do for the contribution.**

It HELPS: the model-versus-harness axis is not invented here. The field measures harness effect
at fixed model, publishes it, and finds it large. That is the premise our routing decision
rests on, and it is now cited rather than assumed.

It CONSTRAINS: "pick the best harness" is therefore NOT novel -- a leaderboard variant already
does exactly that, statically, per benchmark. Our claim has to stay narrower and sharper:
routing **per trajectory**, between spending it on the model or on the harness, with
**execution-grounded credit** from the auditor. Static best-harness selection is the baseline
we must beat, not the contribution.

**A concrete baseline this gives us for free:** best-reported-harness-per-benchmark is a strong,
already-published static policy. If per-trajectory routing cannot beat "always use the best
harness for this benchmark", that is a clean negative result and should be reported as one.

### Harness-swap at FIXED model: the cheapest instrument for the harness axis

Raised 2026-08-31: run Frontis-MA1 under **its own** harness first, because a model
post-trained on OpenMLE's four operators may underperform under a foreign loop -- and only
then swap the harness. That train/inference coupling is a real confound and the ordering is
right.

It also turns into the cleanest experiment available for the harness axis:

| arm | model | harness |
|-----|-------|---------|
| baseline | Frontis-MA1-30B | OpenMLE-Evo (its native loop) |
| swap | Frontis-MA1-30B | LongHorizon-Harness via `AgentAdapter` |
| swap | Frontis-MA1-30B | another strong harness (Claude Code / Codex integrations ship with LHH) |

Model fixed, harness varied. It measures **how much of the achievable gain lives in the
harness for a fixed policy** -- the quantity the model-versus-harness routing decision needs
and that nobody has reported. LHH's `AgentAdapter` exists precisely to do this "without
modifying their native agent loops", so the instrument is provided rather than built.

**Two outcomes, both publishable.** If the swap HELPS, the harness axis carries real value at
fixed model and routing to it is worth deciding. If the swap HURTS, that is train/inference
harness coupling measured directly -- a finding neither paper reports, and one that constrains
every "just use a better harness" claim.

**Sequenced before the routing work, not after.** Routing between model and harness is only
worth building once the harness delta at fixed model is known to be non-trivial. Measuring it
first also avoids the trap this project already fell into: building a controller before
establishing that the thing it controls moves the benchmark.

### VERIFIED: the code is released, it is MIT, and the audit signal is structured

Cloned `github.com/AMAP-ML/LongHorizon-Harness` and inspected it, 2026-08-31. This checks the
one assumption the decision below rests on.

* **License: MIT.** Not CC BY-NC. This matters more than it sounds -- OpenRSI's NC term is
  inherited by derivatives and would block a permissive artifact release, and its vendored
  AIRA-Dojo is separately NC. An MIT base removes that constraint entirely for anything built
  here.
* **Scale: 208 MB, 761 Python files, 231,666 lines.** `src/lh_harness` (the harness core),
  plus `eval/` containing **three** ready harnesses: `TB-harness` (Terminal-Bench 2.1, our
  B5), `WeaveBench-harness`, `OSWorldv2-harness`.
* **The MEA roles are real modules**, not prose: `manager.py` (carries `task_state`,
  `role_verified_context_chars`, a round-indexed `_human_gate`), `auditor_agent.py`,
  `role_prompts.py`, plus `environment/`, `plugins/`, `supervisor/`.

**The finding that matters.** `auditor_agent.py:182`:

```
def audit_report_from_episode_result(result: EpisodeResult, round_index: int, *,
                                     language: str = "en") -> AuditReport
```

with the docstring: *"The role manager stores auditor output as natural language, but the final
report and stop checks need a compact status, integrity flag, and artifact-deletion ledger."*
The returned `AuditReport` carries `round_id`, a `status` (observed values include `blocked`,
`missing`, `runtime_failed`, `delete_declared_unverified`), an integrity flag and a deletion
ledger.

So the auditor's output is **structured, per-round, and derived from the environment rather
than from the model's self-assessment**. That is precisely the execution-grounded, per-decision
credit signal this project measured itself to be missing -- available as a typed object, not
as free text to be parsed.

**What this does NOT establish.** Nothing has been run: no episode executed, no `AuditReport`
observed from a live rollout, no Terminal-Bench task attempted here. Whether the status
granularity actually separates *model-value* from *harness-value* per trajectory is falsifier
(1) in the decision below, and it remains open. Reading a type signature is not measuring a
signal.

### DECISION (2026-08-31): what this paper should be

**Claim.** Long-horizon agent trajectories carry value for two different targets -- the policy
and the scaffold -- and *which* target a trajectory should be spent on is decidable per
trajectory, from execution-verified evidence, better than by the fixed rule Co-Harness uses or
the uniform treatment both 2c and 2d apply.

**Why it can beat SOTA rather than tie it.** The comparison is against a FIXED base model with
a FIXED harness, adding only the routing decision. That is a delta at fixed base -- the only
kind of result a reviewer does not discount as scaling. The baselines are strong and released,
which makes the comparison meaningful rather than convenient.

**Why it is innovative.** The orthogonal model/harness axis generalises Co-Harness's forced
partition into an ablation; the credit signal is execution-grounded rather than reward-scalar;
and the negative results already measured here (token routing null, group routing null, the
LinUCB collapse proof) are the evidence that the naive versions do NOT work, which is what
makes the working version worth reporting.

**What this decision RETIRES.** Grinding further 1.5B GRPO routing arms. Result 7 already shows
three credit signals producing one identical capability outcome; a fourth arm is a fourth
indistinguishable number. Those runs become the paper's *negative-result* section and its
methodology contribution, not its headline.

**Falsifiers, named now.** (1) If the auditor's verdicts turn out to be too coarse to separate
model-value from harness-value per trajectory, the credit problem is unsolved and the claim
fails exactly as the 1.5B arms did. (2) If a fixed rule (Co-Harness's) matches the router at
matched cost, the routing adds nothing and should be reported as such. (3) If harness gains
dominate so completely that model updates never win the comparison, the axis is real but
degenerate, and that is also a reportable finding.

**Unverified:** nothing from 2608.01964 has been downloaded or run; code availability was not
established from the abstract page. The numbers above are theirs as reported.

## 2c. Frontis-MA1 / OpenRSI (2607.28568) — read 2026-08-31, and it changes the positioning

**What it is.** Yang et al., Frontis-MA1: a 35B agent post-trained for machine-learning
engineering, plus **OpenMLE**, a released full-stack system for recursive self-improvement.
Four atomic program-evolution operators -- **Draft** (write a program), **Improve** (refine a
parent using execution feedback), **Debug** (repair a failure), **Crossover** (recombine two
parents) -- composed into long-horizon search. Trained with execution-grounded SFT *and* RL on
data deduplicated against every evaluation benchmark.

**Reported.** MLE-Bench Lite, 12h/task on ONE RTX 4090 capped at 12 GB: base **39.39%** medal
average, OpenMLE-Evo **60.61%**, OpenMLE-Evo-Max **71.21%**. Held-out NatureBench Lite:
Match-SOTA 50% -> 70% by swapping in the trained model.

**Released.** `github.com/FrontisAI/OpenRSI` under **CC BY-NC 4.0** (non-commercial; NOTICE
covers third-party terms). Weights on HF: `FrontisAI/Frontis-MA1-35B` and
`FrontisAI/Frontis-MA1-30B`, plus GGUF. Training code is included, not just inference:
`OpenMLE-ERL/SFT/`, `OpenMLE-ERL/RL/`, `OpenMLE-Evo/`.

**They state their own limitation**, which is worth copying: the Evo-Max number "changes the
search system through benchmark-independent experience priors and asynchronous search, and
should not be interpreted as a pure model gain." That is the same discipline this project uses
for its own arms.

### Why this is a problem for the current framing, stated plainly

"Self-evolving LLM" now has a 35B system with released weights, a released stack, and a
39->71 improvement on a real MLE benchmark. **Our measured position is a three-way null at
1.5B** (see the credit-signal entry in EXPERIMENTS.md): batch credit, per-prompt credit and
random assignment are indistinguishable on MATH-500 and OlympiadBench. A per-group routing
result at 1.5B is not competitive with that as a systems contribution, and pretending
otherwise would not survive review.

### Why it is also the best opening this project has had

1. **It solves the scale blocker without training.** 27B and 32B both failed to train on these
   boxes. Frontis-MA1-30B does not need training -- it needs serving, and an H200 at 141 GB
   can do that. The base model stops being the bottleneck.
2. **It supplies the harness action space M10 lacks.** Our `HarnessAction` axis is
   `PROPOSE/VALIDATE/NONE`, is inert (nothing writes `can_evolve_harness`, `actor.py:283`
   discards `.harness`), and is thin. Draft/Improve/Debug/Crossover is a concrete, trained,
   evaluated operator set. It is exactly the consumer the axis has never had.
3. **It supplies a frontier benchmark we can actually afford.** MLE-Bench Lite runs at
   12h/task on a single 12 GB card. This is the code/SDE benchmark the project has wanted and
   repeatedly deferred as unaffordable; it is not.
4. **Their operators partition by FUNCTION, not by model-vs-harness.** Draft/Improve/Debug/
   Crossover all evolve the program. Nothing in their operator set decides whether a
   trajectory should improve the *policy* or the *harness* -- which is precisely the
   orthogonal axis this project designed and which Co-Harness (2607.22688) forces into a
   partition. That gap is where a contribution could sit.

### The honest shape of a contribution on top of it

Their stack decides WHICH operator to apply. It does not decide whether a trajectory's value
lies in updating the model or in evolving the harness, and it does not address credit
assignment for that decision. This project has: a measured result that per-batch scalar credit
provably collapses every LinUCB arm to one parameter vector; a per-prompt credit mechanism;
matched-proportion controls; and an orthogonal model/harness axis. Those transfer to their
operator set in a way they do not transfer to a 1.5B GRPO run.

**This is a pivot, and it should be called one.** The 1.5B routing results do not carry over
as a headline; they carry over as METHODOLOGY (the controls, the credit analysis, the noise
floors). Any adoption must respect CC BY-NC 4.0 and attribute both the paper and the stack.

**Not yet done, and none of it should be claimed until it is:** nothing from OpenRSI has been
downloaded, no weight has been served, MLE-Bench Lite has never been run here, and the
operator set has not been mapped onto `HarnessAction`. The numbers above are THEIR reported
numbers, read from the paper and repo on 2026-08-31, not reproduced.

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

7. **NULL: the learned router does not beat its matched control.** `router=contextual` vs
   `router=random` at the contextual arm's MEASURED proportions, both at `globalstep149`:
   MATH-500 **-0.0020**, OlympiadBench **+0.0000** (675 problems, both at a cap where neither
   arm is budget-bound). The arms differ only in WHICH unit gets which mode, so at matched
   proportions the per-unit decision buys nothing. Mechanism measured, not guessed: a single
   per-batch scalar credited to all 64 decisions converges every LinUCB arm to the same
   parameter vector, and the router's mode mix stayed at uniform thirds for 129 steps.

**Nothing yet demonstrates the method works.** Results 1–4 are evaluation contributions;
results 5 and 7 are nulls; result 6 is a property of the problem, not of the method. Result 7
is the important one: it falsifies "this router with this credit signal beats its control",
and it localises the failure to credit assignment rather than to routing. The `ctxpc` arm
(`credit="prompt"`) is the designed test of that localisation and is running.

> **The composition numbers are CORRECT as published (retraction, 2026-08-31).** An earlier
> note here claimed the solved share was inflated ~1.8x. That was an artifact of a regex whose
> `solved_group_fraction` pattern also matched `unsolved_group_fraction`, interleaving the two
> series. With a non-aliasing match the identity `silent == solved + unsolved` holds to 1e-5
> across every run checked, and `results.tex`'s table reproduces exactly (step0l batches
> 44-61: silent 0.3592, solved 0.3145, unsolved 0.0447, share 0.8755). Both the 87.5% figure
> and the 7x reach argument stand.

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
1b. **RUNNING, and trending null.** `router=contextual` (A100) against `router=random` at the
   contextual arm's MEASURED proportions (H200). At 129 steps the learned router shows no
   preference, and the cause is measured: a single per-batch scalar credited to all 64
   decisions carries no information separating the arms. Expect a null against the matched
   control, and report it as one.
1c. **Per-prompt credit assignment** -- the concrete fix implied by 1b, and the first thing
   that could make a learned controller work here. Credit a prompt's change in solve rate
   between the batch where it was routed to mode m and its next appearance. Needs prompt
   identity carried through the pipeline; today's `unit_id` is batch-local by construction.
2. **Give `route_batch` a caller** -- otherwise M2 and the new harness axis are dead code.
   NOTE this is now the ONLY remaining "no caller" gap; the group-level seam is closed.
3. **Re-measure the split on OlympiadBench.** 0.875 is a property of GSM8K at a high solve
   rate, not a constant; a harder task moves mass to the unsolved branch.
4. **M9 rule evolve-policy** -- cold start *and* the baseline the learned one must beat.
   **BUILT and AUDITED 2026-08-31** (`selfevo/routing/rule_policy.py`, `router=rule`), and the
   audit changed what this item can deliver. The rule is behaviourally identical to
   `router=solve_rate` under binary graders, so running both is double-reporting, not two
   arms. Running the rule against `ctx` is still worth doing, but it CANNOT be reported as
   "learned beats written at matched inputs": the 1-feature-vs-7 confound survives, because
   only `reward_std` has a measurement behind it. Nor is it matched in proportions -- with no
   teacher the rule emits only `rl`/`skip` against `ctx`'s rl 0.295 / sft 0.353 / skip 0.353,
   so each arm still needs its own `router=random` control at its own measured proportions.
   The cheapest thing that would actually move this item is a measurement grounding a SECOND
   predicate -- the M15 feature ablation -- not another controller.
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
