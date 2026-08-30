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

### Method

| # | requirement | status | evidence / gap |
|---|---|---|---|
| M1 | Per-**token** routing | **DONE** | wired into `grpo_loss_fn`; 3 audits; measured reach **1.7%** |
| M2 | Per-**cluster** routing | **PARTIAL** | `ClusterRouter` built + audited, but makes **0 decisions** `SolveRateRouter` doesn't, and `route_batch` has no caller |
| M3 | Per-**sample** routing | **DONE** | `SolveRateRouter`, registered |
| M4 | Per-**task** routing | **NOT BUILT** | and it is SIA's, so low value |
| M5 | RL/reverse-KL mix per token | **DONE** | extends AReaL's own `rl_loss_weight`/`distill_loss_weight` |
| M6 | SFT / forward-KL modes | **PARTIAL** | registered as names; only RL↔reverse-KL reaches the loss |
| M7 | **Teacher** supplying routed units | **NOT BUILT** | ← blocks the whole claim; without it routing only *deletes* gradient |
| M8 | Learned meta-controller | **NOT BUILT** | `EVOLVE_POLICIES` = bare strings |
| M9 | Rule evolve-policy (cold start + baseline) | **NOT BUILT** | needed before "learned" is falsifiable |
| M10 | Evolve model / harness / reward | **NOT BUILT** | `EVOLVE_TARGETS` = bare strings |
| M11 | Cadence: frozen / alternating / simultaneous | **NOT BUILT** | `CADENCES` = bare strings |
| M12 | Evolvable reward formula | **NOT BUILT** | |
| M13 | MEDS reward shaping | **NOT BUILT** | `SHAPERS = {"none": None}` |
| M14 | BigBang two-level critic | **NOT BUILT** | `"two_level": None` |
| M15 | Trajectory observability → policy inputs | **NOT BUILT** | |
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
| E6 | Both boxes busy | **PARTIAL** — repeatedly idle; now structurally fixed with long training jobs |
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

**Nothing yet demonstrates the method works.** Results 1–4 are evaluation contributions;
result 5 is a null.

---

## 6. Critical path

1. **M7 — supply a teacher.** Until routed units *gain* a signal, every arm is gradient
   deletion, and no positive result would mean what the paper claims.
2. **Measure the 57% split** (solved vs unsolved) with the fixed metric — decides what that
   channel routes *to*.
3. **Give `route_batch` a caller** — otherwise M2 is dead code.
4. **M9 rule evolve-policy** — cold start *and* the baseline the learned one must beat.
5. Then M8 learned controller, then B2/B3 benchmarks.
