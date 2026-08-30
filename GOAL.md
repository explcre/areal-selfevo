# GOAL — self-evolving LLM agents

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

### Benchmarks

| # | requirement | status |
|---|---|---|
| B1 | Frontier math | **DONE** — MATH-500, AIME, AMC, HMMT, OlympiadBench all scored |
| B2 | Frontier code | **NOT BUILT** — LiveCodeBench data obtained, generation/grading unwired |
| B3 | Enterprise SQL | **NOT BUILT** — Spider2 + BIRD on disk, never run |
| B4 | BioMysteryBench | **NOT BUILT** |
| B5 | GeneBench-Pro | **NOT BUILT** |
| B6 | Ornith / BigBang / EvoTrainer models | **PARTIAL** — Ornith-9B and 35B-A3B downloaded + scored; BigBang not pulled |

---

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
