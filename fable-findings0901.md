# Fable findings — 2026-09-01

Literature verification and positioning review for the self-evolving-LLM paper. Three parallel
searches (Opus 5 subagents, web-grounded, all URLs fetched today) plus the synthesis judgement.
Everything below is cited; anything the searches could not confirm is marked UNVERIFIED rather
than guessed. Sections 4 and 5 are appended as the last search lands.

---

## 1. The prior art we cite, verified — and where GOAL.md is wrong about it

| # | ID | Actual title | Decision-maker | Granularity | GOAL.md description |
|---|---|---|---|---|---|
| 1 | [2605.27276](https://arxiv.org/abs/2605.27276) | SIA: Self Improving AI with Harness & Weight Updates | **frozen prompted LLM** ("Feedback-Agent") | per-iteration | **wrong** — not "task-level algorithm selection" |
| 2 | [2607.22688](https://arxiv.org/abs/2607.22688) | Co-Harness: Co-Evolving Harnesses and Model Weights | **fixed rule** + LLM critic in a fixed taxonomy | per-trajectory → per-round | correct |
| 3 | [2606.03108](https://arxiv.org/abs/2606.03108) | EvoTrainer: Co-Evolving LLM Policies and Training Harnesses | **frozen prompted LLM** (Claude Sonnet 4.6), human-gated | per training version | correct |
| 4 | [2604.11297](https://arxiv.org/abs/2604.11297) | The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping (MEDS) | none — no routing at all | per-prompt memory | mechanism correct; **it is not routing support** |
| 5 | [2605.12484](https://arxiv.org/abs/2605.12484) | Learning, Fast and Slow: Towards LLMs That Adapt Continually | none — both channels always on | per-task + weights | numbers exact; **"bounds cadence claims" is unsupported** — no bounds, no cadence analysis |
| 6 | [2607.12227](https://arxiv.org/abs/2607.12227) | Rethinking the Evaluation of Harness Evolution for Agents (AI2 + UW) | n/a — evaluation paper | n/a | **narrower and more adversarial than described** — see §1.2 |
| 7 | [2504.14945](https://arxiv.org/abs/2504.14945) | LUFFY: Learning to Reason under Off-Policy Guidance (**NeurIPS 2025**) | fixed | per-group + per-token | mechanism is **batch construction AND gradient shaping** (`f(x)=x/(x+γ)`, γ=0.1) |

### 1.1 The framing sentence is factually wrong for two of three

GOAL.md's novelty rests on "prior work decides with a FIXED RULE, we learn it." Only Co-Harness
has a genuine fixed rule. SIA and EvoTrainer decide with a **frozen prompted LLM** — neither a
rule nor a learned policy. The correct contrast is *frozen-prior vs learned*, and GOAL.md already
concedes (M23 positioning note) that the LLM-router idea is "a combination of existing lines."

### 1.2 SIA states our contribution as its own future work, verbatim

SIA v2 §9: *"Meta-RL over the action-selection policy. The Feedback-Agent currently selects
between harness and weight updates using a frozen LLM prior. A more principled approach treats
the selection policy itself as the object to be learned: run SIA across a distribution of
tasks, treat each (trajectory, action, outcome) triple as a transition in an outer MDP, and
train the selector via RL on that outer MDP."*

Not blocking — nobody executed it — but publicly staked as of May 2026. The paper must cite
this as the origin of the framing and position as execution + measurement, or a reviewer who
knows SIA will do it for us. SIA also selects among RL algorithms (PPO+GAE, GRPO, entropic
advantage weighting, REINFORCE+KL, BoN-BC, DPO) "conditioned on trajectory observations rather
than a fixed schedule" — so the {loss} axis is theirs too, at per-iteration granularity.

### 1.3 The AI2 paper is a required adversarial control, not a friendly citation

2607.12227 demands harness evolution be compared against "simple task-level search baselines
under **matched feedback and inference budgets**" on **held-out** tasks, and reports that on
Terminal-Bench 2.1 with GPT-5.4 and Claude Opus 4.6, *"automatic harness evolution does not
consistently outperform simple test-time scaling methods and exhibits limited generalization."*
Four of its authors (Zhu, Yuan, Chen, Xiao) also wrote Co-Harness three days later; the same
group shipped the method and the critique, and the critique's v2 (27 Aug) is the newest revision.
GOAL.md §4 currently has **Matched inference budget: NOT MET** and **Matched feedback budget:
NOT MET** — those two rows are exactly what this paper will be used to reject us on.

### 1.4 Two sentences in GOAL.md to delete or rewrite

- "Learning Fast and Slow bounds the cadence claim" — it does not; it is a plasticity paper and
  the 70%-less-KL result is drift from base, not update scheduling.
- MEDS as support for a routing narrative — it is anti-repetition reward shaping with HDBSCAN
  over latter-half layer logits at the final answer token, `r̃ = r − min(α log(|C_k|+1), β)`.
  The clustering feature is real and reusable; the routing framing is not in the paper. Also:
  its abstract claims "up to 4.13 pass@1" while the body reports single cells like Qwen3-8B
  OlympiadBench 44.69→61.12 — cite the abstract figure, not the cell.

### 1.5 UNVERIFIED

Venues for all five 2026 preprints (no acceptance visible); SIA's full affiliation list; LUFFY's
OpenReview record (bot-blocked; NeurIPS 2025 confirmed via the authors' README instead);
absence of same-group follow-ups for SIA / EvoTrainer / MEDS / Fast-and-Slow (searched, none
found, cannot prove absence). Not opened: HarnessX [2606.14249](https://huggingface.co/papers/2606.14249),
TTHE [2607.08124](https://arxiv.org/html/2607.08124v1).

---

## 2. What has appeared since April 2026 — the threat map

**Headline:** no paper found does the full claim — nothing *learns* per-prompt-group routing over
{evolution target} × {loss/source} from trajectory features. But every wall of that cell is
now built, and two papers have staked the framing.

### 2.1 DIRECT THREATS

| paper | ID | date | what it does | why we survive | what it costs us |
|---|---|---|---|---|---|
| **SIA** | [2605.27276](https://arxiv.org/abs/2605.27276) | May 26 | both our axes in one working system; LawBench 45.0→70.1, GPU kernels, scRNA | decision is a frozen Claude prior, per-iteration; learned selection is their future work | must cite as origin; must beat their frozen-LLM selector head-to-head |
| **Next-Gen Agentic RL Systems Enable Self-Evolving Agents** (Ant/HKUST/Tsinghua) | [2607.01120](https://arxiv.org/abs/2607.01120) | Jul 2 | a "unified agent evolution control plane" that "automatically decides, based on trajectory statistics, when to update policy weights or evolve the in-context harness" | position paper; hand-written symptom→surface map; prototype implements only the weight surface; no baselines | claims our framing sentence verbatim; reviewers who read it will say the idea is not new |
| **HASE** | [2607.03935](https://arxiv.org/abs/2607.03935) | Jul 4 | a single Qwen3-8B policy learns solve-vs-edit-harness **end-to-end under GRPO** with no hard-coded planner; frozen-weights ablation shows co-evolution is necessary | routing is implicit in the policy, no explicit feature controller; per-trajectory not per-group; GRPO is the only loss | kills "learned, not hand-written" as a standalone contribution |

### 2.2 PARTIAL OVERLAP — must cite and distinguish, several are mandatory baselines

| paper | ID | date | mechanism | relation to us |
|---|---|---|---|---|
| **SRPO** | [2604.02288](https://arxiv.org/html/2604.02288v1) | Apr 2 | routes each rollout in a group of 8 to GRPO or self-distillation by `z = (1−c_i)·m_i` (wrong AND a correct sibling exists); all-wrong groups → GRPO; entropy-aware token weighting downstream; Qwen3-8B 77.4 vs 74.0 GRPO | **our architecture with the learning removed. Mandatory baseline.** |
| **HPT** | [2509.04419](https://arxiv.org/abs/2509.04419) | Sep 2025 / Jan 2026 | `α = 1 if P > γ else 0` (RL), `β = 1 if P ≤ γ` (SFT), P = mean verifier score over n=8 | **the solve-rate threshold our rule collapsed to in 102/102 contexts.** The rule a learned router must strictly dominate |
| **EvoTrainer** | [2606.03108](https://arxiv.org/abs/2606.03108) | Jun 2 | Claude Sonnet 4.6 revises the training recipe from dead-group ratio, reward std within groups, truncation rate, length distribution; selects GRPO/GSPO cores, reward structures, filters | owns "trajectory features → training decision"; leaves us granularity + learnedness |
| **RSTG** | [2608.00782](https://arxiv.org/abs/2608.00782) | Aug 1 | negative zero-variance groups → teacher distillation weighted by teacher confidence; else GRPO; +4.02 math, +3.05 code | per-group loss routing on all-wrong groups, by rule |
| **LSPO** (NVIDIA) | [2607.27787](https://arxiv.org/html/2607.27787v1) | Jul 30 | detects zero-reward-cliff groups; SFT on gold updates only a LoRA adapter, RL updates only the base — strictly disjoint gradient routing; 15/16 cells win, +10.7 AIME24 pass@4; ~43% of cliff groups converted | **this is the gold-on-unsolved batch-construction arm we were about to build.** Now a baseline |
| **TREK** (LinkedIn) | [2607.05339](https://arxiv.org/abs/2607.05339) | Jul 6 | low-pass-rate prompts → verified solutions → forward-KL phase → back to GRPO; AIME25 36.9→40.3 | per-prompt, staged, threshold-routed |
| **GAC** | [2605.26184](https://arxiv.org/pdf/2605.26184) | May 25 | adaptive SFT/RL mixing weights from online gradient variance + disagreement; beats fixed AND rule-based schedules | adaptive-but-not-learned, continuous-not-discrete quadrant |
| **BRIDGE** (ICML 2026) | [2509.06948](https://arxiv.org/abs/2509.06948) | | bilevel: LoRA adapter meta-learns to coordinate SFT and RL gradients | the strongest *learned* SFT/RL mixing prior — continuous adapter, not feature-conditioned |
| **LLMZero** (AWS) | [2606.18388](https://arxiv.org/abs/2606.18388) | Jun 16 | LLM agents tree-search training strategies per checkpoint | stage granularity, LLM-proposed |
| **E-SPL** | [2602.14697](https://arxiv.org/abs/2602.14697) | Feb | joint prompt-population evolution + RL on weights inside one loop | cleanest prior instance of simultaneous weight+harness evolution in an RL loop |
| **Continual Harness** | [2605.09998](https://arxiv.org/abs/2605.09998) | May 11 | LLM refiner CRUD-edits harness mid-episode + online PRM co-learning | one of two works Lilian Weng cites for harness+weight co-evolution |
| **AGPO** | [2605.20722](https://arxiv.org/abs/2605.20722) | | group reward dispersion, skewness, probe-vote entropy, policy entropy, KL drift → adaptive clipping + temperature | closest feature set among pure-GRPO variants |

### 2.3 Evaluation threats — these shape what counts as evidence

- **[2607.12227](https://arxiv.org/abs/2607.12227)** — held-out tasks + budget-matched test-time-scaling baselines, or the gain "is just search."
- **Harness Updating Is Not Harness Benefit** [2605.30621](https://arxiv.org/pdf/2605.30621) (May 28) — updates from a 9B yield gains comparable to Opus 4.6; benefit is non-monotonic in capability; invest in the solver, not the updater.
- **SFT-then-RL Outperforms Mixed-Policy Methods** [2604.23747](https://arxiv.org/abs/2604.23747) (Apr) — many mixed-policy gains rest on loss-weighting bugs in DeepSpeed/OpenRLHF. **Any RL/SFT-mixing claim must audit its loss normalisation against this.** E1's 96/96 bit-identical rollback proves vanilla is reachable, not that the mixed weights are correct.
- **Evo-Bench** [2608.09096](https://arxiv.org/html/2608.09096) (Aug) — harness evolution shows "early saturation," best evolved 46.3 vs human 47.5.
- **Do Post-Training Algorithms Actually Differ?** [2603.19335](https://arxiv.org/pdf/2603.19335) — scale-dependent ranking inversions; argues *against* any fixed global choice (a motivation for us).
- **AdaBack** [2506.18110](https://arxiv.org/abs/2506.18110) — adaptive methods give limited benefit once RL saturates training reward. The closest thing found to "when adaptivity collapses."

### 2.4 The all-wrong-group landscape (post-LUFFY), for related work

GPPO [2606.01281](https://arxiv.org/abs/2606.01281) · Advantage Collapse in GRPO [2605.21125] · EP-GRPO [2605.04960] · GEPO [2607.16850] · Scaf-GRPO [2510.19807](https://arxiv.org/abs/2510.19807) (ICLR 2026, tiered hints on plateau) · GCPO [2510.07790] (golden answers when all wrong) · HiLL [2604.00698](https://arxiv.org/pdf/2604.00698) (learned hint policy tracking all-incorrect ratio) · NGRPO [2509.18851] · Reuse your FLOPs [2601.18795] · Prompt Replay [2603.21177] · Actor-Curator [2602.20532] (bandit over *data*, not algorithms) · Selector-Guided Autonomous Curriculum [2605.01823] · ExGRPO (ICLR 2026, LUFFY follow-up, own off-policy experience).

### 2.5 Broader self-evolving landscape (RELATED ONLY)

EvolveNet [2608.04968] · SHAPER [2608.11350] · Prime Agent [2608.23552] · CAFE [2608.24794] · Adaptive Auto-Harness [2606.01770] · DemoEvolve [2605.24539] · SHE [2608.09885] · MetaEvolve [2607.21971] · MetaSkill-Evolve [2607.05297] · surveys [2607.13104], [2507.21046] · SEAL [2605.24426] · Recursive Harness Self-Improvement [2607.15524](https://arxiv.org/abs/2607.15524) · **Lilian Weng, "Harness Engineering for Self-Improvement," Jul 4 2026** — cites only SIA and Continual Harness for harness+weight co-evolution, is skeptical of SIA ("confounding choices," "provisional"), and does not discuss learned routing or per-group loss selection.

### 2.6 Searches that came up EMPTY — stated explicitly

1. A learned router (classifier / bandit / RL meta-policy) selecting the training algorithm or loss at per-prompt or per-group granularity in LLM RL post-training. Every hit was rule-based, LLM-prompted, or continuous reweighting.
2. **Bandit-over-training-modes** in LLM post-training. Bandits exist over prompts/data (Actor-Curator), over LoRA experts at inference (Red-Bandit 2510.07239), over model routing (BaRP 2510.07429) — never over training algorithms.
3. **Any paper measuring whether learned per-sample algorithm selection degenerates to a fixed threshold.** Nearest is AdaBack's saturation remark. *"This is an open question you can own."*
4. NeurIPS 2026 / ICLR 2027 OpenReview submissions matching the claim — OpenReview full text not searchable; ICLR 2027 not yet public. **Residual blind spot: do a manual OpenReview sweep before submission.**
5. Major-lab releases since April doing learned what-to-evolve routing.

### 2.7 Ranked — the five most threatening

1. **SIA** [2605.27276] — both axes in one working system; learned selection is *their stated future work*. That sentence is our paper.
2. **Next-Gen Agentic RL Systems** [2607.01120] — claims the framing sentence verbatim, as a position paper.
3. **HASE** [2607.03935] — the only paper where what-to-evolve is learned end-to-end under GRPO, with the frozen-weights ablation.
4. **SRPO** [2604.02288] — our per-sample loss-routing architecture with learning removed. If we don't strictly beat it, the learned router is unmotivated.
5. **EvoTrainer** [2606.03108] — already uses dead-group ratio, reward std and truncation as decision features.

Runners-up: HPT (the threshold to dominate), RSTG and LSPO (per-group loss routing on all-wrong groups), and as methodology threats 2607.12227 and 2604.23747.

### 2.8 The defensible gap, in the scanner's words

*"Learned + explicitly feature-conditioned + per-prompt-group + jointly over {evolution target} ×
{loss/source}, with an ablation showing when it collapses to a threshold. No paper found occupies
that cell. The collapse analysis is entirely unclaimed — make it the paper's second contribution,
because it is the part nobody has done and it inoculates against the 'adaptive gains are noise'
reviewer."*

---

## 3. What this does to the project's own measurements

Read against §2, the project's recorded results stop looking like three failures and start
looking like the unclaimed contribution:

| measurement (already in hand) | what it now means |
|---|---|
| rule policy ≡ solve-rate predicate in **102/102** contexts under binary graders | the hand-written rule collapsed to **HPT's threshold**. This is the degeneracy nobody has measured (§2.6 item 3) |
| GPU null: `router=contextual` vs `router=random` at matched proportions, MATH-500 −0.002, OlympiadBench +0.000 (GOAL.md Result 7) | a learned router over outcome features does not beat its rate-matched control — the null the field has not reported because nobody built the control |
| credit identity: batch-scalar credit contributes zero bits about the mode; rotating contexts rotates fitted parameters to `atol=1e-12` | explains the null mechanistically; **no bandit-over-training-modes exists in the literature to have hit this** (§2.6 item 2) |
| L1-from-uniform rises 0.067→0.173 on the router proven incapable of learning | a metric trap the adaptive-training literature will hit; the subset-contrast + shuffle-control replacement is reusable |
| per-prompt credit: 0.098 → 0.779 subset contrast, shuffle → 0.102 (simulator) | the escape route exists when a **non-outcome** feature carries signal — untested on GPU |
| binary-reward collapse theorem (every estimator = α(k)·(r−r̄)) | the structural reason: under verifiable binary rewards all group outcome statistics are functions of the pass count k |
| unsolved = 60.9% of channel, no target by construction; gold cannot enter via the advantage seam | the k=0 branch — now well-trodden by LSPO/RSTG/GCPO, which become baselines rather than the contribution |

---

## 4. Pending: credit-assignment and degeneracy novelty search

*(appended when the third search lands: is the identity a known folk result; is "everything
collapses to the solve-rate predicate under binary graders" already stated; is self-selection
drift masquerading as learning documented.)*

## 5. Pending: judgement

*(appended after §4.)*
