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

## 4. Are our three "methodological findings" already known? — Yes, all three, in some form

Third search, web-grounded, ~40 targeted queries; empty searches listed at the end.

### 4.1 The credit identity — a special case of a 1999 result; the fix is a re-derivation

| claim | status | canonical citation |
|---|---|---|
| batch-scalar credit ⇒ a linear meta-controller cannot learn a preference | **special case** of Wolpert & Tumer's *zero-learnability* limit in the COIN framework (a global utility handed unchanged to every agent has learnability → 0); in bandit terms the degenerate case `r(x,a)=r(x)` where every policy is optimal — "arguably too trivial to have been written down." Exact linear-estimator identity NOT found in 14 phrasings | Wolpert & Tumer [cs/9905004](https://arxiv.org/abs/cs/9905004); COMA [1705.08926](https://arxiv.org/abs/1705.08926) (difference rewards) |
| every working predecessor measures credit per-arm or restores contrast by resampling | Graves et al. 2017 §3.3 analyses estimator bias and uses per-arm prediction gain; Learning-to-Teach and AutoLoss use a single terminal scalar but run ~50 independent episodes so different action sequences get different rewards — **our controller is single-stream, so the shared scalar never varies with the action** | [1704.03003](https://arxiv.org/abs/1704.03003), [1805.03643](https://arxiv.org/abs/1805.03643), [1810.02442](https://arxiv.org/abs/1810.02442) |
| per-prompt credit across time as the fix | **ESTABLISHED, not novel.** SPO keeps a persistent per-prompt Beta value tracker with KL-adaptive discount, replacing GRPO's group baseline (+3.4pp maj@32); DBB, BV-Blend, GRESO (per-prompt zero-variance streak for a *routing* decision), and **PAC (Aug 31 2026)** — Thompson sampling over per-task arms with progress = least-squares slope of that task's own reward history — are the same prescription | SPO [2509.13232](https://arxiv.org/abs/2509.13232); DBB [2603.18444](https://arxiv.org/abs/2603.18444); BV-Blend [2606.28707](https://arxiv.org/abs/2606.28707); GRESO [2506.02177](https://arxiv.org/abs/2506.02177); PAC [2608.30528](https://arxiv.org/abs/2608.30528) |
| adaptive SFT/RL mode control | all *global* weights or heuristic switches: AMFT (meta-gradient global μ), SRFT (entropy), SuperRL, SASR, DyME, GAC. None is a per-prompt bandit; none addresses controller credit | AMFT [2508.06944](https://arxiv.org/abs/2508.06944) Table 1 |

**Recommendation from the search:** frame the identity as a *lemma* — "a concrete instantiation of the zero-learnability regime for a linear meta-controller in LLM RL post-training" — cite Graves §3.3 / L2T / AutoLoss for the design contrast, and cite SPO and PAC for the fix. Claiming a novel theorem invites an easy rebuttal.

### 4.2 The pass-count collapse — published repeatedly, once as an entire paper's thesis

| result | where it is already in print |
|---|---|
| GRPO / Dr.GRPO / DAPO are three operations on one number `σ = √(k(G−k))/G`; DAPO's dynamic sampling "is simply a keep rule on the same scalar: retain the group when σ>0" | **Bay & Yearick, [2607.00152](https://arxiv.org/abs/2607.00152), 30 Jun 2026 — Theorem 1** |
| reward entropy `H(p)`, group-filter survival `S_N(p)`, RLOO advantage energy `k(N−k)/(N−1)²`, pair count `k(N−k)` are four closed forms in the one scalar `p`, all peaking at 0.5 — "all identify the same target" | Rollout Pass-Rate Control (Baidu), [2605.05112](https://arxiv.org/abs/2605.05112), Eq. 1 |
| the binary-reward family is SGA on `E_x[h(p_θ(C|x))]`; the per-prompt update depends on the prompt only through `p` | Davis & Recht, [2510.13651](https://arxiv.org/abs/2510.13651) |
| `Σ|A_i| = 2G√(p(1−p))`; `A⁺ ∝ √((1−p)/p)`; gradient variance ∝ `p(1−p)` | [2602.05548](https://arxiv.org/abs/2602.05548), [2605.27765](https://arxiv.org/abs/2605.27765), [2602.01601](https://arxiv.org/abs/2602.01601) |
| **the RL/SFT/SKIP solve-rate rule our policy collapsed to is itself a published method** | **DyME** [2506.23061](https://arxiv.org/abs/2506.23061): `mode(x) = GRPO if max_k r(ỹ_k)=1 else SFT` — ablated against reward thresholding, SFT annealing, SFT budget, and claimed "parameter-free, efficient, and empirically optimal." Plus **DAPO** [2503.14476](https://arxiv.org/abs/2503.14476): `0 < |correct| < G` for skip. The three-way combination is unclaimed; both halves are not |
| successors, all pass-rate predicates | GRESO, DA3PO [2608.27982] (`λÂ if R>0 and p̂<0.5`), Skywork-OR1, Online Difficulty Filtering [2504.03380] (EACL 2026 — improvement lower-bounded by variance of task success probs), Prompt Replay, Prompt Curriculum Learning [2510.01135] (ICLR 2026), CurES, MoPPS |

So the memory-file theorem "every estimator = α(k)·(r−r̄)" is confirmed and already published — it is exactly 2607.00152. **This kills the "pass-count bottleneck" reframing as a headline.** It survives only as the *setup* for §4.3.

### 4.3 The one thing NOT found — and the two papers that claim the opposite

**"Length / truncation / token-level entropy carry no routing information beyond p" — NOTHING FOUND either way (8 queries).** No theorem forces covariates to collapse (the memory-file table already notes GraphRPO *escapes* because its baseline conditions on length). So our 102/102 behavioural-equivalence result is potentially a **novel negative result** — but two papers actively claim the converse and must be addressed head-on:

- **RL-ZVP / No Prompt Left Behind** [2509.21880](https://arxiv.org/abs/2509.21880) (ICLR camera-ready Feb 2026) — uses **token-level** entropy to shape advantages on zero-variance prompts (p∈{0,1}), where a solve-rate predicate has zero resolution: +8.61 over GRPO. The distinction that matters: the entropy of the *binary reward distribution* is `H(p)` and collapses; **token/policy entropy does not.**
- **HIVE / Train at the Moving Edge** [2603.25184](https://arxiv.org/abs/2603.25184) — a routing paper fusing pass-rate history with prompt entropy, with the ablation (Fig. 6d) that removing either stage worsens the accuracy–rollout tradeoff. The most direct challenge to a 102/102 equivalence claim.
- Beyond Variance [2602.03452] attacks the *optimality* of the p-threshold, not its sufficiency.

**Obvious reviewer objection to our 102/102:** the 102 contexts may not span the p∈{0,1} degenerate regime where both papers locate their gains. **Before claiming anything, verify which of the 7 observability features are pure functions of the reward vector and which are covariates.**

### 4.4 Self-selection drift masquerading as learning — it has a theorem and a name

**"Incomplete learning."** Kalvit & Zeevi, NeurIPS 2021, [2106.02126](https://arxiv.org/abs/2106.02126), Theorem 2: two arms both Bernoulli(q); under Thompson sampling with q=1, `N₁(n)/n ⇒ Uniform[0,1]` — the allocation converges to a **non-degenerate random variable**, not to ½. Their words: sample paths "evolve in such a way that the algorithm is 'deceived' into incorrectly believing one of the arms to be inferior … a perpetual 'imbalance' in the sample-counts." Consequence for our metric: `L1 = 2|N₁/n − ½|` has limiting mean ½ from near 0 at small n — **it rises monotonically to a positive constant under an exact null.** Generalised by Han [2601.21131](https://arxiv.org/abs/2601.21131): pull counts are deterministic iff the arm is suboptimal or the unique optimum; under a null (all arms optimal) allocation is always a random limit. Stable TS [2505.23260] (COLT 2026), Optimism-stabilised TS [2602.06014], null-hypothesis BRAR [2510.01734] are the fixes; Chen et al. [2205.03820] is the plain-language warning: TS "tends to 'select' an arm … even when there is no treatment difference … this can falsely promote the idea that the experimental arm is better."

**Undocumented in our exact framing (~20 queries):** nobody uses L1-from-uniform of a *training-mode controller's* mix as a learning metric, shows it rising on a provably-non-learning controller, or names a null-controller baseline as the fix. That transfer — plus the subset-contrast + shuffle-control replacement — is the reusable methodological piece. The correct diagnostics per the literature: Allocation Probability Test [2111.00137], randomization tests for adaptively collected data [2301.05365].

### 4.5 Coverage

Semantic Scholar API returned 429 both attempts (reached only via web search). OpenReview full text not searchable; ICLR 2027 not public. Empty query lists are in the subagent transcript.

---

## 5. Judgement

### 5.1 Answering the three questions asked

**Q1 — Is "learned routing beats fixed-rule routing" strong enough? No, not as framed, and the reframing I floated is dead too.**
The framing is factually wrong for two of three comparisons (frozen LLM, not fixed rule); the idea is publicly staked (SIA §9, 2607.01120); the fixed-rule baseline we collapsed to is a *published method* (DyME + DAPO); HASE already learns what-to-evolve end-to-end; and our own GPU measurement (Result 7) is a null at matched proportions. The "pass-count bottleneck" reframing — every routing rule over outcome statistics is a partition of {0..G} — is a true and useful setup, but 2607.00152, 2605.05112 and Davis–Recht already own the theorem. **Making the confound the subject is right; making the theorem the headline is not available.**

**Q2 — Harness or model axis? Model axis, decisively.** The harness space has ≥9 papers since April, a published null at matched budget from AI2 (with Co-Harness co-authors), Lilian Weng's public skepticism of SIA, and Evo-Bench's early-saturation result. Our dispatcher varies a generation-length ladder, which no reviewer will accept as "harness evolution." Use truncation→length **only** as one covariate in the test below; do not headline it.

**Q3 — Is the sharpest claim already measured? Partly — but the parts that are measured are the parts that are already published.** The identity is a Wolpert–Tumer corollary; the fix is SPO/PAC; the collapse is 2607.00152; the drift is Kalvit–Zeevi. What is *not* published, and what the project is uniquely positioned to measure, is the question all of those results leave open:

> **Under verifiable binary rewards, can any covariate — length, truncation, token-level entropy, representation-space clusters — carry routing information beyond the pass count, at matched budget?**

Nothing found either way. Two papers claim yes for specific cases (RL-ZVP on p∈{0,1} via token entropy; HIVE via pass-rate history + entropy). We have the instrument nobody else has built: a learned per-group router over {target}×{loss}, rate-matched and permutation-controlled, with per-prompt credit so it *can* learn, and the subset-contrast metric so we know whether it *did*.

### 5.2 The strongest available paper

**A measurement paper whose instrument is the learned router, framed as the escape route from a published bottleneck.** Structure:

1. **Setup (cite, don't claim):** under binary rewards every outcome statistic is a function of `k` (2607.00152, 2605.05112, Davis–Recht). *Extend it one step to routing*: therefore every routing rule over outcome statistics — DAPO, DyME, HPT, SRPO, RSTG, LSPO, GRESO — is a partition of {0..G}, and a learned router over outcome features can at best learn that partition. Our 102/102 collapse is the empirical instance. This extension is small but, as far as three searches found, unstated.
2. **Why learned routers have been failing (lemma + transfer):** batch-scalar credit is the zero-learnability regime (Wolpert–Tumer); L1-from-uniform rising under a null is incomplete learning (Kalvit–Zeevi). Both stated for training-mode controllers for the first time, with the subset-contrast + shuffle-control + null-controller protocol as the reusable fix. Cite SPO/PAC for per-prompt credit; do not claim it.
3. **The experiment (the paper's content):** covariate-conditioned learned router vs the k-partition (DyME+DAPO — which is also our own rule policy), vs SRPO, at matched rollout and feedback budget, on OlympiadBench and LiveCodeBench at 32B, held-out split, with a repeated-sampling TTS baseline per AI2. Each covariate tested with its rate-matched control: truncation (via the length ladder — the *only* legitimate use of the harness machinery), token-level entropy (RL-ZVP's claim), MEDS clusters (the feature GOAL.md already wants). Report positive or negative with the error bar on the difference.
4. **Framing chosen by the result, honestly:** if a covariate carries signal → "the first learned router that beats the pass-count partition, and why prior ones couldn't"; if none does → "a bound: at 32B on frontier math and code, adaptive routing collapses to the pass count, and here is the protocol that shows it." The 2026 venue record accepts both genres (2603.19335, 2604.23747, 2605.30621, 2607.12227 are all sobering-measurement papers at strong groups).

**What this is not:** it is not "beats Ornith-1.5 on Terminal-Bench." That was never reachable with 8×A100 + 4×H100 for two days. It *can* be "beats DyME/SRPO/HPT at matched budget on OlympiadBench+LCB at 32B" — a real SOTA claim in the adaptive-post-training niche — if and only if a covariate carries signal.

### 5.3 What must be true for a top venue, and what would change my mind

Necessary, in order of how fast we learn the answer:
- **Feature audit (hours, CPU):** classify the 7 observability features into pure-`k` vs covariate. If all 7 are functions of the reward vector, the 102/102 is a corollary of 2607.00152 and there is no escape route in the current feature set — MEDS and token entropy must be added before anything runs.
- **Truncation covariate arm (H100, in progress):** treatment vs rate-matched control. A clean positive at matched budget is the single result that would move this from bound-paper to method-paper. A null is informative and publishable but drops the venue ceiling.
- **Loss-weighting audit against 2604.23747** before any SFT/RL-mixing number is reported. E1's bit-identical rollback proves vanilla is reachable, not that mixed weights are correct.
- **Matched inference and feedback budgets** (GOAL.md §4: both NOT MET). AI2 will reject on exactly these rows.
- **Baselines that must be run:** DyME rule, SRPO rule, HPT threshold, LSPO for the k=0 branch, repeated-sampling TTS at matched budget.

What would change my mind toward the original framing: a covariate-router win over SRPO *and* DyME at matched budget on both benchmarks with a held-out split. What would change it toward "not a top venue in any framing": the feature audit showing all 7 are `k`-functions *and* the truncation arm null *and* token-entropy null — then the honest paper is a short bound with the protocol, aimed one tier down.

### 5.4 Corrections to make regardless

- GOAL.md 2b: done today (`3ccc18e9`). M8 row: cite Wolpert–Tumer, SPO, PAC, Kalvit–Zeevi; demote "identity" to "lemma."
- The memory-file theorem `finding_binary_reward_collapse` should cite 2607.00152 as its published form.
- The k=0 gold arm: do not build; run LSPO as a baseline instead.
- Stop describing the harness ladder as harness evolution anywhere.

---

## 6. The PI's extended proposal, assessed

Proposed 2026-09-01: frozen LLM as router cold start, then a LoRA on the router trained by RL;
per-task LoRA on the policy and, within a task, per-MEDS-cluster LoRA; distillation as a
routable source with every loss choice; per-task harness co-evolution cold-started from each
task's best known harness, down to per-sample-group.

### 6.1 As a whole system — structural objection, not taste

Every axis routes on the same information. Under binary rewards that is `k` plus whatever
covariates carry signal (§4.3). Actuators — LoRA experts, harnesses, distillation sources —
add places to route TO; none adds information about WHERE. If covariates carry nothing beyond
`k`, every actuator is routed by `k` and the system reduces to DyME + DAPO + LSPO with more
parts. The feature audit (§5.3) remains the gate for the whole proposal.

Credit multiplies: four simultaneous decisions per group against one per-prompt credit scalar
per step is a harder attribution problem than the single-decision one just solved, and the
meta-controller literature solves even that only by many independent episodes (L2T, AutoLoss;
§4.1), infeasible when an episode is a training run.

Venue: it would be the fifth system paper in the cell (SIA, EvoTrainer, 2607.01120, HASE). A
system with more axes wins only by beating them at matched budget, which the compute window
cannot run.

### 6.2 Two pieces worth extracting — each is a covariate test WITH a mechanism

**(a) MEDS-cluster-routed LoRA experts.** Parameter-space routing by discovered behavioural
cluster, generalising LSPO's two-expert split (`k=0` → adapter, else → base) to N clusters.
Hypothesis with a measurable mechanism: subpopulations need conflicting updates and a shared
adapter averages them — **gradient interference**. Risks: MEDS clusters RESPONSES, so
inference-time routing needs cluster-from-prompt (circular); LSPO avoids this by merging at
inference, making the benefit training-time only. Cluster-then-merge is adjacent to task
arithmetic (TIES, DARE); the novelty is "discovered clusters during RL", not the merge.
Novelty search launched (results appended as §6.4).

**(b) Frozen-LLM router, LoRA-trained via per-prompt credit.** SIA §9's future work, executed.
Its value is being the richest covariate extractor — it reads trajectory text, exactly what
escapes the pass-count collapse. Costs an LLM call per group per batch; must clear the noise
floor against a free threshold (GOAL.md's own scepticism). Credit must be cited as SPO/PAC.

Not contributions: per-task LoRA at task granularity (standard multi-task adaptation);
per-sample-group harness (Adaptive Auto-Harness 2606.01770 does per-input harness routing, and
the AI2 null applies).

### 6.3 The interference measurement — ready to run when a GPU frees, no new training needed

On an existing checkpoint (any lora30b or smoke32b ckpt), one rollout batch, per-group GRPO
loss:
1. Cluster groups two ways: MEDS (HDBSCAN over latter-half layer logits at the answer token)
   and a **random partition with the same cluster sizes** (the control — cluster labels drawn
   feature-blind).
2. For each cluster `c`, compute the LoRA-parameter gradient `g_c` of the summed GRPO loss over
   its groups.
3. Report pairwise `cos(g_c, g_c')`, the fraction of pairs with negative cosine (conflict
   rate), and the norm ratio `||Σ_c g_c|| / Σ_c ||g_c||` (cancellation). Interference is
   real only if MEDS clusters show lower cosine / higher cancellation than the size-matched
   random partition — otherwise the "clusters" are noise and per-cluster adapters would only
   add parameters.
4. Repeat on 3 checkpoints (early / mid / late) since interference plausibly grows as the
   policy specialises.

Cost: forward+backward over one batch per checkpoint, ~minutes on one H100. This is the
cheapest possible test of whether (a) has a mechanism, and it is measurable before any
per-cluster adapter is built. If the size-matched random control shows the same conflict, do
not build (a).

### 6.4 Pending: cluster-routed-LoRA novelty search
