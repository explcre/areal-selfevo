# Prior art and the evaluation bar, after reading five papers (2026-08-30)

Two of these change what we may claim. One sets a bar our current design does not clear.
Two are interesting but do not bear on the method, and are recorded as such rather than
cited to look thorough.

---

## 1. Rethinking the Evaluation of Harness Evolution for Agents (arXiv 2607.12227)

Wang, Zhu, Hu, Yuan, Chen, Senthil, Hajishirzi, Tsvetkov, Dasigi, Xiao. AI2 / UW.

**This is the most consequential paper for us, and it is a threat before it is an
opportunity.** Its claim is that reported gains from automatic harness evolution may come
from *search*, not from better harness design, because harness evolution is itself an
iterative search that repeatedly evaluates candidates against task feedback. It demands two
controls:

1. **Matched-budget baselines.** Harness evolution must be compared against simple
   test-time scaling and discovery baselines *under comparable feedback and inference
   budgets*. An evolution loop that queries the task 200 times must be compared against a
   baseline allowed 200 queries, not against one-shot.
2. **A held-out task split.** Search and final evaluation must not share a benchmark, or
   the reported gain is overfitting to that task set.

On Terminal-Bench 2.1 with GPT-5.4 and Claude Opus 4.6 they report that automatic harness
evolution **does not consistently beat simple test-time scaling** and generalises poorly.

### What this costs us, stated plainly

Our framework has `evolve_target` as an axis, with harness evolution as one setting. **We
currently have no matched-budget baseline and no held-out task split for any evolution
claim.** Under this paper's protocol our harness-evolution arm is unevaluated, not
promising. Any result we produce for it before adding those two controls is not publishable
and, worse, would probably be wrong in the direction we would like it to be right.

Our own simulation already pointed here: §12 records that routing against the criterion is
worse than its own matched control, and that an always-SFT baseline attains 0.6146 against
the criterion's 0.6127. We found the same shape by accident. This paper makes it a standard.

### The opportunity

The bar is stated precisely enough to clear deliberately. A paper that reports harness or
signal-routing gains **with** matched-budget baselines and a held-out split answers the
field's live objection instead of inviting it. That is worth more than a larger number
without the controls.

---

## 2. SIA: Self Improving AI with Harness & Weight Updates (arXiv 2605.27276)

Hexo Labs. Unifies the two schools we describe: a Feedback-Agent reads the trajectory and
decides between rewriting the harness, triggering a LoRA weight update, or both.

**This is direct prior art for the meta-controller, and it goes further than we assumed.**
SIA already selects *among training algorithms per task*: PPO with GAE where step-level
rewards are dense, GRPO where rollouts are cheap, entropic advantage weighting for
right-skewed rewards, REINFORCE with KL-to-base where capability regression is the risk,
best-of-N behavioural cloning as a cold start when E[r] ~= 0, and DPO where the verifier
ranks but cannot score.

So "a controller that picks the training algorithm" is published. We cannot claim it.

### What it leaves open, and this is our opening

- **Granularity.** SIA selects per *task*. Our claim is per *sample* and per *token*, which
  is a different object: it needs a validity condition for when a unit may receive which
  signal, not a heuristic for which optimiser suits a domain.
- **Learned vs prompted.** SIA's selection is an LLM agent reasoning from rules of thumb.
  Ours is a policy fit to outcomes, which is testable against the rules-of-thumb baseline
  and, per AI2 above, must be.
- **Their own deferral.** The paper states a fuller treatment of algorithm selection across
  tasks is deferred to a later version. That is the gap, named by the authors.
- **Their stated failure mode is the one we designed against.** SIA reports "coupled
  co-evolutionary Goodhart": harness search and weight updates both optimise against the
  same fixed verifier, reaching a Nash equilibrium between two optimisers blind to each
  other. Our `cadence` axis (frozen / alternating with `critic_update_every` > 1 /
  simultaneous) and `frozen_eval_reward` exist precisely to break that coupling. We can run
  their failure mode as an ablation arm rather than only citing it.

### Numbers: handle with care

Reported figures differ between versions (v1 abstract vs v2 tables) -- e.g. LawBench gains
quoted as 25.1% in one place and a 13.5 -> 70.1 span in another. Cite the version explicitly
or cite the structure, not the number.

---

## 3. Continual Harness: Online Adaptation for Self-Improving Foundation Agents (arXiv 2605.09998)

Karten, Zhang et al. Princeton / ARISE / DeepMind. Reset-free online harness refinement on
Pokemon Red and Emerald; the agent alternates acting and refining its own prompt,
sub-agents, skills and memory.

Relevant to us for one mechanism: **an online process-reward co-learning loop in which an
open-source agent's rollouts through the refining harness are relabelled by a frontier
teacher, and used to update the model.** That is teacher-relabelled on-policy distillation
inside an evolving harness -- the same shape as our teacher-based routing modes, in a domain
where resets are impossible. Worth citing as the closest precedent for the teacher arm, and
worth noting that their setting has no held-out split either.

---

## 4. Verbalizable Representations Form a Global Workspace in Language Models (Anthropic, 2026-07-06)

The J-space / Jacobian-lens work on Claude Opus 4.6. A privileged internal space holding
concepts the model is "thinking about" without emitting them.

**Recorded as not load-bearing.** It is an interpretability result about a specific
proprietary model obtained with a tool we cannot run on our checkpoints. It is tempting to
invoke it as motivation for "trajectory observability metrics as evolve-policy inputs", but
we would be borrowing authority for a claim it does not make about models it did not study.
If we want observability features in the evolve-policy, we must justify them with features
we actually compute and ablate. Cite only if we build something that reads internal state.

---

## 5. Reasoning by Superposition (arXiv 2505.12514, NeurIPS 2025)

Proves a two-layer transformer with D steps of continuous CoT solves directed graph
reachability (D = diameter), against O(n^2) decoding steps for constant-depth transformers
with discrete CoT, because a continuous thought vector is a superposition encoding multiple
search frontiers -- parallel BFS -- which emerges without explicit supervision.

**Relevant only if we do continuous CoT.** It is the strongest existing argument that the
*token type* changes what a training signal can express, which is adjacent to our
token-level routing claim. But our token-level work is over discrete tokens with per-token
weights, so this is background, not support. Cite as motivation for a future continuous-CoT
routing mode; do not present it as evidence for the current one.

---

## The bar our experiments must clear, as a checklist

Derived from 2607.12227 and applied to every evolution claim we make, not only harness ones.

| Control | Status | Note |
|---|---|---|
| Matched inference budget vs baseline | **NOT MET** | No budget accounting exists in our harness |
| Matched feedback budget (task queries) | **NOT MET** | Evolution loop query count is not even logged |
| Held-out task split (search != eval) | **NOT MET** | Routing sim and MATH-500 both score the search set |
| Simple test-time-scaling baseline | **NOT MET** | Never run |
| Fixed-mode baseline (always-SFT / always-RL) | MET (sim only) | §12: always-SFT 0.6146 vs criterion 0.6127 |
| Matched permutation control | MET | `MatchedPermutationControl` in routing/proportions.py |
| Held-out benchmark for capability | MET | MATH-500, and it overturned the train-reward reading |
| Frozen evaluator not co-optimised | Designed, unused | `frozen_eval_reward` axis exists; no run uses it |
| Multiple seeds / replicate | **NOT MET** | No replicate of any measurement exists |

Four of nine are unmet and two more are met only in simulation. **The honest position is
that we have one solid held-out capability measurement and no evaluated evolution claim at
all.** That is a smaller claim than we have been implicitly making, and it is the one the
evidence supports.

## What changes, concretely

1. **Log budgets.** Every evolution run records task-feedback queries and inference tokens,
   so a matched baseline is constructible after the fact rather than never.
2. **Split the benchmark before any evolution run.** Search on one half, report on the
   other. For MATH-500 this means a fixed, committed problem-id split, not a re-draw.
3. **Add the test-time-scaling baseline** (best-of-N / self-consistency at the same budget)
   as a first-class arm, not a footnote.
4. **Run SIA's failure mode deliberately** as the `simultaneous` cadence arm against our
   `alternating` and `frozen` arms. Their limitation becomes our ablation.
5. **Do not claim the meta-controller as novel at task granularity.** Claim it at
   sample/token granularity, with a validity condition, measured against a task-level
   controller of SIA's shape as the baseline to beat.
