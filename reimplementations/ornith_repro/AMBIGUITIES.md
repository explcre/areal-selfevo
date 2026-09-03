# Ornith-1.5 loop: what the release states, what it does not, and what we chose

This file is the contract for `ornith_repro/`. Every row is either (a) quoted structure from
the public release, or (b) a choice we made because the release does not determine it.
A reader judging our numbers needs column 3, so it is written before the code, not after.

## 0. Status of this directory: REIMPLEMENTATION, not reproduction

A faithful reproduction of Ornith-1.5 is **impossible from the public release** and we do not
claim one. Established by an earlier source-level audit of the release (2026-08-19,
`self-evo-agent/codex_docs/19-ornith-diagnosis--{ornith_diag.tex,notes.md}`), which inspected
the two vendor blogs, all 18 Hugging Face repositories, and the complete ten-commit history of
`ornith-ai/Ornith-1`:

* `ornith-ai/Ornith-1` contains a README, a LICENSE, a .gitignore and four images. **No source
  file, configuration, log, or training loop exists in any commit.**
* There is no arXiv paper and no first-party technical report. The method exists only as two
  vendor blog posts.
* No task buffer, per-task or per-candidate outcome record, harness, rollout trace, reward,
  advantage, ablation, raw benchmark output, compute ledger, or trainable config was released.
* The released README itself documents **unpublished modifications to its serving stack and
  chat template**, and two of its benchmarks are judged by a **proprietary model**.
* Released weights are 9B dense, 35B-A3B MoE and 397B MoE. **There is no 32B Ornith model**, so
  exact scale matching was never available.

What follows is therefore a reimplementation of the *described method*. Where our numbers differ
from Ornith's, that is not evidence about Ornith: it is evidence about the described method as we
were able to reconstruct it.

## 1. The published equations

Taken verbatim from the release blog as transcribed in the 2026-08-19 audit:

```
R_task    = V(q,s) * D(q,s,{tau_i}) * N(q)
D         = exp( -(p_hat - p*)^2 / (2 sigma^2) ),  p_hat = (1/k) sum_i 1{s(q,tau_i)=success}
p*        = 0.2
N(q)      = 1 - max_{q_j in B} sim(q, q_j)
R_harness = C(q,h) * F(h,{tau_i}) * H(h)
R_rollout = h(q, tau_i)
```

The blog states that question generation, harness generation and solution rollouts are each
optimised with GRPO, and that rollout reward propagates through the stages. It states
`V in [0,1]` and that validity "can also be treated" as a hard gate. It calls `B` a buffer of
"previously generated or trained-on tasks". It uses `s` as the success evaluator in `D` and
introduces `h` as the harness in `R_rollout`, and **never resolves whether `s = h`**.

## 2. Undisclosed quantities, and our choice for each

These are the parameters and definitions the release does not supply. `A#` ids are cited from
code and from `paper_src/results.tex`.

| id | Undisclosed | Our choice | Why, and what it costs |
|----|-------------|-----------|------------------------|
| A1 | rollout count `k` per task | `k = 8`, configurable | Matches our own GRPO group size. Ornith never states `k`. **Consequence, from the audit:** at `theta = p* = 0.2` the standard error of `p_hat` is `sqrt(.16/8) = .1414`, which is *wider than any sigma we would plausibly pick*, so `D` is partly selecting on sampling noise. We do not hide this; we measure it (`experiments/gate_selection.py`). |
| A2 | kernel width `sigma` | `sigma = 0.15`, configurable | No value is published. We swept it rather than tuning it, and report the sweep. `sigma = 0.1` at `k = 8` makes the gate narrower than its own noise. |
| A3 | GRPO group size `G` | `G = 8` | Our house default; lets the degeneracy prediction `theta^G + (1-theta)^G` be checked directly. |
| A4 | whether `V` was binary (hard gate) or continuous | **both implemented**, `validity_mode in {"gate","soft"}`, default `"soft"` | The blog says `V in [0,1]` and that it *can* be a gate. Asserting the gate as a release fact would be wrong. The choice changes the degenerate-group rate, so it is a config axis, not a constant. |
| A5 | whether `s = h` (is the success test the generated harness?) | `s_is_h` config, default `True` | The natural reading and the one that makes the loop closed. Under `True` the loop has **no external correctness signal at all**, which is the property our anchored-reward contribution exists to fix. Under `False` a separate `s` must be supplied. |
| A6 | `sim` function and encoder for `N` | token-level Jaccard by default; pluggable `SimilarityFn` | No encoder is named. Cosine on an unnamed embedder would be a *fabricated* detail. Jaccard is stated, cheap, and bounded in `[0,1]` so `N in [0,1]`. Note cosine would permit `N in [0,2]` and an unnormalised dot product would permit `N < 0`. |
| A7 | `max` over an **empty** buffer | `N = 1.0` (maximally novel), and the event is recorded | Undefined in the source. Returning `0.0` would zero `R_task` for the very first task and silently kill the first group: a silent-zero path. |
| A8 | buffer retention / capacity / insertion order | append-only, unbounded, insert **after** scoring | The blog says "previously generated or trained-on tasks", which does *not* imply append-only. Inserting before scoring would make every task its own nearest neighbour and force `N = 0`: another silent-zero path. Guarded (`G5`). |
| A9 | operational definition of `V, C, F, H` | judge-model rubrics, prompts in `judges.py`, each returning `[0,1]` | The release gives no definition or evaluator for any of the four. **This is the single largest reconstruction gap.** Any number we produce is conditional on our rubrics, not on Ornith's. |
| A10 | update order across the three stages | solver -> harness -> proposer, one step each per iteration, all on the *same* rollout batch | "Jointly optimised" does not specify an order. Ours is recorded and configurable (`stage_order`). Order matters because the stages share rollouts. |
| A11 | what is frozen | judges (`V,C,F,H`) are **frozen** and never trained | The release does not say. Training the judges on the same rollouts would close the last loop and make reward hacking unmeasurable even in principle (audit defect F). |
| A12 | GRPO variance convention and `epsilon` | `(R - mean) / (std + 1e-6)`, population std | Unstated. Matters only for degenerate groups, which we detect explicitly rather than letting `epsilon` decide. |
| A13 | rejection / resampling of degenerate groups | **none**; degenerate groups are counted and reported, not resampled | DAPO-style resampling would change the compute accounting and hide the defect we want to measure. |
| A14 | base model | **parameter**, not a constant: `base_model` in `OrnithConfig` | Ornith released 9B / 35B-A3B / 397B and no 32B. Exact matching was never possible. Target `Qwen/Qwen3.8-27B`; fallback `Qwen/Qwen3-32B`; last resort `Qwen2.5-32B-Instruct`. See section 4. |
| A15 | compute | not disclosed by Ornith; ours is reported in full | One deleted 9B revision contains `latest_checkpointed_iteration.txt = 1000` with no batch size, token count or hardware. That is not a compute disclosure. |
| A16 | whether sampling is NESTED (task -> several scaffolds -> several rollouts each) or FLAT | **nested** is treated as faithful; flat kept as an ablation (`NestedConfig.sampling`) | **The release figure and the release prose disagree.** The figure `ornith_self_improvement_loop` shows a task producing multiple scaffolds and each scaffold producing several rollouts. The prose says the policy produces "a solution rollout", singular, while `D` references a set `{tau_i}`. The figure is the more specific evidence so we follow it, but this is a genuine conflict in the source, not a gap. It matters more than any hyperparameter: GRPO's advantage is group-relative, so the group structure IS the algorithm. Nested creates TWO comparison levels (rollouts within a scaffold; scaffolds within a task); flat creates one. **Our first implementation was flat and additionally fed the rollout advantages to all three stages, so the scaffold and task stages formed no groups at all. That was wrong and is fixed in `nested.py`.** |
| A17 | scaffolds per task, and rollouts per scaffold | `n_scaffolds = 3`, `n_rollouts_per_scaffold = 8`, both named parameters | Neither is disclosed; the figure's counts are schematic. Arms must be matched on `total_rollouts = n_scaffolds * n_rollouts_per_scaffold`, not on step count, because nested at 3x8 spends 24 rollouts per task where flat at 1x8 spends 8. |
| A18 | how a scaffold's reward aggregates its rollouts | `F(h,{tau_i})` over that scaffold's OWN rollouts, per the published signature | This creates the difficulty gate's winner's curse **one level up**: a scaffold that drew lucky rollouts scores well for reasons unrelated to the scaffold. `run_iteration_nested(holdout_texts_by_scaffold=...)` supports the same fresh-block correction, and the fresh-block number is the one to report. |

## 3. Silent-zero paths we refuse by construction

Each has bitten this project before, so each is a guard with a test that makes it **fire**
(`tests/test_guards_fire.py`), not merely a guard that has never been seen to fail.

| id | Silent-zero path | Guard |
|----|------------------|-------|
| G1 | an **aborted** generation graded as a wrong answer, inflating difficulty | `RolloutOutcome` is a three-valued enum `{SUCCESS, FAILURE, ABORTED}`. `p_hat` is computed over non-aborted rollouts only, the abort rate is always recorded, and if fewer than `min_valid_rollouts` survive the task is **refused**, not scored `0`. Grading an abort as `FAILURE` requires setting `abort_policy="failure"` explicitly, which the guard logs as a known-bad setting. |
| G2 | a token cap **above the server's context**, producing refusals scored as failures | `assert_token_budget_fits` compares `prompt_tokens + max_new_tokens` against the *served* context length reported by the backend, and raises. Not against a hardcoded constant. |
| G3 | an **empty reward batch** scored as `0` | `assert_group_nonempty` raises on an empty or singleton group instead of returning a mean of `0`. |
| G4 | a **degenerate (constant-reward) group** silently contributing `0` and being averaged into "the gradient is small" | `grpo_advantages` returns advantages *and* a `degenerate` flag; `assert_batch_not_all_degenerate` raises when every group in a batch is degenerate, which is the false-negative that 90-of-128 unanimous groups produced here before. |
| G5 | the scored task already present in the buffer, forcing `N = 0` | `assert_task_not_in_buffer` raises if the task being scored is already a buffer member. |
| G6 | `N` computed against an empty buffer returning `0` instead of `1` | `novelty_reward` returns `1.0` and sets `empty_buffer=True` in the record; guarded and tested. |
| G7 | a stage that **did not run** but whose artifact still looks plausible | every stage record carries a `provenance` token derived from that stage's actual inputs and outputs; `verify_provenance` recomputes it on read-back. A defaulted or copied record fails. |

## 4. Base model

The base model is a **parameter** (`OrnithConfig.base_model`), never a constant, and every
artifact records the resolved model id and the backend's `/v1/models` reply, because an
unregistered model id serves the base model silently with a 200 and no warning.

* **Target:** `Qwen/Qwen3.8-27B`, present on the H100 box at `~/hf_cache/hub/models--Qwen--Qwen3.8-27B` (52 GB).
* **Open dependency, not assumed:** whether that model loads, serves and trains in this stack is
  being determined by another agent as of 2026-09-02. It is a hybrid-attention vision-language
  model rather than a drop-in text model, so the serving backend, the adapter target modules and
  the chat template are each places it may fail. **Nothing in this directory depends on that
  question resolving either way.** The CPU-side results below were produced with a deterministic
  stub client and are unaffected by it.
* **Fallback:** `Qwen/Qwen3-32B` (also already on the box).
* **Last resort:** `Qwen2.5-32B-Instruct`, which is what the rest of the project already uses and
  which would undercut the "strong current base, evolved further" story. If we land here, say so.

Whichever base actually ran is recorded in `run_meta.json` and must be stated in the write-up,
because that choice is part of the result.

## 5. Benchmarks: there is no head-to-head, and we must not imply one

Ornith reports SWE-bench Verified and Terminal-Bench 2.1. It reports **no competition
mathematics, no LiveCodeBench and no SQL**. Our project reports MATH-500, OlympiadBench and
LiveCodeBench. **The intersection is empty.**

Consequences, which any table we produce must carry:

1. There is **no benchmark on which a published Ornith number and a published Qwen3.8 number can
   be placed side by side.** Any comparison table is one *we* construct.
2. Ornith's 35B headline numbers (79.0 SWE-bench Verified, 68.5 Terminal-Bench 2.1 with Claude
   Code, 67.8 with Terminus-2) are **self-reported five-run averages with no per-run values, no
   intervals, and no per-instance records**, and two of its benchmarks are judged by a
   proprietary model.
3. Therefore we compare **our reimplementation against our own controls on our own benchmarks**,
   and we label every such table as ours. We do not print an Ornith number next to one of ours as
   though the two were measured under a shared protocol.

## 6. Controls (non-negotiable)

Every learned component in this project so far has tied with a size-matched random control at the
treatment's own measured proportions -- the learned router and the MEDS clustering both did, and
both were retired for it. The same control is pre-registered here on each generated axis:

* **Generated-task axis:** `random_task` control draws from a fixed task pool, size-matched, at
  the *measured* empirical proportions of the treatment's own accepted tasks over observable
  covariates (family, length bin, realised `p_hat` bin). Not at uniform, and not at the
  proportions we hoped for.
* **Generated-scaffold axis:** `random_scaffold` control draws scaffolds from a fixed pool at the
  treatment's measured scaffold covariate proportions.

If the loop ties with either control, that is the finding and it gets reported as the finding.
