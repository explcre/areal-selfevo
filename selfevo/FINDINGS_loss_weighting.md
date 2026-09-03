# How this repo weights SFT-mode rows against RL-mode rows inside one GRPO loss

2026-09-01. CPU only, read-only audit; no source file was modified. Written here rather than
into `EXPERIMENTS.md` or `GOAL.md` because both were held by other agents for the whole of
this session; fold it in from here. Everything below is pinned by
`selfevo/tests/test_loss_weighting_audit.py` (15 tests), which drives the real
`grpo_loss_fn` rather than restating its arithmetic.

The question comes from arXiv 2604.23747, which attributes several reported mixed-policy
gains to loss-weighting bugs: an SFT term normalised per sequence sharing a step with an RL
term normalised per token, so a length difference between the two silently rescaled one
against the other. Arms **A4** (LSPO-style gold-SFT rows) and **A5** (DyME-style gold
substitution) put SFT-like rows and RL rows in the same microbatch, so the weighting has to
be a checked fact before either arm's number is reported.

## 1. The normalisation: one token mean over the whole global batch

There is exactly one reduction, and it is per TOKEN.

| where | line | what it does |
| --- | --- | --- |
| `areal/utils/functional/functional.py` | **506** | `loss_mask_count = loss_mask.count_nonzero() or 1` — the denominator, fixed before rejection sampling can narrow the mask |
| `areal/utils/functional/functional.py` | **571** | `pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count` — sum over tokens, divide by token count |
| `areal/trainer/ppo/actor.py` | **1230** | `loss_weight_fn=lambda x: x["loss_mask"].count_nonzero()` — a microbatch's weight is its token count |
| `areal/engine/core/train_engine.py` | **60** | `total_weight = sum(loss_weight_fn(mb) for mb in mbs)`, all-reduced over DP |
| `areal/engine/fsdp_engine.py` | **2216-2217** | `loss_scale = local_weight / total_loss_weight`; the microbatch loss is multiplied by it |

The composition is `sum_over_all_tokens(surrogate) / total_tokens`. The per-microbatch
division at 571 and the per-microbatch rescale at 2216 cancel exactly, so the global objective
is a single token mean and the per-microbatch measurement in the test is the whole story.
CISPO (629/660) and SAPO (744/775) use the identical denominator, so the answer does not
change with the surrogate. There is **no** `token_mean`/`seq_mean` switch, no per-sequence
reduction and no per-group reduction anywhere on this path. `mb_spec` is
`MicroBatchSpec(n_mbs=ppo_n_minibatches)` (actor.py:1201-1204) and affects only how the sum is
partitioned. `importance_sampling_level='sequence'` (GSPO) changes the *ratio* and replaces the
advantages with their per-sequence means (functional.py:56-77) but leaves the reduction at 571
untouched — it is not a per-sequence normalisation of the loss.

### Both modes go through that one reduction

`selfevo/integration/group_apply.py` does not touch the loss. Its seam is
`_APPLIED = (RL, SFT, SKIP)` (line 46) and all three modes are expressed **as values in the
advantage tensor** (`apply_decisions`, lines 237-272): RL leaves `advantages` alone, SFT
*replaces* the response-token entries with the constant `sft_weight`, SKIP writes zero. The
fixed-rule path in the actor does the same thing by addition instead of replacement
(actor.py:966-989), which coincides on a solved group because its advantages are identically
zero. Either way, by the time `grpo_loss_fn` runs there is no mode label left — only numbers
in `input_data["advantages"]`.

So the M19 self-target becomes a gradient by the same route an RL advantage does. At the live
configuration the ratio is exactly 1 (`ppo_n_minibatches=1` with `recompute_logprob`, measured
`importance_weight` avg=min=max=1.0 and `clip_ratio` 0.0 over four runs, recorded at
actor.py:902-912), so for both modes

    dL/dlogprob_t  =  -A_t / N       with N = the microbatch's masked token count

and a group's contribution is `|A| x (its token count) / N`. That is an unclipped REINFORCE
step for the SFT group; nothing in the clip bounds it, because a ratio of 1 is interior.

### Why `adv_norm` has to be `None` (GOAL.md 3, residual 2.139 vs 0.0)

`Normalization._compute_mean` (`areal/utils/data.py:1686-1688`) forms
`(x * mask).sum() / mask.sum()`. That is a **token-weighted** mean. With per-row-constant
advantages `a_i` spread over `L_i` response tokens it equals `sum_i L_i a_i / sum_i L_i`,
which coincides with the per-ROW mean `sum_i a_i / G` only when every `L_i` is equal.
`reward_norm` has already centred the rewards per group, so the advantages arrive with
`sum_i a_i = 0` per group; subtracting a token-weighted mean destroys that, and generation
lengths are never equal. `mean_level='group'` (data.py:1587-1610) does it inside each group,
`mean_level='batch'` (1577-1586) does it across the batch — hence the measured residuals 0.0 /
2.139 / 0.867 recorded at `areal/api/cli_args.py:2129-2136`. The test
`test_group_level_adv_norm_is_token_weighted_and_breaks_the_zero_sum` reproduces both halves:
unequal lengths leave a residual, equal lengths leave none.

## 2. The key question (item 3): YES, length changes the relative contribution

**Measured**, two-group packed microbatch, SFT constant `c = 0.5` on two rows, RL advantages
`+/-1.0` on two rows of 4 tokens each, gradient magnitude read off `logprobs`:

| SFT row length | `|g_sft|` | `|g_rl|` | ratio | token-mean prediction | sequence-mean prediction |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.250000 | 0.500000 | **0.5000** | 0.5000 | 0.5 |
| 8 | 0.333333 | 0.333333 | **1.0000** | 1.0000 | 0.5 |
| 16 | 0.400000 | 0.200000 | **2.0000** | 2.0000 | 0.5 |

Doubling only the SFT rows' length doubles the SFT group's share of the update, exactly. A
sequence-averaged loss would have returned 0.5 three times — confirmed by mutation: replacing
`ppo_actor_loss_fn` with a per-sequence-mean variant makes the ratio 0.5000 at all three
lengths and kills both normalisation tests. The effect is per TOKEN, not per row: 2x8 SFT
tokens and 1x16 SFT tokens weigh identically at fixed batch width. Prompt length is neutral,
because the denominator counts masked tokens only.

The seam's own docstring already prices this at
`selfevo/integration/group_apply.py:143-149` — "a token-mean loss gives a row at the
generation cap ~2.5x the gradient mass of a terminating one" — which is why
`exclude_truncated_from_sft` exists. That is the same mechanism as the 2604.23747 one, seen
from inside a single arm.

## 3. Item 4: the importance ratio on a row the policy never sampled

The PPO ratio is `exp(logprobs - prox_logp)` (functional.py:538). It does **not** read
`input_data["logprobs"]`. That field (`old_logp`, actor.py:1300) is read only by rejection
sampling, by M2PO masking, and by the KD branch. Behaviour by path, all measured:

| what `substitute.py` puts in the field | result |
| --- | --- |
| NaN in `prox_logp` | **RuntimeError**, loud, `actor.py:1727-1731`. Also for +/-Inf. |
| NaN in `logprobs`, `rejection_sampling=None` | **Silently ignored.** Finite loss, gradient bit-identical to the clean batch. |
| NaN in `logprobs`, live `rejection_sampling` | **Silent ratio = 1.** `functional.py:233` rewrites every non-finite log-ratio to 0.0, so `behave_imp_weight = exp(0) = 1`, `filtered_fraction = 0.0`, nothing rejected. The gold row is scored as perfectly on-policy. |
| 0.0 placeholder, live `rejection_sampling` | **Silently down-weighted.** `behave_imp_weight = exp(prox_logp) < 1` multiplies the surrogate at functional.py:568; at `prox_logp = -1` the row keeps 0.368 of its weight and no metric reports it. |
| `prox_logp` key absent | ValueError, actor.py:1686-1691. |
| `logprobs` key absent | KeyError, actor.py:1300. |
| `prox_logp_method='reuse_train_logp'` | Ratio exactly 1 for every token (actor.py:1307-1308); the field is irrelevant. |

There is a worse path **upstream of the loss**, and it is the one that decides the answer.
`PPOActor._compute_advantages` reads `data["logprobs"]` for the KL reward at
**actor.py:741**, `rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)`. `kl_ctl`
is `0.0` in the live config and that does not save it: `-0.0 * NaN` is NaN. Measured on the
real `_compute_advantages`:

* `adv_norm=None`: the NaN reaches that row's advantages and stays there (1 of 8 rows).
* `adv_norm: mean_level=batch` — **the live setting**, `examples/math/gsm8k_grpo_lora.yaml:85-87` — the batch mean is NaN and **all 8 of 8 rows** come out NaN. One gold row destroys the update for every other row in the batch.

**Rule for `selfevo/gold/substitute.py`: write a FINITE `logprobs` for every gold row, and
make it equal to the trainer's own recomputed `prox_logp` for those tokens.** That is the
only value giving `behave_imp_weight = exp(0) = 1`, i.e. leaving the surrogate exactly as the
gold row's advantage intends. NaN is not a safe sentinel here (it is silently laundered into
ratio 1 by the filter, and it is fatal one stage earlier), and 0.0 is not either (it silently
shrinks the row). Note this is what `use_decoupled_loss=false, recompute_logprob=true` already
does for *all* rows at actor.py:725; the live config takes the other branch
(actor.py:727-730), so substitution has to supply the value itself.

## 4. Does the 2604.23747 bug class apply here?

**The specific bug does not; the effect it describes does, and it is not neutral for A4/A5.**
The paper's failure is *inconsistent* normalisation — two different denominators for the two
modes in one step. This repo cannot have that: there is a single reduction (functional.py:571)
and both modes reach it as plain numbers in one advantage tensor, so an SFT row and an RL row
of identical length and identical `|A|` receive identical weight, and the comparison A4/A5 vs
A0 is not corrupted by a normalisation mismatch. What *does* apply is the consequence the
paper's bug produced: because the shared normalisation is per token, **any systematic length
difference between the SFT-like rows and the RL rows is a silent reweighting of the two
modes**, measured above as exactly proportional. Gold solutions are typically shorter and far
lower-variance in length than sampled rollouts, and truncated rollouts sit at the generation
cap, so a length difference is the default expectation, not an edge case.

What A4 and A5 must therefore do to be comparable to A0, in order of preference:

1. **Report the token budget, not the row budget.** Log `sft_tokens / total_tokens` per step
   alongside `route/*_groups`. Two arms matched on gold-row COUNT can differ by 2-4x in gold
   token MASS, and none of the existing `route/*` keys would show it. Without this number the
   arms are not matched on anything the loss reads.
2. **Match on token mass, or reweight explicitly.** Either hold the gold-token fraction fixed
   across arms, or scale `sft_weight` per row by `mean_response_len / row_len` so a group's
   SFT mass is length-invariant. Prefer the explicit reweight to a per-sequence loss: changing
   the reduction at functional.py:571 would alter every A0 number too and orphan the run
   history.
3. **Keep `adv_norm=None` and `kl_ctl=0`.** Already required by GOAL.md 3 for the routing
   rule, and independently required here: batch-level `adv_norm` is what turns one bad gold
   row into a whole-batch NaN, and it is what breaks the per-group zero-sum the SFT constant's
   interpretation depends on.
4. **Set gold `logprobs` as in section 3.** A gold row whose `logprobs` are NaN or 0.0 does
   not fail — it reports a plausible number that is not the arm that was configured.

Doing nothing is defensible only if the gold rows are length-matched to the sampled ones, and
that is an empirical claim about the gold corpus which nobody has checked yet.

## Reproduce

    ~/venv312b/bin/python -m pytest selfevo/tests/test_loss_weighting_audit.py -q
