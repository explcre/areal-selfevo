# M25 experiment plan — MEDS-routed LoRA experts

The paper's method (GOAL.md M25). One spec so every arm is run the same way and no agent
improvises a control. Decision rules are written down BEFORE any number exists.

## Gate 0 — the interference probe (no training, minutes on one H100)

`selfevo/cluster_lora/interference_probe.py` on a 32B checkpoint + one saved rollout batch,
FOUR partitions on the same batch: (1) MEDS behavioural clusters, (2) size-matched random,
(3) ELREA-style prompt-gradient clusters, (4) task-label calibration if the batch spans tasks.
Per partition: pairwise cosine of per-cluster LoRA gradients, conflict rate, cancellation
`||Σg_c||/Σ||g_c||`, sizes, bootstrap CI on mean cosine. Three checkpoints (early/mid/late) if
available, since interference plausibly grows as the policy specialises.

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
