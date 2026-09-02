# The gold batch-construction path: what was built, what it costs, and what it changes

2026-09-01. CPU only; no GPU job was started, stopped or touched. Written here rather than
into `EXPERIMENTS.md` or `GOAL.md` because both were held by other agents for the whole of
this session; fold it in from here.

## Why a batch operation and not an advantage one

`selfevo/tests/test_gold_target_reachability.py` established three things that together fix
the design. On MATH, 25.5% of all groups are UNSOLVED and carry no self-target. No router in
the registry can reach them, because every routing mode that needs a target gates on
`RoutingContext.has_target`, which an unsolved group cannot satisfy. And the advantage tensor
is the wrong altitude regardless: an advantage is a per-token coefficient on the
log-probability of a token the model actually emitted, so on an all-wrong group a positive
constant is a step toward the wrong answer. That is why `unsolved_advantage` is sign-guarded
to be `<= 0`.

So the gold has to enter as a ROW, and the ordinary estimator then acts on it — LUFFY's
construction (arXiv 2504.14945), DyME's (ICLR 2026), and what both mandatory baselines need
from this repo.

## 1. The dataset flag, and where the gold is tokenised

`areal/dataset/competition_math.py` grows `keep_solution: bool = False`, plus `gold_template`
and `append_eos`. It is wired through `dataset_kwargs`, which already reaches the adapter on
both the in-process path (`areal/dataset/__init__.py`) and the data-service path
(`areal/infra/data_service/worker/app.py:121`), so no config plumbing was added:

    train_dataset:
      path: DigitalLearningGmbH/MATH-lighteval
      type: rl
      dataset_kwargs: {keep_solution: true}

**Rollback is asserted, not assumed.** With the flag off, the adapter's output over all 7500
training rows hashes to `dbe5c602f6ca651b71ea8367e49d96ac8f9f103681ab16acfc9dc10dc2255033`,
which is the digest measured before the change. A column-name check would have passed on an
adapter that kept its columns and changed their contents, so the test hashes every
`messages`/`answer` pair instead.

**Tokenised at ADAPTATION, not at workflow time**, and the reason is measured. All 7500
solutions tokenise in 3.29s once with the live 30B tokenizer and are then cached by `datasets`
fingerprinting, so a resumed run pays nothing. The workflow alternative encodes per ROLLOUT:
at the live `gconfig.n_samples=8` over 10 epochs that is 80 encodes of the same string per
prompt, inside the async rollout loop of every rollout worker. Adaptation-time also makes the
length distribution knowable before a GPU is booked — median 163 tokens including the EOS,
max 2495, none empty, 99.13% at or under 1024.

`gold_template` is a parameter and not a constant because a gold row is spliced in after a
PROMPT, and the prompt's chat template decides what a valid continuation is. Measured on the
live model: the prompt ends at `<|im_start|>assistant\n<think>\n`, and the assistant turn that
follows is `\n</think>\n\n<answer><|im_end|>\n`. A gold that does not close that block first
would train the model to answer inside a thinking block it never leaves. There is no default
that is right for every template and guessing one silently is how a gold arm trains a shape
the model never emits, so a thinking model must set
`gold_template: "\n</think>\n\n{solution}"`.

`append_eos` defaults True because a rollout's `output_tokens` end with the stop token
(`ModelResponse.output_tokens_without_stop` strips it; `multi_turn.py:115` re-adds it when it
is absent), so a gold row without one would be the only row in the batch that never
terminates.

## 2. The collation trap, and why the padding is at construction

`check_trajectory_format` only WARNS when a tensor's second dim differs from `input_ids`',
and `concat_padded_tensors` pads dims 1..N-1 to the per-KEY maximum across trajectories. So an
unpadded gold does not fail at collation: it becomes `(B, max_gold_len)` beside an `(B, T)`
batch. `pack_tensor_dict` then packs only tensors whose `shape[1] == seq_len`, so the gold
stays 2-D and unpacked while every sibling becomes 1-D, and the break lands inside the engine
one stage later wearing a shape error that names neither gold nor collation.

`selfevo/gold/attach.py` therefore pads the gold to the TRAJECTORY's own width, at
construction. Both halves are tested: a gold-carrying batch survives
`concat_padded_tensors` -> `pack_tensor_dict` -> `split_padded_tensor_dict_into_mb_list` with
every gold tensor 1-D and the same length as `input_ids`, AND an unpadded gold is shown to
pass collation unremarked and arrive at packing still 2-D. Without the second test the first
would pass just as well on a pipeline that rejected the mismatch, in which case nothing needed
padding.

Everything the path emits is `(B, T)`, including `is_gold`. A `(B,)` per-row tensor does not
survive microbatch splitting and packing and arrives at the loss with the wrong length, which
`_compute_advantages` records as "exactly how the first routed run died" for `group_ids`.

The seam is called from two places so that both ways of building a trajectory are served by
one tested function: `RLVRWorkflow.arun_episode`, and `WorkflowExecutor._execute_workflow`,
which is where the tensor-dict path and the OpenAI-proxy path the live MATH runs actually use
converge. Both are guarded on `"gold_ids" in data`, so a run that did not ask for gold is
bit-identical and `areal` still imports without `selfevo` on the path.

## 3. The pure function

    selfevo/gold/substitute.py

    substitute_gold_rows(
        batch: Mapping[str, Any],
        rule: GoldRule | str,                      # "dyme" | "lspo_cliff" | "none"
        *,
        group_sizes: Sequence[int] | int | None = None,
        logprob_policy: GoldLogprobPolicy | str = GoldLogprobPolicy.PROX_RECOMPUTE,
        gold_reward: float = 1.0,
        pad_value: float = 0.0,
    ) -> tuple[dict[str, Any], GoldStats]

    substitute_in_place(trajectories: list[dict], rule, **kwargs) -> (list[dict], GoldStats)
    reconcile_gold_logprobs(data, *, logprob_policy=...) -> (dict, int)
    assert_gold_logprobs_filled(data) -> None

Pure: the input is never mutated (asserted), nothing imports the actor, the trainer or an
engine. The row becomes `prompt ++ gold ++ padding`, with `loss_mask` 1 on the gold only,
`attention_mask` true over prompt+gold, `rewards` 1.0 so the ORDINARY estimator gives it a
positive advantage, `versions` -1 (a gold token came from no policy version), `turn_ids` 0 on
the gold, and a per-token `is_gold` for LSPO's adapter router to read.

### The off-policy choice, and that it was overturned by measurement

My provisional default was `nan_recompute`: NaN as a tripwire, on the argument that the only
honest value for a token that was never sampled is "no value", and that a loud failure beats a
silent lie. `selfevo/FINDINGS_loss_weighting.md` (landed 2026-09-01, commit `dce4a91d`)
measured that this is wrong on both counts, and the default was changed before anything was
committed.

* NaN is not loud. `functional.py:233` rewrites a non-finite log-ratio to 0.0, so a NaN row is
  scored as perfectly on-policy with `filtered_fraction = 0.0`.
* NaN is fatal one stage earlier. `_compute_advantages` reads `data["logprobs"]` for the KL
  reward at `actor.py:741`, and `kl_ctl = 0.0` does not protect it because `-0.0 * NaN` is
  NaN. Under the live `adv_norm: mean_level=batch` the audit measured all 8 of 8 rows coming
  out NaN from one poisoned row.
* 0.0 is also wrong: with the live rejection sampling `behave_imp_weight = exp(prox_logp) < 1`
  multiplies the surrogate, keeping 0.368 of the row's weight at `prox_logp = -1`, unreported.

The audit's rule is a FINITE `logprobs` equal to the trainer's own recomputed `prox_logp`. That
value does not exist before the forward pass and this function is pure, so the default policy
`prox_recompute` is a two-phase protocol:

1. `substitute_gold_rows` writes `GOLD_LOGP_SENTINEL = +1.0` on the gold tokens. Finite, so it
   cannot poison an advantage even if every guard is bypassed; and strictly positive, so it is
   not a possible value of a log-probability and an unfilled row cannot be mistaken for a
   filled one.
2. `reconcile_gold_logprobs`, called after `compute_logp`, replaces it with `prox_logp`.
3. `assert_gold_logprobs_filled` refuses any batch whose gold tokens still carry a positive or
   non-finite behaviour log-probability. `reconcile_gold_logprobs` calls it on its own output,
   so an unfilled gold row cannot reach the loss.

**The coordinate shift is the whole difficulty and it is silent.** `logprobs` from inference is
in TOKEN coordinates (`[0.0] * input_len + output_logprobs`); `prox_logp` comes from
`gather_logprobs(logits, roll(input_ids, -1))` (`fsdp_engine.py:2116-2121`) and is in EMITTER
coordinates. `_compute_advantages` rolls `logprobs` LEFT by one to compare them, so what must
be written is `prox_logp` rolled RIGHT by one. Writing it unrolled shifts every gold ratio by
one position and raises nothing. This is asserted as a round trip — after the actor's own roll
the two must agree exactly on the gold tokens — with an anti-vacuity assertion that the
UNSHIFTED write disagrees there, so the test is about the shift and not about two tensors
happening to be equal.

Driven through the real `ppo_actor_loss_fn` with the live rejection-sampling config, a matched
row gives loss `-1.0` (`behave_imp_weight = exp(0) = 1`) against `-0.368` for the 0.0
placeholder — the audit's number, reproduced in this repo's own test file.

`ratio_one` is the swappable alternative, kept because the axis should stay swappable if the
loss ever grows an `is_gold` branch. It is NOT the default and it is not silently accepted:
nothing in `grpo_loss_fn` reads `is_gold` today, so `reconcile_gold_logprobs` refuses it rather
than performing a no-op that looks like a fix.

## 4. The reach guard, and the counts

`GoldStats` reports `n_groups`, `n_rows`, `groups_qualifying`, `rows_substituted`,
`gold_tokens`, `loss_tokens`, `groups_no_gold`, `groups_no_fit`, and the group and row indices.
`GoldMissingError` carries the counts that explain it, so a refusal caught per-prompt inside
`substitute_in_place` still contributes its lost reach to the batch report — the first draft
did not, and the list form silently understated `groups_qualifying` and `groups_no_gold`,
which is exactly the quantity the counters exist to expose. That was caught by the tests, not
by review.

Refusals, all typed:

| state | behaviour |
| --- | --- |
| rule on, batch carries no `gold_ids`/`gold_mask` | `GoldMissingError` |
| rule on, every `gold_mask` empty | `GoldMissingError`, with counts |
| groups qualified, none could be served | `GoldMissingError`, naming no-gold vs no-fit separately |
| no group qualified | NO refusal, counts returned as zero — a batch that needed no gold |
| `prox_logp`/`ref_logp`/`teacher_logp` already present | `GoldOrderingError` |
| gold tokens still carry the sentinel at loss time | `GoldOrderingError` |
| group sizes do not partition the batch | `GoldShapeError` |
| unknown rule or policy name | `GoldPolicyError` |

The distinction in rows 3 and 4 is the one that matters: an all-solved step must not fail, and
a step that needed gold and got none must.

### Reach on a fixture batch

Four groups of four — one all-wrong with a usable gold, one all-wrong whose row has no gold
text, one all-wrong whose gold is too long for the batch width, one solved:

    n_groups 4   n_rows 16
    groups_qualifying 3   rows_substituted 1
    groups_no_gold 1      groups_no_fit 1
    gold_tokens 3         loss_tokens 78    token_mass 0.0385

That is the shape of report that distinguishes "no gold was needed" from "gold was needed
three times and landed once".

### Token mass, not row count

`selfevo/FINDINGS_loss_weighting.md` section 1 establishes the objective as a single per-token
mean over the global batch (`functional.py:506,571`, whose per-microbatch division cancels the
FSDP rescale at `fsdp_engine.py:2216`), and section 2 measures a row's share of the update as
proportional to its TOKEN count: 0.5 / 1.0 / 2.0 for SFT rows of 4 / 8 / 16 tokens against
4-token RL rows. So two arms matched on gold-ROW count can differ several-fold in what the loss
reads, and no existing `route/*` key reports it. `GoldStats.as_metrics()` emits
`gold/token_mass = gold_tokens / loss_tokens` alongside the row counts; on the two-group
fixture it is 0.079 against a row fraction of 0.125, and it doubles when the gold doubles at
fixed row count.

### The reach limit nobody can remove from here

A gold is substituted only if `prompt_len + gold_len <= T`, where `T` is the batch's realised
width. 99.13% of MATH golds are at or under 1024 tokens, but `T` is the max over the batch of
prompt+response, so a group whose eight rollouts were all short gives a small `T` and can
refuse a gold that is well under the generation cap. The rate is therefore a property of the
run and not of the corpus, which is why `groups_no_fit` is a per-step metric rather than a
number quoted here. It is also why a gold longer than the row is REFUSED rather than
truncated: a cut-off derivation is a wrong target that still looks like a target.

## 5. How the two baselines' rules map on

**DyME** (`~/baselines/DyME/trainer/DyMETrainer.py:655-698`) is three rules, and only the first
is a batch operation:

1. all-wrong (`has_correct == 0`) -> replace the group's FIRST rollout with the gold and pin it
   to advantage 1. This is `GoldRule.DYME`. The victim index matches DyME's
   `i % num_generations == 0` so the baseline is a reproduction and not a variant.
2. all-correct -> advantage 0. Advantage-level; NOT included here.
3. the non-gold rows of an all-wrong group -> advantage 0. Advantage-level; NOT included here.

Rules 2 and 3 are deliberately absent rather than quietly folded in, and
`GoldStats.qualifying_groups` carries the group indices the actor seam needs to apply them
without re-deriving the predicate and drifting from it. **What the estimator gives those
sibling rows instead is measured below**, and it is not zero.

**LSPO** (arXiv 2607.27787, no code) needs the same gold row plus adapter-disjoint routing. Its
cliff set is `C = {x : sum_k R(x, y^(k)) = 0}`, implemented as the paper writes it and NOT
aliased to DyME's predicate: the two coincide only on rewards in {0,1}, and a group scoring
`[-1, +1]` sums to zero and is a cliff by LSPO's definition while DyME sees a correct sample
and declines. Tested. The routing half is another agent's; what this supplies for it is
`is_gold`, per token so it survives packing, which is what an adapter router has to read to
send gold rows to the adapter and everything else to the base.

## 6. What the solved/unsolved routing sees afterwards

This is the M19/M20 interaction, driven through the REAL `PPOActor._compute_advantages` rather
than argued.

A gold row scores 1.0, so a served group's raw rewards go from `[0,0,0,0]` to `[1,0,0,0]`:

* `unsolved = (g.max() <= 0.5)` becomes **False**. `unsolved_advantage` (M20) stops applying to
  that group, which is correct — unlikelihood on known-wrong samples has no business on a group
  that now contains a correct one — but it is a change to M20's reach, not a no-op.
* `solved = (g.min() > 0.5)` stays False. M19 is untouched, and a solved group's advantages are
  bit-identical with and without substitution (asserted).
* The group stops being SILENT. Group-level reward normalisation gives the gold row a positive
  advantage and the three wrong rollouts negative ones — the ordinary GRPO signal the
  substitution exists to create, and NOT DyME's rule 3, which would zero them.

**Consequence for reporting.** A gold arm's `unsolved_group_fraction` and
`silent_group_fraction` are lower than an ungrounded arm's by exactly the number of groups that
received a gold. The two arms' silence panels are therefore not comparable at face value, and
`gold/rows_substituted` is the term that reconciles them.

The finite sentinel is also checked on this path: with it, every advantage in the batch stays
finite, which is the property the audit says the choice of value has to buy.

## 7. The call sites left to wire

They cannot be collapsed into one, and they cannot live in the actor, because `compute_logp`
runs BETWEEN them and it is the TRAINER that calls it. Substituting inside
`_compute_advantages` would leave `prox_logp` describing tokens that are no longer in the
batch — same shape, wrong content, nothing downstream to notice — which is why
`substitute_gold_rows` refuses a batch that already carries it.

**Site 1** — `areal/trainer/rl_trainer.py`, after `prepare_batch` returns and before the
critic block, i.e. after line 721:

    rollout_batch, gold_stats = substitute_in_place(rollout_batch, config.actor.gold_rule)
    stats_tracker.scalar(**gold_stats.as_metrics())

**Site 2** — same file, immediately after the `prox_logp` loop at line 806:

    rollout_batch = [reconcile_gold_logprobs(t)[0] for t in rollout_batch]

Both are inert when the rule is `"none"`: site 1 returns the list unchanged with zero counts,
and site 2 is a no-op on a batch with no `is_gold`. `config.actor.gold_rule` does not exist
yet; it is one `str = "none"` field on `PPOActorConfig`, which is in the actor's file and so was
left alone.

## 8. What this does NOT establish

Nothing about whether a gold arm helps; that needs a GPU run. Nothing about the rate at which
golds fit a real batch, which depends on the realised `T` and is only observable in a run —
the counters exist to report it. And the executor seam is checked at the SOURCE, not driven,
because driving it needs an inference engine; the logic it calls is driven directly, but a
defect inside that guard would not be caught by this suite.

One recorded blind spot. `test_gold_target_reachability.py::test_gold_is_absent_at_both_ends_or_present_at_both`
reads the string keys of dict LITERALS inside `arun_episode` and requires that a gold in the
schema coexist with an apply seam that can accept a target. The gold now reaches the trajectory
through a function call rather than a literal, so that guard still passes — and it should,
because its "consumable" half asks about `group_apply`, which is precisely the seam this path
establishes gold cannot use. But it is a weaker statement than it was, and
`test_the_old_reachability_guard_no_longer_covers_this_supplier` pins both halves so the
reduced scope is a recorded fact rather than an accident.

## 9. Mutation kill table

`selfevo/tests/mutate_gold_path.py <copy>` against an rsynced copy of the checkout, sha256
verified identical to live before the first mutation and after the last, with every mutant
compile-checked and byte-diffed so an inert one is reported as SKIPPED rather than counted
either way. **33/33 killed, 0 survived, 0 skipped.**

| # | defect | file | killed by |
| --- | --- | --- | --- |
| 1 | gold column kept unconditionally | adapter | default-digest |
| 2 | tokenizer guard dropped | adapter | refusal test |
| 3 | EOS never appended | adapter | eos assertion |
| 4 | gold template ignored | adapter | template test |
| 5 | workflow stops attaching | rlvr | workflow gold test |
| 6 | executor stops attaching | executor | executor seam test |
| 7 | executor attaches unguarded | executor | executor seam test |
| 8 | gold keeps its natural length | attach | collation/packing test |
| 9 | mask counts padding as gold | attach | mask-sum assertions |
| 10 | over-long gold truncated not refused | attach | refusal test |
| 11 | prompt boundary ignores empty-response rows | attach | empty-response test |
| 12 | DyME predicate inverted | substitute | qualifying counts |
| 13 | LSPO cliff aliased to DyME | substitute | signed-reward divergence |
| 14 | last row sacrificed, not DyME's first | substitute | `substituted_rows == (0,)` |
| 15 | old response mask survives | substitute | exact `loss_mask` |
| 16 | attention-mask tail not cleared | substitute | exact `attention_mask` |
| 17 | gold row keeps the wrong reward | substitute | `rewards[0] == 1.0` |
| 18 | prompt overwritten too | substitute | prompt-preserved test |
| 19 | `is_gold` becomes per-row | substitute | packing test |
| 20 | sentinel becomes NaN | substitute | finite/positive test |
| 21 | sentinel becomes 0.0 | substitute | `GOLD_LOGP_SENTINEL > 0` |
| 22 | unfilled-sentinel guard never fires | substitute | unreconciled-row refusal |
| 23 | reconcile writes prox unshifted | substitute | coordinate round trip |
| 24 | reconcile shifts the wrong way | substitute | coordinate round trip |
| 25 | reconcile overwrites every row | substitute | non-gold rows untouched |
| 26 | ratio_one silently reports success | substitute | ratio_one refusal |
| 27 | reach guard never fires | substitute | no-fit refusal |
| 28 | all-empty golds accepted | substitute | empty-gold refusal |
| 29 | substitution after compute_logp allowed | substitute | ordering refusal |
| 30 | non-partitioning group sizes accepted | substitute | list-form grouping test |
| 31 | off arm copies the batch | substitute | `out[k] is v` |
| 32 | token-mass denominator taken pre-substitution | substitute | `loss_tokens` assertion |
| 33 | a refusal drops its counts | substitute | list-form mixed batch |

Two of these were SURVIVORS on the first pass and are the reason the harness was worth
running. #11: no test had a row whose rollout emitted nothing, so the `argmax` fallback in
`prompt_lengths` was unconstrained — and that row is reachable, via sglang's abort path, and
would have had its PROMPT overwritten by the gold. #30: the existing grouping test used a
uniform int, which is caught by an earlier divisibility check, so the guard on an explicit
LIST of sizes was never exercised. Both now have tests; neither was found by reading the code.

## 10. Suite

    ~/venv312b/bin/python -m pytest selfevo experiments -q

1882 passed, 3 skipped, in 339s, with these changes in the tree. The baseline taken at the
start of this session was 1631 passed; the difference is NOT all this work --
`selfevo/tests/test_gold_batch_path.py` contributes 47 and three other agents landed tests in
the same tree while this ran. What is attributable here is that 47, and that nothing else in
the suite changed state.
