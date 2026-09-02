# The seven observability features: which are functions of `k`, and what that does to the
# 102/102 result

2026-09-01. CPU only, no GPU touched. Written here rather than in `EXPERIMENTS.md` or
`GOAL.md` because both were held by another agent for this session; fold it in from here.

## The question

It is published that for a binary reward vector `r ∈ {0,1}^G` a prompt group is fully
described by `k = #correct`, and that every group-relative advantage and every
reward-derived statistic is therefore a function of `k` alone (arXiv 2607.00152 Thm 1,
2605.05112 Eq. 1, 2510.13651). This project separately measured (EXPERIMENTS.md 2026-08-31,
GOAL.md M9) that `RulePolicyRouter` is behaviourally identical to `SolveRateRouter` in
102/102 contexts under each teacher setting. Whether that measurement is a corollary of the
theorem or a genuinely new negative result about covariates turns entirely on one question:
of the seven features in `selfevo/observability.py`, which are pure functions of `r`, and
which read a covariate -- response length, truncation, log-probabilities -- that `r` does not
determine?

The classification below is a checked fact, not a reading.
`selfevo/tests/test_feature_covariate_audit.py` (20 tests) asserts it in both directions:
every k-function is required to be bit-identical when a covariate moves with `r` fixed, and
every covariate is required to actually differ under a perturbation that leaves `r` and the
k-functions untouched.

## The table

All seven come from `group_features(rewards, loss_mask, logprobs, group_sizes, *,
max_response_len, reward_threshold=0.5)`. Write `ln_i = loss_mask[i].sum()` for sample `i`'s
response length and `lp_i = sum_t loss_mask[i,t]*logprobs[i,t] / ln_i` for its mean per-token
log-probability. Nothing in the covariate rows depends on `r`; neither k-function depends on
`loss_mask`, `logprobs` or `max_response_len`.

| feature | formula as computed | class | reads |
| --- | --- | --- | --- |
| `solve_rate` | `#(r_i > 0.5) / G` = `k/G` | **k-function** | `rewards` |
| `reward_std` | population std of `r` = `sqrt(k(G-k))/G` for binary `r` | **k-function** | `rewards` |
| `mean_response_len` | `mean_i ln_i` | **covariate** | `loss_mask` |
| `len_dispersion` | `std_i(ln_i) / mean_i(ln_i)` | **covariate** | `loss_mask` |
| `mean_logprob` | `mean_i lp_i` | **covariate** | `logprobs`, `loss_mask` |
| `logprob_dispersion` | `std_i(lp_i) / abs(mean_i lp_i)` | **covariate** | `logprobs`, `loss_mask` |
| `truncated_fraction` | `#(ln_i >= max_response_len) / G` | **covariate** | `loss_mask`, `config.max_new_tokens` |

Two k-functions, five covariates, nothing mixed. The two closed forms are asserted exactly
(`test_the_two_k_functions_are_the_closed_forms_in_k_and_g`), so "k-function" here is
quantitative rather than a label: at every `G` in {2,4,8,16} and every `k` in `0..G`,
`solve_rate = k/G` and `reward_std = sqrt(k(G-k))/G`.

**On the "entropy" question, which would have been decisive: there is no entropy feature.**
Reward entropy `H(k/G)` would be a k-function and add nothing; the sampler's token-level
entropy would be a covariate and would be the escape route. Neither is in `FEATURE_NAMES`,
and `test_no_feature_is_an_entropy_of_either_kind` pins the absence so a future feature
called `entropy` cannot be read as either. The nearest thing present is `mean_logprob`, a
confidence proxy over the *sampled* tokens rather than an entropy over the distribution; it
is a covariate, but it is not the MEDS/token-entropy signal the literature places gains on.

## What the rule policy actually reads

`selfevo/routing/rule_policy.py` declares `READ_FEATURES = ("solve_rate", "reward_std",
"truncated_fraction")` -- two k-functions and exactly one covariate -- and that constant is
now checked against the partition rather than trusted
(`test_the_rule_reads_two_k_functions_and_exactly_one_covariate`). Branch by branch:

The silence test is `reward_std <= 1e-6`, a k-function; above it the mode is `RL`. On the
silent side the branch is picked by `ctx.has_self_target`, which is `solve_rate > 0`, again a
k-function; solved-and-silent goes to `solved_mode` (`SKIP` by default) and
unsolved-and-silent to `teacher_mode` if `ctx.has_target` else `SKIP`. The single covariate,
`truncated_fraction`, is read on one line only, `propose = trunc >= self.truncated_threshold`,
and it can change nothing but the `HarnessAction` -- and only when `ctx.can_evolve_harness` is
True, which is False by default and False in every run launched so far.

So on the mode axis the rule branches on k-functions exclusively. That is asserted directly:
`test_the_one_covariate_the_rule_reads_cannot_change_the_mode` sweeps `truncated_fraction` in
{0, 0.5, 1} on both silent branches and the informative one, under both teacher settings and
both harness settings, and the mode never moves. `SolveRateRouter` for its part reads only
`ctx.solve_rate` and `ctx.group_size`, i.e. `k` and `G`, and no features at all.

The strongest form is
`test_the_rule_decision_is_a_function_of_k_alone_under_the_shipped_configuration`: at every
`(G, k)`, applying all five covariate perturbations leaves the whole decision (mode *and*
harness) unchanged, so the shipped rule's decision function factors through `k`. That test was
mutation-checked -- a subclass adding one branch on `mean_logprob < -1.0` is caught at `G=2,
k=0` -- so it constrains rather than merely passing.

The one place the rule is *not* a k-function is with a harness arm wired:
`test_truncated_fraction_moves_the_harness_axis_only_when_a_harness_arm_exists` shows two
unsolved-and-silent groups with identical `k` producing `HarnessAction.NONE` versus `PROPOSE`
when their truncation differs. The covariate is inert in the configuration, not in the code.

## The k-distribution of the 102 contexts

The sweep is rebuilt exactly as
`test_rule_policy.test_the_rule_is_equivalent_to_solve_rate_on_binary_rewards` runs it, and
the rebuild is validated against the published number before anything is concluded from it
(`test_the_rebuilt_sweep_reproduces_the_published_102_over_102`: 102 per teacher setting, zero
disagreements).

`G` in {2, 4, 8, 16}, `k` in `0..G` inclusive at every `G` -- 3 + 5 + 9 + 17 = 34 compositions
-- times `truncated_fraction` in {0, 0.5, 1} = 102 contexts, times two teacher settings = 204.
**Both degenerate ends are present**: `k = 0` and `k = G` appear at all four group sizes under
both teacher settings, so a coverage objection aimed at the `p` in {0,1} regime where RL-ZVP
(2509.21880) and HIVE (2603.25184) locate their gains does not land. Coverage of `k` is not the
audit's limitation.

The limitation is the covariates, and it is now an assertion
(`test_the_102_contexts_hold_four_of_the_five_covariates_constant`). Every context in the sweep
was produced from a uniform length-6 loss mask and a uniform -0.5 log-probability, so across
all 204 contexts `mean_response_len` takes one value (6.0), `len_dispersion` one (0.0),
`mean_logprob` one (-0.5) and `logprob_dispersion` one (0.0). Four of the five covariates never
varied at all. The fifth, `truncated_fraction`, did vary over {0, 0.5, 1} -- but it is read
only on the harness axis, no context in the sweep set `can_evolve_harness`, and the comparison
was `.argmax()`, which does not look at the harness axis. Grouping the sweep by
`(G, k, has_teacher)` collapses every group to a single `(mode, harness)` pair.

## Verdict

**The 102/102 result is a corollary of 2607.00152 Thm 1, and it constrains nothing about
covariates.** Two of the seven features are functions of `k`; the rule branches on those two
for its mode and reads exactly one covariate, on an axis that has no consumer in any run so far
and that the comparison did not even inspect. Given the published theorem, the rule's mode is a
function of `k` before any sweep is run, `SolveRateRouter`'s decision is a function of `k` by
construction, and the only content left in "102/102" is that two particular partitions of
`{0..G}` happen to coincide -- a small checkable fact, not a measurement about information in
the feature set. The sweep could not have detected covariate information even in principle:
four covariates were held at a single value throughout, and the fifth could not reach the
compared output. It follows that the result must not be reported as evidence that length,
truncation or confidence carry no routing information beyond `p`. The honest claim is the
narrow one M9 already retracted to: a hand-written rule defensible under this repo's "every
threshold cites a measurement" standard collapses to one predicate because only one feature has
a measurement behind it, so the "1 feature vs 7" confound is a property of the available
evidence rather than something a better rule removes.

What survives as a real, non-corollary question is untouched by this audit, and it is where a
covariate claim would have to be earned. The *learned* controller
(`selfevo.routing.contextual.ContextualRouter`) defaults to the same full `FEATURE_NAMES`, so
its input space is genuinely not k-collapsed -- five of its seven inputs are covariates.
Whether those five carry routing signal beyond `k` is an open empirical question that requires
a run in which a covariate varies with `k` held fixed AND reaches the measured output. The
102-context sweep is not that run, and the harness axis (`can_evolve_harness`) is the one seam
in the current code where a covariate already reaches a decision.
