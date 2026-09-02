# Credit assignment for the learned router: the diagnosis verified, and what the number to
# watch actually is

2026-09-01. CPU only. Written here rather than in `EXPERIMENTS.md` or `GOAL.md` because those
two files were held by another agent for the whole of this session; fold it in from here.

## The claim under test

The recorded diagnosis: `ContextualBanditRouter` develops no preference not because the bandit
is broken but because a single per-batch scalar, credited to every decision in the batch,
carries nothing that distinguishes the arms. The recorded fix: per-prompt credit across time.

Both hold. One part of how the failure is *reported* does not, and it is the part that
`GOAL.md` tracks.

## 1. The diagnosis, as an identity rather than a threshold

`batch_outcomes` hands every unit of a batch the same `value`, and
`ContextualBanditRouter.observe` applies `A_m += x x^T` and `b_m += (value / cost) * x`. With
`value` shared, neither update mentions the mode except as the dictionary key the result lands
in. So an arm's fitted parameters are a function of **which contexts it received** and of
nothing else, and the mode label contributes exactly zero bits.

Tested by forcing the assignment through the public seam. A router with `cold_start_rounds`
past the end of the run selects round-robin, so unit `i` of a batch of nine takes
`sorted(modes)[i % 3]`. Presenting the *same* contexts rotated by one hands each arm precisely
the set of contexts a different arm received in the unrotated run, with the batch scalar
unchanged, and the fitted parameters rotate with them to `atol=1e-12`:

    theta_rl(rotated) == theta_sft(base)      exactly
    theta_sft(rotated) == theta_skip(base)    exactly
    theta_skip(rotated) == theta_rl(base)     exactly

Make the credited value depend on the mode and the identity breaks (minimum pairwise gap
> 0.1). That is the difference between "the bandit is broken" and "the signal is
uninformative", and it is now an assertion rather than an argument
(`test_credit_discrimination.py`, first two tests).

## 2. Where the diagnosis is incomplete: L1 from uniform is not evidence

`GOAL.md` tracks the L1 distance of the mode mix from uniform, and the failing signature is
quoted as 0.056 -> 0.027 over 129 steps. In a controlled world where a right answer exists,
the batch-credited router's L1 **rises**, to 0.067 -> 0.173 by quarter over 8 seeds, while its
targeting stays at chance throughout. The reason is a feedback loop the algebra above does not
cover: the assignment is not independent of the context, because the router chooses it, so the
arms drift apart on their own sampling noise. Away-from-uniform is what an arm that learned
nothing looks like too.

So a fix validated on L1 alone can be validated by noise. Every result below is carried by
**subset contrast** instead -- half the total variation distance between the mode distribution
on the prompts one mode helps and the mode distribution on the rest. It is 0 for a router with
a favourite mode however lopsided that favourite makes the mix, and it is checked for direction
separately with a per-subset targeting fraction, so a router that found the structure and
inverted it cannot pass.

## 3. The simulated world, and why one was needed

The only evidence for the fix was one GPU arm (`ctxpc`, 46 steps) whose mode mix moved. That
arm has no control and no ground truth, so it cannot separate "found the structure" from
"estimates got noisier". `selfevo/routing/credit_sim.py` supplies both: half the prompts are
helped only by SFT and half only by RL, the half is written into one of the seven observability
features and the other six are noise, and a common headroom-scaled upward drift reproduces the
confound the live run hit. Everything downstream of a decision is the shipped code -- the same
router, the same `batch_outcomes`, the same `PromptCreditLedger`. A run is 0.6 s.

## 4. Results, 8 paired seeds, means by quarter

| arm | L1 from uniform | subset contrast | final contrast |
|---|---|---|---|
| `batch` | 0.067 0.065 0.104 **0.173** | 0.058 0.071 0.090 0.098 | 0.098 |
| `prompt` | 0.164 0.412 0.412 0.217 | 0.373 0.864 0.846 0.779 | 0.779 |
| `prompt_centered` | 0.121 0.355 0.333 0.251 | 0.351 0.770 0.801 0.752 | 0.752 |
| `prompt` + `baseline="self_mean"` | 0.193 0.467 0.444 0.279 | 0.089 0.869 0.870 0.828 | **0.828** |
| `prompt`, credit SHUFFLED | 0.078 0.215 0.274 0.204 | 0.049 0.100 0.125 0.102 | 0.102 |
| `self_mean`, credit SHUFFLED | 0.216 0.366 0.249 0.247 | 0.017 0.112 0.159 0.143 | 0.143 |

Paired differences on final-quarter contrast, error bar on the difference:

    prompt   - batch            = +0.682 +- 0.021   (33 sigma, 8/8 seeds)
    prompt   - prompt SHUFFLED  = +0.678 +- 0.018   (38 sigma, 8/8 seeds)
    selfmean - selfmean SHUFF   = +0.685 +- 0.024   (29 sigma, 8/8 seeds)
    selfmean - prompt_centered  = +0.075 +- 0.018   (4.2 sigma, 7/8 seeds)
    selfmean - prompt           = +0.048 +- 0.012   (4.2 sigma, 7/8 seeds)
    centered - prompt           = -0.027 +- 0.010   (-2.7 sigma, 1/8 seeds)

Per-subset targeting under `prompt` is above 0.72 on both subsets for every seed, so the
preference points at the mode that is actually better and not merely away from uniform.

## 5. The control did its job, and one shipped variant does not survive it

The shuffled-credit control -- same ledger, same pairings, same multiset of credit values, only
the prompt-to-credit correspondence destroyed -- collapses both prompt arms to the batch arm's
level. So the effect is targeting and not added variance. A batch-credited run cannot be
shuffled at all (every unit already holds the same scalar) and `simulate` refuses to pretend
otherwise: a control that cannot fail is not a control.

**`credit="prompt_centered"` is not the fix and should not be the arm that gets a GPU.** It
subtracts the batch's mean delta, which is one number shared by every arm -- the same class of
quantity per-prompt credit exists to get away from. It measures at or below plain `"prompt"` in
both regimes tested (-0.027 +- 0.010 here; -0.064 +- 0.016 and -0.007 +- 0.017 over 20 seeds in
an earlier sweep). It was added on the reasoning that the common training trend had to be
removed. That reasoning was right; the quantity chosen to remove it was not.

## 6. The rule added: `PromptCreditLedger(baseline="self_mean")`

Credit for choosing mode `m` on prompt `p` is `(v_now - v_prev) - mean(that prompt's earlier
deltas)`: how this decision fared against what this prompt usually does. It is the within-group
centring GRPO applies across a rollout group, applied across a prompt's appearances in time.
Four design points, all measured or argued in the docstrings:

* **First appearance**: no credit, there is nothing to compare against.
* **First delta**: withheld, and counted in `prompt_credit/cold_baseline_skips`. A zero
  baseline would hand the first-credited mode the whole common trend, which is the bias that
  made the live `ctxpc` arm abandon RL at the exact step credit began flowing.
* **Seen once and never again**: ages out of the capacity-bounded LRU, counted in `evicted`.
* **Twice in one batch**: refused, unchanged from before -- the two observations would be at
  the identical policy, which measures rollout noise, not a mode.
* **Bounded memory**: two floats on the prompt's existing record, so a prompt's whole history
  is one entry that one eviction removes. Never a list, which would grow with the run.
* **Leave-current-out**: the baseline is the mean of strictly earlier deltas. Folding the
  current delta in first would make every credit its own control and shrink the largest, most
  informative observations the most.

`"last"` remains the default, so every arm run before today reproduces bit-for-bit.

## 7. Not done, and why

`credit="prompt_self_baseline"` is **not reachable from config**. Wiring it is two lines --
adding the value to the tuple in `GroupRoutingConfig.__post_init__` and to
`use_prompt_credit = _credit in (...)` in `actor.py`, plus passing `baseline=` to the ledger
there. It was left out deliberately: `actor.py` is imported by a live GPU tree through
`PYTHONPATH`, and both of those exact lines are anchors in
`selfevo/tests/mutate_prompt_credit_wired.py`, so the edit must land together with an update to
that harness and a re-run of it. That belongs to whoever next owns `actor.py`.

## 8. What a CPU cannot settle

The simulator's world is a model. It fixes the *sign* and the *mechanism* -- that per-prompt
credit lets this router discriminate and that a shared batch scalar cannot -- and it can rank
credit rules against each other under a stated confound. It cannot say what a real prompt's
solve rate does between appearances of a 1.5B policy on GSM8K, whether 29 steps is short enough
for the pairing to mean anything, or whether a router that targets correctly produces a better
checkpoint. The last of those is a benchmark question and the null on MATH-500 stands until an
arm is scored.

## 9. Mutation results

`selfevo/tests/mutate_credit_discrimination.py`, run against a copy at `~/mutcopy` with the
live checkout passed as the reference: **24 of 24 applied mutations killed, 0 skipped**, and
every mutated file verified sha256-identical to the live checkout both before and after the
run. Ten mutations on the new baseline in `prompt_credit.py`, thirteen on `credit_sim.py`, one
on `contextual.py`.

Four of those were added only after a first reading showed the tests would not have caught
them. Two would have made the `prompt` / `prompt_centered` pair the same arm run twice, and two
were scale errors in the reported metrics -- a half missing from the total variation distance,
and a uniform reference of 1/2 instead of 1/3 -- which no behavioural test could catch, because
every threshold in the suite would have scaled with them. They are killed now by three
hand-built traces with known answers. A measurement instrument that cannot be mis-set is one
nobody has checked.

Test counts: 1462 passing before this change, 1505 after. Twenty-eight of the new tests are
this change; the other fifteen arrived in this shared checkout from another agent's work during
the same session.
