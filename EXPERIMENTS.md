# Experiment log

Measured results and negative results, newest first. A claim only belongs here once it has
been observed end to end; a prediction belongs in GOAL.md until then.

## 2026-08-31 — The centred-credit arm COLLAPSED into non-termination, and the control proves it is the model

`ctxpcc` (`router=contextual`, `credit="prompt_centered"`) ran to full length, 290/290 -- the
only arm that ever did. Its checkpoint at `globalstep289`, scored on the frozen suite:

| benchmark | accuracy | truncated | cap |
|-----------|----------|-----------|-----|
| MATH-500 | 0.2560 | **500/500 (100%)** | 3072 |
| AMC23 | 0.0500 | **40/40 (100%)** | 3072 |
| AIME24 | 0.0000 | **30/30 (100%)** | 8192 |
| AIME25 | 0.0000 | **30/30 (100%)** | 8192 |

**Every single generation ran to the cap.** Not a high rate -- all of them.

**Control first, because the scoring path on this box was built an hour earlier and untested
tooling is exactly how the last false finding happened.** The untouched base model,
Qwen2.5-1.5B-Instruct, scored through the SAME path on the SAME box: `math500 acc=0.4000
n=30/30 trunc=1`. One truncation in thirty. The path is fine; the model is not.

**So this is a real capability result, and it is the FIRST time any arm in this project moved
the benchmark at all.** Every routing intervention until now moved MATH-500 by less than the
0.020 noise floor. This one moved it from ~0.52 (the other 1.5B arms at step 149) to 0.2560 --
a catastrophic *negative* effect, produced by destroying the model's ability to stop.

**Why that matters more than a null.** It falsifies the comfortable reading of the earlier
nulls, which was that the routing seam is inert. It is not inert -- it has enough leverage to
wreck a model. The earlier arms did not move the benchmark because they did not push hard
enough or long enough, not because the mechanism cannot reach capability. That is a materially
different conclusion and it changes what the negative results mean.

**Mechanism, hypothesised and NOT yet established.** The arm suppressed RL to near zero for
~70 steps and ran at roughly half SFT / half SKIP thereafter. SFT here writes a positive
constant onto a solved group's response tokens, which raises the likelihood of the tokens that
were produced -- with no term anywhere that rewards emitting EOS. Trained long enough with the
policy-gradient signal largely removed, "never stop" is consistent with what was optimised. The
solved branch was already measured inert at 0.5 and *harmful* at 2.0; this looks like the same
direction taken much further.

**Being localised now:** MATH-500 truncation rate at checkpoints 49, 149, 224 and 289, to find
whether the collapse is progressive or abrupt, and whether it coincides with the step where
per-prompt credit began (29) or with the RL suppression phase.

**Not yet known:** whether the raw `prompt` arm does the same (it stalled at 152 and its 149
checkpoint scored 0.5160 with normal truncation, so at that step it had NOT collapsed), and
whether `credit="batch"` at full length collapses too (`ctx2` ran 290 steps and is unscored).

## 2026-08-31 — The base model is worth 3.5x what every routing intervention was worth

OlympiadBench, 675 problems, cap 16384, identical harness and grader:

| model | accuracy | Wilson 95% | truncated |
|-------|----------|------------|-----------|
| 1.5B `ctx149` (batch credit) | 0.1837 | [0.156, 0.215] | 78 |
| 1.5B `rnd149` (random control) | 0.1837 | [0.156, 0.215] | 64 |
| **Frontis-MA1-30B** | **0.6400** | [0.603, 0.675] | 222 (32.9%) |

**+0.456, and the 30B number is still a lower bound** -- `cap_limited` fires at 32.9%
truncation, so its true score is higher.

**Put beside the routing results this is the whole story of the project so far.** Three credit
signals -- per-batch scalar, per-prompt, per-prompt centred -- produced three visibly different
training trajectories and a spread of **0.0015** on this exact benchmark. Swapping the base
model moved it **0.456**. The intervention we spent GPU-weeks measuring is worth roughly a
three-hundredth of what the base model is worth here.

That is not an argument that routing cannot matter. It is a measurement of where the leverage
was NOT, at the scale we were working, and it is the strongest justification yet for the pivot
recorded in GOAL.md Sec. 2c-2d: stop grinding 1.5B arms, work where the effects are large
enough to separate.

**It also retires a defence.** "The intervention might matter at a scale where the model is
competent" was the standing excuse for every null. At 1.5B the benchmark reads 0.18 for
routing-on, routing-off and random alike; at 30B the same benchmark reads 0.64. The scale
confound is no longer hypothetical -- it is measured, and it dwarfs everything else measured
here.

**Caveats, stated.** Different model families (Qwen2.5-1.5B-Instruct vs a Qwen3 MoE), so this
is not a clean parameter-count ablation -- it is base-model-and-training vs base-model-and-
training. The 30B is post-trained for agentic MLE work by Frontis, which plausibly helps on
competition math too. And both 30B and 1.5B numbers remain cap-limited at different rates, so
the gap is bounded below rather than pinned.

## 2026-08-31 — The base model is worth 3.5x what every routing intervention was worth

OlympiadBench, 675 problems, `max_tokens=16384`, identical protocol and identical cap for all
three:

| model / arm | accuracy | Wilson 95% | truncated |
|-------------|----------|------------|-----------|
| 1.5B `ctx149` (batch credit) | 0.1837 | [0.156, 0.215] | 78 (11.6%) |
| 1.5B `rnd149` (random control) | 0.1837 | [0.156, 0.215] | 64 (9.5%) |
| **Frontis-MA1-30B** | **0.6400** | [0.603, 0.675] | 222 (32.9%) |

**+0.456, and the 30B number is still a lower bound** -- `cap_limited` fires at 32.9%
truncation, so its true score is higher again.

**This is the number that should reframe the paper.** Every routing intervention measured here
moved OlympiadBench by **0.0000 to 0.0015** at 1.5B. Swapping the base model moved it by
**0.456**. The interventions this project spent GPU-weeks on are worth roughly **1/300th** of
the base model on the same benchmark, under the same harness, at the same cap.

Three consequences, stated plainly:

1. **The scale confound was real and is now measured, not argued.** "The model is too small for
   the intervention to matter" was untestable for the whole project. At 1.5B, `ctx` and `rnd`
   are identical to four decimal places; at 30B the same benchmark reads 0.64. Any future
   routing claim has to be made at a scale where the base is competent, or it is measuring
   noise around a floor.
2. **A method result must be a delta at FIXED base.** This is exactly why the paper decision
   (Sec. 2d of GOAL.md) fixes the model and varies only the routing or the harness. A number
   obtained by swapping in a better base is a scaling result and a reviewer will discount it,
   correctly -- as this table shows, it can be 300x larger than the method effect and says
   nothing about the method.
3. **It raises the bar for what counts as a real effect.** On OlympiadBench at 30B, a method
   would need to move ~0.05 to be worth reporting next to a base-model swap worth 0.456. Every
   effect this project has measured is two orders of magnitude below that.

**Caveat that matters:** the 30B is a different model family (`Qwen3MoeForCausalLM`), differently
post-trained (on MLE operators, per 2607.28568), at a different active-parameter count. This is
not a controlled scaling study and must not be reported as one. It is a demonstration that the
base dominates, which is all it needs to be.

## 2026-08-31 — First 30B numbers, and our token caps are calibrated for the WRONG model class

Frontis-MA1-30B on the frozen suite, versus the 1.5B `ctx149` checkpoint:

| benchmark | 1.5B `ctx149` | **30B** | 30B truncation |
|-----------|---------------|---------|-----------------|
| MATH-500 | 0.5240 | **0.5720** | 251/500 (**50.2%**) |
| AMC23 | 0.1750 | **0.3250** | 30/40 (75%) |
| AIME24 | 0.0000 | **0.1667** | 25/30 (83%) |

**AIME goes from unusable to usable.** It reads 0.000 for every 1.5B arm -- recorded as B2's
status and the reason it cannot separate anything -- and 0.1667 [0.073, 0.336] here. The scale
confound behind every null this project has measured is now not merely testable but visibly
different.

**And every one of these numbers is a LOWER BOUND, by a lot.** Truncation runs 50-83%. A
truncated generation is graded WRONG, so the 30B beats the 1.5B on all three benchmarks *while
half or more of its generations are being cut off mid-reasoning*.

**The finding: a token cap is a property of (benchmark x MODEL CLASS), not of the benchmark.**
The per-task table was measured on Qwen2.5-1.5B-Instruct, where MATH-500 truncated at 7.8% and
raising 3072 -> 8192 moved accuracy by less than the noise floor. The same caps applied to a
30B reasoning model truncate 50% of MATH-500. The earlier conclusion "raising the cap does not
help" was true of that model and is false of this one -- a scoped claim that would have been
wrong if it had been stated unscoped.

`cap_limited` (>10% truncation) fires on all three rows, which is what it was built for. But
its threshold was chosen against 1.5B truncation rates; at these levels the flag is not
flagging an edge case, it is saying the measurement is invalid.

**Consequence.** No 30B-versus-1.5B comparison should be reported from this run, and no arm
comparison at 30B should use these caps. The suite needs re-running with caps set for a
long-reasoning model, and the per-task table needs a model-class dimension rather than a single
global value per benchmark.

**Not yet done:** OlympiadBench at 16384 was still generating when this was written, so its
truncation rate at 30B is unknown; it may be adequate or may need the same treatment.

## 2026-08-31 — The scale blocker is gone: a 30B serves and generates on the A100

Frontis-MA1-30B (2607.28568, CC BY-NC 4.0) downloaded -- 57 GB, 12 shards, public and ungated.
Config: `Qwen3MoeForCausalLM`, 48 layers, hidden 2048, vocab 151936, i.e. a Qwen3 MoE, so the
ACTIVE parameter count is far below 30B and inference is cheap relative to the total.

Served with sglang, `--tp 4 --mem-fraction-static 0.75` on GPUs 0-3: endpoint came up, and a
chat completion returned real reasoning ("Another way is to use the distributive property:
17 * ..."). 68 GB per card at load. Torn down cleanly afterwards.

**Why this matters more than it looks.** Every null this project has measured is at 1.5B, and
"the model is too small for the intervention to matter" has been an untestable confound
throughout. The 27B and 32B attempts failed at TRAINING -- AReaL materialises full weights per
rank before sharding, and the preflight modelled post-shard steady state rather than load
peak. The pivot does not need training. It needs serving, and serving works.

So the confound is now testable: the same routing questions can be asked at a scale where the
base policy is actually competent at the task, rather than at a scale where AIME reads 0.000
for every arm.

**Not yet done:** no benchmark has been scored with this model, MLE-Bench has never been run
here, and no routing arm has touched it. This entry records only that it loads, serves and
generates.

## 2026-08-31 — Three credit signals, three training behaviours, ONE capability outcome

All three arms at `globalstep149`, greedy, FULL benchmarks (`n_graded == n_problems` in every
row), OlympiadBench at its per-task cap of 16384:

| arm | credit signal | MATH-500 (n=500) | OlympiadBench (n=675) |
|-----|---------------|------------------|------------------------|
| `ctx` | per-batch scalar | 0.5240 [0.480, 0.567] | 0.1837 [0.156, 0.215] |
| `rnd` | random at ctx's measured proportions | 0.5260 [0.482, 0.569] | 0.1837 [0.156, 0.215] |
| `ctxpc` | per-prompt delta | 0.5160 [0.472, 0.560] | 0.1852 [0.158, 0.216] |

**Spread on MATH-500 is 0.010 -- HALF the measured 0.020 noise floor.** On OlympiadBench it is
0.0015 against a jitter of roughly one point (the same ctx checkpoint scored 0.1941 and 0.1837
at two caps). Every interval overlaps every other. Nothing here separates.

**This is a stronger result than the earlier null, and it is the one worth reporting.** The
earlier finding was that a learned router does not beat random assignment at matched
proportions. This adds that an arm whose TRAINING BEHAVIOUR was completely different lands in
the same place. `ctxpc` suppressed RL to near zero for roughly 70 steps, drove its mode mix
across an L1 range of 0.24-1.14 where the batch-credit arm never left 0.027-0.069, and revised
its preferences repeatedly. `ctx` did none of that. `rnd` made no decisions at all. All three
score the same on held-out math.

So at this scale and on this task, **how the per-group routing decision is made does not reach
the benchmark.** The intervention changes what the trainer does, visibly and measurably, and
does not change what the model can do.

**Scope, stated so this is not over-read.** One seed per arm, one checkpoint, one training task
(GSM8K), one model scale (1.5B), and no `router=off` arm scored at 149 for an absolute
reference. It does NOT show that routing is worthless in general; it shows that at 1.5B these
three credit signals are interchangeable as far as held-out math capability is concerned, and
that training-time metrics (mode mix, L1 from uniform, preference formation) are NOT a proxy
for capability here. That last clause is the practical lesson: several hours were spent
reading mode-mix trajectories that turned out to predict nothing.

**Caveat on `ctxpc`:** it stalled at step 152 (see the stall entry) and its checkpoint at 149 is
the last saved. That is the same step at which `ctx` and `rnd` were scored, so the comparison
is matched -- but `ctxpc` never ran to 290 and no arm has a full-length score.

## 2026-08-31 — Second stall at the same place: this is a pattern, not an incident

`ctxpc` stopped advancing at **step 152/290**. `ctx` stopped at **step 162/290** earlier the
same day. Two runs, two boxes' worth of hours, both dying in the 150-165 band.

Signature at the stall, identical in both:

* the train log stops advancing but keeps being WRITTEN -- so log-age alone is a weaker signal
  than it looks; what is written is `ProxyRolloutServer INFO: Cleaned up N stale sessions`,
  repeating with N in the 8-45 range;
* all 8 GPUs drop to **0% utilisation** while still holding memory;
* the worker processes stay alive (5 of them here), so nothing crashes and nothing is reaped;
* `supervise.sh` does not fire, because its stall timeout is 1800s and the run sits under that
  while the log keeps ticking with cleanup messages.

**What this rules out.** Not OOM (no allocation failure, memory held steady). Not the router
-- `ctx` was `credit="batch"` and `ctxpc` was `credit="prompt"`, and the H200 arm `ctx2` ran
the SAME batch-credit config to a clean 290/290. Not the box: `ctx` and `ctxpc` both stalled
on the A100 while `ctx2` and `ctxpcc` both passed the same step band on the H200.

**The remaining candidate, not yet tested:** the rollout session pool. "Cleaned up N stale
sessions" is the only thing the process still emits, and generation is what stops. The A100
runs use `sglang.mem_fraction_static=0.7` on 80 GB cards; the H200 runs use the same fraction
on 141 GB cards, so the absolute KV budget differs by ~1.7x. That is a difference between the
two boxes that tracks the stall, and it is cheap to test by lowering the fraction on the A100.

**Cost so far:** two runs killed at roughly half length, and the arm comparison at 290 steps
is now unavailable on the A100 for any arm.

**What was salvaged.** `saver.freq_steps=25` meant `ctxpc` had a checkpoint at `globalstep149`
-- the SAME step at which `ctx149` and `rnd149` were scored -- so the stall cost the tail of
the run but not the comparison. Scored rather than relaunched.

## 2026-08-31 — AUDIT: the M9 rule is behaviourally IDENTICAL to the router it replaced

An adversarial audit of `c30d27eb` found that the claim M9 was built to support does not
hold. Reproduced independently before acting on it, and the reproduction agrees exactly.

**F1 (the claim). `RulePolicyRouter` and `SolveRateRouter` make the same decision on every
binary group composition.** Swept G in {2,4,8,16} x k solved in 0..G x truncated_fraction in
{0, 0.5, 1}: **102/102 agreement with no teacher, 102/102 with one**. The mechanism is my own
inertness argument turned around -- for a BINARY reward, `reward_std > 0` and "the group was
not unanimous" are the same predicate, and `areal/reward/gsm8k.py` and `boba_grpo.py` both
return exactly 1.0 or 0.0. A feature sweep on the shipped object confirms only `reward_std`
ever moves the mode.

So the stated purpose -- de-confounding "written vs learned" from "1 feature vs 7" -- is NOT
achieved, and "consumes all seven features" is true only in the sense of requiring their
presence.

**The resolution was to fix the claim, not to invent branches.** Keying a second branch on
`mean_logprob` or `len_dispersion` would need a threshold, and this project's standard is
that a threshold cites a measurement. There is none -- GOAL.md M15 records exactly that gap.
So the honest result, and the actual finding of M9:

> A defensible hand-written policy over these seven features collapses to ONE predicate,
> because only one of the seven has a measurement behind it. The "1 feature vs 7" confound in
> a rule-vs-learned comparison is **not removable by writing a better rule**. It is a property
> of the evidence available, and any such comparison must be reported carrying it.

The equivalence is now a TEST rather than a retracted claim, so a future divergence in either
router is a failure rather than a silent duplicate arm, plus a second test pinning the
condition (a partial-credit grader, where the rule keeps a live gradient `SolveRateRouter`
deletes -- no grader here is partial-credit).

**F2 (a real defect, now fixed). `reward_std > 0` is not an identity in float32.** A
unanimous group of eight rewards of 0.8 reduces to `reward_std = 5.96e-08`, and the shipped
router sent it to **rl** with the reason "reward_std=0.0000 > 0". Swept: the residue tops out
at **1.192e-07** (one ULP at 1.0, G=8, value 0.99); G=2 and G=4 leak nothing, and 21 of 27
reward values leak at every G >= 8, so it is the common case at production group sizes.
Latent rather than active -- binary rewards are exact at every G -- but it is precisely the
"an rl group that changes no weight" failure `__post_init__` raises to prevent for
`solved_mode='rl'`, and it would corrupt the mode proportions the matched-proportion control
rests on.

Fixed with a MEASURED tolerance rather than a guessed one, both bounds recomputed in a test
so the constant cannot drift out of the band: noise floor **1.192e-07**, smallest real
dispersion **5.0e-03** (`[1.0, 0.99]`), constant `_UNANIMITY_EPS = 1e-6`, roughly 8x above
the noise and 5000x below the signal.

**F3 (a real gap, now closed). One of my two "provably equivalent" mutation exclusions was
not equivalent.** `std > 0.05` survived all 41 tests and the full 961-test suite while
flipping the decision on realistic partial-credit groups (`[1.0,0.96,1.0,0.96]` 2.0e-02,
`[0.5,0.55]` 2.5e-02, `[1.0,0.99]` 5.0e-03) -- the live gradient the `reward_std` keying
exists to preserve. The exclusion was self-refuting on its own terms: 0.1 lies inside the
class I declared equivalent and was itself listed and killed. The real tested boundary was
0.1, and (0, 0.1] was unconstrained.

The general lesson, and it is the one this log already records in another guise: a
behavioural test can only rule out a threshold that flips a case it happens to contain, so a
constant needs a test on the CONSTANT -- measure the correct implementation's drift, measure
the smallest real effect, assert the constant lies between them. Five threshold mutations now
span the band and the property test kills all of them.

**Lower severity, corrected in place.** `HarnessAction.PROPOSE` cannot fire in any current
run and could not be observed if it did (nothing writes `can_evolve_harness`;
`actor._route_groups` keeps only `.argmax()`), so listing it as shipped behaviour without the
caveat M10 states honestly was wrong -- the caveat is now in the docstring, the GOAL row and
the table below. `EVOLVE_POLICY_FACTORIES["rule"]` is declarative only: that registry is read
by `_stub_problems`, never to build anything.

**Judgement on whether it should ship.** Kept, with the claim demoted from DONE to PARTIAL,
because what remains different is real but narrow and must not be reported as a behavioural
difference: each branch is cited to its measurement and individually mutation-covered, the
boundary is the silence condition rather than an `I_RL` threshold that `threshold_is_inert`
proves cannot change a decision, the context contract is the learned router's, and
`solved_mode` is the seam for the solved-branch A/B. Under today's config it is the same arm
as `router=solve_rate`, and running both as if they were two arms would be double-reporting.

## 2026-08-31 — M9 built: the rule the learned controller now has to beat

> **PARTIALLY RETRACTED by the audit entry above.** The headline claim of this entry --
> that deciding from the same seven features separates "written vs learned" from
> "1 feature vs 7" -- is **false**. The router is behaviourally identical to
> `SolveRateRouter` under this repo's binary graders. The branch groundings, the
> registry discipline and the mutation method below stand; the de-confounding claim
> does not. Two defects it reported as verified were also wrong: `reward_std > 0` is
> not a float32 identity, and one of its two "provably equivalent" mutation exclusions
> was not equivalent.

A build-and-verification record, not a result -- no arm has trained with it yet, and that is
stated in GOAL.md's M9 row rather than implied away.

**Why it was worth building before anything else on the critical path.** The learned router
has already been measured null against its matched RANDOM control (MATH-500 -0.0020,
OlympiadBench +0.0000). That falsifies "the per-unit decision beats a coin at matched
proportions". It cannot falsify "the per-unit decision beats THINKING", because nothing
written down was decided from the same inputs. `selfevo/routing/rule_policy.py` is that
something.

**The `rule` slot on the `evolve_policy` axis was not a baseline before.** It pointed at
`SolveRateRouter`, which (a) decides from one scalar where `learned_weights` decides from
seven, so an arm difference would confound "written vs learned" with "1 feature vs 7", and
(b) is provably inert at this granularity -- `criteria.threshold_is_inert` shows every
threshold in `(0, I_RL(1/G, G)]` induces the identical partition, and its default 0.1 is
inside that interval at every G run here: `I_RL(1/G, G)` is 0.68 at G=4, 0.66 at G=8 and
0.64 at G=16, and `criteria.threshold_is_inert` returns True at all three. Its one tunable
number
cannot change a decision. `SolveRateRouter` stays reachable as `router=solve_rate`.

**The rule, and the measurement each branch rests on.**

| branch | action | grounded in |
|---|---|---|
| `reward_std > 0` | RL | identity: `A_i = r_i - rbar` is non-zero somewhere iff the raw rewards differ. No threshold at all |
| silent, `solve_rate == 1` | SKIP | the free self-target measured **inert at 0.5, harmful at 2.0** -- the only two operating points ever measured |
| silent, `solve_rate == 0` | teacher mode if one exists, else SKIP | no self-target by construction; no run wires a teacher, so in practice SKIP |
| unsolved and `truncated_fraction >= 1` | `HarnessAction.PROPOSE` (**cannot fire today** -- nothing writes `can_evolve_harness` and `_route_groups` discards `.harness`; same gap as M10) | truncation is non-termination, not budget: 8192 -> 16384 moved it 79 -> 78 (ctx) and 61 -> 64 (rnd), and `n_truncated == n_no_box` in every MATH/AMC/AIME row |

**One number is NOT measurement-pinned and the docstring says so** rather than inventing a
justification: the truncation threshold's 1.0 is the only value at which *every* sample in
the group failed to terminate, which is the same guarantee-preserving reasoning
`CoHarnessRouter` documents for its own 1.0/0.0 defaults. What is measured is the branch's
premise, not the cut point.

**Keying on `reward_std` instead of `solve_rate` is not cosmetic.** A group with rewards
`[1.0, 0.8, 1.0, 0.8]` grades as all-correct while its advantages are +-0.1 and its gradient
is live. A solve-rate rule deletes that gradient; this one does not. That case is one of the
25 mutants -- "identity keyed on the OUTCOME split instead of the reward split" -- and it is
killed by the group the observability suite already pins as the realistic non-binary one.

**Verification.** 41 tests, every behavioural one through `ROUTERS["rule"]()` with NO
arguments, because the last two arm failures in this project were both dataclass defaults
reaching training through `factory()` (`random`'s `{rl: 1.0}`, `contextual`'s
`cold_start_rounds=0`). Three tests go through the real `PPOActor._compute_advantages`: on a
fully silent batch the rule arm is **bit-identical** to the off arm, which is the sharp form
of "SKIP spends nothing", and with `solved_mode="sft"` the constant reaches the response
tokens and not the prompt. **25/25 mutants killed**; two further mutations are omitted as
provably equivalent and named in the harness docstring (`has_target` -> `has_teacher` on a
branch where `solve_rate == 0` makes them equal; a `reward_std` threshold below the smallest
attainable non-zero std of a binary group). Target files verified intact after the run.

**What is NOT established.** That the rule is any good. It has not been run, and when it is,
it is not a matched-proportion comparison against `ctx` -- with no teacher it emits only
`rl`/`skip` against the contextual arm's rl 0.295 / sft 0.353 / skip 0.353, so each arm still
needs its own `router=random` control at its own measured proportions.

## 2026-08-31 — RETRACTION: the "silent-channel identity violation" was my own regex bug

Two entries below -- "MEASUREMENT INTEGRITY: the decomposition violates its own identity",
"DIAGNOSED: the violation is truncation", and "Sizing the unclassified bucket retroactively"
-- are **WRONG and are retracted**. So is the PROVISIONAL warning they put on GOAL.md's
composition numbers.

**The bug.** `solved_group_fraction` is a SUBSTRING of `unsolved_group_fraction`, so

    re.findall(rf"{key}\s*│\s*([0-9.eE+-]+)", txt)

for the solved key also matched every unsolved row. The two series were interleaved, which is
why "unsolved" came out exactly equal to "silent" in one run and why residuals appeared to
swing positive and negative. Fixed with a left boundary, `(?<![A-Za-z0-9_])`.

**What the corrected extraction says.** The identity
`silent == solved + unsolved` holds to a maximum residual of **1e-5** -- float32 rounding --
across every run checked (step0l 273 batches, g16 116, sa2 145, math-off 87). There is no
violation. The apparent negative residuals were all at the 1e-5 level.

**The paper was right.** `results.tex` reports silent 0.3592, solved 0.3145, unsolved 0.0447,
solved share 0.875 on step0l over "18 logged batches". Reproduced exactly: batches 44-61, the
first 18 after the metric starts reporting (`solved` is identically 0 before batch 44, which
is the earlier metric bug the paper itself documents). Measured means over that window:
silent 0.3592, solved 0.3145, unsolved 0.0447, share **0.8755**. L1 error against the
published table: 0.0001.

**Consequences.**

* The composition ratios stand as published. "87.5% of the silent channel is solved" is
  correct for its stated operating point; my "inflated 1.8x, really 0.34-0.54" claim was an
  artifact and is withdrawn.
* The reach argument stands, as it always did.
* `unclassified_group_fraction` was added on a false premise. It is KEPT, because the case it
  counts is genuinely reachable -- a group whose every row has an empty loss mask reads as
  silent while its rewards are mixed, which `test_silence_identity.py` demonstrates on the
  real path -- but it is a guard for a case that does NOT occur in these runs, not the
  explanation of an observed anomaly. Its comment has been corrected to say so.

**How this got past me.** I measured a residual, found a mechanism that could produce a
positive residual, wrote a test that reproduced that mechanism synthetically, and treated the
agreement as confirmation. The test genuinely passes and the mechanism is genuinely real --
it just was not what the logs were showing. A synthetic reproduction of a plausible cause is
not evidence that the cause is the one operating. The check I skipped was the cheap one:
verifying the extractor against a case with a known answer before trusting 7 runs of output
from it.

## 2026-08-31 — Per-prompt credit makes the router LEARN, and the first thing it learns is to stop doing RL

`ctxpc` (`router=contextual`, `credit="prompt"`), 46 steps, GSM8K / Qwen2.5-1.5B. Mode mix as
a fraction of 64 groups per step:

| window | rl | sft | skip | L1 vs uniform |
|--------|-----|-----|------|----------------|
| Q1 (steps 0-11) | 0.819 | 0.087 | 0.093 | 0.972 |
| Q2 | **1.000** | 0.000 | 0.000 | 1.333 |
| Q3 | 0.636 | 0.275 | 0.089 | 0.606 |
| Q4 | **0.000** | 0.566 | 0.434 | 0.667 |

**The signal, not the bandit, was the problem.** The batch-credit arm's L1 from uniform never
left 0.027-0.069 over 129 steps and trended FLATTER. This arm swings between 0.61 and 1.33 and
revises its preference. That is the prediction from the credit-assignment analysis, confirmed:
the same LinUCB router, the same features, the same exploration -- only the credit changed,
and it went from developing no preference to developing strong ones.

**The timing is the evidence, not a coincidence.** `prompt_credit/observed_units` is 0 for
steps 0-28 (no prompt has recurred yet) and first goes non-zero at step 29. The first step with `rl_groups == 0` is **step 30**, the step immediately
following. Before credit arrives the arm sits at ~100% RL, which is
the documented tie-break behaviour: with no `observe()` calls the arms' parameters stay at
their initial values, the scores tie, and `argmax` resolves alphabetically to `rl`. The moment
real credit arrives the router abandons RL entirely.

**A CONFOUND that probably explains the direction, and it was predicted in the module
docstring.** The credited value is a prompt's change in solve rate across ~29 steps of
training. The policy improves generally over that window, so most prompts' solve rates rise
whatever mode was applied to them -- the signal conflates "the model got better" with "this
mode helped this prompt". Every arm gets rewarded, and the arm that was applied most during
the improving window accumulates the most positive evidence. So the swing away from RL is not
yet evidence that RL is worse; it may be evidence that the credit is dominated by a common
trend.

**The implied fix, and it is cheap:** centre the per-prompt delta by subtracting the batch's
mean delta, so the common training-progress component cancels and only mode-relative effect
survives. That is one line in the wiring plus a test, and it should be run as its own arm
rather than folded in silently.

**What this arm is NOT.** With `rl_groups == 0` no group receives an ordinary policy gradient
-- every group is either SFT on its own correct sample or skipped. That is a drastic regime,
and the solved branch was already measured inert at 0.5 and harmful at 2.0. Whether it costs
capability is a benchmark question, not a metric question, and this arm has not been scored.

**Pairing rate, for the record.** `observed_units` plateaus around 16 of 64 groups (~25%), not
the ~100% expected from a clean epoch boundary -- consistent with sampling that does not
partition the dataset into exact epochs. `evicted` stays 0, so nothing is lost to capacity;
the router simply receives ~16 informative credits per batch instead of 64 uninformative ones.

## 2026-08-31 — The null holds on the frontier target, and the +1pt gap was jitter

OlympiadBench, both arms at `globalstep149`, re-run at two caps so the comparison could be
made where neither arm is budget-bound:

| cap | ctx | rnd | ctx - rnd | ctx trunc | rnd trunc |
|-----|-----|-----|-----------|-----------|-----------|
| 8192 | 0.1941 | 0.1837 | **+0.0104** | 79/675 (11.7%) | 61/675 (9.0%) |
| **16384** | **0.1837** | **0.1837** | **+0.0000** | 78/675 (11.6%) | 64/675 (9.5%) |

**The gap vanished.** ctx fell 0.1941 -> 0.1837 on the same checkpoint and the same 675
problems with nothing changed but the token budget, while rnd did not move at all. So the
+0.0104 at 8192 was measurement jitter, and `cap_limited` flagged the comparison as unfair
before it was reported -- which is what it was built for.

Two things worth taking from the second row. First, **the null now holds on the frontier
target as well as MATH-500**: -0.0020 there, +0.0000 here on 675 problems with a ~5.7pt
interval. Second, a single greedy score at this n carries about a 1-point jitter, so any
future arm difference below ~2 points on OlympiadBench is not evidence.

**The truncation is NOT a budget problem, and that revises the flag's meaning.** Doubling
8192 -> 16384 barely moved it (79 -> 78 for ctx, 61 -> 64 for rnd). These generations run away
and never emit `\boxed{}` at any budget. So `cap_limited` should be read as "this many
generations never terminate usefully", not "this score would rise with more tokens".

**A real behavioural difference between the arms, which is not an accuracy difference.** ctx
produces consistently more non-terminating generations than rnd -- 79 vs 61 at 8192, 78 vs 64
at 16384 -- across both caps. It does not translate into accuracy, and it is one seed, so it
is recorded as an observation to check rather than an effect.

## 2026-08-31 — Launched: the first arm trained on per-prompt credit

`ctxpc` = `router=contextual` with `group_routing.credit="prompt"`, otherwise identical to
`ctx`. This is the designed test of whether the measured null is the ROUTER or the SIGNAL: at
`credit="batch"` one scalar is credited to all 64 decisions and every arm provably converges
to the same parameter vector, which is why the mode mix stayed uniform for 129 steps and the
arm scored like random. The prompt path credits each decision with its own prompt's change in
solve rate between appearances.

What to watch, and what would falsify the fix rather than confirm it:
`prompt_credit/observed_units` per batch (zero means the router's `pending_cap` evicted the
prior context before the prompt returned, and the arm is silently a no-op),
`prompt_credit/same_batch_skips` (duplicate prompts halving the pairing rate), and the mode
mix's L1 distance from uniform, which stayed flat-to-decreasing for the batch-credit arm.

## 2026-08-31 — A killed mutation harness left the target MUTATED, and the next run made it the baseline

Worth recording because the second failure is worse than the first, and both were silent.

A mutation harness writes a mutation, runs the tests, and restores in a `finally`. A 2-minute
tool timeout killed one mid-mutation, so the restore never ran and
`key = str(ids_cpu[row])` stayed on disk in place of
`key = prompt_key(ids_cpu[row], mask_cpu[row])`.

The next run then read the ALREADY-MUTATED file as its `original`. So it reported
`baseline green` on a corrupted tree, reported that one mutation's anchor `appears 0x` and
counted it as a survivor rather than an error, applied every other mutation on top of the
corruption, and would have written the corrupted text back as the "restore". Five of six
mutations were correctly restored; the sixth was permanent until caught.

**Two independent defects had to line up, and both are recorded elsewhere in this log.** The
harness left the file dirty, AND the wiring tests could not tell prompt-only keying from
whole-row keying, because they reused IDENTICAL rows across the two sightings. Real rollouts
of one prompt differ in their responses, so a whole-row key would never pair them -- the test
used an unrealistic case, exactly like the earlier "hash only the first token" survivor whose
test prompts differed at position 0.

Fixed: the tests now vary the response tokens while holding the prompt fixed, which is the
discriminating case. Re-run: **6/6 killed**, and the target file verified intact afterwards.

**How to apply.** Run mutation harnesses in the background, never in a foreground call that
can hit a tool timeout. After any harness run -- especially an interrupted one -- verify the
target with `git diff` or by grepping for each mutation's replacement text. A
`SKIP: anchor appears 0x` on a mutation that worked before is the tell that the file is
already mutated. `baseline green` only says the tests pass on whatever is on disk.

## 2026-08-31 — OlympiadBench really is the long-output benchmark, and the cap flag earned its keep

The per-task cap hypothesis, tested. Same `ctx@149` checkpoint, OlympiadBench only:

| cap | accuracy | Wilson 95% | truncated | cap_limited |
|-----|----------|------------|-----------|-------------|
| 3072 | 0.1778 | [0.151, 0.208] | 103/675 (15.3%) | n/a (pre-flag) |
| 8192 | **0.1941** | [0.166, 0.226] | 79/675 (11.7%) | **True** |

**Confirmed, and it is the exception.** On MATH-500 and AIME, 3072 -> 8192 moved accuracy by
less than the noise floor. Here it gained +0.0163 and cut truncation by a quarter. So a
uniform cap really was wrong for this suite, and this benchmark really is the one with long
answers -- which is what the per-benchmark table was built to express.

**The flag did the job it was built for.** At 8192, `cap_limited` fired for ctx (79/675,
11.7%) and NOT for rnd (61/675, 9.0%). An arm comparison across that asymmetry is unfair: the
arm that truncates more is penalised more, and truncated generations are graded wrong. So the
observed ctx - rnd = **+0.0104** (0.1941 vs 0.1837, intervals [0.166,0.226] vs [0.156,0.215])
is NOT reportable as an effect. It is a hint that has to be re-measured where neither arm is
budget-bound. Raised to 16384 and both arms relaunched.

Note what this would have looked like without the flag: a +1 point gap on the frontier
benchmark, intervals overlapping, and nothing in the output to say one arm was
budget-truncated more than the other.

## 2026-08-31 — Per-prompt credit wired into the actor, and a within-batch defect it exposed

`group_routing.credit` now selects the signal the router learns from: `"batch"` (default,
unchanged, what every prior run used) or `"prompt"`. The prompt path keys each group by its
prompt tokens, credits the PRIOR decision for that prompt with the change in its own solve
rate, and records the current one. Default untouched, so rollback is exact and the two are an
ablation pair on one axis.

**A test written to check that different prompts do not pair failed, and it was the code that
was wrong.** Two groups in ONE batch can carry the same prompt, and the ledger paired them --
crediting one group's decision with another group's solve rate at the IDENTICAL policy. That
delta measures sampling noise between two rollout groups, not the effect of a mode; the entire
value of this ledger is that the two observations are separated in training time. The ledger
now refuses to pair within a batch, keeps the earlier record so the decision that pairs with
the prompt's next appearance is the one applied first, and counts `same_batch_skips` so a
batch full of duplicate prompts is visible rather than silently halving the pairing rate.

19 ledger tests (11/11 mutants) and 7 wiring tests driven through the real
`_compute_advantages`. One mutation SKIPped on a stale anchor after the refactor and was
rewritten rather than left as a hole -- a skipped mutation is an untested defect wearing a
green tick.

**Feasibility constraint worth recording:** the router's own `pending_cap` (4096) must exceed
groups-per-step x steps-per-epoch (64 x 29 = 1856 here) or the prior unit's context is evicted
before its prompt returns and prompt credit silently reaches nothing.
`prompt_credit/observed_units` is logged per batch so that failure is visible.

**Not yet run.** No arm has trained with `credit="prompt"`. It is the designed test of whether
the null is the router or the signal, and it needs a GPU arm.

## 2026-08-31 — RESULT: the learned router is indistinguishable from its matched control

The first real arm comparison. Both checkpoints at `globalstep149`, both scored greedily at
`max_tokens=3072`, and the control ran at the contextual arm's MEASURED mode proportions
(rl 0.295 / sft 0.353 / skip 0.353), so the two arms differ only in WHICH unit gets which
mode -- not in the mixture.

| benchmark | ctx | rnd | diff | ctx CI | rnd CI |
|-----------|-----|-----|------|--------|--------|
| **MATH-500** | 0.5240 | 0.5260 | **-0.0020** | [0.480, 0.567] | [0.482, 0.569] |
| AMC23 | 0.1750 | 0.1250 | +0.0500 | [0.087, 0.320] | [0.055, 0.261] |
| AIME24 | 0.0000 | 0.0000 | 0.0000 | [0.000, 0.114] | [0.000, 0.114] |
| AIME25 | 0.0000 | 0.0333 | -0.0333 | [0.000, 0.114] | [0.006, 0.167] |

**MATH-500 is the only row with resolution, and it reads -0.0020** -- one TENTH of the 0.020
systematic noise floor measured for greedy scoring here. AMC23 and AIME have 40 and 30
problems, so their intervals span 20+ points and cannot separate anything.

**This is the null the credit-assignment finding predicted**, and it is worth stating as a
positive claim rather than an absence: at matched mode proportions, choosing WHICH unit gets
which mode -- as this router chooses it -- buys nothing over assigning modes at random. What
the routing arm does is set the MIXTURE; the per-unit decision carries no measurable value.

It is consistent with the mechanism, not merely with bad luck: the same router was separately
measured to develop no preference over 129 steps because a single per-batch scalar credited to
64 decisions cannot separate the arms. A router that has learned nothing SHOULD score like
random at the same proportions, and it does.

**Scope, stated so this is not over-read.** One seed, one checkpoint, one task (GSM8K
training), one model scale (1.5B). It falsifies "this learned router, with this credit
signal, beats its matched control" -- not "routing cannot help". The per-prompt credit ledger
built earlier is the designed test of the second question and is not yet wired.

## 2026-08-31 — Token cap: measured, and my budget-artifact claim was WRONG

Same checkpoint, same benchmarks, only `--max-tokens` changed:

| benchmark | acc @3072 | acc @8192 | trunc @3072 | trunc @8192 |
|-----------|-----------|-----------|-------------|-------------|
| MATH-500 | 0.5240 | 0.5260 | 39 | 36 |
| AMC23 | 0.1750 | 0.1500 | 3 | 4 |
| AIME24 | 0.0000 | 0.0000 | 11 | 8 |
| AIME25 | 0.0000 | 0.0333* | 3 | 2 |

Raising the cap 2.7x moved accuracy by less than the noise floor and barely moved truncation.
**In every row `n_truncated` equals `n_no_box`**: these generations never emit `\boxed{}` at
all rather than being cut off mid-solution. So they are genuine failures, and the earlier
claim in this log that "the AIME zero is partly a budget artifact" is **refuted** -- it was a
hypothesis stated with more confidence than the evidence supported.

Per-benchmark generation config was still built, for two reasons that survive the refutation:
a uniform cap is wrong in principle (7.8% truncation on MATH-500 versus 15.3% on
OlympiadBench at the same value), and the results row previously recorded only `seed` and
`temperature`, so two runs generated at different budgets looked comparable. Every generation
parameter is now resolved per benchmark and RECORDED in the row, and a `cap_limited` flag
fires above 10% truncation so a budget-bound score cannot be silently compared against one at
a different cap.

OlympiadBench remains the open case: 15.3% truncation at 3072, the highest in the suite, and
never re-measured at a higher cap. Its 8192 override is a hypothesis, and `cap_limited` is
what will confirm or refute it.

## 2026-08-31 — First benchmark scores for a routed arm: `ctx` @ globalstep149

The gating measurement, finally taken. `router=contextual`, GSM8K training,
Qwen2.5-1.5B-Instruct, scored greedily on the frozen suite via
`experiments/bench/run_math.sh` (sglang server, repo copy of `math_bench.py`).

| benchmark | n | accuracy | Wilson 95% | truncated |
|-----------|---|----------|------------|-----------|
| MATH-500 | 500 | **0.5240** | [0.480, 0.567] | 39 |
| AMC23 | 40 | 0.1750 | [0.087, 0.320] | 3 |
| AIME24 | 30 | 0.0000 | [0.000, 0.114] | 11 |
| AIME25 | 30 | 0.0000 | [0.000, 0.114] | 3 |

**Read MATH-500 only.** AIME reads 0.000 at this scale, which is already recorded as B2's
status -- it cannot separate any two arms here. AMC23's 40 problems give a 23-point interval,
so it cannot either. MATH-500 is the only row with enough resolution to compare arms, and even
it carries a 0.020 systematic noise floor from greedy-scoring irreproducibility.

**Truncation is not negligible and biases DOWNWARD.** 39/500 MATH-500 generations hit the
3072-token cap and were graded wrong, as did 11/30 on AIME24 -- so the AIME zero is partly a
budget artifact rather than pure inability. Every number here is a lower bound at this token
budget, and arm comparisons are only fair at the SAME budget.

**No comparison yet.** This is one arm. The matched control `rnd` at the same
`globalstep149`, run at the contextual arm's measured proportions, is being scored next; until
that lands this is a number, not a result. The `ctx` run stalled at step 162 so the arms cannot
be compared at 290; a full-length contextual run (`ctx2`) was launched on the H200 to recover
that, with the cold-start fix applied there first -- that box did NOT have it, so a contextual
run started there before this hour would have been inert.

## 2026-08-31 — The `ctx` run stalled at step 162, and the orphan check could not see it

The watchdog worked and the cleanup did not. `supervise.sh` logged
``WATCHDOG: train.log stalled 1824s (> 1800s); dumping stacks then killing`` at 15:29:31,
dumped ``/proc/<pid>/status`` for the workers, and then the supervisor itself exited -- while
nine training processes stayed alive holding all 8 GPUs at 0% utilisation. The train log ends
mid-metrics-table, so the stall happened while writing a step summary rather than during
generation.

**Why STEP 1 of the hourly loop could not catch this.** The orphan check compares the
experiment name on GPU-holding processes against the run that is supposed to be live. Here the
orphans ARE `ctx` -- the same experiment -- so the check reported eight healthy `ctx`
processes on a box that had made no progress for 40 minutes. Name matching cannot distinguish
a live run from its own corpse. **Log age is the only signal that catches this**, which is
what STEP 2 exists for, and it is why STEP 2 must never be skipped when STEP 1 looks clean.

**Cause of the stall itself: not yet diagnosed.** The supervisor's stack dump is in
`~/runs/ctx/supervisor.log`. Recorded as unexplained rather than guessed at; what is
established is only that the process group survived the watchdog's kill attempt, which is the
same orphan-survives-cleanup pattern seen twice already today.

**Recovered rather than restarted.** The run had written 11 checkpoints (`saver.freq_steps=25`),
the newest at `globalstep149`, a complete HF model directory. Rather than relaunch and lose the
box for hours, the checkpoint went straight to the gating measurement that has been blocked all
day by both boxes being busy: scoring on the frozen math suite. A matched comparison against
`rnd` is available because that arm has a checkpoint at a comparable step.

**Consequence for the arm comparison.** `ctx` stopped at 162/290 and `rnd` is at 245/290, so
the two arms cannot be compared at their final steps. They CAN be compared at a matched
checkpoint (~149), which is the honest comparison and is the one to report.

## 2026-08-31 — NEGATIVE: the learned router receives feedback and still develops no preference

Measured on the live `ctx` run at 129 steps, GSM8K / Qwen2.5-1.5B, `router=contextual` with
the cold start fixed. Mode mix by quarter, as a fraction of 64 groups per step:

| window | rl | sft | skip | L1(mean vs uniform) | mean per-step L1 |
|--------|------|------|------|---------------------|------------------|
| Q1 | 0.305 | 0.341 | 0.354 | 0.056 | 0.238 |
| Q2 | 0.299 | 0.338 | 0.363 | 0.069 | 0.335 |
| Q3 | 0.351 | 0.303 | 0.346 | 0.061 | 0.248 |
| Q4 | 0.344 | 0.320 | 0.336 | **0.027** | 0.277 |
| all | 0.325 | 0.325 | 0.350 | 0.033 | 0.275 |

**The aggregate mix is indistinguishable from uniform thirds and the trend is flat to
DECREASING** -- the last quarter is the most uniform of the four. A learner should move away
from uniform as its arm estimates separate; this does the opposite. Per-step L1 stays around
0.27, so the router does make different decisions step to step, but they average out instead
of accumulating into a preference.

**This is not the earlier exploration bug, and not a plumbing failure.** Feedback is flowing:
128 feedback records, `n_units=64`, and exactly ONE `confounded_skips` in the whole run.
CORRECTED 2026-08-31 against a boundary-anchored re-read of the log -- the first version of
this entry overstated three of these:

* `weak_attribution` is **not** 0.0 throughout. It is 0 in 108 of 128 records and reaches 0.5.
* `n_modes` is **not** 3 in every batch. It reads 3 in 95 of 128 records, minimum 2.25. Both
  counters are averaged over four data-parallel ranks, so a sub-3 value means one rank's view
  was short a mode.
* The single confounded skip is at **step 57**, not the first batch. The "no predecessor"
  attribution was invented, not measured.

The claim these support -- that feedback was flowing rather than blocked -- survives, and is
backed additionally by `dominant_share` staying in [0.352, 0.766]. The router got 128 updates
and learned nothing.

**Mechanism, and it is structural rather than a tuning problem.** `batch_outcomes` credits a
single scalar -- the change in mean raw reward between consecutive batches -- to every
decision in the batch. LinUCB updates `A_m += x x^T` and `b_m += r x`. When `r` is shared
across all units in the batch and the mode assignment is (initially, by round-robin cold
start) independent of the context `x`, then for every arm

    b_m ~ r * n_m * xbar,   A_m ~ n_m * E[x x^T],
    theta_m = A_m^-1 b_m ~ E[x x^T]^-1 * r * xbar,

which is **the same vector for every arm**, and the `n_m` cancels. The arms therefore start
identical and stay near-identical, selection is driven by the UCB bonus alone, and the mix
stays uniform. The credit signal contains no information that distinguishes one arm from
another, so no amount of it will separate them.

**What this rules out and what it implies.** It rules out "the router needs more steps" -- an
uninformative signal does not become informative with repetition. It implies the fix is
credit ASSIGNMENT, not the bandit: a per-batch scalar cannot train a per-group policy.

The naive within-batch contrast does NOT fix it: comparing the rewards of groups routed to
different modes inside one batch measures which PROMPTS were easy, not which mode was better,
because the decision affects the next policy rather than the reward already observed. A
correct signal has to hold the prompt fixed and vary the mode across time -- credit a prompt's
change in solve rate between the batch where it was routed to mode m and its next appearance.
That needs prompt identity carried through the pipeline, which the current
`unit_id = f"{step}:{i}"` deliberately does not provide (it is batch-local by construction, to
stop cross-batch collisions).

Recorded as a NEGATIVE result for M8. The learned meta-controller is reachable, explores, and
receives clean feedback -- and is still, on this evidence, not learning. The `rnd` control at
matched proportions is running so the comparison is measured rather than assumed.

## 2026-08-31 — The matched control was also inert, for two independent reasons

Same defect class as the contextual router's zero cold start, in the same registry, found the
same way: watch what the arm actually emits. The `rnd` arm ran with `rl_groups=64`,
`sft_groups=0`, `skip_groups=0` at every step -- bit-identical to the off arm while reporting
as the control that every claim about the learned router has to beat.

**Cause 1: no usable default, and one was taken anyway.** `RandomRouter.proportions` defaults
to `{rl: 1.0}` and `_route_groups` builds routers with `factory()` and no arguments. The
class docstring already said the control "must be run at the proportions the criterion router
actually produced, measured, not assumed" -- so there is no correct default, and the factory
now refuses unless `SELFEVO_RANDOM_PROPORTIONS` (or an explicit argument) supplies them.

**Cause 2: the control could not emit the mode it was matching.** `RandomRouter` gated
teacher-requiring modes on `ctx.has_teacher`, but `has_target = has_teacher or
has_self_target`. No run here wires an external teacher, so EVERY sft draw degraded to skip
and the mix collapsed to rl/skip -- even once the proportions were right. A group with
`solve_rate > 0` supplies its own target; that is the method's central claim, it is what
`apply_decisions` already implements with no teacher tensor, and it is what the contextual
router does in live training. Fixed to gate on `has_target`.

An existing test, `test_random_router_degrades_to_skip_without_a_teacher`, asserted the buggy
behaviour. It was rewritten rather than deleted, to the stronger contract the system actually
needs: sft with a self-target, skip with no target at all, sft with an external teacher alone.

**Proportions, measured not assumed.** From the live `ctx` run over its last 40 steps:
rl **0.2946**, sft **0.3527**, skip **0.3526** (sd ~0.12 per step, 64 groups/step). The
control was relaunched at exactly these.

**Worth noting about the bandit itself:** after 61 steps the contextual router's average mode
mix is close to uniform thirds. It varies step to step, so it is not stuck the way the
zero-cold-start version was, but it has not yet developed a strong preference. If that holds,
the matched-proportion control is the right comparison and the honest outcome may be a null
-- which is the result either way, and is why the control had to be fixed before it ran.

**Operational, and it invalidated my STEP 1 orphan check on that box:** `nvidia-smi` on the
H200 reports HOST pids from a different PID namespace, so `ps -o cmd= -p <pid>` returns empty
and `kill -9 <pid>` is a no-op there (and could in principle hit an unrelated host process).
Reaping on that box must use in-namespace `pgrep -f` patterns; the A100 is not affected.

## 2026-08-31 — Sizing the unclassified bucket retroactively: one headline halves, one survives

The residual `silent - (solved + unsolved)` is exactly what the new
`unclassified_group_fraction` counts, so it is recoverable from every completed run without
rerunning anything. Second-half means across seven runs:

| run | task | arm | silent | solved | unclass | %unclass | **quoted** s/(s+u) | **correct** s/silent | neg steps |
|-----|------|-----|--------|--------|---------|----------|--------------------|----------------------|-----------|
| g16 | GSM8K | off | 0.4553 | 0.1541 | 0.2767 | 60.8% | 0.8627 | **0.3385** | 8/116 |
| step0m-off | GSM8K | off | 0.5906 | 0.2880 | 0.2688 | 45.5% | 0.8949 | **0.4876** | 13/145 |
| step0m-on | GSM8K | on | 0.6144 | 0.3099 | 0.2623 | 42.7% | 0.8802 | **0.5044** | 29/178 |
| sa2 | GSM8K | on@2.0 | 0.5745 | 0.3087 | 0.2207 | 38.4% | 0.8725 | **0.5374** | 27/145 |
| math7b | MATH | off | 0.5632 | 0.3285 | 0.1080 | 19.2% | 0.7218 | **0.5834** | 18/58 |
| math7b-on | MATH | on | 0.6950 | 0.3311 | 0.2412 | 34.7% | 0.7296 | **0.4764** | 12/58 |
| math-off | MATH | off | 0.4389 | 0.2038 | -0.0173 | -3.9% | 0.4468 | 0.4644 | 57/87 |

**The composition claim was inflated by roughly 1.8x.** "87.5% of the silent channel is
solved" is the `solved/(solved+unsolved)` ratio, and it reproduces here (0.87-0.89 on GSM8K).
With the unclassified mass in the denominator where it belongs, the solved share is
**0.34-0.54**. The channel is not overwhelmingly solved; it is roughly half solved and, on
GSM8K, nearly as much *neither* -- truncated groups that the correctness split cannot
describe at all.

**The reach argument survives, and this matters for the critical path.** The "31.4% self-target
vs ~4.5% teacher, a 7x difference" comparison uses `solved_group_fraction`, which is a direct
mean over ALL groups (0.288-0.331 on GSM8K here, matching the quoted 31.4%). It never divided
by the contaminated denominator, so the re-ordering it justified stands. Only the composition
RATIO was wrong, not the reach.

**A second mechanism is still unexplained and is not small.** Negative residuals appear in
every run, and in `math-off` they are the majority (57 of 87 steps), which is why its
"unclassified" reads negative. A missing bucket cannot produce a negative residual, so the
retroactive numbers above are contaminated by whatever does: treat the magnitudes as
approximate and the direction as solid. Runs launched after 2c3a5b1a log the bucket directly
and will settle it.

**Incidental, and confounded:** `math7b-on` shows a HIGHER silent fraction than `math7b`
(0.695 vs 0.563) and more than double the unclassified mass. These are separate runs rather
than an arm-matched pair, so it is recorded as an observation to check, not an effect.

## 2026-08-31 — DIAGNOSED: the decomposition violation is truncation, and the metric was incomplete

Follow-up to the integrity finding below, resolved on the real path rather than by argument.

`silent` is read from `seq_adv = (advantages * loss_mask).sum(-1)`. A sequence with NO
response tokens contributes exactly 0 whatever its advantage, so a group of fully-truncated
sequences reads as SILENT while its raw rewards are mixed -- and a mixed group is neither
all-solved (`min > 0.5`) nor all-unsolved (`max <= 0.5`). Those groups fell out of the
decomposition entirely.

Reproduced as a falsifiable test before changing anything: with every sequence carrying
response tokens the identity holds on four different reward patterns; zeroing one group's
loss mask produces `silent=1.0, solved=0.5, unsolved=0.0`, a **+0.5** residual -- the same
sign and mechanism as the **+0.277** mean seen in `g16`.

**Fix.** A third bucket, `unclassified_group_fraction = mean(silent * (1-solved) *
(1-unsolved))`, so the decomposition is complete by construction and ratios have a correct
denominator. 5 tests, 4/5 mutants killed; the survivor ("compute silence without the loss
mask") is EQUIVALENT under the shipped config and is documented as such in the harness --
advantages are constant across a sequence there, so masked and unmasked sums are zero
together. Killing it would need token-level advantages (`gae_lambda < 1` with a value model),
which no config here uses.

**What this does and does not restore.** It explains and removes the POSITIVE residual, and
it means the composition question is now well-posed: a run reports how much of its silent
channel is solved, unsolved, or merely truncated. It does NOT explain the NEGATIVE excursion
(-0.109 at g16 step 112), which cannot arise from a missing bucket -- subsets cannot exceed
their superset. That remains open, and the historical composition numbers stay PROVISIONAL
until a rerun with the third bucket is available to compare against.

**Consequence worth stating.** If `unclassified` turns out to be large in real runs, part of
what this project has been calling the silent channel is not about correctness at all but
about truncation, and neither the self-target argument nor the teacher argument applies to
it. That is measurable on the next run and is now measured by default.

## 2026-08-31 — G=16: doubling the group size does NOT halve the silent channel

The cheapest decisive experiment named in GOAL.md. If a group is silent because every member
happened to land on the same side of the reward threshold -- a binomial tail with one solve
rate p per prompt -- then silence falls as p^G, and doubling G should SQUARE it. If instead
silence is driven by prompt HETEROGENEITY, with many prompts effectively always-solved or
always-unsolved, silence barely moves with G because those prompts are silent at any G.

Matched pair, GSM8K / Qwen2.5-1.5B-Instruct / routing OFF, second-half means:

| G | run | silent_group_fraction | sd | steps |
|---|-----|----------------------|-----|-------|
| 8 | `step0m-off` (H200) | **0.5906** | 0.0417 | 145 |
| 16 | `g16` (A100) | **0.4553** | 0.0756 | 116 |

Homogeneous-binomial prediction, calibrated on the G=8 run itself: silence(16) =
silence(8)^2 = 0.5906^2 = **0.3488**. Observed 0.4553 is **1.31x** that -- doubling G bought
a 22.9% relative reduction where the homogeneous account predicts 41%.

**Reading.** Directionally this supports heterogeneity: a large share of the silent channel is
prompts that are silent at ANY group size, so buying signal by raising G is expensive and
saturating. It is NOT the pure-heterogeneity extreme either, which would predict no movement
at all; some prompts genuinely sit in the binomial-tail regime.

**Confounds, stated because this is a between-run comparison and not a controlled sweep.**
Different boxes; different batch size (256 vs 64 prompts); different epoch counts; one run per
G, so the sd columns describe within-run step variation, not run-to-run variation. A clean
version is a single-box sweep at G in {4, 8, 16} with batch size held fixed and >=2 seeds.
Until that exists this is a strong directional result, not a measured coefficient.

## 2026-08-31 — MEASUREMENT INTEGRITY: the silent-channel decomposition violates its own identity

By construction `solved_group_fraction = mean(silent * solved)` and
`unsolved_group_fraction = mean(silent * unsolved)`, both elementwise-bounded by
`silent_group_fraction = mean(silent)`. Since a group cannot be both all-solved
(`min > 0.5`) and all-unsolved (`max <= 0.5`), the identity

    silent_group_fraction == solved_group_fraction + unsolved_group_fraction

must hold at every step, and averaging over microbatches preserves it by linearity.

**It does not hold.** In `g16` the residual `silent - (solved + unsolved)` has a second-half
mean of **+0.277**, exceeds 0.01 at **104 of 116 steps**, and reaches **-0.109** at step 112 --
negative, which is impossible for a decomposition into subsets. Step 0 satisfies it exactly
(0.1875 = 0.1406 + 0.0469), so the computation is right at least initially and something about
how the three scalars are aggregated or reported diverges afterwards. The same pattern appears
in `sa2` (2nd-half mean +0.221, 113/145 steps, min -0.344).

**Consequence, and it is not small.** The composition numbers this project has been quoting --
"87.5% of the silent channel is solved", the MATH 39.1% / 81.6% figures, the 7x reach argument
that RE-ORDERED the critical path -- are all ratios of these two metrics. Until the identity
violation is explained they cannot be used quantitatively, and any claim resting on them is
provisional. `silent_group_fraction` itself is the directly computed primary metric and is not
implicated by this specific failure, so the G=16 result above still stands.

**Not yet diagnosed.** Candidate explanations (sequence-level `seq_adv` summing to ~0 while
tokens carry gradient; cross-rank aggregation weighting; a reported statistic that is not the
plain mean) are guesses and are recorded as guesses. The next step is a CPU test that asserts
the identity on the real `_compute_advantages` path, which is cheap and decisive -- exactly
the check that should have existed before these numbers were quoted.

## 2026-08-31 — Editing a shell script that bash is currently executing

`bash` reads a script LAZILY, by byte offset, not into memory. A long-running script sits
blocked on its final command with a file offset stored; inserting lines ABOVE that point
shifts every later byte, so when bash resumes it reads from the old offset into the middle of
a now-different line.

I patched `experiments/harness/step0m.sh` to add the router arm while `g16` was still running
from that same file, ~30 lines above the trainer invocation. The training itself was never at
risk -- the python process is already exec'd and its metrics are already in the log -- but the
teardown lines after it (`rc=${PIPESTATUS[0]}`, the exit-code echo, `exit "$rc"`) are read
after the patch, and a garbled read there produces a wrong exit code, which makes the
supervisor restart a run that actually succeeded.

**How to apply.** Never edit a script an active run launched from. Write the change to a NEW
filename and point the next launch at it -- which is what the H200 got
(`step0m_router.sh`). This is the shell-script analogue of the rule already recorded for
tensors in `group_apply`: do not mutate what a caller still holds.

## 2026-08-31 — The Router→advantage seam is live, and a uniform batch starves its own feedback

**Built and verified.** `actor.py::_route_groups` had no test: it was called from
`_compute_advantages` but nothing established that a Router's decision reaches the tensor the
loss reads. Added `selfevo/tests/test_actor_router_seam.py` -- 11 tests driven through the
REAL `_compute_advantages`, deliberately not through the helper, because a test that calls
the helper cannot catch the helper being unreachable. `mutate_actor_router_seam.py`: **7/7
killed**, including "router rebuilt every batch" (a learned router that silently never
accumulates), "unit ids drop the batch prefix" (feedback credited to the wrong unit), and
"unregistered router name silently ignored" (an arm that reports as run and never ran).

Two of my initial assertions were wrong and both were informative:

1. **The prompt region is not zero before routing.** The actor leaves real GAE values there
   (the seam's own docstring measures -0.87 for an informative group). The correct claim is
   that routing does not MOVE them, asserted against the unrouted tensor.
2. **A fixed-mode router never produces feedback.** `batch_outcomes` credits one scalar --
   the change in mean raw reward between consecutive batches -- across a batch's decisions.
   If every group took the same mode, that scalar cannot be divided among them, so the update
   is refused as `ConfoundedUpdate`.

**(2) is a design constraint, not a test artifact.** A learned router that CONVERGES to one
mode stops receiving feedback entirely: it is not punished for converging, it goes blind, and
any later change in which mode is right is invisible to it. Exploration is the precondition
for the learning signal existing at all, not merely a way to improve it. A converged-but-wrong
router and a converged-and-right router look identical from the feedback stream; the
`feedback/confounded_skips` counter is the only diagnostic that separates them.

This predicts a specific failure mode for the LLM-as-router variant (M23): an LLM asked to be
decisive collapses the mode distribution faster than a bandit with explicit exploration, and
therefore starves its own feedback sooner. Worth measuring rather than assuming.

**Still unverified:** that a learned router DECIDES BETTER than the fixed rule. Reachable is
not effective. That is a GPU arm, now item 1b on the critical path.

## 2026-08-31 — Orphaned workers from a failed run silently disable the box

**Measured.** Both boxes sat at ~0% GPU utilisation for hours while appearing "busy".

* A100: four `areal.infra.rpc.rpc_server` processes from the failed `lora27` experiment
  survived their parent and held 72.8 GB on each of GPUs 0-3 (291 GB total). The
  pre-launch guard in `step0m.sh` did its job -- it refused to launch three times with
  `REFUSING TO LAUNCH: GPUs already hold memory` (rc=4) -- so the supervisor exhausted its
  restarts against a condition no restart could clear.
* H200: leftovers from the killed `lora32b` run still held distributed port 22794, and the
  next run died with `EADDRINUSE` at `create_process_group`, surfacing as the misleading
  `Worker 'actor/0' failed with exit code 0`. After a clean reap the *identical* config
  reached `step 1/58` with all 8 GPUs at 100%.

**Why it was misdiagnosed.** The H200 config diff between the working `math7b` run and the
failing `math7b-on` run showed exactly one functional difference, `group_routing.enabled`,
which pointed straight at the routing code. That was a coincidence: `step0m-on` had already
run 178 steps on the same box with the same field set. Config diffing found a difference,
not the cause.

**Guard.** Reap by PID from `nvidia-smi --query-compute-apps=pid`, not by pattern. See the
next finding for why patterns are worse than they look.

## 2026-08-31 — `pkill -f` self-matches the SSH command that carries it

**Measured, three times in one session.** A remote command of the form
`ssh host 'pkill -9 -f "areal" ; ...'` matches *its own* command line, because the pattern
is a substring of the argv of the shell running it. The shell dies mid-command, so the
statements after the `pkill` never run -- including the relaunch. Symptom: the tool returns
no output at all, and the box is left in whatever state the partial cleanup produced.

The bracket idiom (`rpc_serve[r]`) fixes the pattern itself but NOT the rest of the line:
`pgrep -f "supervis[e]\.sh"` still matched the literal `supervise.sh` appearing later in the
same command's relaunch half.

**Guard.** Never combine a kill and a launch in one remote command line. Ship the launch as
a script file (`scp` then `bash launch_x.sh`) so no pattern can match it. Extends
`finding_self_matching_pgrep_watcher`.

## 2026-08-31 — G=16 OOMs at the shipped KV-cache fraction on 80 GB cards

**Measured.** `gconfig.n_samples=16` with `train_dataset.batch_size=64` (rollout budget
matched to the G=8 baseline at 1024 sequences) died on the A100 in
`allocate_balanced_mbs_synced -> dist.all_gather_object` with
`ncclUnhandledCudaError ... Failed to CUDA calloc 6291456 bytes`. A 6 MB allocation
failing is a full card, not a fragmentation problem: `sglang.mem_fraction_static=0.8`
reserves 64 GB of each 80 GB board, and at G=16 the training side plus NCCL buffers no
longer fit in what is left.

`sglang.mem_fraction_static=0.55` with `rollout.max_concurrent_rollouts=128` reaches
`step 4/116` with all 8 GPUs at 85-100%.

**Consequence for the portable script.** `run_portable.sh` defaults `MEM_FRACTION=0.8`,
which is correct at G=8 and wrong at G=16. Anyone running `N_SAMPLES=16` on 80 GB cards
must pass `MEM_FRACTION=0.55`.

## 2026-08-31 — Hydra `+key=value` fails on a key that already exists

`+sglang.mem_fraction_static=0.55` exits rc=1 because the key is already in
`gsm8k_grpo.yaml`. The supervisor faithfully retried the same broken command. Use the bare
`sglang.mem_fraction_static=0.55` for an override, `+` only for a genuinely new key.

## 2026-08-31 — Correction: the group-routing guard is fully mutation-covered

An earlier run of `selfevo/tests/mutate_group_routing.py` reported 4/7 killed with two
survivors keyed on `silent * solved` and `silent * unsolved`. Re-run against the live repo
with the venv interpreter: **7/7 killed**. The survivor report came from a stale checkout
whose test file predated `test_routing_keys_on_silence_not_on_the_outcome`. Verified
independently by applying the `silent * solved -> solved` mutation by hand: that test fails,
as designed.

No code change was needed. Recorded because "the tests do not constrain this" was written
down once and was wrong -- a mutation harness is only as trustworthy as the checkout it
mutates, and it should be run against the same tree the tests import.

## 2026-08-31 — Weighted mixtures land, and four things an adversarial audit found in them

`RoutingDecision.weights` has always been a `Mapping[str, float]`, so a router could always
say "60% SFT, 40% RL". Nothing downstream could hear it: the actor called `.argmax()` and
the seam took `modes: list[str]`. `apply_mixtures` + `group_routing.decision` close that.
On response tokens, for normalised `{rl: a, sft: b, skip: c}`:

    new = a * original_advantage + b * (sft_weight * loss_mask) + c * 0

The pure cases are computed by CALLING `apply_decisions` for the extremes and blending, so
the reduction to today's behaviour is structural rather than an agreement between two copies
of the same arithmetic. An independent adversarial audit could not break it: 0 tensor-bit
mismatches over a specials grid (`+/-0.0, NaN, +/-inf, denormals, +/-3.4e38`), seven
spellings of each one-hot, four partitions, zero-length batches, and 4000 randomised trials
across fp32/fp64/fp16/bf16, compared as int views so `-0.0 != 0.0`. What it did break is
below, and all four are now fixed.

**1. Per-weight finiteness is not enough; the SUM overflows.** `_normalised_mixture` checked
each weight finite and then `if total <= 0`. `{rl: 1e308, sft: 1e308}` has two finite weights
and an infinite sum; every `w / inf` is `0.0`, so the mixture normalised to all-zeros, no
term was assembled, and the group trained as SKIP with `counts` logging no mass at all — a
mixture arm applying a decision nobody chose and reporting neither. Exactly the failure class
the guards were written to prevent, reached by a value the guards did not model. Fixed as
`if total <= 0 or not math.isfinite(total)`. **Guard:** when validating a reduction, validate
the reduction, not only its inputs.

**2. A bit-identity justification that was false, protecting code that was redundant.** The
pure-RL `continue` carried a comment saying it existed because `1.0 * x + 0.0` is not
identical to `x` for `x = -0.0`. The code never computed that expression — zero-weighted
terms were already dropped, so the alternative was exactly `1.0 * x`, which is
bit-preserving. The auditor deleted the short circuit and got 174 passed and 0 bit
mismatches. The line was quoted in a commit message as load-bearing; it was not. It IS
load-bearing, for a different reason found while fixing (3): `changed_rows`. **Guard:** a
justification in a comment is a claim. If no test fails when the code it defends is removed,
either the claim or the test is wrong — and a mutation harness that never mutates the line
will not tell you which.

**3. The reduction was bit-identical in the tensor and NOT in the statistic.** `changed_rows`
was diffed once at the end as `out != advantages`. `NaN != NaN` is True, so on a NaN batch a
pure-RL mixture counted every NaN row as reached while the argmax path — which never writes
an RL group and never compares it — counted zero: 700/700 divergences in the auditor's sweep.
The tensors matched throughout. Fixed by accumulating per group and BEFORE the write, which
is also what `apply_decisions` does. This is what makes the (2) short circuit real: a
written-but-unchanged NaN group would self-report as changed. **Guard:** "bit-identical"
covers everything the function returns. Ours returns `(tensor, stats)`, and only the tensor
was being checked.

**4. The argmax-credit coarsening corrupts the attribution instrument, optimistically.** It
was documented that a mixture arm "learns from a coarser signal than it acts on". Not
documented, and worse: two materially different mixtures sharing an argmax make a batch look
uniform, and `batch_outcomes` refuses the whole update as confounded — the router is starved
by decisions that genuinely differed. And two nearly identical mixtures with different
argmaxes report `dominant_share=0.5, weak_attribution=0.0`, i.e. MAXIMALLY attributable, for
groups that got almost the same gradient. The number whose job is to say how attributable an
update was errs in the direction that makes a run look better than it is, so it cannot be
used to discount a mixture result the way it discounts a `credit="batch"` one. Documented on
`GroupRoutingConfig.decision`; no code change, because splitting a scalar reward across a
mixture's components is not something we can currently measure.

**Also.** `apply_mixtures` built the all-SFT extreme eagerly and so RAISED where
`apply_decisions` succeeds (`sft_weight > 65504` overflows `full_like` on a float16 batch)
for mixtures that never read it; now built lazily. The actor normalised weights before
handing them over, so the overflow in (1) was reported as `{rl: 0.0, sft: 0.0}` — a
diagnosis naming weights no router emitted; it now forwards `decision.weights` raw and lets
the seam normalise once. `route/mixed_groups` was logged on the mixture branch only, so the
two arms did not emit the same key set; now logged on both.

**KNOWN GAP, and it is the important one.** No registered router emits a soft decision.
Measured over 200 randomised probes each (independently reproduced at 2000): `static`,
`solve_rate`, `cluster`, `coharness`, `random`, `contextual` all returned one-hot weights
0/200. `StaticRouter` accepts an arbitrary mapping, but `_route_groups` builds routers with
`factory()` and no kwargs, so nothing reaches them from config — which is why the `random`
router reads `SELFEVO_RANDOM_PROPORTIONS` from the environment instead. **So
`decision=mixture` today produces a run bit-identical to `decision=argmax`, with
`route/mixed_groups` at 0.0.** The plumbing is real, tested end to end through
`PPOActor._compute_advantages`, and mutation-covered; the router that would exercise it is
not written. `experiments/harness/preflight_group_routing.py` now prints the genuine-mixture
count per router, so this cannot be discovered after a GPU-week.

**Harness finding.** `mutate_group_mixture.py` claimed to print a digest check as its last
line on interrupt. It did not: `SystemExit` raised from the signal handler unwound past the
verification block, so the restore happened but was never proven — and the pytest child,
launched in its own session, kept running against a checkout being restored under it. The
handler now kills the child's process group, restores, prints the INTACT/MUTATED table, and
`os._exit`s. Extends the 2026-08-31 entry on killed mutation harnesses: it is not enough for
the restore to happen, the run has to be able to show that it did.
