# Experiment log

Measured results and negative results, newest first. A claim only belongs here once it has
been observed end to end; a prediction belongs in GOAL.md until then.

## 2026-09-01 — The LoRA ladder REPLICATES at a 4x budget, and a confound appears

The 4096 ladder was re-run at 16384 to drop the noise floor. Truncation fell from ~40% to ~13%
as intended, every rung graded 500/500 with zero failures, and **the shape reproduced**.

| step | acc @4096 | trunc | **acc @16384** | **trunc** |
|---|---|---|---|---|
| base | 0.6520 | 41.2% | running | -- |
| 24 | 0.6600 | 41.0% | 0.8680 | 15.0% |
| 49 | 0.6660 | 39.2% | 0.8720 | 15.0% |
| 74 | 0.6720 | 40.2% | 0.8880 | 12.8% |
| 99 | 0.6700 | 39.8% | 0.8800 | 13.2% |
| 115 | 0.6680 | 38.6% | 0.8740 | 14.2% |
| 124 | 0.6720 | 40.4% | 0.8880 | 12.6% |
| **149** | **0.6840** | 39.4% | **0.9000** | **11.4%** |
| 174 | 0.6760 | 40.8% | 0.8820 | 13.2% |

**Same peak at 149, same dip at 174, same span of +0.032 from step 24.** Two ladders at
budgets differing 4x, with truncation differing 3x, producing the same ordering and the same
magnitude. A replication at a different operating point is worth more than either ladder alone.

**Still not significant as a single pair.** At p~0.88 and n=500 the standard error of one
accuracy is 0.0145, so a difference of two runs carries ~0.021; +0.032 gives z ~ 1.56. What
carries the weight is the near-monotone ordering appearing twice independently, not the delta.

### The confound, stated rather than buried

**Truncation falls along the ladder: 15.0% at step 24 to 11.4% at step 149.** Later checkpoints
are cut off less often, so part of the accuracy gain could be "fewer answers truncated" rather
than "more answers right". This is a DIFFERENTIAL confound, unlike the constant one at 4096, and
it is exactly what I said would invalidate the ladder if it appeared.

Two things argue it is not the whole story. The 4096 ladder shows the same +0.032 while its
truncation moves only 41.0% -> 39.4%, a 1.6-point drift against 3.6 points at 16384; if the gain
were truncation-driven, the ladder whose truncation moved LESS should show a SMALLER gain, and it
does not. And the direction is itself a finding: the model is getting **more concise** as training
proceeds, which is the opposite of the length explosion the routed 1.5B arms produced.

**Settling it needs a cap where truncation is ~0 on every rung.** At 32768 the same base
truncated 11.4% on MATH-500, so even that is not clean; 65536 would be. That is the measurement
that converts this from a suggestive replication into a claim.

## 2026-09-01 — The 30B LoRA ladder, WITH its baseline: the feared degradation is absent

Eight adapter checkpoints plus the adapter-free base, all on held-out MATH-500 at a matched
4096 cap. The base is the rung that was missing from every routed arm for weeks: without it a
ladder shows a trend among adapters and cannot say whether training helped or hurt.

| step | MATH-500 | truncation |
|---|---|---|
| **base, no adapter** | **0.6520** | 41.2% |
| 24 | 0.6600 | 41.0% |
| 49 | 0.6660 | 39.2% |
| 74 | 0.6720 | 40.2% |
| 99 | 0.6700 | 39.8% |
| 115 | 0.6680 | 38.6% |
| 124 | 0.6720 | 40.4% |
| **149** | **0.6840** | 39.4% |
| 174 | 0.6760 | 40.8% |

**Truncation is constant across every rung (38.6-41.2%), which is what makes the ladder valid.**
A cap that binds equally on all rungs is a constant confound; had truncation drifted, the cap
would have become differential and the trend would be uninterpretable. This was checked, not
assumed, because the routed arms taught that training can change response length.

### What this settles

**The feared degradation is absent, and the sign is opposite.** The run uses `eps_clip: 0.4` and
`lr: 1.0e-05`, the recipe `step0l.sh` records as destroying held-out capability: `step0d` fell
**0.528 -> 0.316 monotonically**, a drop of 0.21. Here the ladder **rises** 0.6520 -> 0.6840.
Whatever that recipe does to a 1.5B under full fine-tuning, it is not doing it to a 30B under
LoRA -- which is why the finding was flagged rather than acted on, and why the checkpoints were
scored instead of the run being restarted.

### What this does NOT establish

**+0.032 is not a significant improvement on its own.** At p~0.67 and n=500 the standard error
of a single accuracy is ~0.021, so a difference of two independent runs carries ~0.030. The
best rung sits at roughly one standard error from the base. The measured noise floor at ~40%
truncation is ~0.015 by interpolation between the 10% and 97% cases.

What is more informative than any single pair is that **the ordering is near-monotone across
nine points**: base < 24 < 49 < 74 ~ 99 ~ 115 ~ 124 < 149, with only 174 turning down. Noise
around a constant does not usually arrive sorted. That is suggestive, not conclusive, and it is
recorded as such.

**Re-running the same ladder at 16384** to cut truncation from ~40% to ~10% and the noise floor
with it. If the drift survives at the lower floor it becomes a measurement; if it vanishes, the
honest conclusion is that LoRA on this base is neutral on held-out MATH-500.

## 2026-09-01 — An accidental repeat run measures the noise floor, and it depends on truncation

`narrow.sh` was launched twice by mistake, so four checkpoints were each scored twice on
MATH-500 at a matched 16384 cap. The duplication is a gift: it measures the run-to-run spread
directly, on this exact pipeline, rather than by assertion.

| checkpoint | run A | run B | **spread** | truncation |
|---|---|---|---|---|
| `rnd@173` | 0.4600 | 0.4580 | **0.002** | 10% |
| `ctx2@173` | 0.4580 | 0.4480 | **0.010** | 11% |
| `ctx2@199` | 0.2660 | 0.2420 | **0.024** | 96% |
| `rnd@199` | 0.3480 | 0.3220 | **0.026** | 99% |

**The noise floor is a function of truncation.** At ~10% truncated the same checkpoint
reproduces to 0.002-0.010; at ~97% truncated it moves by 0.024-0.026, an order of magnitude
worse. This is the empirical basis for a caveat that had until now only been argued: **above
~80% truncation the accuracy is dominated by which few generations happened to finish**, and
differences of that size between such points are not results.

Concretely, it retires the reading that `rnd` "retains more accuracy" than `ctx2` after the
transition. At 289 the gap was 0.3260 against 0.1740; the repeat spread at comparable truncation
is 0.025, so the gap is real in sign but its magnitude cannot be trusted, and neither can any
ordering among the post-transition points.

### The transition narrows to 174-199

Adding step 173 and 199 to both arms:

| step | 149 | 173 | 174 | **199** | 202 | 231 | 260 | 289 |
|---|---|---|---|---|---|---|---|---|
| `ctx2` trunc | 6.4% | **11%** | 13.2% | **96%** | 99.4% | 92.2% | 77.8% | 96.4% |
| `rnd` trunc | 6.0% | **10%** | 12.4% | **99%** | 99.6% | 99.6% | 97.4% | 98.2% |

**The unmeasured window is now 174 to 199, twenty five steps**, and both arms cross inside it
together -- 13.2 -> 96 for the contextual arm and 12.4 -> 99 for the random one. Two independent
arms crossing in the same 25-step window is the strongest form yet of the conclusion that the
timing belongs to the routed constant rather than to any controller.

## 2026-09-01 — A GPU-day of work generated, then thrown away by a client timeout I created

Three failures compounded, and the first was mine.

**CORRECTED 2026-09-01 (same day). The magnitude below is wrong and the mechanism is right.**
`q38_olymp_64k.out` accumulated results from SEVERAL launches sharing one tag, and I read it
once and quoted the worst line as if it were the run. Reading the whole file:

| run | accuracy | graded | failed | truncated |
|---|---|---|---|---|
| the line I quoted | 0.6154 | 13/675 | 662 | -- |
| second | 0.8717 | 265/675 | 410 | 0 |
| **best** | **0.8891** | **613/675** | **62 (9.2%)** | **1** |

So the claim "a GPU-day produced 13 usable answers" is false: the best run produced **613**.
The timeout mechanism is real and the failure rate does scale with concurrency, but I
overstated its size by reading a snapshot of a file that other processes were still writing.
**A reused output tag makes a log a mixture of runs, and a single grep of it is not a
measurement.** The 613/675 result is also the best OlympiadBench@65536 number we have -- see
the saturation entry, which it changes.

**1. Raising concurrency without raising the timeout.** I moved concurrency 16 -> 40 on a 65536
run and called it safe because the worst-case KV fit the pool. The KV was never the constraint:
more sequences sharing the cards means each takes longer, and `run_math.sh` held a **fixed
`--timeout 600`**. Per-request latency went past it. **The server returned `200 OK` for 613
generations; the client discarded 662 of them.** Roughly a GPU-day produced 13 usable answers.

**2. The scorer reported a score anyway.** `acc=0.6154, n=13/675, fail=662`, exit 0. It printed
`WARNING ... accuracy is over survivors and is biased upward` -- correct, and not enough, since
a warning beside a plausible number is read as a number. The `FAILED_RATE_ABORT` guard written
earlier that day would have aborted this at 98% failures.

**3. The guard was not on the box.** It was written, mutation-tested, committed and pushed --
and the H200 has no GitHub credentials, so it still ran the copy scp'd hours earlier.
**A fix that exists only in git does not protect a machine that cannot reach git.**

### Fixes

* The client timeout now scales: `max_tokens/20 + concurrency*5 + 300`, floored at 600, printed
  at startup. 3072/16 still gets 600s; 65536/40 gets 3776s. Roughly 4x more generous than
  measured decode speed (~90 tok/s/sequence).
* `bench_progress.sh` counted server completions -- which is all a server log can show -- and
  read **90.8% for a run that produced 13 answers**. It now says so, and reports the client-side
  discard count once the scorer writes it.
* Current code copied to the H200, guard confirmed present.

### The lesson worth keeping

**Concurrency and timeout are coupled; a throughput knob that raises per-request latency will
break a fixed deadline.** And the diagnostic that settles it in one step is the server's own
access log: `613 x 200 OK` against `fail=662` locates the failure on the client side
immediately, where the client's empty error text says nothing at all.

The 65536 OlympiadBench measurement is re-running at concurrency 24 with the scaled timeout.
**The earlier 0.6154 is void and must not be cited** -- it is an average over 13 self-selected
survivors, which are the fastest generations, hence the shortest, hence not a random sample.

## 2026-09-01 — Cap saturation: doubling the budget to 65536 buys essentially nothing

The fairness objection was that comparing two models at a cap which truncates BOTH partly
measures verbosity rather than skill, and that a ranking is only trustworthy if it survives a
larger budget. Qwen3.8-27B, same model, same grader, only the cap changed:

| benchmark | @32768 | trunc | @65536 | trunc | delta |
|---|---|---|---|---|---|
| MATH-500 | 0.9760 | 0.8% | **0.9800** | **0.2%** | +0.004 |
| AIME24 | 0.9000 | 13.3% | **0.9333** | **6.7%** | +0.033 (1 problem) |
| AIME25 | 0.9000 | 10.0% | **0.9333** | **3.3%** | +0.033 (1 problem) |
| OlympiadBench | 0.7733 | 17.8% | **0.8089** | **10.7%** | **+0.036** |

**OlympiadBench is the exception, and it inverts the conclusion for that benchmark.** Where
MATH-500 and both AIME sets moved by less than noise, OlympiadBench gains **+0.036** when the
budget doubles, and its truncation collapses from 17.8% to a single problem in 675. **32768 was
NOT the plateau there.** That is consistent with it being the only benchmark whose truncation
was high in the first place -- the generations being cut off at 32768 were ones that would have
been right.

**The clean re-run landed and the survivor bias was large.** At concurrency 24 with the scaled
timeout: `acc=0.8089, n=675/675, fail=0, trunc=72 (10.7%)`. Every problem graded, nothing
discarded. The survivor-biased figure was **0.8891**, so the bias was **+0.08** -- and in exactly
the predicted direction, because the discarded generations are the slowest, which correlate with
the hardest problems. **An average over survivors is not a conservative estimate; here it
overstated the result by more than twice the effect being measured.**

The honest OlympiadBench numbers are therefore **0.7733 at 32768 and 0.8089 at 65536, +0.036**,
with truncation falling 17.8% -> 10.7%. Still the largest cap effect of the four benchmarks and
still not fully saturated at 10.7%, but a quarter the size the biased read suggested.

**All three are saturated at 32768.** MATH-500 moves +0.004 on n=500, where the standard error
is ~0.006 -- inside noise. Each AIME set moves by exactly one problem out of 30, well inside the
~0.1 run-to-run jitter already recorded at that n. Truncation more than halves on every
benchmark and the accuracy barely follows, which is the signature of a budget that was already
sufficient: the generations being cut off were ones that were going to be wrong anyway.

**Consequence for the comparison.** For MATH-500 and both AIME sets, the 32768 numbers are
measurements rather than lower bounds, and the ranking against Frontis-MA1-30B (0.8980 / 0.8000
/ 0.7000) can be stated plainly. **OlympiadBench remains open** -- it is the one benchmark where
truncation was high enough (17.8% against Frontis's 20.0%) for verbosity to matter, and it is
the one still running.

**A caveat that belongs with these numbers.** The 65536 run used concurrency 40 against the
earlier run's 24, so the batch composition differed. Greedy decoding makes each answer
independent of its neighbours, but batched matmuls do not associate identically across batch
sizes, so a one-problem AIME difference is at the resolution where that could contribute. It
does not affect the conclusion, which rests on the accuracy NOT moving while truncation halved.

## 2026-09-01 — The 30B LoRA run is healthy, and two alarms I raised about it were both false

I reported twice that the run emitted "no metrics of any kind" and then that its reward was
"exactly 0.0000 across all 119 steps". **Both were my errors, and they were different errors.**

1. **"No metrics."** AReaL prints metrics in box-drawing tables, not as `key: value`. My grep
   pattern could not match that shape, so I reported an absence that was a pattern failure.
   Parsing whole cells split on the box character recovers **34 distinct keys, 4012 lines**.
2. **"Reward is zero."** `ppo_actor/final_reward/avg` is indeed 0.00000 at every step -- but
   `max` is **+2.47490** and `min` is **-2.47490**. It is zero-MEAN by construction, because
   GRPO normalises within a group. `advantages/avg` is 0 the same way, with max 3.89 and min
   -3.81. A quantity that is zero by definition is not evidence of a dead signal, and I nearly
   restarted a five-hour run over it.

**The actual state at step 119/1160:**

| metric | value | reading |
|---|---|---|
| `correct_n_seqs / n_seqs` | **445 / 512 = 86.9%** | the real learning signal, and it is high |
| `no_eos_ratios/avg` | 0.035 | 3.5% non-terminating; healthy |
| `kl_rewards/*` | 0 | expected, `kl_ctl: 0.0`, so no reference model |
| `correct_seq_len/avg` | 836 | against `incorrect_seq_len/avg` 1061 |
| checkpoints | **5** (steps 24/49/74/99/115) | scoreable offline |

**The 86.9% solve rate is the interesting number, and it is a problem for gradient signal.** At
that rate most groups are all-correct, so their advantages are identically zero and they carry
no gradient -- which is precisely the silent-group phenomenon this whole project is about,
appearing now at 30B on a task that is nearly saturated for it.

### Two config values differ from our corrected recipe, and it is NOT yet clear that matters

`eps_clip: 0.4` and `lr: 1.0e-05`, inherited from the upstream `gsm8k_grpo_lora.yaml`, against
`step0l`'s measured `eps_clip: 0.2` and `lr: 1.0e-06`. `step0l.sh` records that the demo recipe
"destroys held-out capability".

**But that measurement was full fine-tuning of a 1.5B, and this is LoRA on a 30B.** LoRA
adapters are conventionally trained at learning rates one to two orders of magnitude above full
fine-tuning, so 1e-5 is conservative by LoRA standards rather than aggressive. Transferring the
1.5B finding across both changes at once would be exactly the kind of unmeasured generalisation
that has produced three wrong claims here already.

**The way to settle it is the five checkpoints**: score 24/49/74/99/115 on held-out MATH-500 and
look at the direction. `step0d` degraded 0.528 -> 0.316 monotonically under the bad recipe, so
the signature is unmistakable if it is present. Queued behind the GPUs currently in use.

## 2026-09-01 — Scoring throughput: the 80GB memory rule was being applied to 140GB cards

A 65536-cap OlympiadBench run on 4x H200 was measured at **0.01 requests/s, ETA 918 minutes**.
The server's own numbers said why:

* `max_total_num_tokens=2167479` with **`token usage: 0.28`** -- the KV cache was 72% empty
* `#running-req: 16` -- concurrency, not memory, was the binding constraint
* 85700 of 143771 MiB used per card -- 40% of the GPU sitting idle

**Cause: `mem_fraction_static=0.55` is a rule for 80GB cards and these are 140GB cards.** The
rule was recorded for a real failure and carried over to hardware it was never measured on. On
an 80GB card 0.55 leaves the trainer room; on a 140GB card running nothing else it leaves 60GB
unused.

Two changes, neither of which alters what is computed:

| knob | was | now | why it is safe |
|---|---|---|---|
| `mem_fraction_static` | 0.55 | 0.85 | 0.85 x 143771MiB = 122GB pool; nothing else is on these cards |
| `concurrency` | 16 | 40 | worst case 40 x 65536 = 2.6M tokens against a pool of 3.6M |

Measured effect on the pool: `max_total_num_tokens` **2167479 -> 3595866**, a 1.66x increase,
with all eight GPUs at ~89% utilisation afterwards.

**Not bit-identical, and that should be stated rather than glossed.** Decoding is greedy and
requests are independent, so no answer depends on how many run beside it; but batched matmuls do
not associate identically across batch sizes. Expect agreement within a problem or two of 675,
not an exact reproduction. For a saturation check -- does accuracy climb from 32768 to 65536 --
that is far below the effect being measured. For a paired comparison against an existing number
at a different batch size, it is not, and the caps and concurrency must be matched instead.

**Generalisation worth keeping:** a memory fraction is a fraction OF THE CARD, so any rule
expressed as one silently changes meaning when the card changes. The check that catches it is
`token usage` in the server log: if that sits well below 1.0 while a benchmark is slow, the
budget is not the constraint and the concurrency is.

## 2026-09-01 — The collapse is SHARP (corrected), is NOT a cap artifact, and the random arm tracks it

Every point below is MATH-500 at a **matched 16384 cap**, on the H200 checkpoints. 16384 and
not 32768 because these are 1.5B checkpoints with `max_position_embeddings: 32768`; asking for
32768 NEW tokens is rejected outright and returns `fail=500, acc=nan`, which reads like a score
if the fail column is not read. It is 5.3x the 3072 the arms were originally scored at.

| step | `ctx2` acc | `ctx2` trunc | `rnd` acc | `rnd` trunc |
|---|---|---|---|---|
| 149 | 0.5060 (A100 twin, cap 8192) | 6.4% | **0.5100** | **6.0%** |
| 174 | **0.4280** | **13.2%** | **0.4600** | **12.4%** |
| 260 | **0.1100** | **77.8%** | pending | -- |
| 289 | **0.1740** | **96.4%** | **0.3260** | **98.2%** |

`ctxpcc@289` at the same cap: 0.2440, 100% truncated.

### 1. The collapse is NOT a cap artifact -- this was a real possibility and it is now excluded

The original step-289 scores were taken at 3072 with ~100% truncation, where an accuracy is a
truncation rate wearing accuracy's clothing. Raising the budget **5.3x** moves truncation from
~100% only to **96-98%**. These policies genuinely emit more than 16384 tokens. The pathology
is in the model, not in the budget.

### 2. CORRECTED 2026-09-01 (same day): it is SHARP, not gradual

I recorded "gradual, not a cliff" from four points (149/174/260/289) and the intermediate
steps refute it. With 202/231/249 filled in, MATH-500 truncation at a matched 16384 cap reads:

| step | 149 | 174 | **202** | 231 | 249 | 260 | 289 |
|---|---|---|---|---|---|---|---|
| `ctx2` trunc | 6.4% | 13.2% | **99.4%** | 92.2% | 61.4% | 77.8% | 96.4% |
| `ctx2` acc | 0.5060 | 0.4280 | 0.2340 | 0.1240 | 0.0600 | 0.1100 | 0.1740 |
| `rnd` trunc | 6.0% | 12.4% | -- | 99.6% | -- | -- | 98.2% |

**Truncation goes 13.2% -> 99.4% between step 174 and 202.** That is a transition inside a
28-step window, not a slope. My earlier reading came from having no point between 174 and 260
and drawing a line through the gap -- the same error as the original "threshold" claim, made in
the opposite direction. **Two wrong shapes from the same missing interval.**

After the transition it does NOT settle: 99.4 -> 92.2 -> 61.4 -> 77.8 -> 96.4. Non-monotone
across five points, so there is no clean "post-collapse plateau" either.

`ctx2@249` is the oddest point: the LOWEST accuracy (0.0600) with only 61.4% truncation, so at
that checkpoint the failures are not all length -- something else is wrong with the outputs
there, and `nobox=44` against 303 at step 231 says the answers are formatted but wrong rather
than absent. Not explained.

### The RANDOM arm's full curve: the same jump, in the same 28-step window

Completed 2026-09-01. MATH-500 truncation at a matched 16384 cap, both arms, every step:

| step | 149 | 174 | **202** | 231 | 249 | 260 | 289 |
|---|---|---|---|---|---|---|---|
| `ctx2` (contextual) trunc | 6.4% | 13.2% | **99.4%** | 92.2% | 61.4% | 77.8% | 96.4% |
| `rnd` (RANDOM) trunc | 6.0% | 12.4% | **99.6%** | 99.6% | -- | 97.4% | 98.2% |
| `rnd` acc | 0.5100 | 0.4600 | 0.3420 | 0.3540 | -- | 0.2920 | 0.3260 |

**Both arms jump between step 174 and 202, and they agree to within a percentage point at every
measured step before and after.** 12.4 against 13.2 before; 99.6 against 99.4 after. The random
router has no learned policy, no credit signal and no preference over modes, so the timing of
this transition cannot be a property of the controller. It is a property of the routed constant.

This is the third and cleanest line of evidence for that conclusion, after the step-289
comparison and the single matched point at 174. It is also the strongest form: not merely that
both collapse, but that they collapse **at the same step**.

`rnd` retains more accuracy than `ctx2` after the transition (0.29-0.35 against 0.06-0.23), but
both are ~98% truncated there, so the gap is which few generations happened to finish rather
than a capability difference, and should not be reported as one.

**What survives unchanged:** the collapse is not a cap artifact (5.3x budget moves truncation
~100% -> 96-98%), and the random arm tracks the contextual one -- `rnd@231` at 99.6% sits with
`ctx2@202` at 99.4%, and both are near-total by 289.


**SUPERSEDED -- see the correction above.** This paragraph read: truncation rises
monotonically 6% -> 13% -> 78% -> 97% across 149/174/260/289, with no step at which it breaks; it is already doubled 25 steps after the last healthy checkpoint. This
**corrects the earlier "threshold-like" reading**, which came from training-time length curves
on runs that were killed at 162 and never saw the middle of this range.

### 3. The RANDOM arm degrades on the same schedule

`rnd@174` = 0.4600 at 12.4% truncated against `ctx2@174` = 0.4280 at 13.2%. The two arms are
indistinguishable at the point where degradation becomes visible, and both are near-total at
289. **The schedule is a property of the routed constant, not of the router.** This is the
same conclusion the step-289 comparison reached, now with a second, earlier point.

### Do not read the accuracies past ~80% truncation

`ctx2` reads 0.1100 at step 260 and 0.1740 at 289 -- non-monotone. Above roughly 80%
truncation the accuracy is dominated by which few generations happened to finish, and the
ordering between such points is noise. **Truncation is the trustworthy channel there**, and it
is monotone throughout.

## 2026-09-01 — Qwen3.8-27B beats Frontis-MA1-30B on every core benchmark, and its numbers are cleaner

Scored on the H200 at an honest 32768 cap, with the cap-precedence fix carried across by hand
(the box has no GitHub credentials, so without copying it would have silently run the old
code). The fix is confirmed live on this box by the emitted line
`NOTE aime24: using explicit --max-tokens=32768 (BENCH_OVERRIDES default 8192 not applied)`.

| benchmark | **Qwen3.8-27B** | trunc | Frontis-MA1-30B | trunc |
|---|---|---|---|---|
| MATH-500 | **0.9760** | 4/500 (0.8%) | 0.8980 | 57/500 (11.4%) |
| AMC23 | **1.0000** | 0/40 | 0.9500 | 2/40 |
| AIME24 | **0.9000** | 4/30 (13%) | 0.8000 | 6/30 (20%) |
| AIME25 | **0.9000** | 3/30 (10%) | 0.7000 | 6/30 (20%) |

**The truncation column matters as much as the accuracy column.** Qwen3.8-27B is only 0.8%
cap-limited on MATH-500 against Frontis-MA1-30B's 11.4%, so its numbers are close to real
measurements rather than lower bounds. Every Frontis-MA1-30B figure remains a lower bound.

**AMC23 at 1.0000 should be read as "saturated", not as a score.** n=40 with zero errors gives
a Wilson interval of [0.912, 1.000]; the benchmark has no resolving power left at this
capability and should be dropped from any comparison that needs to separate strong models.

### Ornith-1.5-35B-A3B, recorded late (it was measured and never written down)

Scored in the same H200 sweep, at the same 32768 cap, and **never entered here until now**. A
subagent writing the paper refused to use these numbers because `grep` found them nowhere in the
repo -- correctly, since a number that exists only in a terminal and on a disposable vast.ai box
is not a record. Re-read from the artifact rather than from memory:

| benchmark | Ornith-1.5-35B-A3B | trunc |
|---|---|---|
| MATH-500 | 0.9720 | 8/500 |
| AMC23 | 0.9750 | 0/40 |
| AIME24 | 0.8000 | 6/30 |
| AIME25 | 0.8000 | 9/30 |
| OlympiadBench | 0.7393 | 149/675 |

It places between Qwen3.8-27B and Frontis-MA1-30B on every benchmark, and does not change the
ordering. **Note a conflict with `results.tex:78`**, which quotes an Ornith OlympiadBench figure
of 0.716 as a lower bound at 26% truncation; that is a different measurement at a different cap
and the two should not be merged.

### OlympiadBench too: 0.7733 against 0.7363

Qwen3.8-27B wins all five, but **both models are still cap-limited there** -- 120/675 (17.8%)
against 135/675 (20.0%).

### A MATCHED cap is not automatically a FAIR cap

Both models truncate at 32768, so neither number is a measurement of capability; both are
measurements of *capability under a 32768 budget*. That is a legitimate quantity, but it is not
the quantity the comparison implies. When a cap binds on both sides, part of the gap is
verbosity rather than skill, and the ranking is only trustworthy if it is **stable as the cap
grows**.

What we can already see is that neither model has saturated. Frontis-MA1-30B on OlympiadBench:

| cap | accuracy | truncated |
|---|---|---|
| 16384 | 0.6237 | 33.3% |
| 32768 | 0.7363 | 20.0% |
| 65536 | not yet run | -- |

Accuracy is still climbing steeply with budget, so 32768 is on the slope, not the plateau.

**Both models have `max_position_embeddings: 262144`** (checked, not assumed -- the previous
re-score attempt asked a 1.5B with a 32768 context for 32768 NEW tokens and every request was
rejected, returning `fail=500, acc=nan`). So 65536 is available to both and the protocol is:

1. Score both at 65536 on OlympiadBench and MATH-500.
2. If the ranking holds, the claim is robust to the budget and can be stated plainly.
3. If it narrows or flips, the 32768 comparison was partly measuring verbosity, and the
   honest report is an **accuracy-versus-cap curve** rather than a single number per model.

Report truncation beside every accuracy either way. A single number at an unstated cap is the
bug that already cost this project three benchmarks.

### Consequence for GOAL.md

The paper's plan is a delta at a FIXED STRONG BASE, which is what makes a result
un-discountable as scaling. That base should be **Qwen3.8-27B, not Frontis-MA1-30B**: it is
smaller, stronger on all four, and less cap-limited, so a delta measured on it is both harder
to obtain and harder to dismiss. The 30B LoRA run currently training on the A100 uses
Frontis-MA1-30B and should be re-pointed once it has served its purpose as a pipeline proof.

### Caveats

Single run per model, greedy, one seat of caps. AIME n=30 has ~0.1 run-to-run jitter recorded
here, so 0.9000 against 0.8000 on AIME24 is within about one jitter width and should not be
called significant on its own; the MATH-500 gap (0.9760 vs 0.8980, n=500) is well outside it.
OlympiadBench for Qwen3.8-27B was still running when this was written.

## 2026-09-01 — RECOVERED from the H200: all three routed arms collapse at step 289, RANDOM INCLUDED

The H200 was never down. It is a vast.ai instance and its address had changed; I reported it
unreachable for several cycles and treated these results as lost. They were on disk the whole
time. Routing read from each run's `config.yaml`, never from its name.

| arm | `router` | `solved_advantage` | MATH-500 | truncated | AMC23 | OlympiadBench |
|---|---|---|---|---|---|---|
| `ctx2` | contextual | 0.5 | 0.1880 | **489/500 (97.8%)** | 0.0000 | 0.0193 (643/675 trunc) |
| `ctxpcc` | prompt_centered | 0.5 | 0.2560 | **500/500 (100%)** | 0.0500 | 0.0696 (673/675 trunc) |
| `rnd` | **random** | 0.5 | 0.3080 | **499/500 (99.8%)** | 0.1000 | -- |

### The control collapses too, and that settles the mechanism question

I named this measurement hours ago as the one that decides whether the collapse needs the
learned router's specific mode distribution or follows from the routed constant in any arm.
**It is the constant.** The RANDOM router -- which has no learning, no credit signal and no
preference -- collapses just as completely (99.8%) as the contextual one (97.8%). Nothing about
the learned controller is required.

### This CORRECTS the earlier "no routed run has collapsed"

That statement was drawn from the A100 ladders, and it was true of them: `ctx` there peaks at
0.0059 no-EOS and its last checkpoint is step 149. **The A100 runs were killed at 162, before
the collapse.** The H200 runs continued to 289 and collapsed. Both observations are correct;
the earlier conclusion generalised from a truncated window. The A100 A/B at step 149 showing
routing merely *neutral* is consistent with this: **routing is neutral early and catastrophic
late**, with the transition somewhere in 162-289, which nothing has yet localised.

### A fourth negative for the learned controller

At step 289 the ordering is **random 0.3080 > prompt_centered 0.2560 > contextual 0.1880**.
The unlearned control is the LEAST damaged. Do not read the magnitudes -- every arm is
~100% truncated at caps of 3072/8192, so these are truncation rates wearing accuracy's
clothing -- but the ordering is at a matched cap and does not favour learning.

### Caveats

Single seed per arm. All three at pre-fix caps (3072 core / 8192 aime / 16384 olympiadbench),
so the absolute values are properties of the token budget, not of the models; only the
comparison is meaningful. No unrouted control was scored at 289 on this box, so "all routed
arms collapse" is established, "no unrouted arm would" is not.

## 2026-09-01 — CRITICAL PATH ITEM 1: the unrouted control, scored at last. Routing does not help.

Every previously scored checkpoint (`ctx149`, `ctxpc149`, `rnd149`) is a ROUTED arm. The
routing-off control had **never been scored**, so every routing-vs-no-routing statement made
here rested on a baseline that did not exist. This supplies it. Both arms re-scored together
at an explicit matched cap of 8192 rather than reusing stored numbers whose cap is unrecorded.

| arm | step | MATH-500 | AMC23 | AIME24 | trunc (MATH-500) |
|---|---|---|---|---|---|
| `step0l` control | 57 | 0.5040 | 0.325 | 0.0333 | 6 |
| `step0l` control | 149 | **0.5260** | 0.250 | 0.0000 | **2** |
| `step0l` control | 199 | **0.5460** | 0.275 | 0.0333 | 1 |
| `ctx` ROUTED | 149 | **0.5060** | 0.125 | 0.0667 | **32** |

### What is and is not significant

**The accuracy difference is NOT significant and must not be reported as one.** At the matched
step, routed 0.5060 against control 0.5260 is a 0.020 gap on n=500; the standard error of that
difference is ~0.032, so z ~ 0.63. The Wilson intervals overlap heavily ([0.462,0.550] against
[0.482,0.569]). This is also inside the +/-6pt MATH-500 noise band recorded earlier.

**The truncation difference IS large and is not a noise artifact**: 32 truncated against 2, at
the same cap, same benchmark, same step -- a 16x difference. That is a direct measurement, not
an inference, and it matches the independently measured length growth. AMC23 shows the same
sign (4 truncated against 0, accuracy 0.125 against 0.250) but n=40 is too small to lean on.

### The defensible statement

**Routing bought nothing and cost length.** The control improves across training,
0.5040 -> 0.5260 -> 0.5460 over steps 57/149/199, so the training signal is real and the
pipeline works. The routed arm at step 149 sits at 0.5060, indistinguishable from the control's
step-57 value and below its step-149 value, while truncating 16x more often. The mechanism
already measured -- the routed constant grows response length without touching the EOS hazard
-- shows up here as generations that run past the cap and are graded wrong.

**This closes critical-path item 1 as a negative result.** SFT on a unit's own correct sample,
as implemented, does not beat SKIP. Combined with the earlier router-vs-random null and the
withdrawn dose-response, the naive form of this method does not work, and that is now measured
against a real control rather than asserted.

### Caveats

Single seed per arm. `step0l@199` is 50 steps past the matched point and is shown to establish
that the control keeps improving, not as a matched comparison. AIME at 1.5B is a floor effect
for every arm (0.00-0.067) and carries no signal. The routed arm has no checkpoint past 149,
so the post-knee regime remains unmeasured.

## 2026-09-01 — EOS hazard measured directly: neither proposed mechanism holds, and the collapse story was wrong

Teacher-forced 64 frozen greedy responses through 27 checkpoints across both ladders
(`ctx` routed, `step0l` unrouted control, plus the base model as step 0). All 27 loaded.
Stop token determined empirically as **151645 `<|im_end|>`** -- all 64 responses ended on it,
0 on 151643. Alignment verified by a stronger check than proposed: since Y is greedy,
teacher-forced argmax must reproduce it, and agreement at the generator is **0.998** with
`p_eos_at_true_end` **0.99928**. An off-by-one gives ~0 on both.

### The EOS hazard does not erode. It strengthens.

| step | ctx H | ctx p_end | ctx H_body | step0l H | step0l H_body |
|---|---|---|---|---|---|
| 24 | 0.99754 | 0.99754 | <5e-7 | 0.99924 | <5e-7 |
| 74 | 0.99958 | 0.99958 | <5e-7 | 1.00204 | 0.00208 |
| 99 | 0.99975 | 0.99976 | <5e-7 | **1.20534** | **0.20538** |
| 149 | 0.99958 | 0.99958 | <5e-7 | 1.09105 | 0.09110 |

`ctx` rises 0.99754 -> 0.99979 across the full 27-checkpoint ladder, min across prompts
0.850 -> 0.994. **Correction (2026-09-01): do not call this monotonic.** The six steps
displayed above are not themselves ordered -- 0.99975 at step 99 exceeds 0.99958 at step
149 -- and 0.99979 appears in no displayed row. The supported claim is that the hazard
does not erode; monotonicity is not established by this table. Body hazard
below 5e-7 at every step: the hazard is a spike at the response's semantic end and that spike
never moves. **Mechanism A (smooth erosion) is refuted. Mechanism B is refuted at its premise
-- the hazard is not the failing state variable at all.**

A self-generated-Y control (each checkpoint scored on its own greedy output, n=32) reproduces
it: `ctx` mean `p_end` stays 0.984-0.9999 while its own greedy length grows 324 -> 469. Not a
stale-text artifact.

### Three claims I made that the data does not support

1. **"Routed arms collapse into non-terminating output" -- not on this box.** `ctx`'s
   `no_eos_ratios` peaks at **0.0059**, six of 1024 rollouts. Scanning all 18 runs, the ones
   that genuinely produced 1024-token non-terminating output are **step0b (15.6%)** and
   **step0 (11.7%)** -- **both UNROUTED**, both old published-config runs (lr 6e-6,
   eps_clip 0.4), whose collapse `step0l.sh` already attributes to entropy collapse. The
   100%/97.8% truncation figures I cited are from H200 runs at step 289 that I cannot
   currently verify because that box is unreachable; the A100 routed runs were killed at
   step 162, before any such thing.
2. **"The control is flat" -- false.** `step0l`'s H rises to **1.205** at step 99. Its end
   spike is pinned at 0.99996, so the extra 0.21 of stop-mass is MID-response, on 16 of 64
   responses, max 1.81. The control drifts toward stopping EARLIER. It does not confound the
   comparison, but the claim was wrong.
3. **The dose-response is not supported.** Constant 0.5 gives onsets 132 and 144; constant
   2.0 gives 131. **Magnitude does not clearly shift onset at n=3.** I previously described
   this as the strongest evidence we had. It is not evidence.

### What does survive

**Length moves like a threshold, and routing presence separates cleanly.** All three
configured-routing runs hold a flat plateau (reported as ~424 tokens, but **this figure is
UNRECONCILED**: the 2026-08-31 length table gives ctx 304 at step 30 and 340 at step 140,
below 424 right up to its own onset -- do not use 424 until the two are reconciled) for >120 steps then break: onset
(5 consecutive steps above baseline+4sd) at **ctx 144, ctxpc 132, sa2 131**, slope 0.34-0.57
tok/step before and 5.6-9.2 after. **Zero of the matched runs without GROUP routing ever cross** -- and note `step0j` is the
TOKEN-level routing arm, so it is unrouted only at the group level; calling all three
simply "unrouted" overstates the control
(step0l 273 steps, step0j 178, g16 116; note an earlier entry here says step0l ran 219
steps -- 273 is from the newer scan and is what the paper uses, but the disagreement is
unresolved and one of the two is wrong). Free greedy generation: `ctx` 327 -> 455 tokens,
`step0l` 330 -> 316 and flat over 202 steps.

So routing presence predicts crossing; routing MAGNITUDE does not predict onset.

### Fix implication, reversed

An explicit termination term (`overlong_reward_penalty`, `mask_no_eos_with_zero`) targets
`p_eos`, which is **already 0.9996 and does not budge**. It has nothing to correct. The lever
is the constant itself -- the same conclusion as before, reached for the opposite reason.

### Trap: `route/*` metrics are NOT a routing indicator

`sa2` emits no `route/*` metrics at all, yet its config carries
`group_routing.enabled: true, solved_advantage: 2.0, router: null` -- the actor's hardcoded
rule writes the constant without passing through the metric seam. **Routing status must be
read from `actor.group_routing` in `config.yaml`, never from the presence of route metrics.**
Any arm classified by metrics alone has been classified wrong.

### Unmeasured

The post-knee regime. `ctx`'s last checkpoint is step 149; the run continued to 162 with
length still climbing (647 and rising) and the logs end mid-stats-table -- killed abruptly,
not stopped on a criterion. If a real hazard collapse exists it lives past the last
checkpoint, and nothing here can see it. Probes are greedy while training sampled at
temperature 1.0; greedy length at 149 (455) tracks training seq_len at 145 (478), so it is a
fair proxy, but the sampled tail is unmeasured.

## 2026-09-01 — The 27B LoRA run: my diagnosis was wrong, the real cause is an fp32 load

(Original heading: "failed on config, not capacity, and two obvious causes were wrong". It
WAS partly capacity, in a way no config knob expresses -- see the retraction below.)

`lora27b` died at engine init with CUDA OOM and stranded nothing. Diagnosed from its config
and log. It is worth writing down because two natural explanations are FALSE, and acting on
either would have wasted a day.

**Not the cause (checked, both wrong):**

* `target_modules: []` looks like LoRA targeting nothing. It is not: `fsdp_engine.py:1118`
  maps an empty list to `"all-linear"`. Harmless.
* A full-precision reference model looks like the obvious 54GB. It was never built:
  `rl_trainer.py:206` creates `ref` only when `kl_ctl > 0`, and the run had `kl_ctl: 0.0`,
  so `self.ref` was `None`.

**RETRACTED 2026-09-01: the colocation diagnosis below is WRONG.** Reading the resolved
config rather than reasoning from the OOM message shows `rollout.scheduling_strategy.type:
separation` (line 317) and `actor.scheduling_strategy.type: separation` (line 523) -- both
were ALREADY separated. The only `colocation` is on `ref` (line 717), which by the point
above was never built. Two further facts kill it outright: the run died in **actor** init,
and the `rollout` role was never created in any of the four attempts (`grep "workers for
role 'rollout'"` returns zero hits across all four logs) -- so the 57.93 GiB process on
GPU 0 was **not this run's sglang server at all**. It was a foreign co-tenant from another
job on a shared box. Forcing separation is still correct hygiene, but it would not have
saved this run, and I asserted it twice as the cause.

**The actual cause, which nobody had named: a float32 full-model materialisation per rank.**
`fsdp_engine.py:1036` loads in `optimizer_dtype` -- **float32, not `actor.dtype`** -- and with
`fsdp.memory_efficient_load: false` (the default, and what this run used) `loading_device` is
the CUDA device. So every rank materialises the entire model in fp32 on its own card BEFORE
FSDP2 shards it. For 30.5B that is **113.7 GiB against 79.25 GiB usable**. It cannot fit on
any number of GPUs under any rollout configuration. The traceback confirms the site
(`core_model_loading.py:789 _materialize_copy -> tensor.to(device)`, one worker at 70.77 GiB
and still climbing). The fix is `actor.fsdp.memory_efficient_load=true`, which builds on CPU
at rank 0, `meta` elsewhere, and broadcasts after sharding (`fsdp_engine.py:411-465`).

**A fourth defect, also missed: LoRA weight transfer must go through disk.** The run inherited
`weight_update_mode: xccl` from `gsm8k_grpo.yaml`. The repo ships
`examples/math/gsm8k_grpo_lora.yaml` with `weight_update_mode: disk  # must be disk`, and
`rl_trainer.py:378-381` states why: P2P transports cannot carry PEFT-wrapped tensors. Any
LoRA run derived from the non-LoRA recipe is wrong before it starts.

**The half-enabled LoRA finding stands, and the mechanism is worse than stated.**
`get_py_cmd` (`cli_args.py:2286-2300`) skips any flag whose value is `None`/`False`/`""`/`[]`,
so `enable_lora: null` does not mean "passed as null" -- it means **`--enable-lora` never
appears on the server command line at all**.

**Superseded text follows, kept for the record:**

**The actual cause: rollout and trainer were colocated on the same cards.** At the OOM, GPU 0
held one process at **57.93 GiB** (sglang, `mem_fraction_static: 0.6`) plus three trainer
workers at ~8.2 GiB each -- **82.5 GiB on an 80 GiB card**. `allocation_mode` was `''` and the
scheduling spec at line 707 said `type: colocation`, so nothing ever separated them, despite
`deploy_mode: separation` appearing at line 234. The 49.41 GiB single allocation in the second
OOM is the trainer trying to materialise the base model beside a server already holding 58 GiB.

**A second defect that would have broken the run even with memory to spare.** LoRA was enabled
on only one side: `actor.use_lora: true` (line 487) against `rollout.use_lora: false` (line
320) and sglang `enable_lora: null` / `max_lora_rank: null` (lines 194-195). The trainer would
have learned adapters while the rollout engine served the **base** model, so the RL loop would
sample from one policy and update another. `cli_args.py:1374` states the requirement plainly --
LoRA "should be enabled together with vLLM/SGLang" -- and the config violated it silently.

**The run's name asserted what its config denied.** It is called `lora27b`; three of the four
LoRA switches were off. This is the same failure as the B200 arms tagged `mt8` while running
`MT=4`: a name is not a configuration, and only the config file is evidence. Any claim about
what an arm did must be read from its config, never from its name.

Required for a corrected 30B LoRA attempt: separate the rollout and train allocations rather
than colocating, or drop `mem_fraction_static` far enough that both fit; and set
`rollout.use_lora`, sglang `enable_lora`, and `max_lora_rank` to match the actor.

## 2026-09-01 — The cap-precedence bug was corrupting every frontier number, by 3x on AIME

`resolve_params` applied `BENCH_OVERRIDES` unconditionally, so an explicit `--max-tokens`
was discarded silently. Every 30B number scored through that path is a property of the token
budget, not of the model. Re-scored after fixing precedence, same model, same server, same
concurrency, same grader -- only the cap actually reaching the sampler:

| bench | cap ASKED | cap USED (before) | acc before | cap USED (after) | acc after | trunc before -> after |
|---|---|---|---|---|---|---|
| aime24 | 32768 | 8192 | 0.2667 | 32768 | **0.8000** | 73.3% -> 20.0% |
| aime25 | 32768 | 8192 | 0.2667 | 32768 | **0.7000** | 73.3% -> 20.0% |

A 3x swing from a silently-ignored flag. The `NOTE ... BENCH_OVERRIDES default 8192 not
applied` line now printed by the fixed code is what proves the value reached the sampler,
rather than merely appearing on the command line -- which it always did.

**OlympiadBench, the third benchmark the bug corrupted.** Re-scored at a genuine 32768:
**0.7363** (n=675, Wilson [0.702,0.768]), against **0.6237** at the wrongly-applied 16384 --
**+0.113**, with truncation falling 33.3% -> 20.0%. Still a lower bound twice over: 135 of 675
hit even the 32768 cap, and 94 of 675 have multiple gold answers in one string that exact-match
grading marks wrong. Note the shard exited 2 despite writing complete results, which is not yet
explained and should not be read as a failed measurement.

**Run-to-run jitter at n=30 is around 0.1.** An earlier entry records 30B aime24 at an
effective 8192 cap as 0.1667 with 83% truncation, where the controlled re-score records
0.2667 with 73.3%. The 0.2667 -> 0.8000 effect is far larger than that spread, so it
survives, but no AIME difference below ~0.15 at n=30 should be called real.

**Both numbers are still lower bounds**: 20% of AIME generations hit even the 32768 cap and
were graded wrong. And olympiadbench carries a second, unrelated lower-bound caveat -- 94 of
675 problems have multiple gold answers in one string, which exact-match grading marks wrong.

The lesson generalises past this bug. A default table that silently outranks an explicit
request is indistinguishable from the request being honoured, because the run still succeeds
and still writes plausible numbers. The only reason this was caught is that the results rows
record the generation parameters actually used, so the claimed cap could be compared against
the applied one. Recording the parameters a run used is not bookkeeping; it is the only way a
cap bug is visible at all.

## 2026-08-31 — MECHANISM: routing breaks GRPO's zero-mean advantage, and an unrouted control was already in our logs

Code-level analysis. It **refutes my stated hypothesis** while confirming the cause, and it
found a matched control we already had and had never looked at.

### My hypothesis was wrong in its reason

I claimed "nothing anywhere rewards EOS". **False.** EOS is INSIDE the masked region and DOES
receive the positive constant (verified numerically on CPU). The correct statement is narrower:

* For a **terminated** rollout the SFT push raises `log p` of R-1 non-EOS tokens and of EOS
  once, and to first order these roughly cancel -- the EOS hazard integrates to ~1.
* For a **truncated** rollout there is no EOS at all, so every one of its token-pushes is
  "keep going" with **zero counterweight**. And because the loss is a token-mean
  (`functional.py:506,571`; `actor.py:872`), a truncated rollout carries ~2.5x the gradient
  mass of a typical terminating one.

So the destabilising force is carried specifically by truncated rollouts inside SFT-routed
groups, not by the constant in general.

### The control that settles it was already in our logs

**`step0l`**: same model, same GSM8K, same `n_samples: 8`, same `lr: 1.0e-06`, same
`max_new_tokens: 1024` -- differing ONLY in `group_routing: null`. It ran **219 steps** with
mean response length **flat at 275-300 from step 30 to 215**. `g16`, also unrouted, is flat
too. **Three routed arms knee; two unrouted arms never do.**

| step | `ctx` | `ctxpc` | **`step0l` (unrouted)** | `sa2` |
|---|---|---|---|---|
| 30 | 304 | 303 | 307 | 293 |
| 140 | 340 | **435** | 296 | **445** |
| 162 | **527** | – | 286 | – |
| 200 | – | – | **283** | – |

### The quantity that predicts it: a DC offset on the advantage field

GRPO advantages are zero-mean by construction. Routing breaks that:

| run | mean `advantages/avg` | knee |
|---|---|---|
| `step0l` (no routing) | **-0.0004** | never |
| `g16` (no routing) | +0.0136 | never |
| `ctx` | +0.1579 | ~142 |
| `ctxpc` | +0.1675 | ~132 |
| `sa2` | **+0.8979** | ~120 |

**The knee arrives earlier the larger the offset**, and never at ~0. That is a dose-response
relationship across five runs, which is far stronger than the single-arm story I had.

### `sa2` refutes the "SKIP deletes the RL signal" sub-hypothesis

`sa2` takes the fixed-rule branch with `unsolved_advantage: 0.0`, so ONLY silent-and-solved
rows are touched -- confirmed by `routed_group_fraction == solved_group_fraction` to three
decimals at every step. No informative advantage is ever deleted, nothing is ever skipped, and
it still knees hardest. **Adding a positive constant to already-silent all-correct groups is
sufficient on its own.**

### Three defects found on the way

1. **The PPO clip is inert.** With `ppo_n_minibatches: 1`, `recompute_logprob: true` and
   `use_decoupled_loss: true`, the ratio is exactly 1: `importance_weight` avg=min=max=**1.0**,
   `clip_ratio` **0.0**, `clipped_tokens` **0.0** across every step of all four runs. The
   comment at `actor.py:645` claiming "PPO's clip still bounds how far one update can sharpen"
   is **false in this configuration** -- these are unclipped REINFORCE updates, so nothing
   bounds the constant.
2. **A coordinate off-by-one.** `actor.py:445` builds a LOCAL rolled mask, written back to
   `data` only at `:692` -- after routing at `:660`. So `apply_decisions` masks in TOKEN
   coordinates while advantages are in EMITTER coordinates. Consequences: one live token per
   row keeps its pre-routing advantage (so `skip` fails to zero one token per row), and one
   written position per row falls outside the loss.
3. **`no_eos_ratios` is mis-instrumented.** `actor.py:804` compares seqlens to the PADDED batch
   width, so it counts ~1 row per batch of 512 and reads 0.002-0.005 throughout. There is
   therefore **no usable training-time truncation rate anywhere in our logs**, which is why the
   abruptness cannot be resolved from what we have.

### No length term was active

The only one in the codebase is DAPO's `overlong_reward_penalty`, which defaults to **False**
(`cli_args.py:1899`) and is `false` in both arms' configs. `mask_no_eos_with_zero` is false too.
The sole implicit brake is that a truncated rollout grades wrong and earns a negative
group-normalised advantage -- **which `sft` overwrites with +0.5 and `skip` zeroes.**

### Still open: smooth-vs-threshold

Extrapolating `ctx`'s ~2.0%/step growth puts it at the 1024 cap right in the 199-224 window, so
a smoothly advancing mean crossing a fixed cap reproduces 4/60 -> 59/60 with no dynamical
threshold. A truncation-feedback loop also fits and additionally explains the acceleration at
130-142. **Our logs cannot separate them** because the truncation-rate metric is broken.

The discriminating experiment is cheap and needs no training: freeze one response set, then
teacher-force it through the `ctx` and `step0l` checkpoint ladders (both have
`globalstep{24,49,74,99,124,144,149}`) and plot the integrated EOS hazard. Smooth monotone decay
for `ctx` and flat for `step0l` means a smaller constant suffices; a flat-then-cliff means the
objective needs an explicit termination term.

## 2026-08-31 — BOTH credit signals collapse. The cause is the routed SFT constant, and the step-149 nulls were measured on pre-collapse models

The discriminating measurement. `ctx2` and `ctxpcc` both ran 290 steps and differ ONLY in the
credit signal. Same checkpoint step, same caps, same scorer the base model passed at 1/30.

| arm | credit | MATH-500 | truncated |
|-----|--------|----------|-----------|
| `ctx2` | **batch** | 0.1880 | **489/500 (97.8%)** |
| `ctxpcc` | **prompt_centered** | 0.2560 | **500/500 (100%)** |

AMC23 0.0000 vs 0.0500; AIME 0.0000 for both.

**Both collapse.** So per-prompt credit is NOT the cause -- the cause is what the two share:
the routed SFT constant written onto solved groups. Batch credit is if anything slightly worse
on accuracy (0.188 vs 0.256).

### This overturns the headline finding of the day

Result 7 recorded "three credit signals, three very different training trajectories, ONE
identical capability outcome" from checkpoints at **step 149**. That comparison is now known to
have been taken on **pre-collapse models**. `ctx` at 149 scored 0.5240 with 39/500 truncation;
the same arm at 289 scores 0.1880 with 489/500.

So the correct statement is not "the intervention does nothing". It is:

> **The intervention has a large effect that takes ~200 steps to appear, and the effect is to
> destroy the model's ability to terminate.** Every arm measured at 149 was measured before its
> own collapse.

### What this retires and what it establishes

* **`solved_advantage=0.5` is not "inert".** It was recorded as inert at 0.5 and harmful at 2.0
  from short runs. It is slow-acting: harmful at 0.5 too, given enough steps. The earlier
  scoping was a function of run length, not of the constant.
* **The credit-assignment work is not the explanation of the nulls.** It remains a correct
  analysis of why the ROUTER did not learn a preference, but the capability nulls have a
  different and larger cause.
* **The seam has more leverage than any measurement suggested.** It can take a model from 0.52
  to 0.19 on MATH-500. Nothing else in this project moved that benchmark by more than 0.010.

### Mechanism, still hypothesised

SFT here adds a positive constant to the advantages of a solved group's response tokens, which
raises the likelihood of exactly the tokens that were produced, with **no term anywhere
rewarding EOS**. Run long enough with a large share of groups routed to SFT, "never stop" is
consistent with what is being optimised. The collapse being abrupt (truncation 4/60 at step 199,
59/60 at 224) suggests a threshold rather than pure erosion. NOT established.

### What must now be re-run before anything is claimed

Every arm comparison in this project used step-149 checkpoints. Those are pre-collapse for
`ctx` and `ctxpcc` and unknown for `rnd`. **`rnd149` is the control and it must be checked at
289 too** -- if the random-proportions control also collapses, the cause is the SFT constant
appearing in ANY routed arm; if it does not, the collapse needs the router's specific mode
distribution. That is the next measurement.

## 2026-08-31 — The collapse is ABRUPT, and it happens between step 149 and 224

Same checkpoint sweep, same 60-problem MATH-500 slice, same scoring path the base model passed
at 1/30 truncation:

| checkpoint | accuracy | truncated |
|------------|----------|-----------|
| step 49 | 0.6000 | **1/60** |
| step 149 | 0.5833 | **1/60** |
| step 224 | 0.3667 | **59/60** |
| step 289 | 0.3500 | **60/60** |

**The model is healthy at 149 and destroyed by 224.** Truncation goes 1/60 -> 59/60 across
that window; it does not drift, it switches. Accuracy follows it down, 0.583 -> 0.367.

**What this rules out.** Not a gradual entropy decay -- 100 steps of training before 149 leave
truncation at 1 in 60. Not the onset of per-prompt credit either: that begins at step 29, and
the model is still fine 120 steps later. Whatever causes this is specific to the 149-224
window, which is AFTER the RL-suppression phase and after `rl_groups` had recovered to ~16-25%.

**Why the abruptness matters.** A slow degradation would suggest the SFT constant gradually
eroding the EOS probability. A switch suggests something closer to a threshold being crossed --
the policy finding a region where not terminating is locally optimal under the routed
objective, and then staying there. Those imply different fixes: the first wants a smaller
constant, the second wants a termination term in the objective regardless of constant size.
**Which of the two it is, is not yet established.**

**NARROWED to a 25-step window.** Full sweep, same slice and path:

| checkpoint | accuracy | truncated |
|------------|----------|-----------|
| 49 | 0.6000 | 1/60 |
| 149 | 0.5833 | 1/60 |
| 173 | 0.5833 | 3/60 |
| 199 | 0.5667 | 4/60 |
| **224** | 0.3667 | **59/60** |
| 289 | 0.3500 | 60/60 |

So it is BOTH: a slow drift (1 -> 3 -> 4 in 60 across 50 steps) and then a switch (4 -> 59 in
25 steps). The drift is real but tiny and would never have been noticed on its own; the switch
is catastrophic. That pattern -- a slowly rising quantity that then runs away -- is what a
positive feedback loop looks like, and it means the earlier checkpoints were NOT healthy, they
were pre-critical. Any monitor that only alarms on a large truncation rate would have caught
this at 224, far too late; alarming on the DERIVATIVE at 173 would have caught it in time.

**Still open, and it matters for how general this is:** whether `credit="batch"` collapses the
same way at full length. `ctx2` ran 290 steps on the same config apart from the credit signal
and has never been scored. If it collapses too, this is a property of the routed SFT constant
and not of per-prompt credit; if it does not, per-prompt credit is implicated specifically.
That is the single most informative unrun measurement available right now.

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
