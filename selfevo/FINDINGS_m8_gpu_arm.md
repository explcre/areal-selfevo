# M8: the learned meta-controller reaches a GPU --- what had to be fixed first

2026-09-02/03. Written here rather than in `EXPERIMENTS.md` because that file is held by
another agent with uncommitted work; fold it in from here. Numbers from the arms themselves
go to `paper_src/results.tex`, which is where results live.

GOAL.md lists M8 as the top item on the critical path with two gaps: **not reachable from
config**, and **no arm has ever trained with it**. Both are addressed here. Everything runs on
the 8xB200 GCP box (`b200x8-keep-us-east1-b`, SPOT); the A0 baseline on the 4xH100 was not
touched and advanced from step 745 to 852 across this session.

### 1. The config gap, closed (commit on `selfevo/a100`)

`credit="prompt_self_baseline"` is now selectable, and `_route_groups` honours it by building
`PromptCreditLedger(baseline="self_mean")`. Two further seams landed with it because the arm
cannot be MEASURED without them:

* **`credit_shuffle_seed`** — the correspondence control. Same ledger, same pairings, same
  multiset of credit values, only the prompt-to-credit correspondence destroyed. **Refused on
  `credit="batch"`**, where every unit already holds the identical scalar: a control that
  cannot fail is not a control, which is the rule `credit_sim.simulate` already applies.
* **`decision_trace_path`** — one JSON record per routed group per step (mode + the seven
  observability features) plus one per credited decision. Subset contrast needs each unit's
  mode beside the features that define the subsets; `route/{mode}_groups` is an aggregate.
  This is precisely why the batch-credited arm's null had to be re-derived in simulation
  rather than read from its own 129 steps.

**Evidence.** 16 new tests through the REAL `_compute_advantages`;
`selfevo/tests/mutate_prompt_credit_wired.py` extended by six mutations and re-run:
**14/14 killed** (11/14 before the sixth test was added — the survivor was "the shuffle fires
on every arm", which only a test pinning the UNSHUFFLED arm to exact credit values can kill; a
test that merely compares the two arms passes when both shuffle).
**No-op proof:** the whole `selfevo` suite on the patched and unpatched trees gives **50 failed
/ 9 errors on both** (all in other agents' in-flight work: cluster_lora, gold,
harness_route_trainer, loss_hook), with 2024 passing against 2009 — exactly the 15 new tests at
that point and no changed outcome.

### 2. THE FINDING THAT WOULD HAVE KILLED THE ARM: per-prompt credit needs prompt RECURRENCE

Per-prompt credit pairs a decision with the **next sighting of the same prompt**, so its
recurrence period is `n_prompts / batch_size` **steps**.

    A0's own configuration: 102,835 DeepMath rows / batch 8 = 12,853 steps per epoch.

A0 is at step 852 of a 12,853-step epoch. **An M8 arm launched on that corpus at that batch
size would have made ~5,000 routing decisions and received ZERO credits** — and every metric
except `prompt_credit/credited` would have reported it healthy. That is not a null result, it
is a no-op, and it is the exact failure mode the `credit_sim` study could not see, because the
simulator's world recurs by construction (its live reference was GSM8K at ~29 steps per epoch).

**Consequence, applied:** the M8 arms train on `deepmath_decontam_512`, a seeded 512-row subset
of the same decontaminated corpus (`SUBSET.json`, seed 20260902, sha256 recorded), giving a
**64-step epoch** — confirmed live: `Epoch 1/10 Step 1/64`. Each prompt is seen ~10 times over
the run, which is the regime the fix was validated in. The corpus size is a **consequence of
the recurrence requirement, not a tuned hyper-parameter**, and it is a limitation to state:
per-prompt credit is unavailable on a corpus the run does not revisit.

**Second-order consequence, also recorded:** under `baseline="self_mean"` a prompt's FIRST
delta is withheld, so the treatment arm receives its first credit at roughly the THIRD
sighting (~step 128), whereas the batch-credited arm is credited from step 2. The per-prompt
arm therefore pays a warm-up of ~2 epochs in exchange for a signal that is not shared across
arms. Any comparison over a short window flatters the batch arm.

### 3. Stack findings on Blackwell (sm_100), for whoever rebuilds this box

* **sglang's default `attention_backend="fa3"` cannot run here.** `FlashAttention v3 Backend
  requires SM>=80 and SM<=90`; the rollout server dies with SIGKILL *before becoming healthy*,
  and the trainer reports only `exited with code -9`. Fix: `++sglang.attention_backend=flashinfer`
  (a `+` single-override fails — the key is not in the struct).
* **Python 3.12, not 3.10.** The H100's venv is 3.10 and works there; on sm_100 the serving
  path forces FlashInfer, which raises a *TypeError* (not ImportError) on 3.10. The whole
  H100 pin set installs unchanged on 3.12 from its own `pip_freeze_full.txt` with
  `uv pip install --no-deps` — `--no-deps` is required, because the areal dependency set is
  internally unsatisfiable on `openai` (litellm needs >=2.8.0, sglang demands ==2.6.1).
* **A 32B fp32 actor fits ONE B200 unsharded.** `fsdp:d1p1t1` + `sglang:d1p1t1` = **2 GPUs per
  arm**, measured peak **133.7 GB of 178.35** at the PPO update, so four arms run concurrently
  on one box. This is not only a saving: at `d2` the actor path runs in two workers, each with
  its own router and its own ledger seeing half of every batch, and this experiment is about
  what ONE router learns. Confirmed by the trace: a single `trace.pid*.jsonl` shard.
* **`reap` must kill process GROUPS.** AReaL's data-service workers are python children of a
  bash wrapper and only the wrapper's command line carries the run's log path; killing by
  pattern left three orphaned data services registered and holding ports across two launches.

### 4. The one-sample end-to-end pass (`m8_smoke`, `R_EXIT=0`)

Six steps on a 16-row corpus deliberately sized so an epoch is two steps and the credit path
must fire. It proved, in order: 8 router decisions per batch from the contextual router
(`route/rl_groups` 3, `sft` 3, `skip` 2 at step 0); the trace written and read back (60
records = 48 decisions + 12 credits); **per-prompt credit assignment firing on GPU** at step 4,
each record carrying `credited_step` and the `unit_id` that earned it (e.g.
`{"credited_step": 3, "unit_id": "3:1", "mode": "sft", "value": -0.25}`); the self-baseline
withholding first deltas as designed (`cold_baseline_skips` 0 -> 4 -> 12 before the first
credit); the optimiser step (`Memory-Usage ppo update: 133.70/178.35 GB`); and a checkpoint
that reloads with **512/512 tensors changed, all finite, max |delta| 5.11e-4**.

Two dead-on-arrival failures were caught here rather than after a long run: the fa3 backend
above, and `random.Random((seed, step))` — a tuple seed, which Python 3.11+ refuses with a
TypeError. The second would have crashed the correspondence control on its first real step.

### 5. The measurement instrument, validated against a null AND a positive control

`m8_analysis.py` computes subset contrast (half the total variation distance between two
subsets' mode distributions) with a per-subset targeting fraction, bootstrapped over STEPS
rather than groups, and always as a PAIRED DIFFERENCE against a control. Two labellings:
`silence` (the fixed rule's own partition — the floor, not the finding) and `covariate` (a
median split on one covariate WITHIN the informative stratum, so the pass count is held and any
contrast is attributable to something k does not carry — GOAL.md M9's untested axis).

Driven through the real `_compute_advantages` for 200 steps with two hand-built routers
(`trace_selftest.py`): a **feature-blind** router that draws modes from fixed proportions, and
a **sighted** router that reads `mean_logprob` and nothing else.

| arm | contrast on `mean_logprob` | on `logprob_dispersion` | silence contrast |
|---|---|---|---|
| sighted (reads `mean_logprob`) | **0.9882** [0.8313, 1.0000] | 0.2071 | 0.0098 |
| blind (reads nothing) | 0.0616 [0.0200, 0.2278] | 0.0440 | 0.0528 |
| paired difference | **+0.9266** [+0.6700, +0.9542] | +0.1632 | -0.0430 |

So the statistic fires at 0.99 on the covariate a router actually used, reports 0.06 for one
that used nothing, and does not mistake a router that ignores the silence sides for one that
separates them. A covariate that is constant in a window is reported **undefined**, never 0.

### 6. Arms launched

All four differ from each other on exactly ONE axis and share the corpus, the seed, the batch
shape (8 prompts x 8 samples), the cap (1024 new tokens), `lr=1e-4`, `kl_ctl=0`, `adv_norm=null`
and LoRA r=32 on q/k/v/o — the A0 configuration, changed only where the box forced it.

| arm | GPUs | router | credit | why it exists |
|---|---|---|---|---|
| `m8_prompt` | 0,1 | contextual | `prompt_self_baseline` | the treatment |
| `m8_control` | 6,7 | random | (inert) | **the mandatory control**: rate-matched at m8_prompt's MEASURED proportions |
| `m8_shuffle` | 4,5 | contextual | `prompt_self_baseline` + `credit_shuffle_seed` | the correspondence control |
| `m8_batch` | 2,3 | contextual | `batch` | the recorded null, re-run beside its own fix |
| `r_rule` | 6,7 | rule | (inert) | the hand-written rule GOAL.md M8 asks the learned router to beat |

`preflight_r.py` pins the router, the credit rule, the shuffle and the trace to the run's NAME,
so a mislabelled arm is refused rather than reported.

### 7. Concurrent arms on one host: port selection replays the experiment seed

Launching a fourth arm failed twice --- once for the rule arm, once for the matched random
control --- with

    ValueError: Could only find 0 free ports out of 1 requested after 10 attempts

and never at two or three arms, which is why it first looked like load rather than
determinism. The trainer then died again in teardown with `AttributeError: 'PPOTrainer' object
has no attribute 'saver'`, which is the message that actually reaches the log and says nothing
about ports.

**Cause, reproduced directly rather than argued.** `find_free_ports` drew from the GLOBAL
`random` module, and `seeding.set_random_seed(config.seed, key=...)` has already seeded that
module by the time a port is allocated. Two launches with the same `config.seed` --- which is
exactly what a paired treatment/control pair must have --- draw the SAME candidate sequence:

    set_random_seed(1, key="trainer0"); [random.randint(10000, 32767) for _ in range(10)]
      -> [18349, 13413, 26594, 17672, 22794, 19872, 17909, 19968, 16866, 11717]
      -> is_port_free: [False] * 10

13413 and 17672 were the ports the first arm's `rpc.guard` and `rpc_server` were holding, read
straight out of `ps`. With `max_attempts = count * 10` the fourth arm has nothing left to try.

**Fix** (committed): ports come from a private `random.SystemRandom`. A port is not part of any
result, and binding it to the experiment seed couples runs that must be independent. Nothing
else in the module consumes randomness, so every arm still reproduces bit for bit.
`selfevo/tests/test_port_allocation_is_not_seeded.py` pins it and **fails on the unpatched
module** --- verified on a clean copy, where both calls returned `[25915, 27641, 29763, 31093]`.
With the fix the fourth arm launched on the same GPUs that had just refused it.

### 8. The arms are NOT per-step paired, and the analysis must not claim they are

Rollout generation is not deterministic across sglang servers, so two arms with the same
`config.seed` and the same data order still see different completions from step one, hence
different observability features and different routing contexts. Measured directly: over the
first ~40 steps `m8_prompt` and `m8_shuffle` --- which are byte-identical in configuration until
the first credit arrives at ~step 128 --- already have different mode mixes.

Consequences applied to the analysis: the two arms are resampled INDEPENDENTLY in the bootstrap
(a two-sample difference, not a paired one), and every comparison is made over the same STEP
WINDOW rather than the same wall-clock window, because a router's contexts come from the policy
at that step and the arms were launched at different times. `m8_analysis.py --window LO:HI`.
