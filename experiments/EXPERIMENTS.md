
## 2026-08-28 - Step 0 rerun (step0b), and two findings that killed a planned fix

### Finding: `gconfig.min_new_tokens` is a DEAD FIELD in AReaL
Declared at `areal/api/cli_args.py:173` and listed as OpenAI-unsupported at :305, but
**never read anywhere else in the repo**:
    grep -rn "min_new_tokens" --include=*.py .   # -> only those two lines
I had planned to set `min_new_tokens=1` to prevent a degenerate all-EOS generation.
That fix would have been completely inert. Always grep for a config fields *reads*

## 2026-08-28 — Step 0 rerun (step0b), and two findings that killed a planned fix

### Finding: `gconfig.min_new_tokens` is a DEAD FIELD in AReaL
Declared at `areal/api/cli_args.py:173` and listed as OpenAI-unsupported at :305, but
**never read anywhere else in the repo**:

    grep -rn "min_new_tokens" --include=*.py .   # -> only those two lines

I had planned to set `min_new_tokens=1` to prevent a degenerate all-EOS generation from
500-ing the sglang server. That fix would have been completely inert. Always grep for a
config field's *reads* outside its own schema before building a run on it.

### Finding: the flash-attn wheel is ABI-incompatible with torch 2.9.1

    ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
    _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb

`STEP0_FAILURE.md` claimed the fix for the Step 0 stall was "install flash-attn and stop
deviating from the reference config". That was wrong on the merits: the stall was in
generation/retry handling, which `attn_impl` does not touch. sdpa stays, and the deviation
is documented in `step0b.sh` rather than mistaken for the bug.

### Evidence loss
The original 3.6 GB `step0.log` is now 0 bytes; only a 2 MB head survives. Grepping that
head for errors yields only false matches ("5.000e+00" contains the substring "500"). It
shows PPO metric tables through `step 51/233`, i.e. training *was* stepping. The stall
diagnosis therefore rests on evidence that no longer exists, so step0b re-derives the
failure instead of assuming it.

### step0b guards
- `experiments/harness/logfilter.py` collapses repeated log signatures (first 200 verbatim,
  then a periodic tally). A retry storm now costs O(1) disk instead of the ~3.46M lines that
  destroyed the last run's evidence.
- `experiments/harness/watchdog.sh` samples the `step N/233` counter TWICE, 1800s apart, and
  kills only after two consecutive no-progress strikes. Sampling a monotonic counter once
  cannot distinguish progress from a stall — that is how the last run was mis-reported as
  healthy. It kills by recorded PGID, never by pgrep pattern: a pattern kill previously
  matched the watcher's own command line and killed the controlling SSH session.
- `gconfig.max_new_tokens` is no longer overridden to 512 (that truncated chain-of-thought
  and read out as a 0 solve rate); the published 1024 stands.

## 2026-08-28 — Step 0 rerun (step0b), and two findings that killed a planned fix

### Finding: `gconfig.min_new_tokens` is a DEAD FIELD in AReaL
Declared at `areal/api/cli_args.py:173` and listed as OpenAI-unsupported at :305, but
**never read anywhere else in the repo**:

    grep -rn "min_new_tokens" --include=*.py .   # -> only those two lines

I had planned to set `min_new_tokens=1` to prevent a degenerate all-EOS generation from
500-ing the sglang server. That fix would have been completely inert. Always grep for a
config field's *reads* outside its own schema before building a run on it.

### Finding: the flash-attn wheel is ABI-incompatible with torch 2.9.1

    ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
    _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb

`STEP0_FAILURE.md` claimed the fix for the Step 0 stall was "install flash-attn and stop
deviating from the reference config". That was wrong on the merits: the stall was in
generation/retry handling, which `attn_impl` does not touch. sdpa stays, and the deviation
is documented in `step0b.sh` rather than mistaken for the bug.

### Evidence loss
The original 3.6 GB `step0.log` is now 0 bytes; only a 2 MB head survives. Grepping that
head for errors yields only false matches ("5.000e+00" contains the substring "500"). It
shows PPO metric tables through `step 51/233`, i.e. training *was* stepping. The stall
diagnosis therefore rests on evidence that no longer exists, so step0b re-derives the
failure instead of assuming it.

### step0b guards
- `experiments/harness/logfilter.py` collapses repeated log signatures (first 200 verbatim,
  then a periodic tally). A retry storm now costs O(1) disk instead of the ~3.46M lines that
  destroyed the last run's evidence.
- `experiments/harness/watchdog.sh` samples the `step N/233` counter TWICE, 1800s apart, and
  kills only after two consecutive no-progress strikes. Sampling a monotonic counter once
  cannot distinguish progress from a stall — that is how the last run was mis-reported as
  healthy. It kills by recorded PGID, never by pgrep pattern: a pattern kill previously
  matched the watcher's own command line and killed the controlling SSH session.
- `gconfig.max_new_tokens` is no longer overridden to 512 (that truncated chain-of-thought
  and read out as a 0 solve rate); the published 1024 stands.

### Two design gaps in the reference config, found while step0b ran

**1. Near-saturated trunk.** `task_reward/avg` on Qwen2.5-1.5B-Instruct starts around
**0.76** and moves to **0.790** by step 80 (first-quarter vs last-quarter mean, delta
+0.029) with per-step swings of 0.67-0.85. A task the base model already solves ~76% of
the time leaves very little headroom to demonstrate any method. GSM8K with a 1.5B
instruct model is a plumbing check, not an experimental substrate. Any claim of a method
gain measured here would sit inside the step-to-step noise band.

**2. One eval point, no curve.** `evaluator.freq_epochs: 1` with `total_train_epochs=1`
fires the evaluator exactly once, at the end. There is no pre-training baseline and no
validation curve, so "did it improve" cannot be answered from held-out data — only from
train reward on shifting data, which is not the same question. The `valid_*` keys in the
log (`valid_tokens`, `valid_generated_tokens`, `valid_tokens_in_loss`) are FSDP token
counts, not accuracy.

**Action for every subsequent run:** set `evaluator.freq_steps` (e.g. 20) to get a real
validation curve, and evaluate the base model on the same harness with the same flags
before training so there is a floor to compare against.

### RESOLVED with evidence: the Step 0 failure was our own `max_new_tokens=512` override

**Correction to the "evidence loss" note above.** The evidence was never lost. `~/step0.log`
was 0 bytes, but AReaL writes its own per-worker copies under
`~/areal-runs/logs/ubuntu/step0/t1/` — `merged.log` (4.2 GB) and `rollout.log` (3.6 GB)
were intact the whole time. I retracted a correct diagnosis because I looked in the wrong
file. Check the framework's own log directory before declaring evidence gone.

**Root cause**, from `rollout.log`:

    openai.InternalServerError: Error code: 500 - {'detail': 'All output_tokens are EOS or
    PAD tokens; cannot strip stop tokens without removing entire output.'}

Counts in the failed run: **3,468,076** occurrences of that error; 79,444
`OpenAIProxyWorkflow WARNING: Agent task failed`; 20,503 `RemoteInfEngine ... Workflow
execution failed`. sglang 500s when a generation is entirely EOS/PAD because stop-token
stripping would empty the output; AReaL's proxy workflow then retried without bound, which
produced both the stall and the 3.6 GB log.

**Controlled comparison.** step0b is the same config with the `max_new_tokens=512` override
removed (published value 1024 restored). At step 120/233, across all eight worker logs:

| run | all-EOS 500s | `Agent task failed` |
|---|---|---|
| step0 (max_new_tokens=512) | 3,468,076 | 79,444 |
| step0b (published 1024) | **0** | **0** |

So the degenerate all-EOS generations were induced by truncating chain-of-thought at 512
tokens. The failure was ours, not AReaL's. Note the earlier `STEP0_FAILURE.md` advice
("stop deviating from the reference config") was right for the wrong reason: it blamed the
missing flash-attn, which is unrelated, while the harmful deviation was the token cap.

**Residual risk.** step0b avoids the trigger; it does not make AReaL robust to it. An
all-EOS generation from any other cause would still retry without bound. The principled fix
is a bounded retry in the proxy workflow, since `gconfig.min_new_tokens` — the obvious knob
— is a dead field (see above). Treat unbounded retry as a live upstream fragility.

### FINDING: AReaL scores infrastructure failures as reward 0.0 (silent-zero path)

`areal/v2/inference_service/controller/workflow.py:207-231`. The rollout member wraps the
whole trajectory in `except Exception as exc:` and then:

    await self._set_last_reward(http_session, 0.0, session_api_key)
    return None

So the mechanism is NOT "unbounded retry" as I first wrote — the log line is *"This
trajectory will be rejected"*, and the trajectory is dropped. But before dropping it, the
handler **assigns reward 0.0**. Any failure — a sglang 500, a connection drop, a tool
error, a parse failure — is indistinguishable from a genuinely wrong answer in the reward
stream.

Two consequences, both bad, and both silent:

1. **Reward contamination.** In the failed step0 run, 3,468,076 all-EOS 500s each landed as
   a 0.0. A downward-biased reward curve looks like "the model got worse" when the true
   cause is "the harness broke." This is exactly why that run read out a ~0 solve rate.
2. **Batch starvation.** `return None` drops the trajectory. If a persistent condition makes
   most generations fail, the batch never fills and the run stalls with no error raised —
   the observed "stalled at 159/233".

**Why this matters for a self-evolving agent specifically.** Our method deliberately
generates its own tasks and data, so degenerate and malformed generations are *expected*,
not exceptional. A reward channel that silently encodes "infrastructure broke" as "score 0"
will teach the meta-policy to avoid whatever trips the harness rather than whatever is
actually hard. That is a confound sitting directly on the quantity we intend to measure.

**Required instrumentation before any self-evolving run:** count failures by exception type
per step, log the rejected-trajectory fraction as a first-class metric, and separate
"failed" from "scored 0" in the reward aggregation. A run whose rejection rate exceeds a
threshold should abort loudly rather than train on contaminated zeros.

### Related config: `mask_no_eos_with_zero` conflates "out of budget" with "wrong"

`cli_args.py:1999` — *"Mask truncated generations (no EOS token) and exclude from
training"*, `default=False`. Applied in `trainer/ppo/actor.py:262`. In step0b it logs as
**0.0 (False)**, i.e. inactive.

With it False, a generation that hits `max_new_tokens` without emitting EOS is kept and
scored by the reward function — which sees no final answer and returns 0. So a truncated
CoT trains as a *wrong answer* rather than being excluded as an incomplete sample. Under
our `max_new_tokens=512` override that turned every over-long CoT into a reward-0 gradient
signal, independently of the all-EOS 500s.

This is the same silent-zero family as the exception handler above, and it shows up in the
emitted metrics: AReaL logs `ppo_actor/correct_n_seqs` and `ppo_actor/incorrect_n_seqs`,
with no third bucket. Truncated, failed, and genuinely-wrong sequences all land in
`incorrect_n_seqs` and are indistinguishable there.

Useful seam, already present: `areal/utils/perf_tracer.py:467` carries a per-trajectory
status of `pending|accepted|rejected|failed|dropped` plus a rejection reason. Instrumenting
failure accounting should hook that existing tracer rather than add a parallel mechanism.

### step0b OUTCOME: ran to completion, and FAILED by entropy collapse

step0b fixed step0's failure (0 all-EOS errors, passed the old 159/233 stall point) and
then failed a different way. It does not crash or stall; it completes while the policy
degenerates, which is the harder failure to notice.

| metric | early | late |
|---|---|---|
| `task_reward/avg` | 0.85 | 0.23 -> **0.00** |
| `update/entropy/avg` | 4.13 | **1.2e-06** |
| `correct_seq_len/avg` | 245 | 1127 (pinned at the 1024 cap) |

Entropy fell by six orders of magnitude: the policy became deterministic, emitted
maximum-length degenerate output that never reached EOS, and scored zero.

**Cause: our own `train_dataset.batch_size=32` against a published 256** -- an 8x
reduction, applied "to fit one node" on hardware that is at 30% memory. The reference
config has no guard rails to absorb that: `kl_ctl: 0.0` (no anchor to the reference
policy), `eps_clip: 0.4` (loose), `reward_scaling: 10.0`, `gconfig.n_samples: 4` (small
GRPO groups). An 8x smaller batch with 4-sample groups makes the advantage estimate
high-variance, and nothing regularizes the resulting drift.

**Both Step 0 failures were our own deviations from the published config**, not AReaL
bugs: `max_new_tokens=512` in step0, `batch_size=32` in step0b. The lesson survives its
original wrong justification (flash-attn) intact: change one thing at a time, and change
nothing the reference sets unless the hardware forces it.

**A cautionary note on my own reporting.** I read `task_reward/avg` at step ~80, saw
0.71-0.81, and reported "Step 0 is reproducing / healthy". A first-quarter vs last-quarter
comparison on that early window even showed +0.029. The full series has OLS slope
**-0.0021 per step, permutation p < 0.0001** (n=171): the reward was already collapsing
while I called it healthy. An early window of a diverging run looks exactly like a healthy
one. Fit the trend on everything available and state the test, or say nothing about
direction.

### Guard-harness audit: my watchdog would have killed a healthy run

A subagent audit found 15 defects; two were critical and both were mine.

**D1 (critical).** `watchdog.sh` had `now=$(grep -oE step [0-9]+/[0-9]+ "$LOG" ...)` --
the pattern UNQUOTED. grep took `step` as the pattern and `[0-9]+/[0-9]+` plus `$LOG` as
two file operands, so it prefixed output with the filename and `2>/dev/null` swallowed the
error. The sample was the constant `/home/ubuntu/runs/step0b/train.log:step` for the life
of the run, so strikes accumulated unconditionally: the watchdog was scheduled to
`kill -TERM -1226891` a perfectly healthy run at 01:18:42. Verified on the live file, then
the watchdog was killed (its own session leader, PGID disjoint from the trainer's).

Root cause of the defect: I wrote the file through a quoted SSH heredoc, escaping `cur()`'s
quotes as `'"'"'` but writing that one line with bare `'...'`, which the outer single-quoted
argument consumed. **Write scripts to a file and copy them; do not compose them inside
nested shell quoting.** All fixed files were written locally and scp'd.

**D2 (critical).** `logfilter.py`'s `CAP = 200` applied to the progress line too. Every
`Step N/233` line shares one signature, so steps 201-233 never reached the log -- the
liveness signal the watchdog reads was being suppressed by the compression policy. Observed
live: the filtered log stopped at `step 200/233` while the unfiltered worker log continued
to 220.

Other confirmed defects: the launcher always exited 0 so a failed trainer reported success
(D5); `: > "$LOG"` truncated with no concurrency guard (D7); `stdbuf` cannot unbuffer
CPython (D8); the signature table grew unbounded on unique-per-line storms (D9); pgid `0`
would make the watchdog kill its own group (D10); an unreadable pgid file silently retired
the guard with exit 0 (D4); `cur()` was dead code that returned the FIRST step number
(D6); `[0-9a-fA-F]{6,}` collapsed ordinary words like "facade" (D13).

All fixed in `experiments/harness/`, with 9 behavioural tests that reproduce the critical
findings: an advancing counter survives, a frozen one is killed, step 233/233 reaches the
log, and invalid pgids keep the guard alive instead of retiring it.

### Theory link: the group-size law flagged this config before we ran it

Our own analysis (Bay & Yearick, `2607.00152`, carried in `paper_src/`) gives the GRPO
group-size requirement

    G  >=  1 / (8 * eps * p * (1 - p))

for a group of size G to resolve the sign of the advantage at tolerance `eps`, where `p`
is the solve rate. At the observed early solve rate **p = 0.76** (`p(1-p) = 0.1824`):

| eps  | required G | config's G = 4 |
|------|-----------|----------------|
| 0.20 | 3.4       | ok             |
| 0.10 | 6.9       | **below**      |
| 0.05 | 13.7      | **below**      |

`gconfig.n_samples: 4` is below the threshold for any tolerance tighter than 0.2. Our
`batch_size=32` then compounded it: the number of groups contributing to each update fell
from 256 to 32, so the per-update advantage estimate was both individually noisy (small G)
and averaged over 8x fewer groups.

**Stated precisely, to avoid overclaiming.** The law bounds the group size needed to
recover the *sign* of the advantage at a given tolerance. It does not itself predict
entropy collapse; collapse is a downstream consequence of repeatedly taking large,
badly-signed steps with `kl_ctl = 0.0` (no anchor to the reference policy) and
`eps_clip = 0.4` (loose trust region). So this is *consistent with* the theory and
retrospectively explained by it, not a preregistered prediction that it confirmed. The
honest claim is: the diagnostic we already had would have flagged this config as
underpowered, and we did not run it before deviating.

**Two consequences for the paper.**
1. The group-size law has practical diagnostic content -- it identifies an underpowered RL
   config from `p` and `G` alone, before any GPU time is spent. That is worth stating as a
   usable check, not just an inequality.
2. It constrains our own experimental design. Any self-evolving loop we build on GRPO must
   report `G`, `p`, and the implied `eps`, because a method effect measured with an
   underpowered group is indistinguishable from advantage noise. This is the same
   underpowered-comparison trap recorded previously: put the standard error on the
   *difference*, and check the group is large enough to resolve it at all.

### Science eval kits staged on the A100 host (`~/evalkits/`)

`genebench-pro` (16 MB) and `bio-mystery` (11 MB) copied from the READ-ONLY local tree
`~/text-dna/genebench-pro/` to a NEW folder `~/evalkits/` on the A100 host. Excluded
`.git`, `__pycache__`, `.pytest_cache`, `.hf_cache`, `results` -- prior results stay on the
source machine so they cannot be confused with ours. Source tree verified clean
(`git status --porcelain` empty) after the copy: nothing in it was modified.

**Integrity verified by `rsync -naic --checksum`: zero files would re-transfer, i.e. every
file is byte-identical.**

A first attempt to verify with a hand-rolled manifest (`find -printf '%P %s'`) reported a
checksum mismatch. That was the *check's* bug, not the copy's: `%P` strips the starting-point
prefix, so `genebench-pro/README.md` and `bio-mystery/README.md` both collapse to
`README.md` and the two trees' namespaces merged. A failing integrity check is a claim that
needs its own verification before it is acted on -- use `rsync --checksum` and let the tool
that understands the trees do the comparison.

### Controlled comparison: batch size sets the entropy-decay RATE

Same config, same model, same data order; only `train_dataset.batch_size` differs
(32 vs the published 256). Compared at **matched sequences consumed**, which is the fair
axis -- step0c does 1024 sequences per step, step0b 128, so step0c step 5 == step0b step 40.

| sequences consumed | step0b (batch 32) | step0c (batch 256) |
|---|---|---|
| 1,024 | 4.387 | 4.372 |
| 2,048 | 2.405 | 4.338 |
| 3,072 | 1.689 | 4.337 |
| 4,096 | 0.716 | 4.062 |
| 5,120 | 0.436 | 3.685 |
| 6,144 | (collapsed) | 2.917 |

Entropy is not a function of data consumed; it is a function of how many *updates* that
data was split into. step0b took 40 noisy steps to consume what step0c took 5 clean steps
to consume, and paid for it with an 8x lower entropy at the same point. This is the
variance argument made visible: with G=4 and 32 groups per update, the advantage sign is
unreliable (see the group-size law note), and repeatedly stepping on unreliable signs with
`kl_ctl=0.0` and `eps_clip=0.4` drives the policy deterministic.

### PREDECLARED TEST (written before the outcome is known)

step0c's entropy is still falling monotonically: 4.37, 4.34, 4.34, 4.06, 3.69, 2.92. The
published config has no entropy bonus and no KL anchor, so nothing in it obviously arrests
the decay. Two outcomes, declared now so the result cannot be rationalized after the fact:

- **If entropy stabilizes above ~1.0 and `task_reward/avg` rises over 290 steps**, the
  published config is sound and our batch cut was the whole story.
- **If entropy falls below 0.1 before step ~15 and reward declines**, then the published
  config collapses too, only more slowly, and batch size is a rate parameter rather than
  the cause. In that case Step 0 does not reproduce as published on this hardware, and the
  honest report is that we could not reproduce it -- not that we found a better setting.

Either way the number that decides it is `update/entropy/avg` against
`ppo_actor/task_reward/avg`, read across all 290 steps, not an early window. Reading an
early window is exactly the mistake made on step0b.

### FINDING: default fd limit (1024) breaks large-batch rollouts, and the failure is silent

step0c at the published `batch_size: 256` with `gconfig.n_samples: 4` runs 1024 concurrent
rollouts against `max_concurrent_rollouts: 256`. The default soft limit of **1024** file
descriptors is not enough:

    OSError: [Errno 24] Too many open files
    aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host ... [Too many open files]
    [RemoteInfEngine Rank N] ERROR: Workflow execution failed: ... [Too many open files]

Counts at step 18: 6 fd errors, 2 failed workflows, 43 `callback/rollout_complete` read
timeouts (the timeouts are downstream of the same pressure). The soft limit was 1024
against a **hard limit of 1048576** -- headroom was always available, nothing was tuned.

**Why this is dangerous rather than merely noisy.** A failed workflow does not raise; per
`areal/v2/inference_service/controller/workflow.py:207-231` it is assigned
`_set_last_reward(..., 0.0)`. So fd exhaustion is laundered into the reward signal as
"the model got this wrong". At 2 failures in ~18,000 sequences the bias is negligible, but
it grows with concurrency and it is invisible in the reward curve. This is the same
silent-zero family as the `max_new_tokens` truncation.

**Fix applied without restarting.** `prlimit --pid <p> --nofile=131072:1048576` on all 335
processes in the trainer's process group raised the limit in place, so the 18 completed
steps were kept. `ulimit -n 131072` is now set in `step0c.sh` for future runs.

**Operational rule:** raise the fd limit before any large-batch rollout run, and count
failures by exception type per step. A reward curve alone cannot distinguish a harness
failure from a wrong answer.

### CORRECTION: held-out evaluation IS wired; the metric key is `eval-rollout/reward`

Two earlier statements in this file are wrong and are corrected here.

1. *"`evaluator.freq_epochs: 1` with one epoch fires the evaluator exactly once, so there
   is no validation curve."* Wrong.
2. *"`gsm8k_rl.py` never instantiates an Evaluator, so `evaluator.freq_steps` is inert."*
   Also wrong -- I concluded this from `grep -n "Evaluat" examples/math/gsm8k_rl.py`
   returning nothing.

What is actually true: `gsm8k_rl.py` passes `valid_dataset`, `eval_workflow` and
`eval_workflow_kwargs` (with `temperature=0.6`, vs 1.0 for training) into
`PPOTrainer.train()`. The Evaluator is constructed inside `PPOTrainer`, not in the example
script, and it runs. The reason nothing matched is that the metric is logged as
**`eval-rollout/reward`**, which contains neither "Evaluat" nor "valid". There are also
dedicated `eval-rollout.log` and `proxy-eval-rollout.log` worker logs.

The lesson is the one already recorded for `min_new_tokens`, applied in the opposite
direction: grepping for a *name I expected* is not evidence of absence. For the dead field
the check was right (no reads anywhere); here I grepped one file for a class name and
generalised to "the feature does not run". Confirm a feature is off by finding the metric
it would emit, not by failing to find the word.

### First held-out numbers

| step | train `task_reward/avg` | held-out `eval-rollout/reward` |
|------|------------------------|-------------------------------|
| ~20  | ~0.80                  | 0.7210                        |
| ~40  | ~0.85                  | 0.7204                        |

Train reward is climbing (0.69 -> 0.86) while held-out reward is **flat** (0.7210 ->
0.7204). Two points cannot distinguish "no generalisation yet" from "overfitting to the
train split" from noise, so no conclusion is drawn here -- but this is the number that
matters for the predeclared test, not the train curve, and it is the number to read at
step 290.

Entropy did not collapse. It bottomed near 0.21 around step 16-23 and has since recovered
(0.2541, 0.2912, 0.3084, 0.3175). The predeclared failure branch (entropy < 0.1 with
declining reward) has not triggered.

### ROOT-CAUSE FIX: single-threaded callback server discards finished rollouts

**Symptom.** step0c logged a growing count of
`Callback to http://<host>:<port>/callback/rollout_complete failed: HTTPConnectionPool: Read timed out`
(43 at step 18, 103 by step 39), alongside a few `Too many open files`.

**Chain, end to end.**

1. `WorkflowExecutor._send_callback` (`areal/infra/workflow_executor.py:371-385`) posts one
   callback per finished rollout, `timeout=30`, **fire-and-forget**: a failure is logged and
   swallowed, with no retry.
2. The receiver is built at `areal/infra/controller/rollout_controller.py:680` as
   `make_server(host, port, app, threaded=False)` -- **single-threaded**. It serves one
   callback at a time.
3. At the published `batch_size: 256` with `gconfig.n_samples: 4`, 1024 rollouts finish and
   post concurrently. They serialise behind one handler thread; later senders exceed 30s.
4. On the controller, `asyncio.wait_for(future, timeout=request_timeout)` then raises, and
   `_handle_rollout` takes `except TimeoutError` (`rollout_controller.py:921-927`):

       self._pending_futures.pop(task_id, None)
       manager.on_rollout_rejected()
       return None                      # <-- never fetches the result

   The success path fetches the trajectory via `wait_for_task`; the timeout path does not.
   So a rollout whose generation had **already completed** is discarded.

**Why the batch still looked fine.** `n_seqs` is 1024 on every step, because the executor
submits replacements for rejected rollouts. Nothing is missing from the batch -- the cost is
wasted GPU work, and the choice of which rollouts survive is made by callback-queue
position rather than anything meaningful.

**Reproduction** (`experiments/harness/test_callback_threading.py`), at the real
concurrency, handler work 0.02s, client timeout 5s:

| server | callbacks OK | timed out | wall |
|---|---|---|---|
| `threaded=False` | 377 | **647 / 1024** | 10.07s |
| `threaded=True`  | **1024** | **0** | 4.31s |

At 64 concurrent the serial server does not yet drop anything but is already 6.4x slower
(1.47s vs 0.23s) -- perfectly serial, 64 x 0.02s. The failure only appears once serial
service time crosses the client timeout, which is why this surfaced when we moved from our
under-sized `batch_size=32` (128 rollouts) to the published 256 (1024 rollouts).

**Fix.** `threaded=True`. This is a correction, not a tuning choice: the handler was
already written to be called concurrently -- `_resolve_task_future` pops under
`self._futures_lock` and resolves the asyncio future via `loop.call_soon_threadsafe`, and
the `.pop()` makes double-resolution impossible. `threaded=False` prevented the concurrency
the handler was designed for.

**Residual design flaw, NOT fixed here.** The `except TimeoutError` branch throws away a
result that is very likely still retrievable -- it could attempt the same `wait_for_task`
fetch the success path uses before rejecting. Fixing that changes upstream control flow, so
it is recorded rather than changed: with `threaded=True` the branch should now be rare.
Worth revisiting before any self-evolving run, where rollouts are longer and callbacks
burstier.

### step0c error triage: all three accounted for

Verified on the relaunched run (with the `threaded=True` fix) at step 8, against the old
run which had 103 callback failures by step 39.

| error | count before | count after | status |
|---|---|---|---|
| `callback/rollout_complete ... Read timed out` | 103 | **0** | root cause fixed (`threaded=True`) |
| `OSError: [Errno 24] Too many open files` | 6 | **0** | fixed (`ulimit -n 131072` in step0c.sh) |
| `Failed to get yes/no token IDs dynamically` | 4 | 4 | **benign, no action** |

**The third is not ours and is not an error.** It comes from
`sglang/srt/entrypoints/openai/serving_rerank.py:43` -- sglang caching yes/no token IDs for
**Qwen3 reranker scoring**. It is a `logger.warning`, it fires during CUDA-graph capture at
server startup (hence its appearance inside the "Capturing batches" progress bar, not per
rollout), and the function carries its own fallback to hardcoded IDs (9693/2152). We never
call the `/v1/rerank` endpoint, so nothing we do depends on it.

Recorded rather than silenced: a warning from an unused endpoint is noise, but suppressing
it would also hide it if we ever did use reranking.

### ROOT-CAUSE FIX: all-EOS generations 500 sglang and are scored as reward 0.0

**The error returned at the PUBLISHED config**, which corrects my earlier attribution.
step0's 3,468,076 copies of

    500 {'detail': 'All output_tokens are EOS or PAD tokens; cannot strip stop tokens
                    without removing entire output.'}

were blamed on our `max_new_tokens=512` override. That made it catastrophic; it did not
cause it. step0c runs the published 1024 and hit the same error: 2 occurrences at step 27,
**32 one step later**, while entropy fell to 0.0296.

**Mechanism, and why it is self-reinforcing.**
1. As entropy falls the policy starts sampling EOS as the *first* token.
2. The completion is then entirely EOS/PAD. sglang's stop-token stripping would empty the
   output, so it returns 500.
3. `OpenAIProxyWorkflow` catches it, and AReaL assigns the trajectory **reward 0.0**
   (`controller/workflow.py:207-231`).
4. A false zero enters training and pushes entropy lower, which makes step 1 more likely.

Entropy collapse and this error are not two problems; the error is a *consequence* of low
entropy that then *accelerates* it.

**Fix: force at least one non-EOS token.** `extra_body={"min_tokens": 1}` in the GSM8K
workflow kwargs.

The field name is the whole subtlety, and it is exactly the dead-field trap twice over:
- AReaL's `gconfig.min_new_tokens` is declared in `cli_args.py` and **read nowhere**, so
  setting it does nothing.
- sglang's OpenAI adapter declares **`min_tokens`** on `ChatCompletionRequest` and maps it
  in `to_sampling_params()` as `"min_new_tokens": self.min_tokens` (`protocol.py:771`), so
  sending `min_new_tokens` over the API is also silently ignored.

Verified directly (`experiments/harness/test_min_tokens.py`):

| sent on the request | resulting `min_new_tokens` |
|---|---|
| `min_tokens=1` | **1** |
| (nothing) | **0**  <- what every run so far used |
| `min_new_tokens=1` | **0** (ignored) |
| `max_completion_tokens=1024` | `max_new_tokens=1024`, undisturbed |

**This is a deviation from the published config and is recorded as one.** It changes
sampling by suppressing EOS at position 0. The justification is that the alternative is not
"unmodified behaviour" but "a 500 that is silently converted into a training signal of
zero". A generation of length 0 is not a datum about the policy's quality.

**What it does NOT fix.** Entropy still collapses (4.46 -> 0.0296 by step 28) under
`kl_ctl: 0.0` with no entropy bonus. `min_tokens=1` removes the crash and the false zeros
so the collapse can be measured cleanly; it does not prevent it. Whether the published
config avoids collapse over 290 steps is still the open reproduction question.
