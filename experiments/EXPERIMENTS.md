
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

### RETRACTION: failed rollouts are NOT scored as reward 0.0

Several entries above claim that AReaL "launders infrastructure failures into the training
signal as reward 0.0", and build a self-reinforcing feedback loop on top of it
(all-EOS -> 500 -> false zero -> entropy falls -> more all-EOS). **That claim is wrong and
is retracted here.** The entries are left in place rather than edited, so the reasoning
error stays visible.

**Where it came from.** The exception handler at
`areal/v2/inference_service/controller/workflow.py:207-231` calls
`_set_last_reward(http_session, 0.0, session_api_key)`. I read that line and concluded the
zero enters training. I never checked the very next line, `return None`, or what happens to
a None trajectory.

**What `_set_last_reward` actually is.** It POSTs `{"interaction_id": None, "reward": 0.0}`
to the gateway and returns a trajectory id -- session bookkeeping. The `return None` on the
same path is what decides whether the rollout reaches the batch, and None trajectories are
dropped and replaced.

**The measurement.** `experiments/harness/eos_reward_correlation.py` counts all-EOS
failures between consecutive step markers and pairs each count with that step's
`task_reward/avg`. If N failures out of a 1024-sequence batch became zeros, reward would
fall by roughly `reward * N / 1024`.

| log | steps with failures | max in one step | corr(failures, reward) | observed slope | slope predicted if zeros were injected |
|---|---|---|---|---|---|
| step0c pre-fix | 2/21 | 12 | **+0.182** | +0.00245 | -0.00073 |
| step0c current | 4/18 | 62 | **+0.401** | +0.00090 | -0.00075 |

The observed correlation is *positive* in both runs, and the single worst step (62 failures)
carried the **highest** reward in its window (0.8135). `n_seqs` is exactly 1024 at every
step. Failures are dropped and replaced.

**What this invalidates.**
- The "self-reinforcing loop" account of all-EOS. Entropy collapse still *causes* all-EOS
  generations, but all-EOS does not push entropy down through false zeros. There is no loop.
- The framing that four failure paths "all converge on reward 0.0". The shared consequence
  is wasted compute and queue-position-determined selection, not signal contamination.
- The urgency behind three restarts. They were chasing corruption that was not occurring,
  at a cost of roughly an hour of GPU time.

**What survives.**
- The callback fix (`threaded=True`) is still correct and still worth having: it stopped
  discarding finished rollouts and cut step time from 50.7s to 38.2s. Only my stated
  *reason for its severity* was inflated.
- The fd-limit fix stands on the same footing.
- `min_new_tokens` genuinely was a dead field, and forwarding it in
  `SGLangBackend.build_generation_request` is a real fix -- but it did **not** reduce
  all-EOS here (132 by step 19), because `OpenAIProxyWorkflow` reaches sglang through
  `/v1/chat/completions`, not the native generate path that builder serves. So it is a
  correct patch to a path this workflow does not use.

**The lesson, stated so it is checkable next time.** Three separate errors in this session
share one shape: I confirmed a value or behaviour at one layer and asserted it about the
whole path. Twice for `min_new_tokens` (config field; then request protocol), once for
reward contamination (handler line, not batch membership). The check that would have caught
all three is the same: measure the effect end to end on data the running system produced,
before asserting the mechanism.

## Step 0 reproduction: outcome, and three upstream findings

### The reproduction FAILED, per a criterion committed before the outcome (e25a30f5)

step0c ran AReaL's published GSM8K GRPO config with no deviation except the forced
`attn_impl=sdpa`. It collapsed:

| step | entropy | train reward | seq_len |
|---|---|---|---|
| 6 | 4.032 | 0.761 | 341 |
| 22 | **0.029** | 0.775 | 367 |
| 62 | 0.026 | 0.748 | 362 |
| 76 | 0.037 | **0.039** | 209 |

Held-out `eval-rollout/reward`: 0.677, 0.677, 0.707, 0.687 -> **0.378**.

**The shape is the lesson.** Entropy collapsed by step 22 and the policy then looked
healthy -- reward ~0.78 -- for *forty more steps* before falling off a cliff. Any run
sampled in steps 22-62 reads as a successful reproduction. This is the same error made
earlier in this session when an early window was called "reproducing"; here the misleading
window was 40 steps wide.

The honest report is that the published config does not reproduce on this setup -- not that
a better setting was found.

### Why it has no defence against collapse

- **AReaL has no entropy regulariser at all.** No `entropy_coef`/`entropy_bonus` anywhere;
  entropy is `.detach()`ed and logged as a metric. This is presumably why MEDS patches its
  own in at `verl/workers/actor/dp_actor.py:560`.
- The reference config sets `kl_ctl: 0.0`, so the reference policy is not even loaded.
- `reward_scaling: 10.0` is **not** a lever: `adv_norm` normalises by batch mean *and std*,
  so scaling rewards scales the advantage and its std together and cancels exactly.
- `importance_weight` is exactly 1.0 (min and max) and `clip_ratio` exactly 0.0, so with
  `ppo_n_minibatches: 1` the update is plain on-policy policy gradient with no trust region.

So nothing in the published configuration constrains the update. `kl_ctl` is the only
regulariser the framework offers, and it is genuinely wired (`actor.py:255`).

### FINDING: `min_new_tokens` is unusable with AReaL

Not a plumbing problem -- the plumbing is correct through all 10 hops
(`experiments/harness/verify_chain.py`). The feature itself is incompatible:

    AttributeError: 'NoneType' object has no attribute 'additional_stop_token_ids'
    sglang/srt/sampling/penaltylib/min_new_tokens.py:24

sglang's min-new-tokens penalizer reads `req.tokenizer`, and AReaL sends raw `input_ids`
with no tokenizer attached to the request. Every prefill kills the scheduler. Observed with
`min_new_tokens=1`: all four rollout servers dead, 28,146 500s, **zero training steps**.

So all-EOS completions cannot be prevented this way. Do not retry it.

Getting here took four attempts, each of which fixed one link and asserted the whole chain:
`gconfig.min_new_tokens` (dead field), `extra_body min_tokens` (dropped when ArealOpenAI
rebuilds the gconfig), `sglang_remote` forwarding (forwards a value nothing sets), and then
forwarding to only one of two sub-clients (`TypeError` at startup). The verifier now checks
every hop individually and names the one that fails.

### FINDING: infrastructure failures are dropped, NOT scored as reward 0.0

Retracted claim, corrected by measurement. `_set_last_reward(..., 0.0)` in the exception
handler is gateway session bookkeeping; the `return None` on the same path decides batch
membership, and None trajectories are dropped and replaced. Per-step failure counts
correlate **positively** with that step's reward (+0.18 and +0.40 in two runs), and the
single worst step (62 failures) carried the highest reward in its window. `n_seqs` is 1024
every step.

The cost of these failures is wasted compute and queue-position-determined selection, not
signal contamination.

### Fixed, and worth keeping

- **Callback server was single-threaded** (`rollout_controller.py:680`, `threaded=False`),
  serialising 1024 concurrent rollout-completion POSTs. Senders exceeded the 30s
  fire-and-forget timeout and the controller then REJECTED rollouts that had already
  finished generating. Reproduced: 647/1024 callbacks lost vs 0 with `threaded=True`. Side
  effect: step time 50.7s -> 38.2s.
- **File-descriptor soft limit 1024** against a hard limit of 1048576, exhausted by 1024
  concurrent rollouts. Raised in place with `prlimit` (no restart), `ulimit -n 131072` in
  the launcher.

## The four failed runs are a known failure mode, not a botched reproduction

**"Self-Improvement Can Self-Regress: The Rise-and-Collapse Failure Mode of LLM
Self-Training"**, arXiv 2606.21090.

The paper reports: *"pass@1 shows a robust rise-then-collapse pattern: it peaks within tens
of gradient steps and then falls back, sometimes to near zero."* Observed on Qwen-2.5-3B
and Qwen-2.5-7B with a binary CodeGrader reward over 20-step campaigns.

Our data matches it point for point, on the same model family:

| paper | our runs |
|---|---|
| peaks within tens of gradient steps, then falls to near zero | step0c held reward ~0.78 through step 62, then **0.748 -> 0.039** by step 76 |
| **"KL- and EWC-style constraints do not prevent it"** | step0d added `kl_ctl=0.01`. Collapse not prevented -- entropy still fell to 0.017 |
| **"GRPO raises the floor but does not remove the cliff"** | step0d's reward *floor* held at ~0.51 (vs step0c's 0.039), but the policy still degenerated, via length blowup to the 1024 cap |
| cause: *"within-task policy over-optimization on a fixed distribution"*, NOT catastrophic forgetting | GSM8K, fixed distribution, p=0.76 |
| Qwen-2.5-3B / 7B | Qwen-2.5-1.5B-Instruct |

**This reframes the Step 0 result.** We did not fail to reproduce AReaL through
incompetence, and the published config is not uniquely broken. AReaL's demo config is
aggressive (lr 6e-6 and eps_clip 0.4, against lr 1e-6 and eps_clip 0.2 in their own
`scaffolding/gsm8k_rlvr_scaffolding.yaml`), which brings the cliff forward -- but the cliff
is a property of the training procedure, not of that config alone.

It also explains a day of wasted GPU: **the paper had already established that KL does not
prevent this**, and step0d spent hours rediscovering it.

### The mitigation that actually works, and is cheap

The paper tests three control loops. The one that transfers immediately:

* **ES (early stop)** -- *"a within-campaign early-stop rule that rolls forward the peak
  checkpoint and sets the next budget to peak_step+3"*. On Qwen-2.5-7B it reached 22.2%.
* **CARE** -- between-campaign memory with a capability posterior, transfer gate, and
  regression-aware belief revision. Heavier; needs a campaign structure we do not have.
* **GRPO** -- group-relative normalisation. We already use it, and the paper is explicit
  that it *"raises the floor but does not remove the cliff"*.

So the immediately actionable change is **early stopping on a held-out signal, keeping the
peak checkpoint** -- not another hyperparameter sweep. Note this requires an eval that runs
often enough to find the peak, which is why `evaluator.freq_steps` and
`experiments/bench/math_bench.py` matter: with `freq_epochs: 1` the peak is invisible.

### Co-evolution stability, and what it means for our framework

Three further papers bear on the evolving-critic question:

* **MetaSkill-Evolve** (arXiv 2607.05297) -- *"a two-timescale framework separates fast
  task-skill from slow meta-skill evolution"*, i.e. alternating optimisation with explicit
  timescale separation.
* **Multi-Agent Evolve** (arXiv 2510.23595) -- co-evolution with **critic freeze**
  mechanisms; notes that prior LLM-interaction approaches *"suffer from instability and
  collapse during training"*.
* **Self-evolving LLM agents with in-distribution Optimization** (arXiv 2606.07367) --
  *"If the evaluator and policy adapt too closely to one another without external
  grounding, the entire loop can spontaneously collapse into shared shortcuts rather than
  the intended objective."*

That last sentence is the published form of the guard already in `selfevo/compose.py`: an
evolving reward requires a frozen anchor, and a learned policy that can evolve its own
reward must be scored by the frozen one. **What the guard does NOT yet express is
alternation**: the critic and the policy must not both update on every step. Two-timescale
separation -- freeze one while the other optimises -- is the standard remedy and is
currently unrepresented in the axes.

BigBang does this implicitly: their meta-critic compares critic judgements against held-out
outcomes on a *periodic* cadence, not continuously. We encoded the anchor and missed the
cadence.

### CAUGHT IN THE WILD: survivor bias would have reported Qwen3.8-27B at 100% on AIME

The math-harness audit predicted this failure in the abstract:

> *a partial outage silently shrinks the denominator... timeouts correlate with long/hard
> generations -> upward bias*

It happened in production within hours, on the first frontier-model measurement:

    aime24   acc=1.0000  se=0.0000  n=17/30  fail=13
    aime25   acc=1.0000  se=0.0000  n=16/30  fail=14

**100% accuracy on AIME**, computed over survivors, with 13-14 of 30 problems timed out.
The problems that finished are the ones the model solved *quickly* -- the easy ones. The
hard ones exceeded the 600s timeout and vanished from the denominator entirely. `se=0.0000`
compounded it by asserting zero uncertainty, which is what a binomial SE does at 17/17.

Had the log been trusted, the reported result would have been
"Qwen3.8-27B: 100% on AIME24" -- a number that is wrong, headline-shaped, and would have
survived casual review because AIME accuracy near 100% is not obviously absurd for a 2026
frontier thinking model.

**Two mechanisms, both now fixed.**

1. The run used the STALE `$HOME/math_bench.py`, not the audited repo copy, because
   `~/run_math.sh` resolved `$(dirname $0)` to `$HOME`. The audit flagged exactly this
   ("nothing enforces it -- a stale copy would produce numbers credited to audited code").
   Both `$HOME` paths are now **symlinks into the repo**, so a stale copy is impossible
   rather than merely discouraged.
2. The parameters were wrong for a thinking model: 600s timeout at concurrency 32 with
   8192 max tokens. Re-running at timeout 2400, concurrency 8, 16384 tokens.

The audited harness would have caught it regardless: it counts `n_failed` separately from
wrong answers, warns whenever `n_graded < n_problems` rather than only at zero, and reports
a Wilson interval instead of an SE that reads 0.0000 at the extremes.

**The invalid run is quarantined, not deleted**, at
`~/runs/math/qwen38_27b_INVALID_survivor_bias/`. A wrong number that is kept and labelled
is evidence; a wrong number that is deleted is a lesson someone repeats.

### First trustworthy frontier measurement: Qwen3.8-27B on AIME24

    aime24  acc=0.7333  wilson=[0.556, 0.858]  n=30/30  fail=0  trunc=13  nobox=4

Audited harness, all 30 problems graded, zero generation failures, 865 KB of generations
persisted for re-audit. Settings: temperature 0, seed 0, max_tokens 16384, concurrency 8,
timeout 2400s, sglang with CUDA graphs disabled, TP=4 on A100-80GB.

**This is a LOWER BOUND, not a point estimate.** 13 of 30 generations hit the 16384-token
cap and were graded as wrong answers. A thinking model that has not finished thinking has
not answered incorrectly -- it has not answered. The true score at a larger budget is
somewhere in [0.733, 0.733 + 13/30], and the only way to close that is to re-run with a
larger cap and compare. The harness surfaces `trunc` precisely so this cannot be reported
as a settled number.

Contrast with the run this replaced, quarantined at
`runs/math/qwen38_27b_INVALID_survivor_bias/`:

| | accuracy | n graded | failures |
|---|---|---|---|
| invalid (stale harness, 600s timeout) | **1.0000** | 17/30 | 13 |
| audited (2400s timeout) | **0.7333** | 30/30 | 0 |

The invalid run's 100% was survivor bias: at a 600s timeout the problems that finished were
the ones solved quickly, and the hard ones left the denominator.

### What this means for headroom, which was the reason to measure

| model | AIME24 | AIME25 |
|---|---|---|
| Qwen2.5-7B-Instruct | 10.0% | 3.3% |
| Qwen3.8-27B | **73.3%** (>=) | pending |

AIME24 has **less headroom than the frozen-benchmark rationale assumed**. At 73.3% with a
further 13/30 truncated, a method effect has perhaps 25 points of room, and the truncation
uncertainty alone is comparable to any plausible effect size. This is the same saturation
problem as GSM8K at 76%, arriving one benchmark later.

Consequences to act on rather than note:
1. **AIME25 and HMMT are the better axes** for this base; AIME24 is close to spent.
2. The truncation rate must be driven near zero before any AIME number is used in a
   comparison, or the token budget becomes a confound with the method.
3. The claim "AIME leaves real headroom" was true for Qwen2.5-7B and is **not** established
   for a frontier base. It was recorded as an assumption to be measured; it has now been
   measured and partly falsified.

### The AIME numbers measure the token budget, not the model

Complete audited run, Qwen3.8-27B, max_tokens 16384:

    aime24  acc=0.7333  wilson=[0.556,0.858]  n=30/30  fail=0  trunc=13  nobox=4
    aime25  acc=0.7333  wilson=[0.556,0.858]  n=30/30  fail=0  trunc=11  nobox=7

Sensitivity, measured from the persisted generations
(`experiments/bench/trunc_analysis.py`):

| bench | n | correct | truncated | trunc WITH a box | of those, correct | acc | hard upper |
|---|---|---|---|---|---|---|---|
| aime24 | 30 | 22 | 13 | 5 | **5/5** | 0.733 | **1.000** |
| aime25 | 30 | 22 | 11 | 3 | **3/3** | 0.733 | **1.000** |

Two facts make this decisive rather than a caveat:

1. **Every truncated generation that emitted a box at all was correct** (8/8 across both
   benchmarks). The rest never reached an answer. There is no evidence that truncation is
   selecting for *wrong* reasoning -- it is selecting for *unfinished* reasoning.
2. **Truncated generations are ~5x longer than completed ones** (aime24 median 45,084 chars
   vs 8,292). Truncation lands on the hard problems, exactly where a method effect would
   have to show up.

So the true score lies in **[0.733, 1.000]** and the interval from truncation (up to 26.7
points) dwarfs the Wilson interval (+/-15 points) that the harness reports. **At 16384
tokens, an AIME comparison on this model measures the token budget, not the policy.**

**Consequence, which is a methodology rule and not a to-do.** No AIME number from a
thinking model enters a comparison until `trunc` is near zero; otherwise the token budget
is confounded with the method, and a method that merely produces shorter reasoning would
score higher for the wrong reason. This is the same shape as the earlier survivor bias --
a plausible-looking number produced by a mechanism unrelated to what is being measured --
and it is caught only because the harness reports `finish_reason` and persists generations.

The previous entry called 0.733 a lower bound and said truncation "must be driven near
zero". That was right but understated: the bound is so loose that the measurement carries
almost no information about the model.

### LiveCodeBench v6 obtained; integration needs no code changes

175 problems, all with private test cases, downloaded to the HF cache
(`livecodebench/code_generation_lite`, `test6.jsonl`, 134 MB).

| property | value |
|---|---|
| difficulty | **hard 80**, medium 52, easy 43 |
| platform | atcoder 112, leetcode 63 |
| contest dates | 2025-01-04 to 2025-04-06 |
| private test cases | 175 / 175 |

The **hard** bucket is the largest, which makes this the sharpest difficult-code axis
available, and the contest window post-dates most training cutoffs, so contamination is
bounded by construction rather than by assertion. The count also confirms EvoTrainer's
coding evaluation *is* LCB v6: they report "175 held-out problems", which is exactly this
release.

**Integration requires no patches.** Three things were checked rather than assumed:

1. LCB ships `lcb_runner/runner/oai_runner.py`, an OpenAI-compatible backend, so unlike
   AZR's math harness it does not need vLLM.
2. That runner constructs `OpenAI(api_key=os.getenv("OPENAI_KEY"))` with no `base_url`
   argument, which looked like a blocker. It is not: the installed `openai==3.5.0` reads
   `OPENAI_BASE_URL` from the environment. Verified directly -- setting it yields a client
   whose `base_url` is the local endpoint. So pointing LCB at our sglang server is two
   environment variables.
3. Every dependency is already present in the venv (`datasets`, `openai`,
   `huggingface_hub`, `pyarrow`, `fsspec`), and `pyext` -- an unmaintained package that
   would have been the real risk -- is **commented out** in `evaluation/testing_util.py`,
   replaced by `types.ModuleType`.

**A correction worth recording.** An earlier pass reported three missing dependencies
including a fragile `pyext`, and concluded the integration was costly. That was wrong: the
check had run against the system Python 3.10 rather than the venv's 3.12. Querying the
wrong interpreter produced a confident and entirely inverted assessment -- the same shape as
verifying a value at one layer and asserting it about the whole path.

**Not yet done, and not claimed:** the dataset's own loading script is incompatible with
`datasets 5.0.1` ("Dataset scripts are no longer supported"), so the raw `test6.jsonl` is
read directly instead. Generation and sandboxed execution-grading are not wired end to end;
what is established is the data, the difficulty profile, and that the endpoint integration
costs nothing.

---

## step0d scored on a held-out benchmark: the train reward was blind to a real collapse

Six checkpoints were saved across step0d (globalstep 28/57/86/115/144/173). All six, plus
the untrained `Qwen2.5-1.5B-Instruct` anchor, were scored on the full MATH-500 (500
problems, greedy, 8192-token cap, one job per GPU under a hard `timeout`).

The anchor matters: without it, "flat" and "already collapsed before the first checkpoint"
are indistinguishable.

| ckpt  | entropy | train reward | MATH-500 | 95% CI          | acc\|boxed | nobox | trunc | mean len (ch) |
|-------|---------|--------------|----------|-----------------|-----------|-------|-------|---------------|
| base  | --      | --           | 0.528    | [0.484, 0.571]  | 0.537     | 0.016 | 0.012 | 1728          |
| gs028 | 0.2533  | 0.5469       | 0.454    | [0.411, 0.498]  | 0.467     | 0.028 | 0.006 | 1398          |
| gs057 | 0.1344  | 0.2090       | 0.440    | [0.397, 0.484]  | 0.489     | 0.100 | 0.014 | 1445          |
| gs086 | 0.1436  | 0.7207       | 0.466    | [0.423, 0.510]  | 0.478     | 0.026 | 0.020 | 1812          |
| gs115 | 0.0989  | 0.4932       | 0.356    | [0.315, 0.399]  | 0.408     | 0.128 | 0.082 | 3068          |
| gs144 | 0.0253  | 0.5459       | 0.292    | [0.254, 0.333]  | 0.366     | 0.202 | 0.490 | 12898         |
| gs173  | 0.0182  | 0.4814       | 0.316    | [0.277, 0.358]  | 0.404     | 0.218 | 0.598 | 15482         |

**1. Training reward cannot see the collapse.** Over 200 GRPO steps the GSM8K task reward
oscillates between 0.209 and 0.721 with no trend, while held-out MATH-500 falls from 0.528
to 0.316. The base and final Wilson intervals do not overlap, so the decline is not noise.
Ranking these checkpoints by train reward puts gs086 (0.721) first and gs057 (0.209) last;
ranking by MATH-500 puts gs086 third and gs057 fourth. The train signal does not merely
understate the damage, it orders the checkpoints wrongly. Every claim in this project that
rested on GSM8K train reward was measuring something that does not track capability.

**2. The mechanism is length degeneration, not sharpening.** Policy entropy falls 14x
(0.253 -> 0.018), which would ordinarily suggest a policy collapsing onto a confident mode.
The generations show the opposite: mean output grows 9x (1728 -> 15482 characters) and the
fraction hitting the token cap goes from 1.2% to 59.8%. The policy becomes locally
deterministic and globally non-terminating -- it commits hard to each next token while
losing the ability to stop.

**3. Why training never noticed.** step0d trains on GSM8K with
`gconfig.max_new_tokens=1024`. GSM8K answers are short, so a 1024-token cap is not binding
for a healthy policy and the rambling has no room to express itself in-distribution. The
degeneration is only visible on harder, longer problems with a budget large enough to let
the model run. A short-answer training set with a tight generation cap hides exactly this
failure.

**Stated as a limitation, not a result.** `acc|boxed` is survivor-biased and must not be
read as "the model still solves 40%". At gs144 and gs173 roughly half the generations never
produce a parseable answer, and the ones that do are disproportionately the short, easy
problems the model can still finish. The honest reading is that the -0.212 drop in overall
accuracy is real, that the -0.132 drop in `acc|boxed` shows the loss is not purely a
formatting artefact, and that the two cannot be cleanly separated at this truncation rate.
Establishing the split would need a much larger cap on the collapsed checkpoints.

**Not yet done:** the same series has not been scored on AIME24/25 or AMC23. At 30 problems
those cannot resolve a 4-point difference, so MATH-500 was run first on purpose.

Reproduce with `experiments/bench/sweep_entropy.sh` then
`experiments/bench/analyze_sweep.py <suite-dir>`. The analyser recomputes every accuracy
from the raw generations and aborts if it disagrees with the harness's own `results.json`;
an earlier version of it silently reported 0.000 for all seven checkpoints because it read
a key name (`graded`) that the current artefact schema does not use.

### Correction to the above: the grader was biased against the checkpoints being measured

An adversarial audit of the sweep found one real defect, and it points the wrong way.

`extract_boxed` took only the **last** `\boxed{` and returned `None` when it was unbalanced,
by an explicit design decision ("a completion cut off mid-answer has not actually
answered"). On a degenerate looping policy whose token cap lands mid-box, that discards a
correct answer the model already emitted -- often hundreds of times. Measured cost, by
regrading the persisted generations with the same `grade()`:

| ckpt  | as reported | regraded | delta  | items recovered |
|-------|-------------|----------|--------|-----------------|
| base  | 0.528       | 0.528    | +0.000 | 0               |
| gs028 | 0.454       | 0.454    | +0.000 | 0               |
| gs057 | 0.440       | 0.440    | +0.000 | 0               |
| gs086 | 0.466       | 0.466    | +0.000 | 0               |
| gs115 | 0.356       | 0.364    | +0.008 | 4               |
| gs144 | 0.292       | 0.334    | +0.042 | 21              |
| gs173 | 0.316       | 0.358    | +0.042 | 21              |

The base model gains **exactly zero** and every degraded checkpoint gains, monotonically
with degradation. 100% of recovered items are `finish_reason=="length"`, and the audit
measured that 92-93% of those repeat the same boxed value three or more times before the
cut -- the model committed long before the cap. The rule was not measuring "did not
answer", it was measuring "rambled", and charging it only to one side of the comparison.

The audit caught a single case outright: gs173 problem idx=2 graded **correct** at a
2048-token cap and **wrong** at 8192, on byte-identical prefix text. A larger budget
produced a lower score. A grader whose bias tracks the effect under study cannot be used to
study it. Fixed in `math_bench.py` (fall back to the last *balanced* box), with four
regression tests, one pre-existing test reversed and annotated, and 6/7 mutants killed --
the survivor is equivalent (with no boxes the loop body never runs and control reaches the
same `return None`).

**Two claims from the entry above are withdrawn.**

1. **"Monotone" is wrong.** Corrected and uncorrected series both invert twice: gs057 <
   gs086 (0.440 vs 0.466) and gs144 < gs173 (0.334 vs 0.358). Both inversions sit inside
   the harness's own documented +/-1.6pp run-to-run band, so neither direction is real.
   Only three tiers are defensible: base ~0.53; gs028/gs057/gs086 ~0.44-0.47;
   gs115/gs144/gs173 ~0.33-0.36.

2. **"Truncated items were all graded wrong" is wrong.** 82 of gs144's 245 and 111 of
   gs173's 299 truncated completions grade correct, because the loop happened to end on a
   balanced box.

**The decline survives, and is stronger than reported.** The Wilson intervals were the
wrong test: they ignore that all seven models answer the *same* 500 problems. A paired
McNemar test gives, against base:

| ckpt  | discordant (base-only / ckpt-only) | p        |
|-------|------------------------------------|----------|
| gs028 | 80 / 43                            | 1.17e-03 |
| gs057 | 92 / 48                            | 2.79e-04 |
| gs086 | 66 / 35                            | 2.83e-03 |
| gs115 | 105 / 23                           | 8.10e-13 |
| gs144 | 123 / 26                           | 3.55e-15 |
| gs173 | 118 / 33                           | 8.15e-12 |

Every checkpoint is significantly worse than the base, including the three whose Wilson
intervals overlapped it. The audit further showed the decline holds with truncation removed
from the picture entirely: on the 201 problems gs173 did *not* truncate, base scores 0.458
against gs173's 0.338, and that subset is not the easy one (base scores higher on the
truncated problems than the rest).

**Also verified clean by the audit, and worth recording because each would have been fatal:**
identical chat template (byte-identical to the base's, md5 `9a1b1065...`) and identical
token ids across all seven; identical prompt, problem order and golds; identical sglang,
torch, transformers, dtype, rope and context length; 0 formatting false negatives out of
1715 wrong-with-box items under aggressive LaTeX normalisation; 0/3500 regrade
disagreements; no survivor bias (`n_graded=500/500`, `n_failed=0` everywhere).

**Two things remain unverified.** (a) Decoding is argmax over *repetition-penalised*
logits: `sampling_defaults='model'` silently applies `repetition_penalty=1.1` and
`top_k=20` from `generation_config.json`. It is uniform across all seven so it does not
confound the comparison, but the methods text must not say "greedy, temperature 0" without
it. (b) **No replicate of this sweep exists**, so the series carries no run-to-run error
bar -- and the harness's own docstring notes that error dominates problem-sampling error.
The two inversions above are exactly the size that a replicate would settle.

---

## The in-training evaluator deadlocks the run -- and it was our addition, not the reference's

step0g hung at step 59/290 with every process alive, 37 GB held on GPUs 0-3 and 68 GB on
4-7, and **0% utilisation on all eight**. Not a crash: a deadlock. `py-spy dump` on the
trainer names the exact site:

    wait            (threading.py:359)
    wait_results    (areal/infra/workflow_executor.py:615)
    wait            (areal/infra/controller/rollout_controller.py:996)
    _evaluate_fn    (areal/trainer/rl_trainer.py:1447)
    evaluate        (areal/utils/evaluator.py:62)
    _evaluate       (areal/trainer/rl_trainer.py:1505)
    train           (areal/trainer/rl_trainer.py:931)

The main thread is blocked inside **evaluation**, waiting on rollouts that never arrive.
The log agrees: `openai.APITimeoutError`, `httpx.ReadTimeout` and `ProxyRolloutServer
WARNING: Removing stale session` each suppressed 2000x by the logfilter.

**The evaluator is ours.** The reference `gsm8k_grpo.yaml` sets `evaluator.freq_steps:
null`. step0e, step0f and step0g each add `evaluator.freq_steps=20`, and the script header
justifies it as *"one addition, which changes measurement and not training."* That
statement is false. It changes training by hanging it.

This retro-explains the whole run of failures. step0e and step0f were previously recorded
as "killed abruptly, cause not in the logs" -- both stopped mid-line with no traceback,
which reads as SIGKILL. The more likely account now is that they hit the same deadlock and
their watchdogs killed them, truncating the log mid-write exactly as observed. Three
consecutive runs died and the only thing all three shared that the reference lacks is the
evaluator. Recorded as the leading explanation, not as proven: no py-spy dump was taken
while step0e or step0f was hung, and that evidence is gone.

**The fix is not to make the evaluator work.** Its signal is GSM8K eval reward -- the exact
quantity measured above to be directionless and mis-ordering with respect to held-out
capability. Repairing it would buy a signal already shown to be useless. step0h therefore
restores the reference default (`evaluator.freq_steps: null`) and keeps
`saver.freq_steps=25`, so progress is measured the way the sweep measures it: checkpoints
scored offline on held-out MATH-500, on GPUs the trainer is not using.

**A second-order lesson.** The watchdog was doing its job -- it had recorded
`progress=59 prev=57 strikes=1` and would have killed the run within the hour. What it
could not do is say *why*, and without a cause the obvious response is to relaunch, which
would have deadlocked again. A stall watchdog needs a stack dump on strike 1, not just a
kill on strike N.

---

## The scaffolding recipe preserves held-out capability; the demo recipe destroys it

First paired A/B on a committed held-out half. Both runs train Qwen2.5-1.5B-Instruct with
GRPO on GSM8K and differ only in optimisation aggressiveness: step0d uses AReaL's demo
values (lr 6e-6, eps_clip 0.4), step0h uses AReaL's own scaffolding values (lr 1e-6,
eps_clip 0.2). Scored on the `report` half of MATH-500 (250 problems), at the five global
steps present in BOTH runs, with paired McNemar.

| step | demo (lr 6e-6, clip 0.4) | scaffold (lr 1e-6, clip 0.2) | diff | McNemar p |
|------|--------------------------|------------------------------|--------|-----------|
| base | 0.532                    | 0.540                        | --     | --        |
| 28   | 0.468                    | 0.540                        | +0.072 | 2.8e-02   |
| 57   | 0.440                    | 0.536                        | +0.096 | 5.3e-03   |
| 86   | 0.484                    | 0.524                        | +0.040 | 1.9e-01   |
| 115  | 0.380                    | 0.568                        | +0.188 | 1.2e-08   |
| 144  | 0.356                    | 0.556                        | +0.200 | 1.2e-08   |

Scaffold is higher at 5/5 steps, significantly at 4/5, and the gap widens with training.

**The claim is preservation, not improvement.** Scaffold moves 0.540 -> 0.556 across 144
steps, which is +0.016 and inside the noise established below. The honest statement is that
144 steps of the scaffolding recipe leave held-out capability where it started, while the
same number of steps of the demo recipe cost 0.176. RL here is not yet buying capability on
this benchmark; it is a question of whether it destroys it.

**A replicate, finally, and it was free.** The base model appears in both sweeps -- same
checkpoint, same 250 problems, two independent runs. It scores 0.5320 and 0.5400, so
run-to-run noise is 0.0080, and the step-115/144 effect is 24x that. Per-PROBLEM the picture
is noisier: 34 of 250 problems (13.6%) flip between the two identical runs, from batching
nondeterminism at temperature 0. Those flips are symmetric, so they inflate both McNemar
discordant cells roughly equally and make the test conservative rather than optimistic --
but any future claim resting on a handful of problems must account for them.

**A wrong table was published to the session before this one, and this is how it happened.**
`math_bench.py` wrote each generation's `idx` as its position in the CURRENT run rather than
its row in the source file. A full run therefore emitted 0..499 and a `--split report` run
emitted 0..249, so the paired comparison matched step0d's problem k against step0h's problem
k where those are different problems. It produced a confident, plausible table
(mean 0.426 vs 0.550, 5/5 wins) that was meaningless. It was caught only because a sanity
check on the shared base model reported 129 paired problems where 250 were expected, and
44.2% of them disagreeing -- a per-problem flip rate far too high for one model against
itself.

Fixed at the writer (`idx` is now the source row, `run_pos` the position within the run) and
the existing artifacts were remapped, with the remap ASSERTED rather than assumed: for every
row the gold answer at the remapped index had to equal the gold recorded in the artifact,
and the file was to be left untouched on a single mismatch. All 1500 rows matched.

The corrected numbers happen to tell the same story as the wrong ones, which is the
uncomfortable part: a broken pairing produced a table close enough to the truth that nothing
in it looked wrong. The check that caught it was not the result's plausibility but an
invariant that had to hold regardless of the result.

---

## Correction: the held-out split was applied where nothing was held out

The A/B above scored the `report` half of MATH-500 (250 problems) rather than all 500. That
was premature, and it cost half the statistical power for nothing.

The split exists for one reason: arXiv 2607.12227 shows that a method which SEARCHES against
a benchmark and then REPORTS on it overstates its gain. That condition does not hold here.
Both runs train on GSM8K; MATH-500 is never queried during training and never informs a
decision. It is already entirely held out, so splitting it protects against a contamination
route that does not exist while halving n.

**Standing rule from here:**

- **Capability measurement uses the FULL benchmark.** MATH-500 means all 500 problems.
- **The split is used only when MATH-500 feedback drives a decision** -- a routing rule, an
  evolve-policy, a stopping criterion, any harness search. Then search on `search` and report
  on `report`, and say so.

The split infrastructure stays: the moment we fit anything against MATH-500 it becomes
mandatory, and building it after the fact invites choosing the half that flatters.

**Datasets of record.**

- Training: `openai/gsm8k`, `split: train` -- the full 7,473-problem train split, batch 256,
  29 optimizer steps per epoch, 290 steps for 10 epochs. Nothing subsampled.
- Held-out capability: MATH-500, all 500.
- Available and not yet used: AIME24 (30), AIME25 (29), AMC23 (40), and the wider suite
  under the AZR eval tree (olympiadbench, minerva_math, college_math, gaokao*, gpqa,
  mmlu_stem). AIME and AMC are recorded as too small to resolve the effects we are
  measuring -- 30 problems cannot separate a four-point difference -- and are for reporting
  alongside, not for deciding anything.

**What this obliges us to redo.** step0d already has full-500 generations. step0h's six
checkpoints were scored on the report half only, so its search half must be run before the
A/B table can be restated at n=500. The current table stands at n=250 and is labelled as
such; it is not wrong, only weaker than it needed to be.

---

## The A/B restated at the full 500, and a noise floor that grew

step0h's search half was scored, so the demo-vs-scaffold comparison no longer runs on half
the benchmark. Merging is checked rather than assumed: the two halves must be disjoint by
source index and must together cover exactly the 500 rows, or the script refuses.

| step | demo (lr 6e-6, clip 0.4) | scaffold (lr 1e-6, clip 0.2) | diff | McNemar p |
|------|--------------------------|------------------------------|--------|-----------|
| base | 0.528                    | 0.506                        | -0.022 | same model |
| 28   | 0.454                    | 0.530                        | +0.076 | 7.3e-04   |
| 57   | 0.440                    | 0.530                        | +0.090 | 1.7e-04   |
| 86   | 0.466                    | 0.526                        | +0.060 | 4.1e-03   |
| 115  | 0.364                    | 0.536                        | +0.172 | 2.1e-13   |
| 144  | 0.334                    | 0.522                        | +0.188 | 9.3e-15   |

Significance improves from 4/5 to **5/5** at the larger n, as it should.

**The noise floor rose and that qualifies the smallest effect.** The base model appears in
both series, so its difference is pure measurement noise: 0.0080 at n=250, but **0.0220** at
n=500. That is the wrong direction for a larger sample and the reason is procedural, not
statistical -- step0h's base was scored in TWO HALVES at different times on different server
processes, so it accumulates between-run variance that step0d's single sweep does not. A
merged series is not the same measurement as a contiguous one.

Consequences, stated rather than buried:

* The +0.060 at step 86 is only 2.7x the noise floor and should not be reported as a
  standalone result.
* The +0.172 and +0.188 at steps 115 and 144 are ~8x it and survive comfortably.
* The direction is unchanged at every step, and the claim remains PRESERVATION, not
  improvement: scaffold sits at 0.506-0.536 throughout, which brackets its own base.

**The lesson for future sweeps:** score a series in ONE pass whenever the comparison depends
on the base as an anchor. Splitting to fit a GPU budget buys statistics and costs
comparability, and here it cost more than it bought.
