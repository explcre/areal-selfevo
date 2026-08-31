
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

---

## OlympiadBench: the collapse replicates, and the frontier tier finally has a target

Two results from scoring on OlympiadBench (675 problems, harder than AIME and large enough
to resolve a few points).

**1. The step0d collapse is not a MATH-500 artefact.** The same six checkpoints plus the
base, on a benchmark that shares no problems with MATH-500:

| checkpoint | MATH-500 | OlympiadBench |
|------------|----------|---------------|
| base       | 0.528    | 0.188 |
| gs028      | 0.454    | 0.170 |
| gs057      | 0.440    | 0.190 |
| gs086      | 0.466    | 0.148 |
| gs115      | 0.364    | 0.135 |
| gs144      | 0.334    | 0.108 |
| gs173      | 0.358    | 0.107 |

0.188 -> 0.107 is a 43% relative drop, same direction as MATH-500's 0.528 -> 0.358, on an
independent problem set. A finding that reproduces on a second benchmark is worth more than
the same finding measured more precisely on the first.

OlympiadBench is also *more* sensitive to the degeneration: gs173 truncates 328 of 675
generations (49%) against MATH-500's 60%, and the accuracy floor is proportionally lower.

**2. OlympiadBench is the frontier target AIME could not be.** Ornith-1.5-35B-A3B scores
**0.716** with a 95% interval only 7 points wide, against AIME24's 0.867 with an interval 24
points wide on 30 problems. That is ~28 points of headroom measured at a resolution that can
actually see a routing effect.

Stated as a bound, not a score: 174 of 675 generations (26%) hit the 24576-token cap, so
0.716 is a LOWER bound and the true headroom is smaller than 28 points. The number to quote
after a higher-budget rerun, not this one.

**Reading across the three scales**, and this is what makes the benchmark usable: base 1.5B
0.188, Ornith-35B-A3B 0.716. Both ends are far from the ceiling and far from the floor, which
is the property MATH-500 (0.966 at 27B) and AIME (0.000 at 1.5B) each lack at one end.

---

## The routing rule requires adv_norm.mean_level=group, and raw GRPO is not enough

The token-level rule needs `sum_i A_i = 0` within each GRPO group. An audit measured that
the repo's live setting breaks it; this measures every available setting to find one that
does not. Two groups of two with UNEQUAL generation lengths -- the case that breaks
token-weighted centring -- run through AReaL's own `Normalization`:

| mean_level | std_level | max per-group \|sum A_i\| |
|------------|-----------|---------------------------|
| (raw GRPO, no adv_norm) | -- | **2.0000** |
| batch | batch / group / None | 0.8102 / 1.1379 / 1.2308 |
| **group** | batch / group / None | **0.0000 / 0.0000 / 0.0000** |
| None | batch / group / None | 1.3093 / 0.8944 / 2.0000 |

**Only `mean_level=group` works, and it works exactly, at every `std_level`.** That is a
supported AReaL setting, not a hack, so the rule has a valid configuration to run in.

**The result that matters more: raw GRPO does NOT satisfy the precondition.** With
`adv_norm` disabled entirely the group sum is 2.0000, not 0. GRPO centres the per-sequence
REWARD, which gives `sum_i A_i = 0` counted per sequence; the prefix-cancellation argument
needs it counted per TOKEN, and those differ whenever members generate different numbers of
tokens -- which is always. The rule was therefore never valid under the plain recipe, and
the framing "GRPO advantages sum to zero, so shared prefixes are dead" is wrong as stated
for any real batch.

Restated correctly: shared-prefix tokens are RL-dead when the advantages sum to zero **in
the token-weighted sense**, which requires group-level mean normalisation. Anything else --
batch normalisation, or none at all -- leaves a live gradient there.

**Consequences.**

* Any run that enables token routing must set `actor.adv_norm.mean_level=group`. The guards
  already refuse otherwise, so this cannot be forgotten silently.
* That is itself a deviation from the demo recipe and must be measured, not assumed
  harmless: a `mean_level=group` arm with routing OFF is the control the routed arm needs.
  Without it, any gain would be attributable to the normalisation change rather than to
  routing.
* `kl_ctl` must also be 0. It was measured to break the sum independently by 14-43%, and
  the reference default is 0.0 -- our 0.01 was our own addition.

The cheapest useful next run is therefore three arms, not two: demo baseline,
`mean_level=group` with routing off, and `mean_level=group` with routing on.

---

## The first routed run, and how much of the batch the rule actually reaches

step0j is the first run in this project with token routing active. It took six attempts;
every failure was a real defect and every one was caught by a guard rather than by producing
a wrong number: missing `cu_seqlens`, group ids shaped per-sequence instead of per-token,
`adv_norm` set to the value that breaks the precondition, GRPO groups split across
microbatches, and a Hydra override that could not reach `mb_spec`.

**How much is routed, measured rather than assumed:**

| step | routed fraction | entropy | task reward |
|------|-----------------|---------|-------------|
| 13   | 0.0047          | 3.939   | 0.730 |
| 25   | 0.0051          | 3.175   | 0.738 |
| 37   | 0.0084          | 2.087   | 0.793 |
| 49   | 0.0154          | 1.457   | 0.775 |
| 61   | 0.0166          | 1.140   | 0.791 |

**Two things worth stating plainly.**

**1. The reach is small.** At most 1.7% of loss-carrying tokens are RL-dead under this rule.
Whatever the method does, it cannot do much through a 1.7% channel, and any downstream
difference larger than that needs a different explanation. This is a ceiling on the
token-level claim and it should be reported as one rather than discovered by a reviewer.

**2. The reach GROWS as entropy falls -- 3.5x while entropy drops from 3.94 to 1.14.** That
is the mechanism behaving as the theory predicts: group members agree on longer prefixes as
the policy sharpens, so more positions become provably RL-dead. The rule becomes most active
exactly when the policy is degenerating, which is when redirecting that budget is most
valuable. It also means the fraction measured early understates the fraction late, and a
short run understates the method.

**What this arm is NOT.** With no teacher configured, `teacher_logp` is absent and the KD
branch never runs, so routed tokens lose their RL weight and gain nothing. Measured
separately by an audit: routing removes 19.5% of the gradient norm and adds nothing back.
So step0j is a *gradient-deletion ablation* -- "withhold RL from provably dead tokens" -- and
not the RL-to-teacher redirect the method claims. The full claim needs a teacher arm.

**Trajectory health, for context, not as a result.** At step 61 entropy is 1.14 and reward
0.791, against step0d's collapse to 0.018 entropy with directionless reward. That is almost
certainly the conservative recipe (lr 1e-6, eps_clip 0.2, kl_ctl 0, adv_norm off) rather
than the 1.7% routing channel, and the three-arm design exists precisely so the two cannot
be confused.

---

## NEGATIVE RESULT: token-level routing has no measurable effect at 1.7% reach

The first held-out measurement of a routed run. step0j (routing ON) scored on full MATH-500
at the steps it shares with the other two arms, paired McNemar against the routing-off arm:

| step | demo (0d) | scaffold (0h) | scaffold+routing (0j) | 0h vs 0j p |
|------|-----------|---------------|------------------------|------------|
| base | 0.528     | 0.506         | 0.536                  | 0.092 |
| 28   | 0.454     | 0.530         | 0.516                  | 0.534 |
| 57   | 0.440     | 0.530         | 0.526                  | 0.920 |
| 86   | 0.466     | 0.526         | 0.522                  | 0.923 |
| 115  | 0.364     | 0.536         | 0.496                  | 0.065 |
| 144  | 0.334     | 0.522         | 0.526                  | 0.922 |

**Routing changes nothing detectable.** Every p is >= 0.065 and most sit near 0.92. The
entire distance from the demo recipe is explained by the conservative optimiser settings,
which the scaffold arm already has without any routing.

**This was predicted and should be read as consistent, not disappointing.** The rule reaches
at most 1.7% of loss-carrying tokens, and an intervention on 1.7% of the gradient cannot
move a benchmark by more than noise. The measurement confirms the ceiling rather than
discovering a new problem. It also means any future arm claiming a routing effect larger
than this must first explain what changed about the reach.

**The between-arm noise floor is larger than the effect.** The base model appears in both
the 0h and 0j series and scores 0.506 and 0.536 -- a 0.030 gap on the same checkpoint, wider
than any routing difference in the table. No conclusion at this reach can survive that.

**Confound, stated rather than buried.** step0j differs from step0h in THREE ways, not one:
routing, `kl_ctl` (0.01 -> 0) and `adv_norm` (batch -> off). Those had to move together
because the rule's precondition does not hold otherwise. So even a significant difference
here could not have been attributed to routing alone. The clean control is a routing-OFF arm
at step0j's exact config, and it does not exist yet -- that is the single most useful run to
add next, and it costs one training run.

**What would make the token tier worth keeping.** Either raise the reach (route on the
group-silence channel, where whole unanimous groups contribute zero gradient rather than 1.7%
of tokens), or supply a teacher so routed tokens gain a signal instead of merely losing one.
The present arm deletes 19.5% of gradient norm and adds nothing back, which is a
gradient-deletion ablation, not the method.

---

## 57% of groups are RL-silent -- 34x the token-level channel

Measured live on step0l, 64 groups per batch:

    silent_group_fraction   = 0.574
    n_groups                = 64

**More than half of every batch contributes exactly zero gradient.** A group whose members
all score alike has every advantage identically zero, so the entire group -- all eight
sequences, every token -- is RL-dead. Against the token-level shared-prefix rule's 1.7%,
this channel is **34 times larger**.

That reframes where the method's leverage is. The token tier was measured to have no
detectable effect on held-out capability, and the 1.7% reach explains why: an intervention on
1.7% of the gradient cannot move a benchmark. A 57% channel can.

**It also explains the negative result without appealing to noise.** Nothing was wrong with
the token rule; it was simply operating on the wrong tier.

**A bug in the accompanying split, found and fixed.** The metric also reported
`solved_group_fraction = 0.000` and `unsolved_group_fraction = 0.574` -- 0% solved at a
measured 82% solve rate, which is impossible. Cause: the split thresholded `reward_score`,
which by that point has had bias, scaling, clipping AND reward_norm applied, so a 0.5
threshold classifies nothing. It now uses the raw `data["rewards"]` captured before any
transformation. The runs in flight carry the old code, so their solved/unsolved numbers
should be ignored; `silent_group_fraction` is unaffected because it is computed from the
advantages being zero, which is correct either way.

**Why the split matters and is not bookkeeping.** The two silent regimes need OPPOSITE
responses, and their sum hides that:

* silent because SOLVED (p_hat = 1): the model already answers it. Spend less compute here;
  adding SFT would sharpen an already-correct policy and burn entropy for nothing.
* silent because UNSOLVED (p_hat = 0): the model cannot answer it. RL has nothing to push
  on, but a teacher would -- this is precisely the case distillation exists for.

Until the fixed metric runs, we know 57% of groups are silent but not how that splits. That
number decides which signal the 57% should be routed to, and it is the next measurement.

---

## GPU utilization is not a liveness signal: a dead run held 4 GPUs at 100% for 46 minutes

`step0l` raised at step 44 and then sat there. `nvidia-smi` reported **100% utilization on
all four training GPUs the entire time**, which is what a routine check reads as "healthy and
busy". It was a corpse: the log had not grown in 46 minutes.

    log mtime 21:15:50, checked 22:01:49   -> 2759s stale
    20s byte-delta on the log              -> 0

**Cause.** The rollout server dropped during a weight push:

    HTTPUtils WARNING: HTTP request to <addr>/update_weights_from_distributed failed with
    ServerDisconnectedError: Server disconnected (attempt 1/1)
    RuntimeError: Failed after 1 retries each.

`attempt 1/1` is the whole story. `areal/infra/remote_inf_engine.py` hardcodes
`max_retries=1` at both weight-sync call sites, and `DEFAULT_RETRIES = 1`. A single transient
disconnect during weight sync therefore aborts a multi-hour run. The rank that raised exits;
the other three stay in their collective and spin, which is what produces 100% utilization
with zero progress.

**Why this matters beyond one run.** Every "are the GPUs busy?" check in this project has
been reading `nvidia-smi`. That check cannot distinguish training from a dead NCCL wait. The
last 46 minutes of GPU time on this box were reported as fully utilized and produced nothing.

**Three fixes, all landed.**

1. `_weight_sync_retries()` reads `AREAL_WEIGHT_SYNC_RETRIES`, default **1**. The default
   reproduces upstream exactly, so this is a rollback-clean change; runs opt in with the env
   var. Current runs use 5.
2. `experiments/harness/supervise.sh` watches **log growth**, not utilization: if
   `train.log` is stale for more than `stall_seconds` (default 1200) it dumps py-spy stacks
   for the actor ranks, kills the run, and relaunches, up to `max_restarts`.
3. `step0l.sh` takes `$EXTRA_ARGS`, so the supervisor can add `recover.mode=auto` on a
   restart without a second copy of the script.

**One honest note on the restart.** `recover.mode=auto` did *not* resume from the saved
`globalstep28`; the run restarted at step 0. AReaL's recover checkpoints are a different
artefact from the saver checkpoints, and `auto` finds nothing when only the latter exist. For
this particular run that is the better outcome anyway -- step0l is the routing-off control
for step0j, and a contiguous fresh run is comparable to step0j's in a way a resumed run with
reset Adam moments would not be. Recorded rather than glossed, because the flag did not do
what its name suggests.

**Standing consequence.** Liveness is log growth. Utilization is at best a secondary signal
and at worst, as here, a confident lie.

---

## The silent channel is 87.5% SOLVED -- so most of it needs no teacher at all

The measurement the critical path was waiting on. `step0l` (routing OFF, the control arm)
relaunched on the corrected metric, 18 logged batches at 64 groups each = 1152 group
observations:

| quantity | mean | range |
|---|---|---|
| `silent_group_fraction`   | 0.3592 | [0.2891, 0.4414] |
| `solved_group_fraction`   | 0.3145 | [0.2461, 0.4219] |
| `unsolved_group_fraction` | 0.0447 | [0.0117, 0.0703] |
| **solved share of the silent channel** | **0.875** | [0.827, 0.959] |

**Internal consistency, checked rather than assumed.** `solved + unsolved == silent` holds
with a maximum residual of `1.0e-05` across all 18 batches, which is float32 rounding. The
earlier version of this metric failed exactly this check (it reported 0.000 solved at an 82%
solve rate), so the check is the first thing that runs on it now.

**This re-orders the critical path.** The standing plan had "supply a teacher" as item 1, on
the reasoning that without a teacher every routed arm is gradient deletion. That reasoning
was right about the mechanism and wrong about the magnitude:

* **31.4% of all groups are silent because they are SOLVED.** Those groups contain at least
  one correct sample, so a supervised target already exists inside the rollout. This is
  rejection-sampling fine-tuning and it costs no teacher, no extra inference, and no extra
  GPU. It is 87.5% of the silent channel.
* **4.5% of all groups are silent because they are UNSOLVED.** Only these need an external
  teacher -- and they are precisely the units the harness arm is for, since with no target
  and no gradient the harness is the only consumer that can use them at all.

So the external teacher is a lever on ~4.5% of groups, not on the whole silent channel. It
drops from critical-path item 1 to a smaller, later one, and the self-target path takes its
place. That is a 7x difference in reach between the two options, and we had it backwards.

**Scope, stated because it limits the claim.** This is GSM8K, Qwen2.5-1.5B-Instruct, early
training (the first ~18 logged batches), where the solve rate is high. A harder task or a
weaker model moves mass from solved to unsolved and the balance shifts toward needing a
teacher. The 87.5% is a property of this operating point, not a constant, and the split
should be re-measured on OlympiadBench before any claim leans on it.

**What it does not say.** Nothing here shows that training on the solved half helps. There is
a real argument that it hurts: SFT on a policy's own correct output sharpens an
already-correct distribution and spends entropy, and entropy collapse is the failure mode
this project has already measured twice. The measurement says where the reachable mass is,
not what to do with it. The A/B that answers that is the next run, and until it exists the
solved branch should default to SKIP rather than SFT.

---

## PREDECLARED: step0m, the solved-branch A/B (written before any outcome is known)

Registered before the run produces a checkpoint, so the reading of the result cannot be
chosen after seeing it.

**Question.** 31.4% of all groups are RL-silent because they are SOLVED, and those groups
already contain a correct sample. Does training on that free self-target help held-out
accuracy, or does it merely sharpen an already-correct policy and spend entropy?

**Design.** One script, two arms, `ARM=off|on`. Identical model
(Qwen2.5-1.5B-Instruct), data, seed, optimiser, normalisation and token budget; the arms
differ in exactly one flag, and the off arm is bit-identical to vanilla GRPO because
`group_routing` defaults to None. `solved_advantage=0.5`, chosen as half a typical
standardised advantage -- a deliberate first value, NOT tuned, and it will be reported that
way. Arms run sequentially on the same 8 H200s, so hardware is matched.

**Preconditions, all verified before launch rather than assumed:**

* `preflight_group_routing.py` resolves the config through the trainer's own loader and
  confirms the override arrives as a real `GroupRoutingConfig` with both values intact, that
  the control arm is still exactly `None`, and that the sign guard raises through the CLI.
* 17 tests drive the real `PPOActor._compute_advantages`; 7/7 mutations killed.
* The run must emit a NON-ZERO `routed_group_fraction`. If it is zero, the arm did nothing
  and no comparison may be reported.

**What each outcome means, fixed now:**

| outcome on held-out MATH-500, paired McNemar | reading |
|---|---|
| ON > OFF, p < 0.05, effect above the noise floor | the free self-target is usable; this is the method's main positive result |
| ON < OFF significantly | sharpening a correct policy costs capability. A real finding, and the one the entropy argument predicts |
| no significant difference | the 31.4% channel is reachable but inert at this weight. Report as a null, then vary `solved_advantage` ONCE before abandoning |

**The noise floor applies and is not negotiable.** The base checkpoint appears in both arms;
its difference is measurement noise and bounds what may be claimed. On the last comparison
that floor was 0.022 at n=500, so an effect below ~0.03 will not be reported as real
regardless of its p-value.

**Entropy is a co-primary readout, not a diagnostic.** The mechanism's predicted failure is
entropy collapse. Entropy will be reported for both arms whatever the accuracy shows,
because an accuracy win bought with collapsed entropy is not a win -- that is exactly the
pattern the demo-vs-scaffold comparison already caught once.

**Known limitation, stated in advance.** This is GSM8K training at 1.5B. The 87.5%-solved
split is a property of that operating point; a harder task moves mass to the unsolved branch
where no free target exists. A positive result here does NOT transfer to the frontier tier
without re-measuring the split there.

---

## The silent channel GROWS as training proceeds -- on both arms, so it is not our doing

Both live runs, same 1.5B model and config, differing only in `group_routing`:

| run | arm | batches | silent (mean / last) | solved (mean / last) | unsolved (mean / last) |
|---|---|---|---|---|---|
| step0l (A100)    | OFF | 87 | 0.518 / 0.609 | 0.481 / 0.598 | 0.037 / 0.012 |
| step0m-on (H200) | ON  | 45 | -- / --       | 0.450 / 0.602 | 0.037 / 0.020 |

**The solved fraction rises from ~0.27 to ~0.60 within 60-90 batches on the OFF arm.** That
matters more than the level. As the policy improves, more groups become unanimous, so a
larger share of every batch has identically-zero advantages. GRPO's *effective* batch shrinks
as the model gets better.

**Checked before claiming it.** The obvious alternative explanation was that self-SFT on
solved groups sharpens the policy and manufactures its own unanimity -- a self-reinforcing
loop that would look like progress and be an entropy-collapse warning instead. The OFF arm
rules that out: it rises just as much with no routing at all. The growth is a property of RL
training, not of the intervention.

**Why this is the motivation rather than a curiosity.** The channel the method acts on is
*largest exactly where RL is weakest*, and it widens over a run. Extrapolated to the regime
frontier models occupy -- high solve rates -- vanilla GRPO spends most of its batch computing
zeros. That is the opposite of the usual trick, which helps most on weak models and washes
out at scale.

**One difference NOT to read as a result.** At comparable batch index the ON arm sits higher
than the OFF arm (~0.60 vs ~0.49 at batch 45). That is confounded -- different box, different
data order, different step count -- and the matched OFF arm on the same H200 is the run that
settles it. Recorded as a hypothesis to test, not a finding.

**What still decides the paper.** Composition, not size. Here the channel is solved-dominated
(87.5%), where targets are free. On a harder training set the same channel would be
unsolved-dominated, where no target exists without a teacher or a harness. A router that keys
on the SIDE of silence adapts automatically across that shift; that claim is untested and is
the next experiment worth designing.

---

## The right baseline is DAPO, and its cost on our own measurements is 1.5x-2.4x and rising

**DAPO acts on exactly the set this method acts on.** Its dynamic sampling keeps a prompt
only when the group's reward standard deviation is positive (verl,
`recipe/dapo` / `meds_ray_trainer.py`):

    kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()
                        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]

`std == 0` is unanimity, which is precisely the zero-advantage condition. So DAPO *detects*
the silent channel and **discards** it, oversampling until the batch refills. We *reuse* it.
Same phenomenon, opposite response -- which makes DAPO the baseline the claim must be
measured against, not a related method to cite.

**Its cost, computed from our own silent fractions on the control arm (98 batches):**

| point in run | silent fraction s | groups DAPO must generate per accepted group, 1/(1-s) |
|---|---|---|
| first 10 batches | 0.327 | **1.49x** |
| whole run so far | 0.525 | **2.10x** |
| last 10 batches  | 0.583 | **2.40x** |
| max observed     | 0.633 | **2.72x** |

Ours is **1.00x at every point**. Late in the run DAPO pays ~140 extra generations per 100
accepted groups.

**And the gap widens with training,** because the silent fraction grows as the policy
improves (0.33 -> 0.58 within 98 batches, measured on the arm with no routing at all, so the
growth is not our doing). A method whose overhead *increases* as the model gets better is
exactly the wrong shape for the frontier regime, where solve rates are high.

**Stated as a prediction, not a result.** 1/(1-s) is the expectation under the assumption
that regenerated groups are unanimous at the same rate. That assumption is testable and the
instrumentation for it already exists -- AReaL emits `rollout/accepted` and
`rollout/rejected` -- so the DAPO arm will measure the true multiplier rather than inherit
this estimate. If regenerated groups are *less* unanimous the real cost is lower, and this
table is an upper bound; if the sampler is biased toward the same easy prompts it is a lower
bound. Recorded before running so the number cannot be chosen afterwards.

**The comparison axis is matched compute, not matched steps.** DAPO's kept batch is all
informative, so per-step its gradient is denser; ours trains on the informative groups
normally and additionally converts the silent ones. Comparing at equal step counts would
flatter whichever arm sees more tokens. Equal generation budget is the honest axis, and it
is the axis on which the 1.49-2.40x matters.

**Implementation.** `selfevo/baselines/dapo.py` implements the rule against verl's own code,
using population standard deviation to match `np.std` and keeping the singleton carve-out.
It plugs into AReaL's existing `should_accept_fn` hook -- AReaL already ships `dynamic_bs`
and a DAPO config, but NOT the dynamic-sampling criterion itself, so the hook was there with
nothing in it. Default `dynamic_filter_fn=None` leaves upstream behaviour untouched.

---

## CORRECTIONS to the DAPO entry above, both found by audit before any run

Appended rather than edited into the entry above, per this file's rule.

### C1. `dynamic_bs` does the OPPOSITE of what the config help said, and DAPO needs it OFF

The help text written for `dynamic_filter_fn` claimed rejection "only has an effect together
with `dynamic_bs=true`, which is what makes the batch refill." That is backwards. From
`areal/infra/workflow_executor.py:740-757`:

    if not is_accepted:
        if dynamic_bs:
            total_attempts += 1
            if total_attempts >= batch_size: break     # counts ATTEMPTS
        continue
    ...
    if dynamic_bs:
        if total_attempts >= batch_size: break         # accepted + rejected
    elif accepted_cnt >= batch_size: break             # keeps generating until FULL

So `dynamic_bs=True` stops after `batch_size` *attempts* and returns a **shrunken** batch of
whatever was accepted; `dynamic_bs=False` keeps generating until `batch_size` are **accepted**,
which is DAPO's oversampling. Measured on the real dispatcher with half the groups unanimous
and `batch_size=8`: `dynamic_bs=False` returned 8 groups, `dynamic_bs=True` returned 4.

**Consequence had this not been caught:** the DAPO arm would have trained on half-size
batches with no oversampling — a hobbled baseline, and a hobbled baseline makes our own
result worthless. The correct arm is
`+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling` with `dynamic_bs` left at
its default `false`. Note that upstream's own `examples/math/gsm8k_dapo_dynamic_bs.yaml` sets
`dynamic_bs: true` with no filter, which is the wrong combination on both counts.

### C2. The rejection count did NOT reach the trainer, so the cost was not measurable

The entry above states that the instrumentation for measuring DAPO's true multiplier
"already exists" because AReaL emits `rollout/accepted` and `rollout/rejected`. That was
wrong, and wrong in the way this project keeps re-learning: the call existed, the *value* did
not arrive.

Both are recorded through `stats_tracker` as SCALARs, so the exported figure is the mean of a
stream of ones — `1.0` — and the actual number lives in a `__count` key. Under the
single-controller layout these runs use, `RolloutController.export_stats` used every
`__count` only as a denominator and then dropped it:

    worker-side : {'rollout/rejected': 1.0, 'rollout/rejected__count': 5, ...}
    trainer saw : {'rollout/accepted': 1.0, 'rollout/rejected': 1.0}
    trainer now : {..., 'rollout/accepted__count': 6, 'rollout/rejected__count': 10}

Fixed additively (`final_stats.update(counts)`); `tests/test_rollout_controller.py` still
passes 55/55. Without this the 1.49-2.40x prediction could not have been checked against
anything, and the ratio 1.0/1.0 would have looked like a measurement.

### C3. Smaller divergences from verl, and one gap left open

* An EMPTY group was accepted (`numel() <= 1` swept it into the singleton carve-out). verl
  drops it: `np.std([])` is NaN and `len([]) == 1` is False. Fixed to `n_samples == 1`.
* Population vs sample std does **not** change any accept decision (proven by re-running the
  suite with `unbiased=True`: all decision tests still pass); it changes only the reported
  value, which is now pinned against `np.std`.
* `traj["rewards"]` corresponds to verl's `seq_reward` / `acc`, not `seq_final_reward`. It is
  the raw scalar from the reward function, before `reward_bias`/`reward_scaling`. AReaL has
  no `seq_final_reward` analogue because it applies KL as a loss term rather than folding it
  into the reward. MEDS' own run script filters on `acc`, which for a binary math reward is
  the same number.
* **Left open:** verl raises after `max_num_gen_batches` (10) regenerated batches. We have no
  equivalent, so a dataset on which every group is unanimous would regenerate forever rather
  than fail. Recorded, not fixed — it needs a decision about what the right behaviour is.
* Also unguarded: a multi-turn agent emits one interaction per turn, so `len(rewards)` would
  exceed `n_samples` and the std would mix turns with samples. The configured `MathAgent` is
  single-turn, so it does not bite today.

**Test state:** `selfevo/tests/test_dapo_baseline.py`, 48 tests; suite 293 -> 341 passing;
9/9 mutations killed with no survivors, run against a copy of the repo rather than the live
checkout so a mutated file could never be imported by the running job.

---

## INTERIM (not a result): the ON arm sharpens faster mid-run, then lands on the same floor

Predeclared as a co-primary readout, so it is reported while the runs are still going rather
than after the accuracy is known. Entropy (`ppo_actor/update/entropy/avg`) compared at
MATCHED BATCH INDEX -- the series decays steeply, so comparing run means would have been an
artefact of the arms having different batch counts (149 vs 129), and the first version of
this comparison did exactly that before being redone.

| batch window | OFF (step0l) | ON (step0m-on) | ON - OFF |
|---|---|---|---|
| 1-20   | 4.0997 | 3.5938 | -0.5059 |
| 21-50  | 3.1099 | 1.0616 | **-2.0483** |
| 51-80  | 1.6503 | 0.3744 | **-1.2759** |
| 81-129 | 0.2228 | 0.2460 | +0.0231 |
| all    | 1.8273 | 0.9846 | -0.8427 |

ON is below OFF in 88/129 matched batches (68.2%). At the final matched batch the two are
level: OFF 0.1940, ON 0.2272.

**Reading, with the ambiguity left in.** Training on a group's own correct samples makes the
policy sharpen faster in the middle of the run -- which is what the mechanism predicts and
what the entropy-collapse worry is about. But the effect is transient: both arms reach the
same floor near 0.22, and by the end the ON arm sits marginally *higher*. So this is not
differential collapse.

Faster entropy decay is consistent with BOTH readings -- a policy that is learning faster
and one that is collapsing faster -- and entropy alone cannot separate them. Held-out
accuracy is what disambiguates, and that is the A/B's job. Recording the observation now
without choosing a reading for it.

**Confound, stated.** Different boxes (A100 vs H200), different data order, different step
counts. The matched OFF arm on the same H200 is the run that settles it, and it is queued
behind the ON arm. Until then this is an indication, not a comparison.

---

## PROTOCOL DECISION: the A/B arms run 5 epochs, and the ON arm was stopped at 177

Recorded because changing run lengths mid-experiment is the kind of thing that must be
written down rather than remembered, and because it could otherwise look like a run was
stopped once its numbers were known.

**What changed.** `step0m-on` was launched for the config default of 10 epochs (290 steps).
It was stopped at step 177, and the remaining arms (`step0m-off`, then the DAPO arm) run
`total_train_epochs=5` (~145 steps).

**Why 145 and not 290.** The established comparison protocol in this project scores steps
**28, 57, 86, 115, 144** -- it is `compare_runs.py`'s default and the step set used for every
prior arm (step0d, step0h, step0j). 5 epochs reaches 145, which covers all five. Steps beyond
144 are not used by any comparison, so running to 290 would have cost roughly 11 more GPU-hours
-- the DAPO arm especially, since it must generate ~2.4x the rollouts -- for points nothing
reads.

**Why this is not stopping a run once its numbers are known.** No held-out score existed for
any `step0m` checkpoint at the time of the decision, and none exists now; nothing has been
graded. The stop point was chosen from the checkpoint LIST, verified to contain all five
comparison steps before anything was killed:

    24 28 49 57 74 86 99 115 124 144 149 173 174     <- step0m-on, gs28/57/86/115/144 all present

**What it costs.** The ON arm has data past 144 that the other arms will not have, so any
claim about longer-horizon behaviour is unavailable and will not be made. The entropy
comparison already reported is at matched batch index within the shared range, so it is
unaffected.

**What it buys.** Three arms on identical hardware, contemporaneous within a few hours,
instead of one arm with a long tail and two arms that might not finish before the box is
reclaimed. Given both machines are rented and can be taken back at short notice, an
incomplete three-arm comparison would have been worth less than a complete one.

---

## Code-as-policy: what the AST allowlist cannot do, and what closes it instead

`evolve_policy="learned_code"` means a generated Python function decides the routing. An
adversarial audit ran ~150 hostile policies against the validator. Recorded because the
result is a bound on what static validation can achieve, not a bug list.

**Closed by the allowlist** (all pinned by tests, so widening the list fails one): imports by
any route; `__builtins__` / `__globals__` / `__class__` / `__subclasses__` recovery;
attribute access; `eval`/`exec`/`compile`/`open`/`getattr`/`type`; f-strings, walrus, lambda,
comprehensions, generators, `global`/`nonlocal`, `del`, `assert`, `raise`, `try`, `with`,
`yield`, `await`, `match`, class definitions. No `BaseException` is reachable, so nothing can
slip past the `except Exception` on the call path. NFKC identifier normalisation is caught --
`_＿builtins＿_` arrives already normalised and hits the dunder check.

**Closed during the audit**, four of them, each an evaluation site the checks did not cover:
decorators (`@len` above `def route`, which raised out of the *constructor* rather than as
`PolicyRejected`); nested function definitions; **nested defaults** -- `def inner(x=9 ** 9 ** 9)`
was accepted AND fired; and deep expression nesting, where a `RecursionError` escaped a
validator that only caught `SyntaxError`.

One of those four was inert only by accident: an annotation `def route(features: 9 ** 9 ** 9)`
did not evaluate because `compile()` inherited `from __future__ import annotations` from this
module's own frame. Recompiled with `dont_inherit=True` the identical source hangs at
construction. The flag is now passed explicitly rather than inherited, which is the
difference between a property and a coincidence.

**NOT closable by an AST rule, and stated as a limit:**

    def route(features): return "rl" if 9 ** 9 ** 9 else "skip"             # one opcode
    def route(features): return "rl" if len("%999999999d" % 1) else "skip"  # ~1 GB from 30 chars
    def route(features): return "rl" if len([0] * 10 ** 9) else "skip"      # 8 GB

None uses a loop, so rejecting loops does not help, and all three need only `Pow`/`Mult`/`Mod`,
which ordinary policies use. An in-process timeout cannot help either: a signal handler runs
*between* bytecodes and `9 ** 9 ** 9` is a single one.

**What closes it: vetting in a subprocess, and measuring rather than surviving.**
`selfevo/routing/policy_vetting.py` runs a candidate under kernel-enforced `RLIMIT_CPU` and
`RLIMIT_AS`, once, before a run adopts it. The accepted policy then runs in-process at full
speed, so the cost is one subprocess per candidate, not per decision.

The first version only asked "did it survive the limit", and that was not enough --
`"%999999999d" % 1` allocates ~0.93 GiB and PASSED under a 1 GiB cap. A policy that fits just
under the cap then pays that cost on every group of every batch. The child now measures its
own peak RSS and CPU after imports, so cost is a number rather than a verdict:

    good           ok=True   rss=  0.0 MiB   cpu=0.00s
    9**9**9        ok=False  killed by signal 9 (RLIMIT_CPU)
    "%999999999d"  ok=False  MemoryError
    [0]*10**9      ok=False  MemoryError
    "x"*10**8      ok=False  rss= 95.6 MiB   -- COMPLETED, rejected on measured cost

The last row is the point. It is the only one a pass/fail limit lets through.

**A defect of mine the audit found, worth recording separately.** The teacher guard did not
apply to the fallback. `CodePolicyRouter(fallback="sft")` on a unit with no target emitted
`sft` on all three rejection paths -- including the path that logs `"code policy chose sft
with no target"`, increments `teacher_blocked`, and then emitted `sft` anyway. The guard
defeated itself and said so in the log while doing it. It now degrades to SKIP.

Also fixed: a `reason` string carrying an unbounded repr (a policy returning `"x" * 10 ** 7`
produced a 10,000,044-character reason that `RoutingDecision` documents as going into logs);
`allowed_modes=()` silently meaning "every mode" via `or`; and a rejection message that
reported `async def route` as "1 top-level statements".

609 tests passing. 26/26 mutants killed, after a first pass left one survivor -- sharing a
single `_SAFE_BUILTINS` dict between policies instead of copying it, so one policy could
poison another's builtins.

---

## The stall watchdog detected a dead run and then killed nothing, for two independent reasons

`step0l` stalled at step 213. The watchdog fired correctly on log-stall -- the mechanism
added earlier this session worked -- and then failed at both of the things it does next.

**1. The kill pattern matched nothing.** `supervise.sh` ran

    pkill -u "$USER" -f "experiment-name ${TAG}"        # hyphen, space

against processes whose command line reads

    experiment_name=step0l                              # underscore, equals

so it killed nothing at all. Four and a half hours later the original trainer was still
alive, the supervisor had started a second attempt beside it, and the two collided in
`/tmp/areal/name_resolve/ubuntu/step0l` -- which is why the relaunch produced 16 minutes of
startup warnings and no step. A watchdog that logs `WATCHDOG: stalled` and then leaves the
run running is worse than no watchdog: the log says the situation was handled.

Fixed to `experiment_name=${TAG}`, plus the launch script name, plus an explicit sweep for
`inference_service.sglang.launch_server` -- sglang servers carry no experiment name on their
command line, so they survive any experiment-scoped pattern and hold ~118 GB of GPU memory
into the next attempt. The watchdog now logs how many processes still match after the kill,
so a future mismatch is visible instead of silent.

**2. py-spy could not attach.** Every dump returned

    Permission Denied: Try running again with elevated permissions

so the stall was killed with no stack trace, and the cause of THIS stall is unrecoverable --
recorded as unknown rather than guessed at. `/proc/<pid>/wchan` and `/proc/<pid>/status` are
always readable and distinguish a process blocked in a collective from one blocked on I/O, so
they are captured alongside py-spy now.

**Why this matters beyond one run.** The earlier entry in this file records the opposite
failure -- a watchdog that would have killed a HEALTHY run. Together they are the same
lesson: the guard was never exercised against the case it exists for. The detection half was
verified when it was written (a stale log does fire it); the kill half never was, because
verifying it requires actually killing something.

**Cost.** ~4.5 hours of A100 time on a run that was dead, and step0l ends at step 213 of 290.
Its comparison points (28/57/86/115/144) are all present, so nothing the analysis needs is
lost -- but that is luck, not design.

---

## FIRST MEASUREMENT of DAPO's cost, and a confirmation that its filter does what it claims

Both arms on the same 8xH200, same model, same data, differing in one flag.

**1. The filter removes exactly the silent groups.** In the DAPO arm,
`silent_group_fraction = 0.0000` on every one of the first 12 logged batches, against ~0.47
on the OFF arm at the same point. Dynamic sampling rejects a group iff its reward std is
zero, and zero std is unanimity, which is the zero-advantage condition -- so a DAPO training
batch contains no RL-silent groups at all. That is the mechanism working, measured rather
than assumed, and it is the cleanest possible confirmation that DAPO and this method act on
the same set.

**2. Cost, measured as wall-clock per accepted step:**

| arm | steps | intervals | median s/step | mean s/step |
|---|---|---|---|---|
| off (vanilla GRPO) | 1-145 | 144 | **38.0** | 38.6 |
| dapo               | 1-14  | 13  | **70.0** | 55.7 |

**Ratio on medians: 1.84x.** Predicted from the OFF arm's early silent fraction
(s = 0.327, so 1/(1-s) = 1.49x). Measured is higher, which is expected: wall-clock includes
the filtering pass and partially-filled batches, not only the regeneration the 1/(1-s) model
counts. So 1/(1-s) is a LOWER bound on the practical cost, not an estimate of it.

**Preliminary, and here is why.** n = 13 intervals for DAPO against 144 for OFF, and the DAPO
arm is at step 14 of 145. The multiplier should GROW over the run, because the silent
fraction grows (0.33 -> 0.58 measured on the control), so this early number is the smallest
it will be. Its mean (55.7s) sitting below its median (70.0s) also says the early intervals
are not yet stable. To be re-measured at completion.

**Why wall-clock rather than a generation count.** The generation counters do not reach the
trainer in this controller layout. `RolloutController.export_stats` was fixed to stop
dropping the `__count` keys, and that fix IS present on this box, but `rollout/accepted__count`
never appears in the log -- so the run uses a stats path the fix does not cover. Recorded as
an open gap rather than papered over, since an earlier entry in this file already claims that
instrumentation was repaired: it was repaired in the place I looked, and that place is not
the one this run uses.

Wall-clock is arguably the better headline anyway. It is what a practitioner pays, it needs
no counter to be exported, and it captures the filtering overhead a pure generation count
would miss. The generation count remains worth having because it separates "regenerating" from
"slower for some other reason", and that separation is not yet made.

---

## The actor now calls a Router, and what that exposed

`group_routing.router` names a key in `ROUTERS`. When set, the router decides the mode for
EVERY group from observability features; when unset, the previous fixed rule applies. This
is the wiring that turns `router=contextual` and `router=code_policy` from registry entries
into arms, and it was the single step GOAL.md named as blocking six PARTIAL rows.

Verified through the real `PPOActor._compute_advantages`, one silent group and one
informative group:

| router | silent group | informative group | differs from vanilla |
|---|---|---|---|
| none (fixed rule) | +0.500 | -0.866 | yes |
| `coharness` | **+0.500** | -0.866 | **yes** |
| `solve_rate` | 0.000 | -0.866 | no |
| `static` | 0.000 | -0.866 | no |
| `contextual` (untrained) | 0.000 | -0.866 | no |
| any, `enabled=False` | -- | -- | **bit-identical** |

**Three routers changing nothing is the correct result, not a broken seam.** `solve_rate`
sends a solved group to SKIP, and SKIP on an already-silent group is a no-op by construction;
`static` returns RL; an untrained `contextual` has every arm tied at zero and breaks the tie
by name. Only `coharness` chooses SFT on a self-target, and it is the one that moves the
tensor. Chasing this down was worth more than a green test would have been: the seam looked
inert and was not.

**The gap this exposes, stated plainly.** `contextual` can now ACT in a real run and cannot
LEARN in one. Nothing computes a `DecisionOutcome` and calls `observe()` -- the feedback
channel has a consumer and no producer, exactly mirroring how `RoutingContext.extra` had a
consumer and no producer until this session. So a run configured with `router=contextual`
today is a fixed policy with a UCB tie-break, and `validate()` already refuses that
combination unless `require_feedback=True`, which is the right guard and is not yet
satisfiable.

That makes the ordering explicit: the outcome producer is now the single blocking step, as
the actor call was before it. Until it exists, the learned arms are not runnable as learned
arms, and no result may describe them that way.

---

## The learned controller now learns in the real loop -- and then stops, for a structural reason

The feedback channel had a consumer and no producer. `selfevo/routing/outcomes.py` is the
producer, and the actor now closes the loop: it observes the previous batch before routing
the next one, because a decision's outcome is not observable until after the update it took
part in.

Driven through the real `PPOActor._compute_advantages`, 8 batches:

    batch 0: routed=2  updates=0  modes=[rl, sft]
    batch 1: routed=4  updates=2  modes=[rl, skip]
    ...
    batch 5: routed=12 updates=10 modes=[rl, skip]
    batch 6: routed=14 updates=12 modes=[sft]        <- cold start ends, UCB takes over
    batch 7: routed=16 updates=12 modes=[sft]        <- updates STOP

**Two findings, and the second is the important one.**

**1. Without a cold start the controller provably never learns.** Every arm begins at
`theta = 0` with the same `A`, so every UCB score is identical, the tie breaks
deterministically by mode name, and every unit in the batch takes the SAME mode. A
batch-level scalar over a single-mode batch carries no comparative information and is refused
by `batch_outcomes`, so no update arrives and the arms stay tied forever. This is not a
theoretical worry: driven through the actor loop, the update count sat at zero indefinitely.
`cold_start_rounds` cycles the arms round-robin for the first N calls, which makes the early
batches mixed and attributable. Round-robin rather than a random tie-break so a routing
ablation stays reproducible; the default is 0, which preserves the previously pinned
behaviour exactly.

**2. Once the controller converges, it stops receiving information.** At batch 6 UCB takes
over, picks the same mode for every unit, and the batch becomes single-mode again -- so
`batch_outcomes` refuses it and `updates` freezes at 12. This is not a bug in any component:
it is what batch-level attribution *is*. A controller learns only from batches in which its
own decisions disagreed, and a converged controller stops producing those.

**Consequence for the design, stated before running the arm rather than after.** With
batch-level credit, a learned controller needs *within-batch* mode diversity indefinitely,
not just at the start. Two ways out, and they are not equivalent:

* keep forcing a fraction of units per batch onto a non-argmax mode -- cheap, keeps the
  channel open, and costs whatever those units would have gained from the better mode;
* attribute per unit instead of per batch -- strictly better information, and it needs a
  counterfactual the pipeline does not currently produce.

Until one of them exists, a `router=contextual` run learns during cold start and is a fixed
policy afterwards. `feedback/confounded_skips` counts the refused batches, so this is visible
in a run rather than inferred -- a controller that has silently stopped learning looks exactly
like one that has finished learning, and the counter is what separates them.

---

## RESULT: the free self-target is reachable and INERT. step0m answers its own predeclaration

The A/B predeclared earlier in this file has run. Both arms on the same 8xH200, same model,
data, seed and optimiser, differing in one flag; scored on the full MATH-500, paired McNemar.

| checkpoint | off (control) | on (solved to SFT) | diff | McNemar p | diff minus base offset |
|---|---|---|---|---|---|
| **base** (identical weights) | 0.5360 | 0.5160 | **-0.0200** | 0.245 | -- noise floor |
| gs028 | 0.5400 | 0.5180 | -0.0220 | 0.289 | -0.002 |
| gs057 | 0.5220 | 0.5140 | -0.0080 | 0.738 | +0.012 |
| gs086 | 0.5360 | 0.5160 | -0.0200 | 0.378 | 0.000 |
| gs115 | 0.5360 | 0.5160 | -0.0200 | 0.373 | 0.000 |
| gs144 | 0.5400 | 0.5180 | -0.0220 | 0.347 | -0.002 |

n = 500 problems, paired. Significant at **0 of 5** steps.

**The base checkpoint is the whole story.** It is the same weights in both sweeps, so its
-0.0200 is pure measurement noise -- two sweeps sample at temperature and disagree. Every
arm difference is that same -0.020, and the difference-in-differences is **0.000 to 0.002 at
four of five steps**. After removing the sweep offset the intervention did nothing at all.
This is a cleaner null than a table of insignificant p-values, because it identifies exactly
what the observed gap was: not a weak effect, but the offset that was there before training
started.

**It is not a reach problem.** `routed_group_fraction` was measured at 0.23 rising to 0.60,
and `routed == solved` exactly. The intervention touched 24-60% of every batch. It reached
the units; reaching them changed nothing on held-out accuracy.

**The likely reason, and it is not encouraging for this branch.** For a solved group the
model ALREADY assigns high probability to the correct answers -- that is what made the group
solved. SFT raises that probability further, PPO's clip bounds how far, and the marginal
effect on problems the model has not seen is nil. Training on what a model already gets right
teaches it what it already knows. The measured entropy behaviour agrees: the ON arm sharpened
faster mid-run and landed on the same floor, i.e. the intervention changed the optimisation
PATH and not the destination.

**What this does to the claim.** The silent channel is 87.5% solved, and the solved part is
now measured to be worth nothing at this weight. That moves the value to the UNSOLVED part --
which is exactly the part with no self-target, where a teacher or the harness is the only
available consumer, and which is only ~4.5% of groups on GSM8K. On a harder task that share
grows. So the honest reading is that this operating point (an easy task, a high solve rate)
is the WORST case for the method, not the best, and the composition-flip prediction is now
the load-bearing claim rather than a nice-to-have.

**Predeclared next step, and it was fixed in advance:** "report as a null, then vary
`solved_advantage` ONCE before abandoning". Running `solved_advantage=2.0` -- 4x the typical
standardised advantage rather than half of it. If that is also null, the solved branch is
abandoned on evidence rather than on taste.

**One methodological note worth keeping.** The full-500 comparison was used rather than the
250-problem report half, and that is legitimate HERE only because this A/B searched nothing:
the arms differ in one flag fixed before either ran. The protocol number on the report half is
also null (0/5 significant, base gap 0.036), and is in `compare_runs.py` output.

---

## Greedy scoring is only 17% reproducible, and that is the floor under every comparison here

Chasing the A/B's noise floor produced a result that bounds every number this project reports.

The base checkpoint -- **identical weights** -- was scored twice, greedy (`--temperature 0.0`,
`--n 1`), same box, same harness, same code. Comparing the two runs problem by problem:

| quantity | agreement |
|---|---|
| identical generated TEXT | **87 / 500 (17.4%)** |
| identical boxed answer | 304 / 500 (60.8%) |
| identical grade | 440 / 500 (88.0%) |
| accuracy | 0.5360 vs 0.5160, gap **0.0200** |

**The grader is not the problem, and I checked rather than assumed.** It is deterministic:
500/500 self-consistent on identical input, and 500/500 of the stored flags agree with a
fresh regrade of their own text. Of the 60 problems graded differently, **all 60 also had
different text**; zero cases of same-text-different-grade.

**So the generation is nondeterministic at temperature 0.** Four of five completions differ
between two greedy runs of the same weights. `math_bench.py` already warns about this in a
comment -- "avg@n at temperature 0 measures batching nondeterminism, not model uncertainty" --
but the size of it was not measured until now. Batch composition and scheduling change the
floating-point reduction order in the sglang kernels, which flips an argmax somewhere, and the
completion diverges from that token on.

**Consequences, in order of how much they matter:**

1. **Every single-run accuracy in this project carries roughly ±0.02 of systematic
   uncertainty that is not binomial and does not shrink with more problems.** A reported
   0.536 could have been 0.516 on a rerun of the same weights.
2. **Unpaired comparisons of two scorings are close to worthless at this effect size**, and
   several earlier entries in this file compare arms scored in separate sweeps. Their paired
   McNemar numbers survive; any difference-of-two-accuracies reading of them does not.
3. **The A/B null stands and is in fact strengthened.** Paired McNemar is the right test
   precisely because it conditions on the problem: the base pairing was 35/25 discordant,
   p=0.245, i.e. the churn is symmetric. Every arm difference matched the base offset to
   within 0.002, which is what "no effect" looks like once this floor is understood.
4. **What would actually reduce it**: score each checkpoint more than once and average, fix
   the server's batch composition, or compare only paired. Averaging over 5 runs is what
   Ornith-1.5's own protocol does, and this is the reason it does it -- a detail worth
   copying rather than treating as ceremony.

**A correction to my own reading an hour earlier.** A first pass at this comparison reported
"100% identical completions, 12% graded differently", which would have meant a nondeterministic
GRADER -- a much worse bug. That was wrong: the script read `completion`/`answer`/`bench`,
while the schema is `text`/`gold`/`benchmark`, so it compared empty strings to empty strings
and every field defaulted. The real numbers are above. The lesson is the same one this file
keeps recording: a comparison that reads a field name that does not exist returns a clean,
confident, meaningless answer.

---

## Smoke test of run_portable.sh: the refusal path works, and two defects only a run would show

Run on the A100 box with all 8 GPUs held by the `sa2` arm -- i.e. the exact shared-box
condition the script exists for. `MIN_GPUS=4`, `WAIT_FOR_GPUS_S=0`.

**What worked, end to end and unattended:**

* cloned/updated the repo at the pinned branch;
* found a usable Python environment rather than building one;
* prefetched the model (7 files) and GSM8K (7473 train / 1319 test) into its own `HF_HOME`;
* detected that no GPU was below the free-memory threshold and **refused with exit 4** instead
  of launching into contention with the neighbouring job;
* wrote the structured manifest **on the failure path**, with `status: "failed"` and the
  reason -- which is the property that matters, since a manifest that only appears on success
  cannot report a failure.

**Defect 1: the manifest silently records empty strings for anything not exported.**

    "solved_advantage": "",   "model": "",   "wandb": {"project": ""}

The manifest heredoc reads `os.environ`, but those values are set as ordinary shell variables
inside the script and never exported. `mode`, `arm` and `run_name` were populated only because
the smoke test happened to export them on the command line. So a collaborator running with
defaults gets a manifest whose most useful fields are blank, and nothing anywhere says so.
This is the same silent-empty shape this file keeps recording: the field exists, the value
does not, and the artefact looks complete.

**Defect 2: `die()` appends the exit code to the human-readable note.**

    "note": "fewer than MIN_GPUS=4 free (raise WAIT_FOR_GPUS_S to wait) 4"

`die` is `die "msg" 4` and passes `"$*"`, which is both arguments. Cosmetic, but the note is
the field a human reads first.

**Why this entry exists.** I had already written that the script "should not be handed over
until it has been run end to end". Both defects are invisible to `bash -n`, invisible to
review, and would have reached a collaborator as a manifest full of blanks. Running it once,
against a genuinely busy box, cost about a minute.

**Still unverified:** the training path itself, the eval path, the retry/re-claim loop, and
`cleanup_ours` sparing a neighbour's sglang server. Those need free GPUs and, for the last
one, a second job to not kill. They are the parts most likely to hurt someone else's work, so
they are the parts that must not be assumed.

---

## The portable runner would have killed a collaborator's job. Twelve defects, all fixed

Audited before handing it to anyone, on a box where a real neighbouring job (`sa2`, all 8
GPUs) was running throughout. The neighbour finished the audit alive, at step 39, with all
four of its sglang servers intact.

**The headline invalidates the design's core safety assumption.** `cleanup_ours` was built on
`pkill -u "$USER"`, on the theory that a collaborator's job runs as a different user. On this
box the neighbouring job runs as **the same user**, and its sglang cmdlines match the pattern
exactly. So `-u $USER` protects nobody, and the only thing standing between this script and
someone else's training run was one `grep` on `CUDA_VISIBLE_DEVICES` -- which was wrong in
BOTH directions:

* **False positive.** The grep was an unanchored SUBSTRING match. `CVD=0,1` matches a
  neighbour on `0,1,2,3`. `CVD=""` matches **everything**, including a neighbour on a
  completely disjoint `4,5,6,7`. Verified read-only against the real neighbour PIDs: with an
  empty CVD, all four of their servers matched.
* **False negative.** AReaL gives each sglang server exactly ONE device id (`=7`, `=5`, ...),
  so our full-list pattern `=0,1,2,3` can never match our OWN servers. On retry they survive
  holding ~20 GB each, the re-claim then sees those GPUs as busy, and the script exits 4 --
  having permanently squatted GPUs on someone else's machine.

And the empty-CVD case was reachable: `claim_gpus` returned an empty list **as a success**.
At `n=1`, `cut -d, -f1-$((n-1))` is `f1-0`, which errors, blanks the variable, and still
`return 0`s -- reachable through the documented `MIN_GPUS=1`. So the catastrophic row above
was one config flag away.

**The other nine, each reproduced:** rounding applied AFTER the MIN test, so `MIN_GPUS=5` with
5 free returned 4; `awk -F", "` misparsed a driver that emits `0,0` without a space, making
`free_gpus` return `0,0,1,0` and claim boards regardless of memory; the manifest recorded
empty strings because the values were never exported; `write_manifest` wrote nothing at all
when python3 was absent, silently, returning success; `MODE=eval` could never find a
checkpoint because the glob searched one level and AReaL nests four; `pip install wandb`
targeted a REUSED environment -- on this box, the venv the live neighbouring job is importing
from; and `USER`/`HOME` unset under cron aborted under `set -u` before anything was logged.

**The trap was the worst of the quiet ones.** `trap ... INT TERM` called `write_manifest` and
did not exit. Confirmed: after SIGTERM the script kept running, `wait` returned 143, and the
retry loop **relaunched the training the operator had just asked to stop**. An unattended
script that cannot be stopped by Ctrl-C on a machine you do not own is worse than one that
crashes.

**What this says about the process.** `bash -n` was clean throughout. shellcheck at `-o all`
was clean and flagged **none** of the twelve -- every one was semantic. The smoke test I ran
earlier caught two (the empty manifest fields and a cosmetic note); the audit caught ten more,
including all the ones that would have hurt a collaborator. Neither alone was sufficient, and
"it ran once and refused correctly" was not evidence of safety.

Fixed: ownership is now set-containment plus an age test rather than a substring grep; an
empty claim owns nothing; rounding precedes the MIN test; awk tolerates both CSV shapes and
validates both fields as integers; manifest-visible variables are exported and a shell-only
JSON fallback covers a missing python3; `on_signal` cleans up and exits 130 and an `on_exit`
fuse covers every remaining path including the lock-busy one; wandb installs only into a venv
we built ourselves.

840 tests (774 + 66 new, none needing a GPU). All 11 fixes were mutation-tested by reverting
them one at a time: 11 caught, 0 survivors.

**Still unverified and stated as such:** that `cleanup_ours` actually reaps a real AReaL run
(the patterns are tested for shape, not effect); the retry loop end to end; and whether
`GPU_FREE_MIB=4096` is high enough, since a neighbour still loading weights can sit under
4 GiB and look free. Reading `--query-compute-apps` would close that last one and is a design
change I did not make unasked.

---

## run_portable.sh passes end to end on 8 GPUs; the partial-box path does not

Run exactly as a collaborator would: fresh `git clone` of the public branch on the H200, then
the script. Nothing pre-staged, no human intervention.

    status         succeeded
    step           29/29
    gpus_claimed   0,1,2,3,4,5,6,7
    checkpoints    2, with full paths
    commit         bf5e30a7
    wandb          {"mode": "offline", "degraded_from_online": true, ...}
    GPUs after     all 8 at 0 MiB

Every field populated -- no empty strings, which is what the pre-audit version produced. The
W&B entry is the honest one: online was requested, no key existed on that box, it degraded to
offline and **said so in the artefact**.

**What the run proved that no amount of reading could:**

* the whole unattended path works from a clean clone -- setup, prefetch, claim, train,
  checkpoint, manifest;
* GPUs are RELEASED on exit. An earlier run of the same script left 113 processes and 4 GPUs
  holding ~131 GB each after giving up, because neither `die` nor `on_exit` called cleanup;
* a second concurrent launch was REFUSED by the flock and wrote
  `collab8.manifest.lockbusy.json` without touching the holder;
* online W&B with no API key does not warn -- it raises inside PPOTrainer construction and
  kills the run before step 1. A collaborator would have hit this first, on a machine we
  cannot debug.

**The partial-box path is NOT verified, and that is the case this script exists for.** With
`MAX_GPUS=4` on the same H200 the run reached `_update_weights_from_distributed` and died in
`dist.broadcast`, with rollout servers at ~140 GB of 141 GB -- no headroom for the broadcast
buffers. The 8-GPU run puts them at 118 GB, so the difference is how AReaL splits train and
rollout when fewer GPUs are claimed, not the script's arithmetic. `MEM_FRACTION` is now passed
explicitly (default 0.8, matching the yaml) so a partial claim can be given less; that
mitigation is untested.

So: usable on a whole box today, and to be handed over for a partial box only after the
4-GPU path is either fixed or the script is made to refuse a claim it cannot serve. Stating
which of the two configurations was actually exercised matters more than the summary
"it works".

---

## The solved branch is abandoned on evidence: inert at 0.5, HARMFUL at 2.0

The predeclaration said "report the null, then vary `solved_advantage` ONCE before
abandoning". Done. Both arms against the same control, full MATH-500, paired McNemar.

| checkpoint | off (control) | sa 0.5 | sa 2.0 | p (2.0 vs off) |
|---|---|---|---|---|
| **base** (same weights) | 0.5360 | 0.5160 | 0.5140 | 0.193 -- noise floor |
| gs028 | 0.5400 | 0.5180 | 0.5420 | 1.00 |
| gs057 | 0.5220 | 0.5140 | 0.5000 | 0.278 |
| gs086 | 0.5360 | 0.5160 | 0.5160 | 0.395 |
| gs115 | 0.5360 | 0.5160 | 0.5380 | 1.00 |
| **gs144** | 0.5400 | 0.5180 | **0.4860** | **0.0141** |

**At 4x the weight the branch stops being inert and starts costing capability.** gs144 is
-0.054 against the control, 2.5x the 0.022 noise floor, and the difference-in-differences
after removing the base offset is **-0.032** -- a real effect, not the sweep offset that
explained every difference in the 0.5 arm.

**Multiplicity, stated rather than omitted.** This is one significant result among five
paired tests. A Bonferroni threshold at 0.05/5 = 0.01 would NOT admit p=0.0141, so on the
p-value alone this is suggestive, not conclusive. Two things argue it is real anyway, and
both were visible before the p-value: the direction is negative or zero at four of five
steps, and the largest effect is at the LAST checkpoint, which is the shape a cumulative
over-sharpening would produce and is not the shape of a random hit.

**Mechanism, consistent with what was already measured.** A solved group is solved because the
model already assigns high probability to the correct answers. A small push adds nothing
(0.5, null). A larger push sharpens an already-correct distribution, spends entropy, and
costs held-out accuracy -- and the entropy trace recorded earlier already showed the routed
arm descending faster mid-run. The two measurements agree without either being fitted to the
other.

**Consequence: the solved branch is abandoned**, on evidence and on a rule fixed in advance
rather than on taste. That is 31.4% of groups -- the larger half of the silent channel -- and
it is worth nothing at best.

**What survives.** The remaining value is in the UNSOLVED half: no self-target, a teacher or
the harness the only possible consumer, and only ~4.5% of groups on GSM8K. So the entire
method now rests on the composition-flip claim -- that a harder task moves mass from solved
to unsolved. That run (MATH-lighteval, routing off, measuring the natural composition) is
live on the A100 as of this hour. If the composition does not flip, the honest paper is a
negative one: a large, cheaply reachable channel in GRPO that is not worth reaching.

---

## THE COMPOSITION FLIPS. Registered in advance, and it holds

The claim the method rests on, after the solved branch was abandoned: *a harder task moves the
silent channel from solved-dominated to unsolved-dominated*. It was written into GOAL.md and
this file BEFORE the run, as the thing that would decide whether the paper is positive or
negative. Measured now on the same model (Qwen2.5-1.5B-Instruct), same config, differing only
in the training task.

| | GSM8K (step0l) | **MATH-lighteval (math-off)** |
|---|---|---|
| batches measured | 98 | 11 |
| `silent_group_fraction` | 0.359 | **0.419** |
| `solved_group_fraction` | 0.315 | **0.164** |
| `unsolved_group_fraction` | 0.045 | **0.255** |
| **solved share of the silent channel** | **87.5%** | **39.1%** |
| **unsolved share** | 12.5% | **60.9%** |

**The channel does not shrink -- it inverts.** Silence stays about the same size (0.36 vs
0.42), but on the harder task the MAJORITY of it is unsolved rather than solved. In absolute
terms the unsolved fraction of all groups goes from 4.5% to 25.5%, a **5.7x** increase, while
the solved fraction roughly halves.

**Why this rescues the method from the negative result.** The solved branch was measured inert
at weight 0.5 and actively harmful at 2.0, and it was 87.5% of the channel on GSM8K -- which
made the whole silent channel look worthless. On MATH that branch is only 39% of it. The
unsolved branch, which has no self-target and for which a teacher or the harness is the ONLY
possible consumer, is now the larger half and 5.7x bigger in absolute terms. The method's
value was always predicted to live there; this is the first evidence that "there" is a real
place rather than a rounding error.

**Stated limits, because n is small and this is the load-bearing result:**

* **n = 11 batches** against 98 for the GSM8K figure. The effect is enormous relative to the
  spread (solved share 39.1% vs 87.5%, with GSM8K's per-batch range never dipping below 82.7%)
  so it is not a sampling artefact, but the MATH numbers will be restated at full n.
* This is ONE model at ONE scale. Whether the flip also tracks model capability is being
  measured now on the H200 (same task, Qwen2.5-7B-Instruct), which is the second axis.
* It says where the reachable mass IS, not that acting on it helps. The solved branch taught
  exactly that lesson: reachable and worthless are compatible. The unsolved branch still has
  no teacher and no harness consumer, so nothing has yet been shown to USE this mass.

**What it changes about the paper.** The honest framing stops being "a large channel that is
not worth reaching" and becomes "a large channel whose composition -- and therefore the right
intervention -- depends on task difficulty, measured across two tasks that invert it".
Routing on the SIDE of silence is what adapts across that shift, and that is a claim a fixed
rule keyed on a single scalar cannot make.

---

## The governing variable is the SOLVE RATE, and the silence is prompt heterogeneity, not binomial tail

Three (model, task) pairs, all Qwen2.5 with $G=8$:

| run | solve rate | silent | solved share of silent |
|---|---|---|---|
| 1.5B on GSM8K | 0.858 | 0.359 | 87.5% |
| 7B on MATH | 0.743 | 0.702 | 81.6% |
| 1.5B on MATH | 0.474 | 0.411 | 38.2% |

**Composition is monotone in the solve rate, not in the task and not in the model.** The
earlier framing -- "a harder task flips the composition" -- was half the picture. A harder
task lowers the solve rate and a stronger model raises it; they push in opposite directions
and what governs the split is where the pair lands. 7B on MATH sits at 0.743 and behaves like
1.5B on GSM8K (0.858), not like 1.5B on MATH (0.474). The variable is EFFECTIVE difficulty.

**The closed form fails, and the failure is the finding.** For homogeneous per-sample success
$p$, $P(\\text{all solved}) = p^G$ and $P(\\text{all unsolved}) = (1-p)^G$, giving:

| run | predicted silent | measured | predicted solved share | measured |
|---|---|---|---|---|
| 1.5B GSM8K | 0.294 | 0.359 | 1.000 | 0.875 |
| 7B MATH | 0.093 | **0.702** | 1.000 | 0.816 |
| 1.5B MATH | 0.008 | **0.411** | 0.303 | 0.382 |

The predicted silent fraction is wrong by up to **50x** (0.008 against 0.411). A homogeneous-$p$
model says a group is unanimous only by sampling luck, which at $p \\approx 0.5$ is nearly
impossible with $G=8$. The measurement says otherwise, so the assumption is what is wrong:
**per-prompt difficulty is strongly bimodal**. Most silent groups come from prompts this model
either always solves or never solves, not from the tail of a binomial.

**Three consequences, and the third is the useful one:**

1. The mean solve rate is a poor summary. Two runs at the same mean can have very different
   silence if their prompt-difficulty distributions differ.
2. The silent channel is a property of the DATA-model pairing, not of sampling. It is stable,
   not noise.
3. **Increasing $G$ will not shrink it.** If silence were binomial tail mass, more samples per
   prompt would collapse it and the whole method would be unnecessary -- just sample more.
   Because it is prompt heterogeneity, more samples per prompt mostly buys more identical
   answers to the prompts that were already unanimous. That is a falsifiable prediction and
   the cheapest experiment left: rerun one config at $G=16$ and check whether the silent
   fraction falls as $p^{16} + (1-p)^{16}$ would demand (it should nearly vanish) or stays
   near its $G=8$ value (heterogeneity).

Until that test runs, "more samples does not fix this" is an inference, not a measurement, and
is labelled as such.
