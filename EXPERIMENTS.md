# Experiment log

Measured results and negative results, newest first. A claim only belongs here once it has
been observed end to end; a prediction belongs in GOAL.md until then.

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
