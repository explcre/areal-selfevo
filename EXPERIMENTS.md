# Experiment log

Measured results and negative results, newest first. A claim only belongs here once it has
been observed end to end; a prediction belongs in GOAL.md until then.

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
