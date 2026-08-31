# Collaborator task — Terminal-Bench 2.1 harness swap at fixed model

**Honesty first: this has NOT been run end to end here.** I verified every step I could on our
A100 and could not complete one, because Docker is unusable on that box. The prerequisites
below are what I actually established by running things, and the one blocker I hit is stated
as a blocker rather than glossed. Run the preflight before committing any time.

**Why this replaces the 1.5B routing run you were sent.** That task is retired. We measured
three different credit signals producing three very different training trajectories and **one
identical capability outcome** (MATH-500 spread 0.010 against a 0.020 noise floor). A fourth
arm would be a fourth indistinguishable number.

**The question.** For a FIXED model, how much of the achievable gain lives in the *harness*?
LongHorizon-Harness (arXiv 2608.01964, MIT) reports Qwen3.7-Plus 69.7 -> 77.2 on
Terminal-Bench 2.1 by changing only the scaffold. We want that delta measured on a model we
control, because it is the quantity our contribution depends on and neither released paper
reports it.

## PREFLIGHT — run this first, it takes two minutes

```bash
# 1. Docker with a WORKING daemon is a hard requirement (Terminal-Bench runs tasks in
#    containers). This is what blocked us:
docker info >/dev/null 2>&1 && echo "docker OK" || echo "DOCKER BLOCKED - stop here"
#    Ours failed with: permission denied ... unix:///var/run/docker.sock
#    Usually means the user is not in the docker group. Fix before continuing.

# 2. harbor installs in a PLAIN venv - conda is NOT required despite the README.
#    Verified: harbor 0.18.0 installs clean and `harbor --help` works.
python3.12 -m venv ~/tb-env && ~/tb-env/bin/python -m pip install -q harbor==0.18.0
~/tb-env/bin/harbor --help >/dev/null && echo "harbor OK"

# 3. The shipped script REQUIRES both of these or it exits 2 before doing anything:
[ -n "$ANTHROPIC_API_KEY" ] && [ -n "$ANTHROPIC_BASE_URL" ] && echo "anthropic env OK" \
  || echo "MISSING ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL"
```

If any line reports a problem, tell us rather than working around it — the workaround is
probably more interesting to us than the result.

## Setup

```bash
git clone https://github.com/AMAP-ML/LongHorizon-Harness.git
cd LongHorizon-Harness/eval/TB-harness
```

Tasks are NOT shipped. Place Terminal-Bench 2.1 tasks at
`datasets/terminal-bench-2-1/tasks` (a symlink is fine). `harbor download` exists and may
fetch them directly — try it before hunting elsewhere.

Read `eval/TB-harness/README.md` — it is short and is the source of truth over this file.

## What the shipped example actually is

`Scripts/run_tb21_full_cua_harness_claudecode_qwen37_enable_thinking.sh` runs **Claude Code as
the agent harness driving Qwen3.7 through an Anthropic-compatible endpoint**
(`ANTHROPIC_BASE_URL`). It sets `PYTHONPATH=Harness/src` and calls
`harbor run -c <config>.yaml`. So the "LHH arm" is a config, not a bespoke program.

**There is no shipped baseline arm.** This matters: the comparison needs a second config that
runs the SAME model WITHOUT the MEA loop. Defining that is the real work of this task. If the
cleanest baseline available is a different Terminal-Bench agent at the same model and decoding
settings, use it and say exactly what it was.

## The two arms — model held fixed

| arm | model | harness |
|-----|-------|---------|
| A (baseline) | your served model | no MEA loop; simplest agent that completes tasks |
| B (swap) | **the same** model, same endpoint, same decoding | LongHorizon-Harness MEA loop |

A and B must differ in the harness ONLY. Same weights, same decoding parameters, same task
set, same attempt budget. If anything else differs, the comparison is not interpretable.

## What to send back

1. Both arms' **PassRate**, with tasks attempted and completed.
2. **Per-task results, not just aggregates.** We need per-task outcomes to run a paired test;
   an aggregate difference cannot be tested properly.
3. Exact model, decoding settings, LongHorizon-Harness git SHA, harbor version, and both
   config files.
4. **Anything that differed between the arms besides the harness**, however minor. A stated
   confound is worth more to us than a clean number.
5. Failures, crashes and logs. Negative and partial results are wanted.

## What NOT to do

- Do not tune either arm — we want the out-of-the-box delta.
- Do not drop tasks that error; report them as failures. Silently excluding hard tasks
  inflates the score and we have been bitten by exactly that.
- Do not run only arm B. A swap without its baseline measures nothing.

## Unrelated bug you should know about

The old `run_portable.sh` had a broken GPU-pin check: a quoting bug made bash emit
`syntax error near unexpected token '('` at line 226 and then CONTINUE, so pin validation
silently never ran. Fixed on `selfevo/a100`. You found it; thank you.
