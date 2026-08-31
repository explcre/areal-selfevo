# Collaborator task — Terminal-Bench 2.1 harness-swap at fixed model

**Why this and not the previous task.** The 1.5B routing run you were sent is retired. We
measured three different credit signals producing three very different training trajectories
and **one identical capability outcome** (MATH-500 spread 0.010 against a 0.020 noise floor).
A fourth arm would be a fourth indistinguishable number. This task instead measures something
nobody has reported.

**The question.** For a FIXED model, how much of the achievable gain lives in the *harness*?
LongHorizon-Harness (arXiv 2608.01964, MIT) reports Qwen3.7-Plus going 69.7 -> 77.2 on
Terminal-Bench 2.1 by changing only the scaffold. We want that delta measured ourselves, on a
model we control, because it is the quantity our routing contribution depends on.

## Setup

```bash
git clone https://github.com/AMAP-ML/LongHorizon-Harness.git
cd LongHorizon-Harness/eval/TB-harness
conda env create -f environment-tbench21.yml      # env name: terminal-bench-2-1
```

The repo does NOT ship the benchmark tasks. Obtain Terminal-Bench 2.1 tasks and place them at:

```
datasets/terminal-bench-2-1/tasks          # a symlink is fine
```

Read `eval/TB-harness/README.md` fully before running — it is short and it is the source of
truth, not this file.

## What to run — TWO arms, model held fixed

| arm | model | harness |
|-----|-------|---------|
| A (baseline) | your served model | the model's default / native agent loop |
| B (swap) | **the same** served model | LongHorizon-Harness MEA loop |

The whole value is that A and B differ in the harness ONLY. Same model, same weights, same
decoding parameters, same task set, same number of attempts. If you change anything else, the
comparison is not interpretable and we would have to discard it.

## What to send back

1. **Both arms' PassRate**, with the number of tasks attempted and completed.
2. **The raw per-task results**, not just the aggregate — we need per-task outcomes to compute
   a paired test. An aggregate difference without per-task pairing cannot be tested properly.
3. **The exact model, decoding settings and harness versions** for each arm (git SHA of
   LongHorizon-Harness, model path/revision).
4. **Anything that differed between the arms besides the harness**, even if it seems minor.
   A stated confound is far more useful to us than a clean-looking number.
5. Failures and crashes, with logs. Negative and partial results are wanted.

## What NOT to do

- Do not tune either arm. We want the out-of-the-box delta, not a tuned one.
- Do not drop tasks that error — report them as failures. Silently excluding hard tasks
  inflates the score, and we have been bitten by exactly that.
- Do not run only arm B. A swap result without its baseline measures nothing.

## Known issue in the OLD script, now fixed

If you still have `run_portable.sh`, note it had a broken GPU-pin check: a quoting bug made
bash emit `syntax error near unexpected token '('` at line 226 and then CONTINUE, so the pin
validation silently never ran. That is fixed on `selfevo/a100`. It is unrelated to this task
but worth knowing if you reuse the script.
