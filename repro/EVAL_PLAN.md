# Post-training evaluation plan (science regime)

## The logistics problem
The two science harnesses live on the **local lab machine** at
`~/text-dna/genebench-pro/` (READ-ONLY, 59 MB total). The trained checkpoint lands on the
**A100 box** under `~/areal-runs/`. Three ways to close that gap:

1. Copy the checkpoint to the lab machine and serve it there — needs a SLURM GPU job, and
   moves GBs over the network.
2. Serve on the A100 box and point the local harness at it — requires exposing the sglang
   port publicly. Rejected: needlessly exposes an unauthenticated inference endpoint.
3. **Copy the harness to the A100 box and run everything there.** 59 MB, no exposed ports,
   no large transfer. Chosen.

Copying *from* the read-only tree is a read; nothing in `~/text-dna/genebench-pro/` is
modified. Destination on the A100 box is a NEW folder: `~/evalkits/` (never `~/baselines/`,
which is for third-party clones we did not write).

## Split by judge dependency

| Harness | Grading | Runs fully on the A100 box? |
|---|---|---|
| GeneBench-Pro (`gbp`) | `grading.py`, deterministic | **Yes** — self-contained |
| BioMysteryBench (`bmb`) | `judge.py` shells out to **Codex CLI** | **No** — solve there, judge where Codex is |

So: run `gbp` end-to-end on the A100 box. For `bmb`, run the solve phase on the A100 box
(`--backend served`), copy the transcripts back, and judge on the machine that has Codex
(`scripts/rejudge.py` does exactly this — **verified**: its docstring states grading is
"replayable against saved transcripts", and re-running solvers to re-grade "would also
confound the change with solver variance". Args: `jsonl --dataset --judge-model
--judge-effort --dry-run`.)

## Verified CLI contracts

GeneBench-Pro (`gbp/runner.py`), 10 public problems by default:

    python -m gbp.runner --backend openrouter --base-url http://127.0.0.1:8404/v1 \
      --model <served-name> --attempts 1 --effort high \
      --out ~/runs/eval/gbp_<tag>.jsonl \
      --workspace-root ~/scratch/gbp_ws --transcripts ~/runs/eval/gbp_<tag>_transcripts

`--out` is REQUIRED and is appended to. `--base-url` set => no API key needed
(`runner.py:190`). Other flags: `--problems` (eval_ids, default all 10), `--package`,
`--max-steps`, `--exec-timeout`, `--append`.

BioMysteryBench (`bmb/runner.py`), solve phase:

    python -m bmb.runner --backend served --base-url http://127.0.0.1:8404/v1 \
      --model <served-name> --subset all --attempts 1 --max-steps 25 --workers <n> \
      --out ~/runs/eval/bmb_<tag>.jsonl \
      --workspace-root ~/scratch/bmb_ws --transcripts ~/runs/eval/bmb_<tag>_transcripts

Then judge with `--judge-model gpt-5.6-sol --judge-effort medium` where Codex is available.

## Reporting rules (from the harnesses' own docs)
- Every BioMysteryBench number must state its judge model and effort. `judge.py:22-24`:
  numbers are not exchangeable across judges.
- `runner.py:61` drops attempts with `status != ok` or `judge_error` from the denominator
  and counts them in `excluded`. **Read `excluded` on every run** — a judge outage silently
  shrinks n, which is exactly the silent-zero failure mode to watch for.
- Never pool backends in one number; both runners warn about this explicitly.

## Baseline needed for any claim
A trained-model score is meaningless alone. Run the **same harness, same flags, same judge**
against the untrained base model (Qwen2.5-1.5B-Instruct) first. Note upfront: a 1.5B model
will likely score at or near zero on both of these frontier science benchmarks, in which
case they cannot resolve our method and a larger base model is required. Establish that
floor before designing anything around these two benchmarks.
