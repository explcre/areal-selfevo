# `openrsi_repro` — reproducing OpenRSI / Frontis-MA1

**No OpenRSI code is vendored here.** OpenRSI is CC BY-NC 4.0 with a `NOTICE` file, so it
is cloned to a working directory outside our tree and left unmodified. This directory
holds only *our* notes, run scripts and results, which reference it by upstream commit.

* Upstream: `https://github.com/FrontisAI/OpenRSI` at commit **`1f477c48`** (30 commits, 830 files, 110 MB).
* Licence: **CC BY-NC 4.0**, non-commercial, attribution required. `NOTICE` additionally
  carries upstream terms for THUDM/slime (Apache-2.0), the vendored AIRA-Dojo runtime
  (CC BY-NC 4.0), `wangweiz03/gym`, Kaggle-derived artifacts and the Qwen bases.
  Academic use is fine. **Do not relicense, and do not copy their files into our tree.**
* Paper: *Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in
  Machine Learning Engineering*, **arXiv:2607.28568** (verified to resolve).

## Verified status, 2026-09-02 (CPU only, no GPU used)

Everything below was checked against artifacts, not against the README's prose.

| claim | verdict |
|---|---|
| repo exists, is public and clones | **yes**, 110 MB |
| licence is CC BY-NC 4.0 with a NOTICE | **yes** |
| `OpenMLE-Evo` harness ships | **yes** (294 files) |
| `OpenMLE-Gym` sandbox ships | **yes** — `node_controller`, `node_router`, `node_workers`, `node_client`, with `docker-compose.yaml` and Dockerfiles |
| `OpenMLE-ERL` SFT+RL ships | **yes** (368 files), with slime vendored for SFT |
| package installs and imports | **yes** — `openmle-tts-eval` and `aira-dojo` install editable; `import tts_search` OK |
| their own test suite passes | **yes — 66/66** in `OpenMLE-Evo/tests`, plus **53/53** in the vendored `aira-evo/tests` (119 total) |
| Frontis-MA1-35B / -30B weights on HF | **yes**, both return 200 |
| base `Qwen/Qwen3.6-35B-A3B` on HF | **yes** |
| NatureBench task data | **public and ungated** on HF `FrontisAI/NatureBench` (56,188 files) |
| a single task package downloads | **yes** — `s42256-023-00611-x`, 2.9 MB, and it contains `problem/data`, `evaluation/evaluator.py` **and** `evaluation/ground_truth/` |

**Caveat on the environment:** their `pyproject.toml` declares
`requires-python = ">=3.11,<3.13"`. Python 3.13 is *excluded* and their suite does not
collect under it. We use 3.12.14.

**A false defect I nearly reported.** Running their suite with the interpreter invoked by
a relative path (`../../py312/bin/python`) fails 2 of 66 tests, because `sys.executable`
then contains `..` segments while their code correctly resolves the real path. Their code
is right; the tests are merely brittle to a non-normalised `sys.executable`. With an
absolute interpreter path it is 66/66. Their release is clean.

## Corrections to how the target was described to us

Two things need restating before anyone plans around them.

1. **The published 39.39 → 60.61 delta is a MODEL swap with the harness held FIXED.**
   Both rows run OpenMLE-Evo. It is not a "no-scaffold → scaffold" delta. Reproducing it
   requires serving *two* models and running the search twice. Their own table is explicit
   that these are "model–harness results, not standalone one-shot model scores".
2. **The NatureBench hidden evaluation set is NOT shipped inside OpenRSI.** `NOTICE` says
   NatureBench task packages, hidden evaluation data, the evaluation service and container
   images are not redistributed there; only the adapter, config templates and manifest are.
   The data is obtainable, but from the separate public `FrontisAI/NatureBench` repo and
   HF dataset, not from OpenRSI.

## What reproduction actually costs

Their protocol fixes a **wall-clock** budget per task, so a faster GPU does **not** buy it
down — it only changes how much search fits inside the budget, which would deviate from
the protocol. This is the single most important planning fact here.

MLE-Bench Lite, per their evaluation protocol (22-task official split, three independent
runs, 12 h per-task sandbox budget on one RTX 4090 capped at 12 GB VRAM):

```
per (model, harness) cell : 22 tasks x 3 runs x 12 h        =   792 sandbox-hours
base + Frontis-MA1        : 2 cells                          = 1,584 sandbox-hours  (~66 days on one sandbox)
plus                      : a served 35B-A3B model for the whole duration
plus                      : Docker + Kaggle competition data + leaderboard metadata
```

NatureBench Lite (10 tasks, 4 h effective model+sandbox budget, 6 h wall-clock cap,
<=160 nodes): **40 model+sandbox-hours per cell**, up to 60 h wall-clock. Two cells ~80 h.
This is 20x cheaper than the MLE-Bench route and is the same claim shape.

## Proposed ladder (each rung gates the next). NO GPU USED YET — awaiting approval.

| rung | what | cost | needs |
|---|---|---|---|
| 0 | `run_naturebench_local.py --smoke` — one task, **one candidate** | minutes | 1 GPU serving a model |
| 1 | one full NatureBench task, `s42256-023-00611-x` | <=6 h wall-clock | 1 GPU serving |
| 2 | NatureBench Lite, 10 tasks, base vs Frontis-MA1-35B | ~80–120 h | serving + CPU sandbox |
| 3 | MLE-Bench Lite, the published 39.39 → 60.61 | ~1,584 sandbox-h | Docker, Kaggle data, serving |

Rung 0 is the one-sample end-to-end pass the brief demands: one task, one scaffold, one
rollout, one reward, artifacts written and read back. It is also the cheapest possible
check that the whole stack is wired correctly, and it should run before anything else.

Serving note: Frontis-MA1-35B is a 35B-A3B MoE, roughly 70 GB in bf16 — 2xH100, or one
H200, or one card in FP8. The 30B variant is the cheaper first target.

## The controls still apply

Their published delta is a headline, not a control. At every rung where we make a claim,
the size-matched random control at the treatment's measured proportions comes too. If
their loop ties with random at matched budget, that is the finding and we report it.
