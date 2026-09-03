# Rung 0 (OpenRSI one-sample end-to-end): ATTEMPTED, BLOCKED at the reward stage

Date 2026-09-03. Machine: H200 box, GPU 3 only (GPUs 0-2 were another agent's; 3-7 idle).
No other machine touched.

## What ran

| stage | result |
|---|---|
| task construction | OK — `s42256-023-00611-x` built from the public HF task package |
| eval service start | OK — listening on 127.0.0.1:8321, `/health` 200 |
| prompt / scaffold | OK — full task brief, data description, runtime contract assembled |
| generation | **OK** — `Frontis-MA1-30B` returned a real 5338-byte `run.py` (LightGBM solution), not a template |
| candidate materialised | OK — written to `workspaces/..._validation_1/run.py` |
| **evaluation / reward** | **FAILED — HTTP 404 `unknown endpoint` on `POST /register`** |

Evidence in `rung0_evidence/`. Rung 1 was NOT started, per the pre-agreed stop rule that
rung 0 must yield a read-back artifact carrying a per-stage reward.

## Root cause: version skew between two first-party repos

* NatureBench `d38eb9c` (2026-08-28) *"fix(evaluation): protect control endpoints with
  service token"* put `/register`, `/start_timer`, `/resume_timer`, `/pause_timer` behind
  `_has_control_access()`, which returns **404** (masking the endpoint) unless the caller
  sends `CONTROL_TOKEN_HEADER` matching a token the service auto-creates at
  `NatureBench/eval_logs/eval_control_token`.
* The check does **not** exist at `d38eb9c^`.
* OpenRSI at HEAD `1f477c48` has no notion of it: `base_task._post_json` sends only
  `Content-Type: application/json`, and `control_token` / `eval_control_token` appear
  **nowhere** in `OpenMLE-Evo`.
* NatureBench HEAD is `cd9de65` (2026-08-31), three days after the breaking commit.

So OpenRSI's own documented NatureBench quickstart cannot complete a single task against
NatureBench HEAD today. This says nothing about the validity of their published numbers,
which predate the skew.

## What their harness got RIGHT (checked, not assumed)

`summary.csv` shows `grade_avg@k = 1.0` next to `success_rate = 0.0` and `unknown = 1`.
That is **not** a phantom perfect score. In `eval_utils.build_submit_grade_and_medal`,
grade is a normalised leaderboard rank where **lower is better** (`submit_grade_best` is a
`min`), and `1.0` with medal `N/A` is the explicit fallback for `submit_score is None`. The
failure is recorded conservatively and flagged in three columns. Their handling is correct.

## Two ways forward — NOT equivalent, PI's choice

1. **Pin NatureBench to `d38eb9c^`.** Reproduces the stack OpenRSI was written against, and
   runs their code unmodified. But it re-opens control endpoints to model-generated code,
   which is precisely the exposure their fix closed. Poor idea on a shared box.
2. **Teach OpenRSI's client to send the token.** A one-header change in `_post_json`. Cheap
   and safe, but we are then no longer running their code unmodified, which has to be
   declared wherever the result is reported.

## Reproducing what we ran

`repro_rung0.sh` in this directory. Serving is one command and takes ~2 minutes to load.
