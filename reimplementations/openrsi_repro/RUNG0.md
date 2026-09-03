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

## UPDATE 2026-09-03: the control-token patch was applied, and it is NOT sufficient

Per instruction we took the patch route and did **not** pin NatureBench to the pre-fix
commit, because pinning re-opens control endpoints to model-generated code on a box that is
also running other people's jobs.

**The change we made, declared.** `patches/0001-send-naturebench-control-token.patch`
(60 lines, verified to apply cleanly to a pristine OpenRSI `1f477c48`). It adds a module
helper `_load_naturebench_control_token()` and makes `base_task._post_json` attach

    X-NatureBench-Control-Token: <token>

reading `NATUREBENCH_CONTROL_TOKEN`, else the file named by
`NATUREBENCH_CONTROL_TOKEN_PATH` (the service auto-creates it at
`NatureBench/eval_logs/eval_control_token`). It returns `""` and changes nothing when no
token is configured, so a service that does not gate control endpoints is unaffected. Both
insertion sites are bracketed by `LOCAL MODIFICATION, NOT UPSTREAM` comments.

**It works, as far as it goes.** After the patch the eval service logs:

    "POST /register HTTP/1.1" 200
    "POST /start_timer HTTP/1.1" 200
    "POST /evaluate HTTP/1.1" 400

so the authentication half of the skew is closed. Registration and the timer now succeed.

**But `/evaluate` changed shape in the same hardening, and that is a second, larger break.**
NatureBench `/evaluate` now takes `{"eval_token": "<opaque per-task token>"}` and resolves
the token to a registered task. OpenRSI still posts the pre-hardening body
`{"task_name", "batch_name", "output_dir"}`, so the service answers
`400 missing eval_token`. `git log -S eval_token` shows the token API arrived in the same
two commits as the gate (`d38eb9c`, `5304604` "enforce source-paper isolation").

**Why we did not just add the token and continue.** Under the new API the service evaluates
`<registered out_dir>/workspace/output`, taken from the state bound at registration --- it
no longer honours a per-call `output_dir`. Those are different directories in a real run:

    registered out_dir : .../rung0b_out/workspaces/_eval_service/rung0_smoke_patched/s42256-023-00611-x
    attempt output dir : .../rung0b_out/workspaces/s42256-023-00611-x_validation_1/output

OpenRSI registers once per task (`_ensure_eval_service_registered` latches on
`self._eval_service_registered`) but evaluates once per attempt, passing that attempt's
directory. Simply adding an `eval_token` and dropping `output_dir` would make the evaluator
score a fixed path the candidate never wrote to. That is a silent-wrong-number, not an
error: it would return a score, and the score would be of the wrong directory.

Making this correct requires re-registering per attempt with a constructed `out_dir` (the
`force_eval_register` flag suggests re-registration is anticipated) and threading a token
through the attempt lifecycle. That is roughly 30-50 lines of inferred semantics, not one
header, and the inference is exactly where a wrong number would enter. **So we stopped
here rather than expand the deviation unilaterally.**

## Scope correction, stated plainly

Option 2 was chosen on the premise that it was "one header, a smaller deviation". That
premise turned out to be false: the token gate and an `/evaluate` API change shipped
together. The security argument for not pinning is untouched and we still endorse it --- we
are not proposing a pin. The question is only how much of OpenRSI's client we are willing to
rewrite, and that is a call for the PI.

| option | size | risk |
|---|---|---|
| 2a. token + per-attempt re-registration | ~30-50 lines of inferred lifecycle | a mis-bound `out_dir` scores the wrong directory and returns a plausible number |
| 2b. ask the OpenRSI authors / open an issue | zero code | slow, but this is their bug and a fix would help everyone |
| 2c. run NatureBench's own `run_naturebench.py` end to end instead | unknown | tests NatureBench, not OpenMLE-Evo, so it does not serve our comparison |
| 2d. MLE-Bench route instead of NatureBench | large | avoids this skew entirely, but is the 1584-sandbox-hour path |

If 2a is chosen, the mitigation is an assertion that the directory the service evaluated is
the directory the attempt just wrote, checked on every call rather than assumed.

## What their harness got RIGHT (checked, not assumed)

`summary.csv` shows `grade_avg@k = 1.0` next to `success_rate = 0.0` and `unknown = 1`.
That is **not** a phantom perfect score. In `eval_utils.build_submit_grade_and_medal`,
grade is a normalised leaderboard rank where **lower is better** (`submit_grade_best` is a
`min`), and `1.0` with medal `N/A` is the explicit fallback for `submit_score is None`. The
failure is recorded conservatively and flagged in three columns. Their handling is correct.

## Reproducing what we ran

`repro_rung0.sh`. Export `NATUREBENCH_CONTROL_TOKEN_PATH=~/NatureBench/eval_logs/eval_control_token`
and apply `patches/0001-send-naturebench-control-token.patch` first.
