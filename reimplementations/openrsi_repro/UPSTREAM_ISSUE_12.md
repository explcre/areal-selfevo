Hi — thanks for releasing the full stack; the code installs cleanly and your own suites pass 119/119 for us (66 in `OpenMLE-Evo/tests`, 53 in the vendored `aira-evo` tests).

While running the documented **NatureBench local quickstart**, we found that it cannot evaluate a task against current NatureBench HEAD. Two independent changes on the NatureBench side are involved, and `OpenMLE-Evo` has not mirrored either. Reporting in case it helps others.

**Versions**
- OpenRSI `1f477c48` (HEAD at time of testing)
- NatureBench `cd9de65` (HEAD), with the relevant changes in `d38eb9c` and `5304604`

**1. `/register` is gated behind a control token the client never sends**

NatureBench `d38eb9c` ("fix(evaluation): protect control endpoints with service token") puts `POST /register`, `/start_timer`, `/resume_timer` and `/pause_timer` behind `_has_control_access()`, which returns **404 `unknown endpoint`** (deliberately masking the endpoint) unless the request carries

```
X-NatureBench-Control-Token: <token>
```

The service auto-creates the token at `NatureBench/eval_logs/eval_control_token`. `base_task._post_json` sends only `Content-Type: application/json`, and `control_token` does not appear anywhere in `OpenMLE-Evo`, so the run fails at registration:

```
RuntimeError: NatureBench eval service /register failed with HTTP 404: {"error": "unknown endpoint"}
```

The check does not exist at `d38eb9c^`.

**2. `/evaluate` also changed shape, so the header alone is not enough**

After supplying the token locally, `/register` and `/start_timer` return 200 and the run reaches `/evaluate`, which then returns:

```
HTTP 400: {"error": "missing eval_token"}
```

`/evaluate` now takes an opaque per-task `{"eval_token": ...}` bound at registration, while `_post_evaluate` still posts the pre-hardening body `{"task_name", "batch_name", "output_dir"}`. `git log -S eval_token -- eval_service.py` places this in the same two commits as the gate.

**3. The part we would flag as the more subtle hazard**

Under the new API the service scores `<registered out_dir>/workspace/output` from state bound at registration, and no longer honours a per-call `output_dir`. `OpenMLE-Evo` registers once per task (`_ensure_eval_service_registered` latches) but evaluates once per attempt, passing that attempt's directory. In our run those were different paths:

```
registered out_dir : .../workspaces/_eval_service/<batch>/<task>
attempt output dir : .../workspaces/<task>_validation_1/output
```

So a minimal fix that just adds `eval_token` and drops `output_dir` would evaluate a directory the candidate never wrote to — and would **succeed**, returning a score for the wrong path rather than an error. We stopped rather than guess the intended lifecycle. If per-attempt re-registration (the `force_eval_register` flag?) is the intended pattern, that would be worth documenting.

**Scope**

This is a compatibility/reproducibility issue only. Your published results predate these NatureBench commits and are unaffected, and nothing here suggests a problem with them. Repro steps and full logs are happy to be shared if useful.
