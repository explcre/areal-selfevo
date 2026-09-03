# selfevo architecture: map, seams, and a plan to extend them

Read-only audit, 2026-09-02. No source file was modified and no job was started or stopped.
The `lora30b` job was live on all eight A100s throughout (step ~755/1160).

Every number here was measured on this box against
`git merge-base HEAD origin/main` = `1488cd43` (upstream AReaL, *perf(engine): stream Megatron
microbatches from CPU (#1622)*). Where a claim is a judgement rather than a measurement, it
says so.

**A note on this checkout.** It is shared and other agents write to it concurrently: during
this audit the working set went from four modified files to clean, and `GOAL.md`,
`selfevo/FINDINGS_cluster_lora.md` and `selfevo/tests/mutate_cluster_lora.py` were rewritten
under us. Every extraction proposed below must re-diff immediately before touching a file,
and every commit must be pathspec-limited.

---

## 0. The safety envelope, measured rather than assumed

This decides what is safe to change today, so it comes first.

`/home/ubuntu/venv312b/lib/python3.12/site-packages/areal.pth` contains one line,
`/home/ubuntu/areal-selfevo/`. It is a **bare path entry, not an import hook** — no `import`
statement, so Python appends the directory to `sys.path` and executes nothing. There is no
`sitecustomize.py` in the venv, and grepping non-test `selfevo/` for `setattr(areal`,
`monkeypatch` and `import_module("areal` returns **zero hits**. Nothing is patched at import
time; `import areal` resolves straight to the working tree. That is why an import-time break
in `areal/` kills the next worker relaunch.

`selfevo/` is a different case, at two levels.

**Static closure.** All 20 `areal → selfevo` import sites are function-local and guarded, so
`selfevo` is reachable only through those guards. Walking the import graph from the 13
`selfevo.*` modules `areal/` names — *including package `__init__.py` side effects, which
matter: `from selfevo.gold.attach import ...` runs `selfevo/gold/__init__.py:33` and therefore
pulls in `substitute.py`* — gives the set of modules a routed run would import.

| Not in the closure (safe to refactor whenever) | In the closure (treat as live) |
|---|---|
| `harness/selectors.py`, `routing/proportions.py`, `routing/credit_sim.py`, `routing/policy_vetting.py`, `baselines/dapo.py`, `cluster_lora/{interference_dump,interference_analyze,merge}.py`, `harness/mini_swe.py`, all of `selfevo/tests/`, all of `experiments/` | `compose.py`, `critics.py`, `meta_critics.py`, `observability.py`, `integration/{group_apply,packed,token_routing}.py`, `gold/{attach,substitute}.py`, `harness/{base,dispatch}.py`, `clustering/meds.py`, `cluster_lora/{wiring,partition,adapters,features,reach,sketch}.py`, and 12 of 16 `routing/*.py` |

**This run specifically.** The guards do not fire:

| Guard | Sites | Live value |
|---|---|---|
| `if "gold_ids" in data` | `workflow_executor.py:1254`, `workflow/rlvr.py:191` | dataset is `openai/gsm8k`, `keep_solution` unset — **false** |
| `if self.harness_variants is not None` | `api/cli_args.py:1989` | `group_routing: null`, so `__post_init__` never runs — **false** |
| `if tr is not None and tr.enabled` | `actor.py:306` | `token_routing: null` — **false** |
| `if gr:` | `actor.py:419-422, 443, 477-478, 586-587` | `group_routing: null` — **false** |
| `os.environ["SELFEVO_CLUSTER_LORA"]` | `actor.py:540` | unset in the live processes — **false** |
| `os.environ["SELFEVO_CLUSTER_LORA_ADAPTERS"]` | `fsdp_engine.py:1293` | unset — **false** |

(`/home/ubuntu/areal-runs/logs/ubuntu/lora30b/t1/config.yaml` lines 556-557 and 738-739.)

So *no* `selfevo` module is imported by the current run at all. **The plan in section 8
nonetheless ranks by the static closure, not by this run's config**, because the config is a
property of one launch and the next one will differ. Treat "this run imports nothing" as
slack, not as licence.

---

## 1. The map

`selfevo/` is 39,586 lines across 130 Python files, of which **25,882 (65%) are tests**.
Non-test `selfevo/` is 13,704 lines. `experiments/` adds 9,666 Python lines across 47 files
plus 40 shell scripts and is our code too.

| Subpackage | Files | LOC | Owns | Depends on |
|---|---|---|---|---|
| `routing/` | 16 | 3,783 | The decision. `RoutingContext` / `RoutingDecision` / the `Router` Protocol, the open `_MODES` registry, and eight routers. Deliberately torch-free so decisions are CPU-testable. | `observability` (×4) |
| `cluster_lora/` | 10 | 4,640 | The current method (M25). Behavioural features, MEDS and random-matched partitions, per-cluster adapter sets, engine wiring, the interference probe. | `clustering` (×5), `routing` (×3), `areal.trainer.ppo.actor` (×1, lazy) |
| `harness/` | 5 | 1,692 | The harness-evolution axis. `HarnessVariant` / `VARIANTS`, `HarnessDispatcher`, the truncation selector and its rate-matched control. | `routing` (×2) |
| `gold/` | 3 | 1,111 | Gold-as-a-batch-row: attach at trajectory width, substitute into qualifying groups, reconcile log-probabilities after `compute_logp`. Pure tensor functions, no engine import. | none |
| `integration/` | 4 | 903 | Turning a decision into tensors: `apply_decisions` / `apply_mixtures` (the `_APPLIED` seam), token-level routing, packed-tensor helpers. | `routing` (×1) |
| `clustering/` | 2 | 236 | MEDS trajectory clustering, vendored verbatim as a feature. | none |
| `baselines/` | 2 | 114 | DAPO dynamic sampling. | none |
| root: `compose`, `critics`, `meta_critics`, `observability` | 5 | 1,225 | The factorial axis registries and their compatibility rules; group feature extraction. | none at module level |
| `tests/` | 83 | 25,882 | see section 6 | — |

The dependency graph is shallow and acyclic, which is the best structural property this tree
has:

```
clustering ──▶ cluster_lora ──▶ routing ──▶ observability
                    ▲              ▲
harness ────────────┘              │
integration ───────────────────────┘
gold      (no selfevo dependencies at all)
compose   (no module-level selfevo dependencies at all — 12 lazy factory functions)
```

`compose.py` having zero module-level `selfevo` imports is deliberate and load-bearing: it
keeps configuration validation importable without dragging in the routing criteria. Keep it.

**Across the vendor boundary the dependency direction is inverted.** `areal/` imports
`selfevo` at 20 sites in 6 files; `selfevo/` imports `areal` at exactly one non-test site
(`selfevo/cluster_lora/wiring.py:389`, `from areal.trainer.ppo.actor import
_infer_prompt_lens`, itself function-local). The vendor tree depends on the fork tree. That is
backwards, and it is the root cause of section 2.

---

## 2. The vendor boundary

`areal/` is upstream AReaL: 467 files, ~129k lines. Our delta against the merge base is
**14 files, +2,020 / −11**. Counting the whole vendor tree (upstream's `tests/` and
`examples/` too) it is **16 files, +3,051 / −11**.

Only **11 lines have been deleted in total** — 99.5% of the delta is insertion, which is the
fork's best property. But an insertion into the *middle* of an upstream function conflicts on
merge just as reliably as a deletion, and **~26 of the 45 `-U0` hunks touch pre-existing
upstream lines**. Only the three new files are conflict-free.

### 2.1 Per-file

| File | +/− | Hunks on existing lines | Class | Could it live in `selfevo/`? |
|---|---|---|---|---|
| `areal/dataset/competition_math.py` | +190/−0 | 0 (new file) | ADDITIVE | **Yes, verbatim.** Nothing requires it to sit in `areal/`. |
| `areal/utils/group_stats.py` | +312/−0 | 0 (new file) | ADDITIVE | **Yes, verbatim — and it is dead.** Its only non-test caller is a `Normalization.__init__` default; nothing ever passes `recorder=`. |
| `tests/test_group_stats.py` | +1,029/−0 | 0 (new file) | ADDITIVE | **Yes.** 1,029 test lines in the *vendor* test tree for a 312-line module with no production consumer. |
| `areal/api/cli_args.py` | +396/−0 | 3 | ADDITIVE (2 new dataclasses = 347 lines, 3 new fields) | **Mostly.** Move `TokenRoutingConfig` / `GroupRoutingConfig` to `selfevo/config.py`; keep one upstream field resolved through the existing `areal.utils.dynamic_import.import_from_string`. 396 → ~8. |
| `areal/trainer/ppo/actor.py` | +819/−1 | 9 | **Mixed: ~490 additive / ~329 structural** | Largely — see 2.2. |
| `areal/engine/fsdp_engine.py` | +165/−5 | 2 | Mixed (one new method, +138 additive; two structural) | **Yes**, behind `import_from_string(engine)`. |
| `areal/utils/data.py` | +31/−1 | 4 | STRUCTURAL (signature change + new call site in `Normalization.forward`) | **Yes, and it should simply go** — nothing constructs `Normalization(cfg, recorder=...)`. |
| `areal/experimental/openai/client.py` | +12/−0 | 9 | STRUCTURAL (3 `__init__` signatures, 2 call sites) | **Upstream PR.** Generic `min_new_tokens` plumbing, not selfevo logic. |
| `areal/infra/remote_inf_engine.py` | +29/−2 | 2 | Mixed | **Upstream PR.** The two `max_retries=1` sites are inside `ProcessPoolExecutor` helpers, so a parent-process patch is unreliable under `spawn`. |
| `areal/infra/controller/rollout_controller.py` | +16/−1 | 3 | STRUCTURAL | **Upstream PR.** Both are upstream bug fixes with measurements attached (647/1024 callbacks lost at `threaded=False`). |
| `areal/dataset/__init__.py` | +12/−0 | 1 | STRUCTURAL — an order-dependent `elif` spliced into a hardcoded chain | **Yes.** Highest conflict probability per line in the fork: upstream has no dataset registry, only an if/elif chain, and our branch must precede `gsm8k`. |
| `areal/workflow/rlvr.py` | +15/−1 | 1 | STRUCTURAL (rewrites `arun_episode`'s return) | **Yes.** `remote_inf_engine.py:749` resolves `workflow` by string; ship `selfevo/workflows/GoldRLVRWorkflow(RLVRWorkflow)`. |
| `areal/infra/workflow_executor.py` | +12/−0 | 1 | STRUCTURAL (import spliced into the trajectory-collection loop) | **Yes.** A wrapping `RolloutWorkflow` reaches both trajectory paths through the same string plug point. |
| `areal/engine/sglang_remote.py` | +10/−0 | 1 | STRUCTURAL-lite (key added to a request dict) | **Upstream PR.** `gconfig.min_new_tokens` was a dead field upstream. |
| `areal/experimental/openai/proxy/proxy_rollout_server.py` | +1/−0 | 1 | STRUCTURAL-lite | **Upstream PR**, same change-set as `client.py`. |
| `examples/math/gsm8k_rl.py` | +2/−0 | 1 | ADDITIVE | Yes, trivially. |

### 2.2 `actor.py` (+819/−1), the largest liability

| Block | Hunk | Upstream function | Class |
|---|---|---|---|
| B1 | `+5` | module | ADDITIVE — `import os` |
| B2 | `+69,201` | module | ADDITIVE — 4 new functions: `_truncated_rows`, `route_all`, `_refuse_dropped_harness`, `_recentre_advantages` |
| B3 | `+301,13` | `PPOActor.__init__` | STRUCTURAL — builds `_token_routing_spec` |
| B4 | `+395,288` | class body | ADDITIVE — new method `_route_groups` (288 lines) |
| B5 | `+707,4` | `_compute_advantages` | STRUCTURAL — inserts `raw_reward` before reward scaling |
| B6 | `+826,243` | `_compute_advantages` | **STRUCTURAL, the worst block** — `group_ids`, silence statistics, the fixed-rule branch, `_route_groups` dispatch, `_recentre_advantages`; reassigns `advantages` and `data["advantages"]` |
| B7 | `+1072,12` | `_compute_advantages` | STRUCTURAL — `data["gen_mask"]` |
| B8 | `+1249,1` | `ppo_update` | STRUCTURAL — adds `token_routing=` to the loss `partial` |
| B9 | `+1317,1` | `grpo_loss_fn` | STRUCTURAL — signature change |
| B10 | `+1345,41` | `grpo_loss_fn` | STRUCTURAL — `route_token_weights`, rescales advantages **inside the per-microbatch hot path** |
| B11 | `−663 +1468,8` | `grpo_loss_fn` | STRUCTURAL — the file's only deletion |
| B12 | `+1487,6` | `grpo_loss_fn` | STRUCTURAL — one stats line |

`FSDPPPOActor` (`fsdp_engine.py:2417`) does `self.actor = PPOActor(config, self)`. A `selfevo`
subclass overriding `_compute_advantages` (call `super()`, then apply B5-B7 post-hoc) and
`ppo_update` absorbs B3-B8 — roughly 750 of the 819 lines. The irreducible remainder is
B9-B12, which sit inside `grpo_loss_fn`'s numerics after packing; the clean answer there is to
**vendor that one function** into `selfevo/` as `routed_grpo_loss_fn` (~120 lines, one
maintainable copy unit) and pass it as the loss — the actor already passes the loss as a
`functools.partial`, so the seam exists.

### 2.3 Upstream seams that exist and were not used

| Seam | Location | Would have absorbed |
|---|---|---|
| `import_from_string(engine)`, validated `issubclass(TrainEngine)` | `areal/infra/rpc/guard/engine_blueprint.py:383` (default at `:345`) — **the live run's own path** | all of `fsdp_engine.py` and most of `actor.py` |
| `import_from_string(workflow)` | `areal/infra/remote_inf_engine.py:749`, `rl_trainer.py:1625` | `rlvr.py` + `workflow_executor.py` |
| `import_from_string(should_accept_fn)` | `remote_inf_engine.py:854`, `rollout_controller.py:846` | already used correctly by `dynamic_filter_fn` — the good example in the tree |
| `import_from_string` for `gae_lambda` | `areal/trainer/ppo/lambda_fn.py:134` | precedent that pluggable-by-string is the house style |

One caveat: `rl_trainer._create_train_engine` (`:1151`) picks `actor_cls` with a hardcoded
if/elif on `alloc.backend`, not a config string. The engine-subclass route therefore needs
either one small upstream field or a module-attribute rebind on `areal.engine` from a real
`.pth` import hook — the import there is function-local, so a rebind works. That single hook,
plus the two workflow plug points above, would displace roughly 1,020 of the 2,020 vendor
lines in `areal/` (`actor.py` 819 + `fsdp_engine.py` 165 + `rlvr.py` 15 + `workflow_executor.py`
12 + `dataset/__init__.py` 12).

### 2.4 Summary

- **1,531 lines (50% of 3,051)** sit in three new files and could move to `selfevo/` today
  with zero behaviour change. **1,341 of those are already dead in production**
  (`group_stats.py` plus its test file).
- **984 lines (32%)** are `actor.py` + `fsdp_engine.py`, movable behind the existing
  `import_from_string` engine seam plus one vendored loss function; another 39 across
  `rlvr.py`, `workflow_executor.py` and `dataset/__init__.py` move behind the workflow and
  dataset plug points.
- **~70 lines (2%)** across `sglang_remote.py`, `openai/client.py`,
  `proxy_rollout_server.py`, `remote_inf_engine.py` and `rollout_controller.py` are **generic
  upstream bug fixes** carried as fork delta for no reason. PR them and delete them here.
- Highest conflict *probability* per line: `areal/dataset/__init__.py`. Highest conflict
  *volume*: `actor.py` block B6.

One inconsistency to fix on sight: `cli_args.py:1676` says the routing spec "is not imported
here to keep `areal.api` free of a selfevo dependency", and `cli_args.py:1989` then does
`from selfevo.harness.base import VARIANTS`. The comment is false as written.

---

## 3. The registries and seams

### 3.1 Inventory — six seams, five shapes

| Seam | Site | Form | Discovery | Unknown key |
|---|---|---|---|---|
| `ROUTERS` | `selfevo/compose.py:209` | `dict[str, Callable \| None]`, 8 literal entries | explicit literal dict | loud, but **only at first batch, on GPU** |
| `VARIANTS` | `selfevo/harness/base.py:111` | `dict[str, HarnessVariant]` + `register_variant()` | **import side effect** | loud, twice, one of them at config time |
| `_APPLIED` | `selfevo/integration/group_apply.py:46` | private 3-**tuple** | literal | loud |
| `TRAINING_PARTITIONS` | `selfevo/cluster_lora/partition.py:63` | module-level **tuple** + a separate if/elif chain | literal | loud for an unknown string, **silent for a known one with no branch** |
| `GoldRule` | `selfevo/gold/substitute.py:166` | closed **Enum** | members | loud |
| `key_fn` | `selfevo/routing/cluster.py:131` | mutable dataclass field | assignment at `wiring.py:504` | n/a |
| `SELECTORS` | — | **does not exist** | — | — |

Six seams, five shapes: dict-of-factories, dict-plus-register-function, module-level tuple,
closed Enum, and a bare callable field. A newcomer cannot learn one pattern and apply it.

### 3.2 The four findings that matter

**(a) `SELECTORS` is documented but absent.** `EXPERIMENTS.md:246` describes
"`GroupRoutingConfig.harness_selector` + `harness_selector_args`, resolved through a
`SELECTORS` registry" with pre-GPU validation. `grep -rn SELECTORS` over every `.py` returns
zero hits, and `GroupRoutingConfig` (`areal/api/cli_args.py:1703`, fields at 1947-1955) has no
such field. Git explains it: `ecf97f84` added it, `3677694c` removed it, and **neither is an
ancestor of HEAD**. The selector *classes* survive (`selfevo/harness/selectors.py:417` and
`:638`), but the only way to install one is the `selector=` argument of `HarnessDispatcher`
(`dispatch.py:284`), which `build_dispatcher` (`:502-532`) never passes. **Production always
gets `round_robin` (`:311`) — the placeholder whose own docstring says it answers a different
question than the one the paper asks.** The rate-matched harness control is unreachable from
any arm.

**(b) `partition_from_config` has a silent fallthrough.** `partition.py:694-709` reads
`if mode == "none": …` / `if mode not in TRAINING_PARTITIONS: raise` / `if mode == "meds":
return meds` / then an unconditional `return random_matched_partition(meds, seed=seed)`.
Adding a third name to `TRAINING_PARTITIONS` *without* adding a branch silently yields
`random_matched` — a new arm running the control's behaviour under its own label. That is
exactly the failure the module docstring exists to prevent, left open in the module that
prevents it.

**(c) Routers get no pre-GPU validation.** `GroupRoutingConfig.__post_init__`
(`cli_args.py:1957-2015`) validates `credit`, `decision` and `harness_variants` and **never
`router`**. The only static check is `compose.py:460` on `PipelineConfig`, which has no
production caller. A `router` typo survives config parse and dies after model load. Contrast
`harness_variants`, which *is* resolved against `VARIANTS` before any GPU is touched — the
right pattern, already in the tree, applied to one axis of four. Two parallel config
dataclasses exist for the same axis: `compose.PipelineConfig.router` (validated, unused) and
`cli_args.GroupRoutingConfig.router` (used, unvalidated).

**(d) `key_fn` has an undeclared second half.** `ClusterRouter` types it as
`Callable[[RoutingContext], str]`, but the only real implementation, `ClusterLoRAKeyFn`
(`cluster_lora/features.py:354`), is a stateful object that *additionally* requires
`begin_batch(unit_ids, features, group_ids=…)` to be driven out of band from
`wiring.begin_cluster_batch` (`wiring.py:539`). A contributor who writes to the declared type
gets a `key_fn` that raises on every lookup.

Minor but worth a one-line fix: `compose.py:7`'s module docstring advertises
`static_* | solve_rate | bandit` against a registry of eight members containing no `bandit`,
and missing five of the eight real names.

### 3.3 Files you must touch to extend

**(a) A ninth router — 2 files mandatory, 4-5 realistic.**

| # | File | Change |
|---|---|---|
| 1 | `selfevo/routing/<new>.py` | NEW. `route(ctx) -> RoutingDecision`; add `route_batch` if it partitions, `observe` if it learns |
| 2 | `selfevo/compose.py` | **two** edits: a `def _<new>_router(**kw)` lazy-import wrapper near `:70-207`, and one line in the `ROUTERS` literal at `:209` |
| 3 | `selfevo/routing/__init__.py` | optional re-export |
| 4 | `selfevo/compose.py:7` | the stale docstring axis table |
| 5 | `GOAL.md` / `EXPERIMENTS.md` | arm listing |

No test edit is needed: `selfevo/tests/test_gold_target_reachability.py:164-187` auto-discovers
through `sorted(compose.ROUTERS.items())`. That is the one place discovery is done right, and
it should be the model for the others.

The one-file ideal is **not** achievable, for two reasons. `register_router` exists at
`compose.py:337` with **zero non-test callers**, and nothing eagerly imports the routing
modules (`routing/__init__.py` imports only `base`, `criteria`, `routers`), so a
self-registering file would be dead code. And `_route_groups` calls `factory()` with **no
kwargs** (`actor.py:432`), so every experiment-deciding default must be baked into the wrapper
— which is precisely why `_random_router` and `_contextual_router` carry retraction comments
at `compose.py:114-137` and `:156-172` recording two arms that ran bit-identical to the off
arm. The wrapper *is* the seam; the registration is a function, not a line.

**(b) A third partition — 1 file mandatory (4 edits), 6 realistic.**

| # | File | Change |
|---|---|---|
| 1 | `selfevo/cluster_lora/partition.py` | `:63` the tuple; a new `<new>_partition() -> Partition`; **`:696-709` an explicit branch before the trailing `return random_matched_partition(...)`**; `:43-56` `__all__`; `:5-25` docstring |
| 2 | `selfevo/cluster_lora/wiring.py` | `:526` the behavioural-forward branch, if the mode does not need features; `:55`, `:81`, `:85` docstrings |
| 3 | `selfevo/cluster_lora/features.py:377` | docstring |
| 4 | `selfevo/cluster_lora/__init__.py` | `:33` `__all__`, `:47` import |
| 5 | `selfevo/tests/test_cluster_lora_partition.py:230` | `parametrize("mode", [...])` |
| 6 | `selfevo/FINDINGS_cluster_lora.md:128` | doc |

Validation *is* centralised here — `wiring.py:119` and `features.py:387` both read
`TRAINING_PARTITIONS` and need no edit. This is the closest axis to the ideal. What stops it
being *safe* is finding (b) above, plus a mode list duplicated into four prose docstrings that
no test checks.

**(c) A new loss mode alongside RL|SFT — 8+ code files and 3 test files. The worst seam in the
tree by an order of magnitude.**

| # | File | Change |
|---|---|---|
| 1 | `selfevo/routing/base.py:100-112` | `register_mode(...)` on `TrainingMode` |
| 2 | `selfevo/integration/group_apply.py:46` | add to `_APPLIED` — necessary, nowhere near sufficient |
| 3 | same, `:239-270` | `apply_decisions` is an if/elif over RL/SFT with one `sft_weight` scalar |
| 4 | same, `:461-520` | `apply_mixtures` hardcodes **exactly two** blend terms (`:470-471`) and builds two extremes by recursive `apply_decisions` calls (`:437`, `:496`) |
| 5 | `areal/api/cli_args.py:1947-2015` | a magnitude field beside `solved_advantage` / `unsolved_advantage`, plus sign validation |
| 6 | `areal/trainer/ppo/actor.py` | `_route_groups` passes `sft_weight=` and the `sft_rows` veto; `exclude_truncated_from_sft` is SFT-specific |
| 7 | 8 router files — `contextual.py:96`, `feedback.py:131`, `cluster.py:118-130`, `routers.py:100/162/215`, `rule_policy.py:250-251`, `harness.py:100`, `code_policy.py:181`, `credit_sim.py:112-113` | each carries its own hardcoded `(RL, SFT, SKIP)` tuple or `teacher_mode` default |
| 8 | `tests/test_group_apply.py:45`, `test_group_mixture.py:58`, `test_contextual_router.py:56` | three separate `MODES` literals |

**The codebase already contains the proof that this is broken.** `TrainingMode.DISTILL` is
registered at `routing/base.py:111` and was never added to `_APPLIED`, so it is a fully
registered mode that every router may legally emit and that `group_apply.py:215` rejects at
runtime. `routing/cluster.py:121-126` records the consequence verbatim: *"a unit routed to
DISTILL pays full cost and never learns. An audit measured 400/1000 units in exactly that
state on the motivating batch."*

The mode registry `_MODES` is genuinely open. The *application* seam is closed and duplicated
across ~11 files. **That asymmetry is the single largest extension debt in the tree, and it
sits directly in the way of section 5**, because every data-acquisition source is a mode
needing a target tensor that `_APPLIED` cannot presently carry.

---

## 4. Reuse we are missing

A repo-wide AST near-duplicate sweep (≥0.70 similarity, ≥4 code lines) over all 177 files
excluding `mutate_*.py` returned **13 pairs, all trivial**. That is a meaningful negative
result: **outside the mutation harnesses there is essentially no function-level copy-paste in
the library code.** What duplication exists in `selfevo/` is *sub-function idiom* repetition —
the group walk, the matched-control deck, the metric dict — which is why the fixes below are
new shared modules rather than deletions.

### 4.1 Mutation harnesses — by far the largest duplication

34 harnesses (29 in `selfevo/tests/`, 5 in `experiments/bench/`), **5,620 lines**, 755
mutation cases.

| Section | Lines | Share |
|---|---|---|
| Header (module docstring + path/target constants) | 854 | 15% |
| `MUTATIONS` / `MUTANTS` list — the real content | 2,767 | **49%** |
| Driver (`run_tests` → EOF) | **1,999** | **36%** |

Exact-body duplicate detection (comments, docstrings and string literals normalised) finds
**651 removable lines repo-wide, 570 of them (88%) inside `mutate_*.py`**. `main()` is
**byte-identical** across 7 files (`mutate_bench_per_task_config.py`,
`mutate_contextual_cold_start.py`, `mutate_credit_assignment.py`,
`mutate_olympiadbench_wired.py`, `mutate_prompt_credit.py`, `mutate_prompt_credit_wired.py`,
`mutate_silence_identity.py`); `_assert_isolated` is identical across 4; `run_tests` is
identical in 4 clusters covering 12 files.

The two largest drivers, `mutate_harness_dispatch.py:331-467` and
`mutate_harness_selectors.py:309-446`, are 137 lines that differ by **15**, of which only
**three lines are real parameterisation** (the import-probe string and its `want` list). `_sha`,
`_assert_matches_live` and the whole 70-line `main` are character-for-character identical.

**Extraction is quantified:** the 2,767 mutation-list lines are irreducible content; the 1,999
driver lines collapse to one ~140-line driver plus ~34 eight-line `main()` stubs ≈ 410 lines,
and ~250 lines of repeated path constants become a small config object. **~1,590-1,700 lines
removable, ~30% of the 5,620.**

> **Shared home — `selfevo/tests/mutation_driver.py`**, exporting `Mutation(label, target,
> find, replace)` *keyword-only* · `MutationHarness(repo, live=None, tests=[…], imports={…})` ·
> `run_mutations(harness, mutations) -> Report` · `run_tests(repo, tests, timeout) ->
> (bool, failing_test_id)` · `sha256_of` · `assert_isolated` · `assert_matches_live` ·
> `kill_child_on_signal` · `EXPECTED_SURVIVORS`.
> **Zero live-path risk, and 88% of all exact duplication in the repo. Do this first.**

Two hazards the extraction should close, both real today:

- **Six harnesses still mutate the live checkout.** `selfevo/tests/mutate_packed.py:4` writes
  to `~/areal-selfevo/selfevo/integration/token_routing.py` directly, with executable code at
  module scope. `mutate_harness_selectors.py:6-10` states why that is now forbidden: *"The
  training run imports this tree with `PYTHONPATH=/home/ubuntu/areal-selfevo` across worker
  processes that relaunch, so a mutated `selectors.py` sitting on disk for even a few seconds
  could be imported by a live run."* With an 8×A100 job on this box those six are unrunnable
  as written. The only thing making them safe to have on disk is that `pyproject.toml` leaves
  `python_files` at the default, so `mutate_*.py` is never collected by pytest.
- **The mutation-row tuple order is inconsistent between two live harnesses.** Seven distinct
  unpack shapes exist across the 34; `mutate_harness_selectors.py` uses `(label, rel, …)` and
  `mutate_group_mixture.py` uses `(rel, label, …)` — same arity, swapped fields. Copy a row
  between them and it silently looks for a source file named after the defect description. A
  keyword-only `Mutation` dataclass removes the trap by construction.

### 4.2 Bootstrap and statistics helpers

The duplication is **entirely in `experiments/bench/`; `selfevo/` library code has none.**

| Helper | Copies | Sites | Similarity |
|---|---|---|---|
| `norm_cdf` | 4 | `compare_runs.py:25`, `merge500.py:44`, `regrade.py:33`, `threearm.py:49` | **identical** one-liners |
| `mcnemar` | 5 | `compare_runs.py:28`, `merge500.py:47`, `paired_full.py:35`, `regrade.py:35`, `threearm.py:52` | same algorithm, **four different return arities** |
| `wilson` | 3 | `math_bench.py:480`, `analyze_sweep.py:26`, `regrade.py:26` | last two identical; `math_bench`'s is hardened |
| per-problem loader | 5 | `threearm.py:34`, `merge500.py:25`, `compare_runs.py:41`, `paired_full.py:20`, `regrade.py:54` | all read `<suite>/<tag>/generations.jsonl` → `{idx: grade(...)}` |

**Two of these are semantic divergences, not style:**

- `wilson(k, 0)` returns `(0.0, 0.0)` in `analyze_sweep.py` and `regrade.py` but `(nan, nan)`
  in `math_bench.py:485`. An empty benchmark prints a *confident* interval `[0.000, 0.000]` in
  two of the three scripts.
- `paired_full.py:35`'s `mcnemar` has **no normal-approximation branch** — exact binomial at
  all `n` — while the other four switch to a continuity-corrected normal at `n ≥ 25`. Two
  scripts can report different p-values for the same table.

~138 lines across 17 copies → ~45.

> **Shared home — `selfevo/stats.py`**: `wilson(k, n, z=1.96)` (adopt `math_bench`'s NaN and
> clamp semantics) · `norm_cdf(x)` · `mcnemar(a, b) -> McNemarResult(n01, n10, p, n_shared)`
> (one NamedTuple kills the arity drift) · `bootstrap_percentile_ci(values, n_boot, seed,
> alpha=0.05)` (the one real implementation is `cluster_lora/interference_analyze.py:139`) ·
> `paired_diff_se(a, b)`. Plus **`experiments/bench/suite_io.py`** exporting
> `per_problem(suite, tag, keep=None)` for the five loaders.
> **No live-path risk.**

### 4.3 Group and batch reshaping

No copy-pasted functions — a sub-function idiom repeated at 9 source sites and 4 test
re-derivations.

**The walk** (`sl = slice(start, start + g); start += g`), five mechanically identical sites,
**all live**: `integration/group_apply.py:242-245` (`apply_decisions`) and `:460-463`
(`apply_mixtures`, a second copy in the same file), `observability.py:167-170`,
`cluster_lora/wiring.py:434-437`, `gold/substitute.py:467-470`.

**The validator** (`sum(sizes) != n_rows` plus `any(g < 1)`), four sites, with divergences:

| Site | Checks | Note |
|---|---|---|
| `group_apply.py:193-200` | `any(g<1)` then `sum != b` | fullest; a comment explains the ordering |
| `observability.py:141-143` | `sum != b` then `any(g<1)` | reversed order |
| `cluster_lora/wiring.py:382-386` | `sum(sizes) != n_rows` **only** | **no `g < 1` check.** A negative size passes the sum check and the `per_row[start:start+size]` walk at `:436` then silently pools one group's rows into another — the exact failure `group_apply.py:194-197` documents. |
| `gold/substitute.py:281-316` `_normalise_group_sizes` | handles `int | None | Sequence`, raises `GoldShapeError`; `_safe_sizes:562-573` is the non-raising twin | superset of the others |

`areal/utils/data.py` has the padded/packed machinery (`concat_padded_tensors:242`,
`split_and_unpad_tensor:314`, `unpack_sequence:437`) but nothing for per-prompt group slicing,
and `areal/utils/group_stats.py` is only the `GroupStats`/`GroupStatsRecorder` pair — so there
is no upstream home and a `selfevo/` module is right.

> **Shared home — `selfevo/grouping.py`**: `normalise_group_sizes(group_sizes, n_rows)` (the
> `substitute.py` superset) · `validate_group_sizes(group_sizes, n_rows, *, what)` ·
> `group_slices(group_sizes) -> Iterator[slice]` · `iter_groups(seq, group_sizes)`.
> **DEFERRED — all five source sites are in the live closure.** Land the module now against
> the four test re-derivations (`test_group_apply.py:528-530`, `test_observability.py:262-265`,
> `test_group_mixture.py:636-639`, `test_group_plumbing.py:23`), migrate the call sites at the
> next run boundary. **`wiring.py:384`'s missing `g < 1` check is a correctness bug worth
> raising independently of the refactor.**

### 4.4 Metric-emission shapes

**There is no single emitter.** Every module builds its own dict, and the actor also builds
`route/*` dicts inline, bypassing all of them.

| Namespace | Refs | Emitters |
|---|---|---|
| `route/` | 118 | `integration/group_apply.py:93-115` (`RouteStats.as_metrics`) · `harness/dispatch.py:222-248` · `harness/selectors.py:390-414` · **plus 5 inline in `areal/trainer/ppo/actor.py`**: `:665` `route/mixed_groups`, `:673` and `:1017` `route/sft_excluded_rows` (**same key, two separate literals**), `:963` `route/truncated_row_fraction`, `:1056-1057` `route/adv_mean_before` / `_after` |
| `cluster_lora/` | 35 | `cluster_lora/adapters.py:110-121` · `cluster_lora/reach.py:89-108` |
| `prompt_credit/` | 21 | `routing/prompt_credit.py:253-260` |
| `gold/` | 13 | `gold/substitute.py:255-278` |
| `feedback/` | 5 | `routing/outcomes.py:57-64` |

Six sites construct keys by f-string with nothing pinning the resulting key set:
`group_apply.py:112` `f"route/{m}_groups"`, `dispatch.py:247`
`f"route/harness_active_{name}"`, `adapters.py:115/117/119`
`f"cluster_lora/{loss,grad_norm,rows}/{name}"`, `reach.py:107` `f"cluster_lora/size/{name}"`.

**Only one namespace declares its keys** — `SELECTOR_METRIC_KEYS` at
`harness/selectors.py:89-96`, asserted by `test_harness_selectors.py:813`. The codebase
already records the failure this prevents: `actor.py:953-956` notes a routed run (`sa2`) that
shipped with an **entirely empty `route/` namespace**, so its routing status had to be
recovered from the config.

> **Shared home — `selfevo/metrics.py`**: `ROUTE`, `CLUSTER_LORA`, `GOLD`, `PROMPT_CREDIT`,
> `FEEDBACK`, `SUPPLY` namespace objects · `declare(prefix, *names) -> KeySet` ·
> `key(ns, name, **fmt)` (the single f-string site) · `emit(tracker, mapping)`.
> **DEFERRED for the migration** — seven of the emitters are live.
> **Do now instead, additive and zero-risk:** a test that imports every `as_metrics` producer,
> unions the keys they can emit, and asserts equality against a declared registry, modelled on
> `test_harness_selectors.py:813`. Note this extraction saves few lines; its value is
> correctness, not LOC.

### 4.5 Rate-matched control construction

**There are four, not two** — and the two the brief named are the two that are *not* live.

| # | Implementation | Site | Lines | Live? | Mechanism |
|---|---|---|---|---|---|
| 1 | `RandomRouter` | `routing/routers.py:144-198` | 55 | **yes** | samples modes from **configured nominal** proportions; private `random.Random`; SKIP fallback with no target |
| 2 | `MatchedPermutationControl` | `routing/proportions.py:58-149` | 92 | no | replays the criterion router's **realised** decisions, shuffled, drawn **with replacement** |
| 3 | `RateMatchedControlSelector` | `harness/selectors.py:638-820` | 183 | no | deck of `moves`×MOVE + `(decisions−moves)`×STAY, shuffled, popped **without replacement**, reshuffled on exhaustion |
| 4 | `random_matched_partition` | `cluster_lora/partition.py:197-247` | 51 | **yes** | `np.random.default_rng(seed).permutation(labels)`; exact size match plus a post-hoc assertion at `:241-246` |

All four share ~15-20 lines of idiom (≈70 lines total): a private RNG seeded at construction
and never global; a refuse-empty guard; feature-blindness *by construction rather than by
configuration*; and an explicit exact-versus-expectation matching argument. Their docstrings
already cross-reference each other — `selectors.py:709` and `partition.py:206-209` both cite
`MatchedPermutationControl` by name.

They differ on: unit sampled (decision / boolean / cluster label); replacement (with /
without / none); match guarantee (in expectation / **exact at every multiple of `decisions`** /
**exact always**); RNG library; and stream length (unbounded / unbounded-reshuffling /
one-shot).

**#3 and #4 are the same object.** Both are "draw a size-matched permutation of a fixed
multiset"; #4 is exactly #3's deck drawn once and never reshuffled. #2 is a genuinely
different sampling regime but shares the base. #1 is the *unmatched* nominal control that #2
exists to replace and should keep its own identity.

One thing not to "unify away": #2's with-replacement draw is deliberate and measured,
documented at `proportions.py:126-131` — an in-order replay realised 8.5% skip against the
criterion's 32%, because SKIP costs nothing so a skipping arm runs longer at the same budget.

> **Shared home — `selfevo/controls.py`**: `MatchedDeck(multiset, seed)` with `.draw()` /
> `.reshuffle()` (serves #3's boolean deck and #4's label permutation) · `permute_labels(labels,
> seed)` · `ReplayPool(pool, seed)` for #2 · a shared `require_nonempty(n, what)`.
> **Partial defer.** `harness/selectors.py` and `routing/proportions.py` are not live —
> refactor now (~275 → ~180 lines). `cluster_lora/partition.py` and `routing/routers.py` are
> live — defer, and migrate them onto the base the safe half has already tested.

### 4.6 Test helpers

| Duplicate | Sites | Size |
|---|---|---|
| `_ctx` / `ctx` — "build a `RoutingContext`" | **11 independent spellings**: `test_cluster_router.py:16`, `test_code_policy.py:68`, `test_contextual_cold_start.py:23`, `test_contextual_router.py:66`, `test_credit_assignment.py:30`, `test_critics.py:22`, `test_feature_covariate_audit.py:246`, `test_harness_router.py:43`, `test_random_control.py:25`, `test_cluster_lora_features.py:235`, `test_routing.py` | three of them carry the *identical* docstring *"A routing context, built by keyword so no field order can silently rebind."* |
| `mode_of` | `test_code_policy.py:76`, `test_contextual_router.py:74`, `test_harness_router.py:51` | 4 × 3, identical |
| `recorder` | `test_harness_dispatch_wired.py:127`, `test_silence_identity.py:53` | 4 × 2, identical |
| `_clear_stats_tracker` | `test_cluster_lora_engine.py:56`, `test_loss_weighting_audit.py:54` | 5 × 2, identical |
| `stub_router` | `test_actor_router_seam.py:66`, `test_routing_stabilisers.py:156` | 18/13, sim 0.73 |
| `make_batch` | 4 files | 1 canonical in `test_group_routing.py` plus 3 redefinitions |

> **Shared home — `selfevo/tests/conftest.py`** (there is none today; see 6.3): `ctx`,
> `mode_of`, `recorder`, `clear_stats_tracker`, `stub_router`. ~35 lines, test-only, zero risk.

---

## 5. The data-acquisition axis: the `Supplier` seam

### 5.1 Why this is a first-class axis

`GOAL.md:66` retracted the M7 teacher demotion on 2026-09-02. The ~4.5% reach figure was
measured on **GSM8K** and generalised as a constant. It is not one: the free self-target
requires at least one correct rollout, so a teacher is needed exactly where there is none, and
reach is therefore a function of difficulty. Same repo, same measurement, harder data — the
unsolved branch is **25.5% of all groups on MATH (60.9% of the routed channel) against 4.5% on
GSM8K, a factor of 5.7**. On genuinely out-of-distribution or held-out data the model cannot
solve, **self-target reach is zero by construction and a supplier is the only source that
exists**.

Worth recording plainly: the live `lora30b` run is on `openai/gsm8k`, which is the very regime
the retracted number came from.

The design consequence is stated once and enforced by the interface below: **a reach number
without its difficulty regime is not comparable, and ordering suppliers at one difficulty does
not order them at another.**

### 5.2 What already exists — and two mechanically different kinds of supply

Supply splits into two kinds that share a router but not a mechanism.

**(A) Target-distribution suppliers (soft labels).** Supply `teacher_logp`, a `(B, T)` tensor
aligned with the model's *own* rollout tokens. **This is already wired upstream and we did not
build it**: `TeacherConfig` at `areal/api/cli_args.py:3724` (`engine_type: rollout | train`,
`rollout: InferenceEngineConfig`, `train: PPOActorConfig`, `path`, `offload`, `rl_loss_weight`,
`distill_loss_weight`), `self.teacher` built at `rl_trainer.py:210-223`, scored at `:758-774`,
consumed by `grpo_loss_fn`'s reverse-KL penalty. Our single `actor.py` deletion (block B11) is
exactly the change that made that penalty **per-token routable** rather than a batch scalar.
`rl_trainer.py` is otherwise **unmodified by us**. No new rows; cost is one extra forward.

**(B) Token suppliers (hard targets, new rows).** Supply `(ids, mask)` for a row substituted
into the batch. Gold is the only one built — `selfevo/gold/`, 1,111 lines, and it is already
generic in everything except where the ids come from: `substitute_gold_rows` selects
qualifying groups by rule, `_write_gold_row` splices after the victim row's prompt,
`GoldStats` counts reach and token mass, `reconcile_gold_logprobs` fixes the log-probabilities
afterwards. Cost is generation.

Note that **the gold path is not yet wired**: `grep gold areal/trainer/ppo/actor.py` finds
nothing. Only `attach_gold_from_data` is called from `areal/` (two sites). `GoldRule` has zero
production callers — a fully built, heavily tested seam reachable from no run. Wiring it is
the prerequisite that also proves out the Supplier seam.

### 5.3 The ordering constraint — the single hardest fact

The trainer's pipeline (`areal/trainer/rl_trainer.py`, unmodified by us):

```
710  actor.prepare_batch(...)                → rollout_batch
734  critic.compute_values(...)              (optional)
751  ref.compute_logp(...)                   → traj["ref_logp"]        (optional)
769  teacher.compute_logp(...)               → traj["teacher_logp"]    (optional)
804  actor.compute_logp(...)                 → traj["prox_logp"]
817  actor.compute_advantages(...)           → _route_groups at actor.py:980
837  actor.ppo_update(...)
```

**The router runs at stage 817.** A token supplier's row must be in the batch before **804** —
`substitute_gold_rows` raises `GoldOrderingError` if `prox_logp`, `ref_logp` or `teacher_logp`
are already present (`_STALE_AFTER_SUBSTITUTION`, `substitute.py:~105`), and it is right to.
A soft-label supplier must be invoked before **769**.

**So the router currently runs two stages too late to gate a supplier.** That is the central
architectural fact for this axis, and it is not a small edit.

The fix is available and cheap, and this is the most useful single thing in this document:
`group_features(rewards, loss_mask, logprobs, group_sizes, …)` (`observability.py:93`) reads
only what a rollout batch already carries at stage 710 — its `logprobs` argument is the
*inference* log-probability ("the sampler already returned it", `:66`), **not** `prox_logp`.
`RoutingContext` needs `solve_rate`, `group_size`, `has_teacher`, `can_evolve_harness`,
`unit_id` and `extra`, and every one of those is constructible immediately after
`prepare_batch`.

**Therefore routing can be lifted from stage 817 to stage 710 with no new information.** Do
that once, and both supplier kinds have a place to stand. Until it is done, `has_teacher` can
only ever be a constant, which is why `actor.py:508` writes the literal `False`.

### 5.4 The protocol

Two dataclasses of declared metadata, one of cost, one of payload, and a three-method
Protocol. Nothing here needs torch except the payload, so suppliers stay CPU-testable like
routers.

```python
# selfevo/supply/base.py

class SupplyKind(Enum):
    TOKENS = "tokens"          # a row: (ids, mask) substituted into the batch
    DISTRIBUTION = "dist"      # teacher_logp aligned to the model's own tokens

class Scorability(Enum):
    NONE = "none"              # cannot score its own items  <- the honest default
    PROXY = "proxy"            # a cheap proxy exists (the run's own grader, unit tests)
    VERIFIED = "verified"      # ground truth (dataset gold answer match)

class Contamination(Enum):
    CLEAN = "clean"            # provably disjoint from every reported benchmark
    SELF = "self"              # the model's own output; no external contamination
    UNKNOWN = "unknown"        # an API model of undisclosed training data

@dataclass(frozen=True)
class SupplyCost:
    calls: float               # requests per SERVED group
    prompt_tokens: float
    completion_tokens: float
    wall_seconds: float
    usd: float = 0.0

@dataclass(frozen=True)
class SupplyCapability:
    kind: SupplyKind
    provides_modes: frozenset[str]   # registered TrainingMode names it can serve
    model_id: str                    # WHAT SERVED. Recorded in the results artifact.
    scorability: Scorability
    contamination: Contamination
    contaminates: frozenset[str]     # named benchmarks it may have seen, e.g. {"livecodebench_v6"}
    needs_network: bool
    is_blocking: bool                # True = must be prefetched, see 5.6
    reach: float                     # fraction of qualifying groups answered...
    reach_regime: str                # ...IN THIS REGIME. Mandatory, free text, e.g.
                                     # "MATH-lighteval train / Qwen3-30B / n=8 / 2026-09-02"
    cost_per_group: SupplyCost

@dataclass(frozen=True)
class SupplyRequest:
    group_index: int
    prompt_ids: torch.Tensor
    rewards: torch.Tensor            # the group's raw rewards, so a supplier may decline
    unit_id: str

@dataclass(frozen=True)
class SupplyItem:
    group_index: int
    ids: torch.Tensor | None         # (L,) for TOKENS
    mask: torch.Tensor | None
    logp: torch.Tensor | None        # (n, T) for DISTRIBUTION
    provenance: str                  # f"{supplier.name}:{capability.model_id}"

@dataclass(frozen=True)
class SupplyResult:
    items: tuple[SupplyItem, ...]
    unserved: Mapping[int, str]      # group_index -> why. NEVER silently empty.
    cost: SupplyCost

class Supplier(Protocol):
    name: str
    def capability(self) -> SupplyCapability: ...
    def probe(self) -> None: ...     # liveness; raises SupplierUnavailable
    def supply(self, requests: Sequence[SupplyRequest]) -> SupplyResult: ...
```

Three deliberate choices:

- **`Scorability.NONE` is the honest default** (see 5.8). A router may gate only on what a
  supplier declares, and it may not assume per-item fitness exists.
- **`reach` and `reach_regime` are one unit.** A capability carrying a reach without a regime
  is refused at registration, so the comparability rule from 5.1 is structural rather than a
  docstring convention.
- **`unserved` is a mapping with reasons, not a count.** The repo's standing rule is that
  every zero must have an artifact behind it.

### 5.5 Where it plugs in

**Phase A, immediately after `prepare_batch` (stage 710, before 751).**

1. Build `GroupFeatures` and `RoutingContext` per group (already possible — 5.3).
2. `has_teacher` is written from the broker, not from a literal: `broker.has_source_for(ctx)`
   replaces `actor.py:508`'s `has_teacher=False`.
3. `decisions = route_all(router, contexts)` — the existing function, moved earlier.
4. For each group whose decision names a teacher-requiring mode, issue a `SupplyRequest`.
5. `TOKENS` items go through the existing gold machinery, generalised: `substitute_gold_rows`
   becomes `substitute_supplied_rows(batch, items, …)` and `GoldRule` becomes the *qualifying
   predicate* it already is, orthogonal to the source. `_write_gold_row` is unchanged.
6. `DISTRIBUTION` items are written to `traj["teacher_logp"]` **for the routed subset only** —
   which is what makes the upstream teacher routable and budgeted, since `rl_trainer.py:769`
   scores the whole batch unconditionally today.

**Phase B, after `compute_logp` (stage 804, before 817).** `reconcile_gold_logprobs` becomes
`reconcile_supplied_logprobs`, unchanged in mechanism: replace `GOLD_LOGP_SENTINEL` (+1.0,
finite and impossible as a log-probability) with the trainer's recomputed `prox_logp` rolled
right by one, so `behave_imp_weight = exp(0) = 1`. `assert_supplied_logprobs_filled` refuses
to let an unfilled row reach the loss. **This half needs no design work — it already exists and
is correct;** it only needs its gold-specific names widened.

### 5.6 What each supplier costs, at the live batch shape

Live shape: 64 prompt groups per step, `n_samples=8` (512 rows), `max_new_tokens=1024`,
`max_tokens=2048`, 1,160 steps.

| Supplier | Kind | Scorable | Contamination | Network | Cost per served group | Notes |
|---|---|---|---|---|---|---|
| `gold_dataset` | TOKENS | VERIFIED | CLEAN | no | **0 GPU.** 7,500 MATH solutions tokenise in 3.29 s once at dataset adaptation and are cached by `datasets` fingerprinting; median 163 tokens/row | the reference implementation; already built |
| `teacher_logp_local` | DISTRIBUTION | NONE | CLEAN | no | one forward over the **routed rows only**: at 25.5% reach, 16 groups × 8 rows × ≤2048 tok ≈ **262k prefill tokens/step**, against ~1.05M if unrouted | upstream engine exists; needs the teacher resident or `teacher.offload: true` |
| `teacher_gen_local` | TOKENS | PROXY (the run's own grader) | CLEAN | no | one generation per served group: 16 × ≤1024 decode tokens/step at teacher size, plus a grader call | dominant cost; a second served checkpoint on the box |
| `teacher_api` | either | PROXY | **UNKNOWN** | **yes** | 1 request/group; 2-30 s latency; USD must be recorded | `is_blocking=True` — see below |
| `self_past_ckpt` | TOKENS | PROXY | SELF | no | as `teacher_gen_local`, plus a second served checkpoint's memory | reach is bounded above by the current model's — an earlier checkpoint solves a subset |
| `scraped_corpus` | TOKENS | NONE | **UNKNOWN** | yes (offline) | amortised; per-group cost is a retrieval | contamination, not cost, is the binding constraint |

**Latency and `is_blocking`.** An API supplier at 2-30 s per call, inside a step whose other
stages are GPU-bound, would stall the trainer. AReaL is asynchronous by design, so a blocking
supplier must **prefetch against the previous step's prompt ids and serve from cache**,
accepting one-step-stale supply. A supplier declares `is_blocking`; a blocking supplier whose
declared `wall_seconds` exceeds a configured budget is **refused at config time**, not
discovered at step 3.

**Model-type choice is explicit and swappable.** `capability().model_id` names what actually
served, and it is carried into `SupplyItem.provenance` and emitted as a run metric. This is
the LiveCodeBench lesson applied: *a score whose supplier is not recorded is not attributable.*
The three model types the PI asked for map onto `teacher_logp_local` /
`teacher_gen_local` (a served local checkpoint — another adapter or a larger base on the same
box), `teacher_api` (a remote API model), and `self_past_ckpt` (the model's own earlier
checkpoint, i.e. self-distillation across time). Each is one registry entry plus config, not a
code edit.

### 5.7 Availability, and what happens when a supplier fails

**`has_teacher` must stop being a constant.** Availability for a group is the conjunction of
three things, each checked at a different time:

1. **Declared** — some registered supplier's `capability().provides_modes` contains the mode.
   Checked at **config parse**, before any GPU, in `GroupRoutingConfig.__post_init__`, the way
   `harness_variants` already is (`cli_args.py:1991`).
2. **Live** — `probe()` succeeded at trainer initialisation. A supplier that cannot answer is
   removed from the broker *and the run refuses to start* if an arm named it.
3. **In budget** — the step's accumulated `SupplyCost` is under the configured cap.

A router then gates on `ctx.has_teacher`, which the broker writes per group. It cannot select
a teacher mode no supplier can serve, because the value is derived from the suppliers rather
than asserted beside them.

**Mid-batch failure is refused, not absorbed.** A silent fallback to SKIP is precisely the
silent-no-op this repo distrusts most, and the tree already contains the pattern to copy:
`_refuse_dropped_harness` (`actor.py:171`) refuses to let a harness action vanish when nothing
consumes it, with the reasoning *"an arm labelled 'harness-evolving' would train identically to
one that is not — two runs whose only difference is a name."* The supplier axis gets
`_refuse_dropped_source` with the same shape, plus a policy field:

- `on_unserved = "refuse"` (**default**) — raise, mirroring `GoldMissingError`, which already
  refuses when a rule is on and no gold reached the update.
- `on_unserved = "demote_and_count"` — permitted only because the count is emitted; every
  demoted group appears in `supply/unserved_*` with its reason string.

There is no third option, and in particular no option that turns an unserved group into a SKIP
without a counter.

### 5.8 Prior art: OpenRSI — cited, not vendored

**Licence first. `~/openrsi` is CC BY-NC 4.0 (NonCommercial), © 2026 Frontis AI and the
OpenMLE contributors (`~/openrsi/LICENSE`). Its source must not be copied into this repo. We
may cite it and reimplement an idea described in words. Anyone reaching for `~/openrsi` while
implementing this section should stop and read this paragraph.**

**What transfers: parent selection.** `~/openrsi/OpenMLE-ERL/RL/program_database.py:54-57`
keeps a population of candidates where each carries an *exploit* coefficient derived from its
own reward, an *explore* coefficient derived from the **variance of its direct children's
rewards**, a *cooling* coefficient derived from its visit count, and the visit count itself
(persisted; schema at `:195-198`, recomputed at `:572-606`). Fitness is the unweighted sum of
the three after min-max normalisation, each defaulting to a neutral 0.5 where undefined, and
cooling is `1 − visits/max_visits`. That is a UCB-shaped answer to "which candidate do I extend
next", with novelty measured as child-reward variance rather than as a distance in an
embedding space. **If a supplier ever maintains a population of generated candidates, that
triple is a concrete, working answer to the selection question and should be named as prior
art.**

**What does not transfer, and why the protocol must say so.** OpenRSI's loop is generate →
score in a sandbox → keep population → select parent → improve. It runs because an MLE program
has a directly measurable task score. **A training example has no such score**: its value is
"does training on this improve held-out performance", which cannot be measured per example
without a training run per example. The search structure therefore has no signal in our
setting. This is exactly why `Scorability.NONE` is the honest default and why a router may
only gate on what a supplier actually declares — a protocol that assumed per-item fitness
would import a signal that does not exist here.

### 5.9 Cost accounting, and the AI2 bar

`GOAL.md:1008-1009` marks **Matched inference budget: NOT MET** and **Matched feedback budget:
NOT MET — query counts not logged**. A teacher arm that is not budget-matched will be rejected
on exactly that row. `SupplyCost` is accumulated per step and emitted under a `supply/`
namespace, following the `GoldStats.as_metrics()` shape (`substitute.py:255-278`) including its
rule that an **off** arm emits the same keys as zeros rather than omitting them, so two arms
stay readable on one panel:

```
supply/calls  supply/prompt_tokens  supply/completion_tokens  supply/wall_seconds  supply/usd
supply/groups_served  supply/groups_unserved  supply/reach
supply/token_mass          <- the one that decides matching
```

**Match on token mass, not on group count.** `GoldStats.token_mass` already establishes why:
the objective is a single per-token mean over the global batch, so a row's share of the update
is proportional to its **token** count — the audit measured 0.5 / 1.0 / 2.0 relative gradient
magnitude for SFT rows of 4 / 8 / 16 tokens against 4-token RL rows. Two supplier arms matched
on served-group count can differ several-fold in what the loss actually reads.

**Contamination is a first-class declaration, not a note.** `contamination` plus
`contaminates` let the experiment plan mechanically exclude a supplier from any run reporting
a benchmark it may have seen. This matters most for `teacher_api` and `scraped_corpus`:
LiveCodeBench v6's entire value is a contest window post-dating training cutoffs, so a
supplier whose model may have seen it voids that benchmark for the arm. The check belongs in
config validation beside the availability check, before any GPU.

### 5.10 Files you must touch to add a supplier

**Target state — 2 files:**

| # | File | Change |
|---|---|---|
| 1 | `selfevo/supply/<new>.py` | NEW: `capability()`, `probe()`, `supply()` |
| 2 | `selfevo/supply/__init__.py` | a lazy factory wrapper plus one line in the `SUPPLIERS` literal — deliberately the same shape as `ROUTERS`, wrapper and all, for the reason given in 3.3(a) |

**Today, before the prerequisites land, it would be ~11 files** — the same list as 3.3(c),
because a supplier's mode is a loss mode. Three things must land first, and they are listed as
S1-S3 in the plan:

- **S1** — close the `_APPLIED` seam (3.3c), or a supplier's mode is rejected at
  `group_apply.py:215` exactly as `DISTILL` is today.
- **S2** — lift routing from stage 817 to stage 710 (5.3).
- **S3** — replace `actor.py:508`'s literal `has_teacher=False` with `broker.has_source_for(ctx)`.

`SUPPLIERS` should be validated in `GroupRoutingConfig.__post_init__` against the registry,
following `harness_variants` — the one axis in this tree that is already checked before a GPU
is booked.

---

## 6. Testing architecture

### 6.1 Inventory

`.pytest_cache/v/cache/nodeids` (a real prior collection, written 05:08 today) holds **1,724
collected nodes under `selfevo/tests/`** — the ~1,700 figure is confirmed — plus 209 under
`experiments/bench/`, 33 under `experiments/harness/` and 341 under upstream `tests/`. The
`selfevo` figure is 1,127 `def test_` across 52 files; the gap to 1,724 is 117 `parametrize`
decorators, several stacked.

| Subpackage | src LOC | test files | test funcs | test LOC | test:src | mutation harnesses |
|---|---|---|---|---|---|---|
| `routing` | 3,783 | 19 | 427 | 6,521 | **1.72** | 13 |
| `cluster_lora` | 4,640 | 10 | 238 | 4,399 | **0.95** | 2 |
| `integration` | 903 | 12 | 156 | 3,462 | **3.83** | 6 |
| `harness` | 1,692 | 5 | 186 | 2,570 | 1.52 | 2 |
| `gold` | 1,111 | 2 | 55 | 1,446 | 1.30 | 2 |
| `clustering` | 236 | 0 dedicated | — | — | — | 0 |
| `baselines` | 114 | 1 | 21 | 518 | 4.54 | 1 |
| **total** | **13,704** | **52** | **1,127** | **19,669** | **1.44** | **29** |

Thin spots, in order of exposure: **`cluster_lora/interference_dump.py` — 1,408 lines, the
single largest module, reached by one test file and zero mutation harnesses**;
`routing/policy_vetting.py` — 167 lines, **zero references from any test**;
`routing/outcomes.py::AttributionStrength` — zero test references; `clustering/meds.py` — 235
lines, no dedicated file. `cluster_lora` has the lowest test:src ratio of any substantial
subpackage while being the largest and the current method.

### 6.2 Style

The dominant style is **adversarial behavioural testing driven through the real production
call**, not classic unit testing. Three signatures recur and all three are good:

1. **Drive the real seam, never re-derive.** `test_group_routing.py:1-12` states it: *"These
   tests construct a `PPOActor` with `engine=None` and call the actual method rather than
   re-deriving its arithmetic in the test file… a sibling test file isolates the `group_ids`
   expression as a local `derive()` helper, which pins a copy of the code and cannot notice
   the copy drifting."*
2. **Premise tests.** Most files open with a test asserting the setup is non-vacuous —
   *"The premise. If this fails, every test below is testing nothing."*
3. **Mandatory controls.** Every targeted-mechanism claim is paired with a rate-matched or
   shuffled control, because three findings in this repo turned on a targeted rule being
   indistinguishable from a random one at the same rate.

Counts over 52 files: `pytest.raises` 41 · `pytest.approx` 21 · `torch.manual_seed` 15 ·
`monkeypatch` 12 · `ast.parse` 4 (asserting on the AST of the shipped module) · `MagicMock` 1 ·
**`hypothesis` 0** · `caplog`/`capsys` 0. 1,935 assertions. No golden files, no property-based
framework; sweeps are hand-rolled `itertools.product` grids.

Hermeticity is good: **zero `torch.cuda` references in any selfevo test**, heavy imports gated
by module-level `pytest.importorskip` (`torch` 10 files, `peft` 4, `sklearn` 3, `datasets` 3).
Three exceptions download: `test_gold_batch_path.py:169-186` and
`test_gold_target_reachability.py:277-283` (a tokenizer and `MATH-lighteval`), and
`test_cluster_lora_probe.py:490` (the suite's only `skipif`).

**Runtime is unbounded and unmarked.** `pyproject.toml:296-304` declares seven markers
(`slow`, `ci`, `gpu`, `multi_gpu`, `sglang`, `vllm`, `integration`) and **selfevo uses exactly
zero of them**. A CI filter of `-m "not slow"` — which is what upstream uses — would exclude
nothing, and the dataset-downloading tests would run in full.

### 6.3 Fixtures

**There is no `conftest.py` anywhere in `selfevo/` or `experiments/`.** The six in the repo are
upstream's and none is on selfevo's collection path. `@pytest.fixture` appears 24 times across
16 files, all function-local.

**The de-facto conftest is a test module.** `selfevo/tests/test_group_routing.py` (260 lines,
13 tests) is imported by **nine** other test files for `make_actor`, `make_batch`, `meta`,
`advantages`, `B`, `T`, `G`, `PROMPT`, `MIXED`. This is deliberate and documented
(`test_cluster_lora_wired.py:10-12`: *"two definitions of 'an actor configured like the live
runs' drift, and the drift is silent"*) and the instinct is right — but importing a fixture
drags in thirteen unrelated tests' module-level constants and couples file ordering under
`pytest -x`. **A real `conftest.py` is the same idea with none of the coupling**, and 4.6 lists
its first occupants: eleven independent spellings of "build a `RoutingContext`", three of them
sharing an identical docstring.

Fake/stub *classes* are genuinely not a problem — 28 class definitions total across 52 files,
one name (`Recorder`) shared by three files, everything else unique. The brief's expected
`FakeConfig`/`StubRouter`/`DummyEngine` sprawl does not exist.

### 6.4 Mutation harnesses

Covered quantitatively in 4.1. Architecturally there are **three generations**:

| Tier | Example | Capabilities |
|---|---|---|
| **v1** — mutates the **live checkout** | `mutate_packed.py` (39 lines), `mutate_critics.py`, `experiments/bench/mutate_{boxed,selfconsistency,split}.py` | `ROOT = Path.home()/"areal-selfevo"`; executable code at module scope; restore in `finally` |
| **v2** — copies via `sys.argv[1]` | the seven identical-`main` files | sha256 restore assertion, unique-anchor SKIP |
| **v3** — multi-target and verified | `mutate_harness_selectors.py` (446 lines), `mutate_group_mixture.py` (328) | adds `_assert_isolated()` (proves pytest imported the copy), `_assert_matches_live()` (sha256 copy ≡ live before *and* after), a four-way SKIP taxonomy, kill attribution by test id, a SIGINT/SIGTERM handler that reaps the child and prints proof of restore before `os._exit(130)` |

**v3 is already 90% of the shared driver.** The non-negotiable behaviours to carry across, all
implemented somewhere today: refuse on a red baseline; SKIP rather than score when the anchor
is not unique, the replacement leaves bytes unchanged, the mutant fails to `compile()`, or the
text contains a literal `\n` (`mutate_harness_selectors.py`'s docstring records all four as
mistakes actually made here); sha256-assert restore per target and again at the end; record
the killing test id, not just red/green.

### 6.5 Convention for a new contributor

This is inferred from the evidence and **is written down nowhere today** — neither `CLAUDE.md`,
`AGENTS.md`, `CONTRIBUTING.md` nor `.claude/rules/testing.md` (all four are upstream AReaL's,
and the selfevo suite deliberately follows almost none of the last one). Making it explicit is
one of the cheapest wins available.

- **Where.** `selfevo/tests/test_<module>.py`, flat — no subdirectories, no mirroring of the
  source tree. A test for a new router goes in that same flat directory.
- **What to assert.** (1) Call the **real production entry point** — for a router, drive it
  through `PPOActor._compute_advantages` via `make_actor`/`advantages`, never
  `_route_groups` directly, and never re-derive the module's arithmetic in the test.
  (2) Open with a **premise test** proving the fixture is non-vacuous. (3) Assert
  **bit-identical rollback** when the feature is off (`torch.equal`, not `allclose`).
  (4) Assert the feature does not touch data it should not. (5) Ship a **control** for any
  targeting or selection claim. (6) Import shared fixtures rather than copying them.
  (7) Gate heavy imports with module-level `importorskip`; stay on CPU.
- **Naming is enforced by tooling, not policy.** `test_*.py` is collected by pytest;
  `mutate_*.py` is **not** — it is a standalone script run as
  `python selfevo/tests/mutate_X.py <repo-copy> [<live-repo>]`. Getting this backwards would
  be dangerous, because several `mutate_*.py` files write to source at import time.
- **Is a harness expected per module?** Not universally — 25 of 52 test files (48%) have one,
  and the pairing is not arbitrary: **every harness targets a module that can change what a
  live run computes.** By that rule the visible gap is `cluster_lora` — nothing mutates
  `interference_dump.py`, `features.py`, `sketch.py` or `merge.py` in isolation.

### 6.6 CI: none of this runs

- `grep -rn "selfevo\|experiments/" .github/workflows/ .pre-commit-config.yaml` → **zero hits.**
- The only CI `pytest` invocation is `.github/workflows/test-areal.yml:325`, with an explicit
  path list (`tests/test_*.py tests/experimental/ tests/infra/ tests/v2/`) that excludes
  `selfevo/tests/` and `experiments/`. It is `workflow_call`/`workflow_dispatch`-only and gated
  on a self-hosted GPU runner.
- `install-test.yml` is path-filtered to `pyproject.toml`, `uv.lock`, `areal/**` — a
  selfevo-only PR triggers nothing.
- `pre-commit.yml` runs on every PR but only ruff, ruff-format, mdformat and a license header.
- No `Makefile`, no `tox.ini`, no `pytest.ini`, no `setup.cfg`.

**Consequence, and it is currently visible.** The suite is green-by-hand only, and
`.pytest_cache/v/cache/lastfailed` (written 03:34 today) records **five failing tests**:
`experiments/bench/test_math_bench.py::test_truncated_final_box_discards_earlier_box`,
`selfevo/tests/test_compose_feedback.py::test_meta_critic_stub_is_caught_like_any_other`,
`selfevo/tests/test_actor_router_seam.py::test_the_prompt_region_is_never_written`,
`selfevo/tests/test_silence_identity.py::test_a_fully_truncated_group_is_counted_silent_but_neither_solved_nor_unsolved`,
`selfevo/tests/test_routing.py::test_random_router_degrades_to_skip_without_a_teacher`.
That is a *record of the last local run*, not a verification against HEAD — the tree has been
rewritten by another agent since, and no suite was run during this audit. It should be
re-checked rather than quoted. Nothing in CI would have caught it either way.

---

## 7. What a large organisation would have that this lacks

Only the items that would earn their cost for two people on a deadline. Everything below is
either a few hours' work or prevents a class of failure that has already occurred here.

**Worth it:**

1. **Pre-GPU config validation for every axis, not one.** `harness_variants` is resolved
   against `VARIANTS` before a GPU is booked; `router`, `partition` and a future `supplier` are
   not. Two arms in this repo already ran bit-identical to their control because a default was
   wrong (`compose.py:114-137`, `:156-172`). One `__post_init__` block per axis.
2. **A declared metric key set per namespace.** `SELECTOR_METRIC_KEYS` already does this for
   one namespace and a test asserts it. A run has already shipped with an entirely empty
   `route/` namespace (`actor.py:953-956`). Additive, no live-path risk.
3. **A `selfevo/CONTRIBUTING.md` of about one page** — 6.5 plus 3.3, i.e. the seam-extension
   tables and the test convention. The convention exists and is consistent; it is simply
   unwritten, so it is re-derived by every contributor and this document is the third artifact
   to reconstruct it.
4. **A CI job that runs the selfevo suite.** The suite exists, is 1,724 nodes, is CPU-only, and
   nothing runs it. The blocker is that no test carries a `slow` marker, so it cannot be
   filtered. Marking the three downloading tests `slow` and adding one `pytest selfevo/tests -m
   "not slow"` job is perhaps two hours and would have caught the five current failures.
5. **A results artifact that records what served.** `capability().model_id` and the `supply/*`
   budget keys, per 5.6 and 5.9. This is the AI2 bar (`GOAL.md:1008-1009`) and it is a
   precondition for the teacher arm being publishable at all.

**Not worth it, explicitly:** type-checking the whole tree in CI (the Protocols that matter are
already runtime-checked at the seams); coverage gates (test:src is already 1.44 and the thin
spots are known by name); an ADR process (`FINDINGS_*.md` already does this better, with
measurements); a plugin/entry-point discovery system for the registries (eight routers do not
need one — a literal dict a newcomer can read is better); and any abstraction over
`experiments/`, which is a scratch surface and should stay one.

---

## 8. Ranked plan

Ranked by value ÷ disruption. **DEFERRED** means it touches a module in the live import
closure of section 0 and must wait for a run boundary. Recall that the *current* run imports
no `selfevo` module at all — that is slack, not licence, because the next launch will differ.

### Safe now — no module in the live closure

| # | Item | Value | Effort | Removes / adds |
|---|---|---|---|---|
| **A1** | **`selfevo/tests/mutation_driver.py`** and migrate the 34 harnesses (4.1) | **Highest.** 88% of all exact duplication in the repo; closes the swapped-tuple trap and retires the six harnesses that mutate the live checkout — a real hazard while a job runs | ~1 day | −1,590 to −1,700 lines |
| **A2** | **`selfevo/tests/conftest.py`** with `ctx`, `mode_of`, `recorder`, `clear_stats_tracker`, `stub_router` (4.6) | Removes 11 spellings of one helper; decouples nine files from `test_group_routing.py` | ~2 h | −35, +1 file |
| **A3** | **Metric key-set test** — union every `as_metrics` producer's keys, assert against a declared registry (4.4) | Catches namespace drift today without touching a live module; a run has already shipped an empty `route/` namespace | ~2 h | additive |
| **A4** | **`selfevo/stats.py`** + `experiments/bench/suite_io.py` (4.2) | Collapses 17 copies and **fixes two live analysis bugs**: `wilson(k,0)` printing `[0.000, 0.000]`, and `mcnemar` disagreeing between scripts | ~3 h | −93 |
| **A5** | **`selfevo/controls.py`**, safe half only — `harness/selectors.py` + `routing/proportions.py` (4.5) | Gives the live half a tested base to migrate onto later | ~4 h | −95 |
| **A6** | **Write `selfevo/CONTRIBUTING.md`** — 6.5 + the 3.3 extension tables | This document is the third reconstruction of an unwritten convention | ~2 h | +1 file |
| **A7** | **CI job for the selfevo suite**: mark the three downloading tests `slow`, add `pytest selfevo/tests -m "not slow"` (6.6) | 1,724 nodes currently run only by hand; five tests are recorded failing | ~2 h | +1 workflow |
| **A8** | **Fix `compose.py:7`'s stale docstring**; delete `areal/utils/group_stats.py`, `tests/test_group_stats.py` and the `areal/utils/data.py` `recorder` plumbing (2.1) | Retires 1,372 dead vendor lines including a structural edit to `data.py` | ~1 h | −1,372 vendor lines |
| **A9** | **PR the ~70 generic upstream fixes** — `sglang_remote.py`, `openai/client.py`, `proxy_rollout_server.py`, `remote_inf_engine.py`, `rollout_controller.py` (2.4) | Removes 15 hunks of merge liability that are not ours in the first place | ~half a day + review latency | −70 vendor lines |

Two correctness items to raise now, independent of any refactor:

- **`cluster_lora/wiring.py:384` omits the `any(g < 1)` check** its four sibling validators
  have, and the walk at `:436` silently mis-pools on a negative size (4.3). *This module is in
  the live closure — file it, do not fix it inline.*
- **`partition_from_config`'s silent fallthrough** (`partition.py:709`) will mislabel the next
  partition anyone adds (3.2b). Same caveat.

### Prerequisites for the Supplier axis — DEFERRED, in this order

| # | Item | Why it must come first |
|---|---|---|
| **S1** | **Close the `_APPLIED` seam** (3.3c): make `apply_decisions`/`apply_mixtures` mode-generic, then delete or implement `DISTILL` | Largest extension debt in the tree; a supplier's mode is a loss mode, so nothing else can land first. `DISTILL` is registered, routable, and rejected at runtime — 400/1000 units measured paying full cost and learning nothing |
| **S2** | **Lift routing from stage 817 to stage 710** (5.3) | The router runs two stages after the point a supplier must act. `group_features` already needs nothing that is unavailable at 710, so this is a move, not a redesign |
| **S3** | **`has_teacher` from the broker**, replacing `actor.py:508`'s literal `False` (5.7) | Until then the gate cannot be honest and no teacher arm is reachable |
| **S4** | **Wire the gold path** — call `substitute_gold_rows` before `compute_logp` and `reconcile_gold_logprobs` after | 1,111 lines already built and tested with zero production callers. It is the reference supplier and it proves S1-S3 end to end before any teacher spend |
| **S5** | **Generalise gold → `selfevo/supply/`**: `substitute_supplied_rows`, `reconcile_supplied_logprobs`, `SUPPLIERS` registry, `SupplyCapability`, budget metrics (5.4-5.9) | Only after S4 has run once. Ship `gold_dataset` and `teacher_logp_local` first — the latter reuses the upstream teacher engine and adds no new serving infrastructure |

### Deferred, lower priority

| # | Item | Blocked on |
|---|---|---|
| D1 | `selfevo/grouping.py` → the 5 live call sites (4.3) | run boundary |
| D2 | `selfevo/metrics.py` → 7 `as_metrics` + 5 actor inlines (4.4) | run boundary |
| D3 | `selfevo/controls.py` → `partition.py`, `routers.py` (4.5) | run boundary; A5 first |
| D4 | **Move `actor.py` and `fsdp_engine.py` behind `import_from_string`** (2.2-2.3): a `selfevo` engine subclass plus a vendored `routed_grpo_loss_fn` | run boundary. Displaces 984 vendor lines, including all 9 of `actor.py`'s hunks on pre-existing upstream lines — the single largest reduction in merge liability available, and the one that most needs an idle trainer |
| D5 | Move `TokenRoutingConfig`/`GroupRoutingConfig` to `selfevo/config.py` behind one upstream field (2.1) | run boundary; pairs naturally with D4 |
| D6 | Replace `areal/dataset/__init__.py`'s spliced `elif` with a wrapper or an upstream registry PR (2.1) | run boundary. Highest conflict probability per line in the fork |
| D7 | Restore the `SELECTORS` registry and its config field (3.2a) | not strictly live-blocked, but pointless before the selector arm has a consumer; `build_dispatcher` hardwires `round_robin` today |

**Net, before the trainer is touched: ~1,815 `selfevo`/`experiments` lines removable at zero
live-path risk, plus 1,372 dead vendor lines retired and ~70 more PR'd upstream — which takes
the `areal/` delta from 3,051 lines to ~1,609. Once the trainer is idle, a further ~1,023
vendor lines and ~135 `selfevo` lines, leaving a vendor footprint of roughly 590 lines in
6 files.**
