# Contributing to `selfevo/`

The root `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md` and `.claude/rules/testing.md` are
upstream AReaL's. This suite deliberately follows almost none of the last one, and the
rules it follows instead were never written down: `ARCHITECTURE.md` is the *third*
reconstruction of them from the evidence in the tree. This file exists so there is no
fourth.

Everything below is a rule the code already enforces on itself somewhere. Where a rule
was learned by breaking something, the incident is named, because a rule with a scar
attached is the only kind anyone keeps.

**On citations.** Symbols and file names are authoritative; line numbers are a
convenience and were verified against HEAD on 2026-09-02. Up to six agents edit this
tree at once and line numbers rot within a day, so if a number disagrees with the
symbol, trust the symbol.

______________________________________________________________________

## 1. The five rules

**1. Mutation-test every guard.** A test that passes is not evidence; a test that fails
when you break the thing it guards is. Ship a `mutate_*.py` harness alongside any module
that can change what a live run computes. 25 of 52 test files have one, and the pairing
is not arbitrary: **every existing harness targets a module that can change a number.**
See §5.

**2. A SKIP is never a SURVIVED, and never a KILLED.** A mutation whose anchor was not
unique, whose replacement left the bytes unchanged, whose text failed to `compile()`, or
whose replacement contained a literal `\n`, has not been *tested*. Report it in its own
column. All four are mistakes actually made in this repo, recorded in
`mutate_harness_selectors.py`'s docstring. A harness that scores them as kills reports a
number higher than the truth, which is worse than reporting nothing.

**3. Stage by pathspec. The checkout is shared.** Several agents work in this tree at
once, and `git commit` commits the *index*, which someone else may have staged into
while you were thinking. Always:

```
git -c user.name="..." -c user.email="..." commit -F msg.txt -- path/one path/two
```

Never a bare `git commit`, never `git commit -a`, never `git add -A`. Before editing a
file, `git status --porcelain -- <file>`: if it is already modified it belongs to
somebody else right now, and a pathspec commit of it would sweep their work into your
history. Leave it and say so.

**4. Anything on the live path is default-off and bit-identical when off.** The trainer
runs for days and imports this tree through a `.pth` on `sys.path`. A new feature ships
behind a config flag that defaults to off, and the off path is asserted with
`torch.equal`, not `allclose` — a feature that is merely "numerically indistinguishable"
from off is a feature that changed the run.
`test_group_routing.py::test_rollback_is_bit_identical` is the shape to copy: it
parametrises over *four* off-configurations, because "absent", "disabled", "enabled with
zero weights" and "weighted but disabled" are four different ways to be off and only one
of them had ever been checked.

**5. Report reach with the regime it was measured in.** "0.31 of groups qualified" means
nothing without the dataset, the model, the group size and the date. This is structural
on the supplier axis — a `SupplyCapability` carrying a `reach` without a `reach_regime`
is refused at registration — and it is expected everywhere else. Two arms measured on
different difficulty mixes are not comparable, and printing them side by side is a claim
you did not test.

______________________________________________________________________

## 2. Where things go

| Kind                     | Location                           | Collected by pytest? |
| ------------------------ | ---------------------------------- | -------------------- |
| Test                     | `selfevo/tests/test_<module>.py`   | **yes**              |
| Mutation harness         | `selfevo/tests/mutate_<module>.py` | **no**               |
| Shared fixture           | `selfevo/tests/conftest.py`        | n/a                  |
| Scratch analysis, sweeps | `experiments/`                     | not in the CI job    |
| Findings                 | `selfevo/FINDINGS_*.md`            | n/a                  |

`selfevo/tests/` is **flat**. No subdirectories, no mirroring of the source tree. A test
for a new router goes in that directory next to everything else.

The `test_` / `mutate_` split is enforced by tooling, not policy: pytest collects the
first prefix and not the second. **Getting it backwards is dangerous**, because several
`mutate_*.py` files write to source at import time, and the first-generation ones write
to `~/areal-selfevo` — the live checkout a training job is reading.

______________________________________________________________________

## 3. Writing a test

The house style is **adversarial behavioural testing driven through the real production
call**, not unit testing. Seven things are expected of a new test file.

1. **Drive the real entry point.** For a router that is `PPOActor._compute_advantages`
   through `make_actor` / `advantages`, never `_route_groups` directly, and never a
   local re-derivation of the module's arithmetic. A sibling file once isolated the
   `group_ids` expression as a `derive()` helper; that pins a *copy* of the code and
   cannot notice the copy drifting from the original.
1. **Open with a premise test.** *"If this fails, every test below is testing nothing."*
   Prove the fixture is non-vacuous before asserting anything about it — that the silent
   group really is silent, that the batch really does contain both kinds of row.
1. **Assert bit-identical rollback** when the feature is off. Rule 4.
1. **Assert the feature does not touch what it should not.** The prompt region stays at
   zero; the informative group is untouched; the input batch is not mutated.
1. **Ship a control for any targeting or selection claim**, rate-matched or shuffled, at
   the same rate. Three findings in this repo turned on a targeted rule being
   indistinguishable from a random one at the same rate. An uncontrolled targeting claim
   will not be believed here, and should not be.
1. **Import shared fixtures, do not copy them.** `selfevo/tests/conftest.py` holds
   `ctx`, `mode_of`, `Recorder` / `recorder`, `clear_stats_tracker`,
   `registered_router`, `stub_router`, and the group-routing batch (`B`, `T`, `G`,
   `PROMPT`, `MIXED`, `SOLVED_AND_UNSOLVED`, `make_actor`, `make_batch`, `meta`,
   `advantages`). Two definitions of "an actor configured like the live runs" drift, and
   the drift is silent. Copy a fixture only when the *difference* is the subject of your
   file, and then say so in the docstring — `test_routing_stabilisers.py` keeps its own
   batch because variable generation lengths are what it tests.
1. **Gate heavy imports and stay on CPU.** Module-level `pytest.importorskip("torch")`
   and friends. There are zero `torch.cuda` references in this suite and it should stay
   that way. A test that reaches the network gets `@pytest.mark.slow`; CI runs
   `-m "not slow"`.

Also true of the existing suite and worth keeping: no golden files, no `hypothesis`, no
`caplog`. Sweeps are hand-rolled `itertools.product` grids. Fakes are plain classes
defined in the file that uses them. Docstrings say what the test *catches*, not what it
does.

______________________________________________________________________

## 4. Adding one of each thing

Five extension axes. They are not equally easy, and the difference is worth knowing
before you start rather than after.

### 4.1 A ninth router — 2 files mandatory, 5 realistic

| #   | File                           | Change                                                                                                                                                                         |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `selfevo/routing/<new>.py`     | NEW. `route(ctx) -> RoutingDecision`; add `route_batch` if it partitions, `observe` if it learns                                                                               |
| 2   | `selfevo/compose.py`           | **two** edits: a `def _<new>_router(**kw)` lazy-import wrapper among the others (`_static_router` … `_rule_router`, `:69-219`), and one line in the `ROUTERS` literal (`:221`) |
| 3   | `selfevo/routing/__init__.py`  | optional re-export                                                                                                                                                             |
| 4   | `selfevo/compose.py` docstring | the axis table at the top of the file                                                                                                                                          |
| 5   | `GOAL.md` / `EXPERIMENTS.md`   | arm listing                                                                                                                                                                    |

No test edit is needed: `test_gold_target_reachability.py:176` auto-discovers through
`sorted(compose.ROUTERS.items())`. That is the one place discovery is done right in this
tree, and the model for the others.

Two things that look wrong and are not. `register_router` exists (`compose.py:349`) with
zero non-test callers, and nothing eagerly imports the routing modules, so a
self-registering file would be dead code — the literal dict is load-bearing. And
`_route_groups` calls `factory()` with **no kwargs** (`actor.py:432`), so every
experiment-deciding default must be baked into the wrapper. **The wrapper is the seam;
the registration is a function, not a line.** `_random_router` (`compose.py:121`) and
the block above `_contextual_router` (`:170`) carry retraction comments recording two
arms that ran bit-identical to the off arm because a default lived somewhere `factory()`
could not reach.

**Known gap.** `GroupRoutingConfig.__post_init__` (`cli_args.py:1957`) validates
`credit`, `decision` and `harness_variants` and never `router`. A typo therefore
survives config parse and dies after model load, on GPU, at the first batch.
`harness_variants` in that same method shows the right pattern — it resolves against
`VARIANTS` before any GPU is booked. If you are already editing that file, extend it.

### 4.2 A third partition — 1 file mandatory (4 edits), 6 realistic

| #   | File                                           | Change                                                                                                                                                                                        |
| --- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `selfevo/cluster_lora/partition.py`            | the `TRAINING_PARTITIONS` tuple (`:63`) **and** `DISPATCHED_PARTITIONS` (`:67`); a new `<new>_partition() -> Partition`; a branch in `partition_from_config`; `__all__`; the module docstring |
| 2   | `selfevo/cluster_lora/wiring.py`               | the behavioural-forward branch, if the mode needs no features; three docstrings                                                                                                               |
| 3   | `selfevo/cluster_lora/features.py`             | docstring                                                                                                                                                                                     |
| 4   | `selfevo/cluster_lora/__init__.py`             | `__all__`, import                                                                                                                                                                             |
| 5   | `selfevo/tests/test_cluster_lora_partition.py` | the `parametrize("mode", [...])` grid                                                                                                                                                         |
| 6   | `selfevo/FINDINGS_cluster_lora.md`             | doc                                                                                                                                                                                           |

Validation is centralised — `wiring.py` and `features.py` both read
`TRAINING_PARTITIONS` and need no edit — which makes this the closest axis to the ideal.

**The trap here has been closed, and the fix is the pattern to copy.**
`partition_from_config` used to end in an unconditional
`return random_matched_partition(...)`, so a fourth name added to `TRAINING_PARTITIONS`
without a branch silently ran the **control's** mechanism under the new arm's label —
and since the returned `Partition` was bit-identical to the control's, nothing
downstream could tell afterwards which one produced the table. It now raises, naming
`DISPATCHED_PARTITIONS`. Add your name to *both* tuples and add the branch. What is
still hand-maintained is a mode list duplicated into four prose docstrings that no test
checks.

### 4.3 A selector — the registry does not exist yet

`EXPERIMENTS.md:246` documents `GroupRoutingConfig.harness_selector` +
`harness_selector_args` resolved through a `SELECTORS` registry, validated before any
GPU is booked. `grep -rn SELECTORS` over every `.py` returns **zero hits**. Git explains
it: `ecf97f84` added it, `3677694c` removed it, and neither commit is an ancestor of
HEAD.

The selector *classes* survive — `TruncationStepLimitSelector`
(`harness/selectors.py:417`) and `RateMatchedControlSelector` (`:638`). The only way to
install one is `HarnessDispatcher(selector=...)` (`harness/dispatch.py:263`), and
`build_dispatcher` (`:502`), the single production entry point, never passes it.
**Production always gets `round_robin`** (`dispatch.py:91`), the placeholder whose own
docstring says it answers a different question than the one the paper asks. The
rate-matched harness control is currently unreachable from every arm.

Adding a selector therefore means restoring the registry first:

| #   | File                                           | Change                                                                                                                                                                |
| --- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `selfevo/harness/selectors.py`                 | NEW class beside the two that exist                                                                                                                                   |
| 2   | `selfevo/harness/dispatch.py`                  | a `SELECTORS` literal; `build_dispatcher` resolves it and passes `selector=` instead of falling through to `round_robin`                                              |
| 3   | `areal/api/cli_args.py`                        | `harness_selector` / `harness_selector_args` on `GroupRoutingConfig`, **validated in `__post_init__` against the registry**, exactly as `harness_variants` already is |
| 4   | `selfevo/tests/test_harness_dispatch_wired.py` | a wired test — a selector proved only through `HarnessDispatcher` cannot tell a reachable arm from an unreachable one                                                 |

Do not add a selector class alone. An unreachable arm that looks configured is how this
axis got into its present state.

### 4.4 A loss mode alongside RL / SFT — 8+ code files, 2 test files

The application seam is the largest extension debt in the tree. The *registry* is
genuinely open (`routing/base.py::register_mode`); the *application* is an if/elif over
exactly two modes, duplicated across roughly eleven files.

| #   | File                                    | Change                                                                                                                                                                                                                          |
| --- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `selfevo/routing/base.py:74`            | `register_mode(name, needs_teacher=..., applicable=True)`                                                                                                                                                                       |
| 2   | `selfevo/integration/group_apply.py:46` | add to `_APPLIED` — necessary, nowhere near sufficient                                                                                                                                                                          |
| 3   | same, `apply_decisions`                 | an if/elif over RL/SFT with a single `sft_weight` scalar                                                                                                                                                                        |
| 4   | same, `apply_mixtures`                  | hardcodes **exactly two** blend terms and builds the two extremes by recursive `apply_decisions` calls                                                                                                                          |
| 5   | `areal/api/cli_args.py`                 | a magnitude field on `GroupRoutingConfig` beside `solved_advantage` / `unsolved_advantage`, plus sign validation in `__post_init__`                                                                                             |
| 6   | `areal/trainer/ppo/actor.py`            | `_route_groups` passes `sft_weight=` and the `sft_rows` veto; `exclude_truncated_from_sft` is SFT-specific                                                                                                                      |
| 7   | 8 router files                          | `contextual.py:96`, `feedback.py:131`, `cluster.py`, `routers.py:100/215`, `rule_policy.py`, `harness.py`, `code_policy.py`, `credit_sim.py` — each carries its own hardcoded `(RL, SFT, SKIP)` tuple or `teacher_mode` default |
| 8   | 2 test files                            | `test_group_apply.py:48` and `test_group_mixture.py:58` each carry a separate `MODES` literal                                                                                                                                   |

**The registry/seam drift is now caught at import, and that mechanism is worth
understanding before you add a mode.** `group_apply.py:55-63` computes
`applicable_modes() - _APPLIED` and its converse and raises a `RuntimeError` at **import
time** if either is non-empty. Registering a mode as applicable without implementing a
branch no longer costs a rollout to discover; it fails to import. `selfevo.routing` is
torch-free and cannot import `group_apply`, which is why the flag lives in the registry
and the assertion lives in the seam.

`TrainingMode.DISTILL` is registered with `applicable=False` and is the reference case:
named, reasoned about, not implemented, and therefore un-selectable by a
default-configured router rather than a mode that pays for a full rollout and never
learns. If your mode is not yet implemented end to end, register it that way rather than
leaving it applicable.

### 4.5 A supplier — 2 files, once the prerequisites land

Target state:

| #   | File                         | Change                                                                                                                                              |
| --- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `selfevo/supply/<new>.py`    | NEW: `capability()`, `probe()`, `supply()`                                                                                                          |
| 2   | `selfevo/supply/__init__.py` | a lazy factory wrapper plus one line in the `SUPPLIERS` literal — deliberately the same shape as `ROUTERS`, wrapper and all, for the reason in §4.1 |

**Today it is ~11 files**, the list in §4.4, because a supplier's mode *is* a loss mode.
Three things must land first:

- **S1** — open the `_APPLIED` seam, or the supplier's mode has no target tensor and is
  rejected exactly where an applicable-but-unimplemented mode would be.
- **S2** — lift routing from stage 817 to stage 710. The router currently runs two
  stages after the point a supplier must act. `group_features` needs nothing that is
  unavailable at 710, so this is a move, not a redesign.
- **S3** — replace `actor.py:508`'s literal `has_teacher=False` with a broker query.
  Until then the gate cannot be honest and no teacher arm is reachable.

Three parts of the protocol are not decoration:

- `Scorability.NONE` is the **honest default**. A router may gate only on what a
  supplier declares; it may not assume per-item fitness exists.
- `reach` and `reach_regime` are **one unit**. A capability carrying a reach without a
  regime is refused at registration, which makes rule 5 structural instead of a
  convention.
- `unserved` is a **mapping of reasons, not a count**. Every zero in this repo has an
  artifact behind it.

`SUPPLIERS` should be validated in `GroupRoutingConfig.__post_init__` against the
registry, following `harness_variants`.

### 4.6 Why the registries do not look alike

`ROUTERS` is a dict of factories. `VARIANTS` is a dict plus a `register_variant()`
populated by import side effect. `_APPLIED` is a private tuple cross-checked against the
mode registry at import. `TRAINING_PARTITIONS` is a module-level tuple *plus* a second
`DISPATCHED_PARTITIONS` tuple and an if/elif chain. `GoldRule` is a closed Enum.
`key_fn` is a bare mutable callable field. `SELECTORS` does not exist. **Six seams, five
shapes**: you cannot learn one and apply it to the next. New axes should copy `ROUTERS`;
existing ones should move toward it when they are being touched for another reason
anyway.

One undeclared contract worth knowing before you write to a type signature.
`ClusterRouter` types `key_fn` as `Callable[[RoutingContext], str]`, but its only real
implementation, `ClusterLoRAKeyFn` (`cluster_lora/features.py:354`), is a stateful
object that *additionally* requires `begin_batch(unit_ids, features, group_ids=...)` to
be driven out of band from `wiring.begin_cluster_batch` (`cluster_lora/wiring.py:451`).
Writing to the declared type gets you a `key_fn` that raises on every lookup.

______________________________________________________________________

## 5. Mutation harnesses

A harness copies the repo, mutates one anchor in the copy, runs a named test selection
against the copy, and records which test id killed it. It is a script, never collected:

```
python selfevo/tests/mutate_<module>.py <repo-copy> [<live-repo>]
```

Three generations exist. **Copy the third.** `mutate_harness_selectors.py` (446 lines)
and `mutate_group_mixture.py` (328) are the reference implementations, and the
behaviours they carry are all non-negotiable:

- **Refuse to start on a red baseline.** A mutation score against a suite that was
  already failing measures nothing.
- **`_assert_isolated()`** — prove pytest imported the *copy*. A harness that silently
  tested the live tree reports every mutation as killed, for the wrong reason.
- **`_assert_matches_live()`** — sha256 the copy against the live file before *and*
  after.
- **A four-way SKIP taxonomy.** Rule 2: non-unique anchor, zero-byte change,
  uncompilable mutant, literal `\n` in the replacement text.
- **Kill attribution by test id**, not a red/green bit. "Something failed" does not tell
  you whether the guard you meant to test is the one that fired.
- **A SIGINT/SIGTERM handler** that reaps the child and prints proof of restore before
  `os._exit(130)`. Interrupting a harness must never leave a mutated source behind.

First-generation harnesses that set `ROOT = Path.home()/"areal-selfevo"` and mutate the
live checkout are a hazard while a job is running. Migrate them; do not imitate them.

The visible coverage gap is `cluster_lora`: nothing mutates `interference_dump.py` (the
single largest module in the subpackage), `features.py`, `sketch.py` or `merge.py` in
isolation, and that subpackage has the lowest test-to-source ratio of any substantial
one while being the current method.

______________________________________________________________________

## 6. CI

`.github/workflows/selfevo-tests.yml` runs `pytest selfevo/tests -m "not slow"` on
`ubuntu-latest`, CPU only, on any pull request touching `selfevo/**` or one of the
handful of `areal/` files this fork modifies. It is the first CI job that has ever run a
selfevo test — `test-areal.yml` passes an explicit path list that excludes this
directory and needs a self-hosted GPU runner, and `install-test.yml` is path-filtered to
`areal/**`.

Two steps in it are load-bearing and should not be pruned as noise:

- **Refuse a vacuous run.** Almost every file here opens with `pytest.importorskip`. A
  missing dependency turns the suite into a green no-op rather than a red failure, so
  the imports it depends on are asserted in a step where absence is an error.
- **Refuse a hollow run.** The JUnit report is parsed and the job fails if fewer than
  1,500 tests actually executed. If that floor ever trips, a subpackage went missing:
  investigate it, do not lower it.

`experiments/` is deliberately outside the job. It is a scratch surface and should stay
one.

______________________________________________________________________

## 7. Before you open a pull request

- `pytest selfevo/tests -m "not slow" -q` is green, or every failure is one you can name
  and did not cause.
- Any guard you added has a mutation harness, you ran it, and the SKIPs are reported
  separately from the kills.
- Anything on the live path is off by default and `torch.equal`-identical when off.
- Every new function and class has a docstring saying what it is for; every new test
  docstring says what it *catches*.
- Any test that reaches the network carries `@pytest.mark.slow`.
- The commit is pathspec-limited and names only files you actually changed.
- Every measured number in the message carries the regime it was measured in.
