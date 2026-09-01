#!/usr/bin/env python3
"""Mutation-test the harness dispatch axis against a byte-identical copy of the repo.

Usage: ``mutate_harness_dispatch.py <repo-copy> [<live-repo>]``

**A copy, not the live checkout, and the reason is on the box.** ``experiments/harness/
lora30b.sh`` runs with ``PYTHONPATH=/home/ubuntu/areal-selfevo`` across a dozen worker
processes that relaunch, so a mutated ``dispatch.py`` or ``actor.py`` sitting on disk for
even a few seconds could be imported by a live training run. ``mutate_harness_router.py``
adopted the same rule for the same reason. The copy is not a re-derivation: every anchor
below is matched against the file's real bytes, and when ``<live-repo>`` is given the sha256
of each target is asserted equal between the copy and the live checkout before the first
mutation and after the last, so a mutation that killed a test killed it in the production
source text and nothing else.

**What counts as evidence, and what does not.** Three things have been mistaken for a
mutation here before -- a replacement identical to what it replaced, a change that is
semantically a no-op, and an escape sequence passed through as two literal characters so the
"mutation" landed inside a comment. So every mutation is verified before it is scored:

* the anchor must appear exactly once, or the mutation is a SKIP;
* the rewritten file must differ from the original in BYTES, or it is a SKIP;
* the rewritten file must still COMPILE, or it is a SKIP -- a SyntaxError fails every test
  in the suite and would otherwise be scored as the strongest kill in the table while
  proving nothing about any guard;
* neither the anchor nor the replacement may contain a literal backslash-n, because that is
  how the escaping trap presents and there is no legitimate use of one here.

A SKIP is reported as a SKIP. It is never reported as SURVIVED and never as killed.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None

DISPATCH = "selfevo/harness/dispatch.py"
ACTOR = "areal/trainer/ppo/actor.py"
CLI = "areal/api/cli_args.py"

TESTS = [
    "selfevo/tests/test_harness_dispatch.py",
    "selfevo/tests/test_harness_dispatch_wired.py",
    # The pre-existing guard tests. A mutation that weakened _refuse_dropped_harness or the
    # consumer flag must be caught by the tests that were already there, not only by mine.
    "selfevo/tests/test_harness_axis_guard.py",
]

BACKSLASH_N = chr(92) + "n"

# (label, file, find, replace). Each is a single defect a careless edit could produce.
MUTATIONS = [
    # ---- can_evolve: the floor that decides whether the axis exists at all -------------
    ("can_evolve floor lowered to 1: a single-variant control reports as evolvable",
     DISPATCH,
     "        return len(self._variants) >= 2",
     "        return len(self._variants) >= 1"),
    ("can_evolve floor raised to 3: a legal two-variant arm cannot move",
     DISPATCH,
     "        return len(self._variants) >= 2",
     "        return len(self._variants) >= 3"),
    ("can_evolve becomes a truthiness check, so any configured set is evolvable",
     DISPATCH,
     "        return len(self._variants) >= 2",
     "        return bool(self._variants)"),
    ("can_evolve turns exclusive, rejecting exactly the smallest evolvable set",
     DISPATCH,
     "        return len(self._variants) >= 2",
     "        return len(self._variants) > 2"),

    # ---- apply(): what each action is allowed to do ------------------------------------
    ("PROPOSE no longer checks whether there is anywhere to go",
     DISPATCH,
     "        if not self.can_evolve:",
     "        if False:"),
    ("VALIDATE falls through to the proposal path and moves the harness",
     DISPATCH,
     "        if action is HarnessAction.VALIDATE:\n            return DispatchRecord(\n"
     "                action, before, before, False, f\"validate current variant "
     "{before!r}\"\n            )",
     "        if action is HarnessAction.PROPOSE and False:\n            return "
     "DispatchRecord(\n                action, before, before, False, f\"validate current "
     "variant {before!r}\"\n            )"),
    ("NONE falls through to the proposal path, so the inert default is not inert",
     DISPATCH,
     "        if action is HarnessAction.NONE:\n            return DispatchRecord(action, "
     "before, before, False, \"no harness action\")",
     "        if action is HarnessAction.VALIDATE:\n            return DispatchRecord(action, "
     "before, before, False, \"no harness action\")"),
    ("a refused proposal reports changed=True, so the switch count flatters the arm",
     DISPATCH,
     "                False,\n                f\"proposal refused: {len(self._variants)} "
     "variant(s) configured, \"",
     "                True,\n                f\"proposal refused: {len(self._variants)} "
     "variant(s) configured, \""),
    ("the selector's output is no longer checked against the configured set",
     DISPATCH,
     "        if chosen not in self._variants:",
     "        if False:"),
    ("the set-membership check is inverted, so every legal proposal is rejected",
     DISPATCH,
     "        if chosen not in self._variants:",
     "        if chosen in self._variants:"),
    ("a selector that stays put is accepted, making a proposal a silent no-op",
     DISPATCH,
     "        if chosen.name == before:",
     "        if False:"),
    ("the action type check is dropped, so a typo dispatches as a real action",
     DISPATCH,
     "        if not isinstance(action, HarnessAction):",
     "        if False:"),
    ("the selection is computed but never stored, so the harness never actually moves",
     DISPATCH,
     "        self._active = chosen",
     "        self._active = self._active"),
    ("the set starts on its last member rather than its first",
     DISPATCH,
     "        self._active = variants[0] if variants else None",
     "        self._active = variants[-1] if variants else None"),

    # ---- construction guards -----------------------------------------------------------
    ("duplicate variant names are accepted, so one scaffold looks like two",
     DISPATCH,
     "        if dupes:",
     "        if False:"),
    ("behaviourally identical variants are accepted, so an arm dispatches to itself",
     DISPATCH,
     "        if len(variants) > 1 and len(behaviours) == 1:",
     "        if False:"),
    ("the behaviour check reads only step_limit, rejecting variants that differ in settings",
     DISPATCH,
     "            (v.step_limit, repr(sorted(v.settings.items(), key=lambda kv: kv[0])))",
     "            (v.step_limit,)"),
    ("the behaviour check fires on a single variant, so the control arm cannot be built",
     DISPATCH,
     "        if len(variants) > 1 and len(behaviours) == 1:",
     "        if len(variants) > 0 and len(behaviours) == 1:"),

    # ---- consume(): the aggregation that keeps a batch from cancelling itself -----------
    ("the batch is marked as having acted before it has, so nothing ever moves",
     DISPATCH,
     "        acted = False",
     "        acted = True"),
    ("the aggregation condition is inverted, so the first proposal is the one discarded",
     DISPATCH,
     "            if action is HarnessAction.PROPOSE and acted:",
     "            if action is HarnessAction.PROPOSE and not acted:"),
    ("the batch forgets it acted, so every proposal rotates and an even count cancels",
     DISPATCH,
     "            if record.changed:\n                acted = True",
     "            if record.changed:\n                acted = False"),
    # NOT a mutation, and deliberately absent: swapping ``record.changed`` for
    # ``record.action is HarnessAction.PROPOSE`` here is an EQUIVALENT mutant. apply()
    # returns changed=True for a PROPOSE exactly when can_evolve is True, and can_evolve
    # cannot change inside a batch, so the two predicates agree on every input this loop can
    # produce. It would alter bytes, compile, run, and survive -- and scoring that as a
    # survivor would report a no-op as a gap in the tests. The reachable defect in the same
    # place is the round-2 entry that keys on "some action happened" instead.

    # ---- round_robin -------------------------------------------------------------------
    ("round_robin returns the current variant, so no proposal can ever move",
     DISPATCH,
     "    return variants[(names.index(current.name) + 1) % len(variants)]",
     "    return variants[names.index(current.name) % len(variants)]"),
    ("round_robin walks backwards, which is invisible on a two-member set",
     DISPATCH,
     "    return variants[(names.index(current.name) + 1) % len(variants)]",
     "    return variants[(names.index(current.name) - 1) % len(variants)]"),
    ("round_robin drops the wrap, so the last variant has nowhere to go",
     DISPATCH,
     "    return variants[(names.index(current.name) + 1) % len(variants)]",
     "    return variants[names.index(current.name) + 1]"),
    ("round_robin stops checking that current is a member",
     DISPATCH,
     "    if current.name not in names:",
     "    if False:"),
    ("round_robin stops checking for an empty set",
     DISPATCH,
     "    if not variants:",
     "    if False:"),

    # ---- metrics: the only thing that makes the axis auditable after the run ------------
    ("switches counts every record, not the ones that moved the harness",
     DISPATCH,
     "        return sum(1 for r in self.records if r.changed)",
     "        return sum(1 for r in self.records)"),
    ("every variant is reported as active, so the logs cannot name the live scaffold",
     DISPATCH,
     "            out[f\"route/harness_active_{name}\"] = float(name == self.active)",
     "            out[f\"route/harness_active_{name}\"] = 1.0"),
    ("the propose counter is wired to the validate count",
     DISPATCH,
     "            \"route/harness_propose\": float(self.count(HarnessAction.PROPOSE)),",
     "            \"route/harness_propose\": float(self.count(HarnessAction.VALIDATE)),"),
    ("can_evolve is logged as a constant, so a control cannot be told from an arm",
     DISPATCH,
     "            \"route/harness_can_evolve\": float(self.can_evolve),",
     "            \"route/harness_can_evolve\": 1.0,"),
    ("only the active variant gets a key, so the emitted key set moves between steps",
     DISPATCH,
     "        for name in self.variant_names:\n            out[f\"route/harness_active_"
     "{name}\"] = float(name == self.active)",
     "        for name in self.variant_names:\n            if name == self.active:\n"
     "                out[f\"route/harness_active_{name}\"] = 1.0"),
    ("the switch count is logged as the proposal count",
     DISPATCH,
     "            \"route/harness_switches\": float(self.switches),",
     "            \"route/harness_switches\": float(self.count(HarnessAction.PROPOSE)),"),

    # ---- build_dispatcher and the adapter seam -----------------------------------------
    ("an unknown variant name is silently skipped instead of refused",
     DISPATCH,
     "    if unknown:",
     "    if False:"),
    ("an empty variant list builds a dispatcher rather than no harness arm",
     DISPATCH,
     "    if not names:\n        return None",
     "    if names is None:\n        return None"),
    ("a missing adapter is no longer refused before a rollout is claimed",
     DISPATCH,
     "        if self.adapter is None:",
     "        if False:"),
    ("a rollout runs under the first variant rather than the active one",
     DISPATCH,
     "        return self.adapter.run(task_id, self._active)",
     "        return self.adapter.run(task_id, self._variants[0])"),

    # ---- actor.py: where can_evolve_harness and the consumer flag are decided -----------
    ("can_evolve_harness hardcoded True: every run claims an evolvable harness",
     ACTOR,
     "                can_evolve_harness=dispatcher is not None and dispatcher.can_evolve,",
     "                can_evolve_harness=True,"),
    ("can_evolve_harness drops the can_evolve conjunct: a one-variant arm may propose",
     ACTOR,
     "                can_evolve_harness=dispatcher is not None and dispatcher.can_evolve,",
     "                can_evolve_harness=dispatcher is not None,"),
    ("can_evolve_harness pinned False, returning the axis to being inert",
     ACTOR,
     "                can_evolve_harness=dispatcher is not None and dispatcher.can_evolve,",
     "                can_evolve_harness=False,"),
    ("harness_consumer hardcoded True: the refusal guard is bypassed, not satisfied",
     ACTOR,
     "        decisions = route_all(router, contexts, harness_consumer=dispatcher is not None)",
     "        decisions = route_all(router, contexts, harness_consumer=True)"),
    ("harness_consumer keyed on can_evolve, so the control arm raises instead of running",
     ACTOR,
     "        decisions = route_all(router, contexts, harness_consumer=dispatcher is not None)",
     "        decisions = route_all(router, contexts, harness_consumer=dispatcher is not None "
     "and dispatcher.can_evolve)"),
    ("harness_consumer left at its default, so a real consumer is refused",
     ACTOR,
     "        decisions = route_all(router, contexts, harness_consumer=dispatcher is not None)",
     "        decisions = route_all(router, contexts)"),
    ("the dispatcher is rebuilt every batch, resetting the selection it exists to keep",
     ACTOR,
     "        if not hasattr(self, \"_selfevo_harness\"):",
     "        if True:"),
    ("the decisions' harness actions never reach the dispatcher",
     ACTOR,
     "            harness_batch = dispatcher.consume(d.harness for d in decisions)",
     "            harness_batch = dispatcher.consume([])"),
    ("the harness metrics are computed and then not logged",
     ACTOR,
     "            stats_tracker.scalar(**harness_batch.as_metrics())",
     "            harness_batch.as_metrics()"),
    ("consumption is gated on can_evolve, so the control arm emits no harness keys",
     ACTOR,
     "        if dispatcher is not None:\n            harness_batch = dispatcher.consume",
     "        if dispatcher is not None and dispatcher.can_evolve:\n            harness_batch "
     "= dispatcher.consume"),

    # ---- cli_args.py: the config guards --------------------------------------------------
    ("the config accepts a repeated variant name",
     CLI,
     "            if dupes:",
     "            if False:"),
    ("the config stops resolving names, deferring an unknown one to the first batch",
     CLI,
     "            unknown = [n for n in self.harness_variants if n not in VARIANTS]",
     "            unknown = []"),

    # ---- Round 2: the reachable cases round 1 did not cover -----------------------------
    ("a validation earlier in the batch counts as having acted, silencing later proposals",
     DISPATCH,
     "            if record.changed:\n                acted = True",
     "            if record.action is not HarnessAction.NONE:\n                acted = True"),
    ("VALIDATE reports changed=True, so a measurement step logs as a switch",
     DISPATCH,
     "                action, before, before, False, f\"validate current variant {before!r}\"",
     "                action, before, before, True, f\"validate current variant {before!r}\""),
    ("NONE reports changed=True, so a wholly inert batch logs switches",
     DISPATCH,
     "            return DispatchRecord(action, before, before, False, \"no harness action\")",
     "            return DispatchRecord(action, before, before, True, \"no harness action\")"),
    ("the batch reports the first variant as active rather than the live one",
     DISPATCH,
     "            active=self._active.name if self._active is not None else None,",
     "            active=self._variants[0].name if self._variants else None,"),
    ("the batch reports can_evolve as a constant",
     DISPATCH,
     "            can_evolve=self.can_evolve,",
     "            can_evolve=True,"),
    ("build_dispatcher reverses the configured order, so run history is not reproducible",
     DISPATCH,
     "    return HarnessDispatcher([VARIANTS[n] for n in names])",
     "    return HarnessDispatcher([VARIANTS[n] for n in reversed(names)])"),
    ("can_evolve_harness computed from the config length rather than from the dispatcher",
     ACTOR,
     "                can_evolve_harness=dispatcher is not None and dispatcher.can_evolve,",
     "                can_evolve_harness=bool(getattr(gr, \"harness_variants\", None)),"),
    ("the config's duplicate test is off by one, so a name repeated twice passes",
     CLI,
     "                {n for n in self.harness_variants if self.harness_variants.count(n) > 1}",
     "                {n for n in self.harness_variants if self.harness_variants.count(n) > 2}"),
    ("the config stops requiring a router, so a harness arm can dispatch nothing",
     CLI,
     "            if not self.router:",
     "            if False:"),
    ("the whole harness_variants validation block is skipped",
     CLI,
     "        if self.harness_variants is not None:",
     "        if False:"),
]


def run_tests() -> tuple[bool, str]:
    """Run the harness-axis suite against ``REPO``.

    Returns:
        ``(passed, first failing test id)``. The test id is carried so the kill table can
        say WHICH assertion caught a mutant, which is the difference between "the suite went
        red" and "the guard is constrained".
    """
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env,
    )
    who = ""
    for line in r.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            who = line.split(" ")[1].split("::")[-1]
            break
    if not who:
        for line in r.stdout.splitlines():
            if "::" in line and (" - " in line or line.startswith("_" * 5)):
                who = line.strip("_ ").split("::")[-1].split(" ")[0]
                break
    return r.returncode == 0, who


def _sha(p: pathlib.Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _assert_isolated() -> None:
    """Refuse to run unless pytest imports the COPY, not any other checkout."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.harness.dispatch as d, areal.trainer.ppo.actor as a; "
         "print(d.__file__); print(a.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=600,
    )
    lines = [pathlib.Path(x.strip()).resolve() for x in r.stdout.splitlines() if x.strip()]
    want = [REPO / DISPATCH, REPO / ACTOR]
    if lines != want:
        raise SystemExit(f"ISOLATION FAILED: imports resolve to {lines}, not {want}\n{r.stderr}")
    print(f"isolated: imports resolve inside {REPO}")


def _assert_matches_live(targets: list[str]) -> None:
    """Assert the copy is byte-identical to the live checkout, so the anchors are real.

    Without this the harness would be mutating a re-derivation, and a kill would say nothing
    about the file training actually imports.
    """
    if LIVE is None:
        print("no live repo given; skipping byte-identity check")
        return
    for rel in targets:
        a, b = _sha(REPO / rel), _sha(LIVE / rel)
        if a != b:
            raise SystemExit(f"COPY DIVERGED from live at {rel}: {a} != {b}")
    print(f"copy is byte-identical to {LIVE} for {len(targets)} target file(s)")


def main() -> int:
    targets = sorted({m[1] for m in MUTATIONS})
    _assert_isolated()
    _assert_matches_live(targets)

    original = {rel: (REPO / rel).read_text() for rel in targets}
    digests = {rel: _sha(REPO / rel) for rel in targets}

    ok, _ = run_tests()
    if not ok:
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    killed, survivors, skips = [], [], []
    for label, rel, find, repl in MUTATIONS:
        path = REPO / rel
        src = original[rel]

        if BACKSLASH_N in find or BACKSLASH_N in repl:
            skips.append((label, rel, "literal backslash-n in the mutation text"))
            print(f"SKIP      [{rel}] {label}: literal backslash-n")
            continue
        n = src.count(find)
        if n != 1:
            skips.append((label, rel, f"anchor appears {n}x"))
            print(f"SKIP      [{rel}] {label}: anchor appears {n}x")
            continue
        mutated = src.replace(find, repl, 1)
        if mutated == src:
            skips.append((label, rel, "replacement left the file byte-identical"))
            print(f"SKIP      [{rel}] {label}: file byte-identical")
            continue
        try:
            compile(mutated, str(path), "exec")
        except SyntaxError as exc:
            skips.append((label, rel, f"mutant does not compile ({exc.msg})"))
            print(f"SKIP      [{rel}] {label}: mutant does not compile")
            continue

        path.write_text(mutated)
        try:
            passed, who = run_tests()
        finally:
            path.write_text(src)
            assert _sha(path) == digests[rel], f"restore failed for {rel}"

        if passed:
            survivors.append((label, rel))
            print(f"SURVIVED  [{rel}] {label}")
        else:
            killed.append((label, rel, who))
            print(f"killed    [{rel}] {label}  <- {who}")

    for rel in targets:
        assert _sha(REPO / rel) == digests[rel], f"final restore check failed for {rel}"
    _assert_matches_live(targets)

    print(f"\n{len(killed)} killed, {len(survivors)} survived, {len(skips)} skipped "
          f"({len(MUTATIONS)} total)")
    if skips:
        print("\nSKIPPED (not applied, so not evidence either way):")
        for label, rel, why in skips:
            print(f"  - [{rel}] {label}: {why}")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, rel in survivors:
            print(f"  - [{rel}] {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
