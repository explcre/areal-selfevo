#!/usr/bin/env python3
"""Mutation-test the feature-driven harness selector and its rate-matched control.

Usage: ``mutate_harness_selectors.py <repo-copy> [<live-repo>]``

**A copy, not the live checkout, and the reason is on the box.** The training run imports this
tree with ``PYTHONPATH=/home/ubuntu/areal-selfevo`` across worker processes that relaunch, so a
mutated ``selectors.py`` or ``dispatch.py`` sitting on disk for even a few seconds could be
imported by a live run. ``mutate_harness_router.py`` and ``mutate_harness_dispatch.py`` adopted
the same rule for the same reason. The copy is not a re-derivation: every anchor below is
matched against the file's real bytes, and when ``<live-repo>`` is given the sha256 of each
target is asserted equal between the copy and the live checkout before the first mutation and
after the last, so a mutation that killed a test killed it in the production source text and
nothing else.

**What counts as evidence, and what does not.** Three things have been mistaken for a mutation
in this repo before -- a replacement identical to what it replaced, a change that is
semantically a no-op, and an escape sequence passed through as two literal characters so the
"mutation" landed inside a comment. Every mutation is therefore verified before it is scored:

* the anchor must appear exactly once, or the mutation is a SKIP;
* the rewritten file must differ from the original in BYTES, or it is a SKIP;
* the rewritten file must still COMPILE, or it is a SKIP -- a SyntaxError fails every test in
  the suite and would otherwise be scored as the strongest kill in the table while proving
  nothing about any guard;
* neither the anchor nor the replacement may contain a literal backslash-n, because that is
  how the escaping trap presents and there is no legitimate use of one here.

A SKIP is reported as a SKIP. It is never reported as SURVIVED and never as killed.

**Deliberately absent, because it is an EQUIVALENT mutant.** Dropping ``math.isfinite`` from
``if not math.isfinite(value) or not 0.0 <= value <= 1.0`` looks like a real defect and is not
one: every non-finite value already fails the range test. ``nan`` fails it because every
comparison with ``nan`` is False, and ``+-inf`` fails it because it lies outside [0, 1]. So the
two predicates accept and reject exactly the same floats, the mutant would alter bytes,
compile, run and survive, and scoring it as a survivor would report a no-op as a gap in the
tests. The reachable defect in the same place -- dropping the RANGE test and keeping the
finiteness one -- is in the table below and is killed.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None

SEL = "selfevo/harness/selectors.py"
DISPATCH = "selfevo/harness/dispatch.py"

TESTS = [
    "selfevo/tests/test_harness_selectors.py",
    # The pre-existing dispatch suite. The refusal seam is an addition to `apply`, and a
    # mutation that widened it must be caught by the tests that were already there -- not
    # only by mine.
    "selfevo/tests/test_harness_dispatch.py",
]

BACKSLASH_N = chr(92) + "n"

# (label, file, find, replace). Each is a single defect a careless edit could produce.
MUTATIONS = [
    # ---- _check_set: programmer errors must not become data conditions -------------------
    ("the empty-set guard is dropped, so a set that does not exist becomes a refusal",
     SEL,
     "    if not variants:",
     "    if False:"),
    ("the membership guard is dropped, so a caller and a dispatcher that disagree are guessed",
     SEL,
     "    if current is None or current.name not in names:",
     "    if False:"),

    # ---- _neighbour: the direction and the distance --------------------------------------
    ("an equal step_limit counts as LONGER, so the rule moves where the budget is the same",
     SEL,
     "        if longer and delta <= 0:",
     "        if longer and delta < 0:"),
    ("an equal step_limit counts as SHORTER, same defect in the other direction",
     SEL,
     "        if not longer and delta >= 0:",
     "        if not longer and delta > 0:"),
    ("the upward filter keeps SHORTER variants, so a starved batch is given a smaller budget",
     SEL,
     "        if longer and delta <= 0:",
     "        if longer and delta == 0:"),
    ("the downward filter keeps LONGER variants",
     SEL,
     "        if not longer and delta >= 0:",
     "        if not longer and delta == 0:"),
    ("the rule jumps to the FARTHEST budget, claiming a magnitude the feature does not carry",
     SEL,
     "        key = (abs(delta), i)",
     "        key = (-abs(delta), i)"),
    ("distance ignores sign, so nearest is decided by raw delta and the ordering inverts",
     SEL,
     "        key = (abs(delta), i)",
     "        key = (delta, i)"),

    # ---- one observation, one decision ---------------------------------------------------
    ("observe stops opening a new epoch, so the harness freezes after its first decision",
     SEL,
     "        self._epoch += 1",
     "        self._epoch += 0"),
    ("the never-observed guard is dropped, so the first decision is taken on nan",
     SEL,
     "        if self._epoch == 0:",
     "        if False:"),
    ("the repeat guard is dropped, so every proposal in a batch becomes its own decision",
     SEL,
     "        if self._decided_epoch == self._epoch:",
     "        if False:"),
    ("the epoch is never marked decided, so the repeat guard can never fire",
     SEL,
     "        self._decided_epoch = self._epoch",
     "        self._decided_epoch = 0"),
    ("repeat calls are not counted, so a caller that forgot observe looks like a quiet run",
     SEL,
     "            self._repeat_calls += 1",
     "            self._repeat_calls += 0"),

    # ---- the audit record and the counters -----------------------------------------------
    ("the unknown-category guard is dropped, so a mistyped refusal is counted nowhere",
     SEL,
     "            if category not in self._refusals:",
     "            if False:"),
    ("refusals are not counted, so the two kinds of decline are invisible",
     SEL,
     "            self._refusals[category] += 1",
     "            self._refusals[category] += 0"),
    ("moves counts every decision, so a rule that only ever declined reports rate 1.0",
     SEL,
     "        return sum(1 for r in self._records if r.moved)",
     "        return sum(1 for r in self._records)"),
    ("an undefined rate is reported as 0.0, merging 'never asked' with 'always declined'",
     SEL,
     "        if not self._records:\n            return math.nan",
     "        if not self._records:\n            return 0.0"),
    ("outcomes reports every decision as a move, so a replayed schedule is all-move",
     SEL,
     "        return tuple(r.moved for r in self._records)",
     "        return tuple(True for r in self._records)"),
    ("the rate metric is wired to the move COUNT, so the panel plots a cumulative total",
     SEL,
     '            "route/harness_sel_rate": float(self.switch_rate),',
     '            "route/harness_sel_rate": float(self.moves),'),
    ("a metric key drifts from SELECTOR_METRIC_KEYS, so the two arms stop sharing a panel",
     SEL,
     '            "route/harness_sel_moves": float(self.moves),',
     '            "route/harness_sel_move": float(self.moves),'),

    # ---- the treatment rule ---------------------------------------------------------------
    ("the raise trigger turns exclusive, so a batch exactly at the threshold does not move",
     SEL,
     "        if t >= self.raise_above:",
     "        if t > self.raise_above:"),
    ("the lower trigger turns exclusive",
     SEL,
     "        if t <= self.lower_below:",
     "        if t < self.lower_below:"),
    ("truncation raises the budget for everything above the LOWER threshold, erasing the band",
     SEL,
     "        if t >= self.raise_above:",
     "        if t >= self.lower_below:"),
    ("a truncated batch is given a SHORTER budget: the prediction, inverted",
     SEL,
     "            target = _neighbour(variants, current, longer=True)",
     "            target = _neighbour(variants, current, longer=False)"),
    ("an untruncated batch is given a LONGER budget: the symmetric half, inverted",
     SEL,
     "            target = _neighbour(variants, current, longer=False)",
     "            target = _neighbour(variants, current, longer=True)"),
    ("the default raise threshold drops to just above the floor, so the dead band vanishes",
     SEL,
     "        raise_above: float = 0.5,",
     "        raise_above: float = 0.06,"),
    ("the default lower threshold rises to meet it, so the dead band vanishes downward",
     SEL,
     "        lower_below: float = 0.05,",
     "        lower_below: float = 0.45,"),
    ("the threshold ordering guard is dropped, so one value can satisfy both branches",
     SEL,
     "        if lower_below >= raise_above:",
     "        if False:"),
    ("the threshold range guard is dropped, so a branch can be configured never to fire",
     SEL,
     "            if not 0.0 <= value <= 1.0:",
     "            if False:"),

    ("the all-same-budget guard is dropped, so the rule silently becomes its own control",
     SEL,
     "        if len(variants) > 1 and len(limits) == 1:",
     "        if False:"),
    ("the all-same-budget guard fires on a single-variant set, hiding the ordinary refusal",
     SEL,
     "        if len(variants) > 1 and len(limits) == 1:",
     "        if len(variants) > 0 and len(limits) == 1:"),

    # ---- the statistic --------------------------------------------------------------------
    ("the empty-batch guard is dropped, so a batch with no groups is a division by zero",
     SEL,
     "        if not rows:",
     "        if False:"),
    ("the missing-feature guard is dropped, so an absent feature is a bare KeyError",
     SEL,
     "            if self.feature not in row:",
     "            if False:"),
    ("the range half of the finiteness guard is dropped, so 1.5 and -0.1 decide a budget",
     SEL,
     "            if not math.isfinite(value) or not 0.0 <= value <= 1.0:",
     "            if not math.isfinite(value):"),
    ("the batch is reduced by MAX, so one pathological group moves a shared artefact",
     SEL,
     "            total += value",
     "            total = max(total, value)"),
    ("the batch is reduced to its LAST group, so the other groups are evidence for nothing",
     SEL,
     "            total += value",
     "            total = value"),
    ("the sum is never divided, so the statistic scales with the number of groups",
     SEL,
     "        self._statistic = total / len(rows)",
     "        self._statistic = total"),

    # ---- the rate-matched control ----------------------------------------------------------
    ("the deck carries one move too many, so the control switches faster than the treatment",
     SEL,
     "            self._deck = [True] * self.target_moves + [False] * (",
     "            self._deck = [True] * (self.target_moves + 1) + [False] * ("),
    ("the deck is padded to twice its length, so the realised rate is halved",
     SEL,
     "                self.block - self.target_moves",
     "                self.block"),
    ("the deck is never shuffled, so every seed produces one fixed, periodic schedule",
     SEL,
     "            self._rng.shuffle(self._deck)",
     "            pass"),
    ("the seed is ignored, so replicate control runs cannot differ",
     SEL,
     "        self._rng = random.Random(seed)",
     "        self._rng = random.Random(0)"),
    ("the control may draw the variant that is already active, silently losing a switch",
     SEL,
     "        others = [v for v in variants if v.name != current.name]",
     "        others = list(variants)"),
    ("the destination is pinned, so the control is a second targeted rule rather than none",
     SEL,
     "        target = others[self._rng.randrange(len(others))]",
     "        target = others[0]"),
    ("the control reseeds itself from the batch, so its schedule reads the data after all",
     SEL,
     "        return None\n",
     "        self._rng = random.Random(len(list(rows or [])))\n"),
    ("the empty-deck guard is dropped, so a control with nothing to match is a no-op arm",
     SEL,
     "        if decisions <= 0:",
     "        if False:"),
    ("the rate range guard is dropped, so a control can be asked for more moves than decisions",
     SEL,
     "        if not 0 <= moves <= decisions:",
     "        if False:"),
    ("from_treatment matches the DENOMINATOR, so the control switches on every decision",
     SEL,
     "        return cls(moves=treatment.moves, decisions=treatment.decisions, seed=seed)",
     "        return cls(moves=treatment.decisions, decisions=treatment.decisions, seed=seed)"),
    ("the reported target rate is off by one in its denominator",
     SEL,
     "        return self.target_moves / self.block",
     "        return self.target_moves / (self.block + 1)"),

    # ---- dispatch.py: the refusal seam ------------------------------------------------------
    ("the refusal seam never catches, so a dead-band batch kills the run",
     DISPATCH,
     "        except HarnessSelectionRefused as refusal:",
     "        except KeyboardInterrupt as refusal:"),
    ("the refusal seam widens to every ValueError, swallowing the dispatcher's own guards",
     DISPATCH,
     "        except HarnessSelectionRefused as refusal:",
     "        except ValueError as refusal:"),
    ("a selector refusal reports changed=True, so declining flatters the switch count",
     DISPATCH,
     '                False,\n                f"selector refused: {refusal}",',
     '                True,\n                f"selector refused: {refusal}",'),
    ("a selector refusal is not marked refused, so it vanishes from route/harness_refused",
     DISPATCH,
     '                f"selector refused: {refusal}",\n                refused=True,',
     '                f"selector refused: {refusal}",\n                refused=False,'),
    ("the dispatcher's own refusal is not marked refused",
     DISPATCH,
     '                "nothing to move to",\n                refused=True,',
     '                "nothing to move to",\n                refused=False,'),
    ("refusals counts every record, so an inert batch reports refusals it never made",
     DISPATCH,
     "        return sum(1 for r in self.records if r.refused)",
     "        return sum(1 for r in self.records)"),
    ("the refusal metric is wired to the switch count",
     DISPATCH,
     '            "route/harness_refused": float(self.refusals),',
     '            "route/harness_refused": float(self.switches),'),
    ("every record defaults to refused, so the counter is a constant rather than a measurement",
     DISPATCH,
     "    refused: bool = False",
     "    refused: bool = True"),
]


def run_tests() -> tuple[bool, str]:
    """Run the selector suite against ``REPO``.

    Returns:
        ``(passed, first failing test id)``. The test id is carried so the kill table can say
        WHICH assertion caught a mutant, which is the difference between "the suite went red"
        and "the guard is constrained".
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
         "import selfevo.harness.selectors as s, selfevo.harness.dispatch as d; "
         "print(s.__file__); print(d.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=600,
    )
    lines = [pathlib.Path(x.strip()).resolve() for x in r.stdout.splitlines() if x.strip()]
    want = [REPO / SEL, REPO / DISPATCH]
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
