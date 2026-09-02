#!/usr/bin/env python3
"""Mutation-test ``selfevo/tests/test_credit_discrimination.py`` against a COPY of the repo.

A copy, never the live checkout. A GPU run imports this tree through ``PYTHONPATH`` and the
training supervisor relaunches on process exit, so a mutated module sitting on disk for even a
few seconds could be picked up by a real run -- and this project has already lost a file to a
mutation harness killed by a tool timeout before its restore ran. Three guards, all of which
have to hold: imports are asserted to resolve inside the copy before anything is written, every
target is asserted sha256-identical to the live checkout at the start AND at the end, and each
target is restored and re-hashed after every single mutation.

Usage: ``mutate_credit_discrimination.py <copy-of-repo> <live-repo>``.

The mutations are single-line defects a careless edit could produce in the code this change
adds: the per-prompt baseline in the ledger, the simulated world and its two reported metrics,
and the shuffled control. Metric mutations are included deliberately -- a scale error in
``subset_contrast`` or ``l1_from_uniform`` would not fail any behavioural test, since every
threshold would scale with it, and a measurement instrument nobody can mis-set is one nobody
has checked.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

TESTS = "selfevo/tests/test_credit_discrimination.py"
LEDGER = "selfevo/routing/prompt_credit.py"
SIM = "selfevo/routing/credit_sim.py"
CONTEXTUAL = "selfevo/routing/contextual.py"

# (target, label, find, replace)
MUTATIONS = [
    # ---- the per-prompt baseline ----
    (LEDGER, "first delta credited against a zero baseline, handing it the whole trend",
     "            elif n_deltas > 0:",
     "            elif n_deltas >= 0:"),
    (LEDGER, "baseline is the previous delta, not the mean of the earlier ones",
     "            mean_delta += (delta - mean_delta) / n_deltas",
     "            mean_delta = delta"),
    (LEDGER, "running mean off by one, so every baseline is diluted",
     "            mean_delta += (delta - mean_delta) / n_deltas",
     "            mean_delta += (delta - mean_delta) / (n_deltas + 1)"),
    (LEDGER, "baseline added instead of subtracted, doubling what it should remove",
     "                out = (prior, delta - mean_delta)",
     "                out = (prior, delta + mean_delta)"),
    (LEDGER, "baseline ignored, so self_mean silently means last",
     "                out = (prior, delta - mean_delta)",
     "                out = (prior, delta)"),
    (LEDGER, "withheld pairings not counted, so the baseline's cost is invisible",
     "                self.cold_baseline_skips += 1",
     "                pass"),
    (LEDGER, "baseline test always true, so the new rule never runs",
     '            if self.baseline == "last":',
     "            if True:"),
    (LEDGER, "baseline test always false, so the DEFAULT arm silently self-baselines",
     '            if self.baseline == "last":',
     "            if False:"),
    (LEDGER, "unknown baseline accepted, so a misspelt arm reports as the one asked for",
     '        if self.baseline not in ("last", "self_mean"):',
     "        if False:"),
    (LEDGER, "delta history never carried forward, so every prompt stays cold forever",
     "            n_deltas, mean_delta = prior.n_deltas, prior.mean_delta",
     "            n_deltas, mean_delta = 0, 0.0"),
    # ---- the simulated loop ----
    (SIM, "centring never applied, so the ablation pair is one arm run twice",
     '                centred=credit == "prompt_centered",',
     "                centred=False,"),
    (SIM, "centring applied to every prompt arm, erasing the ablation the other way",
     '                centred=credit == "prompt_centered",',
     "                centred=True,"),
    (SIM, "self baseline never selected, so the new rule silently means the old one",
     '            baseline="self_mean" if credit == "prompt_self_baseline" else "last"',
     '            baseline="last"'),
    (SIM, "the shuffled control silently becomes its own treatment",
     "    if shuffler is not None:",
     "    if False:"),
    (SIM, "a batch-credited shuffle is accepted, so a guaranteed no-op reads as a control",
     '    if shuffle_credit and credit == "batch":',
     "    if False:"),
    (SIM, "the credited value is a constant, so no prompt delta is ever informative",
     "                    (p, ctx.unit_id, m, world.value(p))",
     "                    (p, ctx.unit_id, m, 0.0)"),
    (SIM, "every prompt shares one ledger key, merging the whole pool into one prompt",
     '        closed = ledger.observe_and_record(f"p{prompt}", unit_id, mode, value, step)',
     '        closed = ledger.observe_and_record("p", unit_id, mode, value, step)'),
    (SIM, "the world's gain no longer depends on the mode, so there is nothing to target",
     "        gain = self.gain if mode == self.best_mode(prompt) else 0.0",
     "        gain = self.gain"),
    (SIM, "the marker no longer decides the subset, so one mode is right everywhere",
     "        return self.high_mode if self._marker[prompt] > 0.4 else self.low_mode",
     "        return self.high_mode"),
    # ---- the two reported metrics ----
    (SIM, "uniform reference is a half rather than one over the number of modes",
     "            sum(abs(s - 1.0 / len(self.modes)) for s in self.mode_shares(w).values())",
     "            sum(abs(s - 0.5) for s in self.mode_shares(w).values())"),
    (SIM, "total variation distance not halved, so every contrast reads double",
     "            out.append(0.5 * sum(abs(sh_hi[m] - sh_lo[m]) for m in self.modes))",
     "            out.append(sum(abs(sh_hi[m] - sh_lo[m]) for m in self.modes))"),
    (SIM, "a mode nobody chose drops out of the share sum instead of counting as zero",
     "        return {m: sum(1 for r in rows if r[0] == m) / n for m in self.modes}",
     "        return {m: sum(1 for r in rows if r[0] == m) / n for m in {r[0] for r in rows}}"),
    (SIM, "targeting counts the decisions that were WRONG",
     "                acc[subset] = sum(1 for c in rows if c == subset) / len(rows)",
     "                acc[subset] = sum(1 for c in rows if c != subset) / len(rows)"),
    # ---- the router these tests are ultimately about ----
    (CONTEXTUAL, "observe credits nothing, so no credit rule could ever separate the arms",
     "    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:",
     "    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:\n        return"),
]


def run_tests(repo: pathlib.Path) -> bool:
    """True if the suite passes inside ``repo``, importing ``repo`` and not the live tree."""
    env = dict(os.environ, PYTHONPATH=str(repo))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True, timeout=1800, env=env,
    )
    return r.returncode == 0


def _digest(path: pathlib.Path) -> str:
    """sha256 of a file's text, the identity every guard here is stated in."""
    return hashlib.sha256(path.read_text().encode()).hexdigest()


def _assert_isolated(repo: pathlib.Path) -> None:
    """Refuse to run unless pytest would import the copy's modules, not the live ones."""
    env = dict(os.environ, PYTHONPATH=str(repo))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.routing.credit_sim as m, selfevo.routing.prompt_credit as p, "
         "selfevo.routing.contextual as c; print(m.__file__); print(p.__file__); "
         "print(c.__file__)"],
        cwd=repo, capture_output=True, text=True, env=env, timeout=300,
    )
    got = [pathlib.Path(line).resolve() for line in r.stdout.split()]
    want = [(repo / f).resolve() for f in (SIM, LEDGER, CONTEXTUAL)]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: imports resolve to {got}, not {want}")
    print(f"isolated: imports resolve inside {repo}")


def _assert_matches_live(repo: pathlib.Path, live: pathlib.Path, when: str) -> None:
    """Every mutated file must be byte-identical to the live checkout, start and finish."""
    for rel in sorted({m[0] for m in MUTATIONS} | {TESTS}):
        a, b = _digest(repo / rel), _digest(live / rel)
        if a != b:
            raise SystemExit(f"{when}: {rel} differs between the copy and {live}")
    print(f"{when}: every target matches the live checkout")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <copy-of-repo> <live-repo>")
    repo = pathlib.Path(sys.argv[1]).resolve()
    live = pathlib.Path(sys.argv[2]).resolve()
    if repo == live:
        raise SystemExit("refusing to mutate the live checkout; pass a copy as the first path")
    _assert_isolated(repo)
    _assert_matches_live(repo, live, "before")

    originals = {rel: (repo / rel).read_text() for rel in {m[0] for m in MUTATIONS}}
    digests = {rel: _digest(repo / rel) for rel in originals}

    if not run_tests(repo):
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors, skipped = [], []
    for rel, label, find, repl in MUTATIONS:
        target = repo / rel
        original = originals[rel]
        n = original.count(find)
        if n != 1:
            # Reported as SKIP, never as SURVIVED: a mutation that could not be applied is an
            # untested defect, not a tested one, and counting it either way would be a lie.
            print(f"SKIP      [{rel}] {label}: anchor appears {n}x")
            skipped.append((rel, label, f"anchor appears {n}x"))
            continue
        mutated = original.replace(find, repl, 1)
        if mutated == original:
            print(f"SKIP      [{rel}] {label}: replacement left the file byte-identical")
            skipped.append((rel, label, "no-op replacement"))
            continue
        target.write_text(mutated)
        try:
            compiles = True
            try:
                compile(mutated, str(target), "exec")
            except SyntaxError as exc:
                compiles = False
                print(f"SKIP      [{rel}] {label}: mutant does not compile ({exc.msg})")
                skipped.append((rel, label, "mutant does not compile"))
            passed = run_tests(repo) if compiles else None
        finally:
            target.write_text(original)
            assert _digest(target) == digests[rel], f"restore failed for {rel}"
        if passed is None:
            continue
        if passed:
            print(f"SURVIVED  [{rel}] {label}")
            survivors.append((rel, label))
        else:
            print(f"killed    [{rel}] {label}")

    tried = len(MUTATIONS) - len(skipped)
    print(f"\n{tried - len(survivors)}/{tried} applied mutations killed "
          f"({len(skipped)} skipped, {len(MUTATIONS)} listed)")
    for rel, label in survivors:
        print(f"  SURVIVOR: [{rel}] {label}")
    for rel, label, why in skipped:
        print(f"  SKIPPED:  [{rel}] {label} -- {why}")
    _assert_matches_live(repo, live, "after")
    return 1 if survivors or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
