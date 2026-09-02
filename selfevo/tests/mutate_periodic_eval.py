"""Mutation test for the periodic evaluation: break it on purpose, one way at a time.

Run against a COPY of the tree, never the live checkout:

    rsync -a --exclude .git ~/areal-selfevo/ /tmp/mut_pe/
    MATH_EVAL_DATA=~/evaldata python3 selfevo/tests/mutate_periodic_eval.py /tmp/mut_pe

The refusal below is not decoration. Six first-generation harnesses in this tree write to
``~/areal-selfevo`` directly, and ``mutate_harness_selectors.py`` records why that is now
forbidden: the training run imports this tree across worker processes that relaunch, so a
mutated source file sitting on disk for even a few seconds can be imported by a live run.

REPORTING. Three columns, not two. A mutation whose anchor was not unique, whose replacement
left the bytes unchanged, or whose result does not compile has not been TESTED, and scoring
it as a kill reports a number higher than the truth. ``SKIP`` is its own outcome and makes
the run fail, exactly as a survivor does.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
from dataclasses import dataclass

TESTS = ["selfevo/tests/test_periodic_eval.py"]

LIVE = pathlib.Path("/home/ubuntu/areal-selfevo").resolve()


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect.

    Keyword-only by construction. Seven different tuple unpack shapes exist across the 34
    harnesses in this tree and two of them differ only by swapping the first two fields, so a
    row copied between harnesses silently looks for a source file named after the defect
    description.

    Attributes:
        label: What the defect is, in the words a reader needs.
        rel: Source file, relative to the repo root.
        find: Exact text to replace; must appear exactly once.
        repl: Its replacement.
    """

    label: str
    rel: str
    find: str
    repl: str


MUTATIONS = [
    Mutation(
        label="THE BRIEF'S FIRST MUTATION: the periodic eval reads the `report` half",
        rel="selfevo/periodic_eval.py",
        find='EVAL_SPLIT = "search"',
        repl='EVAL_SPLIT = "report"',
    ),
    Mutation(
        label="the post-condition split guard never refuses anything",
        rel="selfevo/periodic_eval.py",
        find="    search, report = require_committed_split(bench)\n    seen = ",
        repl="    search, report = require_committed_split(bench)\n    return 0\n    seen = ",
    ),
    Mutation(
        label="an unsplit benchmark falls through to the WHOLE set, which contains `report`",
        rel="selfevo/periodic_eval.py",
        find="    sf = split_path(bench)\n    if not sf.exists():",
        repl="    sf = split_path(bench)\n    if False:",
    ),
    Mutation(
        label="THE BRIEF'S SECOND MUTATION: scoring zero problems is reported as success",
        rel="selfevo/periodic_eval.py",
        find='    graded = int(row.get("n_graded") or 0)',
        repl='    return\n    graded = int(row.get("n_graded") or 0)',
    ),
    Mutation(
        label="a run that graded a small minority is accepted as a comparable point",
        rel="selfevo/periodic_eval.py",
        find="    if graded * 2 < requested:",
        repl="    if False:",
    ),
    Mutation(
        label="THE BRIEF'S THIRD MUTATION: the liveness metric always says LIVE",
        rel="selfevo/periodic_eval.py",
        find="        is_live=int(max_d > cfg.live_eps),",
        repl="        is_live=1,",
    ),
    Mutation(
        label="the liveness verdict is decided on greedy TEXT again (the step-149 false alarm)",
        rel="selfevo/periodic_eval.py",
        find="        is_live=int(max_d > cfg.live_eps),",
        repl="        is_live=int(differ > 0),",
    ),
    Mutation(
        label="a missing logprobs block is treated as `no difference` instead of refused",
        rel="selfevo/periodic_eval.py",
        find="    if not lps:\n        raise LivenessUnavailable(",
        repl="    if not lps:\n        return text, [0.0]\n    if False:\n        raise LivenessUnavailable(",
    ),
    Mutation(
        label="THE BRIEF'S FOURTH MUTATION: best-val keeps the LATEST checkpoint",
        rel="selfevo/periodic_eval.py",
        find="        is_best = score > self.best_score",
        repl="        is_best = True",
    ),
    Mutation(
        label="a tie walks `best` forward, making selection a no-op on a plateau",
        rel="selfevo/periodic_eval.py",
        find="        is_best = score > self.best_score",
        repl="        is_best = score >= self.best_score",
    ),
    Mutation(
        label="patience is off by one and stops the run a step early",
        rel="selfevo/periodic_eval.py",
        find="            should_stop=self.n_since_best >= self.patience,",
        repl="            should_stop=self.n_since_best >= self.patience - 1,",
    ),
    Mutation(
        label="best-val state is never persisted, so a resumed run forgets the best checkpoint",
        rel="selfevo/periodic_eval.py",
        find="        if self.state_path is not None:\n            self.state_path.parent.mkdir",
        repl="        if False:\n            self.state_path.parent.mkdir",
    ),
    Mutation(
        label="an inert adapter is diagnosed from accuracy anyway, hiding `not learning yet`",
        rel="selfevo/periodic_eval.py",
        find="    if not liveness.is_live:\n        return DIAGNOSIS[\"adapter_inert\"]",
        repl="    if False:\n        return DIAGNOSIS[\"adapter_inert\"]",
    ),
    Mutation(
        label="`could not measure` is folded into `adapter is dead`",
        rel="selfevo/periodic_eval.py",
        find='    if liveness is None:\n        return DIAGNOSIS["unknown"]',
        repl='    if liveness is None:\n        return DIAGNOSIS["adapter_inert"]',
    ),
    Mutation(
        label="an unknown throughput cost is reported as free rather than as unknown",
        rel="selfevo/periodic_eval.py",
        find="        return float(\"nan\")\n    return eval_seconds / (freq_steps * step_seconds)",
        repl="        return 0.0\n    return eval_seconds / (freq_steps * step_seconds)",
    ),
    Mutation(
        label="the feature runs at every step regardless of the configured cadence",
        rel="selfevo/periodic_eval.py",
        find="        return bool(self.enabled) and global_step > 0 and global_step % self.freq_steps == 0",
        repl="        return True",
    ),
    Mutation(
        label="a non-primary rank evaluates too, which deadlocks the barrier",
        rel="selfevo/periodic_eval.py",
        find="        if not is_primary:\n            return {}",
        repl="        if False:\n            return {}",
    ),
    Mutation(
        label="a failed evaluation still updates best-val, so a crash can become `best`",
        rel="selfevo/periodic_eval.py",
        find='    if status == STATUS["ok"]:\n        wilson =',
        repl="    if True:\n        wilson =",
    ),
    Mutation(
        label="a missing model id is accepted, so the BASE model is scored in silence",
        rel="selfevo/periodic_eval.py",
        find="        if not model:\n            raise ValueError(",
        repl="        if False:\n            raise ValueError(",
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the OLDEST version in the window is resolved, not the newest",
        rel="selfevo/periodic_eval.py",
        find="        chosen = max(candidates, key=lambda t: t[0])[1]",
        repl="        chosen = min(candidates, key=lambda t: t[0])[1]",
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: a resolution failure returns the BASE model id",
        rel="selfevo/periodic_eval.py",
        find=(
            "    else:\n"
            "        raise AdapterUnresolved(\n"
            '            f"{sorted(served)} contains no version of adapter {name!r}. Refusing to "\n'
        ),
        repl=(
            "    else:\n"
            "        return base_model\n"
            "    if False:\n"
            "        raise AdapterUnresolved(\n"
            '            f"{sorted(served)} contains no version of adapter {name!r}. Refusing to "\n'
        ),
    ),
    Mutation(
        label="the explicit base-model guard never fires, so the base id can be resolved",
        rel="selfevo/periodic_eval.py",
        find="    if base_model and chosen == base_model:",
        repl="    if False:",
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the resolved id is CACHED across evaluations",
        rel="selfevo/periodic_eval.py",
        find="        pinned = await resolve_adapter(session, cfg)",
        repl=(
            "        pinned = getattr(resolve_adapter, '_cache', None) or await resolve_adapter(session, cfg)\n"
            "        resolve_adapter._cache = pinned"
        ),
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the recorded version is the CONFIGURED id, not the resolved one",
        rel="selfevo/periodic_eval.py",
        find="        model=model, version=split_adapter_version(model)[1], n_served=len(served)",
        repl="        model=cfg.model, version=split_adapter_version(cfg.model)[1], n_served=len(served)",
    ),
    Mutation(
        label="the mid-evaluation eviction check is removed, so a point can span two versions",
        rel="selfevo/periodic_eval.py",
        find="        await assert_still_served(session, cfg, pinned)",
        repl="        pass",
    ),
    Mutation(
        label="the harness' SystemExit escapes into the training loop and kills the run",
        rel="selfevo/periodic_eval.py",
        find="    except SystemExit as exc:",
        repl="    except KeyboardInterrupt as exc:",
    ),
    Mutation(
        label="the adapter family is matched by PREFIX, so another arm's adapter can win",
        rel="selfevo/periodic_eval.py",
        find="        if n == name and v is not None:",
        repl="        if mid.startswith(name) and v is not None:",
    ),
    Mutation(
        label="the endpoint is guessed at localhost instead of read off the trainer's engine",
        rel="selfevo/periodic_eval.py",
        find=(
            '    addrs = getattr(rollout, "addresses", None) or getattr(\n'
            '        getattr(rollout, "_engine", None), "addresses", None\n'
            "    )"
        ),
        repl='    addrs = ["127.0.0.1:30000"]',
    ),
    Mutation(
        label="an explicitly configured endpoint is overridden by the engine's address",
        rel="selfevo/periodic_eval.py",
        find="        if self.config.base_url or self._rollout is None:",
        repl="        if self._rollout is None:",
    ),
    Mutation(
        label="declared keys are not filled in, leaving invisible gaps in the W&B series",
        rel="selfevo/periodic_eval.py",
        find="    for key in cfg.metric_keys():\n        metrics.setdefault(key, float(\"nan\"))",
        repl="    for key in []:\n        metrics.setdefault(key, float(\"nan\"))",
    ),
]


def run_tests(repo: pathlib.Path) -> bool:
    """Run the pinned tests inside one tree.

    Args:
        repo: The tree to run in.

    Returns:
        True when every test passed.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    return r.returncode == 0


def main() -> int:
    """Apply every mutation in turn and report killed / survived / skipped.

    Returns:
        0 when every mutation was killed, 1 otherwise.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    repo = pathlib.Path(sys.argv[1]).resolve()
    if repo == LIVE:
        print(
            f"REFUSING: {repo} is the live checkout. A training run imports this tree across "
            f"worker processes that relaunch, so a mutated file on disk can be imported by "
            f"it. Copy the tree first and point this at the copy."
        )
        return 2

    originals = {}
    digests = {}
    for m in MUTATIONS:
        p = repo / m.rel
        if m.rel not in originals:
            originals[m.rel] = p.read_text()
            digests[m.rel] = hashlib.sha256(originals[m.rel].encode()).hexdigest()

    if not run_tests(repo):
        print("BASELINE IS RED -- fix the tree before reading any mutation result")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    killed, survived, skipped = [], [], []
    for m in MUTATIONS:
        p = repo / m.rel
        original = originals[m.rel]
        n = original.count(m.find)
        if n != 1:
            print(f"SKIP      {m.label}  (anchor appears {n}x)")
            skipped.append(m.label)
            continue
        mutated = original.replace(m.find, m.repl, 1)
        if mutated == original:
            print(f"SKIP      {m.label}  (replacement changed no bytes)")
            skipped.append(m.label)
            continue
        try:
            compile(mutated, str(p), "exec")
        except SyntaxError as exc:
            print(f"SKIP      {m.label}  (mutant does not compile: {exc})")
            skipped.append(m.label)
            continue
        p.write_text(mutated)
        try:
            still_green = run_tests(repo)
        finally:
            p.write_text(original)
            assert hashlib.sha256(p.read_text().encode()).hexdigest() == digests[m.rel], (
                f"failed to restore {m.rel}"
            )
        if still_green:
            print(f"SURVIVED  {m.label}")
            survived.append(m.label)
        else:
            print(f"killed    {m.label}")
            killed.append(m.label)

    print(
        f"\nkilled {len(killed)}  survived {len(survived)}  skipped {len(skipped)}"
        f"  of {len(MUTATIONS)}"
    )
    for x in survived:
        print(f"  SURVIVOR: {x}")
    for x in skipped:
        print(f"  SKIPPED:  {x}")
    return 1 if (survived or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
