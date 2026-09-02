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
        label="a missing logprobs block is `no difference` instead of refused (chat path)",
        rel="selfevo/periodic_eval.py",
        find=(
            "    lps = [t.get(\"logprob\") for t in content if isinstance(t, dict) and t.get(\"logprob\") is not None]\n"
            "    if not lps:\n"
        ),
        repl=(
            "    lps = [t.get(\"logprob\") for t in content if isinstance(t, dict) and t.get(\"logprob\") is not None]\n"
            "    if not lps:\n"
            "        return text, [0.0]\n"
            "    if False:\n"
        ),
    ),
    Mutation(
        label="a missing logprobs block is `no difference` instead of refused (token-id path)",
        rel="selfevo/periodic_eval.py",
        find=(
            "    lps = r.get(\"logprobs\") or []\n"
            "    if not lps:\n"
        ),
        repl=(
            "    lps = r.get(\"logprobs\") or []\n"
            "    if not lps:\n"
            "        return r.get(\"text\", \"\"), [0.0]\n"
            "    if False:\n"
        ),
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
        label="THE BRIEF'S MUTATION: an explicitly configured endpoint is overridden by discovery",
        rel="selfevo/periodic_eval.py",
        find='    if configured:\n        return ResolvedEndpoint(_v1(configured), "configured")',
        repl='    if False:\n        return ResolvedEndpoint(_v1(configured), "configured")',
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: discovery falls back to a localhost guess",
        rel="selfevo/periodic_eval.py",
        find="    raise EndpointUndiscoverable(\n        \"no inference server address could be discovered, so there is nothing to evaluate \"",
        repl=(
            '    return ResolvedEndpoint("http://127.0.0.1:30000/v1", "process_cmdline")\n'
            "    raise EndpointUndiscoverable(\n"
            "        \"no inference server address could be discovered, so there is nothing to evaluate \""
        ),
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: a discovery failure returns an address nothing checked",
        rel="selfevo/periodic_eval.py",
        find='    infos = getattr(rollout, "server_infos", None)\n    if not infos:\n        return ""',
        repl='    infos = getattr(rollout, "server_infos", None)\n    if not infos:\n        return "172.28.127.18:32735"',
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the recorded source is always the CONFIGURED one",
        rel="selfevo/periodic_eval.py",
        find='    endpoint = ResolvedEndpoint(cfg.base_url, cfg.base_url_source or "unresolved")',
        repl='    endpoint = ResolvedEndpoint(cfg.base_url, "configured")',
    ),
    Mutation(
        label="the recorded port is a constant rather than the port that was queried",
        rel="selfevo/periodic_eval.py",
        find="    port = None if endpoint is None else endpoint_port(endpoint.base_url)",
        repl="    port = 30000",
    ),
    Mutation(
        label="THE ORIGINAL DEFECT: the controller's own server_infos is not read",
        rel="selfevo/periodic_eval.py",
        find='    infos = getattr(rollout, "server_infos", None)\n',
        repl="    infos = None\n",
    ),
    Mutation(
        label="THE 2026-09-02 MISDIAGNOSIS: worker RPC ports are read as serving addresses",
        rel="selfevo/periodic_eval.py",
        find="    return names.gen_servers(experiment, trial)",
        repl='    return names.worker_discovery(experiment, trial, "rollout", 0)',
    ),
    Mutation(
        label="the process scan ignores trial ownership and can find another run's server",
        rel="selfevo/periodic_eval.py",
        find='        if not host or not port or not _descends_from(pid, owners, by_pid):',
        repl="        if not host or not port:",
    ),
    Mutation(
        label="a command line with no port acquires a default one",
        rel="selfevo/periodic_eval.py",
        find='        host, port = _flag(argv, "--host"), _flag(argv, "--port")',
        repl='        host, port = _flag(argv, "--host"), _flag(argv, "--port") or "30000"',
    ),
    Mutation(
        label="ownership is decided on the serving process alone, not on its ancestry",
        rel="selfevo/periodic_eval.py",
        find="        seen.add(pid)\n        pid = by_pid[pid][0]",
        repl="        seen.add(pid)\n        return pid in owners",
    ),
    Mutation(
        label="two servers for one trial: one is picked instead of refusing",
        rel="selfevo/periodic_eval.py",
        find="    if len(found) > 1:",
        repl="    if False:",
    ),
    Mutation(
        label="the trial-identity gate is removed, so an unscoped scan of the box happens",
        rel="selfevo/periodic_eval.py",
        find="    if not experiment or not trial:",
        repl="    if False:",
    ),
    Mutation(
        label="an undiscoverable endpoint reuses the generic `the server did not answer` code",
        rel="selfevo/periodic_eval.py",
        find='        status = STATUS["endpoint_undiscovered"]',
        repl='        status = STATUS["endpoint_error"]',
    ),
    Mutation(
        label="the discovered address is not normalised to a /v1 endpoint",
        rel="selfevo/periodic_eval.py",
        find='    return addr if addr.endswith("/v1") else addr + "/v1"',
        repl="    return addr",
    ),
    Mutation(
        label="a failed evaluation forgets the adapter it was pinned to (the step-50 wart)",
        rel="selfevo/periodic_eval.py",
        find="    if resolved is None:\n        resolved = pin.get(\"adapter\")",
        repl="    if False:\n        resolved = pin.get(\"adapter\")",
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
