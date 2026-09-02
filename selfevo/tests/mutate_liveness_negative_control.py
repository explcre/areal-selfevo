"""Mutation test for the adapter-liveness controls. A test that catches nothing is not a test.

Breaks :mod:`selfevo.periodic_eval`'s liveness path one distinct way at a time and asserts
that ``test_liveness_negative_control.py`` FAILS for each. The mutations are the ways this
guard has actually gone wrong, or could: comparing a distribution against itself, defaulting
a verdict when there was nothing to compare, routing on the CONFIGURED adapter name rather
than the one actually served, deciding on a maximum that one position can carry, dropping the
in-band control, and going back to generating.

It never touches the checked-out file. Every mutation is applied to a private copy of the
package in a scratch directory, which is what the tests are then run against.

Usage:
    python selfevo/tests/mutate_liveness_negative_control.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = "selfevo/periodic_eval.py"
TESTS = "selfevo/tests/test_liveness_negative_control.py"

#: The test that is red on purpose: it pins the epsilon against a floor measured on A0's own
#: server and stays red until that one-line constant is raised. Excluded so a mutation is
#: judged by the tests it BREAKS, not by one that was already failing.
KNOWN_RED = "test_the_default_epsilon_clears_the_measured_same_weights_noise_floor"

#: ``name -> (old, new)``. Each must be a unique substring of the target file.
MUTATIONS = {
    "compare_a_distribution_against_itself": (
        '_paired_abs_diff(a[x], b[y], f"adapter {model!r} against base")',
        '_paired_abs_diff(a[x], a[y], f"adapter {model!r} against base")',
    ),
    "default_the_verdict_when_nothing_was_compared": (
        '    if n_tok == 0:\n        raise LivenessUnavailable(',
        '    if n_tok == 0:\n        return LivenessReport(\n'
        '            n_probes=len(cfg.probe_prompts), prompts_live_frac=1.0,\n'
        '            max_abs_dlogprob=0.0, mean_abs_dlogprob=0.0,\n'
        '            noise_max_abs_dlogprob=0.0, noise_mean_abs_dlogprob=0.0,\n'
        '            is_live=1, n_tokens_compared=0)\n    if False:\n        raise LivenessUnavailable(',
    ),
    "route_on_the_configured_name_not_the_served_one": (
        'lora = "" if model == cfg.base_model else mb.adapter_route(model, caps)',
        'lora = "" if cfg.model == cfg.base_model else mb.adapter_route(cfg.model, caps)',
    ),
    "decide_on_the_maximum_a_single_position_can_carry": (
        'is_live=int(mean_d > noise_mean + cfg.live_eps),',
        'is_live=int(cross_max[k_signal] > noise_mean + cfg.live_eps),',
    ),
    "drop_the_in_band_negative_control": (
        'is_live=int(mean_d > noise_mean + cfg.live_eps),',
        'is_live=int(mean_d > cfg.live_eps),',
    ),
    "report_live_unconditionally": (
        'is_live=int(mean_d > noise_mean + cfg.live_eps),',
        'is_live=1,',
    ),
    "report_the_noise_as_zero_without_measuring_it": (
        '        for k, d in enumerate(within):\n            within_sum[k] += sum(d)\n'
        '            within_max[k] = max(within_max[k], max(d))',
        '        for k, d in enumerate(within):\n            within_sum[k] += 0.0\n'
        '            within_max[k] = 0.0',
    ),
    "go_back_to_generating": (
        '"max_new_tokens": 0}',
        '"max_new_tokens": 32}',
    ),
}


def run(root: Path) -> tuple[int, str]:
    """Run the liveness controls against one copy of the package.

    Args:
        root: Directory holding the ``selfevo`` package to test.

    Returns:
        ``(n_failed, summary_text)``.
    """
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(root / TESTS), "-q", "--no-header",
         "-p", "no:cacheprovider", "-k", f"not {KNOWN_RED}"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=600,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()
    return (0 if " failed" not in (r.stdout + r.stderr) else 1), "\n".join(tail[-3:])


def stage(tmp: Path) -> Path:
    """Copy the package into a scratch tree the mutations can be applied to.

    Args:
        tmp: Scratch directory.

    Returns:
        The root of the copy.
    """
    root = tmp / "tree"
    (root / "selfevo").mkdir(parents=True)
    shutil.copytree(REPO / "selfevo", root / "selfevo", dirs_exist_ok=True)
    os.symlink(REPO / "experiments", root / "experiments")
    return root


def main() -> int:
    """Apply every mutation in turn and report which the tests caught.

    Returns:
        Process exit status: 0 when every mutation was caught.
    """
    with tempfile.TemporaryDirectory(dir=os.environ.get("MUT_TMP") or None) as td:
        root = stage(Path(td))
        pristine = (root / TARGET).read_text()

        n_failed, tail = run(root)
        if n_failed:
            print(f"BASELINE IS RED -- fix that first:\n{tail}")
            return 2
        print("baseline: green\n")

        caught, missed = [], []
        for name, (old, new) in MUTATIONS.items():
            if pristine.count(old) != 1:
                print(f"  SKIP  {name}: anchor matched {pristine.count(old)} times")
                missed.append(name + " (anchor drifted)")
                continue
            (root / TARGET).write_text(pristine.replace(old, new))
            failed, tail = run(root)
            (root / TARGET).write_text(pristine)
            if failed:
                caught.append(name)
                print(f"  caught  {name}\n            {tail.splitlines()[-1]}")
            else:
                missed.append(name)
                print(f"  MISSED  {name}  <-- the tests do not constrain this")

        print(f"\n{len(caught)}/{len(MUTATIONS)} mutations caught")
        if missed:
            print("NOT CONSTRAINED: " + ", ".join(missed))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
