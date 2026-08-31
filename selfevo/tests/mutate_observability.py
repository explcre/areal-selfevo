#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_observability.py against a COPY of the repo.

A copy, not the live checkout: the training supervisor relaunches on process exit, so a
mutated module sitting on disk for even a few seconds could be imported by a real run.
Every mutation is a single-line defect a careless edit could produce -- a dropped mask, a
sample standard deviation, a comparison flipped at the boundary -- and the file is restored
and its sha256 re-checked after each one.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
TARGET = REPO / "selfevo/observability.py"
TESTS = "selfevo/tests/test_observability.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    ("mean_logprob drops the mask entirely",
     "    tok_lp = (kept * mask).sum(dim=-1)                            # (B,)",
     "    tok_lp = logprobs.to(torch.float32).sum(dim=-1)               # (B,)"),
    ("masking is done by the multiply alone, so nan * 0 == nan leaks the prompt",
     "    kept = logprobs.to(torch.float32).masked_fill(mask == 0, 0.0)",
     "    kept = logprobs.to(torch.float32)"),
    ("per-sample log-prob is a sum, not a per-token mean",
     "        [torch.tensor(_safe_div(float(s), float(n))) for s, n in zip(tok_lp, lengths)]",
     "        [torch.tensor(float(s)) for s, n in zip(tok_lp, lengths)]"),
    ("per-token mean divides by the padded width instead of the response length",
     "        [torch.tensor(_safe_div(float(s), float(n))) for s, n in zip(tok_lp, lengths)]",
     "        [torch.tensor(_safe_div(float(s), float(mask.shape[-1]))) for s, n in zip(tok_lp, lengths)]"),
    ("reward_std becomes a SAMPLE standard deviation",
     "        r_std = _finite(float(r.std(unbiased=False)))",
     "        r_std = _finite(float(r.std(unbiased=True)))"),
    ("length std becomes a SAMPLE standard deviation",
     "        len_std = _finite(float(ln.std(unbiased=False)))",
     "        len_std = _finite(float(ln.std(unbiased=True)))"),
    ("log-prob std becomes a SAMPLE standard deviation",
     "        lp_std = _finite(float(lp.std(unbiased=False)))",
     "        lp_std = _finite(float(lp.std(unbiased=True)))"),
    ("len_dispersion stops being scale-free",
     "                len_dispersion=_safe_div(len_std, mean_len),",
     "                len_dispersion=len_std,"),
    ("logprob_dispersion loses the abs, so it reads backwards on real log-probs",
     "                logprob_dispersion=_safe_div(lp_std, abs(mean_lp)),",
     "                logprob_dispersion=_safe_div(lp_std, mean_lp),"),
    ("solve_rate turns inclusive at the threshold",
     "                solve_rate=float((r > reward_threshold).to(torch.float32).mean()),",
     "                solve_rate=float((r >= reward_threshold).to(torch.float32).mean()),"),
    ("truncation turns exclusive, so a response that hit the wall is not truncated",
     "            float((ln >= max_response_len).to(torch.float32).mean())",
     "            float((ln > max_response_len).to(torch.float32).mean())"),
    ("an absent budget is guessed as fully truncated",
     "            else 0.0\n        )",
     "            else 1.0\n        )"),
    ("the finiteness guard on reward_std is dropped",
     "        r_std = _finite(float(r.std(unbiased=False)))",
     "        r_std = float(r.std(unbiased=False))"),
    ("the finiteness guard on mean_logprob is dropped",
     "        mean_lp = _finite(float(lp.mean()))",
     "        mean_lp = float(lp.mean())"),
    ("the reward guard reads the raw tensor, not the float32 the features come from",
     "    if not bool(torch.isfinite(rewards.to(torch.float32)).all()):",
     "    if not bool(torch.isfinite(rewards).all()):"),
    ("the reward guard fires on ANY value, refusing every batch",
     "    if not bool(torch.isfinite(rewards.to(torch.float32)).all()):",
     "    if bool(torch.isfinite(rewards.to(torch.float32)).all()):"),
    ("solve_rate compares against a fixed 0.5 instead of the configured threshold",
     "                solve_rate=float((r > reward_threshold).to(torch.float32).mean()),",
     "                solve_rate=float((r > 0.5).to(torch.float32).mean()),"),
    ("_safe_div returns the quotient even when the denominator is zero",
     "    if abs(den) < 1e-12:\n        return default",
     "    if False:\n        return default"),
    ("the finiteness guard on mean_response_len is dropped",
     "        mean_len = _finite(float(ln.mean()))",
     "        mean_len = float(ln.mean())"),
    ("_safe_div stops rejecting an infinite quotient",
     '    return out if out == out and abs(out) != float("inf") else default',
     "    return out if out == out else default"),
    ("_safe_div's near-zero denominator guard narrows to exact zero",
     "    if abs(den) < 1e-12:",
     "    if den == 0.0:"),
    ("the partition check accepts a group_sizes that drops rows",
     "    if sum(group_sizes) != b:",
     "    if sum(group_sizes) > b:"),
    ("an empty group is accepted",
     "    if any(g < 1 for g in group_sizes):",
     "    if any(g < 0 for g in group_sizes):"),
    ("the group cursor advances by one row instead of by the group size",
     "        start += g",
     "        start += 1"),
    ("rewards are read from the whole batch instead of the group",
     "        r = rewards[sl].to(torch.float32)",
     "        r = rewards.to(torch.float32)"),
    ("lengths are read from the whole batch instead of the group",
     "        ln = lengths[sl]",
     "        ln = lengths"),
    ("the non-finite reward guard is removed",
     "    if not bool(torch.isfinite(rewards.to(torch.float32)).all()):",
     "    if False:"),
    ("the 2-D shape guard on loss_mask/logprobs is removed",
     "        if t.ndim != 2:",
     "        if t.ndim != 0:"),
    ("the budget guard accepts a zero budget, which reports every batch truncated",
     "    if max_response_len is not None and max_response_len < 1:",
     "    if max_response_len is not None and max_response_len < 0:"),
]


# Mutants that CANNOT be killed because the mutated guard cannot fire: the input
# validation above them already rules out the only inputs that would reach them. Listed so
# an unreachable guard is a recorded fact rather than a hole in the suite.
EXPECTED_SURVIVORS = {
    "the finiteness guard on reward_std is dropped":
        "rewards are refused unless finite in float32, and torch's std is stable, so "
        "r.std() over finite float32 values cannot itself be non-finite",
    "the finiteness guard on mean_logprob is dropped":
        "per-sample log-probs come out of _safe_div, which is total: their mean is finite "
        "by construction",
}


def run_tests() -> bool:
    """True if the suite passes."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=900, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c", "import selfevo.observability as m; print(m.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=300,
    )
    got = pathlib.Path(r.stdout.strip()).resolve()
    if got != TARGET:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {TARGET}")
    print(f"isolated: imports resolve to {got}")


def main() -> int:
    _assert_isolated()
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors, expected = [], []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        passed = run_tests()
        TARGET.write_text(original)
        assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, "restore failed"
        if passed and label in EXPECTED_SURVIVORS:
            print(f"EQUIVALENT {label}")
            expected.append(label)
        elif passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        elif label in EXPECTED_SURVIVORS:
            print(f"killed    {label}  (was listed as equivalent -- delist it)")
        else:
            print(f"killed    {label}")

    killed = len(MUTATIONS) - len(survivors) - len(expected)
    print(f"\n{killed}/{len(MUTATIONS)} killed, {len(expected)} equivalent by construction")
    for label in expected:
        print(f"  = {label}: {EXPECTED_SURVIVORS[label]}")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
