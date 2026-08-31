"""Vet a generated policy in a subprocess under hard resource limits, before a run uses it.

The AST allowlist in :mod:`selfevo.routing.code_policy` cannot bound cost. An adversarial
audit found two families it provably cannot close without breaking legitimate policies:

    def route(features): return "rl" if 9 ** 9 ** 9 else "skip"       # one opcode, no loop
    def route(features): return "rl" if len("%999999999d" % 1) else "skip"   # ~1 GB

Neither uses a loop, so rejecting loops does not help; both need only ``Pow``/``Mult``/``Mod``,
which ordinary policies use. Worse, an in-process timeout cannot save you: a signal handler
runs *between* bytecodes and ``9 ** 9 ** 9`` is a single one.

So cost is bounded where it can be -- in a child process with ``RLIMIT_CPU`` and
``RLIMIT_AS``, which the kernel enforces regardless of what the code does. A policy is vetted
once, before a run adopts it; the accepted policy then executes in-process at full speed, so
this costs one subprocess per candidate rather than one per routing decision.

This is a *cost* boundary. It does not make executing generated code safe against an
adversary, and :mod:`selfevo.routing.code_policy` says so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

__all__ = ["VettingResult", "vet_policy", "DEFAULT_PROBES"]

# Probe inputs spanning the regimes a policy must survive, including the degenerate ones a
# generated rule is most likely to divide by.
DEFAULT_PROBES: tuple[dict[str, float], ...] = (
    {"solve_rate": 0.0, "reward_std": 0.0, "mean_response_len": 0.0, "len_dispersion": 0.0,
     "mean_logprob": 0.0, "logprob_dispersion": 0.0, "truncated_fraction": 0.0,
     "group_size": 8.0, "has_target": 0.0},
    {"solve_rate": 1.0, "reward_std": 0.0, "mean_response_len": 512.0, "len_dispersion": 0.1,
     "mean_logprob": -0.4, "logprob_dispersion": 0.2, "truncated_fraction": 0.0,
     "group_size": 8.0, "has_target": 1.0},
    {"solve_rate": 0.5, "reward_std": 0.5, "mean_response_len": 1024.0, "len_dispersion": 0.9,
     "mean_logprob": -2.0, "logprob_dispersion": 1.5, "truncated_fraction": 1.0,
     "group_size": 8.0, "has_target": 1.0},
)

# The child MEASURES its own cost rather than only surviving a limit. A hard limit alone
# answers "did it crash", which is the wrong question: a policy allocating just under the
# cap passes vetting and then does it again on every routing decision.
_RUNNER = r'''
import json, resource, sys
cpu, mem = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
if mem > 0:
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
payload = json.load(sys.stdin)
sys.path.insert(0, payload["repo"])
from selfevo.routing.code_policy import compile_policy
fn = compile_policy(payload["source"])
# Baseline AFTER imports, so the interpreter and this package are not charged to the policy.
base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
base_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
out = []
for probe in payload["probes"]:
    out.append(repr(fn(dict(probe))))
u = resource.getrusage(resource.RUSAGE_SELF)
json.dump({
    "returns": out,
    "rss_delta_kb": max(u.ru_maxrss - base_rss, 0),
    "cpu_seconds": max(u.ru_utime - base_cpu, 0.0),
}, sys.stdout)
'''


@dataclass(frozen=True)
class VettingResult:
    """Outcome of vetting one policy.

    Args:
        ok: Whether the policy validated, ran on every probe, and stayed within budget.
        detail: Human-readable reason. On failure this is what a policy author reads.
        returns: What the policy returned for each probe, as reprs. Empty when it failed.
        rss_delta_bytes: Peak memory the policy itself added, measured in the child after
            imports. Reported even on success, because a policy that is merely expensive is
            worth seeing before it runs once per group per batch.
        cpu_seconds: CPU the probes consumed, measured the same way.
    """

    ok: bool
    detail: str
    returns: tuple[str, ...] = ()
    rss_delta_bytes: int = 0
    cpu_seconds: float = 0.0


def vet_policy(
    source: str,
    *,
    repo_root: str,
    probes: tuple[dict[str, float], ...] = DEFAULT_PROBES,
    cpu_seconds: int = 5,
    address_space_bytes: int = 512 * 1024 ** 2,
    max_alloc_bytes: int = 64 * 1024 ** 2,
    wall_timeout: float = 30.0,
) -> VettingResult:
    """Run a candidate policy under hard limits and report whether it is usable.

    Args:
        source: Candidate policy source.
        repo_root: Directory to put on the child's ``sys.path`` so it can import
            ``selfevo.routing.code_policy`` -- the child re-validates rather than trusting
            the parent, so a policy cannot be vetted under one allowlist and run under
            another.
        probes: Feature dicts to evaluate. Defaults span the degenerate regimes.
        cpu_seconds: ``RLIMIT_CPU``. Kernel-enforced, so it stops a single runaway opcode.
        address_space_bytes: ``RLIMIT_AS``; 0 disables. The backstop for an allocation the
            measurement below could not survive to report.
        max_alloc_bytes: Reject a policy whose own peak allocation exceeds this, even though
            it completed. A decision rule needs kilobytes; anything near this is a bomb that
            happened to fit under the hard limit, and it would pay that cost on every group
            of every batch.
        wall_timeout: Backstop for a child that blocks without burning CPU.

    Returns:
        A :class:`VettingResult`. ``ok=False`` is a normal outcome for a generated policy and
        never raises -- a candidate that fails vetting should be discarded and counted, not
        crash the search that produced it.
    """
    payload = json.dumps(
        {"source": source, "probes": [dict(p) for p in probes], "repo": repo_root}
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(cpu_seconds), str(address_space_bytes)],
            input=payload, capture_output=True, text=True, timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        return VettingResult(False, f"policy exceeded the {wall_timeout}s wall clock")

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        last = err[-1] if err else f"exit code {proc.returncode}"
        # A negative return code is a signal: SIGKILL/SIGXCPU is what a resource limit looks
        # like from here, and it is the expected outcome for a runaway policy.
        if proc.returncode < 0:
            return VettingResult(
                False,
                f"policy was killed by signal {-proc.returncode} -- it exceeded "
                f"RLIMIT_CPU={cpu_seconds}s or RLIMIT_AS={address_space_bytes} bytes",
            )
        return VettingResult(False, f"policy failed to run: {last}")

    try:
        got = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return VettingResult(False, f"child produced no result: {proc.stdout[:200]!r}")

    rss = int(got.get("rss_delta_kb", 0)) * 1024
    cpu = float(got.get("cpu_seconds", 0.0))
    returns = tuple(got.get("returns", ()))
    if max_alloc_bytes and rss > max_alloc_bytes:
        return VettingResult(
            False,
            f"policy allocated {rss / 1024 ** 2:.0f} MiB on {len(probes)} probes, over the "
            f"{max_alloc_bytes / 1024 ** 2:.0f} MiB budget. It completed, but this cost "
            f"would be paid for every group of every batch.",
            returns, rss, cpu,
        )
    return VettingResult(True, "ok", returns, rss, cpu)
