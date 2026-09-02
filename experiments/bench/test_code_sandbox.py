#!/usr/bin/env python3
"""Tests for the sandbox that executes model-written code.

These assert the sandbox's GUARANTEES, not its implementation: a program that hangs is
killed, one that allocates without bound dies, one that reaches for the network fails, one
that crashes is reported as a crash rather than as silence, and none of them can take the
grader down or make it wait forever. The point of writing them this way is that the same
assertions hold whichever isolation tier the machine can offer, so they keep constraining
the sandbox on a box where bubblewrap is unavailable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import code_sandbox  # noqa: E402
from code_sandbox import (  # noqa: E402
    SandboxLimits,
    describe_tier,
    detect_tier,
    run_python,
)

FAST = SandboxLimits(wall_seconds=8.0, memory_bytes=512 * 1024 ** 2)


def usable_tiers():
    """Tiers this machine can actually run, measured rather than assumed."""
    out = []
    for tier in code_sandbox.TIERS:
        try:
            if run_python("print(1)", tier=tier, limits=FAST).status == "ok":
                out.append(tier)
        except Exception:
            pass
    return out


TIERS = usable_tiers()


def test_at_least_one_tier_works():
    """If nothing runs, every score from this harness would be a silent zero."""
    assert TIERS, "no sandbox tier runs on this machine"


def test_detect_tier_prefers_the_strongest_available():
    assert detect_tier() == next(t for t in code_sandbox.TIERS if t in TIERS)


def test_unknown_forced_tier_raises():
    """A typo must not quietly downgrade isolation to the weakest tier."""
    with pytest.raises(ValueError):
        detect_tier("bubblewrap")


def test_describe_tier_covers_every_tier():
    for tier in code_sandbox.TIERS:
        assert describe_tier(tier)


@pytest.mark.parametrize("tier", TIERS)
def test_stdout_and_exit_zero(tier):
    r = run_python("print('hi')", tier=tier, limits=FAST)
    assert (r.status, r.returncode, r.stdout.strip()) == ("ok", 0, "hi")
    assert r.tier == tier


@pytest.mark.parametrize("tier", TIERS)
def test_stdin_is_delivered_byte_exactly(tier):
    """Whitespace and non-ASCII must survive: a mangled input is a wrong answer nobody
    can explain afterwards."""
    payload = "5 7\né\ntrailing  \n"
    r = run_python("import sys; sys.stdout.write(repr(sys.stdin.read()))",
                   stdin_data=payload.encode(), tier=tier, limits=FAST)
    assert r.stdout == repr(payload)


@pytest.mark.parametrize("tier", TIERS)
def test_infinite_loop_is_killed_and_reported_as_timeout(tier):
    t0 = time.time()
    r = run_python("while True:\n    pass\n",
                   tier=tier, limits=SandboxLimits(wall_seconds=2.0))
    assert r.status == "timeout"
    assert r.killed_by_parent
    # A test that hangs the grader is a benchmark that silently reports nothing.
    assert time.time() - t0 < 20


@pytest.mark.parametrize("tier", TIERS)
def test_sigterm_ignoring_program_still_dies(tier):
    """The kill must be SIGKILL: untrusted code can decline SIGTERM and often does."""
    src = ("import signal, time\n"
           "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
           "while True:\n    time.sleep(0.05)\n")
    t0 = time.time()
    r = run_python(src, tier=tier, limits=SandboxLimits(wall_seconds=2.0))
    assert r.status == "timeout"
    # Empty detail means it actually died. A signal it could ignore would leave the
    # "survived SIGKILL; possible leak" note, and a leaked process is how a grader starts
    # lying about the box it is running on.
    assert r.detail == ""
    assert time.time() - t0 < 20


@pytest.mark.parametrize("tier", TIERS)
def test_memory_ceiling_stops_a_runaway_allocation(tier):
    r = run_python("x = bytearray(4 * 1024 ** 3)\n", tier=tier, limits=FAST)
    assert r.status in ("error", "timeout")
    assert r.status != "ok"


@pytest.mark.parametrize("tier", TIERS)
def test_network_is_denied(tier):
    r = run_python("import socket\n"
                   "socket.create_connection(('1.1.1.1', 80), timeout=3)\n",
                   tier=tier, limits=FAST)
    assert r.status == "error"


@pytest.mark.parametrize("tier", TIERS)
def test_uncaught_exception_is_an_error_with_a_traceback(tier):
    r = run_python("raise ValueError('boom')\n", tier=tier, limits=FAST)
    assert r.status == "error"
    assert "ValueError" in r.stderr and "boom" in r.stderr


@pytest.mark.parametrize("tier", TIERS)
def test_syntax_error_is_an_error_not_a_crash_of_the_grader(tier):
    r = run_python("def (:\n    pass\n", tier=tier, limits=FAST)
    assert r.status == "error"


@pytest.mark.parametrize("tier", TIERS)
def test_nonzero_exit_is_reported(tier):
    r = run_python("import sys; sys.exit(3)\n", tier=tier, limits=FAST)
    assert (r.status, r.returncode) == ("error", 3)


@pytest.mark.parametrize("tier", TIERS)
def test_program_runs_as_main(tier):
    """Competitive-programming submissions routinely guard on __name__."""
    r = run_python("if __name__ == '__main__':\n    print('main')\n",
                   tier=tier, limits=FAST)
    assert r.stdout.strip() == "main"


@pytest.mark.parametrize("tier", TIERS)
def test_extra_files_are_readable_and_results_are_read_back(tier):
    r = run_python("import json\n"
                   "d = json.load(open('in.json'))\n"
                   "open('out.json', 'w').write(json.dumps(d['a'] + 1))\n",
                   extra_files={"in.json": '{"a": 41}'}, read_back=("out.json",),
                   tier=tier, limits=FAST)
    assert r.status == "ok"
    assert r.files["out.json"] == "42"


@pytest.mark.parametrize("tier", TIERS)
def test_missing_read_back_file_is_none_not_an_exception(tier):
    """A solution that never returns is an ordinary FAIL to be scored, not a crash."""
    r = run_python("pass\n", read_back=("out.json",), tier=tier, limits=FAST)
    assert r.status == "ok" and r.files["out.json"] is None


def test_extra_files_cannot_replace_the_guard():
    """Overwriting _guard.py would silently disable every resource limit while the run
    still reported the tier it asked for."""
    for name in ("_guard.py", "main.py", "_boot.py", "stdin.txt"):
        with pytest.raises(ValueError):
            run_python("pass", extra_files={name: "x"}, limits=FAST)


def test_extra_files_cannot_escape_the_working_directory():
    with pytest.raises(ValueError):
        run_python("pass", extra_files={"../evil.py": "x"}, limits=FAST)


@pytest.mark.parametrize("tier", TIERS)
def test_environment_is_scrubbed(tier):
    """Untrusted code must not inherit this process's credentials, caches or GPUs."""
    os.environ["LCB_TEST_SECRET"] = "leaked"
    try:
        r = run_python("import os, json\n"
                       "print(json.dumps({'secret': os.environ.get('LCB_TEST_SECRET'),\n"
                       "                  'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
                       "                  'n': len(os.environ)}))\n",
                       tier=tier, limits=FAST)
    finally:
        del os.environ["LCB_TEST_SECRET"]
    import json as _json
    got = _json.loads(r.stdout)
    assert got["secret"] is None
    assert got["cuda"] == ""
    assert got["n"] <= 12


@pytest.mark.parametrize("tier", TIERS)
def test_output_is_capped_and_the_truncation_is_flagged(tier):
    """A print loop must not make the grader allocate without bound, and the reader must
    be told the text was cut rather than being handed a plausible short answer."""
    r = run_python("print('x' * 200000)\n", tier=tier,
                   limits=SandboxLimits(wall_seconds=8.0, output_bytes=1000))
    assert r.status == "ok"
    assert len(r.stdout) == 1000 and r.stdout_truncated


@pytest.mark.parametrize("tier", TIERS)
def test_each_run_gets_a_fresh_working_directory(tier):
    """State must not leak between test cases: a solution that caches to disk would
    otherwise be graded on a previous test's answer."""
    first = run_python("open('marker', 'w').write('1')\nprint('ok')\n",
                       tier=tier, limits=FAST)
    second = run_python("import os; print(os.path.exists('marker'))\n",
                        tier=tier, limits=FAST)
    assert first.status == "ok"
    assert second.stdout.strip() == "False"


def test_working_directories_are_cleaned_up():
    import tempfile
    base = Path(os.environ.get("LCB_SANDBOX_TMP") or tempfile.gettempdir())
    before = set(base.glob("lcbsbx-*"))
    run_python("print(1)", limits=FAST)
    assert set(base.glob("lcbsbx-*")) <= before


def test_selftest_reports_every_guarantee():
    checks = code_sandbox.selftest()
    for key in ("runs_ok", "stdin_wired", "timeout_killed", "memory_bounded",
                "network_blocked", "exception_is_error"):
        assert checks[key] is True, f"{key} failed on tier {checks['tier']}"


def test_cpu_backstop_sits_above_the_wall_clock():
    """If RLIMIT_CPU fired first, a slow-but-correct program would be misreported as a
    resource kill and the timeout count would stop meaning what it says."""
    assert SandboxLimits(wall_seconds=10.0).cpu() > 10
    assert SandboxLimits(wall_seconds=10.0, cpu_seconds=3).cpu() == 3
