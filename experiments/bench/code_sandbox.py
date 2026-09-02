#!/usr/bin/env python3
"""Execute untrusted, model-written Python under bounded time, memory and privilege.

This is the security-critical half of the code benchmark. Everything else -- loading
problems, building prompts, calling an endpoint -- is plumbing that fails loudly. This
module runs code an unaligned model wrote, and it has to fail loudly *and* safely.

WHAT THIS SANDBOX DOES
----------------------
Three tiers, strongest available chosen automatically (:func:`detect_tier`), and the tier
actually used is recorded on every result so a score can never be read without knowing how
it was produced.

``bwrap``      bubblewrap: a new user, network, IPC, PID, UTS and cgroup namespace; the
               whole filesystem re-bound READ-ONLY with a fresh tmpfs ``/tmp`` and one
               read-write bind for the job's own scratch directory; a new session (no
               controlling terminal, so no ``TIOCSTI`` injection into the caller's tty);
               ``--die-with-parent`` so the sandbox cannot outlive the grader.
``netns``      ``unshare -rn``: a network namespace with loopback only. No filesystem
               isolation -- code can still write anywhere the invoking user can.
``subprocess`` No namespaces at all. Only the limits in the list below.

EVERY tier additionally gets, applied inside the child before any untrusted byte runs
(:func:`_guard_source`):

* ``RLIMIT_AS``    -- address space, so a runaway allocation dies instead of the box.
* ``RLIMIT_CPU``   -- CPU seconds, a backstop under the wall clock for a process that
                      somehow escapes the parent's timer.
* ``RLIMIT_FSIZE`` -- largest file it may write, which also caps stdout/stderr because
                      those are redirected to files rather than pipes.
* ``RLIMIT_NPROC`` -- fork ceiling.
* ``RLIMIT_CORE``  -- 0, no core dumps.
* a scrubbed environment (no ``HF_*``, no credentials, ``CUDA_VISIBLE_DEVICES=""`` so a
  generated program cannot touch a GPU this box is training on),
* a fresh working directory per execution, deleted afterwards,
* ``start_new_session=True`` and a ``killpg`` on timeout, so a child that spawns children
  is killed as a group rather than orphaned,
* stdio redirected to FILES, not pipes: a program that writes 10 GB cannot deadlock the
  grader on a full pipe buffer, and the parent reads back a bounded prefix.

WHAT THIS SANDBOX DOES *NOT* DO -- read this before trusting it
---------------------------------------------------------------
A subprocess with rlimits and namespaces is **not a container and not a VM**. Concretely:

* **No seccomp filter.** Every syscall the kernel offers is reachable. A kernel privilege
  escalation bug is not defended against.
* **Same kernel, same host.** There is no hypervisor boundary.
* **No CPU or I/O cgroup.** ``RLIMIT_CPU`` bounds one process's CPU time, not the machine's
  load; a wide fan-out of graders can still saturate the box.
* **The Python-level network block is defence in depth only.** In the ``bwrap`` and
  ``netns`` tiers the real block is a kernel network namespace with no route out. In the
  ``subprocess`` tier all that stands is a monkeypatch over :mod:`socket`, and code that
  opens a raw file descriptor walks straight past it. Do not run the ``subprocess`` tier on
  a machine you care about; it exists so the harness is testable on a box without user
  namespaces, and it says so in every result it returns.
* **In the ``netns`` tier the filesystem is fully writable** by the invoking user. Only
  ``bwrap`` makes it read-only.
* **``RLIMIT_NPROC`` counts processes per real UID, not per sandbox.** On a shared account
  already running other jobs the ceiling may be exceeded before the child starts, which
  makes *every* fork inside the sandbox fail. That is the safe direction and it is applied
  identically to reference code and to model code, so it cannot bias a comparison -- but a
  solution that legitimately needs :mod:`multiprocessing` will fail, and that failure is a
  benchmark artefact rather than a wrong answer.
* **No wall-clock guarantee under load.** The timeout is enforced by the parent's timer; if
  the box is oversubscribed a correct program can exceed it and be scored a timeout, which
  is a FAIL. Timeouts are therefore reported separately from wrong answers so that
  confound stays visible in the output rather than being folded into the score.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TIER_BWRAP = "bwrap"
TIER_NETNS = "netns"
TIER_SUBPROCESS = "subprocess"
TIERS = (TIER_BWRAP, TIER_NETNS, TIER_SUBPROCESS)

# Exit status the bootstrap uses when it fails BEFORE reaching the untrusted program. It
# must not be confusable with a submission that merely crashed, or a broken grader would be
# reported as a model that writes broken code.
BOOT_FAILURE_RC = 97

# Environment variable overriding the auto-detected tier, mostly so the tests can exercise
# the weak tier on a box that has bubblewrap.
TIER_ENV = "LCB_SANDBOX"

# Files the sandbox owns. A caller-supplied file may not take one of these names.
RESERVED_NAMES = frozenset({"main.py", "_guard.py", "_boot.py",
                            "stdin.txt", "stdout.txt", "stderr.txt"})


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings for one execution.

    Attributes:
        wall_seconds: Hard wall-clock limit. The parent kills the process group at this
            point regardless of what the child is doing.
        memory_bytes: ``RLIMIT_AS``. 2 GiB by default, far above any competitive
            programming solution and far below this box's memory.
        output_bytes: Most stdout/stderr the PARENT will read back, and the most of any
            read-back file. The child may write more (up to ``file_bytes``); the excess is
            discarded and FLAGGED, because a silently shortened output compares unequal and
            would be scored a wrong answer. 16 MiB is chosen from measurement: the largest
            expected output in livecodebench v6 is 3.28 MiB, and 14 cases across 4 problems
            exceed 1 MiB. At 1 MiB those four problems failed the replay oracle.
        file_bytes: ``RLIMIT_FSIZE``. Because stdio is redirected to files this also bounds
            how much a print loop can put on disk.
        max_processes: ``RLIMIT_NPROC``.
        cpu_seconds: ``RLIMIT_CPU``. ``None`` derives it from ``wall_seconds``.
    """

    wall_seconds: float = 10.0
    memory_bytes: int = 2 * 1024 ** 3
    output_bytes: int = 16 * 1024 ** 2
    file_bytes: int = 64 * 1024 ** 2
    max_processes: int = 64
    cpu_seconds: int | None = None

    def cpu(self) -> int:
        """CPU-second ceiling, derived from the wall clock when not set explicitly.

        Returns:
            Whole seconds, at least 1 and deliberately ABOVE the wall clock so the CPU
            limit is a backstop rather than the primary timeout. If it fired first, a
            slow-but-correct program would be misreported as a resource kill and the
            timeout count would stop meaning what it says.
        """
        if self.cpu_seconds is not None:
            return max(1, int(self.cpu_seconds))
        return max(1, int(self.wall_seconds) + 2)


@dataclass
class SandboxResult:
    """Outcome of one execution.

    Attributes:
        status: ``ok`` (exited 0), ``timeout`` (wall clock or ``RLIMIT_CPU``), ``error``
            (non-zero exit, including an uncaught exception), ``output_limit`` (killed by
            ``SIGXFSZ``), ``harness_error`` (the sandbox itself failed; NEVER a statement
            about the submission).
        returncode: Child exit status, negative for a terminating signal.
        stdout: Bounded prefix of the child's stdout.
        stderr: Bounded prefix of the child's stderr.
        files: Read-back contents of files named in ``read_back``; a name maps to ``None``
            when the child never created it.
        files_truncated: Per read-back file, whether it hit ``output_bytes``. A caller must
            treat a truncated value as UNKNOWN rather than as a mismatch.
        elapsed: Wall seconds the child ran.
        tier: Which tier actually ran it.
        killed_by_parent: True when the wall-clock timer fired, which is what separates
            "we killed it" from "it died on its own".
        detail: Free text for anything a reader would otherwise have to guess at.
    """

    status: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    files: dict = field(default_factory=dict)
    files_truncated: dict = field(default_factory=dict)
    elapsed: float = 0.0
    tier: str = TIER_SUBPROCESS
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    killed_by_parent: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        """Plain-dict form, for the artifact file."""
        return asdict(self)


def _have(cmd: str) -> bool:
    """Whether an executable is on PATH."""
    return shutil.which(cmd) is not None


def _probe(argv) -> bool:
    """Whether ``argv`` runs a trivial command successfully.

    Presence on PATH is not capability: ``unshare -n`` exists on every box and is refused
    without privilege, and bubblewrap fails where unprivileged user namespaces are off. So
    the tier is chosen by RUNNING something, not by looking for a binary.

    Args:
        argv: Command to try, ending in a trivial no-op program.

    Returns:
        True when it exited zero.
    """
    try:
        r = subprocess.run(argv, capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


_TIER_CACHE = {}


def detect_tier(force: str = "") -> str:
    """Strongest isolation tier this machine can actually provide.

    Args:
        force: Tier name to use instead of probing; also read from ``$LCB_SANDBOX``.

    Returns:
        One of :data:`TIERS`.

    Raises:
        ValueError: If an unknown tier is forced. A typo in ``$LCB_SANDBOX`` must not
            silently downgrade the sandbox to the weakest tier, which is exactly the kind
            of quiet degradation that makes a security claim false without any error.
    """
    forced = force or os.environ.get(TIER_ENV) or ""
    if forced:
        if forced not in TIERS:
            raise ValueError(f"unknown sandbox tier {forced!r}; expected one of {TIERS}")
        return forced
    if "auto" in _TIER_CACHE:
        return _TIER_CACHE["auto"]
    tier = TIER_SUBPROCESS
    if _have("bwrap") and _probe(["bwrap", "--unshare-all", "--ro-bind", "/", "/",
                                  "--", "/bin/true"]):
        tier = TIER_BWRAP
    elif _have("unshare") and _probe(["unshare", "-rn", "--", "/bin/true"]):
        tier = TIER_NETNS
    _TIER_CACHE["auto"] = tier
    return tier


def describe_tier(tier: str) -> str:
    """One-line honest statement of what a tier does and does not contain.

    Args:
        tier: One of :data:`TIERS`.

    Returns:
        Human-readable description, carried into every results row so a score is never
        read without its isolation level.
    """
    return {
        TIER_BWRAP: (
            "bubblewrap: new user/net/ipc/pid/uts namespaces, read-only root, rw scratch "
            "dir only, no network route. NOT a VM and NOT seccomp-filtered."
        ),
        TIER_NETNS: (
            "unshare -rn: network namespace only (no route out). Filesystem is NOT "
            "isolated -- the child can write anywhere this user can."
        ),
        TIER_SUBPROCESS: (
            "plain subprocess: rlimits, scrubbed env and a fresh cwd only. The network "
            "block is a Python monkeypatch and is bypassable. Weakest tier."
        ),
    }[tier]


_GUARD_TEMPLATE = '''"""Applied inside the child before any untrusted byte is compiled."""
import resource, socket, sys


def install():
    """Lower this process's resource ceilings and deny the network."""
    lim = [
        (resource.RLIMIT_AS, %d),
        (resource.RLIMIT_CPU, %d),
        (resource.RLIMIT_FSIZE, %d),
        (resource.RLIMIT_NPROC, %d),
        (resource.RLIMIT_CORE, 0),
    ]
    for which, value in lim:
        try:
            soft, hard = resource.getrlimit(which)
            cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (cap, cap))
        except (ValueError, OSError):
            # A ceiling we cannot lower is reported through the tier string, not by
            # pretending it was applied. Never abort here: turning a machine restriction
            # into a crash would score every submission zero.
            pass

    def _denied(*a, **k):
        raise OSError(101, "network access is disabled in the grading sandbox")

    # Defence in depth. In the bwrap/netns tiers the kernel has already removed the route;
    # in the subprocess tier this is ALL there is, and it is bypassable.
    for name in ("socket", "create_connection", "create_server", "socketpair"):
        try:
            setattr(socket, name, _denied)
        except Exception:
            pass
    sys.setrecursionlimit(20000)
'''


def _guard_source(limits: SandboxLimits) -> str:
    """Source of the in-child guard module, with this run's ceilings baked in.

    Written as a separate file rather than prepended to the submission so that line numbers
    in the submission's traceback stay correct, and so that a submission opening with a
    ``__future__`` import or a shebang is not broken by a prefix.

    Args:
        limits: The ceilings to bake in.

    Returns:
        Python source for ``_guard.py``.
    """
    return _GUARD_TEMPLATE % (int(limits.memory_bytes), int(limits.cpu()),
                              int(limits.file_bytes), int(limits.max_processes))


_BOOT = '''"""Bootstrap: install limits, then run the submission as __main__."""
import sys, traceback

try:
    import _guard
    _guard.install()
except BaseException:
    traceback.print_exc()
    sys.exit(97)

import runpy

try:
    sys.argv = ["main.py"]
    runpy.run_path("main.py", run_name="__main__")
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    sys.exit(1)
'''


def child_env() -> dict:
    """Environment for the child: the minimum needed to start Python, nothing else.

    Inheriting the grader's environment would hand untrusted code every credential, cache
    path and accelerator handle this process holds. ``CUDA_VISIBLE_DEVICES=""`` is set
    explicitly rather than merely omitted, because an empty value hides the GPUs while an
    absent variable exposes all of them -- and on this box those GPUs are running a
    training job that must not be disturbed.

    Returns:
        The complete environment dict passed to :class:`subprocess.Popen`.
    """
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }


def _argv(tier: str, python: str, workdir: str) -> list:
    """Command line that launches the bootstrap under the requested tier.

    Args:
        tier: One of :data:`TIERS`.
        python: Interpreter path.
        workdir: Scratch directory, which is the only writable path under ``bwrap``.

    Returns:
        The argv list.
    """
    inner = [python, "_boot.py"]
    if tier == TIER_BWRAP:
        return [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", workdir, workdir,
            "--chdir", workdir,
            "--",
        ] + inner
    if tier == TIER_NETNS:
        return ["unshare", "-rn", "--"] + inner
    return inner


def _read_capped(path: Path, cap: int):
    """Read at most ``cap`` bytes of a file.

    Args:
        path: File to read.
        cap: Byte ceiling.

    Returns:
        ``(text, truncated)``. A program that prints without bound must not be able to make
        the grader allocate without bound, so the parent never reads the whole file.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(cap + 1)
    except FileNotFoundError:
        return "", False
    truncated = len(data) > cap
    return data[:cap].decode("utf-8", "replace"), truncated


def _kill_group(proc) -> None:
    """SIGKILL the child's whole process group.

    SIGKILL rather than SIGTERM because untrusted code can install a SIGTERM handler and
    decline to die; the group rather than the pid because a submission that forked would
    otherwise leave its children running after the grader has already reported a timeout.

    Args:
        proc: The :class:`subprocess.Popen` started with ``start_new_session=True``.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass


def run_python(source, stdin_data=b"", limits=None, extra_files=None,
               read_back=(), tier=None, python=None) -> SandboxResult:
    """Run one Python program under the sandbox and return what it did.

    Args:
        source: The complete program. Written verbatim to ``main.py``; nothing is
            prepended, so tracebacks carry the submission's own line numbers.
        stdin_data: Bytes fed on stdin, through a FILE rather than a pipe.
        limits: Ceilings; :class:`SandboxLimits` defaults when omitted.
        extra_files: Additional files placed in the scratch directory before the run, for
            example a JSON blob of call arguments. Names must be plain file names.
        read_back: Files to read out of the scratch directory afterwards; each appears in
            :attr:`SandboxResult.files`, mapping to ``None`` if never created.
        tier: Force an isolation tier; defaults to :func:`detect_tier`.
        python: Interpreter to run. Defaults to the one running the grader, which is what
            makes the venv's libraries available to submissions.

    Returns:
        A :class:`SandboxResult`. This function does not raise for anything the submission
        does -- a crash, a hang and an unwritten result file are ordinary outcomes that must
        be *scored*, not propagated. ``status="harness_error"`` means the sandbox itself
        could not run, which is a different thing and is reported as such.

    Raises:
        ValueError: If ``extra_files`` names a path rather than a plain filename, or would
            overwrite the bootstrap. Letting a caller replace ``_guard.py`` would disable
            every limit here while still reporting the tier that was requested.
    """
    limits = limits or SandboxLimits()
    tier = tier or detect_tier()
    python = python or sys.executable
    base = os.environ.get("LCB_SANDBOX_TMP") or tempfile.gettempdir()
    workdir = tempfile.mkdtemp(prefix="lcbsbx-", dir=base)
    try:
        w = Path(workdir)
        for name in (extra_files or {}):
            if os.path.basename(name) != name or name in RESERVED_NAMES:
                raise ValueError(
                    f"extra_files name {name!r} must be a plain filename and must not be "
                    f"one of the sandbox's own files {sorted(RESERVED_NAMES)}"
                )
        (w / "main.py").write_text(source)
        (w / "_guard.py").write_text(_guard_source(limits))
        (w / "_boot.py").write_text(_BOOT)
        for name, blob in (extra_files or {}).items():
            mode, payload = ("wb", blob) if isinstance(blob, bytes) else ("w", blob)
            with open(w / name, mode) as fh:
                fh.write(payload)
        (w / "stdin.txt").write_bytes(stdin_data)

        argv = _argv(tier, python, workdir)
        t0 = time.time()
        killed = False
        rc = None
        detail = ""
        try:
            with open(w / "stdin.txt", "rb") as fin, \
                 open(w / "stdout.txt", "wb") as fout, \
                 open(w / "stderr.txt", "wb") as ferr:
                proc = subprocess.Popen(
                    argv, cwd=workdir, stdin=fin, stdout=fout, stderr=ferr,
                    env=child_env(),
                    # A new session means the child and everything it spawns share a
                    # process group we can kill wholesale. Without it a hung grandchild
                    # survives the timeout and holds the box.
                    start_new_session=True,
                )
        except Exception as exc:
            return SandboxResult(
                status="harness_error", returncode=None, stdout="", stderr="",
                elapsed=time.time() - t0, tier=tier,
                detail=f"failed to launch sandbox ({tier}): {exc!r}",
            )
        try:
            rc = proc.wait(timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            killed = True
            _kill_group(proc)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Unkillable (uninterruptible sleep). Reported, never silently ignored: a
                # leaked process is exactly how a grader starts lying later on.
                detail = "process survived SIGKILL; possible leak"
                rc = None
        elapsed = time.time() - t0

        out, out_trunc = _read_capped(w / "stdout.txt", limits.output_bytes)
        err, err_trunc = _read_capped(w / "stderr.txt", limits.output_bytes)
        files, files_truncated = {}, {}
        for name in read_back:
            p = w / name
            if p.exists():
                files[name], files_truncated[name] = _read_capped(p, limits.output_bytes)
            else:
                files[name], files_truncated[name] = None, False

        if killed or rc == -signal.SIGXCPU:
            status = "timeout"
        elif rc == -signal.SIGXFSZ:
            status = "output_limit"
        elif rc == BOOT_FAILURE_RC:
            status = "harness_error"
            detail = detail or "sandbox bootstrap failed before running the submission"
        elif rc == 0:
            status = "ok"
        else:
            status = "error"
        return SandboxResult(
            status=status, returncode=rc, stdout=out, stderr=err, files=files,
            files_truncated=files_truncated,
            elapsed=elapsed, tier=tier, stdout_truncated=out_trunc,
            stderr_truncated=err_trunc, killed_by_parent=killed, detail=detail,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def selftest(tier=None) -> dict:
    """Assert the sandbox's own guarantees on THIS machine, and report what held.

    A sandbox is a claim about a machine, not about a source file: the same code gives a
    kernel network namespace on one box and a bypassable monkeypatch on another. So the
    claim is measured where it runs rather than asserted from the tier name.

    Args:
        tier: Tier to measure; defaults to the auto-detected one.

    Returns:
        A dict of check name to result, including the tier and its honest description.
    """
    tier = tier or detect_tier()
    lim = SandboxLimits(wall_seconds=6.0, memory_bytes=512 * 1024 ** 2)
    checks = {"tier": tier, "describes": describe_tier(tier)}

    r = run_python("print('hello')", limits=lim, tier=tier)
    checks["runs_ok"] = r.status == "ok" and r.stdout.strip() == "hello"

    r = run_python("import sys; sys.stdout.write(sys.stdin.read().upper())",
                   stdin_data=b"abc", limits=lim, tier=tier)
    checks["stdin_wired"] = r.stdout == "ABC"

    r = run_python("while True: pass", limits=SandboxLimits(wall_seconds=2.0), tier=tier)
    checks["timeout_killed"] = r.status == "timeout"

    r = run_python("x = bytearray(4 * 1024 ** 3)", limits=lim, tier=tier)
    checks["memory_bounded"] = r.status in ("error", "timeout")

    r = run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\n",
                   limits=lim, tier=tier)
    checks["network_blocked"] = r.status == "error"

    r = run_python("raise ValueError('boom')", limits=lim, tier=tier)
    checks["exception_is_error"] = r.status == "error" and "ValueError" in r.stderr

    return checks


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2))
