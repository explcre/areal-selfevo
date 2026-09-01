#!/usr/bin/env python3
"""Ground-truth GPU box status, derived from processes that actually hold the GPUs.

The rule this tool exists to enforce: **status is derived from the processes holding
the GPUs, never from a log file found by guessing.** A log directory with no live
owning process is history, and is reported as DEAD -- it can never be mistaken for
the current state of the box, which is the failure this tool was written after.

Verdicts, one per job and one per idle GPU:

  LIVE         a process holds the GPU and its log advanced within --stall-s
  STALLED      a process holds the GPU and its log has NOT advanced -- the real failure
  SERVER-ONLY  an inference server holds the GPU but no client is driving it (wasted)
  PROGRESS-UNKNOWN  a process holds the GPU but the tool cannot show that it advanced
  IDLE         the GPU holds nothing
  DEAD         a run log exists but no live process owns it; shown only under History

Exit status is 0 only when every GPU is held by a job verified LIVE, and 1 when
anything is IDLE, STALLED, SERVER-ONLY or PROGRESS-UNKNOWN, so a supervisor can act
on it. PROGRESS-UNKNOWN counts as degraded on purpose: "I cannot tell whether this
job is advancing" is not the same claim as "this job is advancing", and reporting
the first as the second is the whole class of error this tool is against.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

# A process whose cmdline matches one of these is an inference SERVER: it holds GPU
# memory whether or not any client is driving it, so a server with no client is a
# distinct (and silently wasteful) state rather than a healthy one.
SERVER_PATTERNS = ("sglang::", "sglang.launch_server", "vllm.entrypoints", "trtllm")
# A process whose cmdline matches one of these is a CLIENT: it consumes a server.
CLIENT_PATTERNS = ("math_bench.py", "run_math.sh", "lcb_bench", "harbor ", "swebench")
# A client pattern found beyond this many argv entries, or inside an entry this long,
# is a MENTION of a client rather than an invocation of one. MEASURED on this box: the
# run supervisor's `bash -c "... git add -A experiments/bench/math_bench.py ..."` puts
# 1788 characters in a single argv entry, and matching it counted as a live client --
# which silences SERVER-ONLY for the whole box exactly when every real client has
# finished and the servers are burning GPUs for nothing.
CLIENT_ARGV_ENTRIES = 4
CLIENT_ARGV_MAXLEN = 120
# Lines that indicate a job advanced. Kept deliberately broad; a job whose log never
# matches any of these is reported as PROGRESS-UNKNOWN rather than silently healthy.
#
# MEASURED 2026-09-01 over the last 400 lines of all 131 logs under ~/runs/math: the
# first two alternatives alone left 25 logs with no match, and every one of the 20
# sglang `server.log` files was among them -- i.e. the pattern did not recognise the
# ONE format the GPU-holding processes on this box actually write, so every live job
# reported PROGRESS-UNKNOWN. The `#field: N` and `throughput` alternatives cover
# sglang's "Decode batch, #running-req: 11, #token: 214206, ... gen throughput
# (token/s): 927.69" and cut that to 5 logs, all of them runs that genuinely died
# before emitting any progress.
PROGRESS_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:step|Epoch|epoch|iter|global_step|solved|acc)"
    r"[ =:]*\d|\d+\s*/\s*\d+|#[A-Za-z][\w-]*:\s*\d|throughput[^\d\n]{0,24}\d"
)


def sh(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout, or '' if it fails; never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class Job:
    """One process that holds GPU memory, plus everything derived about it."""

    pid: int
    gpus: list[int] = field(default_factory=list)
    mem_mib: int = 0
    age_s: int = 0
    cmdline: str = ""
    root_pid: int | None = None
    root_cmd: str = ""
    log: str | None = None
    log_age_s: int | None = None
    last_progress: str | None = None
    kind: str = "compute"   # compute | server | client
    verdict: str = "LIVE"
    note: str = ""


def mtime_of(path: str) -> float | None:
    """Modification time of `path`, or None if it cannot be read.

    A log can be rotated or deleted between being listed and being stat'ed, and an
    unguarded ``getmtime`` in a sort key crashes the whole status report over one
    vanished file. Returning None makes the caller decide, which is always "treat
    this file as having no known age" rather than "the box has no status".
    """
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def gpu_index_by_uuid() -> dict[str, int]:
    """Map each GPU's UUID to its index, so compute-apps rows can be placed."""
    out = {}
    for line in sh(["nvidia-smi", "--query-gpu=index,uuid",
                    "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            out[parts[1]] = int(parts[0])
    return out


def all_gpu_indices() -> list[int]:
    """Every GPU index the driver reports, including ones holding nothing."""
    idx = []
    for line in sh(["nvidia-smi", "--query-gpu=index",
                    "--format=csv,noheader"]).splitlines():
        line = line.strip()
        if line.isdigit():
            idx.append(int(line))
    return idx


def compute_apps() -> list[tuple[int, int, str]]:
    """(pid, used_mib, gpu_uuid) for every process currently holding GPU memory."""
    rows = []
    out = sh(["nvidia-smi",
              "--query-compute-apps=pid,used_memory,gpu_uuid",
              "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                mem = int(float(parts[1]))
            except ValueError:
                # The driver prints '[N/A]' for used_memory under MIG and in a few
                # container setups. Dropping the whole row there lost the fact that
                # the GPU is held at all, and the GPU was then reported IDLE while a
                # process was running on it. Keep the row; lose only the number.
                mem = 0
            rows.append((int(parts[0]), mem, parts[2]))
    return rows


def proc_field(pid: int, name: str) -> str:
    """Read one /proc/<pid> text file, or '' if the process is gone."""
    try:
        with open(f"/proc/{pid}/{name}", "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def argv_of(pid: int) -> list[str]:
    """The process's argv as SEPARATE entries, or [] if the process is gone.

    Keeping the entries apart is what distinguishes running a client from naming
    one: `bash -c '<a thousand characters of script>'` holds the whole script in a
    single argv entry, so a client name inside it is a mention, not an invocation.
    """
    raw = proc_field(pid, "cmdline")
    return [a for a in raw.split("\x00") if a] if raw else []


def cmdline_of(pid: int) -> str:
    """The process's argv joined by spaces; falls back to its comm name."""
    raw = proc_field(pid, "cmdline")
    if raw:
        return " ".join(raw.split("\x00")).strip()
    return proc_field(pid, "comm").strip()


def ppid_of(pid: int) -> int | None:
    """Parent PID from /proc/<pid>/status, or None if unavailable."""
    for line in proc_field(pid, "status").splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def age_of(pid: int) -> int:
    """Seconds since the process started, from ps; 0 if it cannot be read."""
    out = sh(["ps", "-o", "etimes=", "-p", str(pid)]).strip()
    return int(out) if out.isdigit() else 0


def ancestors(pid: int, limit: int = 12) -> list[int]:
    """The chain of PIDs from `pid` up towards init, excluding pid 1."""
    chain, cur = [], pid
    for _ in range(limit):
        par = ppid_of(cur)
        if par is None or par <= 1:
            break
        chain.append(par)
        cur = par
    return chain


def open_logs(pid: int) -> list[str]:
    """Paths of .log/.out files the process has open, newest-written first.

    Looking at the process's OWN open descriptors is what ties a job to its log
    without guessing at a directory -- the guess is what produced a false status
    report before this tool existed.
    """
    found = []
    d = f"/proc/{pid}/fd"
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for n in names:
        try:
            tgt = os.readlink(os.path.join(d, n))
        except OSError:
            continue
        if tgt.startswith("/") and (tgt.endswith(".log") or tgt.endswith(".out")):
            if os.path.isfile(tgt):
                found.append(tgt)
    # A file listed a moment ago can be gone by the time it is stat'ed; -1.0 sorts it
    # last instead of raising and taking the entire report down with it.
    return sorted(set(found), key=lambda p: -(mtime_of(p) or -1.0))


def live_pids() -> list[int]:
    """Every PID present in /proc right now."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    return [int(n) for n in names if n.isdigit()]


def open_log_owners(pids: list[int] | None = None) -> dict[str, int]:
    """Map each .log/.out file a LIVE process holds open to that process's PID.

    A run's log is usually held by the ``tee`` in its pipeline, and its client may
    not touch a GPU at all -- ``math_bench.py`` is an HTTP client. Deciding
    deadness from the GPU jobs alone therefore printed a RUNNING benchmark's log
    under History as DEAD (measured on this box: ``f30b_olymp32k_v2/math.log`` was
    held open by live pid 1551620 and still reported dead). That is the mirror of
    the error this tool exists to prevent, and just as misleading.

    Args:
        pids: PIDs to inspect; defaults to every live PID. Injectable so the
            deadness rule is testable without a process table.

    Returns:
        realpath -> the PID holding it open. Costs ~0.13s over 2649 PIDs.
    """
    owners: dict[str, int] = {}
    for pid in (live_pids() if pids is None else pids):
        for path in open_logs(pid):
            owners.setdefault(os.path.realpath(path), pid)
    return owners


def find_log(job: Job) -> str | None:
    """The log for a job: its own open descriptors first, then its ancestors'."""
    for cand in open_logs(job.pid):
        return cand
    for anc in ancestors(job.pid):
        for cand in open_logs(anc):
            return cand
    return None


def tail_progress(path: str, lines: int = 400) -> str | None:
    """The last line in the log that looks like progress, if any."""
    out = sh(["tail", "-n", str(lines), path])
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s and PROGRESS_RE.search(s):
            return s[:150]
    return None


def classify(cmd: str) -> str:
    """Label a process as server, client, or plain compute from its cmdline."""
    if any(p in cmd for p in SERVER_PATTERNS):
        return "server"
    if any(p in cmd for p in CLIENT_PATTERNS):
        return "client"
    return "compute"


def looks_like_client(argv: list[str]) -> bool:
    """True if `argv` INVOKES a client, rather than merely mentioning one.

    Only the leading argv entries are considered, and only short ones -- see
    :data:`CLIENT_ARGV_ENTRIES` and :data:`CLIENT_ARGV_MAXLEN` for the command that
    forced this. A pattern that ends in a space, such as ``"harbor "``, names an
    EXECUTABLE, so it is only allowed to match the program slot: argv[0], or the
    module right after ``-m``. Matching it anywhere would make ``grep harbor`` and
    ``python -c "import harbor"`` count as a running client.
    """
    for i, entry in enumerate(argv[:CLIENT_ARGV_ENTRIES]):
        if len(entry) > CLIENT_ARGV_MAXLEN:
            continue
        is_program = i == 0 or argv[i - 1] == "-m"
        hay = entry + " " if is_program else entry
        if any(p in hay for p in CLIENT_PATTERNS):
            return True
    return False


def find_client(jobs) -> str | None:
    """The command line of a client driving the servers, or None if there is none.

    A client need not hold GPU memory -- ``math_bench.py`` is an HTTP client -- so
    the GPU jobs alone cannot answer this and the process table has to be consulted.
    This tool and its own ancestors are excluded, because running the status check
    from a shell that names a client would otherwise report one, and a single false
    client silences the SERVER-ONLY alarm for the whole box.

    Args:
        jobs: The GPU-holding jobs, checked first so a client on a GPU is found
            without walking /proc.

    Returns:
        The matching command line, or None when nothing is driving the servers.
    """
    for j in jobs:
        if j.kind == "client":
            return j.cmdline
    mine = {os.getpid(), *ancestors(os.getpid())}
    for pid in sorted(live_pids()):
        if pid in mine:
            continue
        argv = argv_of(pid)
        if looks_like_client(argv):
            return " ".join(argv)
    return None


def collect(stall_s: int) -> tuple[list[Job], list[int], str | None]:
    """Build one Job per GPU-holding PID, the idle GPU indices, and the client.

    Args:
        stall_s: A job whose log has not been written in this long is STALLED.

    Returns:
        (jobs sorted by lowest GPU index, sorted idle GPU indices, the command line
        of the client driving the servers or None).
    """
    uuid_idx = gpu_index_by_uuid()
    every = all_gpu_indices()
    jobs: dict[int, Job] = {}
    busy: set[int] = set()

    for pid, mem, uuid in compute_apps():
        gpu = uuid_idx.get(uuid)
        if gpu is not None:
            busy.add(gpu)
        j = jobs.get(pid)
        if j is None:
            j = Job(pid=pid, cmdline=cmdline_of(pid), age_s=age_of(pid))
            j.kind = classify(j.cmdline)
            anc = ancestors(pid)
            j.root_pid = anc[-1] if anc else None
            j.root_cmd = cmdline_of(j.root_pid) if j.root_pid else ""
            j.log = find_log(j)
            if j.log:
                mt = mtime_of(j.log)
                j.log_age_s = int(time.time() - mt) if mt is not None else None
                j.last_progress = tail_progress(j.log)
            jobs[pid] = j
        if gpu is not None and gpu not in j.gpus:
            j.gpus.append(gpu)
        j.mem_mib += mem

    # A server with no client anywhere on the box is holding GPUs for nothing.
    client = find_client(jobs.values())

    for j in jobs.values():
        # nvidia-smi returns compute-apps rows in no guaranteed order, so a
        # multi-GPU PID's list has to be sorted before it is printed or used as the
        # sort key, or the same job moves around the report between invocations.
        j.gpus.sort()
        if j.kind == "server" and client is None:
            j.verdict, j.note = "SERVER-ONLY", "server holds GPUs but no client is driving it"
        elif j.log_age_s is not None and j.log_age_s > stall_s:
            # Staleness is decided by mtime, which is ground truth, BEFORE the
            # progress pattern, which is only a display nicety. The other order let
            # a stalled job whose log format the pattern does not recognise be
            # reported PROGRESS-UNKNOWN -- and PROGRESS-UNKNOWN used to exit 0.
            j.verdict, j.note = "STALLED", f"log has not advanced in {j.log_age_s}s"
        elif j.log is None:
            j.verdict, j.note = "PROGRESS-UNKNOWN", "no log on this process's descriptors"
        elif j.last_progress is None:
            j.verdict, j.note = "PROGRESS-UNKNOWN", "log has no line matching a progress pattern"
        else:
            j.verdict = "LIVE"

    idle = sorted(g for g in every if g not in busy)
    return sorted(jobs.values(), key=lambda x: (x.gpus or [99])[0]), idle, client


def dead_history(roots: list[str], jobs: list[Job], hours: int = 48,
                 held: dict[str, int] | None = None) -> list[tuple[str, int]]:
    """Recently-written run logs that NO live process owns.

    These are reported separately and labelled DEAD. Presenting one of these as the
    box's current state is precisely the error this tool prevents. The converse
    matters just as much: a log a live process still holds open is NOT history, so
    `held` is subtracted too, and paths are compared as realpaths because a
    descriptor's target is fully resolved while a `find` result carries whatever
    symlinks the caller's --log-roots went through.

    Args:
        roots: Directories to scan.
        jobs: The GPU jobs; their own logs are current status, not history.
        hours: How far back a log counts as recent.
        held: realpath -> PID for logs a live process holds open, from
            :func:`open_log_owners`. Defaults to scanning /proc.

    Returns:
        Up to 8 (path, seconds since written) pairs, freshest first.
    """
    if held is None:
        held = open_log_owners()
    owned = {os.path.realpath(j.log) for j in jobs if j.log}
    owned |= set(held)
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        found = sh(["find", root, "-maxdepth", "5", "-type", "f", "-name", "*.log",
                    "-newermt", f"-{hours} hours"])
        for p in found.splitlines():
            p = p.strip()
            if not p or os.path.realpath(p) in owned:
                continue
            mt = mtime_of(p)
            if mt is not None:
                out.append((p, int(time.time() - mt)))
    return sorted(out, key=lambda x: x[1])[:8]


def hms(sec: int | None) -> str:
    """Compact duration, e.g. 3h12m, for display."""
    if sec is None:
        return "-"
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def main() -> int:
    """Print the box's status and return 0 only if every GPU is verified busy."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stall-s", type=int, default=1800,
                    help="a job whose log has not advanced in this long is STALLED")
    ap.add_argument("--log-roots", default=os.path.expanduser("~/areal-runs/logs"),
                    help="comma-separated dirs to scan for DEAD run logs")
    args = ap.parse_args()

    jobs, idle, client = collect(args.stall_s)
    host = os.uname().nodename
    print(f"=== {host}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    if not jobs:
        print("no process holds any GPU")
    for j in jobs:
        g = ",".join(str(x) for x in j.gpus) or "?"
        print(f"[{j.verdict:16s}] gpu {g:9s} pid {j.pid:<8d} {hms(j.age_s):>6s} "
              f"{j.mem_mib // 1024:>3d}G  {j.kind}")
        print(f"    cmd  {j.cmdline[:132]}")
        if j.log:
            print(f"    log  {j.log}  (written {hms(j.log_age_s)} ago)")
        if j.last_progress:
            print(f"    at   {j.last_progress}")
        if j.note:
            print(f"    !!   {j.note}")

    if client:
        print(f"[CLIENT          ] {client[:132]}")

    if idle:
        print(f"[IDLE            ] gpu {','.join(str(g) for g in idle)}  "
              f"-- {len(idle)} GPU(s) holding nothing")

    dead = dead_history([r for r in args.log_roots.split(",") if r], jobs)
    if dead:
        print("\n--- History: logs no live process owns (DEAD; NOT current status) ---")
        for p, age in dead:
            print(f"[DEAD            ] {hms(age):>6s} ago  {p}")

    # Anything that is not a verified LIVE is degraded, PROGRESS-UNKNOWN included:
    # the exit status must not assert progress the tool could not observe.
    bad = [j for j in jobs if j.verdict != "LIVE"]
    print()
    if bad or idle:
        print(f"VERDICT: NOT FULLY BUSY -- {len(idle)} idle GPU(s), "
              f"{len(bad)} degraded job(s)")
        return 1
    print("VERDICT: all GPUs held by jobs that are advancing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
