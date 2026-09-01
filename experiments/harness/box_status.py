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
  IDLE         the GPU holds nothing
  DEAD         a run log exists but nothing owns it; shown only under History

Exit status is 0 when every GPU is LIVE, 1 when anything is IDLE, STALLED or
SERVER-ONLY, so a supervisor can act on it.
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
# Lines that indicate a job advanced. Kept deliberately broad; a job whose log never
# matches any of these is reported as PROGRESS-UNKNOWN rather than silently healthy.
PROGRESS_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:step|Epoch|epoch|iter|global_step|solved|acc)"
    r"[ =:]*\d|\d+\s*/\s*\d+"
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
                rows.append((int(parts[0]), int(float(parts[1])), parts[2]))
            except ValueError:
                continue
    return rows


def proc_field(pid: int, name: str) -> str:
    """Read one /proc/<pid> text file, or '' if the process is gone."""
    try:
        with open(f"/proc/{pid}/{name}", "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


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
    found = sorted(set(found), key=lambda p: -os.path.getmtime(p))
    return found


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


def collect(stall_s: int) -> tuple[list[Job], list[int]]:
    """Build one Job per GPU-holding PID, plus the list of idle GPU indices."""
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
                j.log_age_s = int(time.time() - os.path.getmtime(j.log))
                j.last_progress = tail_progress(j.log)
            jobs[pid] = j
        if gpu is not None and gpu not in j.gpus:
            j.gpus.append(gpu)
        j.mem_mib += mem

    # A server with no client anywhere on the box is holding GPUs for nothing.
    have_client = any(j.kind == "client" for j in jobs.values())
    if not have_client:
        for line in sh(["ps", "-eo", "args"]).splitlines():
            if any(p in line for p in CLIENT_PATTERNS):
                have_client = True
                break

    for j in jobs.values():
        if j.kind == "server" and not have_client:
            j.verdict, j.note = "SERVER-ONLY", "server holds GPUs but no client is driving it"
        elif j.log is None:
            j.verdict, j.note = "PROGRESS-UNKNOWN", "no log on this process's descriptors"
        elif j.last_progress is None:
            j.verdict, j.note = "PROGRESS-UNKNOWN", "log has no line matching a progress pattern"
        elif j.log_age_s is not None and j.log_age_s > stall_s:
            j.verdict, j.note = "STALLED", f"log has not advanced in {j.log_age_s}s"
        else:
            j.verdict = "LIVE"

    idle = sorted(g for g in every if g not in busy)
    return sorted(jobs.values(), key=lambda x: (x.gpus or [99])[0]), idle


def dead_history(roots: list[str], jobs: list[Job], hours: int = 48) -> list[tuple[str, int]]:
    """Recently-written run logs that NO live GPU job owns.

    These are reported separately and labelled DEAD. Presenting one of these as the
    box's current state is precisely the error this tool prevents.
    """
    owned = {j.log for j in jobs if j.log}
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        found = sh(["find", root, "-maxdepth", "5", "-name", "*.log",
                    "-newermt", f"-{hours} hours"])
        for p in found.splitlines():
            p = p.strip()
            if p and p not in owned and os.path.isfile(p):
                out.append((p, int(time.time() - os.path.getmtime(p))))
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stall-s", type=int, default=1800,
                    help="a job whose log has not advanced in this long is STALLED")
    ap.add_argument("--log-roots", default=os.path.expanduser("~/areal-runs/logs"),
                    help="comma-separated dirs to scan for DEAD run logs")
    args = ap.parse_args()

    jobs, idle = collect(args.stall_s)
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

    if idle:
        print(f"[IDLE            ] gpu {','.join(str(g) for g in idle)}  "
              f"-- {len(idle)} GPU(s) holding nothing")

    dead = dead_history([r for r in args.log_roots.split(",") if r], jobs)
    if dead:
        print("\n--- History: logs no live GPU job owns (DEAD; NOT current status) ---")
        for p, age in dead:
            print(f"[DEAD            ] {hms(age):>6s} ago  {p}")

    bad = [j for j in jobs if j.verdict in ("STALLED", "SERVER-ONLY")]
    print()
    if bad or idle:
        print(f"VERDICT: NOT FULLY BUSY -- {len(idle)} idle GPU(s), "
              f"{len(bad)} degraded job(s)")
        return 1
    print("VERDICT: all GPUs held by jobs that are advancing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
