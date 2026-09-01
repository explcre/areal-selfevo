"""Tests for the ground-truth box status reporter.

None of these need a GPU. The box is built out of injected data -- a fake `sh` that
answers nvidia-smi, and fake /proc readers -- so the parsing, the verdict ladder, the
exit status and the LIVE-vs-DEAD separation are all exercised on CPU.

What is being constrained is not "does it print something": it is the two claims the
tool exists to make honestly. A log no live process owns must never appear as current
status, and a status of LIVE must never be asserted for a job the tool could not
observe advancing. Every test below is written so that a defect in one of those two
claims fails it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_status as bs  # noqa: E402

NOW = 1_800_000_000.0


class FakeBox:
    """A GPU box described entirely by data, installed over box_status's readers.

    Every field mirrors one real source of truth: `apps` is nvidia-smi's compute-apps
    table, `fds` is /proc/<pid>/fd, `mtime` is the filesystem, `tail` is the log text.
    Building the box this way is what makes a stalled job, a server with no client and
    a dead log reproducible without hardware.
    """

    def __init__(self):
        self.gpus: list[int] = [0, 1]
        self.apps: list[tuple[int, object, int]] = []   # (pid, used_mib, gpu index)
        self.cmd: dict[int, str] = {}
        self.argv: dict[int, list[str]] = {}
        self.parents: dict[int, list[int]] = {}
        self.fds: dict[int, list[str]] = {}
        self.mtime: dict[str, float] = {}
        self.tail: dict[str, str] = {}
        self.find: dict[str, list[str]] = {}
        self.extra_pids: list[int] = []

    def add_job(self, pid, gpu, cmd, log=None, log_age=5, tail="step 12", mem=40000):
        """Register one GPU-holding process, optionally with a log of a given age."""
        self.apps.append((pid, mem, gpu))
        self.cmd[pid] = cmd
        self.argv.setdefault(pid, cmd.split())
        if log is not None:
            self.fds[pid] = [log]
            self.mtime[log] = NOW - log_age
            self.tail[log] = tail
        return self

    def add_process(self, pid, cmd, argv=None):
        """Register a process that holds no GPU, e.g. a benchmark client or a shell."""
        self.cmd[pid] = cmd
        self.argv[pid] = argv if argv is not None else cmd.split()
        self.extra_pids.append(pid)
        return self

    def _sh(self, cmd, timeout=30):
        """Stand in for every external command box_status shells out to."""
        if cmd[0] == "nvidia-smi":
            if "--query-gpu=index,uuid" in cmd:
                return "".join(f"{i}, GPU-{i}\n" for i in self.gpus)
            if "--query-gpu=index" in cmd:
                return "".join(f"{i}\n" for i in self.gpus)
            if any(c.startswith("--query-compute-apps") for c in cmd):
                return "".join(f"{p}, {m}, GPU-{g}\n" for p, m, g in self.apps)
        if cmd[0] == "tail":
            return self.tail.get(cmd[-1], "")
        if cmd[0] == "find":
            return "".join(f"{p}\n" for p in self.find.get(cmd[1], []))
        return ""

    def install(self, monkeypatch):
        """Point box_status at this fake box instead of the real machine."""
        monkeypatch.setattr(bs, "sh", self._sh)
        monkeypatch.setattr(bs, "cmdline_of", lambda pid: self.cmd.get(pid, ""))
        monkeypatch.setattr(bs, "argv_of", lambda pid: self.argv.get(pid, []))
        monkeypatch.setattr(bs, "ancestors", lambda pid, limit=12: self.parents.get(pid, []))
        monkeypatch.setattr(bs, "age_of", lambda pid: 600)
        monkeypatch.setattr(bs, "open_logs", lambda pid: list(self.fds.get(pid, [])))
        # Injected paths use the injected mtime; real files on disk (the DEAD-history
        # tests use real ones, because deadness is a filesystem question) keep theirs.
        real_mtime = bs.mtime_of
        monkeypatch.setattr(bs, "mtime_of",
                            lambda p: self.mtime[p] if p in self.mtime else real_mtime(p))
        monkeypatch.setattr(bs, "live_pids",
                            lambda: sorted(set(self.cmd) | set(self.extra_pids)))
        monkeypatch.setattr(bs.time, "time", lambda: NOW)
        return self


@pytest.fixture()
def box(monkeypatch):
    """An empty two-GPU box with every external reader stubbed out."""
    return FakeBox().install(monkeypatch)


SGLANG = "sglang::scheduler_TP0"
CLIENT = "python3 experiments/bench/math_bench.py --base-url http://127.0.0.1:8404/v1"
SGLANG_LINE = ("[2026-09-01 00:56:38 TP0] Decode batch, #running-req: 11, #token: 214206, "
               "token usage: 0.10, cuda graph: True, gen throughput (token/s): 930.05, "
               "#queue-req: 0")


# ------------------------------------------------------------------- progress pattern


def test_progress_pattern_matches_the_format_the_gpu_jobs_actually_write():
    """sglang's decode line is THE log format the GPU-holding processes emit here.

    Measured before this test existed: the pattern matched none of the 20 sglang
    server.log files under ~/runs/math, so every live job on the box reported
    PROGRESS-UNKNOWN. A progress pattern that does not recognise the one format the
    jobs write is not broad, it is blind.
    """
    assert bs.PROGRESS_RE.search(SGLANG_LINE)
    assert bs.PROGRESS_RE.search("Capturing batches: 100%|##########| 16/16 [00:15<00:00]")
    assert bs.PROGRESS_RE.search("global_step 41 loss 0.3")


def test_progress_pattern_does_not_match_boilerplate():
    """PROGRESS-UNKNOWN has to stay reachable, or it reports nothing."""
    for line in (
        "Traceback (most recent call last):",
        "warnings.warn(",
        "torch_dtype is deprecated! Use dtype instead!",
        "[Gloo] Rank is connected to peer ranks.",
    ):
        assert not bs.PROGRESS_RE.search(line), line


def test_tail_progress_returns_the_last_matching_line(box):
    box.tail["/runs/a/server.log"] = "boot\n" + SGLANG_LINE + "\nwarnings.warn(\n"
    assert bs.tail_progress("/runs/a/server.log") == SGLANG_LINE[:150]
    box.tail["/runs/b/server.log"] = "Traceback (most recent call last):\nwarnings.warn(\n"
    assert bs.tail_progress("/runs/b/server.log") is None


# --------------------------------------------------------------------- verdict ladder


def test_a_fresh_job_is_live(box):
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=5, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", log_age=5, tail=SGLANG_LINE)
    jobs, idle, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["LIVE", "LIVE"]
    assert idle == []


def test_a_stalled_job_is_stalled_not_live(box):
    """A log older than --stall-s is the failure the tool was written to catch."""
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=9999, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", log_age=9999, tail=SGLANG_LINE)
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["STALLED", "STALLED"]
    assert "9999s" in jobs[0].note


def test_the_stall_boundary_is_strictly_greater(box):
    """Exactly at the threshold is not yet stalled; one second past it is."""
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=1800, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/b/math.log", log_age=1801, tail=SGLANG_LINE)
    verdicts = {j.pid: j.verdict for j in bs.collect(stall_s=1800)[0]}
    assert verdicts == {100: "LIVE", 101: "STALLED"}


def test_a_stalled_job_is_stalled_even_when_its_log_format_is_unrecognised(box):
    """mtime is ground truth for "advanced"; the progress pattern is only display.

    The pattern check used to run first, so a stalled job whose log the pattern did
    not recognise was reported PROGRESS-UNKNOWN -- and PROGRESS-UNKNOWN exited 0. On
    this box every sglang server was in exactly that state.
    """
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=9999,
                tail="Traceback (most recent call last):")
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", log_age=9999,
                tail="Traceback (most recent call last):")
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["STALLED", "STALLED"]


def test_a_job_with_no_log_is_progress_unknown_not_live(box):
    box.add_job(100, 0, CLIENT, log=None)
    box.add_job(101, 1, CLIENT, log=None)
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["PROGRESS-UNKNOWN"] * 2
    assert "descriptors" in jobs[0].note


def test_a_fresh_log_with_no_progress_line_is_progress_unknown(box):
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=5, tail="warnings.warn(")
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", log_age=5, tail="warnings.warn(")
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["PROGRESS-UNKNOWN"] * 2


def test_a_job_finds_its_log_through_an_ancestor(box):
    """A worker that inherited the shell's redirect still has to be tied to that log."""
    box.add_job(100, 0, SGLANG, log=None)
    box.add_job(101, 1, SGLANG, log=None)
    box.parents[100] = [50]
    box.parents[101] = [50]
    box.fds[50] = ["/runs/a/server.log"]
    box.mtime["/runs/a/server.log"] = NOW - 3
    box.tail["/runs/a/server.log"] = SGLANG_LINE
    box.add_process(900, CLIENT)
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.log for j in jobs] == ["/runs/a/server.log"] * 2
    assert [j.verdict for j in jobs] == ["LIVE", "LIVE"]


# ------------------------------------------------------------------------ SERVER-ONLY


def test_a_server_with_no_client_anywhere_is_server_only(box):
    """Servers hold GPU memory whether or not anything is driving them."""
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    jobs, idle, client = bs.collect(stall_s=1800)
    assert client is None
    assert [j.verdict for j in jobs] == ["SERVER-ONLY", "SERVER-ONLY"]
    assert idle == []


def test_a_server_with_a_client_off_the_gpus_is_not_server_only(box):
    """math_bench.py is an HTTP client and holds no GPU; it still counts."""
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    box.add_process(900, CLIENT)
    jobs, _, client = bs.collect(stall_s=1800)
    assert client is not None and "math_bench.py" in client
    assert [j.verdict for j in jobs] == ["LIVE", "LIVE"]


def test_server_only_outranks_a_stale_log(box):
    """Wasting GPUs is the finding; the log age is not why it is degraded."""
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", log_age=9999, tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", log_age=9999, tail=SGLANG_LINE)
    jobs, _, _ = bs.collect(stall_s=1800)
    assert [j.verdict for j in jobs] == ["SERVER-ONLY", "SERVER-ONLY"]


def test_a_mention_of_a_client_is_not_a_client():
    """Observed on the box: the run supervisor's command line names math_bench.py.

    `bash -c "... git add -A experiments/bench/math_bench.py ..."` was counted as a
    live client, which switches off the SERVER-ONLY alarm for the whole box exactly
    when every real client has exited and the servers are burning GPUs for nothing.
    """
    supervisor = ("cd ~/areal-selfevo && git add -A experiments/bench/math_bench.py "
                  "experiments/harness/box_status.py && git commit -q -m 'x' " + "y" * 1600)
    assert not bs.looks_like_client(["bash", "-c", supervisor])
    assert bs.looks_like_client(["python3", "experiments/bench/math_bench.py", "--limit", "0"])
    assert bs.looks_like_client(["bash", "experiments/bench/run_math.sh", "model"])


def test_harbor_is_recognised_in_the_forms_the_repo_actually_launches_it():
    """run_tb_swap.sh sets HARBOR to a binary OR to `<python> -m harbor`, then runs
    `$HARBOR run -c ...` -- both forms have to count, and nothing else may.

    The trailing space in the "harbor " pattern is what keeps `pip install harbor==...`
    out; restricting the match to the program slot is what keeps `grep harbor` and the
    preflight's own `python -c "import harbor"` probe out.
    """
    assert bs.looks_like_client(["/home/ubuntu/venv312b/bin/harbor", "run", "-c", "cfg"])
    assert bs.looks_like_client(["/usr/bin/python3", "-m", "harbor", "run", "-c", "cfg"])
    assert not bs.looks_like_client(["grep", "-rn", "harbor", "experiments/"])
    assert not bs.looks_like_client(["/usr/bin/python3", "-c", "import harbor"])
    assert not bs.looks_like_client(["pip", "install", "-q", "harbor==0.18.0"])


def test_the_status_tool_does_not_count_itself_as_a_client(box, monkeypatch):
    """Running the check from a shell that names a client must not fake one up."""
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", log_age=5, tail=SGLANG_LINE)
    me = os.getpid()
    box.add_process(me, "bash run_math.sh")
    monkeypatch.setattr(bs, "ancestors", lambda pid, limit=12: [])
    assert bs.collect(stall_s=1800)[2] is None


# ------------------------------------------------------------------------- idle + sort


def test_idle_gpus_are_reported(box):
    box.gpus = [0, 1, 2, 3]
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    jobs, idle, _ = bs.collect(stall_s=1800)
    assert idle == [1, 2, 3]
    assert [j.verdict for j in jobs] == ["LIVE"]


def test_a_gpu_whose_memory_the_driver_will_not_report_is_still_busy(box):
    """nvidia-smi prints '[N/A]' for used_memory under MIG.

    Dropping the row lost the fact that the GPU was held at all and reported it
    IDLE, which is a claim about free capacity that a scheduler would act on.
    """
    box.apps.append((100, "[N/A]", 0))
    box.cmd[100] = CLIENT
    box.argv[100] = CLIENT.split()
    box.fds[100] = ["/runs/a/math.log"]
    box.mtime["/runs/a/math.log"] = NOW - 5
    box.tail["/runs/a/math.log"] = SGLANG_LINE
    jobs, idle, _ = bs.collect(stall_s=1800)
    assert idle == [1]
    assert [j.pid for j in jobs] == [100] and jobs[0].mem_mib == 0


def test_a_multi_gpu_job_reports_its_gpus_in_order_and_sums_memory(box):
    """nvidia-smi returns compute-apps rows in no guaranteed order."""
    box.gpus = [0, 1, 2, 3]
    box.apps = [(100, 1000, 3), (100, 1000, 0), (100, 1000, 2), (100, 1000, 1)]
    box.cmd[100] = CLIENT
    box.argv[100] = CLIENT.split()
    box.fds[100] = ["/runs/a/math.log"]
    box.mtime["/runs/a/math.log"] = NOW - 5
    box.tail["/runs/a/math.log"] = SGLANG_LINE
    jobs, idle, _ = bs.collect(stall_s=1800)
    assert jobs[0].gpus == [0, 1, 2, 3]
    assert jobs[0].mem_mib == 4000
    assert idle == []


# ----------------------------------------------------------------------- DEAD history


def _write(path: Path, age_s: float) -> str:
    """Create a log file with a given age and return its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n")
    os.utime(path, (time.time() - age_s, time.time() - age_s))
    return str(path)


def test_a_log_no_process_owns_is_dead_and_is_not_a_job(box, tmp_path):
    """The whole point: an orphan log is history, never current status."""
    orphan = _write(tmp_path / "old" / "server.log", 7200)
    box.find[str(tmp_path)] = [orphan]
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    jobs, _, _ = bs.collect(stall_s=1800)
    dead = bs.dead_history([str(tmp_path)], jobs, held={})
    assert [p for p, _ in dead] == [orphan]
    assert orphan not in {j.log for j in jobs}


def test_a_log_a_gpu_job_owns_is_never_dead(box, tmp_path):
    owned = _write(tmp_path / "live" / "math.log", 10)
    orphan = _write(tmp_path / "old" / "server.log", 7200)
    box.find[str(tmp_path)] = [owned, orphan]
    box.add_job(100, 0, CLIENT, log=owned, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log=owned, tail=SGLANG_LINE)
    jobs, _, _ = bs.collect(stall_s=1800)
    dead = bs.dead_history([str(tmp_path)], jobs, held={})
    assert [p for p, _ in dead] == [orphan]


def test_a_log_a_live_non_gpu_process_holds_open_is_never_dead(box, tmp_path):
    """Measured on this box: `f30b_olymp32k_v2/math.log` was reported DEAD while a
    live `tee` (pid 1551620) held it open for a benchmark that was still running.

    The client that writes it holds no GPU, so the GPU jobs alone cannot answer
    deadness. A false obituary is as misleading as a false status: acting on it
    means restarting a run that was fine.
    """
    held = _write(tmp_path / "live" / "math.log", 10)
    orphan = _write(tmp_path / "old" / "server.log", 7200)
    box.find[str(tmp_path)] = [held, orphan]
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", tail=SGLANG_LINE)
    box.add_process(900, CLIENT)
    box.fds[900] = [held]
    jobs, _, _ = bs.collect(stall_s=1800)
    dead = bs.dead_history([str(tmp_path)], jobs)
    assert [p for p, _ in dead] == [orphan]


def test_open_log_owners_maps_a_log_to_the_pid_holding_it(box):
    box.add_process(900, CLIENT)
    box.fds[900] = ["/runs/a/math.log"]
    owners = bs.open_log_owners()
    assert owners[os.path.realpath("/runs/a/math.log")] == 900


def test_deadness_survives_a_symlinked_log_root(box, tmp_path):
    """A descriptor's target is fully resolved; a find result carries the caller's
    symlinks. Comparing the two as raw strings labels an OWNED log DEAD."""
    real = tmp_path / "real"
    owned = _write(real / "run" / "math.log", 10)
    link = tmp_path / "link"
    link.symlink_to(real)
    box.add_job(100, 0, CLIENT, log=owned, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log=owned, tail=SGLANG_LINE)
    box.find[str(link)] = [str(link / "run" / "math.log")]
    jobs, _, _ = bs.collect(stall_s=1800)
    assert bs.dead_history([str(link)], jobs, held={}) == []


def test_dead_history_skips_a_log_that_vanishes_between_find_and_stat(box, tmp_path):
    """A rotated log must not take the whole status report down with it."""
    gone = str(tmp_path / "gone.log")
    alive = _write(tmp_path / "there.log", 60)
    box.find[str(tmp_path)] = [gone, alive]
    dead = bs.dead_history([str(tmp_path)], [], held={})
    assert [p for p, _ in dead] == [alive]


def test_open_logs_survives_a_file_that_disappears(monkeypatch, tmp_path):
    """The mtime sort key ran on paths that can vanish; unguarded it raised."""
    good = tmp_path / "a.log"
    good.write_text("x")
    gone = tmp_path / "b.log"
    monkeypatch.setattr(bs.os, "listdir", lambda d: ["3", "4"])
    monkeypatch.setattr(bs.os, "readlink",
                        lambda p: str(good) if p.endswith("3") else str(gone))
    monkeypatch.setattr(bs.os.path, "isfile", lambda p: True)
    assert bs.open_logs(1234) == [str(good), str(gone)]


def test_mtime_of_returns_none_rather_than_raising(tmp_path):
    assert bs.mtime_of(str(tmp_path / "nope.log")) is None
    real = tmp_path / "yes.log"
    real.write_text("x")
    assert bs.mtime_of(str(real)) == pytest.approx(os.path.getmtime(real))


# ------------------------------------------------------------------------ exit status


def _main(box, monkeypatch, tmp_path, capsys):
    """Run main() against the fake box and return (exit status, stdout)."""
    monkeypatch.setattr(sys, "argv",
                        ["box_status.py", "--stall-s", "1800", "--log-roots", str(tmp_path)])
    rc = bs.main()
    return rc, capsys.readouterr().out


def test_idle_gpus_force_a_nonzero_exit(box, monkeypatch, tmp_path, capsys):
    """Half the box doing nothing is a finding, even when every job is healthy."""
    box.gpus = [0, 1, 2, 3]
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "[IDLE" in out and "gpu 1,2,3" in out
    assert "NOT FULLY BUSY" in out
    assert "3 idle GPU(s), 0 degraded job(s)" in out


def test_a_fully_busy_box_exits_zero(box, monkeypatch, tmp_path, capsys):
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 0
    assert "all GPUs held by jobs that are advancing" in out
    assert "IDLE" not in out


def test_a_stalled_job_forces_a_nonzero_exit(box, monkeypatch, tmp_path, capsys):
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", log_age=9999, tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", log_age=9999, tail=SGLANG_LINE)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "[STALLED" in out


def test_a_server_only_box_forces_a_nonzero_exit(box, monkeypatch, tmp_path, capsys):
    box.add_job(100, 0, SGLANG, log="/runs/a/server.log", tail=SGLANG_LINE)
    box.add_job(101, 1, SGLANG, log="/runs/a/server.log", tail=SGLANG_LINE)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "[SERVER-ONLY" in out


def test_progress_unknown_does_not_claim_the_box_is_advancing(box, monkeypatch,
                                                              tmp_path, capsys):
    """The tool could not observe progress, so it must not assert progress.

    Every GPU on this box reported PROGRESS-UNKNOWN and the summary line still said
    "all GPUs held by jobs that are advancing", with exit 0. That is the failure
    class the whole tool is against, printed by the tool itself.
    """
    box.add_job(100, 0, CLIENT, log=None)
    box.add_job(101, 1, CLIENT, log=None)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "advancing" not in out
    assert "2 degraded job(s)" in out


def test_a_dead_log_appears_only_under_history(box, monkeypatch, tmp_path, capsys):
    """It must be structurally impossible to read an orphan log as current status."""
    orphan = _write(tmp_path / "old" / "server.log", 7200)
    box.find[str(tmp_path)] = [orphan]
    box.add_job(100, 0, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    box.add_job(101, 1, CLIENT, log="/runs/a/math.log", tail=SGLANG_LINE)
    rc, out = _main(box, monkeypatch, tmp_path, capsys)
    assert rc == 0
    head, _, history = out.partition("--- History")
    assert orphan not in head
    assert orphan in history
    assert "NOT current status" in history
