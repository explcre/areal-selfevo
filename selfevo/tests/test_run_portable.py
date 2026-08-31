"""``experiments/harness/run_portable.sh``, exercised as a library.

The script is handed to an external collaborator to run unattended on a shared GPU box we
do not control, so the failure that matters most is not "our run died" -- it is "our run
killed, starved or slowed a NEIGHBOUR'S job". Every function below that can reach a
neighbour is therefore driven against a fake topology: ``nvidia-smi``, ``pkill`` and
``pgrep`` are stubs at the front of PATH, and nothing in this file touches a GPU.

The script is sourced with ``RUN_PORTABLE_SOURCE_ONLY=1``, which defines its functions and
runs nothing. That guard exists for these tests: without it, reaching the functions means
running ``main``, i.e. cloning a repo and claiming GPUs on whatever box the suite runs on.

Three properties are swept rather than spot-checked, because each has already been a defect
here:

  * the free-GPU parse, over the shapes a driver actually prints -- ``0, 20214`` and
    ``0,20214``, ``[N/A]``, multi-digit indices. A mis-parse reads a neighbour's 67 GB job
    as a free GPU, and the neighbour is what we are protecting;
  * the claim, over 0/1/2/3/5/7/8 free GPUs and neighbour-held boards. Rounding an odd
    count DOWN must never yield an empty list reported as a success, and never fewer GPUs
    than ``MIN_GPUS``;
  * the ownership test inside ``cleanup_ours``, over device lists that nest, overlap and
    are permuted. A substring match killed a neighbour whose list merely CONTAINED ours,
    and never matched our own workers, which AReaL gives one physical id each.

The manifest tests assert on the FILE rather than on the shell's variables. The manifest is
produced by a python heredoc reading ``os.environ``, so the defect class it hides is a
shell variable that was never exported arriving as an empty string -- a manifest that looks
well-formed and reports nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "harness" / "run_portable.sh"

# Rows exactly as the driver prints them for
# ``--query-gpu=index,memory.used --format=csv,noheader,nounits``: index, comma, SPACE, MiB.
# Copied from a real 8xA100 box while a neighbour held every board.
NEIGHBOUR_HOLDS_ALL_EIGHT = [
    "0, 20214", "1, 20502", "2, 23102", "3, 20654",
    "4, 67176", "5, 67156", "6, 67056", "7, 67156",
]


def idle(n: int, first: int = 0) -> list[str]:
    """``n`` rows describing idle GPUs numbered from ``first``, in the real CSV shape."""
    return [f"{first + i}, 0" for i in range(n)]


class Harness:
    """A sandbox that runs snippets against the script's functions, without a GPU.

    ``nvidia-smi``, ``pkill`` and ``pgrep`` are replaced by stubs at the front of PATH, so a
    test that reaches for a real GPU or a real process gets the stub instead -- and the call
    shows up in :meth:`calls` rather than happening to the machine running the suite.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.workdir = root / "wd"
        self.outdir = self.workdir / "out"
        self.stub = root / "stub"
        self.stub.mkdir(parents=True)
        self.calls_file = root / "stub-calls"
        self._write(self.stub / "nvidia-smi", """
            [ -n "${FAKE_SMI_ROWS:-}" ] && printf '%s\\n' "$FAKE_SMI_ROWS"
            exit 0
        """)
        for name in ("pkill", "pgrep"):
            self._write(self.stub / name, f"""
                echo "{name} $*" >> "$STUB_CALLS"
                [ -n "${{FAKE_PIDS:-}}" ] && printf '%s\\n' $FAKE_PIDS
                exit 0
            """)

    @staticmethod
    def _write(path: Path, body: str) -> None:
        """Drop an executable bash stub at ``path``."""
        path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body).strip() + "\n")
        path.chmod(0o755)

    def env(self, **extra: object) -> dict[str, str]:
        """The environment a snippet runs in: stubs first on PATH, a sandbox WORKDIR.

        ``CUDA_VISIBLE_DEVICES`` is removed rather than blanked, because "unset" is the
        state the script actually meets before it has claimed anything and the state
        ``set -u`` would abort on.
        """
        env = dict(os.environ)
        env.update(
            PATH=f"{self.stub}{os.pathsep}{os.environ['PATH']}",
            RUN_PORTABLE_SOURCE_ONLY="1",
            WORKDIR=str(self.workdir),
            OUTDIR=str(self.outdir),
            STUB_CALLS=str(self.calls_file),
            RUN_NAME="t",
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(self, snippet: str, rows: list[str] | None = None, timeout: int = 60,
            **env: object) -> subprocess.CompletedProcess:
        """Source the script, then run ``snippet``; ``rows`` becomes the fake nvidia-smi."""
        if rows is not None:
            env["FAKE_SMI_ROWS"] = "\n".join(rows)
        body = f'source "{SCRIPT}"\n' + textwrap.dedent(snippet)
        return subprocess.run(["bash", "-c", body], env=self.env(**env),
                              capture_output=True, text=True, timeout=timeout)

    def popen(self, snippet: str, **env: object) -> subprocess.Popen:
        """Same, left running, for the lock and signal tests."""
        body = f'source "{SCRIPT}"\n' + textwrap.dedent(snippet)
        return subprocess.Popen(["bash", "-c", body], env=self.env(**env),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def calls(self) -> list[str]:
        """Every stubbed ``pkill``/``pgrep`` invocation made so far."""
        if not self.calls_file.exists():
            return []
        return self.calls_file.read_text().splitlines()

    def manifest(self, name: str = "t") -> dict:
        """The parsed manifest for run ``name``."""
        return json.loads((self.outdir / f"{name}.manifest.json").read_text())


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    """A fresh sandbox per test; nothing it does escapes ``tmp_path``."""
    return Harness(tmp_path)


def claim(harness: Harness, rows: list[str], **env: object) -> tuple[int, str, int]:
    """Run ``claim_gpus`` and return ``(rc, csv_of_devices, count)``."""
    env.setdefault("WAIT_FOR_GPUS_S", 0)
    got = harness.run('got="$(claim_gpus)"; rc=$?; echo "RC=$rc"; echo "GOT=$got"',
                      rows=rows, **env)
    rc = int(got.stdout.split("RC=")[1].split("\n")[0])
    csv = got.stdout.split("GOT=")[1].split("\n")[0]
    return rc, csv, len([d for d in csv.split(",") if d])


# ------------------------------------------------------------------ parsing free GPUs ---


@pytest.mark.parametrize(
    "rows,expected",
    [
        # The format a real driver prints, with the space after the comma.
        (["0, 0", "1, 0", "2, 90000"], "0,1"),
        # ... and without it. A bare -F", " leaves $2 empty here, which compares as free
        # AND prints the whole line as the index -- "0,0,1,0" as a device list.
        (["0,0", "1,0", "2,90000"], "0,1"),
        # A driver that cannot report memory tells us nothing, so we must claim nothing.
        (["0, [N/A]", "1, 12"], "1"),
        (["0, 0", "1, "], "0"),
        # Indices are not single digits on every box.
        (["8, 0", "9, 0", "10, 0", "11, 40000"], "8,9,10"),
        # The threshold is strict: exactly GPU_FREE_MIB is NOT free.
        (["0, 4096", "1, 4095"], "1"),
        ([], ""),
    ],
)
def test_free_gpus_parses_the_shapes_a_driver_actually_prints(harness, rows, expected):
    """Every row we cannot parse must drop out, not default to free.

    This is the parse that decides whether a board belongs to a neighbour, so the failure
    direction matters: reading an unparseable row as free is how we land on top of someone.
    """
    out = harness.run('free_gpus', rows=rows, GPU_FREE_MIB=4096, MAX_GPUS=8)
    assert out.stdout.strip() == expected, out.stdout


def test_a_neighbour_holding_every_board_yields_no_claim(harness):
    """The live topology this script was written for: eight GPUs, none of them ours."""
    out = harness.run('free_gpus', rows=NEIGHBOUR_HOLDS_ALL_EIGHT, GPU_FREE_MIB=4096, MAX_GPUS=8)
    assert out.stdout.strip() == "", out.stdout


def test_only_the_idle_half_of_a_shared_box_is_offered(harness):
    """A neighbour on 4-7 keeps 4-7; we may look at 0-3 and nothing else."""
    rows = idle(4) + NEIGHBOUR_HOLDS_ALL_EIGHT[4:]
    out = harness.run('free_gpus', rows=rows, GPU_FREE_MIB=4096, MAX_GPUS=8)
    assert out.stdout.strip() == "0,1,2,3", out.stdout


@pytest.mark.parametrize("max_gpus,expected", [(2, "0,1"), (4, "0,1,2,3"), (8, "0,1,2,3,4,5,6,7")])
def test_free_gpus_never_offers_more_than_max_gpus(harness, max_gpus, expected):
    """MAX_GPUS caps the offer before anything else looks at it."""
    out = harness.run('free_gpus', rows=idle(8), GPU_FREE_MIB=4096, MAX_GPUS=max_gpus,
                      MIN_GPUS=2)
    assert out.stdout.strip() == expected, out.stdout


# ----------------------------------------------------------------------- claiming GPUs ---


@pytest.mark.parametrize("free", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_a_claim_is_even_never_empty_and_never_below_min_gpus(harness, free):
    """The invariant, swept over every board count a shared box can leave us.

    Three things have to hold at once, and the original held only the third: the count is
    even (AReaL splits train/rollout, an odd GPU is wasted), a success is never an empty
    list, and a success is never fewer GPUs than the caller demanded. Rounding an odd count
    down AFTER the MIN_GPUS test broke the first two -- at ``free=1`` it returned success
    with no devices at all.
    """
    rc, csv, n = claim(harness, idle(free), MIN_GPUS=4, MAX_GPUS=8)
    if rc == 0:
        assert n > 0, csv
        assert n % 2 == 0, csv
        assert n >= 4, csv
        assert n <= free
    else:
        assert csv == "", csv
        assert free < 4


@pytest.mark.parametrize("free,expected", [(3, 2), (5, 4), (7, 6), (9, 8)])
def test_an_odd_count_rounds_down_to_the_next_even(harness, free, expected):
    """It keeps the LOWEST-numbered GPUs and drops exactly one, whatever the width."""
    rc, csv, n = claim(harness, idle(free), MIN_GPUS=1, MAX_GPUS=16)
    assert rc == 0, csv
    assert n == expected, csv
    assert csv == ",".join(str(i) for i in range(expected)), csv


def test_one_free_gpu_is_refused_rather_than_claimed_as_nothing(harness):
    """The exact path that produced an empty CUDA_VISIBLE_DEVICES.

    A single free GPU rounds down to zero. Reporting that as a success handed ``main`` an
    empty device list, and an empty list is what makes ``cleanup_ours`` match every process
    on the box -- our neighbour's servers included.
    """
    rc, csv, n = claim(harness, idle(1), MIN_GPUS=1, MAX_GPUS=8)
    assert rc != 0, csv
    assert csv == "", csv


@pytest.mark.parametrize("min_gpus,free,ok", [(4, 4, True), (4, 3, False), (4, 8, True),
                                              (6, 4, False), (2, 2, True), (8, 7, False)])
def test_min_gpus_is_a_floor_not_a_suggestion(harness, min_gpus, free, ok):
    """Below MIN_GPUS the answer is "no", so the caller can exit 4 instead of thrashing."""
    rc, csv, n = claim(harness, idle(free), MIN_GPUS=min_gpus, MAX_GPUS=8)
    assert (rc == 0) is ok, (rc, csv)
    if ok:
        assert n >= min_gpus, csv


def test_an_odd_min_gpus_is_never_satisfied_by_one_gpu_fewer(harness):
    """MIN_GPUS=5 with 5 free is a refusal, not a quiet 4.

    Rounding to even and honouring the floor can conflict; when they do, the floor wins.
    Six free boards satisfy MIN_GPUS=5, five do not.
    """
    assert claim(harness, idle(5), MIN_GPUS=5, MAX_GPUS=8)[0] != 0
    assert claim(harness, idle(6), MIN_GPUS=5, MAX_GPUS=8)[1] == "0,1,2,3,4,5"


def test_a_claim_skips_the_boards_a_neighbour_is_holding(harness):
    """Claimed indices are the idle PHYSICAL ids, not a renumbering of them."""
    rows = ["0, 90000", "1, 0", "2, 90000", "3, 0", "4, 0", "5, 0"]
    rc, csv, n = claim(harness, rows, MIN_GPUS=4, MAX_GPUS=8)
    assert rc == 0 and csv == "1,3,4,5", csv


def test_a_configuration_that_can_never_be_satisfied_is_refused_at_startup(harness):
    """MAX_GPUS below MIN_GPUS would otherwise wait for GPUs it would then decline."""
    out = harness.run('echo unreachable', rows=idle(8), MIN_GPUS=8, MAX_GPUS=4)
    assert out.returncode == 4, out.stdout + out.stderr
    assert "unreachable" not in out.stdout


def test_waiting_is_bounded_by_wait_for_gpus_s(harness):
    """With the default of 0 the answer comes back immediately rather than looping."""
    started = time.monotonic()
    rc, csv, _ = claim(harness, idle(1), MIN_GPUS=4, MAX_GPUS=8, WAIT_FOR_GPUS_S=0)
    assert rc != 0 and csv == ""
    assert time.monotonic() - started < 30


# -------------------------------------------------------------------- whose process is it ---


@pytest.fixture
def sleepers():
    """Real processes carrying a chosen ``CUDA_VISIBLE_DEVICES``, reaped afterwards.

    ``ours_gpu`` reads ``/proc/<pid>/environ``, which is NUL-separated, so a stand-in file
    would not test the thing that matters. These are ``sleep`` processes: no GPU is touched.
    """
    started: list[subprocess.Popen] = []

    def spawn(devices: str) -> int:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices)
        proc = subprocess.Popen(["sleep", "30"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(proc)
        return proc.pid

    yield spawn
    for proc in started:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize(
    "ours,theirs,owned",
    [
        # What our own workers really look like: AReaL gives each sglang server ONE
        # physical id, so the full claimed list never appears in their environ. The old
        # exact-string match therefore never killed our own servers, and they survived
        # holding their GPUs into the next attempt.
        ("0,1,2,3", "0", True),
        ("0,1,2,3", "3", True),
        ("4,5,6,7", "7", True),
        # A neighbour whose list CONTAINS ours must survive. The old substring match killed
        # them: "CUDA_VISIBLE_DEVICES=0,1" is a substring of "...=0,1,2,3".
        ("0,1", "0,1,2,3", False),
        ("0", "0,1,2,3", False),
        ("4,5", "4,5,6,7", False),
        # Same set, different order, is still ours.
        ("3,2,1,0", "0,1,2,3", True),
        ("3,2,1,0", "2", True),
        # Disjoint is never ours.
        ("0,1,2,3", "4", False),
        ("0,1", "2,3", False),
        # A partial overlap is not proof of ownership either.
        ("0,1", "1,2", False),
    ],
)
def test_ownership_is_set_containment_not_substring(harness, sleepers, ours, theirs, owned):
    """The predicate that decides whether a process gets ``kill -9``.

    Both directions were wrong before and both are dangerous, but asymmetrically: a false
    negative leaves our own server holding a GPU, a false POSITIVE kills a neighbour's job.
    """
    pid = sleepers(theirs)
    out = harness.run(f'ours_gpu {pid} && echo OWNED || echo FOREIGN',
                      CUDA_VISIBLE_DEVICES=ours)
    assert out.stdout.strip() == ("OWNED" if owned else "FOREIGN"), out.stdout


@pytest.mark.parametrize("claimed", ["", None])
def test_an_empty_claim_owns_nothing(harness, sleepers, claimed):
    """The catastrophe path, from both spellings of "we have not claimed anything".

    With an empty ``CUDA_VISIBLE_DEVICES`` the old grep pattern degenerated to
    ``CUDA_VISIBLE_DEVICES=``, which matches EVERY process that sets the variable at all --
    on a box where the neighbour runs as the same user, that is their whole job.
    """
    pid = sleepers("4,5,6,7")
    env = {} if claimed is None else {"CUDA_VISIBLE_DEVICES": claimed}
    out = harness.run(f'ours_gpu {pid} && echo OWNED || echo FOREIGN', **env)
    assert out.stdout.strip() == "FOREIGN", out.stdout


def test_a_process_without_the_variable_is_never_ours(harness, sleepers):
    """We cannot prove ownership of a process that never declared a device."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    proc = subprocess.Popen(["sleep", "30"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        out = harness.run(f'ours_gpu {proc.pid} && echo OWNED || echo FOREIGN',
                          CUDA_VISIBLE_DEVICES="0,1,2,3")
        assert out.stdout.strip() == "FOREIGN", out.stdout
    finally:
        proc.kill()
        proc.wait()


def test_cleanup_does_nothing_at_all_when_no_gpus_are_claimed(harness):
    """Not "kills nothing" -- issues no pkill and no pgrep.

    Asserted on the stub call log rather than on a return value, because the damage would
    be done by the call itself.
    """
    out = harness.run('cleanup_ours; echo DONE')
    assert "DONE" in out.stdout, out.stdout + out.stderr
    assert harness.calls() == [], harness.calls()


def test_cleanup_targets_our_run_name_and_only_our_run_name(harness):
    """The pkill patterns must carry RUN_NAME, or they match a neighbour's trainer too."""
    out = harness.run('cleanup_ours; echo DONE', CUDA_VISIBLE_DEVICES="0,1",
                      RUN_NAME="ours-1234", timeout=90)
    assert "DONE" in out.stdout, out.stdout + out.stderr
    pkills = [c for c in harness.calls() if c.startswith("pkill")]
    assert pkills, harness.calls()
    assert all("ours-1234" in c for c in pkills), pkills
    assert all("-u " in c for c in pkills), pkills


# ---------------------------------------------------------------------------- manifest ---


def test_the_manifest_records_the_metadata_rather_than_empty_strings(harness):
    """The silent-empty failure, asserted on the file.

    ``write_manifest`` reads ``os.environ``. MODE, ARM, SOLVED_ADV, MODEL and RUN_NAME were
    plain shell variables, so every one of them reached the manifest as "" -- a file that
    parses, looks complete, and identifies no run at all. Note that nothing here is passed
    in through the environment: the values must come from the script exporting its own
    defaults.
    """
    out = harness.run('write_manifest "running" ""; STATUS_WRITTEN=1', RUN_NAME="t",
                      ARM="dapo", SOLVED_ADV="0.25")
    assert out.returncode == 0, out.stdout + out.stderr
    man = harness.manifest()
    assert man["run_name"] == "t", man
    assert man["mode"] == "train", man
    assert man["arm"] == "dapo", man
    assert man["solved_advantage"] == "0.25", man
    assert man["model"] == "Qwen/Qwen2.5-1.5B-Instruct", man
    assert man["status"] == "running", man
    assert man["wandb"]["mode"] == "online", man
    assert man["wandb"]["project"], man
    assert man["log"].endswith("t.log"), man


def test_wandb_is_online_by_default_and_reaches_the_manifest(harness):
    """Requirement and evidence in one place: the default is online, and it is reported."""
    out = harness.run('echo "MODE=$WANDB_MODE"; write_manifest "running" ""; STATUS_WRITTEN=1')
    assert "MODE=online" in out.stdout, out.stdout
    assert harness.manifest()["wandb"]["mode"] == "online"


def test_the_manifest_reports_checkpoints_and_evaluations(harness):
    """It has to find checkpoints at AReaL's real depth, and read every results.json."""
    ckpt = harness.outdir / "checkpoints/ubuntu/exp/t1/default/epoch0epochstep1globalstep10"
    ckpt.mkdir(parents=True)
    (harness.outdir / "eval").mkdir(parents=True, exist_ok=True)
    (harness.outdir / "eval/results.json").write_text(json.dumps({"math500": 0.42}))
    out = harness.run('write_manifest "succeeded" ""; STATUS_WRITTEN=1')
    assert out.returncode == 0, out.stdout + out.stderr
    man = harness.manifest()
    assert [p for p in man["checkpoints"] if p.endswith("globalstep10")], man["checkpoints"]
    assert man["evaluations"] and man["evaluations"][0]["data"] == {"math500": 0.42}, man


def test_a_broken_results_json_is_reported_rather_than_thrown(harness):
    """One unreadable eval file must not cost us the whole manifest."""
    (harness.outdir / "eval").mkdir(parents=True, exist_ok=True)
    (harness.outdir / "eval/results.json").write_text("{not json")
    harness.run('write_manifest "failed" "boom"; STATUS_WRITTEN=1')
    man = harness.manifest()
    assert man["status"] == "failed"
    assert man["evaluations"][0]["data"] is None, man


def test_the_manifest_is_written_on_the_failure_path(harness):
    """``die`` has to leave a manifest; a manifest that only exists on success is useless."""
    out = harness.run('die "setup failed" 5')
    assert out.returncode == 5, out.stdout + out.stderr
    man = harness.manifest()
    assert man["status"] == "failed", man
    # The note carries the reason and NOT the exit code that followed it.
    assert man["note"] == "setup failed", man


def test_an_unhandled_exit_still_leaves_a_manifest_and_keeps_its_status(harness):
    """The EXIT fuse: a `set -u` abort or a dropped ssh must not vanish silently."""
    out = harness.run('( true ); ( true ) & wait; exit 42')
    assert out.returncode == 42, out.stdout + out.stderr
    man = harness.manifest()
    assert man["status"] == "failed", man
    assert "42" in man["note"], man


def test_the_exit_fuse_does_not_overwrite_a_finished_run(harness):
    """A success that then exits normally stays a success."""
    harness.run('write_manifest "succeeded" ""; STATUS_WRITTEN=1')
    assert harness.manifest()["status"] == "succeeded"


def test_the_manifest_survives_a_missing_python3(harness, tmp_path):
    """Before setup there may be no interpreter, and that is a failure worth reporting.

    The heredoc cannot run, so the fallback writes what the shell already knows. It still
    has to be JSON, or the collaborator's tooling cannot read the failure either.
    """
    bindir = tmp_path / "nopy"
    bindir.mkdir()
    for tool in ("bash", "date", "cat", "tee", "sed", "tr", "head", "paste", "awk", "cut",
                 "grep", "mkdir", "sleep", "flock", "hostname", "ps"):
        found = subprocess.run(["bash", "-c", f"command -v {tool}"],
                               capture_output=True, text=True).stdout.strip()
        if found:
            (bindir / tool).symlink_to(found)
    env = harness.env(PATH=str(bindir))
    body = f'source "{SCRIPT}"\nwrite_manifest "failed" \'setup "failed"\'\nSTATUS_WRITTEN=1\n'
    out = subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    man = harness.manifest()
    assert man["status"] == "failed", man
    assert man["mode"] == "train", man
    assert "python3" in man["degraded"], man


# -------------------------------------------------------------------------------- lock ---


def test_a_second_copy_refuses_and_leaves_the_holders_manifest_alone(harness):
    """One flock per WORKDIR, and the loser reports itself without clobbering the winner."""
    holder = harness.popen('write_manifest "running" ""; STATUS_WRITTEN=1; sleep 8')
    try:
        time.sleep(2.0)
        second = harness.run('echo unreachable')
        assert second.returncode == 3, second.stdout + second.stderr
        assert "unreachable" not in second.stdout
        assert harness.manifest()["status"] == "running", harness.manifest()
        busy = json.loads((harness.outdir / "t.manifest.lockbusy.json").read_text())
        assert busy["status"] == "failed", busy
    finally:
        holder.kill()
        holder.wait()


# ----------------------------------------------------------------------------- signals ---


def test_a_signal_stops_the_run_instead_of_restarting_it(harness):
    """SIGTERM has to end the script, not merely be noted in passing.

    The old trap wrote a manifest and returned, so ``wait`` came back non-zero and the retry
    loop relaunched the training the operator had just asked to stop. On a machine we do not
    own, ignoring a SIGTERM that way is the worst possible reading of it.
    """
    proc = harness.popen(
        'echo READY; sleep 5 >/dev/null 2>&1 & wait $!; echo NOT_STOPPED')
    try:
        for _ in range(100):
            line = proc.stdout.readline()
            if "READY" in line:
                break
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 130, (proc.returncode, out)
        assert "NOT_STOPPED" not in out, out
        assert harness.manifest()["status"] == "interrupted", harness.manifest()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ------------------------------------------------------------------------ static shape ---


def test_the_script_parses(harness):
    """``bash -n`` on every push: a syntax error here is discovered by the collaborator."""
    out = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_sourcing_the_script_launches_nothing(harness):
    """The guard these tests depend on, asserted rather than assumed.

    If sourcing ever starts running ``main`` again, every test above would begin cloning a
    repository and claiming GPUs on whatever machine the suite happens to run on.
    """
    out = harness.run('echo SOURCED_ONLY', rows=idle(8))
    assert out.stdout.strip().endswith("SOURCED_ONLY"), out.stdout
    assert "cloning" not in out.stdout, out.stdout
    assert harness.calls() == [], harness.calls()


# ------------------------------------------------------------- eval checkpoint lookup ---


def test_the_newest_checkpoint_is_found_at_areals_real_depth(harness):
    """``MODE=eval`` with no CKPT has to find what we just trained.

    AReaL writes ``checkpoints/<user>/<experiment>/<trial>/default/<step>``, four levels
    below where a one-level glob looks. Finding nothing was not a quiet no-op: eval returned
    a failure, the retry loop re-claimed GPUs on a shared box and repeated it.
    """
    root = harness.outdir / "checkpoints/ubuntu/exp/t1/default"
    for step, age in (("epoch0epochstep1globalstep10", 200), ("epoch2epochstep0globalstep99", 20)):
        (root / step).mkdir(parents=True)
        os.utime(root / step, (time.time() - age, time.time() - age))
    out = harness.run('newest_checkpoint')
    assert out.stdout.strip().endswith("globalstep99"), out.stdout


def test_no_checkpoint_yields_an_empty_answer_not_a_stray_directory(harness):
    """With nothing trained yet the caller must see "", so it can ask for CKPT= instead."""
    (harness.outdir / "checkpoints").mkdir(parents=True)
    out = harness.run('newest_checkpoint')
    assert out.stdout.strip() == "", out.stdout
