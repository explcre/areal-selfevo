"""One-sample end-to-end pass: one task, one scaffold, one rollout batch, one reward,
one update, artifacts written and read back.

The point of this file is to prove each stage FIRED rather than defaulting silently.
Every assertion is on observable state -- a value in the returned record or a value read
back off disk -- never on a function that returned True.

It runs on CPU with the deterministic stub client and touches no GPU.
"""

from __future__ import annotations

import pytest

from ornith_repro.buffer import TaskBuffer
from ornith_repro.guards import GuardViolation
from ornith_repro.judges import Judges
from ornith_repro.llm import StubClient
from ornith_repro.loop import (
    OrnithConfig,
    read_artifacts,
    run_iteration,
    verify_provenance,
    write_artifacts,
)
from ornith_repro.types import RolloutOutcome, Scaffold, Task


def _fixture(k=8, success_rate=0.25, abort_rate=0.0, seed=0):
    """Build one task, one scaffold, and k stub rollouts."""
    task = Task(
        task_id="q-001",
        text="compute the number of distinct primes below the given bound",
        family="number_theory",
        length_bin="short",
    )
    scaffold = Scaffold(
        scaffold_id="h-001",
        instructions="compute the number of distinct primes below the given bound step by step",
        tools=("python",),
        grader_kind="exact",
    )
    client = StubClient(abort_rate=abort_rate, success_rate=success_rate)
    rollouts = [client.generate(f"{task.text}||{scaffold.instructions}", 256, seed + i)
                for i in range(k)]
    return task, scaffold, rollouts, client


def test_one_sample_end_to_end_every_stage_fires_and_artifact_round_trips(tmp_path):
    cfg = OrnithConfig(base_model="stub/deterministic-v1", k_rollouts=8, sigma=0.15)
    task, scaffold, rollout_texts, client = _fixture()
    buffer = TaskBuffer()
    judges = Judges()

    assert len(buffer) == 0, "precondition: buffer starts empty"

    applied: list[str] = []
    result = run_iteration(
        cfg, task, scaffold, rollout_texts, buffer, judges,
        updater=lambda stage, adv: applied.append(stage),
    )

    # --- stage 3: the rollouts actually ran, and the model was actually called -------
    assert len(client.calls) == 8, "the stub client was called once per rollout"
    assert len(result.task_record.rollouts) == 8
    assert result.stages_fired == ["rollout", "task", "harness"]

    # --- p_hat is a real measurement, not a default ---------------------------------
    tr = result.task_record
    assert tr.n_valid == 8 and tr.n_aborted == 0
    n_succ = sum(1 for r in tr.rollouts if r.outcome is RolloutOutcome.SUCCESS)
    assert tr.p_hat == pytest.approx(n_succ / 8), "p_hat must equal the observed rate"

    # --- stage 1: every factor of R_task was computed, and none defaulted ------------
    assert 0.0 < tr.V <= 1.0
    assert 0.0 < tr.D <= 1.0
    assert tr.empty_buffer is True, "first task sees an empty buffer"
    assert tr.N == 1.0, "empty-buffer novelty is 1.0, not 0.0 (guard G6)"
    assert tr.R_task == pytest.approx(tr.V * tr.D * tr.N), "R_task = V*D*N exactly"
    assert tr.R_task > 0.0, "a non-degenerate first iteration must not be silently zero"

    # --- stage 2: the harness reward was computed ------------------------------------
    hr = result.harness_record
    assert hr.R_harness == pytest.approx(hr.C * hr.F * hr.H), "R_harness = C*F*H exactly"
    assert hr.C > 0.0, "scaffold shares wording with the task, so alignment is positive"

    # --- the update fired once per stage, in the configured order -------------------
    assert applied == list(cfg.stage_order)
    assert result.updates_applied == list(cfg.stage_order)

    # --- the buffer was written AFTER scoring (A8 / G5) ------------------------------
    assert len(buffer) == 1 and buffer.texts() == [task.text]

    # --- artifacts written and read back, with provenance recomputed ----------------
    path = write_artifacts(tmp_path / "iter.jsonl", result)
    rows = read_artifacts(path)
    assert len(rows) == 1
    row = rows[0]
    verify_provenance(row)  # raises if any stage did not run on the inputs it claims

    assert row["task_record"]["R_task"] == pytest.approx(tr.R_task)
    assert row["harness_record"]["R_harness"] == pytest.approx(hr.R_harness)
    assert row["stages_fired"] == ["rollout", "task", "harness"]
    assert len(row["task_record"]["rollouts"]) == 8


def test_provenance_detects_a_fabricated_artifact(tmp_path):
    """G7: a record edited after the fact must fail provenance recomputation.

    This is the mutation test for the provenance guard. Without it, a stage that never
    ran could leave a plausible-looking artifact behind.
    """
    cfg = OrnithConfig(base_model="stub/deterministic-v1")
    task, scaffold, rollout_texts, _ = _fixture()
    result = run_iteration(cfg, task, scaffold, rollout_texts, TaskBuffer(), Judges())
    rows = read_artifacts(write_artifacts(tmp_path / "iter.jsonl", result))

    verify_provenance(rows[0])  # unmodified row verifies

    rows[0]["task_record"]["R_task"] = 0.99  # tamper
    with pytest.raises(GuardViolation, match="task-stage provenance mismatch"):
        verify_provenance(rows[0])


def test_second_iteration_sees_a_nonempty_buffer_and_novelty_drops(tmp_path):
    """Observable state: after one insert, novelty is measured against a real buffer."""
    cfg = OrnithConfig(base_model="stub/deterministic-v1")
    buffer, judges = TaskBuffer(), Judges()

    task1, scaffold, rt1, _ = _fixture(seed=0)
    r1 = run_iteration(cfg, task1, scaffold, rt1, buffer, judges)
    assert r1.task_record.empty_buffer is True and r1.task_record.N == 1.0

    task2 = Task(task_id="q-002", text=task1.text + " and also list them",
                 family="number_theory", length_bin="short")
    _, _, rt2, _ = _fixture(seed=100)
    r2 = run_iteration(cfg, task2, scaffold, rt2, buffer, judges)

    assert r2.task_record.empty_buffer is False, "buffer is no longer empty"
    assert r2.task_record.N < 1.0, "a near-duplicate must score less than maximal novelty"
    assert len(buffer) == 2
