"""A fresh run must get a fresh tracker identifier, and a reused one must fail loudly.

THE FAILURE THIS FILE EXISTS FOR, observed on A0 on 2026-09-02. The run's W&B id is built
from the experiment name, the trial name and `StatsLoggerConfig.wandb.id_suffix`, whose
default is the CONSTANT `"train"`. Six launches of `a0_math/t1` that day therefore all asked
for the id `a0_math_t1_train`, and with `resume="allow"` each one resumed the last. The
tracker's step counter was already ahead, so every write came back with

    Tried to log to step N that is less than the current step M. Steps must be
    monotonically increasing, so this data will be ignored.

seventy times in four hours, and the run's entire curve -- training metrics and periodic
evaluation points alike -- was discarded. Nothing crashed. Nothing on disk was lost. The only
symptom was the absence of the one artefact this project requires.

WHAT IS AND IS NOT TESTED HERE. The rules are pure functions and are tested as such; the
wiring into `StatsLogger` is tested in `test_run_identity_is_wired_into_the_stats_logger`,
which reads the vendor file rather than importing it, because importing it pulls in wandb,
swanlab, trackio, tensorboardX and torch.distributed on a box where a training run is live.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ri = pytest.importorskip("selfevo.run_identity")

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------- a fresh launch is fresh ----


def test_two_launches_of_the_same_trial_do_not_share_an_identifier():
    """THE DEFECT. The vendor id is a pure function of experiment, trial and a constant."""
    first, _ = ri.resolve_run_id("a0_math", "t1", "train", env={}, token="A")
    second, _ = ri.resolve_run_id("a0_math", "t1", "train", env={}, token="B")
    assert first != second
    assert first.startswith("a0_math_t1_train_")


def test_the_identifier_still_reads_as_the_run_it_belongs_to():
    """A unique id that nobody can recognise in a list of runs is a different problem."""
    run_id, _ = ri.resolve_run_id("a0_math", "t1", "train", env={}, token="TOK")
    assert run_id == "a0_math_t1_train_TOK"


def test_an_absent_suffix_does_not_leave_a_dangling_separator():
    """`id_suffix` is nullable in the vendor config, and `a0_math_t1__TOK` is not a name."""
    assert ri.resolve_run_id("a0_math", "t1", "", env={}, token="TOK")[0] == "a0_math_t1_TOK"


def test_the_launch_token_separates_two_launches_inside_one_second():
    """Time alone is not unique: a crash-relaunch loop restarts inside the same second."""
    same_second = 1788373292.0
    a = ri.launch_token(now=same_second, entropy=101)
    b = ri.launch_token(now=same_second, entropy=102)
    assert a != b
    assert re.fullmatch(r"\d{8}T\d{6}Z_\d{5}", a), a


# ------------------------------------------------------ an intended resume works ----


def test_a_named_identifier_is_used_verbatim_and_marked_as_an_intended_resume():
    """A resumed run is legitimate; forbidding it outright would be the wrong fix.

    AReaL's own recovery relaunches the same trial, and that run's curve should continue
    rather than fork into a second W&B run with half the history.
    """
    run_id, intended = ri.resolve_run_id(
        "a0_math", "t1", "train", env={ri.ENV_RUN_ID: "a0_math_t1_train_ORIGINAL"}
    )
    assert run_id == "a0_math_t1_train_ORIGINAL"
    assert intended is True


def test_an_unnamed_launch_is_not_marked_as_a_resume():
    """The discriminating half: `intended` must not be True by default, or the guard is off."""
    assert ri.resolve_run_id("a0_math", "t1", "train", env={}, token="T")[1] is False


# --------------------------------------------- the collision fails at startup ----


def test_a_fresh_launch_that_landed_on_history_is_refused():
    """The A0 failure exactly: a launch that did not ask to resume came back at step 12000."""
    with pytest.raises(ri.RunIdCollision) as exc:
        ri.assert_id_is_fresh("a0_math_t1_train", 12000, intended_resume=False)
    assert "12000" in str(exc.value)
    assert ri.ENV_RUN_ID in str(exc.value)


def test_a_genuinely_fresh_launch_passes():
    """A guard that always fires guards nothing, and this one sits on every launch."""
    ri.assert_id_is_fresh("a0_math_t1_train_TOK", 0, intended_resume=False)


def test_an_intended_resume_is_allowed_to_land_on_history():
    """The whole point of distinguishing intent: this is the case that must NOT fail."""
    ri.assert_id_is_fresh("a0_math_t1_train_ORIGINAL", 12000, intended_resume=True)


@pytest.mark.parametrize("resumed", [1, 63, 12000])
def test_any_history_at_all_is_a_collision_for_an_unintended_launch(resumed):
    """One dropped step and twelve thousand are the same bug; only the bill differs."""
    with pytest.raises(ri.RunIdCollision):
        ri.assert_id_is_fresh("id", resumed, intended_resume=False)


# ------------------------------------- and the independent check at first write ----


def test_the_first_write_of_a_rewound_resume_is_refused():
    """The case intent cannot cover: resuming on purpose, from a step behind the tracker.

    Deliberately not derived from the startup check -- this one compares the step actually
    about to be WRITTEN with the step the tracker actually sits at, so it fires where the
    other one is silent by design.
    """
    with pytest.raises(ri.RunIdCollision) as exc:
        ri.assert_step_advances("a0_math_t1_train", resumed_step=12000, log_step=0)
    assert "12000" in str(exc.value) and "step 0" in str(exc.value)


def test_a_resume_that_picks_up_where_it_left_off_is_allowed():
    """The tracker merges a write to its current step; only a write BELOW it is discarded."""
    ri.assert_step_advances("id", resumed_step=500, log_step=500)
    ri.assert_step_advances("id", resumed_step=500, log_step=501)


def test_a_brand_new_run_writing_step_zero_is_allowed():
    """The regression this rule invites: a fresh run's first commit really is step 0.

    A rule spelled `log_step > resumed_step` refuses every fresh run at its first write. The
    rule is the tracker's own -- a step LESS THAN the current one is ignored -- and a run with
    no history has no current step to be less than.
    """
    ri.assert_step_advances("id", resumed_step=0, log_step=0)


# ------------------------------------------------------------------- the wiring ----


def test_run_identity_is_wired_into_the_stats_logger():
    """The rules are worthless unless the tracker's id actually comes from them.

    Read from the file rather than imported: importing `areal.utils.stats_logger` pulls in
    wandb, swanlab, trackio, tensorboardX and torch.distributed, and this suite runs on a box
    where a training job owns the GPUs. What is asserted is what the defect was -- the id is
    no longer an f-string of experiment, trial and a constant suffix -- plus the presence of
    both checks on their two paths.
    """
    src = (REPO / "areal" / "utils" / "stats_logger.py").read_text()
    assert 'id=f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}"' not in src
    assert "resolve_run_id(" in src and "id=run_id," in src
    assert "assert_id_is_fresh(" in src
    assert "assert_step_advances(" in src
    # The freshness check must run in `init`, where a refusal costs nothing, and the step
    # check in `commit`, which is the first place the step to be written is known.
    init_body = src.split("    def init(self)", 1)[1].split("\n    def ", 1)[0]
    commit_body = src.split("    def commit(self", 1)[1].split("\n    def ", 1)[0]
    assert "assert_id_is_fresh(" in init_body
    assert "assert_step_advances(" in commit_body
