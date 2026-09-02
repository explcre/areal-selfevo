"""The harness arm as the TRAINER runs it: the real methods, not a re-derivation of them.

These tests bind ``PPOTrainer._harness_init`` / ``_harness_record`` / ``_harness_route``
unbound to a stub carrying only what they read, and drive them in the order ``train()`` calls
them. That matters more than it sounds: a test that reimplemented the loop would pass while
the real one called ``observe`` in the wrong place, and the failure this whole arm exists to
avoid -- two arms that are byte-identical -- is invisible from inside a single run.

The claim under test is the one the experiment rests on: the variant's setting reaches the
dict every rollout task is built from, and the truncation the selector observes is measured
against the budget the batch was ACTUALLY generated under, not against a config constant that
stops tracking the moment the budget moves.
"""

import json

import pytest
import torch

from areal.api.cli_args import GroupRoutingConfig
from areal.trainer.rl_trainer import PPOTrainer
from selfevo.harness.base import VARIANTS


class _Cfg:
    """The two config branches the harness methods read, and nothing else."""

    def __init__(self, group_routing, log_root):
        self.actor = type("A", (), {"group_routing": group_routing})()
        self.stats_logger = type(
            "S",
            (),
            {
                "experiment_name": "harness_unit",
                "trial_name": log_root.name,
                "fileroot": str(log_root),
            },
        )()


class _Trainer:
    """A stub with the real methods bound to it."""

    _harness_init = PPOTrainer._harness_init
    _harness_record = PPOTrainer._harness_record
    _harness_route = PPOTrainer._harness_route
    _harness_apply_budget = PPOTrainer._harness_apply_budget
    _harness_group_routing = PPOTrainer._harness_group_routing

    def __init__(self, group_routing, log_root):
        self.config = _Cfg(group_routing, log_root)


def _batch(lengths, group_size=4):
    """A rollout batch whose rows have the given response lengths, as loss_mask rows."""
    width = max(lengths)
    out, i = [], 0
    while i < len(lengths):
        chunk = lengths[i : i + group_size]
        mask = torch.zeros(len(chunk), width)
        for r, n in enumerate(chunk):
            mask[r, :n] = 1.0
        out.append({"loss_mask": mask, "input_ids": torch.zeros(len(chunk), width)})
        i += group_size
    return out


def _kwargs():
    """What ``examples/math/gsm8k_rl.py`` builds and passes into ``train()``."""
    return {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1536,
        "max_completion_tokens": 1024,
    }


TREATMENT = dict(
    harness_variants=["gen96", "gen160", "gen256"],
    harness_selector="truncation_step_limit",
)


def test_the_opening_budget_replaces_the_yaml_cap(tmp_path):
    """Step 0 already generates under the declared variant, not under gconfig."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    assert kw["max_completion_tokens"] == 96, "variants[0] is the shortest configured rung"
    assert t._harness.active.name == "gen96"


def test_no_harness_config_leaves_the_kwargs_untouched(tmp_path):
    """A run with no harness arm behaves exactly as it did before this code existed."""
    t = _Trainer(None, tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    t._harness_record(_batch([50] * 8))
    t._harness_route(kw, 0)
    assert kw == _kwargs() and t._harness is None


def test_a_truncated_batch_lengthens_the_budget_and_the_kwargs_follow(tmp_path):
    """The end-to-end claim: batch feature in, a different generation length out."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    assert kw["max_completion_tokens"] == 96
    # Every row reached the 96-token budget: truncated_fraction 1.0 >= 0.5.
    t._harness_record(_batch([96] * 8))
    t._harness_route(kw, 1)
    assert kw["max_completion_tokens"] == 160
    assert t._harness.active.name == "gen160"


def test_an_untruncated_batch_shortens_the_budget(tmp_path):
    """The other threshold, from the top rung down."""
    gr = GroupRoutingConfig(
        harness_variants=["gen256", "gen160", "gen96"],
        harness_selector="truncation_step_limit",
    )
    t = _Trainer(gr, tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    assert kw["max_completion_tokens"] == 256, "config order picks the opening rung"
    t._harness_record(_batch([40] * 8))  # nothing near 256
    t._harness_route(kw, 1)
    assert kw["max_completion_tokens"] == 160, "the LADDER, not the config order, moves"


def test_the_dead_band_leaves_the_budget_exactly_where_it_was(tmp_path):
    """A refusal is inert, and a run that only ever refuses is not a run that never decided."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    t._harness_record(_batch([96] * 8))
    t._harness_route(kw, 1)  # -> gen160
    # 2 of 8 rows at 160: truncated_fraction 0.25, inside (0.05, 0.5).
    t._harness_record(_batch([160, 160, 100, 100, 100, 100, 100, 100]))
    t._harness_route(kw, 2)
    assert kw["max_completion_tokens"] == 160
    assert t._harness_selector.refusals == 1 and t._harness_selector.decisions == 2


def test_truncation_is_measured_against_the_ACTIVE_budget_not_a_constant(tmp_path):
    """The bug this would otherwise have: a moving cap read against a fixed one.

    The same batch of 160-token responses is a fully truncated batch under a 160-token
    budget and a fully terminating one under 256. If the measurement used a config constant
    it would report the same number for both, and the controller would be steering on a
    quantity that had stopped describing the run.
    """
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    t._harness_record(_batch([96] * 8))
    t._harness_route(kw, 1)  # now at gen160
    t._harness_record(_batch([160] * 8))
    assert t._harness_observation["truncated_fraction"] == 1.0
    t._harness_route(kw, 2)  # -> gen256
    assert t._harness.active.name == "gen256"
    t._harness_record(_batch([160] * 8))
    assert t._harness_observation["truncated_fraction"] == 0.0, (
        "the identical batch is not truncated under the larger budget"
    )


def test_one_observation_per_step_and_it_is_consumed(tmp_path):
    """A second route on one observation must not reuse it, and must not decide twice."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    t._harness_record(_batch([96] * 8))
    t._harness_route(kw, 1)
    assert t._harness_observation is None, "the observation is consumed exactly once"
    t._harness_route(kw, 2)
    assert t._harness_selector.decisions == 1, "a step with no rollout takes no decision"
    assert t._harness_selector.repeat_observations == 0


def test_a_batch_with_no_loss_mask_is_refused_not_scored_as_zero_truncation(tmp_path):
    """A silently unmeasurable batch would read as 0.0 and ratchet the budget down."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    with pytest.raises(ValueError, match="no 'loss_mask'"):
        t._harness_record([{"input_ids": torch.zeros(4, 8)}])
    with pytest.raises(ValueError, match="no rows"):
        t._harness_record([])


def test_kwargs_without_the_field_the_rollout_reads_are_refused(tmp_path):
    """Writing a key the workflow never reads is the silent no-op, so it is refused."""
    gr = GroupRoutingConfig(**TREATMENT)
    t = _Trainer(gr, tmp_path)
    with pytest.raises(ValueError, match="max_completion_tokens"):
        t._harness_init({"temperature": 1.0, "max_tokens": 1024})
    with pytest.raises(ValueError, match="no workflow_kwargs"):
        t._harness_init(None)


def test_every_step_writes_one_auditable_record(tmp_path):
    """The primary artefact: the decision log a reclaimed run still leaves behind."""
    t = _Trainer(GroupRoutingConfig(**TREATMENT), tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    for step in range(1, 4):
        t._harness_record(_batch([96] * 8))
        t._harness_route(kw, step)
    rows = [json.loads(line) for line in open(t._harness_log_path)]
    assert [r["global_step"] for r in rows] == [1, 2, 3]
    # The budget oscillates, and that is the mechanism rather than a defect. Every batch
    # here is 96 tokens long. Under a 96-token budget that is fully truncated, so the rule
    # lengthens; under the resulting 160-token budget the SAME batch terminates well inside
    # its cap, so the rule shortens again. A measurement taken against a config constant
    # would have reported "truncated" both times and ratcheted upwards forever.
    assert [r["budget_before"] for r in rows] == [96, 160, 96]
    assert [r["budget_after"] for r in rows] == [160, 96, 160]
    assert [r["move"] for r in rows] == [1, -1, 1]
    assert [round(r["truncated_fraction"], 3) for r in rows] == [1.0, 0.0, 1.0]
    assert all("route/harness_budget" in r["metrics"] for r in rows)
    keys = {frozenset(r["metrics"]) for r in rows}
    assert len(keys) == 1, "the same metric keys on every step, moved or not"


def test_the_control_arm_runs_through_the_same_path(tmp_path):
    """The control differs only in which rule reads the (ignored) observation."""
    gr = GroupRoutingConfig(
        harness_variants=["gen96", "gen160", "gen256"],
        harness_selector="rate_matched_control",
        harness_selector_args={"move_rate": 1.0, "up_share": 1.0, "seed": 1},
    )
    t = _Trainer(gr, tmp_path)
    kw = _kwargs()
    t._harness_init(kw)
    budgets = []
    for step in range(1, 4):
        t._harness_record(_batch([10] * 8))  # nothing truncates; the control ignores it
        t._harness_route(kw, step)
        budgets.append(kw["max_completion_tokens"])
    assert budgets == [160, 256, 160], "up, up, then flipped inward at the top"
    assert t._harness_selector.moves == 3 and t._harness_selector.flips == 1


def test_the_registered_ladder_matches_the_measured_probe():
    """The rungs are the ones the length distribution was measured to separate."""
    assert [VARIANTS[n].settings["max_new_tokens"] for n in ("gen96", "gen160", "gen256")] == [
        96,
        160,
        256,
    ]
