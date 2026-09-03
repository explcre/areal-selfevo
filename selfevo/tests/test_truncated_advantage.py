"""The truncated-rollout advantage policy: what each setting does, and that it fires.

A rollout that hits the generation cap stops mid-derivation. The verifier is handed a
response with no answer in it, finds nothing to parse and returns 0 -- the same 0 a
confidently wrong answer earns, and after ``reward_bias`` and ``reward_scaling`` the same
negative number. ``PPOActorConfig.truncated_advantage`` decides what happens next.

These tests pin all three settings against values computed by hand from the group
normalisation formula rather than from the code under test, so a change to that code
cannot move the expectation with it.
"""

import pytest

torch = pytest.importorskip("torch")

from selfevo.tests.conftest import G, make_actor  # noqa: E402

B, T, PROMPT, CAP = 8, 10, 2, 4
SQRT = torch.tensor(0.0).sqrt  # placeholder to keep torch referenced at import


def batch(rewards, resp_lens):
    """A batch whose rows have DIFFERENT response lengths, which conftest's cannot express.

    Args:
        rewards: Per-row reward in {0, 1}, length ``B``.
        resp_lens: Per-row response length in tokens, length ``B``. A row with
            ``resp_len >= CAP`` is what ``_truncated_rows`` calls truncated.

    Returns:
        The tensor dict ``_compute_advantages`` consumes. ``attention_mask`` is kept
        consistent with ``loss_mask`` -- prompt then response, then padding -- because the
        outcome reward is placed at ``attention_mask.sum(-1) - 2`` and would land outside
        the loss region of a short row otherwise.
    """
    loss_mask = torch.zeros(B, T)
    attention_mask = torch.zeros(B, T)
    for i, ln in enumerate(resp_lens):
        loss_mask[i, PROMPT : PROMPT + ln] = 1.0
        attention_mask[i, : PROMPT + ln] = 1.0
    return {
        "input_ids": torch.randint(0, 100, (B, T)),
        "loss_mask": loss_mask,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "old_logp": torch.zeros(B, T),
        "ref_logp": torch.zeros(B, T),
        "logprobs": torch.zeros(B, T),
        "attention_mask": attention_mask,
    }


def actor(policy):
    """An actor configured like the live A0 run, with the cap small enough to hit."""
    from areal.utils.data import TrajBatchMeta

    a = make_actor(reward_bias=-0.5)
    a.config.max_new_tokens = CAP
    a.config.truncated_advantage = policy
    a.truncated_advantage = policy
    a._meta = TrajBatchMeta(n_trajs=B, traj_group_sizes=[G, G], traj_seqlens=[T] * B)
    return a


def seq_adv(a, rewards, resp_lens):
    """Per-row advantage. Constant across a row's tokens under this configuration."""
    data = batch(rewards, resp_lens)
    adv = a._compute_advantages(data, a._meta)["advantages"]
    mask = torch.roll(data["loss_mask"], shifts=-1, dims=-1)
    return (adv * mask).sum(-1) / mask.sum(-1)


# group 0: two right two wrong, one of each truncated.
# group 1: one right three wrong, the last one truncated.
REWARDS = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
LENS = [2, CAP, 2, CAP, 2, 2, 2, CAP]
TRUNC = [False, True, False, True, False, False, False, True]


def test_default_is_keep_and_scores_a_truncation_like_a_wrong_answer():
    """The shipped default reproduces the historical update: no row is special."""
    a = actor("keep")
    assert a.config.truncated_advantage == "keep"
    adv = seq_adv(a, REWARDS, LENS)
    # Group 0 scaled rewards are [.5, .5, -.5, -.5]; mean 0, unbiased std sqrt(1/3)/2.
    expect0 = 0.5 / (torch.tensor(4 * 0.25 / 3).sqrt() + 1e-5)
    assert torch.allclose(adv[:4], torch.tensor([1.0, 1.0, -1.0, -1.0]) * expect0, atol=1e-4)
    # The truncated row carries the SAME magnitude as the terminating wrong row.
    assert torch.isclose(adv[3], adv[2], atol=1e-6)


def test_zero_silences_the_truncated_row_and_moves_nothing_else():
    """'zero' is exactly one change: the unverified row stops voting."""
    keep = seq_adv(actor("keep"), REWARDS, LENS)
    zero = seq_adv(actor("zero"), REWARDS, LENS)
    trunc = torch.tensor(TRUNC)
    assert torch.equal(zero[trunc], torch.zeros(int(trunc.sum())))
    # Bit-identical, not merely close: the baseline its siblings are measured against is
    # untouched, so 'keep' and 'zero' differ in one thing and a test that allowed drift
    # here would pass for an implementation that re-normalised.
    assert torch.equal(zero[~trunc], keep[~trunc])


def test_exclude_also_moves_the_baseline_off_the_unverified_rows():
    """'exclude' additionally drops the row from its group's mean and std."""
    excl = seq_adv(actor("exclude"), REWARDS, LENS)
    trunc = torch.tensor(TRUNC)
    assert torch.equal(excl[trunc], torch.zeros(int(trunc.sum())))
    # Group 0 survivors are [.5, -.5]: mean 0, unbiased std over two rows sqrt(0.5).
    s0 = torch.tensor(0.5).sqrt()
    assert torch.allclose(
        excl[[0, 2]], torch.tensor([0.5, -0.5]) / (s0 + 1e-5), atol=1e-5
    )
    # And it is genuinely a different baseline from 'keep', on rows that were not touched.
    keep = seq_adv(actor("keep"), REWARDS, LENS)
    assert not torch.allclose(excl[[0, 2]], keep[[0, 2]], atol=1e-3)


def test_a_wholly_truncated_group_gets_zero_not_the_raw_scaled_reward():
    """The mask-everything path is the one that can hand back an unnormalised reward.

    ``Normalization`` returns the raw tensor when its mask sums to zero, and its group
    mean over an empty group is 0. Either would put +-reward_scaling/2 straight into the
    loss for rows nobody verified, which is worse than the bug being fixed.
    """
    lens = [2, 2, 2, 2, CAP, CAP, CAP, CAP]  # group 1 entirely truncated
    for policy in ("zero", "exclude"):
        adv = seq_adv(actor(policy), REWARDS, lens)
        assert torch.equal(adv[4:], torch.zeros(4)), policy


@pytest.mark.parametrize("policy", ["zero", "exclude"])
def test_every_row_truncated_still_yields_zero(policy):
    """With no survivors there is no baseline, and no row may carry a raw reward.

    ``Normalization`` returns its input UNNORMALISED when the mask sums to zero, so this
    is the batch on which an implementation that relied on the mask alone would put
    +-reward_scaling/2 into the loss for rows nobody verified.
    """
    adv = seq_adv(actor(policy), REWARDS, [CAP] * B)
    assert torch.equal(adv, torch.zeros(B))


def test_the_instrument_disagrees_with_no_eos_ratios_on_the_same_batch():
    """Why a new metric exists rather than a reading of the old one.

    ``no_eos_ratios`` compares a sequence length to the PADDED BATCH WIDTH, so it flags
    whichever rows happen to be longest and nothing else. On this batch it reports three
    of eight truncated by coincidence of shape; make the batch one column wider and it
    reports none, while the cap-based predicate is unchanged.
    """
    from areal.trainer.ppo.actor import _truncated_rows

    data = batch(REWARDS, LENS)
    am = data["attention_mask"]
    no_eos = (am.sum(-1) == am.shape[-1]).float()
    trunc = _truncated_rows(data["loss_mask"], CAP).float()
    assert torch.equal(trunc, torch.tensor(TRUNC, dtype=torch.float32))
    assert not torch.equal(no_eos, trunc)
    assert float(no_eos.sum()) == 0.0  # nothing reaches the padded width at all


class _ReachedEngine(Exception):
    """Raised in place of the engine call, to mark that the logging block completed."""


def test_the_rate_is_logged_on_every_step(monkeypatch):
    """A run that does not act on truncation must still be able to report it.

    Driven through ``_ppo_update``, the method that actually emits the statistic, with the
    engine hand-off replaced by a sentinel. Asserting on ``_compute_advantages`` instead
    would have been the weaker test that let a NameError into the emitting method: every
    unit test passed while the real first step could not have run.
    """
    import areal.trainer.ppo.actor as actor_mod
    from areal.utils import stats_tracker

    def boom(*_a, **_k):
        raise _ReachedEngine

    monkeypatch.setattr(actor_mod, "stage_batch_for_engine", boom)
    stats_tracker.export_all(reset=True)
    a = actor("keep")
    data = batch(REWARDS, LENS)
    data = a._compute_advantages(data, a._meta)
    with pytest.raises(_ReachedEngine):
        a._ppo_update(data)
    exported = stats_tracker.export_all(reset=True)
    key = next(k for k in exported if k.endswith("truncated_ratios/avg"))
    assert exported[key] == pytest.approx(3 / 8)
    # And the arm says which policy produced it, from the actor rather than the config file.
    arm = next(k for k in exported if k.endswith("truncated_advantage"))
    assert exported[arm] == 0.0


def test_every_emitting_method_binds_the_predicate_it_reports():
    """The defect this file was written around, as a standing guard.

    ``truncated_ratios`` was first emitted from ``_ppo_update`` while ``truncated`` was
    bound only in ``_compute_advantages``: a NameError on the first real step that no test
    driving ``_compute_advantages`` could see. Any function that reports the rate must
    compute it.
    """
    import ast
    import inspect

    import areal.trainer.ppo.actor as actor_mod

    src = inspect.getsource(actor_mod)
    tree = ast.parse(src)
    emitting = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "truncated_ratios" not in seg:
            continue
        emitting.append(node.name)
        bound = {
            t.id
            for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        assert "truncated" in bound, f"{node.name} reports the rate without computing it"
    assert emitting, "nothing emits truncated_ratios any more"


@pytest.mark.parametrize("bad", ["", "Keep", "drop", "true", None])
def test_a_misspelt_policy_is_refused_rather_than_defaulted(bad):
    """A typo must not produce an arm that reports the fix and runs the baseline."""
    from areal.api.cli_args import PPOActorConfig

    with pytest.raises((ValueError, TypeError)):
        PPOActorConfig(path="unused", truncated_advantage=bad)
