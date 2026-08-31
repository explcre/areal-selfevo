"""DAPO dynamic sampling, as a baseline arm.

DAPO (arXiv 2503.14476) addresses the same phenomenon this project measures: a GRPO group
whose samples all score alike has every advantage identically zero and contributes nothing.
DAPO's answer is to **discard** those groups and oversample until the batch refills. Ours is
to reuse them. Both act on the same set, so DAPO is the baseline our claim must be measured
against, and it must be the real rule rather than an approximation of it.

The rule is taken from verl's implementation (``recipe/dapo``; the copy read here is
``verl/trainer/config/algorithm.py`` + ``meds_ray_trainer.py`` in the MEDS checkout), which
keeps a prompt when::

    kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()
                        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]

so: keep if the group's reward standard deviation is positive, and keep singleton groups
unconditionally. ``np.std`` there is the population standard deviation (``ddof=0``), and
``torch.std(unbiased=False)`` reproduces it exactly at every group size -- including the two
degenerate ones, where one sample gives ``0.0`` and no samples give ``NaN``. Since
``NaN > 0`` is False and an empty group is not a singleton, an empty group is dropped, which
is also what verl does with it.

For a group of two or more the estimator does NOT change the accept decision: the sample and
population standard deviations are positive together. ``unbiased=False`` matters for the
returned VALUE and for the singleton (where the sample estimator is NaN), not for the split.

verl filters on the per-sample scalar score (``filter_groups.metric``; ``acc`` in the MEDS
run script, alternatively ``seq_reward``/``seq_final_reward``). ``traj["rewards"]`` is
AReaL's per-sample scalar reward from the reward function, one entry per sample of the group,
which is the same quantity as ``seq_reward`` -- and the same as ``acc`` for a binary-correct
math reward. It is NOT ``seq_final_reward``: AReaL applies KL as a loss term rather than
folding it into the reward, so there is no post-KL variant to filter on.

The oversampling half of DAPO comes from AReaL's ``prepare_batch``, but only under the
DEFAULT ``dynamic_bs=false``: there the collector keeps submitting until ``batch_size``
trajectories have been ACCEPTED, so every rejected group is replaced by a freshly generated
one. Under ``dynamic_bs=true`` collection stops after ``batch_size`` ATTEMPTS and rejected
groups are simply missing, which shrinks the batch instead of refilling it and is NOT DAPO.
Measured on the real collector with half the groups unanimous and ``batch_size=8``:
``dynamic_bs=false`` returns 8 groups from 16 attempts, ``dynamic_bs=true`` returns 4 from 8.

Rejections are counted by the executor as ``rollout/rejected``; the extra generation cost is
the ``rollout/rejected__count`` key, because the bare key is a mean of ones and always 1.0.
Unlike verl, AReaL has no ``max_num_gen_batches`` cap, so a dataset on which every group is
unanimous regenerates forever instead of raising.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["dapo_dynamic_sampling", "group_reward_std"]


def _group_rewards(traj: dict[str, Any]) -> torch.Tensor:
    """The group's per-sample rewards as a flat float64 tensor.

    Args:
        traj: A concatenated group as assembled by the workflow executor. ``rewards`` holds
            one entry per sample in the group.

    Returns:
        A 1-D float64 tensor. The cast is load-bearing: integer rewards are legal input and
        ``torch.std`` refuses integer dtypes.

    Raises:
        KeyError: If the trajectory carries no ``rewards`` -- silently accepting such a
            trajectory would turn the baseline into vanilla GRPO without any sign of it.
    """
    if "rewards" not in traj:
        raise KeyError(
            "trajectory has no 'rewards'; DAPO dynamic sampling cannot decide without it, "
            "and defaulting to accept would silently degrade this arm to vanilla GRPO"
        )
    r = traj["rewards"]
    if not isinstance(r, torch.Tensor):
        r = torch.as_tensor(r)
    return r.flatten().to(torch.float64)


def group_reward_std(traj: dict[str, Any]) -> float:
    """Population standard deviation of a group's per-sample rewards.

    Equal to ``np.std(rewards)`` -- the quantity verl compares against zero -- at every group
    size, including ``0.0`` for one sample and ``NaN`` for none.

    Args:
        traj: A concatenated group.

    Returns:
        The population standard deviation, or NaN for an empty group.
    """
    return float(_group_rewards(traj).std(unbiased=False))


def dapo_dynamic_sampling(traj: dict[str, Any]) -> bool:
    """Accept a group only if its rewards vary, per DAPO's dynamic sampling.

    Args:
        traj: A concatenated group.

    Returns:
        True to keep the group. Singleton groups are kept unconditionally, matching verl.
        A NaN standard deviation -- an empty group, or a group with a NaN reward -- fails the
        comparison and is dropped, again matching verl.
    """
    # _group_rewards raises when 'rewards' is absent, and that must not be short-circuited by
    # an earlier .get(): accepting a rewardless trajectory would turn this arm into vanilla
    # GRPO with nothing in the log to show it.
    n_samples = _group_rewards(traj).numel()
    return bool(group_reward_std(traj) > 0.0 or n_samples == 1)
