"""Does the silent-channel decomposition satisfy its own identity on the real path?

By construction `solved_group_fraction = mean(silent * solved)` and
`unsolved_group_fraction = mean(silent * unsolved)`, and a group cannot be both all-solved
(`min > 0.5`) and all-unsolved (`max <= 0.5`). So

    silent_group_fraction == solved_group_fraction + unsolved_group_fraction

must hold at every step. Measured in the g16 run it does not: the residual exceeds 0.01 at
104 of 116 steps and reaches -0.109. Every "X% of the silent channel is solved" figure this
project has quoted is a ratio of the two implicated metrics, so the violation has to be
explained before those numbers can be used.

These tests drive the real `_compute_advantages` and read what it actually logs, which
separates "the computation is wrong" from "the aggregation or reporting is wrong".

The hypothesis under test is the loss-mask one. `silent` is computed from
`seq_adv = (advantages * loss_mask).sum(-1)`, so a sequence with NO response tokens
contributes exactly 0 whatever its advantage, and a group of such sequences reads as silent
while its raw rewards are mixed -- making it neither all-solved nor all-unsolved, and
inflating `silent` above the sum. Truncation makes this reachable in a real run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from selfevo.tests.conftest import G, make_actor, make_batch, meta  # noqa: E402


def run(rewards, recorder, *, blank_mask_rows=()):
    """Drive the real advantage computation and return the logged decomposition.

    Args:
        rewards: Per-sample rewards, length must be a multiple of G.
        recorder: The Recorder fixture.
        blank_mask_rows: Row indices whose loss_mask is zeroed, simulating a response that
            contributes no tokens (full truncation).
    """
    batch = make_batch(list(rewards))
    for i in blank_mask_rows:
        batch["loss_mask"][i] = 0.0
    make_actor(None)._compute_advantages(batch, meta())
    return (
        recorder.get("silent_group_fraction"),
        recorder.get("solved_group_fraction"),
        recorder.get("unsolved_group_fraction"),
        recorder.get("unclassified_group_fraction"),
    )


ALL_SOLVED_AND_ALL_WRONG = [1, 1, 1, 1, 0, 0, 0, 0]
MIXED_AND_SOLVED = [1, 1, 1, 1, 0, 1, 0, 1]


@pytest.mark.parametrize(
    "rewards", [ALL_SOLVED_AND_ALL_WRONG, MIXED_AND_SOLVED, [1] * 8, [0] * 8]
)
def test_the_identity_holds_on_ordinary_batches(rewards, recorder):
    """With every sequence carrying response tokens the decomposition must partition."""
    s, so, u, other = run(rewards, recorder)
    assert s == pytest.approx(so + u + other, abs=1e-6), (s, so, u, other, rewards)
    assert other == pytest.approx(0.0, abs=1e-6), (
        f"no sequence is truncated here, so nothing should be unclassifiable; got {other}")


def test_a_truncated_group_lands_in_the_unclassified_bucket(recorder):
    """The confirmed mechanism, now pinned as a regression.

    Group 0 has MIXED raw rewards, so it is neither all-solved nor all-unsolved. Zeroing its
    loss mask makes every member's masked advantage sum exactly 0, which is what `silent`
    reads. Before the third bucket existed this produced a +0.5 residual, reproducing the
    +0.277 seen in the g16 run; the group must now be counted, not dropped.
    """
    s, so, u, other = run(MIXED_AND_SOLVED[::-1], recorder, blank_mask_rows=range(G))
    resid = s - so - u - other
    assert resid == pytest.approx(0.0, abs=1e-6), (
        f"identity violated by {resid:+.4f}: silent={s} solved={so} unsolved={u} "
        f"unclassified={other}"
    )
    assert other > 0.0, (
        "the truncated group must land in the unclassified bucket, not vanish; "
        f"silent={s} solved={so} unsolved={u} unclassified={other}"
    )
