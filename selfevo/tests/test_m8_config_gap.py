"""The M8 config gap, driven through the REAL actor.

GOAL.md M8 records the learned meta-controller as PARTIAL for two reasons, and this file is
about the first: the fix that was validated on CPU -- per-prompt credit measured against that
prompt's OWN earlier deltas -- was **not reachable from config**, so no arm could select it
without editing code. Alongside it land the two seams the arm cannot be MEASURED without: the
correspondence control that shuffles credits across the prompts that earned them, and the
per-decision trace the subset contrast is computed from.

Everything is driven through ``_compute_advantages`` rather than ``_route_groups``, because a
test that calls the helper cannot catch the helper being unreachable -- the exact defect this
file exists to close.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig
from selfevo.routing.base import RoutingDecision, TrainingMode
from selfevo.routing.decision_trace import shuffle_correspondence, trace_path
from selfevo.tests.conftest import G, MIXED, make_actor, make_batch, meta, registered_router

STUB = "_m8_config_gap_stub"


class Alternating:
    """Router that alternates SFT and RL and records every ``observe`` call.

    Alternating rather than fixed so consecutive decisions differ: a router with one mode
    gives every prompt the same arm, and a credit rule could then be broken without any test
    noticing.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.observed: list[dict] = []

    def route(self, ctx):
        """Return SFT on odd sightings and RL on even ones."""
        self.seen.append(ctx.unit_id)
        mode = TrainingMode.SFT if len(self.seen) % 2 else TrainingMode.RL
        return RoutingDecision({mode: 1.0}, reason="stub")

    def observe(self, outcomes) -> None:
        """Record one feedback call."""
        self.observed.append(dict(outcomes))


@pytest.fixture
def stub():
    """Register :class:`Alternating` and yield the list of instances built."""
    made: list[Alternating] = []

    def factory(*_a, **_kw):
        r = Alternating()
        made.append(r)
        return r

    with registered_router(STUB, factory):
        yield made


def _actor(**kw):
    """An actor whose group routing uses the stub router plus the named overrides."""
    return make_actor(
        GroupRoutingConfig(enabled=True, solved_advantage=0.5, router=STUB, **kw)
    )


def _prompts():
    """input_ids where each group's rows share prompt tokens and the two groups differ."""
    ids = torch.zeros(8, 6, dtype=torch.long)
    ids[:G, :2] = torch.tensor([11, 22])
    ids[G:, :2] = torch.tensor([33, 44])
    return ids


def _run(actor, rewards, ids, response_tag):
    """One batch through the real advantage path, holding prompts fixed.

    ``response_tag`` varies the RESPONSE tokens only. Two rollouts of one prompt differ in
    their responses, so a key taken from the whole row would never pair them, and a test that
    reused identical rows would pass against exactly that defect.
    """
    b = make_batch(list(rewards))
    row = ids.clone()
    row[:, 2:] = 100 + response_tag
    b["input_ids"] = row
    return actor._compute_advantages(b, meta())


# ------------------------------------------------------------------ reachable from config ---


def test_the_self_baseline_arm_is_accepted_by_config():
    """The gap itself: before this, the value the CPU study recommended was refused."""
    cfg = GroupRoutingConfig(enabled=True, credit="prompt_self_baseline")
    assert cfg.credit == "prompt_self_baseline"


def test_an_unknown_credit_mode_is_still_refused():
    """Widening the tuple must not turn it into a pass-through."""
    with pytest.raises(ValueError, match="credit must be"):
        GroupRoutingConfig(enabled=True, credit="prompt_self_mean")


def test_the_self_baseline_arm_builds_a_ledger_with_the_self_baseline(stub):
    """Accepting the value is worthless if the actor then runs the old baseline."""
    a = _actor(credit="prompt_self_baseline")
    _run(a, MIXED, _prompts(), 0)
    assert a._selfevo_ledger.baseline == "self_mean"


def test_the_plain_prompt_arm_is_unchanged(stub):
    """The ablation rung below it must keep the raw delta, or the two arms are one arm."""
    a = _actor(credit="prompt")
    _run(a, MIXED, _prompts(), 0)
    assert a._selfevo_ledger.baseline == "last"


def test_the_default_arm_still_builds_no_ledger(stub):
    """Rollback must be exact: credit='batch' behaves as it did before this change."""
    a = _actor(credit="batch")
    _run(a, MIXED, _prompts(), 0)
    assert getattr(a, "_selfevo_ledger", None) is None


def test_the_self_baseline_withholds_the_first_delta_and_counts_it(stub):
    """A prompt's first delta has no earlier delta to centre against.

    Crediting it against zero would hand the first-credited mode the whole common training
    trend -- the bias that made the live arm abandon RL at the exact step credit began
    flowing -- so it is withheld, and the withholding is counted rather than silent.
    """
    a = _actor(credit="prompt_self_baseline")
    ids = _prompts()
    _run(a, [0] * 8, ids, 0)
    _run(a, [1] * 8, ids, 1)                      # first delta for both prompts: withheld
    assert stub[0].observed == [] or not stub[0].observed[0]
    assert a._selfevo_ledger.cold_baseline_skips == 2
    _run(a, [1] * 8, ids, 2)                      # second delta: credited against the first
    credited = [o for o in stub[0].observed if o]
    assert credited, "the self baseline never credited anything"


def test_the_self_baseline_measures_against_the_prompt_and_not_the_batch(stub):
    """A prompt that repeats its usual improvement earns ~0; the batch never enters.

    Both prompts rise by the same amount twice. Under ``"last"`` every credit is positive --
    that is the common trend the router must not be taught -- and under the self baseline the
    second delta matches the first, so the credit is zero.
    """
    ids = _prompts()
    a_raw = _actor(credit="prompt")
    _run(a_raw, [0] * 8, ids, 0)
    _run(a_raw, [1, 1, 0, 0, 1, 1, 0, 0], ids, 1)
    _run(a_raw, [1] * 8, ids, 2)
    raw = [o.value for call in stub[0].observed for o in call.values()]
    assert raw and all(v > 0 for v in raw), raw

    a_self = _actor(credit="prompt_self_baseline")
    _run(a_self, [0] * 8, ids, 0)
    _run(a_self, [1, 1, 0, 0, 1, 1, 0, 0], ids, 1)
    _run(a_self, [1] * 8, ids, 2)
    selfv = [o.value for call in stub[1].observed for o in call.values()]
    assert selfv, "the self baseline credited nothing on the third sighting"
    assert all(abs(v) < 1e-6 for v in selfv), selfv


# ------------------------------------------------------------- the correspondence control ---


def test_the_shuffle_is_refused_on_the_batch_scalar():
    """Permuting one number across the units that all hold it changes nothing.

    A control that could not have failed must not be reportable as one, so this is refused at
    config time rather than run as a silent no-op.
    """
    with pytest.raises(ValueError, match="cannot be run on credit='batch'"):
        GroupRoutingConfig(enabled=True, router="static", credit="batch", credit_shuffle_seed=1)


def test_the_shuffle_moves_the_credits_and_keeps_the_multiset(stub):
    """Same values, different owners. That is the whole control.

    The treatment and the shuffled arm are run on identical data, so any difference in which
    unit holds which credit is the permutation and nothing else.
    """
    ids = _prompts()
    plain = _actor(credit="prompt")
    shuffled = _actor(credit="prompt", credit_shuffle_seed=3)
    # Three sightings, not two. With two pairings a permutation is the identity half the
    # time, and at seed 3 the step-1 draw IS the identity -- reported as `inert`, which is
    # honest but is not what this test is about. The step-2 draw swaps.
    #
    # Prompt 0 moves twice as far as prompt 1 at every sighting, so the two credits differ
    # and a swap is observable; equal credits would make a working shuffle and a broken one
    # indistinguishable.
    for tag, rw in enumerate([[0] * 8, [1, 1, 1, 1, 1, 1, 0, 0], [0] * 8]):
        _run(plain, rw, ids, tag)
        _run(shuffled, rw, ids, tag)
    a = {u: o.value for call in stub[0].observed for u, o in call.items()}
    b = {u: o.value for call in stub[1].observed for u, o in call.items()}
    assert a and b, (a, b)
    assert sorted(a.values()) == pytest.approx(sorted(b.values())), (a, b)
    assert a != b, "the shuffle left every credit on the unit that earned it"


def test_without_a_seed_the_credit_stays_on_the_unit_that_earned_it(stub):
    """The other half of the control, and the half a comparison of two arms cannot see.

    A test that only compares a shuffled arm against an unshuffled one passes when BOTH
    shuffle, because two different permutations still differ from each other and still share a
    multiset. So the unshuffled arm is pinned to exact values instead: group 0's prompt went
    from solve rate 0 to 1 and group 1's from 0 to 0.5, and the credits must land on those two
    units in that order.
    """
    ids = _prompts()
    a = _actor(credit="prompt")
    _run(a, [0] * 8, ids, 0)
    _run(a, [1, 1, 1, 1, 1, 1, 0, 0], ids, 1)
    got = {u: o.value for call in stub[0].observed for u, o in call.items()}
    assert got == pytest.approx({"0:0": 1.0, "0:1": 0.5}), got


def test_shuffle_correspondence_reports_itself_inert_when_it_cannot_fail():
    """One pairing, or a step whose credits are all equal, is not a control."""
    assert shuffle_correspondence([("a", 1.0)], 3, 0)[1] == 1
    assert shuffle_correspondence([("a", 2.0), ("b", 2.0)], 3, 0)[1] == 1
    assert shuffle_correspondence([], 3, 0)[1] == 1


def test_shuffle_correspondence_is_a_permutation():
    """The multiset of credits is preserved exactly; only the owners move."""
    pairs = [(chr(97 + i), float(i)) for i in range(12)]
    out, inert = shuffle_correspondence(pairs, 11, 4)
    assert inert == 0
    assert [p for p, _ in out] == [p for p, _ in pairs]
    assert sorted(v for _, v in out) == sorted(v for _, v in pairs)
    assert [v for _, v in out] != [v for _, v in pairs]


# ------------------------------------------------------------------------ the trace ---------


def test_a_trace_without_a_router_is_refused():
    """The fixed rule never reaches the routing path, so its trace would stay empty."""
    with pytest.raises(ValueError, match="decision_trace_path requires a router"):
        GroupRoutingConfig(enabled=True, decision_trace_path="/tmp/x")


def test_no_trace_path_writes_nothing(stub, tmp_path):
    """The default must leave the filesystem untouched."""
    a = _actor(credit="prompt_self_baseline")
    _run(a, MIXED, _prompts(), 0)
    assert list(tmp_path.iterdir()) == []


def test_the_trace_records_one_decision_per_group_with_its_features(stub, tmp_path):
    """Subset contrast needs the mode BESIDE the features that define the subsets."""
    base = str(tmp_path / "trace")
    a = _actor(credit="prompt_self_baseline", decision_trace_path=base)
    _run(a, MIXED, _prompts(), 0)
    rows = [json.loads(l) for l in trace_path(base).read_text().splitlines()]
    decisions = [r for r in rows if r["kind"] == "decision"]
    assert len(decisions) == 2, rows
    for r in decisions:
        assert r["mode"] in (TrainingMode.SFT, TrainingMode.RL, TrainingMode.SKIP)
        assert 0.0 <= r["solve_rate"] <= 1.0
        assert r["group_size"] == G
        # The seven observability features, without which no subset can be defined on
        # anything but the solve rate.
        assert len(r["features"]) == 7, r["features"]


def test_the_trace_records_the_credits_against_the_units_that_earned_them(stub, tmp_path):
    """A credit record with no unit id could not be joined back to a decision."""
    base = str(tmp_path / "trace")
    a = _actor(credit="prompt", decision_trace_path=base)
    ids = _prompts()
    _run(a, [0] * 8, ids, 0)
    _run(a, [1] * 8, ids, 1)
    rows = [json.loads(l) for l in trace_path(base).read_text().splitlines()]
    credits = [r for r in rows if r["kind"] == "credit"]
    units = {r["unit_id"] for r in rows if r["kind"] == "decision"}
    assert credits, rows
    assert {r["unit_id"] for r in credits} <= units
    assert all(r["value"] > 0 for r in credits), credits
