"""Tests for cluster-granularity routing.

The tier exists because a live measurement showed 57.4% of GRPO groups are RL-silent against
the token rule's 1.7%. These tests pin the routing TABLE by cluster, because a router that
partitions correctly and then assigns the wrong signal is indistinguishable from one that
works, if you only check the partition.
"""
from __future__ import annotations

import pytest

from selfevo.routing.base import RoutingContext, TrainingMode
from selfevo.routing.cluster import ClusterAssignment, ClusterRouter, silence_cluster_key


def ctx(p, teacher=True, g=8):
    return RoutingContext(solve_rate=p, group_size=g, has_teacher=teacher)


def mode_of(d):
    return next(iter(d.weights))


def test_each_cluster_gets_its_own_signal():
    """The three clusters need OPPOSITE responses; pin each one, not just the partition."""
    r = ClusterRouter()
    a = r.route_batch([ctx(0.5), ctx(0.0), ctx(1.0)])
    assert mode_of(a.decisions[0]) == TrainingMode.RL, "informative must get RL"
    # SFT, not DISTILL: base.py states hard distillation is deliberately absent and
    # SFT with a teacher-sourced target is the supported path. This assertion used to
    # pin DISTILL, which an audit measured as a dead branch -- units paid full cost
    # and never learned, and this test enshrined it.
    assert mode_of(a.decisions[1]) == TrainingMode.SFT, "unsolved must get a teacher target"
    assert mode_of(a.decisions[2]) == TrainingMode.SKIP, "solved must not be trained further"


def test_solved_is_skipped_not_sft():
    """Adding a supervised gradient to an already-correct policy burns entropy for nothing.

    This asymmetry is the reason the two silent sides are separated at all, so a router that
    collapsed them would pass a partition test and fail here.
    """
    a = ClusterRouter().route_batch([ctx(1.0)])
    assert mode_of(a.decisions[0]) != TrainingMode.SFT
    assert mode_of(a.decisions[0]) == TrainingMode.SKIP


def test_teacher_mode_without_a_teacher_is_refused():
    """Routing to distillation with no teacher trains on a target that does not exist."""
    a = ClusterRouter().route_batch([ctx(0.0, teacher=False)])
    assert mode_of(a.decisions[0]) == TrainingMode.SKIP
    assert "needs a teacher" in a.decisions[0].reason


def test_every_unit_is_assigned_exactly_one_cluster():
    ctxs = [ctx(p) for p in (0.0, 0.125, 0.5, 0.875, 1.0)]
    a = ClusterRouter().route_batch(ctxs)
    assert len(a.decisions) == len(ctxs) == len(a.cluster_of)
    assert sum(a.sizes.values()) == len(ctxs)
    assert abs(sum(a.fractions.values()) - 1.0) < 1e-9


def test_fractions_are_reported_per_cluster_and_never_summed():
    """A combined 'silent fraction' would hide which silent side dominates."""
    a = ClusterRouter().route_batch([ctx(0.0), ctx(0.0), ctx(1.0), ctx(0.5)])
    assert a.fractions["unsolved"] == 0.5
    assert a.fractions["solved"] == 0.25
    assert a.fractions["informative"] == 0.25


def test_an_unnamed_cluster_falls_back_and_says_so():
    """A silently unrouted cluster must not look like a deliberate decision."""
    r = ClusterRouter(policy={"informative": TrainingMode.RL})
    a = r.route_batch([ctx(0.0), ctx(0.5)])
    assert mode_of(a.decisions[0]) == TrainingMode.SKIP
    assert "fell back" in a.decisions[0].reason or "fallback" in a.basis
    assert "fallback" in a.basis


def test_a_supplied_partition_replaces_the_derived_one():
    """How a learned or MEDS-style clustering is compared against the derived split."""
    r = ClusterRouter(policy={"a": TrainingMode.RL, "b": TrainingMode.SKIP},
                      key_fn=lambda c: "a" if c.solve_rate > 0.5 else "b")
    a = r.route_batch([ctx(0.9), ctx(0.1)])
    assert mode_of(a.decisions[0]) == TrainingMode.RL
    assert mode_of(a.decisions[1]) == TrainingMode.SKIP
    assert "caller-supplied" in a.basis


def test_empty_batch_is_a_state_not_an_error():
    a = ClusterRouter().route_batch([])
    assert a.decisions == () and a.sizes == {} and a.fractions == {}


def test_route_single_matches_route_batch():
    """The Router protocol path and the batch path must not diverge."""
    r = ClusterRouter()
    c = ctx(0.0)
    assert mode_of(r.route(c)) == mode_of(r.route_batch([c]).decisions[0])


def test_invalid_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown mode"):
        ClusterRouter(policy={"informative": "telepathy"})
    with pytest.raises(ValueError, match="default_mode"):
        ClusterRouter(default_mode="telepathy")
    with pytest.raises(ValueError, match="threshold"):
        ClusterRouter(threshold=1.5)


def test_assignment_rejects_inconsistent_bookkeeping():
    """Sizes that disagree with the decisions mean a unit was dropped or double-counted."""
    from selfevo.routing.base import RoutingDecision
    d = (RoutingDecision({TrainingMode.RL: 1.0}),)
    with pytest.raises(ValueError, match="cluster label"):
        ClusterAssignment(decisions=d, cluster_of=(), sizes={}, fractions={}, basis="x")
    with pytest.raises(ValueError, match="dropped or double-counted"):
        ClusterAssignment(decisions=d, cluster_of=("a",), sizes={"a": 5},
                          fractions={"a": 1.0}, basis="x")


def test_silence_key_is_derived_from_the_estimator_not_chosen():
    """Group size changes which solve rates are attainable, so the key must depend on it."""
    assert silence_cluster_key(ctx(0.0, g=8)) == "unsolved"
    assert silence_cluster_key(ctx(1.0, g=8)) == "solved"
    assert silence_cluster_key(ctx(0.5, g=8)) == "informative"


# ------------------------------------------------- audit-driven regressions (12/40 survived)


def test_every_teacher_mode_is_refused_without_a_teacher_not_just_one():
    """A mutant restricting the check to one mode survived: only DISTILL was ever tested."""
    r = ClusterRouter(policy={"unsolved": TrainingMode.SFT, "informative": TrainingMode.RL,
                              "solved": TrainingMode.SKIP})
    a = r.route_batch([ctx(0.0, teacher=False)])
    assert mode_of(a.decisions[0]) == TrainingMode.SKIP, "SFT without a teacher must be refused"
    assert a.refused_teacher == 1


def test_route_single_also_refuses_a_missing_teacher():
    """A mutant dropping the refusal from route() survived: the test used teacher=True."""
    assert mode_of(ClusterRouter().route(ctx(0.0, teacher=False))) == TrainingMode.SKIP


def test_cluster_of_records_each_unit_actual_cluster():
    """A mutant labelling every unit 'x' survived: only len(cluster_of) was checked."""
    a = ClusterRouter().route_batch([ctx(0.5), ctx(0.0), ctx(1.0)])
    assert a.cluster_of == ("informative", "unsolved", "solved")


def test_a_non_default_default_mode_is_honoured():
    """A mutant hardcoding SKIP survived: every fallback test used the default."""
    r = ClusterRouter(policy={"informative": TrainingMode.RL}, default_mode=TrainingMode.RL)
    assert mode_of(r.route_batch([ctx(1.0)]).decisions[0]) == TrainingMode.RL


def test_the_key_depends_on_group_size():
    """A mutant hardcoding G=8 survived: all three assertions used g=8.

    At G=2 the only attainable rates are 0, 0.5 and 1, so a rate of 0.125 is not reachable
    and the classification must still be driven by the group size it was given.
    """
    assert silence_cluster_key(ctx(0.5, g=2)) == "informative"
    assert silence_cluster_key(ctx(0.0, g=2)) == "unsolved"
    assert silence_cluster_key(ctx(0.0, g=16)) == "unsolved"
    assert silence_cluster_key(ctx(1.0, g=16)) == "solved"


def test_threshold_zero_is_rejected():
    """At 0.0 every unanimous group reads as informative and the split silently vanishes."""
    with pytest.raises(ValueError, match="silently disappears|must be in"):
        ClusterRouter(threshold=0.0)


def test_basis_states_the_threshold_is_inert_rather_than_implying_it_worked():
    """The smallest non-zero I_RL is 0.64-0.68, so the 0.1 default separates nothing."""
    a = ClusterRouter().route_batch([ctx(0.5), ctx(0.0)])
    assert "INERT" in a.basis
    assert "was the group unanimous" in a.basis


def test_mode_counts_report_what_was_trained_not_how_it_partitioned():
    """With no teacher those differ completely, and only the histogram shows it."""
    a = ClusterRouter().route_batch([ctx(0.0, teacher=False), ctx(0.0, teacher=False),
                                     ctx(0.5, teacher=False), ctx(1.0, teacher=False)])
    assert a.fractions["unsolved"] == 0.5, "the PARTITION still has half unsolved"
    assert a.mode_counts.get(TrainingMode.SFT, 0) == 0, "but nothing was actually trained on a teacher"
    assert a.mode_counts[TrainingMode.SKIP] == 3
    assert a.refused_teacher == 2


def test_assignment_rejects_the_bookkeeping_an_audit_could_smuggle_through():
    from selfevo.routing.base import RoutingDecision
    d = (RoutingDecision({TrainingMode.RL: 1.0}),)
    with pytest.raises(ValueError):                       # empty sizes no longer short-circuits
        ClusterAssignment(decisions=d, cluster_of=("a",), sizes={}, fractions={}, basis="x")
    with pytest.raises(ValueError, match="not 1"):
        ClusterAssignment(decisions=d, cluster_of=("a",), sizes={"a": 1},
                          fractions={"a": 0.5}, basis="x")
    with pytest.raises(ValueError, match="keys"):
        ClusterAssignment(decisions=d, cluster_of=("a",), sizes={"a": 1},
                          fractions={"zzz": 1.0}, basis="x")
    with pytest.raises(ValueError, match="basis"):
        ClusterAssignment(decisions=d, cluster_of=("a",), sizes={"a": 1},
                          fractions={"a": 1.0}, basis="")
