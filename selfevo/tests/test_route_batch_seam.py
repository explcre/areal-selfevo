"""Tests for the batched routing seam in PPOActor._route_groups.

route_batch had no caller outside its own single-context wrapper, which made every
partitioning router a no-op: ClusterRouter.route IS route_batch([ctx]), so a per-unit seam
gives each cluster exactly one member. These tests pin that the batch path is taken, that the
fallback still works, and that a length mismatch is refused rather than silently misaligning
every group with another group's decision.
"""
import pytest

from areal.trainer.ppo.actor import route_all


class _BatchRouter:
    """Router exposing route_batch, recording how it was called."""

    def __init__(self, n_out=None):
        self.batch_calls = []
        self.single_calls = 0
        self._n_out = n_out

    def route_batch(self, contexts):
        self.batch_calls.append(list(contexts))
        n = len(contexts) if self._n_out is None else self._n_out
        return type("A", (), {"decisions": [_D(f"b{i}") for i in range(n)]})()

    def route(self, ctx):
        self.single_calls += 1
        return _D("single")


class _SingleRouter:
    """Router with no route_batch, exercising the fallback."""

    def __init__(self):
        self.calls = []

    def route(self, ctx):
        self.calls.append(ctx)
        return _D(f"s{len(self.calls)}")


class _D:
    def __init__(self, tag):
        self.weights = {tag: 1.0}
        self._tag = tag

    def argmax(self):
        return self._tag


def _drive(router, n):
    """Drive the REAL seam. Calling route_all rather than restating its logic is the point:
    an earlier version of this file re-implemented the control flow, so every mutation to the
    shipped function survived every test here."""
    contexts = [object() for _ in range(n)]
    decisions = route_all(router, contexts)
    return [d.argmax() for d in decisions], contexts


def test_a_batching_router_is_called_once_with_every_context():
    """The whole point: a partitioning router must see the batch, not one unit at a time."""
    r = _BatchRouter()
    modes, contexts = _drive(r, 5)
    assert len(r.batch_calls) == 1, "route_batch must be called exactly once per batch"
    assert r.batch_calls[0] == contexts, "route_batch must receive every context"
    assert r.single_calls == 0, "the per-unit path must not also run"
    assert modes == ["b0", "b1", "b2", "b3", "b4"]


def test_a_router_without_route_batch_still_works():
    """The fallback must survive: most routers implement only route."""
    r = _SingleRouter()
    modes, _ = _drive(r, 3)
    assert len(r.calls) == 3, "one call per unit, no more"
    assert modes == ["s1", "s2", "s3"]


@pytest.mark.parametrize("n_out", [4, 6, 0])
def test_a_length_mismatch_is_refused_not_silently_misaligned(n_out):
    """Too few, too many, or none: every mismatch must raise.

    A short list would zip away the tail; a long one would attach group i to group j's mode.
    Both are invisible downstream, so the failure has to happen here.
    """
    with pytest.raises(ValueError, match="decisions are positional"):
        _drive(_BatchRouter(n_out=n_out), 5)


def test_the_matching_length_is_accepted():
    """The guard must not fire on the healthy case."""
    modes, _ = _drive(_BatchRouter(n_out=5), 5)
    assert len(modes) == 5


def test_an_empty_batch_routes_to_nothing_without_error():
    """An empty batch is a real state, not a misuse; route_batch documents this."""
    modes, _ = _drive(_BatchRouter(), 0)
    assert modes == []
