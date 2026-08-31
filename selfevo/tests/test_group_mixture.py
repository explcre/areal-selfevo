"""Weighted mode MIXTURES, driven through the REAL ``PPOActor._compute_advantages``.

``RoutingDecision.weights`` has always been a ``Mapping[str, float]``, so a router could
always say "train this group 60% by SFT and 40% by RL". The actor threw that away with
``.argmax()`` and the seam took ``modes: list[str]``, so every soft router in the registry
was executed as a hard one. This file is about the path that no longer does that.

Two things are asserted that a helper-only test cannot establish:

* The decision reaches the tensor THROUGH THE ACTOR. A test that calls ``apply_mixtures``
  directly cannot notice ``apply_mixtures`` being unreachable, which is precisely the
  failure mode a config gate introduces -- and the gate is a one-line ``==`` comparison, so
  it is exactly the sort of thing that is silently wrong.
* The REDUCTION property, bit for bit. A pure ``rl`` mixture must leave the advantages not
  merely close to what vanilla GRPO produced but IDENTICAL to it, because rollback runs
  through that case. The comparison is on raw bit patterns rather than ``torch.equal``:
  ``torch.equal`` compares values, and ``-0.0 == 0.0`` is True, so it would accept an
  implementation that turned a negative zero into a positive one.

Fixtures are IMPORTED from :mod:`selfevo.tests.test_group_routing`,
:mod:`selfevo.tests.test_actor_router_seam` and :mod:`selfevo.tests.test_group_apply`
rather than rebuilt: a second copy of "an actor configured like the live runs" drifts from
the first, and the drift is silent.
"""

from __future__ import annotations

import contextlib
import random

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig
from selfevo import compose
from selfevo.integration.group_apply import apply_decisions, apply_mixtures
from selfevo.routing.base import RoutingDecision, TrainingMode

from selfevo.tests.test_actor_router_seam import RecordingRouter  # noqa: E402
from selfevo.tests.test_group_apply import (  # noqa: E402
    WEIGHTS,
    informative,
    masks,
    random_partition,
    silent,
)
from selfevo.tests.test_group_routing import (  # noqa: E402
    G,
    MIXED,
    PROMPT,
    advantages,
    make_actor,
)

STUB = "_mixture_stub"          # own registry key, so this file cannot disturb the seam tests
RL, SFT, SKIP = TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP
MODES = [RL, SFT, SKIP]
W = 0.5                         # solved_advantage for every actor built here
RESPONSE = slice(PROMPT, None)

# Mixtures used wherever a test sweeps: one of each pure case and three genuine blends,
# including an asymmetric one so that swapping the RL and SFT coefficients is visible.
MIXES = [
    {RL: 1.0},
    {SFT: 1.0},
    {SKIP: 1.0},
    {RL: 0.6, SFT: 0.4},
    {SFT: 0.5, SKIP: 0.5},
    {RL: 0.2, SFT: 0.3, SKIP: 0.5},
]


def bits(t: torch.Tensor) -> torch.Tensor:
    """Raw bit patterns of a float32 tensor, so ``-0.0`` and ``0.0`` compare DIFFERENT.

    Args:
        t: A float32 tensor.

    Returns:
        An int32 view of the same storage.

    ``torch.equal`` compares VALUES, and every rollback claim in this repo is stated as
    "bit-identical". Under ``torch.equal`` an implementation that computed ``1.0 * x + 0.0``
    instead of leaving ``x`` alone would pass while flipping the sign bit of every negative
    zero it touched -- invisible to the loss, but not the claim that was made.
    """
    return t.contiguous().view(torch.int32)


class MixtureRouter(RecordingRouter):
    """A Router that returns a fixed WEIGHT MAPPING per group position.

    Subclasses the seam's recording router so the call bookkeeping (``seen``, ``observed``)
    stays in one place; only the decision changes.

    Args:
        mixtures: One ``{mode: weight}`` mapping per group position within a batch, cycled
            if the batch has more groups. Per-POSITION rather than one mapping for the
            whole batch on purpose: with a single mixture, a loop that applied group 0's
            decision to group 1's rows would produce exactly the right tensor and no test
            could see it.
    """

    def __init__(self, *mixtures: dict[str, float]) -> None:
        super().__init__()
        self.mixtures = [dict(m) for m in mixtures]
        self.calls = 0

    def route(self, ctx) -> RoutingDecision:
        """Record the unit and return this position's fixed mixture."""
        self.seen.append(ctx.unit_id)
        mix = self.mixtures[self.calls % len(self.mixtures)]
        self.calls += 1
        return RoutingDecision(weights=mix, reason="mixture stub")


@contextlib.contextmanager
def routing_to(*mixtures: dict[str, float]):
    """Register a :class:`MixtureRouter` under ``STUB`` and restore the registry after.

    Args:
        mixtures: Passed straight to :class:`MixtureRouter`.

    Yields:
        The list of routers the actor actually built, so a test can assert how many times
        it was constructed and what it saw.

    ``compose.ROUTERS`` is module-level state shared with every other test in the process,
    so a test that mutates it without restoring makes an unrelated test fail later, in a
    different file, with no visible connection. Same discipline as the ``stub_router``
    fixture in :mod:`selfevo.tests.test_actor_router_seam`, written as a context manager
    because most tests here need two actors built on different mixtures within one test.
    """
    made: list[MixtureRouter] = []

    def factory(*_a, **_kw):
        r = MixtureRouter(*mixtures)
        made.append(r)
        return r

    previous = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = factory
    try:
        yield made
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = previous


def routed(*mixtures: dict[str, float], decision: str = "mixture", sft_weight: float = W):
    """Advantages from the REAL actor with ``mixtures`` routed per group.

    Args:
        mixtures: One mixture per group position; see :class:`MixtureRouter`.
        decision: ``"mixture"`` or ``"argmax"`` -- the config gate under test.
        sft_weight: ``solved_advantage``, i.e. the magnitude of the SFT component.

    Returns:
        ``(advantages, router)``: the tensor ``_compute_advantages`` produced, and the
        router instance the actor built.
    """
    with routing_to(*mixtures) as made:
        torch.manual_seed(0)
        adv = advantages(
            make_actor(
                GroupRoutingConfig(
                    enabled=True, solved_advantage=sft_weight, router=STUB,
                    decision=decision,
                )
            ),
            MIXED,
        )
    return adv, made[0]


def vanilla() -> torch.Tensor:
    """Advantages with routing switched off entirely -- the rollback reference."""
    torch.manual_seed(0)
    return advantages(make_actor(None), MIXED)


# ============================================================ the reduction property =====
#
# The load-bearing claim: a mixture with all its weight on one mode must be EXACTLY the
# corresponding hard decision. If this does not hold, "mixtures generalise the existing
# modes" is false and the rollback story goes with it.


def test_a_pure_rl_mixture_is_bit_identical_to_no_routing_at_all():
    """Rollback. ``{rl: 1.0}`` must reproduce vanilla GRPO bit for bit, not approximately."""
    got, _ = routed({RL: 1.0})
    assert torch.equal(bits(vanilla()), bits(got)), (vanilla() - got).abs().max()


def test_a_pure_sft_mixture_is_bit_identical_to_the_hard_sft_decision():
    """``{sft: 1.0}`` must write exactly what the argmax path writes, on every group."""
    hard, _ = routed({SFT: 1.0}, decision="argmax")
    soft, _ = routed({SFT: 1.0}, decision="mixture")
    assert torch.equal(bits(hard), bits(soft)), (hard - soft).abs().max()
    # Not vacuous: the write really happened, on the informative group too.
    assert torch.allclose(soft[:, RESPONSE], torch.full_like(soft[:, RESPONSE], W)), soft


def test_a_pure_skip_mixture_is_bit_identical_to_the_hard_skip_decision():
    """``{skip: 1.0}`` must zero the response tokens, exactly as the argmax path does."""
    hard, _ = routed({SKIP: 1.0}, decision="argmax")
    soft, _ = routed({SKIP: 1.0}, decision="mixture")
    assert torch.equal(bits(hard), bits(soft)), (hard - soft).abs().max()
    assert torch.equal(soft[:, RESPONSE], torch.zeros_like(soft[:, RESPONSE])), soft
    # Not vacuous: the informative group carried gradient before this ran.
    assert not torch.equal(soft, vanilla())


@pytest.mark.parametrize("name", list(masks(4, 6)))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("w", [0.0, 0.5, 7.5])
def test_a_one_hot_mixture_equals_the_hard_decision_over_every_mask_and_weight(name, mode, w):
    """The same reduction at the seam, swept over the masks and weights a batch can present.

    The actor tests above see one mask shape. A defect that only appears on a ragged mask or
    at ``sft_weight == 0`` would survive them, and both of those are what a real batch looks
    like.
    """
    lm = masks(4, 6)[name]
    adv, _ = informative()
    hard, hs = apply_decisions(adv, lm, [2, 2], [mode] * 2, sft_weight=w)
    soft, ss = apply_mixtures(adv, lm, [2, 2], [{mode: 1.0}] * 2, sft_weight=w)
    assert torch.equal(bits(soft), bits(hard)), (soft - hard).abs().max()
    assert ss.changed_rows == hs.changed_rows
    assert (ss.n_rows, ss.n_groups) == (hs.n_rows, hs.n_groups)
    assert ss.counts == {m: float(c) for m, c in hs.counts.items()}
    assert ss.mixed_groups == 0


def test_an_unnormalised_one_hot_mixture_still_reduces_exactly():
    """A router emitting scores rather than probabilities must land on the same tensor.

    ``w / w`` is exact in IEEE754, so this is a real bit-identity claim and not a tolerance.
    """
    a, _ = routed({RL: 7.5})
    assert torch.equal(bits(a), bits(vanilla()))
    b, _ = routed({SFT: 3.0})
    c, _ = routed({SFT: 1.0})
    assert torch.equal(bits(b), bits(c))


# ================================================================== the mixture rule =====


def test_a_mixture_lands_the_weighted_sum_on_the_response_tokens():
    """The stated semantics: ``a * original + b * sft_weight + c * 0``.

    Read on the INFORMATIVE group as well as the silent one. On a silent group the original
    advantages are identically zero, so ``a`` multiplies nothing and any value of ``a``
    produces the same tensor -- a test using only ``MIXED``'s solved group would not
    constrain the RL coefficient at all.
    """
    base = vanilla()
    got, _ = routed({RL: 0.6, SFT: 0.4})
    expected = 0.6 * base[:, RESPONSE] + 0.4 * W
    assert torch.allclose(got[:, RESPONSE], expected, atol=1e-6), got[:, RESPONSE] - expected
    # The informative group's original advantages were not zero, so the RL term is live.
    assert base[G:, RESPONSE].abs().max() > 1e-6


def test_a_mixture_is_the_linear_blend_of_the_pure_arms():
    """Stated as a decomposition, so it cannot be satisfied by a coincidence of constants.

    ``a * (pure rl result) + b * (pure sft result) + c * 0`` -- each arm measured through
    the actor, then recombined here. An implementation that blended the wrong two tensors
    (say, the SFT constant against the SFT constant) reproduces neither endpoint's slope.
    """
    rl, _ = routed({RL: 1.0})
    sft, _ = routed({SFT: 1.0})
    got, _ = routed({RL: 0.5, SFT: 0.3, SKIP: 0.2})
    expected = 0.5 * rl[:, RESPONSE] + 0.3 * sft[:, RESPONSE] + 0.2 * 0.0
    assert torch.allclose(got[:, RESPONSE], expected, atol=1e-6), got[:, RESPONSE] - expected


def test_the_rl_and_sft_coefficients_are_not_interchangeable():
    """An asymmetric mixture, so swapping the two coefficients changes the answer.

    ``{rl: 0.6, sft: 0.4}`` and ``{rl: 0.4, sft: 0.6}`` differ on both groups: on the silent
    one through the SFT constant, on the informative one through the RL term.
    """
    a, _ = routed({RL: 0.6, SFT: 0.4})
    b, _ = routed({RL: 0.4, SFT: 0.6})
    assert not torch.allclose(a, b)
    assert not torch.allclose(a[:G], b[:G]), "the silent group did not separate them"
    assert not torch.allclose(a[G:], b[G:]), "the informative group did not separate them"


def test_a_skip_component_shrinks_the_update_towards_zero():
    """``c`` contributes exactly nothing, which is what makes SKIP a proportion of effort."""
    base = vanilla()
    got, _ = routed({RL: 0.25, SKIP: 0.75})
    assert torch.allclose(got[:, RESPONSE], 0.25 * base[:, RESPONSE], atol=1e-6)


def test_weights_need_not_be_normalised_by_the_router():
    """``{rl: 6, sft: 4}`` is the same decision as ``{rl: 0.6, sft: 0.4}``."""
    a, _ = routed({RL: 6.0, SFT: 4.0})
    b, _ = routed({RL: 0.6, SFT: 0.4})
    assert torch.equal(a, b), (a - b).abs().max()


def test_each_group_gets_its_own_mixture():
    """Two groups, two different blends: a slicing defect shows up as one leaking into the other."""
    base = vanilla()
    got, _ = routed({RL: 0.25, SFT: 0.75}, {RL: 0.75, SFT: 0.25})
    assert torch.allclose(
        got[:G, RESPONSE], 0.25 * base[:G, RESPONSE] + 0.75 * W, atol=1e-6
    ), got[:G]
    assert torch.allclose(
        got[G:, RESPONSE], 0.75 * base[G:, RESPONSE] + 0.25 * W, atol=1e-6
    ), got[G:]


def test_the_prompt_region_is_never_written_by_a_mixture():
    """A blended value on a prompt token is gradient on text the model did not choose.

    Asserted against the UNROUTED tensor, not against zero: the actor's GAE carries
    ``lastgaelam`` backwards through the masked prefix, so the informative group arrives
    with NON-ZERO prompt advantages. What must hold is that mixing does not move them.
    """
    base = vanilla()
    assert base[G:, :PROMPT].abs().max() > 1e-6, "the premise: prompt advantages are real"
    for mix in MIXES:
        got, _ = routed(mix)
        assert torch.equal(bits(got[:, :PROMPT]), bits(base[:, :PROMPT])), (mix, got)


def test_the_magnitude_scales_with_the_configured_solved_advantage():
    """A hardcoded constant that happens to match 0.5 would pass every test above."""
    base = vanilla()
    got, _ = routed({RL: 0.5, SFT: 0.5}, sft_weight=0.25)
    assert torch.allclose(
        got[:, RESPONSE], 0.5 * base[:, RESPONSE] + 0.5 * 0.25, atol=1e-6
    ), got


# ======================================================================== the gate =======


def test_the_argmax_gate_still_discards_the_mixture():
    """``decision='argmax'`` must behave exactly as it did before this feature existed.

    The router emits ``{rl: 0.6, sft: 0.4}``; under argmax that is the mode ``rl``, which is
    the identity, so the batch must come back bit-identical to vanilla. This is the test
    that fails if the gate is wired to always take the new path.
    """
    got, _ = routed({RL: 0.6, SFT: 0.4}, decision="argmax")
    assert torch.equal(bits(got), bits(vanilla())), (got - vanilla()).abs().max()


def test_the_mixture_gate_changes_what_the_argmax_gate_returns():
    """The gate is an ablation axis only if the two sides differ on the same router."""
    hard, _ = routed({RL: 0.6, SFT: 0.4}, decision="argmax")
    soft, _ = routed({RL: 0.6, SFT: 0.4}, decision="mixture")
    assert not torch.equal(hard, soft)


def test_the_default_configuration_is_the_argmax_path():
    """Nothing changes unless the flag is set; the shipped default is the old behaviour."""
    assert GroupRoutingConfig().decision == "argmax"
    got, _ = routed({RL: 0.6, SFT: 0.4}, decision=GroupRoutingConfig().decision)
    assert torch.equal(bits(got), bits(vanilla()))


def test_an_unknown_decision_value_is_refused():
    """Falling back to argmax would report a mixture arm in which no mixture ever ran."""
    with pytest.raises(ValueError, match="decision must be"):
        GroupRoutingConfig(enabled=True, router=STUB, decision="mixtures")


def test_a_mixture_without_a_router_is_refused():
    """The fixed solved/unsolved rule emits no weights, so there is nothing to mix."""
    with pytest.raises(ValueError, match="requires a router"):
        GroupRoutingConfig(enabled=True, solved_advantage=W, decision="mixture")


# =============================================================== router bookkeeping ======


def test_the_router_is_asked_exactly_once_per_group():
    """Reading the weights and then the label must not route twice: a router may be stateful."""
    _, r = routed({RL: 0.6, SFT: 0.4})
    assert len(r.seen) == len(MIXED) // G == 2, r.seen
    assert r.calls == 2


def test_feedback_still_flows_under_a_mixture_and_credits_the_argmax_label():
    """A mixture arm must still be able to learn, and what it learns from is documented.

    ``DecisionOutcome`` carries ONE mode name, and there is no measured way to divide a
    prompt's change in solve rate among the components of a mixture, so a mixture is
    credited under its argmax. That is a stated limitation of this axis; this test pins it
    so it cannot become an undocumented one.

    The two mixtures have DIFFERENT argmaxes, because ``batch_outcomes`` refuses a batch
    whose decisions were all the same mode as provably unattributable.
    """
    with routing_to({RL: 0.9, SFT: 0.1}, {SFT: 0.9, RL: 0.1}) as made:
        actor = make_actor(
            GroupRoutingConfig(
                enabled=True, solved_advantage=W, router=STUB, decision="mixture"
            )
        )
        torch.manual_seed(0)
        advantages(actor, MIXED)
        torch.manual_seed(0)
        advantages(actor, MIXED)
    r = made[0]
    assert len(r.observed) == 1, r.observed
    assert {o.mode for o in r.observed[0].values()} == {RL, SFT}


# ================================================================ the seam directly ======


@pytest.mark.parametrize("name", list(masks(4, 6)))
@pytest.mark.parametrize("mix", MIXES)
def test_the_mask_bounds_every_mixture_write(name, mix):
    """Nothing outside the mask moves, in either direction, for any blend.

    Writing a blended value on a prompt token puts gradient where the model was never asked
    to generate; writing a zero there is invisible to the loss but erases the GAE value the
    actor left behind and makes ``changed_rows`` count a row whose gradient did not move.
    """
    lm = masks(4, 6)[name]
    adv, _ = informative()
    out, _ = apply_mixtures(adv, lm, [2, 2], [mix, mix], sft_weight=0.5)
    off = lm == 0
    assert torch.equal(out[off], adv[off]), (out - adv).abs().max()


@pytest.mark.parametrize("mix", MIXES)
def test_the_callers_tensor_is_not_modified(mix):
    """By identity, by storage and by value: the caller may still hold the original."""
    adv, lm = informative()
    snapshot = adv.clone()
    out, _ = apply_mixtures(adv, lm, [4], [mix], sft_weight=0.5)
    assert out is not adv
    assert out.data_ptr() != adv.data_ptr()
    assert torch.equal(adv, snapshot), (adv - snapshot).abs().max()


@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mix", MIXES)
def test_the_advantage_dtype_survives(dt, mix):
    """A dtype change here reaches the loss as a silent upcast of the whole batch."""
    adv, lm = informative()
    out, _ = apply_mixtures(adv.to(dt), lm, [4], [mix], sft_weight=0.5)
    assert out.dtype == dt


def test_a_weight_valued_mask_scales_the_sft_component_as_the_hard_path_does():
    """``loss_mask`` is 0/1 today, and the two entry points must not disagree if it stops being.

    ``apply_decisions`` scales its SFT write by the mask VALUE; the mixture's SFT component
    has to do the same, or the day a weighted mask arrives every SFT magnitude changes on
    one path only.
    """
    adv, lm = silent()
    lm = lm * 2.0
    out, _ = apply_mixtures(adv, lm, [4], [{SFT: 1.0}], sft_weight=0.5)
    hard, _ = apply_decisions(adv, lm, [4], [SFT], sft_weight=0.5)
    assert torch.equal(out, hard)
    assert float(out[0, -1]) == 1.0


# ---------------------------------------------------------------------- statistics ------


def test_counts_report_mode_mass_and_reduce_to_group_counts_when_hard():
    """Mass is the only reading under which "0.6 of this group was RL" can be reported."""
    adv, lm = informative(b=4, t=6)
    _, hard = apply_mixtures(adv, lm, [2, 2], [{RL: 1.0}, {SFT: 1.0}], sft_weight=0.5)
    assert hard.counts == {RL: 1.0, SFT: 1.0, SKIP: 0.0}
    _, soft = apply_mixtures(
        adv, lm, [2, 2], [{RL: 0.6, SFT: 0.4}, {SFT: 0.5, SKIP: 0.5}], sft_weight=0.5
    )
    assert soft.counts == pytest.approx({RL: 0.6, SFT: 0.9, SKIP: 0.5})
    assert sum(soft.counts.values()) == pytest.approx(2.0), "mass must total the group count"


def test_mixed_groups_counts_only_the_genuinely_mixed_decisions():
    """A mixture arm whose router only emits one-hot decisions is an argmax run.

    Without this number that run reports itself as a mixture arm, which is the same class of
    error as a silently ignored bad mixture: an arm that never ran, reported as one that did.
    """
    adv, lm = informative(b=6, t=6)
    _, s = apply_mixtures(
        adv, lm, [2, 2, 2],
        [{RL: 1.0, SFT: 0.0}, {RL: 0.99, SFT: 0.01}, {SKIP: 4.0}],
        sft_weight=0.5,
    )
    assert s.mixed_groups == 1, s
    assert s.n_groups == 3


def test_changed_rows_counts_rows_the_blend_actually_moved():
    """A no-op blend must count zero, or the reported reach is inflated by decisions that
    did nothing -- the specific error this field exists to avoid."""
    adv, lm = silent()
    _, s = apply_mixtures(adv, lm, [4], [{RL: 0.5, SKIP: 0.5}], sft_weight=0.5)
    assert s.changed_rows == 0, s
    adv, lm = informative()
    _, s = apply_mixtures(adv, lm, [4], [{RL: 0.5, SKIP: 0.5}], sft_weight=0.5)
    assert s.changed_rows == 4, s


def test_the_metric_key_set_is_the_same_as_the_hard_paths():
    """A new key would make a mixture run's dashboard incomparable with the argmax run's."""
    adv, lm = informative(b=4, t=6)
    _, soft = apply_mixtures(adv, lm, [4], [{RL: 0.5, SFT: 0.5}], sft_weight=0.5)
    _, hard = apply_decisions(adv, lm, [4], [RL], sft_weight=0.5)
    assert set(soft.as_metrics()) == set(hard.as_metrics())
    assert soft.as_metrics()["route/changed_row_fraction"] == 1.0


def test_an_empty_batch_is_not_an_error():
    """A rollout can produce nothing routable; the metric must be 0.0, not a ZeroDivision."""
    out, s = apply_mixtures(torch.zeros(0, 5), torch.zeros(0, 5), [], [], sft_weight=0.5)
    assert tuple(out.shape) == (0, 5)
    assert (s.n_rows, s.n_groups, s.changed_rows, s.mixed_groups) == (0, 0, 0, 0)
    assert s.as_metrics()["route/changed_row_fraction"] == 0.0


# ---------------------------------------------------------------------- validation ------


@pytest.mark.parametrize(
    "mixtures, needle",
    [
        ([{}, {RL: 1.0}], "empty mixture"),
        ([{RL: 1.0}, {}], "empty mixture"),
        ([{RL: -0.1, SFT: 1.0}, {RL: 1.0}], "must be >= 0"),
        ([{RL: 0.0, SFT: 0.0}, {RL: 1.0}], "cannot be normalised"),
        ([{RL: float("nan")}, {RL: 1.0}], "must be finite"),
        ([{RL: float("inf")}, {RL: 1.0}], "must be finite"),
        ([{"banana": 1.0}, {RL: 1.0}], "mixes modes"),
        ([{TrainingMode.DISTILL: 1.0}, {RL: 1.0}], "mixes modes"),
        ([{RL: 0.5, TrainingMode.DISTILL: 0.5}, {RL: 1.0}], "mixes modes"),
        ([{RL: 1.0}], "one mixture per group"),
        ([{RL: 1.0}] * 3, "one mixture per group"),
    ],
    ids=["empty-first", "empty-second", "negative", "zero-sum", "nan", "inf", "unknown-name",
         "distill", "distill-component", "too-few", "too-many"],
)
def test_an_invalid_mixture_is_refused(mixtures, needle):
    """Loud, not silent. A mixture that is dropped, clamped or renormalised from garbage
    would let a run report decisions it never applied."""
    with pytest.raises(ValueError, match=needle):
        apply_mixtures(torch.zeros(4, 3), torch.ones(4, 3), [2, 2], mixtures, sft_weight=0.5)


def test_the_offending_group_is_named():
    """One bad mixture in a batch of hundreds: 'some group was invalid' is not a diagnosis."""
    with pytest.raises(ValueError, match="group 1"):
        apply_mixtures(
            torch.zeros(4, 3), torch.ones(4, 3), [2, 2], [{RL: 1.0}, {}], sft_weight=0.5
        )


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 4)), "same shape"),
        (dict(advantages=torch.zeros(4), loss_mask=torch.ones(4)), r"must be \(B, T\)"),
        (dict(advantages=torch.zeros(4, 3, dtype=torch.int64),
              loss_mask=torch.ones(4, 3)), "floating point"),
        (dict(group_sizes=[1, 1]), "sums to 2"),
        (dict(group_sizes=[-1, 5]), "must be >= 1"),
        (dict(sft_weight=-0.1), "must be >= 0"),
        (dict(sft_weight=float("nan")), "must be finite"),
    ],
    ids=["shape", "rank", "int-advantages", "sizes-sum", "negative-size", "negative-weight",
         "nan-weight"],
)
def test_the_hard_paths_guards_still_apply_to_mixtures(kwargs, needle):
    """Every guard is delegated to ``apply_decisions`` rather than restated here, so the two
    entry points cannot come to disagree about what a valid batch is. This is the test that
    the delegation actually happens."""
    call = dict(
        advantages=torch.zeros(4, 3), loss_mask=torch.ones(4, 3), group_sizes=[2, 2],
        sft_weight=0.5,
    )
    call.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        apply_mixtures(
            call["advantages"], call["loss_mask"], call["group_sizes"],
            [{RL: 1.0}] * len(call["group_sizes"]), sft_weight=call["sft_weight"],
        )


# ------------------------------------------------------------------------ the sweep -----


@pytest.mark.parametrize("seed", range(25))
def test_random_mixtures_match_the_stated_formula(seed):
    """Random shapes, partitions, mixtures and weights against an INDEPENDENT re-derivation.

    The expected tensor is written from the specification -- ``a * original + b * (weight *
    mask)`` -- rather than from the implementation's own expression, so it cannot pass by
    copying a mistake.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    b, t = rng.randint(1, 9), rng.randint(1, 7)
    adv = torch.randn(b, t)
    before = adv.clone()
    lm = (torch.rand(b, t) < 0.6).float()
    sizes = random_partition(rng, b)
    # 0.05 floor so a mixture can never sum to zero, which is a refusal rather than a blend.
    mixes = [
        {m: rng.random() + 0.05 for m in rng.sample(MODES, rng.randint(1, 3))}
        for _ in sizes
    ]
    w = rng.choice(WEIGHTS)

    out, stats = apply_mixtures(adv, lm, sizes, mixes, sft_weight=w)

    assert torch.equal(adv, before), "the caller's tensor was modified in place"
    off = lm == 0
    assert torch.equal(out[off], before[off]), "a write escaped the loss mask"
    start = 0
    for g, mix in zip(sizes, mixes):
        rows = slice(start, start + g)
        start += g
        total = sum(mix.values())
        a, coef = mix.get(RL, 0.0) / total, mix.get(SFT, 0.0) / total
        expected = a * before[rows] + coef * (w * lm[rows])
        on = lm[rows].bool()
        assert torch.allclose(out[rows][on], expected[on], atol=1e-6), (seed, mix)
    assert stats.changed_rows == int((out != before).any(dim=-1).sum()), (seed, stats)
    assert (stats.n_rows, stats.n_groups) == (b, len(sizes))
    assert sum(stats.counts.values()) == pytest.approx(len(sizes))
    assert stats.mixed_groups == sum(1 for m in mixes if len(m) > 1)
