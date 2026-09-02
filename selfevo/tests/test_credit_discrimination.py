"""Does the credit signal let the router discriminate, and does it point the right way?

Three questions, in the order they have to be answered.

1. **Is the diagnosis true?** The recorded cause of the null is that a shared per-batch scalar
   credits every arm with the same number, so the arms cannot separate. The first two tests
   turn that into an exact algebraic statement and check it: under a shared scalar an arm's
   parameters are fixed entirely by WHICH CONTEXTS it received, and the mode label contributes
   nothing at all -- rotate which arm gets which contexts and the fitted parameters rotate with
   them, to floating-point equality. Under a credit that depends on the mode the same rotation
   does not reproduce them. That is the difference between "the bandit is broken" and "the
   signal is uninformative", stated as an identity rather than a threshold.

2. **Does per-prompt credit fix it?** ``selfevo.routing.credit_sim`` poses a world where one
   mode genuinely helps an identifiable half of the prompts and another mode helps the rest, so
   there is a right answer to find. The batch-credited router does not find it; the
   prompt-credited one does, and holds it.

3. **Is it targeting, or is it noise?** Two controls, because both failure modes have cost
   this project real time. A router whose estimates merely got noisier drifts away from uniform
   without pointing anywhere, so the L1 distance from uniform -- the number the goal file
   tracks -- is reported but never asserted on as evidence: it RISES for the batch arm that
   learns nothing. And a mechanism that scores the same when its per-prompt correspondence is
   destroyed was never using that correspondence, so every prompt-credit result is run again
   with the credits shuffled across the prompts that earned them.
"""

from __future__ import annotations

import numpy as np
import pytest

from selfevo.routing.base import RoutingContext
from selfevo.routing.contextual import ContextualBanditRouter
from selfevo.routing.credit_sim import CREDIT_RULES, ModePreferenceWorld, RunTrace, simulate
from selfevo.routing.feedback import DecisionOutcome
from selfevo.routing.prompt_credit import PromptCreditLedger

SEEDS = tuple(range(8))
_CACHE: dict[tuple, RunTrace] = {}


def _trace(credit: str, seed: int, shuffle: bool = False) -> RunTrace:
    """Memoised run, so the several tests that need the same arm pay for it once."""
    key = (credit, seed, shuffle)
    if key not in _CACHE:
        _CACHE[key] = simulate(credit, seed=seed, shuffle_credit=shuffle)
    return _CACHE[key]


def _final_contrast(credit: str, shuffle: bool = False) -> np.ndarray:
    """Final-quarter subset contrast for every seed, as a paired sample."""
    return np.array([_trace(credit, s, shuffle).subset_contrast()[-1] for s in SEEDS])


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and standard error OF THE DIFFERENCE, which is the quantity a claim needs.

    Seeds are paired -- the same world, the same prompts, the same order -- so the difference
    is far better resolved than either arm's own spread, and quoting separate error bars would
    understate the evidence in one direction and overstate it in the other.
    """
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# ------------------------------------------------- 1. is the diagnosis actually true? ----


def _fixed_context(rng: np.random.Generator, step: int, index: int) -> RoutingContext:
    """A context with every observability feature drawn independently of the mode."""
    from selfevo.observability import FEATURE_NAMES

    extra = {name: float(rng.uniform(0.0, 1.0)) for name in FEATURE_NAMES}
    return RoutingContext(
        solve_rate=extra["solve_rate"], group_size=8, has_teacher=False,
        unit_id=f"{step}:{index}", extra=extra,
    )


def _replay(rotate: int, *, mode_dependent: bool, batches: int = 40, seed: int = 0):
    """Drive a round-robin router over a fixed context stream, rotating the assignment.

    ``cold_start_rounds`` is set past the end of the run so selection is round-robin
    throughout: unit ``i`` of a batch of nine takes ``sorted(modes)[i % 3]``. Rotating the
    order in which the SAME contexts are presented therefore hands each arm exactly the set of
    contexts a different arm received in the unrotated run, with the batch's scalar unchanged.
    This is the only way to force an assignment through the public seam -- ``route`` chooses
    for itself -- and forcing it is the point: the claim is about what the update does with a
    label, so the label has to be the only thing that moves.

    Args:
        rotate: How far to rotate the batch before routing it.
        mode_dependent: Credit each unit by its mode as well as the batch's scalar.
        batches: Batches to run. Nine units each, so every batch starts the cycle at ``rl``.
        seed: RNG seed for the context stream and the batch scalars.

    Returns:
        ``{mode: theta}``, the ridge solution per arm.
    """
    rng = np.random.default_rng(seed)
    router = ContextualBanditRouter(cold_start_rounds=10 ** 9)
    per_mode = {"rl": 0.0, "sft": 1.0, "skip": -1.0}
    for step in range(batches):
        contexts = [_fixed_context(rng, step, i) for i in range(9)]
        scalar = float(rng.normal(0.0, 0.1))
        ordered = contexts[rotate:] + contexts[:rotate]
        modes = [router.route(ctx).argmax() for ctx in ordered]
        router.observe({
            ctx.unit_id: DecisionOutcome(
                mode=mode,
                value=scalar + (per_mode[mode] if mode_dependent else 0.0),
                batch_id=str(step),
            )
            for ctx, mode in zip(ordered, modes)
        })
    return {m: np.linalg.inv(router._A[m]) @ router._b[m] for m in router.modes}


def test_a_shared_batch_scalar_credits_the_mode_label_with_nothing():
    """The diagnosis as an identity: rotate the assignment and the arms rotate exactly.

    ``b_m += (value / cost) * x`` and ``A_m += x x^T``. With one scalar for the whole batch,
    neither update mentions the mode except as the dictionary key it lands in, so an arm's
    parameters are a function of the CONTEXTS it received and nothing else. Two arms fed
    exchangeable streams of contexts therefore converge to the same estimate whatever they
    were called, which is precisely why the mode mix could not sharpen over 129 live steps.
    """
    base = _replay(0, mode_dependent=False)
    rotated = _replay(1, mode_dependent=False)
    # Rotating by one moves arm rl's contexts onto sft, sft's onto skip, skip's onto rl.
    for got, want in (("rl", "sft"), ("sft", "skip"), ("skip", "rl")):
        assert np.allclose(rotated[got], base[want], atol=1e-12), (
            f"arm {got!r} did not reproduce arm {want!r}: the update read the mode label"
        )


def test_the_identity_breaks_as_soon_as_the_credit_depends_on_the_mode():
    """The complement, and the reason this is a diagnosis rather than a tautology.

    The same rotation over the same contexts, with the mode contributing to the credited value.
    Now the arms do NOT rotate, because the label finally carries information. Without this the
    test above would be satisfied by a router that ignored its outcomes entirely.
    """
    base = _replay(0, mode_dependent=True)
    rotated = _replay(1, mode_dependent=True)
    gaps = [float(np.linalg.norm(rotated[g] - base[w]))
            for g, w in (("rl", "sft"), ("sft", "skip"), ("skip", "rl"))]
    assert min(gaps) > 0.1, f"mode-dependent credit still rotated cleanly: {gaps}"


# --------------------------------------------- 2. does per-prompt credit restore it? ----


def test_batch_credit_leaves_the_router_at_chance_on_a_world_with_a_right_answer():
    """The failure, reproduced where the right answer is known.

    Half the prompts are helped only by SFT and half only by RL, and the half is written into a
    feature the router reads. The batch-credited router ends the run treating the two halves
    the same, which is what having learned nothing looks like when there is something to learn.
    """
    trace = _trace("batch", seed=0)
    assert trace.updates > 100, f"the arm never received credit at all: {trace.updates}"
    contrast = trace.subset_contrast()
    assert max(contrast) < 0.30, f"batch credit unexpectedly targeted the subsets: {contrast}"


def test_the_distance_from_uniform_rises_for_an_arm_that_learned_nothing():
    """Why the tracked number cannot carry the claim, stated as a test so it stays true.

    The failing signature is quoted as L1 from uniform going 0.056 -> 0.027. Here the same
    uninformative signal moves L1 the other way -- the router's own choices decide which
    contexts each arm sees, so the arms drift apart on that feedback alone -- while the
    targeting stays at chance. A fix validated on L1 alone could therefore be validated by
    noise, and this test fails if anyone starts believing otherwise.
    """
    for seed in SEEDS:
        trace = _trace("batch", seed)
        assert max(trace.l1_from_uniform()) > 0.08, (
            f"seed {seed}: L1 stayed flat, so this test no longer demonstrates the point"
        )
        assert max(trace.subset_contrast()) < 0.30, (
            f"seed {seed}: the batch arm targeted something, which would refute the diagnosis"
        )


def test_per_prompt_credit_develops_the_preference_and_holds_it():
    """The fix, on the same world and the same router: only the credit rule differs."""
    trace = _trace("prompt", seed=0)
    contrast = trace.subset_contrast()
    assert contrast[0] < contrast[1], f"no preference developed: {contrast}"
    assert min(contrast[1:]) > 0.60, f"the preference did not hold: {contrast}"


def test_the_preference_points_at_the_mode_that_is_actually_better():
    """Away from uniform is not enough; it has to be away in the right direction.

    Checked per subset, so a router that simply collapsed onto one mode fails: it would score
    high on the subset that happens to prefer that mode and at chance on the other.
    """
    for seed in SEEDS:
        final = _trace("prompt", seed).targeting_accuracy()[-1]
        assert len(final) == 2, f"seed {seed}: expected two subsets, got {final}"
        for subset, accuracy in final.items():
            assert accuracy > 0.60, f"seed {seed}: subset {subset!r} routed at {accuracy:.2f}"


def test_the_gap_over_batch_credit_is_far_larger_than_the_seed_noise():
    """Eight paired seeds, and the error bar is on the DIFFERENCE, not on either arm."""
    diff, se = _paired(_final_contrast("prompt"), _final_contrast("batch"))
    assert diff > 0.40 and diff > 10 * se, f"prompt - batch = {diff:.3f} +- {se:.3f}"


# ------------------------------------------------------------- 3. the two controls ----


def test_shuffling_the_credit_across_prompts_destroys_the_preference():
    """The absorption control: same pairings, same credit values, correspondence destroyed.

    If a router given per-prompt credits that have been permuted across the prompts that earned
    them still developed the preference, the preference would not be coming from the targeting
    -- it would be an artifact of receiving a higher-variance signal, and this project has
    repeatedly found a "smart" mechanism performing exactly like a random one.
    """
    treatment, control = _final_contrast("prompt"), _final_contrast("prompt", shuffle=True)
    assert max(control) < 0.35, f"shuffled credit still targeted the subsets: {control}"
    diff, se = _paired(treatment, control)
    assert diff > 0.35 and diff > 10 * se, f"prompt - shuffled = {diff:.3f} +- {se:.3f}"


def test_the_control_also_holds_for_the_self_baselined_rule():
    """The same control on the rule this change adds, since a new rule needs its own."""
    treatment = _final_contrast("prompt_self_baseline")
    control = _final_contrast("prompt_self_baseline", shuffle=True)
    assert max(control) < 0.35, f"shuffled self-baselined credit targeted: {control}"
    diff, se = _paired(treatment, control)
    assert diff > 0.35 and diff > 10 * se, f"self_baseline - shuffled = {diff:.3f} +- {se:.3f}"


def test_shuffling_a_batch_credited_run_is_refused_rather_than_reported_as_a_control():
    """An equivalent mutant is not evidence, and the module refuses to pose as one."""
    with pytest.raises(ValueError, match="no-op"):
        simulate("batch", shuffle_credit=True)


def test_a_per_prompt_baseline_beats_subtracting_the_batchs_mean():
    """The measured reason this change exists rather than using the centring already shipped.

    ``credit="prompt_centered"`` subtracts the batch's mean delta, which is one number shared by
    every arm -- the same class of quantity the whole per-prompt design exists to get away
    from. Centring on the prompt's OWN earlier deltas is a per-prompt quantity and measures
    better. The margin is small next to the batch-vs-prompt gap, so it is asserted with the
    error bar on the paired difference and not on a single seed.
    """
    diff, se = _paired(_final_contrast("prompt_self_baseline"), _final_contrast("prompt_centered"))
    assert diff > 0.03 and diff > 2 * se, f"self_baseline - centered = {diff:.3f} +- {se:.3f}"


# ------------------------------------------------------- the credit rule in isolation ----


def _sightings(ledger: PromptCreditLedger, values, mode="rl"):
    """Show one prompt to the ledger once per step, returning the credit it emits each time."""
    out = []
    for step, value in enumerate(values):
        closed = ledger.observe_and_record("p", f"{step}:0", mode, value, step)
        out.append(None if closed is None else closed[1])
    return out


def test_the_default_baseline_is_the_prompts_previous_value_and_is_unchanged():
    """Every arm run before this change must be reproduced exactly, so this pins it."""
    got = _sightings(PromptCreditLedger(), [0.0, 0.2, 0.5, 0.6])
    assert got[0] is None
    assert [round(v, 10) for v in got[1:]] == [0.2, 0.3, 0.1]


def test_the_self_mean_baseline_subtracts_only_strictly_earlier_deltas():
    """Leave-current-out, checked on numbers, because folding the current delta in is silent.

    Deltas are 0.2, 0.3, 0.1. The first has no history and is withheld. The second is scored
    against 0.2 and the third against the mean of 0.2 and 0.3 -- never against a mean that
    already contains the delta being scored, which would shrink every credit toward zero and
    shrink the largest ones most.
    """
    ledger = PromptCreditLedger(baseline="self_mean")
    got = _sightings(ledger, [0.0, 0.2, 0.5, 0.6])
    assert got[0] is None and got[1] is None
    assert round(got[2], 10) == round(0.3 - 0.2, 10)
    assert round(got[3], 10) == round(0.1 - 0.25, 10)
    assert ledger.as_metrics()["prompt_credit/cold_baseline_skips"] == 1.0
    assert ledger.as_metrics()["prompt_credit/credited"] == 2.0


def test_a_withheld_first_delta_is_counted_and_not_silently_credited_as_zero():
    """A prompt seen exactly twice yields nothing under this baseline, and says so."""
    ledger = PromptCreditLedger(baseline="self_mean")
    assert _sightings(ledger, [0.4, 0.9]) == [None, None]
    assert ledger.as_metrics()["prompt_credit/cold_baseline_skips"] == 1.0
    assert ledger.as_metrics()["prompt_credit/credited"] == 0.0


def test_the_default_baseline_never_withholds_and_never_counts_a_cold_skip():
    """The two rules differ on exactly one axis, so the counter must stay 0 on the old one."""
    ledger = PromptCreditLedger()
    _sightings(ledger, [0.4, 0.9, 0.95])
    assert ledger.as_metrics()["prompt_credit/cold_baseline_skips"] == 0.0
    assert ledger.as_metrics()["prompt_credit/credited"] == 2.0


def test_a_prompts_delta_history_is_evicted_with_the_prompt():
    """Bounded memory: a prompt's baseline is two floats on its record, so eviction takes it.

    The consequence is deliberate -- a prompt that comes back after being evicted is cold again
    rather than centred against a history that no longer exists -- and the alternative, a
    second table keyed by prompt, would grow without limit across a long run.
    """
    ledger = PromptCreditLedger(capacity=1, baseline="self_mean")
    ledger.observe_and_record("a", "0:0", "rl", 0.1, 0)
    ledger.observe_and_record("a", "1:0", "rl", 0.3, 1)
    ledger.observe_and_record("b", "2:0", "rl", 0.5, 2)      # evicts "a" and its history
    assert ledger.as_metrics()["prompt_credit/evicted"] == 1.0
    assert ledger.observe_and_record("a", "3:0", "rl", 0.9, 3) is None
    assert ledger.as_metrics()["prompt_credit/credited"] == 0.0


def test_an_unknown_baseline_is_rejected_rather_than_falling_back():
    """A silent fallback would report a self-baselined arm that in fact ran the old rule."""
    with pytest.raises(ValueError, match="baseline must be"):
        PromptCreditLedger(baseline="prompt_centered")


# ----------------------------------------------------------------- simulator guards ----


def test_every_named_credit_rule_actually_runs():
    """A rule listed but unreachable would be an arm reported and never run."""
    for rule in CREDIT_RULES:
        trace = simulate(rule, seed=0, steps=12)
        assert trace.updates >= 0 and len(trace.decisions) == 12


def test_an_unknown_credit_rule_is_rejected():
    with pytest.raises(ValueError, match="credit must be one of"):
        simulate("prompt_mean", seed=0, steps=4)


def test_a_world_whose_marker_is_not_a_feature_is_rejected():
    """The router reads ctx.extra by name; an unknown marker would be invisible to it."""
    with pytest.raises(ValueError, match="not an observability feature"):
        ModePreferenceWorld(marker="difficulty")


def test_a_world_with_one_right_mode_everywhere_is_rejected():
    """Such a world cannot tell a targeting router from one with a favourite mode."""
    with pytest.raises(ValueError, match="must differ"):
        ModePreferenceWorld(high_mode="sft", low_mode="sft")


def test_a_pool_smaller_than_a_batch_is_rejected():
    with pytest.raises(ValueError, match="at least batch_size"):
        ModePreferenceWorld(n_prompts=8, batch_size=32)


def test_a_run_is_reproducible_from_its_seed():
    """Fixed seed, identical decisions -- otherwise none of the paired differences mean much."""
    assert simulate("prompt", seed=5, steps=20).decisions == simulate("prompt", seed=5, steps=20).decisions


def test_more_windows_than_steps_is_refused_rather_than_averaged_over_nothing():
    trace = simulate("prompt", seed=0, steps=3)
    with pytest.raises(ValueError, match="cannot be split"):
        trace.l1_from_uniform(quarters=8)


# ------------------------------------------------- the arms are the arms they claim ----


def test_the_centred_arm_really_centres():
    """``prompt`` and ``prompt_centered`` must not be the same run wearing two labels.

    Asserted on the decisions rather than on a score: the two rules are close enough that a
    scored comparison can be satisfied by noise, and an ablation pair whose two halves are
    byte-identical is the failure this project keeps finding under other names.
    """
    plain = simulate("prompt", seed=1, steps=30).decisions
    centred = simulate("prompt_centered", seed=1, steps=30).decisions
    assert plain != centred, "centring changed nothing, so the ablation pair is one arm"


def test_the_self_baselined_arm_really_self_baselines():
    """The withheld first delta is the fingerprint of the baseline, so it is checked directly.

    A scored comparison could not tell a self-baselined run from the raw one at this margin;
    the ledger counter can, and it is the number a live run would be read by.
    """
    baselined = simulate("prompt_self_baseline", seed=1, steps=40)
    plain = simulate("prompt", seed=1, steps=40)
    assert baselined.metrics["prompt_credit/cold_baseline_skips"] > 0
    assert plain.metrics["prompt_credit/cold_baseline_skips"] == 0.0


# ---------------------------------------------------- the metrics have their claimed scale ----


def _hand(rows: list[tuple[str, str]]) -> RunTrace:
    """A one-step trace built by hand, so a metric can be checked against a known answer."""
    return RunTrace(decisions=(tuple(rows),), modes=("rl", "sft", "skip"), updates=0)


def test_the_metrics_are_on_the_scale_the_docstrings_claim():
    """Known answers, because a metric read only on live output cannot be wrong out loud.

    Three hand-built traces: perfect targeting, a single mode everywhere, and a uniform mix.
    They pin the constant in the uniform reference (1/3, not 1/2), the half in the total
    variation distance, the direction of the targeting fraction, and -- via the all-RL case --
    that a mode nobody chose still counts as a zero share instead of dropping out of the sum.
    """
    separating = _hand([("sft", "sft"), ("rl", "rl")])
    assert separating.subset_contrast(1) == [1.0]
    assert separating.targeting_accuracy(1) == [{"rl": 1.0, "sft": 1.0}]
    assert round(separating.l1_from_uniform(1)[0], 10) == round(2 / 3, 10)

    one_mode = _hand([("rl", "sft"), ("rl", "rl")])
    assert one_mode.subset_contrast(1) == [0.0]
    assert one_mode.targeting_accuracy(1) == [{"rl": 1.0, "sft": 0.0}]
    assert round(one_mode.l1_from_uniform(1)[0], 10) == round(4 / 3, 10)

    uniform = _hand([(m, s) for s in ("sft", "rl") for m in ("rl", "sft", "skip")])
    assert uniform.l1_from_uniform(1) == [0.0]
    assert uniform.subset_contrast(1) == [0.0]


def test_contrasting_a_window_with_only_one_subset_is_refused():
    """Returning 0 there would read as "no targeting" for a window that could not show any."""
    with pytest.raises(ValueError, match="only one prompt subset"):
        _hand([("rl", "sft"), ("sft", "sft")]).subset_contrast(1)
