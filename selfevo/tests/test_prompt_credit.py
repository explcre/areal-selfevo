"""Per-prompt credit: identity from tokens, and pairing across time.

This is the fix implied by the measured null in `test_credit_assignment.py`: a single
per-batch scalar credited to 64 decisions carries no information separating the arms, so the
signal has to hold the PROMPT fixed and vary the mode across time.
"""

from __future__ import annotations

import pytest

from selfevo.routing.prompt_credit import PromptCreditLedger, prompt_key

# Two rollouts of the SAME prompt: identical prompt tokens, different responses.
PROMPT_A = [11, 22, 33]
ROLLOUT_A1 = (PROMPT_A + [90, 91], [0, 0, 0, 1, 1])
ROLLOUT_A2 = (PROMPT_A + [70, 71], [0, 0, 0, 1, 1])
ROLLOUT_B = ([44, 55, 66] + [90, 91], [0, 0, 0, 1, 1])


def test_the_same_prompt_hashes_the_same_whatever_the_response():
    """Identity must come from the prompt, or a prompt never pairs with itself."""
    assert prompt_key(*ROLLOUT_A1) == prompt_key(*ROLLOUT_A2)


def test_different_prompts_hash_differently():
    """Merging prompts would credit one task's decision to another's outcome."""
    assert prompt_key(*ROLLOUT_A1) != prompt_key(*ROLLOUT_B)


def test_prompts_sharing_a_leading_prefix_still_differ():
    """The realistic case, and the one a naive test misses.

    Every real prompt starts with the same BOS and system prefix, so identity derived from
    only the first token (or first few) would merge the entire dataset into one prompt and
    credit every decision to every other. The two prompts here share their first two tokens
    and differ only at the third.
    """
    shared = ([7, 7, 1] + [90], [0, 0, 0, 1])
    other = ([7, 7, 2] + [90], [0, 0, 0, 1])
    assert prompt_key(*shared) != prompt_key(*other)


def test_identity_depends_on_every_prompt_token():
    """Any position being ignored merges prompts that differ only there."""
    base = [5, 6, 7, 8]
    mask = [0, 0, 0, 0, 1]
    keys = set()
    for i in range(len(base)):
        altered = list(base)
        altered[i] += 1
        keys.add(prompt_key(altered + [90], mask))
    keys.add(prompt_key(base + [90], mask))
    assert len(keys) == len(base) + 1, "some prompt position does not affect identity"


def test_a_longer_response_does_not_change_identity():
    """Response length varies run to run; identity must not."""
    assert prompt_key(PROMPT_A + [1, 2, 3, 4], [0, 0, 0, 1, 1, 1, 1]) == prompt_key(*ROLLOUT_A1)


def test_a_row_with_no_prompt_region_is_refused():
    """Returning a constant would merge every such row into a single phantom prompt."""
    with pytest.raises(ValueError, match="no prompt region"):
        prompt_key([1, 2], [1, 1])


def test_mismatched_lengths_are_refused():
    """A length mismatch means the row was mis-sliced; keying on it hides that."""
    with pytest.raises(ValueError, match="same positions"):
        prompt_key([1, 2, 3], [0, 0])


# ------------------------------------------------------------------ pairing ------------


def test_the_first_sighting_credits_nothing():
    """There is no prior decision to close; crediting would invent a delta."""
    led = PromptCreditLedger()
    assert led.observe_and_record("k", "0:0", "sft", 0.25, step=0) is None
    assert led.credited == 0


def test_the_second_sighting_credits_the_first_decision():
    """The paired observation: same prompt, earlier mode, delta across time."""
    led = PromptCreditLedger()
    led.observe_and_record("k", "0:0", "sft", 0.25, step=0)
    got = led.observe_and_record("k", "29:3", "rl", 0.75, step=29)
    assert got is not None
    prior, delta = got
    assert prior.mode == "sft"
    assert prior.unit_id == "0:0"
    assert prior.step == 0
    assert delta == pytest.approx(0.5)
    assert led.credited == 1


def test_the_delta_is_signed_so_a_harmful_mode_is_punished():
    """A mode that made the prompt worse must produce a negative credit."""
    led = PromptCreditLedger()
    led.observe_and_record("k", "0:0", "sft", 0.80, step=0)
    _, delta = led.observe_and_record("k", "29:0", "rl", 0.20, step=29)
    assert delta == pytest.approx(-0.60)


def test_credit_uses_the_same_observation_it_records():
    """Recording before crediting would compare a value against itself: a constant zero."""
    led = PromptCreditLedger()
    led.observe_and_record("k", "0:0", "rl", 0.4, step=0)
    _, delta = led.observe_and_record("k", "29:0", "rl", 0.4, step=29)
    assert delta == pytest.approx(0.0)
    # A third sighting must pair against the SECOND value, not the first.
    _, delta3 = led.observe_and_record("k", "58:0", "rl", 0.9, step=58)
    assert delta3 == pytest.approx(0.5)


def test_prompts_are_kept_apart():
    """Two prompts in flight must not cross-credit."""
    led = PromptCreditLedger()
    led.observe_and_record("a", "0:0", "sft", 0.1, step=0)
    led.observe_and_record("b", "0:1", "skip", 0.9, step=0)
    prior_a, delta_a = led.observe_and_record("a", "29:0", "rl", 0.4, step=29)
    assert prior_a.mode == "sft" and delta_a == pytest.approx(0.3)
    prior_b, delta_b = led.observe_and_record("b", "29:1", "rl", 0.5, step=29)
    assert prior_b.mode == "skip" and delta_b == pytest.approx(-0.4)


# ------------------------------------------------------------------ capacity -----------


def test_eviction_is_oldest_first_and_counted():
    """Silent eviction starves the router with nothing in the log to say so."""
    led = PromptCreditLedger(capacity=2)
    led.observe_and_record("a", "0:0", "sft", 0.1, step=0)
    led.observe_and_record("b", "0:1", "sft", 0.1, step=0)
    led.observe_and_record("c", "0:2", "sft", 0.1, step=0)   # evicts "a"
    assert led.evicted == 1
    assert led.observe_and_record("a", "1:0", "rl", 0.5, step=1) is None  # "a" is gone
    assert led.observe_and_record("c", "1:2", "rl", 0.5, step=1) is not None


def test_a_reappearing_prompt_refreshes_its_position():
    """Otherwise a frequently-seen prompt is evicted while a stale one survives."""
    led = PromptCreditLedger(capacity=2)
    led.observe_and_record("a", "0:0", "sft", 0.1, step=0)
    led.observe_and_record("b", "0:1", "sft", 0.1, step=0)
    led.observe_and_record("a", "1:0", "sft", 0.2, step=1)   # "a" becomes newest
    led.observe_and_record("c", "1:2", "sft", 0.1, step=1)   # evicts "b", not "a"
    assert led.observe_and_record("a", "2:0", "rl", 0.3, step=2) is not None
    assert led.observe_and_record("b", "2:1", "rl", 0.3, step=2) is None


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        PromptCreditLedger(capacity=0)


def test_metrics_expose_the_starvation_signal():
    led = PromptCreditLedger(capacity=1)
    led.observe_and_record("a", "0:0", "sft", 0.1, step=0)
    led.observe_and_record("b", "0:1", "sft", 0.1, step=0)
    m = led.as_metrics()
    assert m["prompt_credit/evicted"] == 1.0
    assert m["prompt_credit/pending"] == 1.0
    assert m["prompt_credit/credited"] == 0.0
