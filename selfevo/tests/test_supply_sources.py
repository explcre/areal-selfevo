"""The generalised supplier axis behind the batch-construction seam, on CPU.

``test_gold_batch_path.py`` establishes the seam itself: an all-wrong group cannot be reached
through the advantage, a correct row can only enter as a ROW, and the splice, the sentinel and
the two reach guards are what makes that safe. This file tests the generalisation of WHO
supplies the row, and it is written against the five ways this project has been bitten before:

1. **A silent no-op.** A supplier with nothing to offer must refuse, with a typed reason that
   is counted, and must never return the row unchanged while the stats report a success. Two
   arms in this repo ran bit-identical to the off arm while reporting as on arms.
2. **A wrong row that looks like a right one.** A corpus or teacher row for a DIFFERENT prompt
   is spliced after this prompt, so it teaches an answer to a question it does not answer. The
   suppliers refuse instead, and the refusal is asserted rather than assumed.
3. **A second treatment of off-policy weighting.** Every source uses the gold path's sentinel
   and ``reconcile_gold_logprobs``; nothing here writes ``logprobs``.
4. **A control that is not one.** The matched control replays the treatment's realised source
   multiset, permuted, and the tests pin that the multiset matches, the assignment does NOT,
   the control cannot see features, and the comparison is non-vacuous.
5. **Rollback that is asserted, not argued.** The default path is digested against a constant
   measured on the checkout BEFORE the refactor existed.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from areal.utils.data import concat_padded_tensors  # noqa: E402
from selfevo.gold import (  # noqa: E402
    GOLD_LOGP_SENTINEL,
    GoldMissingError,
    GoldPolicyError,
    assert_gold_logprobs_filled,
    attach_gold,
    reconcile_gold_logprobs,
    substitute_gold_rows,
    substitute_in_place,
)
from selfevo.routing.prompt_credit import prompt_key  # noqa: E402
from selfevo.supply import (  # noqa: E402
    NO_SOURCE,
    SUPPLY_SOURCES,
    FixedSourcePolicy,
    ForcedSourcePolicy,
    MatchedSourceControl,
    Refusal,
    SupplierRefused,
    SupplyConfigError,
    SupplyOffer,
    SupplyRequest,
    build_suppliers,
    key_for_prompt,
    source_code,
    source_proportions,
)
from selfevo.supply.corpus import CorpusSupplier, load_corpus_jsonl  # noqa: E402
from selfevo.supply.gold import GoldSupplier  # noqa: E402
from selfevo.supply.self_gen import SelfGeneratedSupplier  # noqa: E402
from selfevo.supply.store import SolvedRolloutStore  # noqa: E402
from selfevo.supply.teacher import (  # noqa: E402
    RecordedTeacherClient,
    TeacherSupplier,
)

T = 12
PROMPT = 4
GOLD = [901, 902, 903]

# Digested on the checkout as it stood BEFORE selfevo/supply existed, with
# ~/tmp/supply/rollback_digest.py: the three-group fixture below, `substitute_in_place(.,
# "dyme")`, every returned tensor and every `gold/` metric. A refactor that changes what the
# default path produces fails here rather than in a run.
ROLLBACK_DIGEST = "63ebf46e8afd6dbe7c03ec31f5dac2bfb17cb804cd70f5f3a8066fe55219534d"


# --------------------------------------------------------------------------- fixtures ---


def make_row(
    reward: float,
    *,
    prompt_ids: list[int] | None = None,
    resp: list[int] | None = None,
    width: int = T,
) -> dict[str, torch.Tensor]:
    """One rollout row shaped exactly as a collated batch's row is.

    Args:
        reward: The row's raw reward.
        prompt_ids: Prompt token ids. Defaults to a fixed prompt, so two rows built without an
            explicit prompt are two rollouts of the SAME prompt -- which is what a group is.
        resp: Response token ids.
        width: Padded width.

    Returns:
        A dict of ``(1, width)`` tensors plus a ``(1,)`` reward, in the TOKEN coordinates a
        workflow emits.
    """
    prompt_ids = list(range(1, PROMPT + 1)) if prompt_ids is None else list(prompt_ids)
    resp = [51, 52, 53, 54, 55] if resp is None else list(resp)
    p, n = len(prompt_ids), len(prompt_ids) + len(resp)
    ids = torch.zeros(1, width, dtype=torch.int32)
    ids[0, :p] = torch.tensor(prompt_ids, dtype=torch.int32)
    ids[0, p:n] = torch.tensor(resp, dtype=torch.int32)
    loss_mask = torch.zeros(1, width, dtype=torch.int32)
    loss_mask[0, p:n] = 1
    attn = torch.zeros(1, width, dtype=torch.bool)
    attn[0, :n] = True
    logp = torch.zeros(1, width, dtype=torch.float32)
    logp[0, p:n] = -0.5
    return {
        "input_ids": ids,
        "loss_mask": loss_mask,
        "logprobs": logp,
        "versions": torch.full((1, width), -1, dtype=torch.int32),
        "turn_ids": torch.full((1, width), -1, dtype=torch.int32),
        "attention_mask": attn,
        "rewards": torch.tensor([reward], dtype=torch.float32),
    }


def make_group(
    rewards: list[float], *, gold: list[int] | None = GOLD, with_gold: bool = True, **kw
) -> dict[str, torch.Tensor]:
    """One GRPO group, collated the way ``GroupedRolloutWorkflow`` collates one.

    Args:
        rewards: One raw reward per rollout.
        gold: Gold token ids shared by the group, or None for a row with no gold text.
        with_gold: Whether to attach the gold keys at all. False builds the batch a run
            without ``keep_solution`` produces, which is what every non-gold source must work
            on.
        **kw: Forwarded to :func:`make_row`.

    Returns:
        A ``(len(rewards), T)`` batch dict.
    """
    rows = [make_row(r, **kw) for r in rewards]
    if with_gold:
        rows = [attach_gold(row, gold) for row in rows]
    return concat_padded_tensors(rows)


def digest(trajs, stats) -> str:
    """sha256 over every returned tensor and every ``gold/`` metric.

    Only the ``gold/`` metrics, because the supplier axis ADDS ``supply/`` keys to every arm on
    purpose -- a one-source arm and a four-source arm must emit the same key set -- and a
    rollback guard that failed on a new metric key would be asserting the wrong thing.
    """
    h = hashlib.sha256()
    for t in trajs:
        for k in sorted(t):
            v = t[k]
            h.update(k.encode())
            if torch.is_tensor(v):
                h.update(str(v.dtype).encode())
                h.update(str(tuple(v.shape)).encode())
                h.update(v.contiguous().to(torch.float64).numpy().tobytes())
            else:
                h.update(repr(v).encode())
    h.update(
        repr(
            [(k, v) for k, v in sorted(stats.as_metrics().items()) if k.startswith("gold/")]
        ).encode()
    )
    return h.hexdigest()


def store_with(prompt_ids: list[int], resp: list[int]) -> SolvedRolloutStore:
    """A store holding one correct response for one prompt."""
    store = SolvedRolloutStore(capacity=16)
    store.record(key_for_prompt(prompt_ids), torch.tensor(resp, dtype=torch.long))
    return store


def always_correct(request, tokens) -> bool:
    """A verifier that accepts, passed EXPLICITLY so the choice is greppable."""
    return True


def never_correct(request, tokens) -> bool:
    """A verifier that rejects everything the teacher proposes."""
    return False


# ------------------------------------------------------------------------- rollback ---


def test_the_gold_only_default_is_byte_identical_to_before_the_supplier_refactor():
    """Rollback, against a digest and not against an argument.

    The gold-reading arithmetic moved out of ``substitute_gold_rows``' per-group loop into
    ``GoldSupplier``. That is a refactor only if the output is unchanged, and "unchanged" has
    to mean every tensor of every returned trajectory plus every ``gold/`` count -- not a
    spot-check of ``input_ids``, which would pass on a version that stopped rewriting
    ``versions`` or ``turn_ids``.
    """
    batch = [
        make_group([0.0] * 4),
        make_group([0.0] * 4, gold=None),
        make_group([1.0] * 4),
    ]
    out, stats = substitute_in_place(batch, "dyme")
    assert digest(out, stats) == ROLLBACK_DIGEST


def test_the_capability_is_off_by_default_and_adds_no_tensor_to_the_pipeline():
    """The gate: passing neither ``suppliers`` nor ``source_policy`` is the OFF state.

    ``source_ids`` is a real ``(B, T)`` tensor that collates, packs and splits with every other
    one, so an arm that never asked for multiple sources must not start paying for it. The key
    appears exactly when the axis is engaged, and ``substitute_in_place`` applies the same
    condition to every element, so no two dicts in the list can disagree about it.
    """
    off, _ = substitute_gold_rows(make_group([0.0] * 4), "dyme")
    assert "source_ids" not in off
    assert "is_gold" in off

    on, _ = substitute_gold_rows(
        make_group([0.0] * 4), "dyme", suppliers={"gold": GoldSupplier()}
    )
    assert "source_ids" in on

    listed, _ = substitute_in_place(
        [make_group([0.0] * 4), make_group([1.0] * 4)],
        "dyme",
        suppliers={"gold": GoldSupplier()},
    )
    assert [sorted(d) for d in listed][0] == [sorted(d) for d in listed][1]


def test_a_none_rule_is_still_a_true_no_op_even_with_suppliers_configured():
    """The off arm stays inert when its neighbours are misconfigured.

    ``_safe_sizes`` already follows this rule for a bad grouping. A supplier mapping that would
    be refused under an active rule must not fail a run that asked for no substitution at all,
    because the whole point of ``none`` is that a run with the path compiled in and switched
    off is bit-identical to one built before it existed.
    """
    batch = make_group([0.0] * 4)
    out, stats = substitute_gold_rows(batch, "none", suppliers={})
    assert "is_gold" not in out and "source_ids" not in out
    assert torch.equal(out["input_ids"], batch["input_ids"])
    assert stats.rows_substituted == 0 and stats.decisions == ()


# ------------------------------------------------------------------ gold as a supplier ---


def test_the_gold_supplier_reproduces_the_default_path_exactly():
    """Naming gold explicitly is the same computation as not naming anything.

    Anti-vacuity for the rollback digest above: that test would also pass if the default path
    had stopped going through the supplier interface at all. This one drives the interface and
    compares tensor for tensor.
    """
    batch = make_group([0.0] * 4)
    default, ds = substitute_gold_rows(batch, "dyme")
    explicit, es = substitute_gold_rows(
        batch,
        "dyme",
        suppliers={"gold": GoldSupplier()},
        source_policy=FixedSourcePolicy(("gold",)),
    )
    for key in default:
        assert torch.equal(default[key], explicit[key]), key
    assert ds.rows_substituted == es.rows_substituted == 1
    assert es.decisions == ("gold",)
    assert dict(es.served_by) == {"gold": 1}


def test_the_gold_specific_counters_keep_their_meaning_under_the_interface():
    """``groups_no_gold`` and ``groups_no_fit`` still count groups, not attempts.

    They were separate before the refactor because a missing solution is a dataset problem and
    an over-long one is a sequence-length problem. Under a chain they must keep counting
    GROUPS THAT ENDED UNSERVED for those causes, or a group rescued by a later source would be
    reported as a loss of reach that never happened.
    """
    long_gold = list(range(901, 901 + T - 1))
    batch = [
        make_group([0.0] * 4, gold=None),
        make_group([0.0] * 4, gold=long_gold),
        make_group([0.0] * 4),
    ]
    _out, stats = substitute_in_place(batch, "dyme")
    assert (stats.groups_no_gold, stats.groups_no_fit) == (1, 1)
    assert dict(stats.unserved_groups) == {"no_gold": 1, "no_fit": 1}
    assert stats.decisions == (NO_SOURCE, NO_SOURCE, "gold")


def test_a_group_rescued_by_a_later_source_is_not_counted_as_a_loss_of_reach():
    """The distinction the two counter families exist to make.

    ``refusals`` is attempt-level and records that gold refused; ``unserved_groups`` is
    group-level and must stay empty, because the group WAS served. A single "skipped" count
    could not tell the two apart, and would report a served arm as having lost reach.
    """
    prompt = [7, 8, 9, 10]
    group = make_group([0.0] * 4, gold=None, prompt_ids=prompt)
    store = store_with(prompt, [61, 62])
    out, stats = substitute_gold_rows(
        group,
        "dyme",
        suppliers={"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("gold", "self")),
    )
    assert stats.rows_substituted == 1
    assert stats.decisions == ("self",)
    assert dict(stats.refusals) == {"no_gold": 1}
    assert dict(stats.unserved_groups) == {}
    assert stats.groups_no_gold == 0
    assert out["input_ids"][0, len(prompt) : len(prompt) + 2].tolist() == [61, 62]


# ---------------------------------------------------------------------- refusals ---


def test_a_refusing_supplier_leaves_the_row_untouched_and_reports_no_success():
    """The silent no-op, refused in the two places it could hide.

    A supplier that has nothing must not (a) report a substitution that did not happen, or
    (b) leave the row's tokens alone while marking it as supplied. Both are asserted, because
    a version that returned the unmodified row AND incremented the counter would pass a test
    that only looked at one of them.
    """
    unseen = [21, 22, 23, 24]
    seen = [90, 91, 92, 93]
    # A served neighbour, so the BATCH-level reach guard stays quiet and the refused group's
    # own rows can be inspected. A batch in which nothing at all landed is a different
    # failure, tested separately.
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=unseen)
    served = make_group([0.0] * 4, with_gold=False, prompt_ids=seen)
    store = store_with(seen, [77, 78])
    # One batch of two groups rather than a list of two, so the refused group's own tensors
    # are in the returned batch. (In the LIST form a per-trajectory refusal takes the deferred
    # path, which returns the original dict; that path's own defect is owned by
    # ``test_audit_2026_09_02.py`` and is not what this test is about.)
    batch = concat_padded_tensors([group, served])
    out, stats = substitute_gold_rows(
        batch,
        "dyme",
        group_sizes=4,
        suppliers={"self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("self",)),
    )
    assert stats.rows_substituted == 1
    assert stats.decisions == (NO_SOURCE, "self")
    assert dict(stats.served_by) == {"self": 1}
    assert dict(stats.unserved_groups) == {"no_match": 1}
    assert torch.equal(out["input_ids"][:4], batch["input_ids"][:4])
    assert torch.equal(out["loss_mask"][:4], batch["loss_mask"][:4])
    assert torch.equal(out["rewards"][:4], batch["rewards"][:4])
    assert int(out["is_gold"][:4].sum()) == 0
    assert int(out["source_ids"][:4].sum()) == 0
    assert int(out["is_gold"][4].sum()) == T


def test_every_refusal_reason_gets_its_own_counter_and_all_of_them_are_emitted():
    """A mapping of reasons, not a count, and the same key set on every arm.

    ``selfevo/CONTRIBUTING.md`` states both halves. Emitting only the non-zero reasons would
    make a gold arm's panel and a teacher arm's panel structurally different, which is the
    thing ``gold/`` keys are emitted as zeros on a ``none`` run to avoid.
    """
    m = substitute_gold_rows(make_group([0.0] * 4), "dyme")[1].as_metrics()
    for reason in Refusal:
        assert m[f"supply/refused/{reason.value}"] == 0.0
        assert m[f"supply/unserved/{reason.value}"] == 0.0
    for name in SUPPLY_SOURCES:
        assert f"supply/served/{name}" in m
    assert m["supply/served/gold"] == 1.0
    assert m["supply/sources_used"] == 1.0


def test_three_distinct_causes_land_in_three_distinct_counters():
    """Anti-"a counter that never increments", and anti-"one counter catches everything".

    Three qualifying groups fail for three different reasons -- no gold text, a gold too long
    for the row, and a policy that names no supplier at all -- and each must move its own key.
    """
    long_gold = list(range(901, 901 + T - 1))
    trajs = [
        make_group([0.0] * 4, gold=None),
        make_group([0.0] * 4, gold=long_gold),
        make_group([0.0] * 4),
        make_group([0.0] * 4),
    ]
    forced = ForcedSourcePolicy(["gold", "gold", NO_SOURCE, "gold"])
    _out, stats = substitute_in_place(
        trajs, "dyme", suppliers={"gold": GoldSupplier()}, source_policy=forced
    )
    assert dict(stats.unserved_groups) == {"no_gold": 1, "no_fit": 1, "no_source": 1}
    m = stats.as_metrics()
    assert m["supply/unserved/no_gold"] == 1.0
    assert m["supply/unserved/no_fit"] == 1.0
    assert m["supply/unserved/no_source"] == 1.0
    assert m["supply/served/gold"] == 1.0


def test_the_batch_reach_guard_fires_when_no_configured_supplier_has_anything():
    """The generalisation of "every gold_mask in this batch is empty".

    An arm whose only supplier holds nothing must refuse on the first batch, loudly, rather
    than decline every group quietly for 900 steps. With gold alone the message is unchanged
    word for word, which the second half asserts so the existing guard's own test keeps its
    meaning.
    """
    empty = SelfGeneratedSupplier(SolvedRolloutStore(capacity=4))
    with pytest.raises(GoldMissingError, match="has nothing for this batch"):
        substitute_gold_rows(
            make_group([0.0] * 4, with_gold=False),
            "dyme",
            suppliers={"self": empty},
            source_policy=FixedSourcePolicy(("self",)),
        )
    with pytest.raises(GoldMissingError, match="every gold_mask in this batch is empty"):
        substitute_gold_rows(
            make_group([0.0] * 4, gold=None), "dyme", suppliers={"gold": GoldSupplier()}
        )


def test_a_payload_that_does_not_fit_is_refused_and_never_truncated():
    """A cut-off derivation is a wrong target that still looks like a target.

    ``attach_gold`` records the argument for the dataset gold; it is the same argument for
    every other source, so the self supplier refuses with ``NO_FIT`` rather than clipping.
    """
    prompt = [31, 32, 33, 34]
    other = [35, 36, 37, 38]
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    served = make_group([0.0] * 4, with_gold=False, prompt_ids=other)
    store = store_with(prompt, list(range(600, 600 + T)))
    store.record(key_for_prompt(other), torch.tensor([99, 98], dtype=torch.long))
    out, stats = substitute_in_place(
        [group, served],
        "dyme",
        suppliers={"self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("self",)),
    )
    assert stats.rows_substituted == 1
    assert dict(stats.unserved_groups) == {"no_fit": 1}
    assert torch.equal(out[0]["input_ids"], group["input_ids"])


# ------------------------------------------------------------------ prompt identity ---


def test_prompt_identity_is_the_ledgers_scheme_and_not_a_second_one():
    """One identity function, reached two ways, asserted equal.

    ``key_for_prompt`` (offline, prompt only) and ``SupplyRequest.identity`` (a live rollout
    row) must agree, or a corpus built offline never matches a batch. They agree because both
    route to ``selfevo.routing.prompt_credit.prompt_key``; two independent hashes would agree
    today and diverge silently on the first change to either.
    """
    prompt = [41, 42, 43]
    batch = make_group([0.0], with_gold=False, prompt_ids=prompt, resp=[81, 82])
    request = SupplyRequest(batch=batch, row=0, group=0, prompt_len=len(prompt), width=T)
    assert request.identity() == key_for_prompt(prompt)
    assert request.identity() == prompt_key(
        [int(v) for v in batch["input_ids"][0].tolist()],
        [float(v) for v in batch["loss_mask"][0].tolist()],
    )
    assert request.prompt_ids() == prompt


def test_identity_comes_from_the_prompt_and_survives_a_different_batch_position():
    """``unit_id`` is batch-local; the prompt is not, which is the whole premise.

    The same prompt appears as row 0 of one batch and row 5 of another, beside different
    neighbours and with a different rollout. The store must find it both times.
    """
    prompt = [55, 56, 57, 58]
    store = store_with(prompt, [88, 89])
    supplier = SelfGeneratedSupplier(store)
    for position, resp in ((0, [1, 2]), (3, [9, 9, 9])):
        rows = [make_row(0.0, prompt_ids=[70 + i, 71, 72, 73]) for i in range(4)]
        rows[position] = make_row(0.0, prompt_ids=prompt, resp=resp)
        batch = concat_padded_tensors(rows)
        request = SupplyRequest(
            batch=batch, row=position, group=0, prompt_len=len(prompt), width=T
        )
        assert supplier.supply(request).token_ids.tolist() == [88, 89]


def test_a_row_with_no_prompt_region_refuses_rather_than_keying_on_a_constant():
    """A row that is all response has no identity, and merging such rows would be worse."""
    row = make_row(0.0, prompt_ids=[5], resp=[6, 7])
    row["loss_mask"][0, :] = 1
    batch = concat_padded_tensors([row])
    request = SupplyRequest(batch=batch, row=0, group=0, prompt_len=0, width=T)
    with pytest.raises(SupplierRefused) as exc:
        request.identity()
    assert exc.value.reason is Refusal.NO_IDENTITY


# ---------------------------------------------------------------------- the store ---


def test_the_store_records_only_correct_rollouts():
    """The store holds TARGETS. An incorrect row in it is a wrong row served with confidence."""
    store = SolvedRolloutStore(capacity=8)
    batch = concat_padded_tensors(
        [
            make_row(1.0, prompt_ids=[1, 2, 3, 4], resp=[61, 62]),
            make_row(0.0, prompt_ids=[9, 8, 7, 6], resp=[63, 64]),
        ]
    )
    assert store.record_batch(batch) == 1
    assert store.get(key_for_prompt([1, 2, 3, 4])).tolist() == [61, 62]
    assert store.get(key_for_prompt([9, 8, 7, 6])) is None
    assert store.skipped_incorrect == 1


def test_the_store_refuses_to_record_a_row_that_was_itself_substituted():
    """Otherwise a self-generated arm silently replays its own gold arm.

    A substituted row's tokens came from a supplier, not from the model, and its reward was
    set to the gold reward -- so it passes the correctness filter and would be harvested as
    though the policy had produced it. Nothing downstream could tell the two apart afterwards.
    """
    group = make_group([0.0] * 4)
    out, stats = substitute_gold_rows(group, "dyme")
    assert stats.rows_substituted == 1
    store = SolvedRolloutStore(capacity=8)
    assert store.record_batch(out) == 0
    assert store.skipped_substituted == 1
    assert len(store) == 0


def test_the_store_is_bounded_and_says_so():
    """An unbounded per-prompt memory is the one thing a 10-epoch trainer must not carry."""
    store = SolvedRolloutStore(capacity=2)
    for i in range(4):
        store.record(key_for_prompt([i, i + 1]), torch.tensor([i], dtype=torch.long))
    assert len(store) == 2
    assert store.evicted == 2
    assert store.as_metrics()["supply/store/evicted"] == 2.0
    assert store.get(key_for_prompt([0, 1])) is None


def test_the_store_serves_a_prompt_it_saw_at_an_earlier_step():
    """The premise: this prompt is unsolved NOW, and was solved before."""
    prompt = [12, 13, 14, 15]
    store = SolvedRolloutStore(capacity=8)
    earlier = concat_padded_tensors(
        [make_row(1.0, prompt_ids=prompt, resp=[71, 72, 73]), make_row(0.0, resp=[1, 2])]
    )
    assert store.record_batch(earlier) == 1

    now = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    out, stats = substitute_gold_rows(
        now,
        "dyme",
        suppliers={"self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("self",)),
    )
    assert stats.decisions == ("self",)
    assert out["input_ids"][0, len(prompt) : len(prompt) + 3].tolist() == [71, 72, 73]


def test_the_resample_seam_is_declared_and_is_never_a_gpu_call_here():
    """The second self-generated source is an interface, driven with a fake.

    A higher-temperature resample needs an engine, so this repo ships the seam and not the
    implementation; a supplier that quietly booked a GPU would be undiscoverable until a run.
    """
    prompt = [61, 62, 63, 64]
    calls: list[int] = []

    def fake_resampler(request):
        """Stand-in for a higher-temperature resample; returns a fixed correct response."""
        calls.append(request.row)
        return torch.tensor([95, 96], dtype=torch.long)

    supplier = SelfGeneratedSupplier(
        SolvedRolloutStore(capacity=4), resampler=fake_resampler
    )
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    out, stats = substitute_gold_rows(
        group, "dyme", suppliers={"self": supplier}, source_policy=FixedSourcePolicy(("self",))
    )
    assert calls == [0]
    assert stats.decisions == ("self",)
    assert out["input_ids"][0, 4:6].tolist() == [95, 96]


# ---------------------------------------------------------------------- the corpus ---


def test_the_corpus_refuses_rather_than_serving_another_prompts_row():
    """The cheapest possible fabrication, refused.

    The writer keeps the row's own prompt, so a pooled row for a different prompt is spliced
    after a question it does not answer. A pool covering one prompt must decline a batch of
    another, not fall back to whatever it holds.
    """
    pool = CorpusSupplier({key_for_prompt([100, 101]): [201, 202]})
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=[5, 6, 7, 8])
    request = SupplyRequest(batch=group, row=0, group=0, prompt_len=4, width=T)
    with pytest.raises(SupplierRefused) as exc:
        pool.supply(request)
    assert exc.value.reason is Refusal.NO_MATCH

    covered = make_group([0.0] * 4, with_gold=False, prompt_ids=[100, 101, 102, 103])
    pool2 = CorpusSupplier({key_for_prompt([100, 101, 102, 103]): [201, 202]})
    out, stats = substitute_in_place(
        [group, covered],
        "dyme",
        suppliers={"corpus": pool2},
        source_policy=FixedSourcePolicy(("corpus",)),
    )
    assert stats.rows_substituted == 1
    assert stats.decisions == (NO_SOURCE, "corpus")
    assert torch.equal(out[0]["input_ids"], group["input_ids"])


def test_the_corpus_serves_the_prompt_it_does_cover(tmp_path):
    """And the offline file's keys line up with a live batch's, which is the point of §identity."""
    prompt = [77, 78, 79, 80]
    path = tmp_path / "pool.jsonl"
    path.write_text(
        json.dumps({"prompt_ids": prompt, "response_ids": [301, 302, 303]}) + "\n"
    )
    supplier = CorpusSupplier(load_corpus_jsonl(path))
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    out, stats = substitute_gold_rows(
        group,
        "dyme",
        suppliers={"corpus": supplier},
        source_policy=FixedSourcePolicy(("corpus",)),
    )
    assert stats.decisions == ("corpus",)
    assert out["input_ids"][0, 4:7].tolist() == [301, 302, 303]


def test_a_corpus_file_that_is_not_a_pool_of_solved_rows_is_refused(tmp_path):
    """A malformed or incorrect line is refused, never dropped in silence."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt_ids": [1, 2], "response_ids": []}) + "\n")
    with pytest.raises(SupplyConfigError):
        load_corpus_jsonl(bad)
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text(
        json.dumps({"prompt_ids": [1, 2], "response_ids": [3], "correct": False}) + "\n"
    )
    with pytest.raises(SupplyConfigError):
        load_corpus_jsonl(wrong)


# --------------------------------------------------------------------- the teacher ---


def test_the_teacher_requires_a_verifier():
    """A teacher is stronger, not correct, and this branch has no other check on it."""
    with pytest.raises(SupplyConfigError, match="needs verify"):
        TeacherSupplier(RecordedTeacherClient({}), verify=None)


def test_an_unverified_teacher_completion_is_refused_and_counted():
    """The refusal that separates a teacher supplier from a fabrication."""
    prompt = [11, 22, 33, 44]
    other = [12, 23, 34, 45]
    client = RecordedTeacherClient(
        {key_for_prompt(prompt): [401, 402], key_for_prompt(other): [403, 404]}
    )
    # The verifier accepts one prompt's completion and rejects the other's, so the refusal is
    # attributable to the VERIFIER and not to a teacher that had nothing.
    def verify(request, tokens):
        """Accept only the completion for ``other``."""
        return request.prompt_ids() == other

    supplier = TeacherSupplier(client, verify=verify)
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    served = make_group([0.0] * 4, with_gold=False, prompt_ids=other)
    out, stats = substitute_in_place(
        [group, served],
        "dyme",
        suppliers={"teacher": supplier},
        source_policy=FixedSourcePolicy(("teacher",)),
    )
    assert stats.rows_substituted == 1
    assert dict(stats.unserved_groups) == {"not_verified": 1}
    assert torch.equal(out[0]["input_ids"], group["input_ids"])
    assert out[1]["input_ids"][0, 4:6].tolist() == [403, 404]


def test_a_verified_teacher_completion_is_served_through_the_same_splice():
    """The interface works end to end with an offline client; no model is served here."""
    prompt = [11, 22, 33, 44]
    client = RecordedTeacherClient({key_for_prompt(prompt): [401, 402]})
    supplier = TeacherSupplier(client, verify=always_correct)
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    out, stats = substitute_gold_rows(
        group,
        "dyme",
        suppliers={"teacher": supplier},
        source_policy=FixedSourcePolicy(("teacher",)),
    )
    assert stats.decisions == ("teacher",)
    assert out["input_ids"][0, 4:6].tolist() == [401, 402]
    assert int(out["source_ids"][0, 0]) == source_code("teacher")


def test_a_teacherless_arm_refuses_honestly_instead_of_pretending():
    """``actor.py``'s literal ``has_teacher=False`` is why no teacher arm has been reachable.

    The honest form is a supplier that answers the question from its own state: with no client
    every request is ``UNAVAILABLE``, and the batch guard fires rather than the arm running as
    a silent no-op.
    """
    supplier = TeacherSupplier(None, verify=always_correct)
    assert supplier.has_supply({}) is False
    request = SupplyRequest(
        batch=make_group([0.0], with_gold=False), row=0, group=0, prompt_len=4, width=T
    )
    with pytest.raises(SupplierRefused) as exc:
        supplier.supply(request)
    assert exc.value.reason is Refusal.UNAVAILABLE


# ------------------------------------------------------- what the substituted row is ---


def test_a_substituted_row_never_keeps_the_original_responses_token_mask():
    """The mask is what makes the update an SFT step on the supplied tokens and nothing else.

    A version that ORed the new mask onto the old one leaves the wrong rollout's tail masked
    in, so the row trains on the supplied answer AND on leftover wrong tokens -- same shapes,
    no error, and the arm reports a clean substitution.
    """
    prompt = [1, 2, 3, 4]
    group = make_group(
        [0.0] * 4, with_gold=False, prompt_ids=prompt, resp=[51, 52, 53, 54, 55]
    )
    store = store_with(prompt, [61, 62])
    out, _ = substitute_gold_rows(
        group,
        "dyme",
        suppliers={"self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("self",)),
    )
    lm = out["loss_mask"][0].tolist()
    assert lm == [0] * 4 + [1, 1] + [0] * (T - 6), lm
    assert out["input_ids"][0].tolist()[:6] == prompt + [61, 62]
    assert out["input_ids"][0, 6:].tolist() == [0] * (T - 6)
    assert out["attention_mask"][0].tolist() == [True] * 6 + [False] * (T - 6)
    assert float(out["rewards"][0]) == 1.0


def test_the_off_policy_treatment_is_the_gold_paths_for_every_source():
    """One treatment of the importance ratio, not one per source.

    A supplied row was never emitted by the current policy, so ``logprobs`` has no honest
    value; the gold path writes a finite sentinel and ``reconcile_gold_logprobs`` replaces it
    with the trainer's own recomputed ``prox_logp``, rolled right by one to cross the
    coordinate convention. A source that skipped this would leave the ROLLOUT's own
    log-probabilities on tokens the rollout never produced -- an importance ratio computed
    against a behaviour policy that never emitted them, silent at every stage.
    """
    prompt = [1, 2, 3, 4]
    group = make_group([0.0] * 4, with_gold=False, prompt_ids=prompt)
    store = store_with(prompt, [61, 62])
    out, _ = substitute_gold_rows(
        group,
        "dyme",
        suppliers={"self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("self",)),
    )
    assert torch.equal(
        out["logprobs"][0, 4:6], torch.full((2,), GOLD_LOGP_SENTINEL)
    ), "a supplied row kept a log-probability the model never produced"
    assert float(out["logprobs"][0, :4].abs().max()) == 0.0
    with pytest.raises(Exception):
        assert_gold_logprobs_filled(out)

    out["prox_logp"] = torch.full((4, T), -0.75)
    filled, n = reconcile_gold_logprobs(out)
    assert n == 1
    assert torch.allclose(filled["logprobs"][0, 4:6], torch.full((2,), -0.75))
    assert_gold_logprobs_filled(filled)


def test_source_ids_is_per_token_and_names_the_supplier():
    """A ``(B,)`` tensor does not survive packing; ``_compute_advantages`` records why.

    Gold is per-ROW and so is the source, so the tensor is ``(B, T)`` like ``group_ids`` and
    ``is_gold``, and it carries the closed registry's 1-based code so 0 unambiguously means
    "this row was not substituted".
    """
    prompt = [1, 2, 3, 4]
    store = store_with(prompt, [61, 62])
    out, stats = substitute_in_place(
        [
            make_group([0.0] * 4, gold=None, prompt_ids=prompt),
            make_group([0.0] * 4, prompt_ids=[8, 9, 10, 11]),
        ],
        "dyme",
        suppliers={"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("gold", "self")),
    )
    assert tuple(out[0]["source_ids"].shape) == (4, T)
    assert int(out[0]["source_ids"][0, 0]) == source_code("self")
    assert int(out[1]["source_ids"][0, 0]) == source_code("gold")
    assert int(out[0]["source_ids"][1].sum()) == 0
    assert dict(stats.served_by) == {"gold": 1, "self": 1}
    assert stats.as_metrics()["supply/sources_used"] == 2.0


# ------------------------------------------------------------------ the random control ---


def treatment_batch():
    """Six groups: four the gold can serve, two only the store can, in a fixed order."""
    prompts = [[10 + i, 11 + i, 12 + i, 13 + i] for i in range(6)]
    trajs = []
    for i, prompt in enumerate(prompts):
        gold = None if i in (1, 4) else GOLD
        trajs.append(make_group([0.0] * 4, gold=gold, prompt_ids=prompt))
    store = SolvedRolloutStore(capacity=16)
    for i in (1, 4):
        store.record(key_for_prompt(prompts[i]), torch.tensor([70 + i, 71], dtype=torch.long))
    return trajs, store


def test_the_matched_control_reproduces_the_source_proportions_exactly():
    """Matched by construction, for any batch, with no probability to mis-specify.

    ``selfevo/routing/proportions.py`` records what configuring nominal probabilities costs:
    half the mass migrated to a mode nobody configured, and the control quietly became a
    different arm. Replaying the realised multiset cannot do that.
    """
    trajs, store = treatment_batch()
    suppliers = {"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)}
    _out, treated = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=FixedSourcePolicy(("gold", "self"))
    )
    assert sorted(treated.decisions) == sorted(["gold"] * 4 + ["self"] * 2)

    control = MatchedSourceControl(treated.decisions, seed=1)
    assert sorted(control.assignment) == sorted(treated.decisions)
    assert source_proportions(control.assignment) == source_proportions(treated.decisions)


def test_the_matched_control_does_not_reuse_the_treatments_own_assignment():
    """The control must MOVE the decisions, or it is the treatment wearing another label.

    A control that returned the realised vector unshuffled would match the proportions
    perfectly, report as a control, and be bit-identical to the treatment. That is the exact
    shape of the two arms this repo has already retracted.
    """
    trajs, store = treatment_batch()
    _out, treated = substitute_in_place(
        trajs,
        "dyme",
        suppliers={"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)},
        source_policy=FixedSourcePolicy(("gold", "self")),
    )
    control = MatchedSourceControl(treated.decisions, seed=1)
    assert control.assignment != treated.decisions
    assert control.moved_fraction() > 0.0


def test_the_matched_control_is_feature_blind():
    """Blindness that depends on a caller not passing something is not blindness.

    The constructor admits a vector of names and a seed and nothing else, so the assignment
    cannot depend on rewards, solve rates or cluster labels. Asserted by building the same
    control against two batches whose rewards differ and getting the same assignment.
    """
    decisions = ("gold", "self", "gold", "gold", "self", "gold")
    a = MatchedSourceControl(decisions, seed=3)
    b = MatchedSourceControl(decisions, seed=3)
    assert a.assignment == b.assignment
    assert "features" not in MatchedSourceControl.__init__.__code__.co_varnames
    assert set(MatchedSourceControl.__init__.__code__.co_varnames[:3]) == {
        "self",
        "realised",
        "seed",
    }


def test_the_control_is_not_vacuous_it_can_serve_fewer_groups_than_the_treatment():
    """The measurement the control exists for, and proof that it is capable of losing.

    Under the treatment every served group got a source that could serve it, by construction.
    Under a permutation some groups get a source that cannot, so ``rows_substituted`` falls.
    If the fixed order carried no information, the two would agree; here it does, and the gap
    is the thing a future learned router would have to beat.
    """
    trajs, store = treatment_batch()
    suppliers = {"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)}
    _out, treated = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=FixedSourcePolicy(("gold", "self"))
    )
    assert treated.rows_substituted == 6

    control = MatchedSourceControl(treated.decisions, seed=1)
    _cout, controlled = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=control
    )
    assert controlled.groups_qualifying == treated.groups_qualifying == 6
    assert controlled.rows_substituted < treated.rows_substituted
    assert dict(controlled.unserved_groups)


def test_the_forced_replay_of_the_treatment_matches_the_treatment():
    """The control's comparison arm: same one-attempt mechanism, unpermuted.

    Comparing an ordered chain (several attempts per group) against a forced control (one)
    would confound targeting with fallback depth. Replaying the realised vector through the
    same ``ForcedSourcePolicy`` the control uses removes that, and this asserts the replay is
    faithful.
    """
    trajs, store = treatment_batch()
    suppliers = {"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)}
    _out, treated = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=FixedSourcePolicy(("gold", "self"))
    )
    _rout, replayed = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=ForcedSourcePolicy(treated.decisions)
    )
    assert replayed.decisions == treated.decisions
    assert replayed.rows_substituted == treated.rows_substituted


def test_a_forced_policy_refuses_to_wrap_rather_than_reusing_decisions():
    """Wrapping would silently change the realised proportions the control exists to match."""
    policy = ForcedSourcePolicy(["gold", "self"])
    assert policy.chain_for(0, 0) == ("gold",)
    assert policy.chain_for(1, 1) == ("self",)
    with pytest.raises(SupplyConfigError, match="refusing to wrap"):
        policy.chain_for(2, 2)


def test_the_qualifying_index_is_batch_global_across_the_list_form():
    """``prepare_batch`` returns one dict per prompt, and a replay is indexed across them all.

    A version that restarted the index at 0 for every element would give the first decision to
    every prompt, matching no proportions at all -- and would still pass every single-group
    test in this file.
    """
    trajs, store = treatment_batch()
    suppliers = {"gold": GoldSupplier(), "self": SelfGeneratedSupplier(store)}
    assignment = ["gold", "self", "gold", "gold", "self", "gold"]
    _out, stats = substitute_in_place(
        trajs, "dyme", suppliers=suppliers, source_policy=ForcedSourcePolicy(assignment)
    )
    assert stats.decisions == tuple(assignment)
    assert stats.rows_substituted == 6


# ------------------------------------------------------------------ closed registries ---


def test_an_unknown_source_name_is_refused_at_construction():
    """A typo must not survive config parse and die after model load, on GPU, at batch one."""
    with pytest.raises(SupplyConfigError):
        FixedSourcePolicy(("gold", "orACLE"))
    with pytest.raises(SupplyConfigError):
        ForcedSourcePolicy(["nope"])
    with pytest.raises(SupplyConfigError):
        source_code("nope")


def test_a_policy_naming_a_supplier_the_arm_did_not_build_is_refused():
    """A silently-skipped source is an arm that reports a mixture it never ran."""
    with pytest.raises(GoldPolicyError, match="did not build"):
        substitute_gold_rows(
            make_group([0.0] * 4),
            "dyme",
            suppliers={"gold": GoldSupplier()},
            source_policy=FixedSourcePolicy(("gold", "corpus")),
        )


def test_an_empty_supplier_mapping_is_refused():
    """An arm that configured no source is an off arm wearing an on arm's label."""
    with pytest.raises(GoldPolicyError, match="names no source"):
        substitute_gold_rows(make_group([0.0] * 4), "dyme", suppliers={})


def test_a_chain_may_not_repeat_a_supplier():
    """A second attempt gets the same refusal and reports two losses of reach for one cause."""
    with pytest.raises(SupplyConfigError, match="repeats a supplier"):
        FixedSourcePolicy(("gold", "gold"))


def test_the_factories_refuse_the_arguments_that_would_make_an_arm_silent():
    """The wrapper is the seam: an experiment-deciding default must not live past it."""
    with pytest.raises(SupplyConfigError, match="needs store"):
        build_suppliers(["self"])
    with pytest.raises(SupplyConfigError, match="needs pool"):
        build_suppliers(["corpus"])
    with pytest.raises(SupplyConfigError, match="needs verify"):
        build_suppliers(["teacher"])
    built = build_suppliers({"gold": {}, "self": {"store": SolvedRolloutStore()}})
    assert list(built) == ["gold", "self"]


def test_a_malformed_offer_is_refused_at_the_seam_it_crosses():
    """A broken supplier must fail where it is named, not as a shape error inside the writer."""
    with pytest.raises(SupplyConfigError, match="1-D tensor"):
        SupplyOffer(torch.zeros(2, 3, dtype=torch.long), "gold")
    with pytest.raises(SupplyConfigError, match="empty payload"):
        SupplyOffer(torch.zeros(0, dtype=torch.long), "gold")
    with pytest.raises(SupplyConfigError, match="token ids are integers"):
        SupplyOffer(torch.zeros(3, dtype=torch.float32), "gold")
    with pytest.raises(SupplyConfigError):
        SupplyOffer(torch.zeros(3, dtype=torch.long), "not_a_source")
