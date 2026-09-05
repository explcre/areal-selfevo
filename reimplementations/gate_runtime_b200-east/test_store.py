#!/usr/bin/env python3
"""Tests for the permanent task store and the routing seam, written to attack them.

The store's whole value is that it refuses ambiguity, so every test here tries to store an
ambiguous record and requires the refusal. The one the coordinator singled out -- that an
older-schema record is read correctly or refused loudly, never reinterpreted silently -- is
test 10.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasksource.routing import SourceProfile, plan, route  # noqa: E402
from tasksource.store import (FLOORS, SCHEMA_VERSION, StoredTask, TaskStore,  # noqa: E402
                              band_for, content_hash, floor_for, make_dedup)

TEXT = ("Find the number of ordered pairs of positive integers (a, b) with a + b = 100 "
        "and gcd(a, b) = 5.")


def good_dedup(text=TEXT):
    return {"held_out": make_dedup("held_out", 0.21, 0.45, text),
            "training_pool": make_dedup("training_pool", 0.30, 0.60, text),
            "run_buffer": make_dedup("run_buffer", 0.11, 0.60, text)}


def good(**over):
    base = dict(text=TEXT, answer="9", source_type="retrieved",
                provenance={"corpus": "math500", "row": 42, "licence": "MIT"},
                verification={"verdict": "verified", "backend": "solver_consensus",
                              "comparator": "symbolic_compare"},
                difficulty={"success_rate": 0.5, "measured_under_prompt": "bare_v1",
                            "n_samples": 8},
                dedup=good_dedup(), cost={"produce_tokens": 0, "score_tokens": 1200})
    base.update(over)
    return StoredTask(**base)


def expect_refusal(fn, why):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError("stored an ambiguous record: %s" % why)


def test_01_distilled_must_name_its_teacher():
    expect_refusal(lambda: good(source_type="distilled",
                                provenance={"prompt_version": "v1"}),
                   "a distilled task without teacher_model cannot be audited for collusion")
    good(source_type="distilled",
         provenance={"teacher_model": "qwen3.8-27b", "teacher_version": "local-2026-09",
                     "prompt_version": "distil_v1"})


def test_02_retrieved_must_carry_corpus_row_and_licence():
    for missing in ("corpus", "row", "licence"):
        p = {"corpus": "math500", "row": 42, "licence": "MIT"}
        del p[missing]
        expect_refusal(lambda p=p: good(provenance=p), "retrieved without %s" % missing)


def test_03_generated_must_name_prompt_version_and_exemplars():
    expect_refusal(lambda: good(source_type="generated", provenance={"prompt_version": "g1"}),
                   "generated without exemplars cannot be checked against its conditioning")
    good(source_type="generated", provenance={"prompt_version": "g1", "exemplars": []})


def test_04_a_verdict_needs_its_comparator():
    for k in ("verdict", "backend", "comparator"):
        v = {"verdict": "verified", "backend": "b", "comparator": "c"}
        del v[k]
        expect_refusal(lambda v=v: good(verification=v), "verification without %s" % k)


def test_05_a_refutation_needs_its_witness():
    expect_refusal(lambda: good(verification={"verdict": "refuted", "backend": "b",
                                              "comparator": "symbolic_compare"}),
                   "a refutation with no witness cannot be rechecked")
    good(verification={"verdict": "refuted", "backend": "b", "comparator": "symbolic_compare",
                       "witness": "12"})


def test_06_a_success_rate_needs_the_prompt_it_was_measured_under():
    expect_refusal(lambda: good(difficulty={"success_rate": 0.5}),
                   "difficulty under a scaffold is a different quantity from difficulty bare")
    good(difficulty={"success_rate": 0.5, "measured_under_prompt": "scaffolded_v2"})
    good(difficulty={})  # no claim made is fine; an unattributed claim is not


def test_07_dedup_needs_all_three_reference_sets_with_floors():
    for ref in ("held_out", "training_pool", "run_buffer"):
        d = good_dedup()
        del d[ref]
        expect_refusal(lambda d=d: good(dedup=d), "dedup collapsed, missing %s" % ref)
    d = good_dedup()
    del d["run_buffer"]["false_positive_floor"]
    expect_refusal(lambda: good(dedup=d), "a dedup score without its floor is uninterpretable")


def test_08_cost_is_required_both_ways():
    for k in ("produce_tokens", "score_tokens"):
        c = {"produce_tokens": 1, "score_tokens": 2}
        del c[k]
        expect_refusal(lambda c=c: good(cost=c), "cost missing %s" % k)


def test_09_identity_is_content_addressed_and_whitespace_stable():
    a = good()
    b = good(text="  " + TEXT.replace(" ", "  ") + "\n")
    assert a.content_hash == b.content_hash, "whitespace changed the content hash"
    assert a.task_id == b.task_id
    c = good(text=TEXT + " Also compute a-b.")
    assert c.content_hash != a.content_hash, "different statements collided"
    assert content_hash(TEXT) == a.content_hash


def test_10_an_unknown_schema_version_is_refused_not_reinterpreted():
    """The one that matters: an older row must never be read with today's field meanings."""
    with tempfile.TemporaryDirectory() as d:
        st = TaskStore(d)
        st.append(good())
        assert len(TaskStore.read(st.path)) == 1, "a current-version record must read back"
        # A record from a different schema, appended directly as an older writer would have.
        with open(st.path, "a") as fh:
            row = good().to_json()
            row["schema_version"] = SCHEMA_VERSION - 1
            fh.write(json.dumps(row) + "\n")
        try:
            TaskStore.read(st.path)
        except ValueError as e:
            assert "schema_version" in str(e) and "Refusing" in str(e)
        else:
            raise AssertionError("an older-schema record was read with today's field meanings")
        # And a row with no version at all must not be guessed at either.
        with tempfile.TemporaryDirectory() as d2:
            p = os.path.join(d2, "x.jsonl")
            row = good().to_json()
            row.pop("schema_version")
            open(p, "w").write(json.dumps(row) + "\n")
            try:
                TaskStore.read(p)
            except ValueError as e:
                assert "no schema_version" in str(e)
            else:
                raise AssertionError("a record with no schema_version was read anyway")


def test_11_the_store_is_append_only():
    with tempfile.TemporaryDirectory() as d:
        st = TaskStore(d)
        st.append(good())
        first = open(st.path).read()
        st.append(good(text=TEXT + " Variant two for appending."))
        after = open(st.path).read()
        assert after.startswith(first), "appending rewrote an existing record"
        assert len(after.strip().splitlines()) == 2


def test_12_floors_are_measured_per_reference_and_length_band():
    short = "Find x if 2x = 6."
    long_ = "A" * 400
    assert band_for(short) == (0, 60) and band_for(long_) == (260, 10**9)
    assert floor_for("run_buffer", 0.60, short) == FLOORS["run_buffer"][0.60][(0, 60)]
    assert floor_for("run_buffer", 0.60, short) > floor_for("run_buffer", 0.60, long_)
    assert floor_for("held_out", 0.45, short) > 0
    assert floor_for("held_out", 0.99, short) is None, "an unmeasured floor must be None"
    d = make_dedup("run_buffer", 0.7, 0.60, short)
    assert d["false_positive_floor"] is not None and d["length_band"] == "0-60"


def test_13_routing_refuses_to_rank_on_an_unmeasured_field():
    profiles = [
        SourceProfile("retrieved", "bounded", False, False, True, key_refuted_rate=0.02,
                      key_coverage=0.4, tokens_per_accepted=0),
        SourceProfile("generated", "unbounded", True, True, False, key_refuted_rate=None,
                      key_coverage=None, tokens_per_accepted=11000),
    ]
    r = route(profiles, "reliable_keys")
    assert r[0][0] == "retrieved"
    assert r[-1][0] == "generated" and "NOT MEASURED" in r[-1][1], (
        "an unmeasured field was ranked as if it were zero, which declares a winner by accident")
    assert [n for n, _ in route(profiles, "reliable_keys", exclude_collusion=True)] == ["retrieved"]
    assert route(profiles, "unbounded_supply")[0][0] == "generated"
    try:
        route(profiles, "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("routed on an unknown need")
    p = plan(profiles)
    assert set(p) and all(isinstance(v, list) for v in p.values())


def test_14_the_two_dedup_booleans_answer_different_questions():
    """A score between the floor and the threshold must read as ABOVE floor, BELOW threshold.

    This is the case v1 could not express: it reported one boolean, named for the floor and
    computed from the threshold, so an overlap that was real but not actionable was recorded
    as if it were nothing at all.
    """
    short = "x" * 40                       # 0-60 band, run_buffer floor 0.2917 at 0.60
    d = make_dedup("run_buffer", 0.40, 0.60, short)
    assert d["false_positive_floor"] == 0.2917
    assert d["above_floor"] is True, "0.40 exceeds the 0.2917 chance floor for this band"
    assert d["above_threshold"] is False, "0.40 does not reach the 0.60 action threshold"
    # and the two must not be the same field under two names
    d2 = make_dedup("run_buffer", 0.70, 0.60, short)
    assert (d2["above_floor"], d2["above_threshold"]) == (True, True)
    d3 = make_dedup("run_buffer", 0.10, 0.60, short)
    assert (d3["above_floor"], d3["above_threshold"]) == (False, False)


def test_15_production_and_verification_costs_are_not_collapsed():
    """A corpus is free to read and can still be the dearest thing to trust.

    Ranking on production+verification added together would have said retrieval was the
    expensive source to produce, when it costs nothing to produce and is merely expensive to
    verify. The two needs must rank differently on the same profiles.
    """
    ps = [SourceProfile(name="retrieved", supply="bounded", targetable=False,
                        collusion=False, licence_constrained=True,
                        tokens_per_accepted=0, verify_tokens_per_accepted=9716),
          SourceProfile(name="generated", supply="unbounded", targetable=True,
                        collusion=True, licence_constrained=False,
                        tokens_per_accepted=5135, verify_tokens_per_accepted=3200)]
    assert route(ps, "cheap_tokens")[0][0] == "retrieved", "free to read must rank first"
    assert route(ps, "cheap_verification")[0][0] == "generated", "cheap to trust differs"
    # and an unmeasured verification cost is not ranked as zero
    ps[1].verify_tokens_per_accepted = None
    assert route(ps, "cheap_verification")[-1][0] == "generated"
    assert "NOT MEASURED" in route(ps, "cheap_verification")[-1][1]


def test_16_a_tied_ranking_says_so_rather_than_implying_an_order():
    """Equal measured values must not be presented as first, second and third.

    Every source refuted 0 of its decided keys, so `reliable_keys` returned an order that
    was really list order. An arbitrary order read as a result is how a non-finding becomes
    a claim.
    """
    ps = [SourceProfile(name=n, supply="unbounded", targetable=True, collusion=False,
                        licence_constrained=False, key_refuted_rate=0.0)
          for n in ("a", "b", "c")]
    got = route(ps, "reliable_keys")
    assert all("TIED" in why for _, why in got), "a three-way tie was presented as a ranking"
    ps[0].key_refuted_rate = 0.5
    got = route(ps, "reliable_keys")
    assert not any("TIED" in why for _, why in got), "a genuine difference was called a tie"
    assert got[0][0] in ("b", "c") and got[-1][0] == "a"



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (t.__name__, e))
        except Exception as e:
            failed += 1
            print("ERROR %s: %r" % (t.__name__, e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
