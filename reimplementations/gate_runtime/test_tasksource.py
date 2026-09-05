#!/usr/bin/env python3
"""Tests for the task-source layer, written to ATTACK it rather than to exercise it.

The four failures an auditor should go for here, each with a test that fails if the property
is absent:

  1. the shared novelty buffer is three buffers with one name;
  2. the contamination check catches only exact matches, not a paraphrase;
  3. provenance is dropped at the first transformation and never reaches the artifact;
  4. a source that produces nothing contributes a silent zero instead of a reported failure.

Plus the ones that matter for money and for correctness of the headline number.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasksource.backends import HostedTeacher  # noqa: E402
from tasksource.base import Provenance, SourceResult, TaskRecord  # noqa: E402
from tasksource.pipeline import TaskPipeline  # noqa: E402
from tasksource.registry import ContaminationFilter, SharedNoveltyBuffer  # noqa: E402
from tasksource.similarity import similarity  # noqa: E402
from tasksource.sources import ModelWrittenSource, RetrievedSource, parse_problem_answer  # noqa: E402

HELD_OUT = [
    "Let n be a positive integer such that n^2 - 3n + 1 divides n^3 - 2n + 5. "
    "Find the sum of all such n.",
    "A bag contains 5 red and 7 blue marbles. Two are drawn without replacement. "
    "What is the probability that both are red?",
]


def prov(src="test", origin="unit", lic="none"):
    """A minimal valid provenance."""
    return Provenance(source=src, origin=origin, licence=lic)


class FakeSource:
    """A source that returns exactly what the test tells it to."""

    def __init__(self, name, tasks, ok=True, reason=""):
        self.name, self._tasks, self._ok, self._reason = name, tasks, ok, reason

    def fetch(self, n, rng):
        if not self._ok:
            return SourceResult.failure(self.name, n, self._reason)
        return SourceResult(self.name, list(self._tasks), attempted=n, ok=True)


def test_1_novelty_buffer_is_shared_across_sources():
    """A near-duplicate from a DIFFERENT source must lose to the one already accepted."""
    original = ("Find the number of ordered pairs of positive integers (a, b) with "
                "a + b = 100 and gcd(a, b) = 5.")
    near_dupe = ("Find the number of ordered pairs of positive integers (a,b) such that "
                 "a + b = 100 and gcd(a,b) = 5.")
    buf = SharedNoveltyBuffer(threshold=0.60)
    pipe = TaskPipeline(buf, ContaminationFilter(held_out=[], threshold=0.45))
    a = FakeSource("generated", [TaskRecord(original, "9", prov("generated"))])
    b = FakeSource("retrieved", [TaskRecord(near_dupe, "9", prov("retrieved"))])
    stats = pipe.run([a, b], 1, random.Random(0))
    assert stats["generated"].accepted == 1, "the first source's task should be accepted"
    assert stats["retrieved"].accepted == 0, (
        "a near-duplicate from a second source was accepted: the buffer is NOT shared")
    assert stats["retrieved"].rejected_duplicate == 1
    assert stats["retrieved"].duplicate_of.get("generated") == 1, (
        "a cross-source collision must name the source that already held the near-duplicate")
    assert len(buf.texts) == 1 and buf.owners == ["generated"]


def test_2_contamination_catches_a_paraphrase_not_only_an_exact_match():
    """A reworded held-out problem must be rejected; that is the whole point of the filter."""
    filt = ContaminationFilter(held_out=HELD_OUT, threshold=0.45)
    exact = HELD_OUT[0]
    paraphrase = ("Suppose n is a positive integer for which n^2 - 3n + 1 is a divisor of "
                  "n^3 - 2n + 5. Compute the sum of all such integers n.")
    unrelated = ("Compute the area of a triangle whose vertices are at (0,0), (4,0) "
                 "and (0,3) in the coordinate plane.")
    ok_e, sim_e, _ = filt.check(exact)
    ok_p, sim_p, _ = filt.check(paraphrase)
    ok_u, sim_u, _ = filt.check(unrelated)
    assert not ok_e, "an exact held-out problem was accepted"
    assert not ok_p, (
        "a PARAPHRASE of a held-out problem was accepted (similarity %.3f < threshold): the "
        "filter only catches exact matches and the held-out set is not protected" % sim_p)
    assert ok_u, "an unrelated problem was rejected as contaminated (similarity %.3f)" % sim_u
    assert sim_p > sim_u, "the paraphrase must score more similar than an unrelated problem"


def test_3_provenance_survives_to_the_artifact():
    """Every emitted record must carry source, origin and licence after all transformations."""
    t = TaskRecord("Find the least positive integer n such that n! is divisible by 2^10.",
                   "12", Provenance(source="retrieved", origin="math500#42",
                                    licence="MIT (AZR redistribution of MATH)",
                                    detail={"corpus": "math500", "row": 42}))
    pipe = TaskPipeline(SharedNoveltyBuffer(), ContaminationFilter(held_out=[]))
    pipe.run([FakeSource("retrieved", [t])], 1, random.Random(0))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tasks.jsonl")
        assert pipe.write_artifacts(path) == 1
        row = json.loads(open(path).read().strip())
    assert "provenance" in row, "provenance was dropped before the artifact"
    for field in ("source", "origin", "licence"):
        assert row["provenance"].get(field), "provenance lost %r on the way out" % field
    assert row["provenance"]["origin"] == "math500#42"
    assert row["provenance"]["detail"]["row"] == 42, "provenance detail was flattened away"


def test_4_a_source_that_produces_nothing_is_a_reported_failure():
    """Both shapes of nothing must be visible: an explicit failure and an empty success."""
    pipe = TaskPipeline(SharedNoveltyBuffer(), ContaminationFilter(held_out=[]))
    stats = pipe.run([FakeSource("dead", [], ok=False, reason="backend refused"),
                      FakeSource("empty", [], ok=True)], 3, random.Random(0))
    assert stats["dead"].ok is False and "refused" in stats["dead"].failure_reason
    assert stats["empty"].ok is False, (
        "a source that returned ok with zero tasks was recorded as a success: that is a "
        "silent zero, which is the failure shape this project has hit five times")
    assert stats["empty"].failure_reason


def test_5_a_refuted_key_is_never_accepted():
    """A task whose key the verifier refutes must not enter the buffer or the artifacts."""
    calls = []

    def verifier(task):
        calls.append(task.task_id)
        return ("refuted", 10) if task.answer == "wrong" else ("verified", 10)

    buf = SharedNoveltyBuffer()
    pipe = TaskPipeline(buf, ContaminationFilter(held_out=[]), verifier=verifier)
    good = TaskRecord("Compute the sum of the first 20 positive odd integers.", "400",
                      prov())
    bad = TaskRecord("Compute the number of primes below one hundred exactly.", "wrong",
                     prov())
    stats = pipe.run([FakeSource("s", [good, bad])], 2, random.Random(0))
    assert len(calls) == 2, "every candidate must go through the verifier"
    assert stats["s"].refuted == 1 and stats["s"].verified == 1
    assert stats["s"].accepted == 1
    assert stats["s"].refuted_rate == 0.5
    assert all("wrong" != t.answer for t in pipe.accepted)
    assert len(buf.texts) == 1, "a refuted task was added to the novelty buffer"


def test_6_task_record_refuses_a_missing_key_or_provenance():
    """A task with no key would score p_hat=0 and look ideally difficult; refuse it."""
    for bad in ("", "   ", None):
        try:
            TaskRecord("A perfectly fine statement that is long enough to pass.", bad, prov())
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("accepted a task with answer %r" % bad)
    try:
        TaskRecord("A perfectly fine statement that is long enough to pass.", "1",
                   {"source": "x"})
    except TypeError:
        pass
    else:
        raise AssertionError("accepted a dict where a Provenance was required")
    try:
        Provenance(source="s", origin="o", licence="")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a provenance with an empty licence")


def test_7_hosted_teacher_cannot_spend_without_authorisation():
    """The paid backend must refuse, and must still be priceable without contacting anyone."""
    t = HostedTeacher(model="some-hosted-model", price_in_per_mtok=3.0,
                      price_out_per_mtok=15.0)
    os.environ.pop(HostedTeacher.AUTH_ENV, None)
    try:
        t.generate(["hello"])
    except PermissionError as e:
        assert "not authorised" in str(e)
    else:
        raise AssertionError("the hosted teacher made a call without authorisation")
    est = t.estimate_cost(n_prompts=100, prompt_tokens=400, output_tokens=1200)
    assert est["usd_total"] > 0 and est["n_prompts"] == 100


def test_8_model_written_source_reports_a_total_parse_failure():
    """If nothing parses, that is a failure with reasons, not an empty success."""
    class Backend:
        name = "stub"

        def generate(self, prompts):
            return (["no fields here at all"] * len(prompts), 42)

    src = ModelWrittenSource("generated", Backend(), "{context}", oversample=2)
    res = src.fetch(2, random.Random(0))
    assert res.ok is False and "none parsed" in res.reason
    assert res.cost_tokens == 42, "a failed source must still report what it spent"


def test_9_retrieved_source_reports_an_unusable_root():
    """A missing corpus must be a failure, not zero tasks."""
    src = RetrievedSource(root="/nonexistent/root", corpora=["math500"])
    res = src.fetch(3, random.Random(0))
    assert res.ok is False and "no usable rows" in res.reason


def test_10_parse_rejects_the_shapes_that_actually_occur():
    """The parse must reject what this base actually emits when it fails."""
    assert parse_problem_answer("")[2] == "empty"
    assert parse_problem_answer("thinking out loud with no fields")[2] == "no PROBLEM field"
    assert parse_problem_answer("PROBLEM: too short")[2] in ("no ANSWER field", "problem too short")
    long_ans = "PROBLEM: " + "x" * 60 + "\nANSWER: " + "y" * 200
    assert parse_problem_answer(long_ans)[2] == "answer not closed form"
    good = ("<think>musing</think>\nPROBLEM: Find the sum of all positive integers n below "
            "50 that are divisible by 7.\nANSWER: 168")
    p, a, why = parse_problem_answer(good)
    assert why == "ok" and a == "168" and "divisible by 7" in p


def test_11_similarity_separates_paraphrase_from_unrelated():
    """The measure the whole layer rests on must order these correctly."""
    a = "Find the number of positive divisors of 360."
    para = "How many positive divisors does the integer 360 have?"
    other = "Evaluate the limit of sin(x)/x as x approaches zero."
    assert similarity(a, para) > similarity(a, other)
    assert similarity(a, a) > 0.99


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
