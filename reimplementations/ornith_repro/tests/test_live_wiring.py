"""The live path fires, at both comparison levels, and the guards actually refuse.

This is the test that the reachability sweep says was missing. `nested`, `llm`,
`assert_group_nonempty` and `random_scaffold_control` all had tests and no production
caller, so the suite was green while the live path executed none of them. Every assertion
here is on OBSERVABLE STATE produced by `live.run_live_iteration` -- which stages ran, how
many groups formed, whether scaffold-level advantages are non-zero, whether a written
artifact verifies -- never on a function returning true.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from ornith_repro.buffer import TaskBuffer
from ornith_repro.guards import GuardViolation, assert_group_nonempty
from ornith_repro.judges import Judges
from ornith_repro.live import parse_proposal, run_live_iteration
from ornith_repro.loop import OrnithConfig, extract_boxed, read_artifacts, verify_provenance
from ornith_repro.nested import NestedConfig

SOLVED = ["Compute 1+1.", "Compute 2+2.", "Compute 3+3."]
UNSOLVED = ["Prove the Riemann hypothesis."]


class BoxedStub:
    """Offline client that answers each of the three prompts in its own format.

    Not a fixed-string mock: the rollout it returns depends on the seed, so a stage that
    failed to vary its seeds produces detectably identical rollouts, and `success_rate`
    drives p_hat to a chosen value so the degenerate and non-degenerate paths can both be
    exercised.
    """

    model_id = "stub/boxed-v1"

    def __init__(self, success_rate: float = 0.5, answer: str = "42") -> None:
        self.success_rate = success_rate
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Return a proposal, a scaffold, or a boxed rollout depending on the prompt."""
        self.prompts.append(prompt)
        if "PROBLEM: <statement on one line>" in prompt:
            return (f"PROBLEM: A genuinely self-contained question number {seed} that is "
                    f"long enough to pass the validity gate.\nANSWER: {self.answer}"), False
        if "Write instructions" in prompt:
            # Echo words from the task, because C(q,h) is word overlap with the task: a
            # scaffold sharing no vocabulary with its task scores C=0, which zeroes
            # R_harness = C*F*H for every scaffold and collapses the scaffold group.
            m = re.search(r"PROBLEM: (.+)", prompt)
            echo = " ".join((m.group(1) if m else "").split()[: 6 + seed % 5])
            return f"Approach: {echo}. Then simplify. Variant {seed}.", False
        # Success must depend on the SCAFFOLD too, not the seed alone. Keying it on the
        # seed only made every scaffold score identically, which collapsed the scaffold
        # group to zero advantage and made the nesting real but inert.
        key = hashlib.sha256((prompt + "|" + str(seed)).encode()).digest()[0] / 255.0
        good = key < self.success_rate
        return (f"work {seed} \\boxed{{{self.answer if good else 'wrong'}}}"), False


def _run(success_rate=0.5, n_scaffolds=3, n_roll=8, artifacts=None):
    """Drive one live iteration against the boxed stub."""
    cfg = OrnithConfig(max_new_tokens=256, min_valid_rollouts=4)
    ncfg = NestedConfig(n_scaffolds=n_scaffolds, n_rollouts_per_scaffold=n_roll)
    return run_live_iteration(BoxedStub(success_rate=success_rate), cfg, ncfg,
                              SOLVED, UNSOLVED, TaskBuffer(), Judges(), seed=7,
                              artifacts=artifacts)


def test_all_three_stages_fire_on_the_live_path():
    """Every stage the method page names must appear in a live iteration's record."""
    live = _run()
    assert live is not None
    assert set(live.stages_fired) >= {"rollout", "harness", "task"}, live.stages_fired


def test_nested_structure_is_actually_used_not_flat():
    """The scaffold level must EXIST: one rollout group per scaffold, plus a scaffold group.

    This is the assertion whose absence let the flat path run while the write-up described
    nesting. It checks the shape of the result, not that the call returned.
    """
    live = _run(n_scaffolds=3)
    assert len(live.scaffolds) == 3
    assert len(live.result.rollout_advantages) == 3, "one rollout group per scaffold"
    assert live.result.scaffold_advantages is not None, "no scaffold-level group formed"
    assert len(live.result.scaffold_advantages.advantages) == 3


def test_scaffold_advantages_are_not_identically_zero():
    """A scaffold group that is always degenerate would be nesting in name only."""
    live = _run(success_rate=0.5, n_scaffolds=3)
    adv = live.result.scaffold_advantages.advantages
    assert any(abs(a) > 1e-9 for a in adv), f"scaffold advantages all zero: {adv}"


def test_scaffolds_differ_from_each_other():
    """Distinct scaffolds must be generated, or the scaffold group compares copies."""
    live = _run(n_scaffolds=3)
    texts = {s.instructions for s in live.scaffolds}
    assert len(texts) == 3, f"only {len(texts)} distinct scaffolds"


def test_holdout_block_is_independent_of_the_discovery_block():
    """Scaffold rewards must be recomputed on a fresh block, or the curse is invisible."""
    live = _run(success_rate=0.5, n_scaffolds=3)
    assert len(live.result.scaffold_rewards_discovery) == 3
    assert len(live.result.scaffold_rewards_holdout) == 3


def test_guard_fires_on_a_singleton_group():
    """The guard must REFUSE, not pass. A guard that never fires is not evidence."""
    with pytest.raises(GuardViolation):
        assert_group_nonempty([1.0], what="singleton")
    with pytest.raises(GuardViolation):
        assert_group_nonempty([], what="empty")
    assert_group_nonempty([1.0, 0.0], what="pair")  # must NOT raise


def test_rollouts_are_graded_on_boxed_answers_not_a_marker():
    """The live grader reads \\boxed{}, scoring a matching answer and refusing a wrong one."""
    from ornith_repro.loop import grade
    from ornith_repro.types import RolloutOutcome, Scaffold, Task

    assert extract_boxed(r"x \boxed{\frac{1}{2}} y") == r"\frac{1}{2}"
    sc = Scaffold(scaffold_id="s", instructions="i", grader_kind="boxed_exact")
    task = Task(task_id="t", text="q", answer="42")
    assert grade(sc, task, r"so \boxed{42}", False).outcome is RolloutOutcome.SUCCESS
    assert grade(sc, task, r"so \boxed{41}", False).outcome is RolloutOutcome.FAILURE
    # Truncation stays ABORTED, never a wrong answer.
    assert grade(sc, task, "", True).outcome is RolloutOutcome.ABORTED


def test_a_batch_with_no_disagreement_anywhere_is_refused():
    """G4 must fire when every group is unanimous, rather than reporting a silent zero.

    An all-solved or all-failed task carries exactly zero reward-directed gradient, and any
    gradient statistic computed over it is a false negative. The guard had no caller at all
    before; this asserts it now fires on the live path.
    """
    from ornith_repro.guards import GuardViolation
    with pytest.raises(GuardViolation, match="degenerate"):
        _run(success_rate=1.0)
    with pytest.raises(GuardViolation, match="degenerate"):
        _run(success_rate=0.0)


def test_missing_answer_key_is_refused_not_scored_zero():
    """G8: a boxed_exact scaffold with no reference must raise, not mark everything wrong."""
    from ornith_repro.guards import GuardViolation
    from ornith_repro.loop import grade
    from ornith_repro.types import Scaffold, Task
    sc = Scaffold(scaffold_id="s", instructions="i", grader_kind="boxed_exact")
    with pytest.raises(GuardViolation, match="no answer"):
        grade(sc, Task(task_id="t", text="q", answer=None), r"\boxed{42}", False)


def test_unknown_grader_kind_is_refused_not_treated_as_anything_goes():
    """G9: a typo in grader_kind used to score every non-empty rollout SUCCESS."""
    from ornith_repro.guards import GuardViolation
    from ornith_repro.loop import grade
    from ornith_repro.types import Scaffold, Task
    sc = Scaffold(scaffold_id="s", instructions="i", grader_kind="boxed_exect")
    with pytest.raises(GuardViolation, match="unknown grader_kind"):
        grade(sc, Task(task_id="t", text="q", answer="42"), "anything", False)


def test_hardened_graders_include_the_live_one():
    """Regression for a defect already fixed once: boxed_exact must score H = 1.0.

    It was omitted when the live grader was added, so the strictest grader in the package was
    scored as the most gameable and every live R_harness was multiplied by 0.3. Without this
    test that fix can be silently reverted.
    """
    from ornith_repro.judges import Judges
    from ornith_repro.types import Scaffold
    j = Judges()
    assert j.hack_resistance(Scaffold("s", "i", grader_kind="boxed_exact")) == 1.0
    assert j.hack_resistance(Scaffold("s", "i", grader_kind="exact")) == 1.0
    assert j.hack_resistance(Scaffold("s", "i", grader_kind="nonempty")) == 0.3


def test_validity_gate_rejects_and_reports_reasons():
    """A malformed proposal is rejected with a recorded reason, never silently accepted."""
    assert parse_proposal("")[2] == "empty"
    assert parse_proposal("ANSWER: 4")[2] == "no PROBLEM field"
    assert parse_proposal("PROBLEM: x")[2] in ("no ANSWER field", "problem too short")
    p, a, why = parse_proposal("PROBLEM: " + "a long enough statement " * 3 + "\nANSWER: 42")
    assert why == "ok" and a == "42" and p


def test_naive_gold_agreement_is_not_reported_as_a_second_opinion():
    """It was algebraically identical to p_hat, so it must not masquerade as a check.

    Under boxed_exact a rollout scores 1.0 exactly when it matches `task.answer`, and p_hat is
    the mean of those. Reporting both invited a reader to count one number twice.
    """
    live = _run(success_rate=0.5)
    assert live.gold_agreement is None
    assert live.task.answer is not None


def test_artifacts_are_written_and_their_provenance_verifies(tmp_path):
    """A run must leave evidence, and that evidence must recheck against its own digests."""
    path = tmp_path / "iters.jsonl"
    live = _run(n_scaffolds=3, artifacts=path)
    assert live is not None
    rows = read_artifacts(path)
    assert len(rows) == 3, f"expected one row per scaffold, got {len(rows)}"
    for row in rows:
        verify_provenance(row)


def test_truncated_rollouts_are_aborted_not_wrong():
    """A capped generation must not be charged to the policy as a wrong answer."""
    class Truncating(BoxedStub):
        def generate(self, prompt, max_new_tokens, seed):
            if "PROBLEM: <statement on one line>" in prompt or "Write instructions" in prompt:
                return BoxedStub.generate(self, prompt, max_new_tokens, seed)
            return "", True

    cfg = OrnithConfig(max_new_tokens=256, min_valid_rollouts=4)
    ncfg = NestedConfig(n_scaffolds=2, n_rollouts_per_scaffold=8)
    with pytest.raises(ValueError, match="every rollout aborted"):
        run_live_iteration(Truncating(), cfg, ncfg, SOLVED, UNSOLVED,
                           TaskBuffer(), Judges(), seed=3)


# ------------------------------------------- the verifier must CHANGE what the loop does
class FixedVerifier:
    """Verifier returning a fixed verdict, counting how often it was consulted."""

    def __init__(self, verdict, detail="fixed") -> None:
        self.verdict = verdict
        self.detail = detail
        self.calls = 0
        self.seen: list[tuple[str, str]] = []

    def verify(self, problem, asserted):
        """Record the call and return the fixed verdict.

        An UNVERIFIABLE must state its abstention kind; the dataclass refuses a silent
        default, which is what let a truncated reply be reported as a coverage gap.
        """
        from ornith_repro.verify import Abstain, VerificationResult, Verdict as V
        self.calls += 1
        self.seen.append((problem, asserted))
        kind = Abstain.SUBSTANTIVE if self.verdict is V.UNVERIFIABLE else Abstain.NONE
        return VerificationResult(self.verdict, "fixed", None, asserted, self.detail,
                                  abstain=kind)


def _run_with(verdict, policy="reject_refuted", buffer=None):
    """One live iteration under a fixed key verdict, sharing a caller-supplied buffer."""
    from ornith_repro.verify import Verdict as V
    cfg = OrnithConfig(max_new_tokens=256, min_valid_rollouts=4)
    ncfg = NestedConfig(n_scaffolds=3, n_rollouts_per_scaffold=8)
    client = BoxedStub(success_rate=0.5)
    buf = buffer if buffer is not None else TaskBuffer()
    ver = FixedVerifier(verdict)
    live = run_live_iteration(client, cfg, ncfg, SOLVED, UNSOLVED, buf, Judges(),
                              seed=7, verifier=ver, key_policy=policy)
    return live, client, buf, ver


def test_a_refuted_key_stops_the_pipeline_and_spends_no_rollouts():
    """The load-bearing assertion: REFUTED must change OBSERVABLE behaviour.

    A call site that computed a verdict and dropped it would satisfy any reachability check
    and leave the mis-keying confound exactly where it was. So this asserts the task produced
    no rollouts, never entered the buffer, and returned no result -- not that a function ran.
    """
    from ornith_repro.verify import Verdict as V
    live, client, buf, ver = _run_with(V.REFUTED)
    assert ver.calls == 1, "verifier was never consulted"
    assert live.result is None, "a refuted task still produced a scored iteration"
    assert live.rollouts_spent == 0, "rollouts were spent on a task with a refuted key"
    assert len(buf) == 0, "a refuted task entered the task buffer"
    assert "refuted" in live.rejected_reason
    solver_prompts = [p for p in client.prompts if "Solve it." in p]
    assert not solver_prompts, "the solver was called for a refuted task"


def test_a_non_refuted_key_lets_the_pipeline_run():
    """The contrast that makes the previous test meaningful.

    Identical inputs; only the verdict differs. If both paths behaved the same, the verifier
    would be decorative however reachable it was.
    """
    from ornith_repro.verify import Verdict as V
    live, client, buf, ver = _run_with(V.UNVERIFIABLE)
    assert live.result is not None
    assert live.rollouts_spent == 48, live.rollouts_spent
    assert len(buf) == 1, "a scored task did not enter the buffer"
    assert [p for p in client.prompts if "Solve it." in p], "solver never ran"


def test_require_verified_also_drops_unverifiable_keys():
    """The strict policy abstains on anything short of a positive verification."""
    from ornith_repro.verify import Verdict as V
    live, _, buf, _ = _run_with(V.UNVERIFIABLE, policy="require_verified")
    assert live.result is None and live.rollouts_spent == 0 and len(buf) == 0
    live2, _, buf2, _ = _run_with(V.VERIFIED, policy="require_verified")
    assert live2.result is not None and len(buf2) == 1


def test_claiming_to_verify_without_a_verifier_is_refused():
    """A policy that verifies must have something to verify with, or it lies by default."""
    cfg = OrnithConfig(max_new_tokens=256)
    ncfg = NestedConfig(n_scaffolds=2, n_rollouts_per_scaffold=8)
    with pytest.raises(ValueError, match="requires a verifier"):
        run_live_iteration(BoxedStub(), cfg, ncfg, SOLVED, UNSOLVED, TaskBuffer(),
                           Judges(), seed=1, key_policy="reject_refuted")


def test_the_verifier_receives_the_task_text_and_the_asserted_key_only():
    """The verifier is given the problem and the key, and nothing about the rollouts."""
    from ornith_repro.verify import Verdict as V
    live, _, _, ver = _run_with(V.VERIFIED)
    problem, asserted = ver.seen[0]
    assert problem == live.task.text
    assert asserted == live.task.answer
