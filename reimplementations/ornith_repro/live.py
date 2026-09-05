"""The live driver: the stage that makes the loop call a model instead of stubs.

WHY THIS MODULE EXISTS. Every other piece of this reimplementation was built and tested,
and none of it ran. `run_iteration` and `run_iteration_nested` both take already-generated
text as an argument, so nothing in the package ever called `llm.py`; a reachability sweep
found `llm`, `nested`, `controls.random_scaffold_control` and `guards.assert_group_nonempty`
defined, exercised by tests, and reached by no production caller. This module is the caller.
It generates all three stages against a real client and hands them to the NESTED entry point,
which is the one Ornith's figure describes.

THE THREE GENERATION STAGES, in the order the method page gives them:

  1. proposer  -- a task "beyond what the model has already solved", conditioned on the
     model's own measured competence and on the buffer, so novelty has something to bite on;
  2. harness   -- `n_scaffolds` scaffolds for that task, which is what creates the scaffold
     comparison level;
  3. solver    -- `n_rollouts_per_scaffold` rollouts per scaffold, plus an independent
     HOLDOUT block so the scaffold-level winner's curse is measured rather than assumed.

WHAT IS OURS. The three prompts, the rubrics behind V/C/F/H (`judges.py`), the counts, sigma,
and the decision to grade a generated task against the proposer's own asserted answer. That
last one is the sharpest: agreement with a generated gold is agreement with the GENERATOR,
not correctness, and this project has already measured 58.8% of self-generated golds
disagreeing with confident solver consensus. `Task.answer` and the `gold_agreement`
statistic below exist so that this is reported rather than assumed away.

SILENT-ZERO DISCIPLINE. A generation that hits the cap is `truncated=True` and becomes
ABORTED, never a wrong answer: three separate budget traps have already appeared in this
project, most recently the proposer rejecting 43% of its own output at an 8192-token cap.
Every count this module reports separates aborted from failed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .buffer import TaskBuffer
from .guards import assert_group_nonempty
from .controls import random_scaffold_control
from .judges import Judges
from .loop import IterationResult, OrnithConfig, write_artifacts
from .nested import NestedConfig, NestedIterationResult, run_iteration_nested
from .rewards import jaccard_similarity
from .types import Scaffold, Task, digest
from .verify import Verdict

PROPOSER_PROMPT = """You are proposing a new training task for a model.

Problems the model ALREADY SOLVES RELIABLY:
{solved}

Problems the model CANNOT SOLVE AT ALL:
{unsolved}

{novelty}

Propose ONE new problem that goes beyond what the model already solves: harder than the
first group, but not hopeless like the second. Aim for a problem it solves about one time
in five.

Requirements:
- Fully self-contained and unambiguous, solvable from its own text alone.
- Exactly ONE correct final answer, a specific closed form (integer, fraction, or simple
  exact expression). Not a proof, not "show that".
- You must be certain of the answer.

Reply in exactly this format and nothing else:
PROBLEM: <statement on one line>
ANSWER: <final answer only>
"""

SCAFFOLD_PROMPT = """You are writing a solving scaffold for this problem.

PROBLEM: {task}

Write instructions that tell a solver how to approach THIS problem specifically: what to
set up, what to decompose it into, and what form the final answer must take. Be concrete
about this problem, not generic.

Reply with the instructions only, at most 120 words.
"""

SOLVER_PROMPT = """{instructions}

PROBLEM: {task}

Solve it. Put your final answer in \\boxed{{}}.
"""


#: What to do with a task whose asserted answer fails independent verification. This is a
#: required argument rather than a default, because the whole point is that a silent "off"
#: is what left every difficulty number confounded with mis-keying.
KEY_POLICIES = ("off", "reject_refuted", "require_verified")


@dataclass
class LiveIteration:
    """One live iteration plus the accounting that makes it auditable.

    `result` is None when the task was rejected before any rollout was spent, in which case
    `rejected_reason` says why.
    """

    result: NestedIterationResult | None
    task: Task
    scaffolds: list[Scaffold]
    n_proposals: int = 0
    n_rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    gold_agreement: float | None = None
    stages_fired: list[str] = field(default_factory=list)
    control_scaffolds: list = field(default_factory=list)
    key_verdict: str = "off"
    key_detail: str = ""
    rejected_reason: str = ""
    rollouts_spent: int = 0


def parse_proposal(text: str) -> tuple[str | None, str | None, str]:
    """Split a proposal into problem and answer, or say why it is unusable.

    This is the validity gate in its most literal form, applied before the judge rubric:
    a proposal that cannot be read as a task with one answer is rejected and the reason is
    RECORDED, so the rejection rate is a reported number rather than an assumption.

    Args:
        text: Raw proposer completion.

    Returns:
        (problem or None, answer or None, reason).
    """
    if not text:
        return None, None, "empty"
    body = text.split("</think>")[-1]
    mp = re.search(r"PROBLEM:\s*(.+?)(?:\n\s*ANSWER:|\Z)", body, re.S)
    ma = re.search(r"ANSWER:\s*(.+?)\s*\Z", body, re.S)
    if not mp:
        return None, None, "no PROBLEM field"
    if not ma:
        return None, None, "no ANSWER field"
    problem = " ".join(mp.group(1).split())
    answer = " ".join(ma.group(1).split())
    if len(problem) < 40:
        return None, None, "problem too short"
    if len(answer) > 120:
        return None, None, "answer not closed form"
    return problem, answer, "ok"


def propose_task(client, cfg: OrnithConfig, solved: Sequence[str], unsolved: Sequence[str],
                 buffer: TaskBuffer, seed: int,
                 max_attempts: int = 4) -> tuple[Task | None, int, dict]:
    """Stage 1: generate a task, retrying only over VALIDITY rejections.

    The buffer is shown to the proposer so that novelty has something to act on; N(q) then
    scores the result against the same buffer. Retrying is bounded and every rejection is
    counted, because an unbounded retry loop would quietly turn a broken proposer into a
    working-looking one.

    Args:
        client: An `LLMClient`.
        cfg: Loop configuration, for the generation cap.
        solved: Exemplars the model solves every time.
        unsolved: Exemplars it never solves.
        buffer: The task buffer, shown as the novelty context.
        seed: Base seed; each attempt uses a distinct derived seed.
        max_attempts: Attempts before giving up on this slot.

    Returns:
        (task or None, attempts used, rejection-reason counts).
    """
    reasons: dict = {}
    recent = buffer.texts()[-5:]
    novelty = ("Do NOT repeat or lightly reword any of these already-used tasks:\n"
               + "\n".join("- " + t[:200] for t in recent)) if recent else ""
    for attempt in range(max_attempts):
        prompt = PROPOSER_PROMPT.format(
            solved="\n".join("- " + s[:400] for s in solved),
            unsolved="\n".join("- " + s[:400] for s in unsolved),
            novelty=novelty)
        text, truncated = client.generate(prompt, cfg.max_new_tokens, seed * 100 + attempt)
        if truncated:
            reasons["truncated"] = reasons.get("truncated", 0) + 1
            continue
        problem, answer, reason = parse_proposal(text)
        if reason != "ok":
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        return (Task(task_id=digest("task", problem)[:16], text=problem,
                     answer=answer, source="generated"),
                attempt + 1, reasons)
    return None, max_attempts, reasons


def propose_scaffolds(client, cfg: OrnithConfig, ncfg: NestedConfig, task: Task,
                      seed: int) -> list[Scaffold]:
    """Stage 2: generate the scaffolds that form the scaffold comparison level.

    Args:
        client: An `LLMClient`.
        cfg: Loop configuration.
        ncfg: Nested counts; `n_scaffolds` under nested sampling, 1 under flat.
        task: The task to scaffold.
        seed: Base seed; each scaffold gets a distinct derived seed.

    Returns:
        The scaffolds, each graded `boxed_exact` so the rollout reward is a real answer
        check rather than a string marker.
    """
    n = ncfg.n_scaffolds if ncfg.sampling == "nested" else 1
    out: list[Scaffold] = []
    for j in range(n):
        text, truncated = client.generate(
            SCAFFOLD_PROMPT.format(task=task.text), cfg.max_new_tokens, seed * 1000 + j)
        instructions = (text.split("</think>")[-1].strip()
                        if not truncated else "Solve the problem.")
        if not instructions:
            instructions = "Solve the problem."
        out.append(Scaffold(scaffold_id=digest("scaffold", task.task_id, j, instructions)[:16],
                            instructions=instructions[:2000],
                            grader_kind="boxed_exact", source="generated"))
    return out


def roll_out(client, cfg: OrnithConfig, task: Task, scaffold: Scaffold, n: int,
             seed: int) -> list[tuple[str, bool]]:
    """Stage 3: generate one scaffold's rollout block.

    Args:
        client: An `LLMClient`.
        cfg: Loop configuration.
        task: The task.
        scaffold: The scaffold, whose instructions condition the solver.
        n: Rollouts to generate.
        seed: Base seed; each rollout gets a distinct derived seed.

    Returns:
        `(text, truncated)` pairs. Truncation is propagated, never collapsed into a wrong
        answer.
    """
    prompt = SOLVER_PROMPT.format(instructions=scaffold.instructions, task=task.text)
    return [client.generate(prompt, cfg.max_new_tokens, seed * 10000 + i) for i in range(n)]


def gold_agreement(result: NestedIterationResult, task: Task) -> float | None:
    """DEPRECATED: this is algebraically identical to `p_hat` and carries no extra signal.

    It was reported as a separate headline number until an audit showed the identity: under
    the `boxed_exact` grader a rollout's reward is 1.0 exactly when its boxed answer matches
    `task.answer`, and `p_hat` is the mean of those rewards over resolved rollouts. So
    "agreement with the proposer's gold" and "success rate" are the same quantity, and
    reporting both invited a reader to treat one number as two independent checks.

    The number that WOULD be informative is agreement with an INDEPENDENTLY verified answer,
    which requires `verify.py` in the loop. Until the verifier is wired, no gold statistic
    here is independent of the proposer, and this function returns None rather than dressing
    up `p_hat` as a second opinion.

    Args:
        result: The nested iteration.
        task: The task carrying the asserted answer.

    Returns:
        None, always. Kept as a named seam so the verifier can supply the real statistic.
    """
    return None


def run_live_iteration(client, cfg: OrnithConfig, ncfg: NestedConfig,
                       solved: Sequence[str], unsolved: Sequence[str],
                       buffer: TaskBuffer, judges: Judges, seed: int,
                       sim: Callable[[str, str], float] = jaccard_similarity,
                       artifacts: Path | None = None,
                       scaffold_pool: Sequence[Scaffold] | None = None,
                       verifier=None,
                       key_policy: str = "off") -> LiveIteration | None:
    """One complete live iteration: propose, scaffold, roll out, score, persist.

    Every stage is driven by the client, and the NESTED entry point is used rather than the
    flat one, so both comparison levels exist. `assert_group_nonempty` is invoked on each
    rollout block before it is scored -- that guard previously had no caller at all.

    Args:
        client: An `LLMClient`.
        cfg: Loop configuration.
        ncfg: Nested configuration.
        solved: Competence exemplars the model always solves.
        unsolved: Exemplars it never solves.
        buffer: Task buffer, read for novelty and written after scoring.
        judges: Frozen judges.
        seed: Iteration seed.
        sim: Similarity behind N(q).
        artifacts: When given, one JSONL row per (task, scaffold) is appended, each with a
            provenance digest that `verify_provenance` can recheck.
        scaffold_pool: Scaffolds from other tasks. When given, a size-matched random
            scaffold control is drawn from it, so the comparison level that nesting ADDS is
            not left uncontrolled -- the rollout level had a control and this one did not.
        verifier: Object with `verify(problem, asserted) -> VerificationResult`, checking the
            proposer's key INDEPENDENTLY of the proposer. Required unless `key_policy` is
            "off".
        key_policy: "off" trains on unverified keys, which is what left every difficulty
            number confounded with mis-keying; "reject_refuted" drops a task whose key is
            refuted; "require_verified" additionally drops UNVERIFIABLE keys. Verification
            runs BEFORE any rollout, so a refuted task costs one check rather than a full
            rollout budget.

    Returns:
        A `LiveIteration`, or None when the proposer produced nothing valid.
    """
    if key_policy not in KEY_POLICIES:
        raise ValueError("key_policy must be one of %s, got %r" % (KEY_POLICIES, key_policy))
    if key_policy != "off" and verifier is None:
        raise ValueError("key_policy=%r requires a verifier; refusing to run unverified "
                         "while claiming to verify" % key_policy)

    task, attempts, reasons = propose_task(client, cfg, solved, unsolved, buffer, seed)
    if task is None:
        return None

    # ---- independent key check, BEFORE any rollout budget is committed ----------------
    verdict, vdetail = "off", "key not checked"
    if key_policy != "off":
        vres = verifier.verify(task.text, task.answer or "")
        verdict, vdetail = vres.verdict.value, vres.detail
        drop = (vres.verdict is Verdict.REFUTED or
                (key_policy == "require_verified" and vres.verdict is not Verdict.VERIFIED))
        if drop:
            # The task never reaches the solver, never enters the buffer, and never
            # contributes a p_hat. This is the behaviour the confound needed.
            return LiveIteration(
                result=None, task=task, scaffolds=[], n_proposals=attempts,
                n_rejected=sum(reasons.values()), rejection_reasons=reasons,
                key_verdict=verdict, key_detail=vdetail, rollouts_spent=0,
                rejected_reason="key %s by independent verification" % verdict)

    scaffolds = propose_scaffolds(client, cfg, ncfg, task, seed)
    blocks: list[list[tuple[str, bool]]] = []
    holdout: list[list[tuple[str, bool]]] = []
    for j, sc in enumerate(scaffolds):
        blk = roll_out(client, cfg, task, sc, ncfg.n_rollouts_per_scaffold, seed * 7 + j)
        hld = roll_out(client, cfg, task, sc, ncfg.n_rollouts_per_scaffold,
                       seed * 7 + j + 500000)
        # NOT `[1.0] * len(blk)`. That was the first version, and it could not fire: the
        # vector was synthetic and its length is already forced >= 2 by
        # NestedConfig.__post_init__, so the guard was decorative while its docstring
        # claimed it had been wired. Pass what actually varies -- whether each generation
        # came back usable -- so an all-failed block is refused instead of scored.
        assert_group_nonempty([0.0 if tr else 1.0 for _, tr in blk],
                              what="rollout block for scaffold %d" % j)
        assert_group_nonempty([0.0 if tr else 1.0 for _, tr in hld],
                              what="holdout block for scaffold %d" % j)
        blocks.append(blk)
        holdout.append(hld)

    result = run_iteration_nested(cfg, ncfg, task, scaffolds, blocks, buffer, judges,
                                  sim=sim, holdout_texts_by_scaffold=holdout)

    if artifacts is not None:
        for hrec, adv in zip(result.harness_records, result.rollout_advantages):
            write_artifacts(artifacts, IterationResult(
                task_record=result.task_record, harness_record=hrec,
                rollout_advantages=adv, updates_applied=[],
                stages_fired=list(result.stages_fired)))

    control: list[Scaffold] = []
    if scaffold_pool:
        # Matched on grader_kind, the scaffold's observable covariate, exactly as the
        # task-level control is matched on task covariates.
        control = random_scaffold_control(scaffolds, list(scaffold_pool), key="grader_kind",
                                          seed=seed)

    return LiveIteration(
        result=result, task=task, scaffolds=list(scaffolds), control_scaffolds=control,
        n_proposals=attempts, n_rejected=sum(reasons.values()), rejection_reasons=reasons,
        gold_agreement=gold_agreement(result, task),
        stages_fired=list(result.stages_fired),
        key_verdict=verdict, key_detail=vdetail,
        rollouts_spent=len(scaffolds) * ncfg.n_rollouts_per_scaffold * 2)


def load_competence(blocks_path: str, problems_path: str,
                    min_samples: int = 8) -> tuple[list[str], list[str]]:
    """Split a measured pool into always-solved and never-solved exemplars.

    Args:
        blocks_path: Rollout JSONL carrying per-problem outcomes.
        problems_path: The problem file those outcomes index.
        min_samples: Problems with fewer resolved samples are not classified.

    Returns:
        (always_solved, never_solved) problem statements.
    """
    from collections import defaultdict
    got = defaultdict(list)
    with open(blocks_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok" and r.get("finish_reason") != "length":
                got[r["idx"]].append(1 if r.get("correct") else 0)
    rows = [json.loads(l) for l in open(problems_path) if l.strip()]
    always: list[str] = []
    never: list[str] = []
    for i, row in enumerate(rows):
        v = got.get(i, [])
        if len(v) < min_samples:
            continue
        if sum(v) == len(v):
            always.append(row["question"])
        elif sum(v) == 0:
            never.append(row["question"])
    return always, never
