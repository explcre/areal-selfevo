"""Mutation test for the token-id evaluation path: break it on purpose, one way at a time.

Run against a COPY of the tree, never the live checkout:

    rsync -a --exclude .git ~/areal-selfevo/ /tmp/mut_tid/
    MATH_EVAL_DATA=~/evaldata python3 selfevo/tests/mutate_token_id_eval.py /tmp/mut_tid

The refusal below is not decoration: the training run imports this tree across worker
processes that relaunch, so a mutated source file sitting on disk for even a few seconds can
be imported by a live run.

REPORTING. Three columns, not two, following ``mutate_periodic_eval.py``. A mutation whose
anchor was not unique, whose replacement left the bytes unchanged, or whose result does not
compile has not been TESTED, and scoring it as a kill reports a number higher than the truth.
``SKIP`` is its own outcome and makes the run fail, exactly as a survivor does.

WHAT THE FIVE STARRED MUTATIONS ARE. They are the five the brief named, and each one is a
defect that would still produce a plausible curve: a decoder that loses the last token, a
point emitted without its budget, the token-id path engaging against a server that has a
tokenizer, a comparability flag stuck at "not comparable", and a decode failure that returns
an empty string the grader then scores as a wrong answer.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
from dataclasses import dataclass

TESTS = [
    "selfevo/tests/test_token_id_eval.py",
    "selfevo/tests/test_periodic_eval.py",
]

LIVE = pathlib.Path("/home/ubuntu/areal-selfevo").resolve()

MB = "experiments/bench/math_bench.py"
PE = "selfevo/periodic_eval.py"


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect.

    Attributes:
        label: What the defect is, in the words a reader needs.
        rel: Source file, relative to the repo root.
        find: Exact text to replace; must appear exactly once.
        repl: Its replacement.
    """

    label: str
    rel: str
    find: str
    repl: str


MUTATIONS = [
    Mutation(
        label="THE BRIEF'S FIRST MUTATION: the decoder drops the final token",
        rel=MB,
        find="        return self.tok.decode(list(ids), skip_special_tokens=True)",
        repl="        return self.tok.decode(list(ids)[:-1], skip_special_tokens=True)",
    ),
    Mutation(
        label="THE BRIEF'S SECOND MUTATION: the budget is not emitted with the score",
        rel=PE,
        find='    "max_tokens",\n    "headline_max_tokens",\n    "budget_matches_headline",\n',
        repl="",
    ),
    Mutation(
        label="THE BRIEF'S THIRD MUTATION: the token-id path engages against a TOKENISING server",
        rel=MB,
        find="    if caps.has_tokenizer:\n        url = chat_url(args.base_url)",
        repl="    if False:\n        url = chat_url(args.base_url)",
    ),
    Mutation(
        label="THE BRIEF'S FOURTH MUTATION: the comparability flag always reads NOT COMPARABLE",
        rel=PE,
        find='        "budget_matches_headline": None if used is None else int(used == headline),',
        repl='        "budget_matches_headline": None if used is None else 0,',
    ),
    Mutation(
        label="THE BRIEF'S FIFTH MUTATION: a decode failure returns \"\" and is graded WRONG",
        rel=MB,
        find=(
            "        try:\n"
            "            text = tio.decode(ids)\n"
            "        except Exception as exc:\n"
            "            _report_token_id_problem(\n"
            '                url, 200, f"could not decode {len(ids)} output ids: {type(exc).__name__}: {exc}"\n'
            "            )\n"
            "            return dict(failed)\n"
        ),
        repl=(
            "        try:\n"
            "            text = tio.decode(ids)\n"
            "        except Exception:\n"
            '            text = ""\n'
        ),
    ),
    Mutation(
        label="the comparability flag always reads COMPARABLE, which is the dangerous direction",
        rel=PE,
        find='        "budget_matches_headline": None if used is None else int(used == headline),',
        repl='        "budget_matches_headline": None if used is None else 1,',
    ),
    Mutation(
        label="the budget post-condition guard never fires",
        rel=PE,
        find="    for bench in benchmarks:\n        score = metrics.get",
        repl="    for bench in []:\n        score = metrics.get",
    ),
    Mutation(
        label="the budget pre-condition guard never fires",
        rel=PE,
        find="    used = (row.get(\"params\") or {}).get(\"max_tokens\")\n    if used is None:\n        raise BudgetUnrecorded(",
        repl="    used = (row.get(\"params\") or {}).get(\"max_tokens\")\n    if False:\n        raise BudgetUnrecorded(",
    ),
    Mutation(
        label="the recorded budget is the one that was ASKED FOR, not the one that ran",
        rel=PE,
        find='    used = (row.get("params") or {}).get("max_tokens")\n    used = None if used is None else int(used)',
        repl='    used = (row.get("params") or {}).get("max_tokens_requested") or (row.get("params") or {}).get("max_tokens")\n    used = None if used is None else int(used)',
    ),
    Mutation(
        label="the trend series is called `accuracy` again, so it lines up with the headline",
        rel=PE,
        find='BENCH_METRIC_SUFFIXES = (\n    "trend_score",',
        repl='BENCH_METRIC_SUFFIXES = (\n    "accuracy",',
    ),
    Mutation(
        label="the headline budget is not read from the table the headline runs used",
        rel=MB,
        find='    v = BENCH_OVERRIDES.get(bench, {}).get("max_tokens")\n    if isinstance(v, int) and v > 0:\n        return v',
        repl='    v = BENCH_OVERRIDES.get(bench, {}).get("max_tokens")\n    if False:\n        return v',
    ),
    Mutation(
        label="the tokenizer-less server is GUESSED at rather than asked",
        rel=MB,
        find="    has_tok = not skip if isinstance(skip, bool) else True",
        repl="    has_tok = True",
    ),
    Mutation(
        label="the server's published context limit is dropped, so the old guard is all there is",
        rel=MB,
        find="        context_limit=ctx if isinstance(ctx, int) and ctx > 0 else None,",
        repl="        context_limit=None,",
    ),
    Mutation(
        label="the cap is never clamped to what the server will accept",
        rel=MB,
        find='    eff, why = clamp_max_tokens(int(params.get("max_tokens") or 0), caps.context_limit)',
        repl='    eff, why = int(params.get("max_tokens") or 0), None',
    ),
    Mutation(
        label="the token budget is sent under a key `/generate` ignores, so the cap does nothing",
        rel=MB,
        find='        "max_new_tokens": params["max_tokens"],',
        repl='        "max_tokens": params["max_tokens"],',
    ),
    Mutation(
        label="an ABORTED generation is graded as a wrong answer (the A0 silent zero)",
        rel=MB,
        find='        if reason == "abort":',
        repl="        if False:",
    ),
    Mutation(
        label="a reply with no output_ids becomes an empty completion, graded WRONG",
        rel=MB,
        find='        ids = d.get("output_ids")\n        if not isinstance(ids, list):',
        repl='        ids = d.get("output_ids")\n        if not isinstance(ids, list):\n            ids = []\n        if False:',
    ),
    Mutation(
        label="the BASE model is routed as a LoRA adapter, which the server has never loaded",
        rel=MB,
        find="    if base and os.path.realpath(model) == os.path.realpath(base):\n        return \"\"",
        repl="    if False:\n        return \"\"",
    ),
    Mutation(
        label="the chat template is not applied, so the model answers a different question",
        rel=MB,
        find=(
            "        text = self.tok.apply_chat_template(\n"
            '            [{"role": "user", "content": content}],\n'
            "            tokenize=False,\n"
            "            add_generation_prompt=True,\n"
            "        )\n"
        ),
        repl="        text = content\n",
    ),
    Mutation(
        label="a tokenizer that is not the server's is accepted, silently changing the prompt",
        rel=MB,
        find="            if mine != theirs:",
        repl="            if False:",
    ),
    Mutation(
        label="the liveness probe keeps using the chat endpoint the server cannot serve",
        rel=PE,
        find="    if caps is None or caps.has_tokenizer:",
        repl="    if True:",
    ),
    Mutation(
        label="a missing logprobs block on the token-id path is read as `no difference`",
        rel=PE,
        find='    lps = r.get("logprobs") or []\n    if not lps:\n        raise LivenessUnavailable(',
        repl='    lps = r.get("logprobs") or [0.0]\n    if False:\n        raise LivenessUnavailable(',
    ),
]


def run_tests(repo: pathlib.Path) -> bool:
    """Run the pinned tests inside one tree.

    Args:
        repo: The tree to run in.

    Returns:
        True when every test passed.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    return r.returncode == 0


def main() -> int:
    """Apply every mutation in turn and report killed / survived / skipped.

    Returns:
        0 when every mutation was killed, 1 otherwise.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    repo = pathlib.Path(sys.argv[1]).resolve()
    if repo == LIVE:
        print(
            f"REFUSING: {repo} is the live checkout. A training run imports this tree across "
            f"worker processes that relaunch, so a mutated file on disk can be imported by "
            f"it. Copy the tree first and point this at the copy."
        )
        return 2

    originals = {}
    digests = {}
    for m in MUTATIONS:
        p = repo / m.rel
        if m.rel not in originals:
            originals[m.rel] = p.read_text()
            digests[m.rel] = hashlib.sha256(originals[m.rel].encode()).hexdigest()

    if not run_tests(repo):
        print("BASELINE IS RED -- fix the tree before reading any mutation result")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    killed, survived, skipped = [], [], []
    for m in MUTATIONS:
        p = repo / m.rel
        original = originals[m.rel]
        n = original.count(m.find)
        if n != 1:
            print(f"SKIP      {m.label}  (anchor appears {n}x)", flush=True)
            skipped.append(m.label)
            continue
        mutated = original.replace(m.find, m.repl, 1)
        if mutated == original:
            print(f"SKIP      {m.label}  (replacement changed no bytes)", flush=True)
            skipped.append(m.label)
            continue
        try:
            compile(mutated, str(p), "exec")
        except SyntaxError as exc:
            print(f"SKIP      {m.label}  (mutant does not compile: {exc})", flush=True)
            skipped.append(m.label)
            continue
        p.write_text(mutated)
        try:
            still_green = run_tests(repo)
        finally:
            p.write_text(original)
            assert hashlib.sha256(p.read_text().encode()).hexdigest() == digests[m.rel], (
                f"failed to restore {m.rel}"
            )
        if still_green:
            print(f"SURVIVED  {m.label}", flush=True)
            survived.append(m.label)
        else:
            print(f"killed    {m.label}", flush=True)
            killed.append(m.label)

    print(
        f"\nkilled {len(killed)}  survived {len(survived)}  skipped {len(skipped)}"
        f"  of {len(MUTATIONS)}"
    )
    for x in survived:
        print(f"  SURVIVOR: {x}")
    for x in skipped:
        print(f"  SKIPPED:  {x}")
    return 1 if (survived or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
