#!/usr/bin/env python3
"""Score a served model on the frozen math suite.

Generation goes through our sglang OpenAI-compatible endpoint; grading uses `math_verify`,
which is what AZR's own `custom_evaluate.py` imports (`from math_verify import parse,
verify`). An independent audit cross-checked this grader against AZR's own cascade over 680
generations and found **zero disagreements**, 1 misgrade in 600 (understatement), and no
false positives -- so grading is not the uncertainty that matters here.

The uncertainties that DO matter, all surfaced by that audit and handled below:

* **Run-to-run nondeterminism dominates problem-sampling error.** The same command twice
  gave MATH-500 74.80% and 76.40%; AMC23 spanned 47.5-60.0% over five runs. A standard
  error computed from problem sampling alone (+/-7.9pp on AMC23) understates the real
  spread. Every number should be reported with the seed and, where it matters, repeated.
* **A binomial SE is the wrong interval at these counts.** At 1/30 the normal interval runs
  negative; at 0/30 or 30/30 the SE is exactly 0, claiming certainty. Wilson is reported
  instead, and it is asymmetric on purpose.
* **Silent-zero paths.** Truncated generations were graded as wrong answers with nothing
  reported; empty 200-responses were dropped from the denominator, which *inflates*
  accuracy; a partial outage shrank the denominator while still printing a normal-looking
  score. Each is now counted and surfaced separately.
* **Nothing was persisted**, so no reported number could be re-audited without regenerating
  it. Every completion is now written to disk.
* **Which weights answered was neither checked nor recorded.** The payload names a MODEL ID
  and nothing else -- sglang routes a LoRA adapter by that id, because `--lora-paths
  NAME=path` registers NAME as one -- and an id the server does not have is answered HTTP
  200 by the BASE model. The harness default was exactly such a name and no results row
  recorded the id, so a run left on the default would have scored the base model in
  silence. Every run now verifies the id against `<base-url>/models` BEFORE generating, and
  records the id, the endpoint and the whole served list.

Why not AZR's runner: `math_eval/eval/math_eval.py` does `from vllm import LLM`, and vLLM
is not installed here. Installing it risks the torch/sglang environment. The runner is the
easy half; the grader is the part worth reusing.

Datasets come from the pinned AZR clone (MIT). Benchmarks with a uniform
`problem`/`answer` schema are included as-is; olympiadbench (`question`/`final_answer`) has
its own adapter in `load` and IS included -- it is the frontier math target, 675 problems
with a 7-point CI, where MATH-500 saturates at 27B and AIME is unusable at 1.5B. Its gold
answers self-verify 675/675 through this grader, measured 2026-08-31, so a low score from it
is the model's and not the harness's. minerva_math (answer embedded in `solution`) still
needs its own extraction and is left out rather than half-supported.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import os

# Root holding one directory per benchmark (each with test.jsonl). Differs per box -- the
# A100 has it under the Absolute-Zero-Reasoner checkout, the H200 under /root/evaldata --
# so it is an environment variable rather than a constant.
DATA = Path(
    os.environ.get(
        "MATH_EVAL_DATA",
        os.path.expanduser(
            "~/baselines/Absolute-Zero-Reasoner/evaluation/math_eval/eval/data"
        ),
    )
)


def require_dataset(name: str) -> Path:
    """Path to one benchmark's ``test.jsonl``, checked to exist.

    A missing dataset must stop the run. Left unchecked it produces an empty problem list,
    which grades to zero and is indistinguishable in the output from a model that answered
    everything wrong -- a silent-zero failure this project has already been bitten by.

    Args:
        name: Benchmark directory name, e.g. ``"math500"``.

    Returns:
        The path to ``<DATA>/<name>/test.jsonl``.

    Raises:
        FileNotFoundError: If the root or the file is absent, naming MATH_EVAL_DATA so the
            fix is obvious from the message alone.
    """
    if not DATA.exists():
        raise FileNotFoundError(
            f"eval-data root {DATA} does not exist. Set MATH_EVAL_DATA to the directory "
            f"holding one folder per benchmark (each with test.jsonl)."
        )
    f = DATA / name / "test.jsonl"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. MATH_EVAL_DATA={DATA} exists but has no {name}/test.jsonl; "
            f"present: {sorted(d.name for d in DATA.iterdir() if d.is_dir())[:12]}"
        )
    return f
SUITE = ["aime24", "aime25", "amc23", "math500", "hmmt_2024", "hmmt_2025",
         "livemathbench", "olympiadbench"]

# --------------------------------------------------------- per-benchmark generation ----
#
# Generation parameters are a property of the BENCHMARK, not of the run. A single global
# value is wrong in both directions: it over-budgets short benchmarks and under-budgets long
# ones, and because the parameters land in the results row it makes two runs look comparable
# when they were generated differently.
#
# MEASURED 2026-08-31, same checkpoint at 3072 vs 8192 max_tokens:
#   math500 0.5240 -> 0.5260 (trunc 39 -> 36), amc23 0.1750 -> 0.1500 (trunc 3 -> 4),
#   aime24 0.0000 -> 0.0000 (trunc 11 -> 8), aime25 0.0000 -> 0.0000 (trunc 3 -> 2).
# So raising the cap 2.7x moved accuracy by less than the 0.020 greedy noise floor. In every
# row n_truncated equals n_no_box, i.e. these generations never emit \boxed{} at all rather
# than being cut off mid-solution -- they are genuine failures, not budget artifacts. The
# caps below are therefore chosen for HEADROOM and PROVENANCE, not because they raise scores.
#
# olympiadbench is the exception, and the hypothesis was CONFIRMED by measurement: unlike
# MATH-500 and AIME, where 3072 -> 8192 moved accuracy by less than noise, it gained
# 0.1778 -> 0.1941 with truncation falling 15.3% -> 11.7% (103/675 -> 79/675). So a uniform
# cap really was wrong for this suite, and this benchmark really is the long-output one.
#
# 8192 is still not enough: `cap_limited` fired at 11.7%, and it fired for one arm and not
# the other (ctx 79/675 vs rnd 61/675), which makes an arm comparison at that cap unfair --
# the arm that truncates more is penalised more. Raised to 16384 to test whether the gap
# survives once neither arm is budget-bound.
#
# Anything absent falls back to the CLI value, so this table states only what differs.
BENCH_OVERRIDES: dict[str, dict[str, object]] = {
    "aime24": {"max_tokens": 8192},
    "aime25": {"max_tokens": 8192},
    "hmmt_2024": {"max_tokens": 8192},
    "hmmt_2025": {"max_tokens": 8192},
    "livemathbench": {"max_tokens": 8192},
    "olympiadbench": {"max_tokens": 16384},
}

# Every knob that changes what the model produces or how it is sampled. Recorded per
# benchmark in the results row so a comparison can VERIFY the two runs matched instead of
# assuming it.
GEN_KEYS = ("max_tokens", "temperature", "top_p", "n", "concurrency", "timeout", "seed")

# Above this share of truncated generations the score is measuring the token budget rather
# than the model, and must not be compared against a run at a different cap.
CAP_LIMITED_RATE = 0.10
# Above this share of REQUESTS that never returned an answer, the run is not a measurement and
# must not be reported as one. Measured 2026-09-01: asking a model whose context is 32768 for
# 32768 NEW tokens makes the server reject every request, and the scorer then completes
# normally and writes acc=nan with fail=500/500. Two models in one sweep were recorded "DONE"
# that way. A crash is caught by the exit status; this is the failure that does NOT crash.
FAILED_RATE_ABORT = 0.10

# Seconds between progress lines. A time cadence, not a count: OlympiadBench at a large cap runs
# for the better part of an hour, and a run that prints nothing cannot be distinguished from a
# stalled one by any watchdog or any person watching.
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", "30"))
_PROGRESS_WIDTH = int(os.environ.get("PROGRESS_WIDTH", "28"))
# Whether to redraw in place. A bar written with carriage returns into a redirected log becomes
# one unreadable line thousands of characters long, so the shape of the output depends on where
# it is going, not on a preference.
_PROGRESS_TTY = sys.stderr.isatty()


# Sentinel marking "the user did not name this flag". argparse only applies a default when
# the attribute is absent from the namespace, so pre-seeding with this reveals which
# generation parameters were EXPLICIT on the command line.
_EXPLICIT: set = set()

_UNSET = object()


# Room left for the prompt when clamping a cap to a model's context. MATH-500 prompts run a few
# hundred tokens; 2048 is generous and still leaves almost all of a large context for output.
PROMPT_HEADROOM = 2048


def model_context_limit(model_path: str):
    """Largest position a model can attend to, read from its own ``config.json``.

    Asking a server for more NEW tokens than the model's context makes it reject EVERY request.
    The run then completes normally and reports ``acc=nan`` with ``fail=N/N``, which reads like
    a score. That has now happened three times on three machines, so it is worth catching from
    data the model ships rather than from a rule someone has to remember.

    Args:
        model_path: Directory holding ``config.json``, or an identifier that is not a local
            path.

    Returns:
        The context length, or ``None`` when it cannot be determined -- an unreadable config is
        not evidence of a small context, so the caller must not clamp on ``None``.
    """
    import json as _json
    import os as _os

    for name in ("config.json",):
        f = _os.path.join(str(model_path), name)
        if not _os.path.isfile(f):
            continue
        try:
            cfg = _json.load(open(f))
        except Exception:
            return None
        for key in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            v = cfg.get(key)
            if isinstance(v, int) and v > 0:
                return v
        text = cfg.get("text_config") or {}
        v = text.get("max_position_embeddings") if isinstance(text, dict) else None
        return v if isinstance(v, int) and v > 0 else None
    return None


def clamp_max_tokens(requested: int, ctx_limit, headroom: int = PROMPT_HEADROOM):
    """Reduce a generation cap that cannot fit inside a model's context.

    Args:
        requested: The cap the caller asked for.
        ctx_limit: The model's context length, or ``None`` if unknown.
        headroom: Tokens reserved for the prompt.

    Returns:
        ``(effective_cap, reason)``. ``reason`` is ``None`` when nothing was changed, so a
        caller can report the clamp rather than applying it silently -- a cap that quietly
        differs from the one requested is the bug this file already carries a fix for.
    """
    if not isinstance(ctx_limit, int) or ctx_limit <= 0:
        return requested, None
    usable = ctx_limit - headroom
    if usable <= 0 or requested <= usable:
        return requested, None
    return usable, (
        f"max_tokens={requested} exceeds the model's context of {ctx_limit} "
        f"(reserving {headroom} for the prompt); clamped to {usable}. Without this every "
        f"request is rejected and the run reports acc=nan with fail=N/N."
    )


def explicit_gen_keys(parser, argv) -> set:
    """Which :data:`GEN_KEYS` the caller named on the command line.

    An explicit ``--max-tokens 32768`` must outrank :data:`BENCH_OVERRIDES`; without this
    distinction the table wins silently and a deliberate cap change is discarded while the
    run still reports success. That is not hypothetical -- it invalidated a full 30B
    re-score, which ran at the table's caps and looked fine.

    Args:
        parser: The argument parser that produced the real namespace.
        argv: The argument list to inspect, excluding the program name.

    Returns:
        The subset of :data:`GEN_KEYS` present in ``argv``.
    """
    ns = argparse.Namespace(**{k: _UNSET for k in GEN_KEYS})
    try:
        parser.parse_known_args(list(argv), namespace=ns)
    except SystemExit:
        # Unreachable from main(), which calls parse_args() first and would already
        # have exited. Reachable for any other caller -- and an empty set here IS the
        # pre-fix behaviour, so failing open silently would restore the bug this
        # function exists to prevent. Say so instead of returning quietly.
        print("WARNING: could not determine which generation flags were explicit; "
              "BENCH_OVERRIDES will win over every one of them", file=sys.stderr)
        return set()
    return {k for k in GEN_KEYS if getattr(ns, k, _UNSET) is not _UNSET}


def resolve_params(bench: str, args, explicit=frozenset()) -> dict:
    """Generation parameters for one benchmark: CLI values with per-benchmark overrides.

    Precedence is explicit CLI > :data:`BENCH_OVERRIDES` > CLI default. The table exists to
    give each benchmark a workable default cap, not to override a cap the caller asked for
    on purpose.

    Args:
        bench: Benchmark name.
        args: Parsed CLI namespace supplying the defaults.
        explicit: Generation keys the caller named on the command line, from
            :func:`explicit_gen_keys`. These are never overridden by the table.

    Returns:
        A dict with exactly :data:`GEN_KEYS`.

    Raises:
        ValueError: If the table names a key that is not a generation parameter. A typo
            there would otherwise be silently ignored and the benchmark would run at the
            default while the results row claimed the override.
    """
    over = BENCH_OVERRIDES.get(bench, {})
    unknown = sorted(set(over) - set(GEN_KEYS))
    if unknown:
        raise ValueError(
            f"BENCH_OVERRIDES[{bench!r}] names unknown generation parameter(s) {unknown}; "
            f"known: {list(GEN_KEYS)}"
        )
    out = {k: getattr(args, k) for k in GEN_KEYS}
    applied = {k: v for k, v in over.items() if k not in explicit}
    for k in sorted(set(over) & set(explicit)):
        if over[k] != out[k]:
            print(
                f"NOTE {bench}: using explicit --{k.replace('_', '-')}={out[k]} "
                f"(BENCH_OVERRIDES default {over[k]} not applied)",
                flush=True,
            )
    out.update(applied)
    # Clamp against the model's OWN declared context. This runs before any GPU work and
    # protects every revision that executes it, unlike a runtime abort which only protects a
    # machine that has pulled it -- three separate boxes have now reported acc=nan because a
    # cap exceeded a context, twice on revisions that predated the abort guard.
    _ctx = model_context_limit(getattr(args, "model_path", "") or "")
    _eff, _why = clamp_max_tokens(int(out.get("max_tokens") or 0), _ctx)
    if _why:
        print(f"NOTE {bench}: {_why}", file=sys.stderr, flush=True)
        out["max_tokens"] = _eff
    return out


PROMPT = (
    "Solve the following math problem. Reason step by step, and put your final answer "
    "within \\boxed{{}}.\n\n{problem}"
)


def load(bench: str, split: str = "all") -> list[dict]:
    """Load a benchmark's problems.

    Raises:
        FileNotFoundError: If the benchmark is absent, rather than silently scoring zero
            problems -- an empty suite would otherwise report a confident 0.0.
        ValueError: If a row lacks the uniform problem/answer schema, or none load.
    """
    f = DATA / bench / "test.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{bench}: {f} not found")
    raw = f.read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    keep = _split_indices(bench, split, raw)
    # Schema adapters. Every benchmark is normalised to problem/answer HERE rather than by
    # relaxing the check below, so a genuinely unknown schema still fails loudly instead of
    # silently scoring zero.
    #
    # OlympiadBench uses question/final_answer, and final_answer is a one-element list for
    # all 675 rows. 94 of them are flagged is_multiple_answer, meaning that single string
    # holds several answers; an exact-match grader will systematically mark those wrong.
    # They are kept (dropping them would quietly change the benchmark) and the count is
    # reported, so the score is read as a lower bound rather than mistaken for the model's
    # true rate.
    n_multi = 0
    for r in rows:
        if "question" in r and "final_answer" in r and "problem" not in r:
            fa = r["final_answer"]
            r["problem"] = r["question"]
            r["answer"] = fa[0] if isinstance(fa, list) and fa else fa
            if r.get("is_multiple_answer"):
                n_multi += 1
    if n_multi:
        print(f"NOTE {bench}: {n_multi}/{len(rows)} problems have multiple gold answers in "
              f"one string; exact-match grading will mark them wrong, so the reported "
              f"accuracy is a LOWER BOUND", file=sys.stderr)

    out = []
    for i, r in enumerate(rows):
        if "problem" not in r or "answer" not in r:
            raise ValueError(f"{bench} row {i}: expected problem/answer, got {sorted(r)}")
        if keep is not None and i not in keep:
            continue
        out.append({"problem": r["problem"], "answer": str(r["answer"]), "idx": i})
    if not out:
        raise ValueError(f"{bench}: no problems loaded")
    return out


def _split_indices(bench: str, split: str, raw: bytes) -> set[int] | None:
    """Indices to keep for `split`, or None for the whole benchmark.

    Searching and reporting on the same task set overstates a method`s gain (arXiv
    2607.12227), so evolution claims must search on one half and report on the other.

    The split addresses problems BY INDEX, so it is only meaningful against the exact file
    it was built from. The checksum is therefore verified on every load and a mismatch is
    fatal: silently scoring a different 250 problems would be undetectable in the output.
    """
    if split == "all":
        return None
    sf = Path(__file__).resolve().parent / f"{bench}_split.json"
    if not sf.exists():
        raise FileNotFoundError(
            f"--split {split} requested but {sf.name} does not exist. Splits are committed, "
            f"not generated per run: regenerating one lets a half be re-rolled until it "
            f"flatters the method. Create it once with make_split.py --write."
        )
    d = json.loads(sf.read_text())
    if split not in d:
        raise ValueError(f"{sf.name} has no '{split}' half; keys are {sorted(d)}")
    actual = hashlib.md5(raw).hexdigest()
    if actual != d["dataset_md5"]:
        raise ValueError(
            f"{bench}: dataset md5 is {actual} but {sf.name} was built against "
            f"{d['dataset_md5']}. The split addresses rows by index, so these indices now "
            f"identify different problems. Refusing to score."
        )
    return set(d[split])


_BOXED = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str | None:
    """Return the content of the last BALANCED \\boxed{...}.

    A regex like ``\\\\boxed\\{([^}]*)\\}`` truncates at the first inner brace, silently
    mangling ``\\boxed{\\frac{1}{2}}`` into ``\\frac{1``. A mangled extraction grades as
    wrong, so that bug surfaces as a plausible lower score rather than an error.

    Falls back to the last BALANCED box when the final one is cut off by the token cap.

    This used to return None whenever the last box was unbalanced, on the reasoning that a
    completion cut off mid-answer has not answered. An audit measured what that costs: it
    flips 0 items for the base model and 21/23 items for the two most degraded checkpoints
    (4.2% and 4.6%), because those completions are verbatim loops that emit the same box
    ~500 times and get cut mid-box on the last one. 100% of the flips are
    finish_reason=="length", and in 92-93% of them the last three boxes hold the SAME
    value -- the model committed long before the cap.

    So the old rule did not measure "did not answer", it measured "rambled", and it charged
    that only to the checkpoints being argued about. One case was caught scoring CORRECT at
    a 2048-token cap and WRONG at 8192 on byte-identical prefix text: raising the budget
    lowered the score.
    """
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    for start in reversed(starts):
        i = start
        depth, buf = 1, []
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
            buf.append(c)
            i += 1
    return None


def grade(pred_text: str, gold: str) -> bool:
    """True when the prediction matches the gold answer.

    Argument order into ``verify(gold, pred)`` matters and is deliberate: the audit found a
    real case where swapping them yields a false positive (gold ``5x-7y+11z+4=0``, pred
    ``0``).
    """
    boxed = extract_boxed(pred_text)
    if boxed is None:
        return False
    cands = [boxed]
    # math_verify parses "10\%" as 0.1, so a percent answer against a bare numeric gold
    # reads as wrong. Only tried when the gold itself carries no percent sign.
    if "%" not in gold and boxed.rstrip().endswith(("\\%", "%")):
        cands.append(re.sub(r"\\?%\s*$", "", boxed).strip())
    try:
        from math_verify import parse, verify

        g = parse(f"${gold}$")
        for c in cands:
            p = parse(f"${c}$")
            if g and p and verify(g, p):
                return True
    except Exception:
        pass
    norm = lambda s: re.sub(r"[\s,$]|\\left|\\right|\\!|\\,|\\?%", "", s).rstrip(".")
    return any(norm(c) == norm(gold) for c in cands)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Used instead of a normal binomial SE because at these counts the SE misleads: at 1/30
    the normal interval runs negative, and at 0/30 or 30/30 it is exactly 0, asserting
    certainty from a single unanimous sample.

    ``n <= 0`` is NOT a measurement and returns NaN. Two copies of this function used to
    return ``(0.0, 0.0)`` there, which prints ``[0.000, 0.000]`` for an EMPTY benchmark --
    a confident interval around zero, indistinguishable in the table from a real result of
    zero. A negative n is refused for the same reason: the clamp inside the square root
    hides it and the interval comes back with ``lo > hi``, outside [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------- endpoint identity ----
#
# WHICH WEIGHTS ANSWERED is not something the request payload says. The chat payload below
# carries no `lora_path`; an adapter is reachable only because sglang's
# `--lora-paths NAME=path` registers NAME as a MODEL ID, so `--model NAME` IS the routing
# decision. And an id the server has never heard of is not an error: sglang answers HTTP
# 200 and silently serves the BASE model. A run left on a plausible-looking default would
# therefore score the base weights while every line of its output looked normal.
#
# So: ask the endpoint what it serves, refuse to generate against an id that is not on the
# list, and record the id, the URL and the whole list in the results row. A score nobody
# can attribute afterwards is not a measurement.

CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"
# Seconds allowed for the model-list query. Deliberately NOT the generation timeout, which
# is minutes: this is one small GET, and a stuck endpoint should fail fast rather than hold
# a run open for the full generation budget before refusing.
MODELS_TIMEOUT = 60.0


def chat_url(base_url: str) -> str:
    """The chat-completions URL for an OpenAI-compatible base url.

    Args:
        base_url: The ``/v1`` base url, with or without a trailing slash.

    Returns:
        The exact URL every completion in a run is POSTed to.
    """
    return base_url.rstrip("/") + CHAT_PATH


def models_url(base_url: str) -> str:
    """The model-list URL for an OpenAI-compatible base url.

    Args:
        base_url: The ``/v1`` base url, with or without a trailing slash.

    Returns:
        The URL listing the model ids this endpoint will route.
    """
    return base_url.rstrip("/") + MODELS_PATH


async def list_served_models(session, base_url: str, timeout: float = MODELS_TIMEOUT):
    """Model ids the endpoint declares it serves.

    Args:
        session: An open ``aiohttp.ClientSession`` (or anything with the same ``get``).
        base_url: The ``/v1`` base url.
        timeout: Seconds to wait for the reply.

    Returns:
        The ids, in the order the endpoint reported them.

    Raises:
        RuntimeError: If the endpoint cannot be reached, answers non-200, or returns
            anything but a non-empty list of ids. It never returns an empty list, because
            "this endpoint serves nothing" and "we could not tell what it serves" must not
            both come out looking like a name that simply is not registered.
    """
    return [m["id"] for m in await list_served_model_records(session, base_url, timeout)]


async def list_served_model_records(session, base_url: str, timeout: float = MODELS_TIMEOUT):
    """Every record ``/v1/models`` reported, not only the ids.

    Split out of :func:`list_served_models` rather than duplicated, so the refusals below are
    the only copy and the two functions cannot answer differently about the same endpoint.

    Args:
        session: An open ``aiohttp.ClientSession``.
        base_url: The ``/v1`` base url.
        timeout: Seconds to wait for the reply.

    Returns:
        The model records, in the order the endpoint reported them.

    Raises:
        RuntimeError: Under exactly the conditions :func:`list_served_models` documents.
    """
    url = models_url(base_url)
    try:
        async with session.get(url, timeout=timeout) as r:
            if r.status != 200:
                raise RuntimeError(f"{url} answered HTTP {r.status}")
            payload = await r.json()
    except RuntimeError:
        raise
    except Exception as exc:  # transport error, timeout, non-JSON body
        raise RuntimeError(
            f"{url} could not be queried: {type(exc).__name__}: {exc}"
        ) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError(f"{url} returned no model list: {str(payload)[:200]!r}")
    recs = [m for m in data
            if isinstance(m, dict) and isinstance(m.get("id"), str)]
    if not recs:
        raise RuntimeError(f"{url} listed no model ids: {str(payload)[:200]!r}")
    return recs


async def verify_model(session, base_url: str, model, timeout: float = MODELS_TIMEOUT):
    """Refuse to generate unless the endpoint really serves ``model``, and describe the run.

    This is the whole point of the endpoint-identity block above: it turns a silent
    wrong-weights run into a loud failure before a single token is generated, and it
    returns the provenance that makes the resulting score attributable.

    Args:
        session: An open ``aiohttp.ClientSession``.
        base_url: The ``/v1`` base url.
        model: The requested model id. Falsy is itself a refusal -- there is no safe
            default, because an unregistered name is served by the base model.
        timeout: Seconds to wait for the model list.

    Returns:
        The attribution block to record in the results row: ``model`` (the resolved id),
        ``endpoint`` (the exact URL the completions are POSTed to) and ``served_models``
        (every id the endpoint listed, so a wrong id is diagnosable after the fact).

    Raises:
        SystemExit: If no model was named, if the id is not served, or if the list could
            not be fetched at all. Every one of those must stop the run: continuing means
            scoring whatever the server happens to have loaded.
    """
    if not model:
        raise SystemExit(
            "REFUSING TO RUN: no model id was given. The model id is the routing "
            f"decision -- {models_url(base_url)} lists what this endpoint serves, and an "
            "UNREGISTERED name is answered HTTP 200 by the BASE model, so there is no "
            "safe default. Pass --model with an id from that list."
        )
    try:
        served = await list_served_models(session, base_url, timeout)
    except RuntimeError as exc:
        raise SystemExit(
            f"REFUSING TO RUN: cannot verify that model {model!r} is served -- {exc}. "
            "Generating without this check risks silently scoring the BASE model, so an "
            "unverifiable endpoint is a hard stop, not a warning."
        ) from exc
    if model not in served:
        raise SystemExit(
            f"REFUSING TO RUN: {models_url(base_url)} does not serve the requested model.\n"
            f"  asked for: {model!r}\n"
            f"  available: {served}\n"
            "sglang answers HTTP 200 for an unregistered id and silently serves the BASE "
            "model, so this run would have scored the wrong weights with nothing in "
            "results.json to say so. A LoRA adapter registers as its own id via "
            "--lora-paths NAME=path; use an id exactly as printed above."
        )
    return {"model": model, "endpoint": chat_url(base_url), "served_models": list(served)}


# ------------------------------------------------------- servers with no tokenizer ----
#
# A TRAINING rollout server is not an evaluation server. AReaL launches sglang with
# `--skip-tokenizer-init`, which is right for the trainer because the trainer speaks token
# ids, and the consequence for us is total: the server holds no tokenizer at all, so EVERY
# text request fails for EVERY model id it serves, base snapshot included. Measured against
# A0's live server on 2026-09-02:
#
#   POST /v1/chat/completions {"messages": [...]}   -> HTTP 500
#       "Internal server error: 'NoneType' object has no attribute 'apply_chat_template'"
#   POST /v1/completions      {"prompt": "hello"}   -> HTTP 400
#       "The engine initialized with skip_tokenizer_init=True cannot accept text prompts.
#        Please provide input_ids or re-initialize the engine with skip_tokenizer_init=False."
#   POST /v1/completions      {"prompt": [ids...]}  -> HTTP 500 "Internal server error: 'text'"
#
# The third line decides the design, and is the reason this speaks `/generate` rather than
# `/v1/completions`. Token ids ARE accepted by `/v1/completions` -- the request gets as far as
# the length check, which is what produces the HTTP 400 for an over-budget cap -- but the
# OpenAI RESPONSE serialiser reads `ret["text"]` unconditionally and a tokenizer-less engine
# never produces it, so the reply cannot be built. `stream`, `echo` and `logprobs` were each
# tried and none of them changes it: the OpenAI surface of this server is unusable in BOTH
# directions.
#
# sglang's native `/generate` is not. It takes `input_ids`, returns `output_ids`, routes a
# LoRA adapter by `lora_path`, and returns per-token logprobs. It is also the endpoint AReaL's
# own bridge uses (`areal/v2/inference_service/sglang/bridge.py`), so this is the interface
# the training run is itself generating through rather than a second path invented for the
# evaluation.
#
# WHAT IS ADDITIVE AND WHAT IS NOT. Everything below engages only when the SERVER SAYS it has
# no tokenizer, and the server is asked rather than guessed at. On an ordinary tokenising
# endpoint -- every headline number now in the paper -- `server_capabilities` reports
# `has_tokenizer=True`, `build_generator` returns the same `generate` closure this file has
# always used, and the results row is byte-identical. `test_math_bench.py` asserts that on a
# fixed input rather than taking it on trust.

#: sglang's native generation endpoint. At the server ROOT, not under `/v1`.
GENERATE_PATH = "/generate"
#: sglang's own description of how it was launched. This is the only place the flags the
#: SERVER was started with are readable; a model's `config.json` cannot see any of them.
SERVER_INFO_PATH = "/get_server_info"


def root_url(base_url: str) -> str:
    """The server root behind an OpenAI-compatible ``/v1`` base url.

    Args:
        base_url: The ``/v1`` base url, with or without a trailing slash.

    Returns:
        The same url with a trailing ``/v1`` removed, because :data:`GENERATE_PATH` and
        :data:`SERVER_INFO_PATH` are served at the root and not under ``/v1``.
    """
    u = base_url.rstrip("/")
    return u[: -len("/v1")].rstrip("/") if u.endswith("/v1") else u


@dataclass(frozen=True)
class ServerCapabilities:
    """What one endpoint can do with TEXT, and how long a request it will accept.

    Read from the server, never inferred from the model. The two disagree here and the
    disagreement is the bug: A0's Qwen2.5-32B declares ``max_position_embeddings: 32768``
    while its rollout server was launched with ``--context-length 4096``.

    Attributes:
        has_tokenizer: False when the server was launched with ``--skip-tokenizer-init``,
            i.e. when no text request of any kind can succeed. Defaults to True for an
            endpoint that does not publish this, so an unknown server behaves exactly as it
            did before this class existed.
        context_limit: The largest total (prompt + completion) the server will accept, or
            None when it does not say.
        tokenizer_path: The tokenizer the server WOULD have used, so a local tokenizer that
            is a different model can be refused instead of silently building a wrong prompt.
        base_model: The served id of the base model, so an id that is NOT it can be routed as
            a LoRA adapter without a second request.
        source: Where the verdict came from, in words, for the results row and the logs.
    """

    has_tokenizer: bool
    context_limit: object
    tokenizer_path: str
    base_model: str
    source: str


async def server_capabilities(session, base_url: str, timeout: float = MODELS_TIMEOUT):
    """Ask the endpoint how it was launched.

    Deliberately does NOT read ``/v1/models``: that list is already fetched by
    :func:`verify_model`, its contents move between requests on a training server, and
    several tests pin the exact number of times it is read.

    Args:
        session: An open ``aiohttp.ClientSession``.
        base_url: The ``/v1`` base url.
        timeout: Seconds to wait.

    Returns:
        The capabilities. An endpoint that does not answer, or answers without the flag, is
        reported as tokenising with no known limit -- which is the behaviour every caller had
        before this function existed, so an unknown server is not degraded by its presence.
    """
    url = root_url(base_url) + SERVER_INFO_PATH
    info = None
    source = f"{SERVER_INFO_PATH} not available on this endpoint"
    try:
        async with session.get(url, timeout=timeout) as r:
            if r.status == 200:
                info = await r.json()
            else:
                source = f"{SERVER_INFO_PATH} answered HTTP {r.status}"
    except Exception as exc:  # transport error, timeout, non-JSON body
        source = f"{SERVER_INFO_PATH} could not be read: {type(exc).__name__}"
    if not isinstance(info, dict):
        return ServerCapabilities(True, None, "", "", source)
    skip = info.get("skip_tokenizer_init")
    has_tok = not skip if isinstance(skip, bool) else True
    ctx = info.get("context_length")
    return ServerCapabilities(
        has_tokenizer=has_tok,
        context_limit=ctx if isinstance(ctx, int) and ctx > 0 else None,
        tokenizer_path=str(info.get("tokenizer_path") or ""),
        base_model=str(info.get("model_path") or ""),
        source=f"{SERVER_INFO_PATH}: skip_tokenizer_init={skip!r} context_length={ctx!r}",
    )


class TokenIO:
    """The tokenizer the EVALUATION holds, for a server that holds none.

    One object so the encode and the decode provably come from the same tokenizer: a prompt
    built by one tokenizer and a completion read by another is a score of something nobody
    asked for, and it would look completely normal in the output.
    """

    def __init__(self, tokenizer, path: str = ""):
        """Wrap a loaded tokenizer.

        Args:
            tokenizer: A loaded HuggingFace tokenizer.
            path: Where it was loaded from, recorded in the results row.
        """
        self.tok = tokenizer
        self.path = str(path)

    @classmethod
    def from_model_path(cls, model_path: str, server_tokenizer_path: str = "") -> "TokenIO":
        """Load the tokenizer from a local snapshot, refusing one that is not the server's.

        Args:
            model_path: Directory holding ``tokenizer.json`` / ``tokenizer_config.json``.
            server_tokenizer_path: What the server says it would tokenise with, from
                :func:`server_capabilities`. Empty when the server does not say, in which
                case no comparison is possible and none is invented.

        Returns:
            The wrapper.

        Raises:
            ValueError: If no path was given, or if it names a DIFFERENT tokenizer from the
                server's. The second is the dangerous one: a different chat template turns
                the same problem into a different prompt, the model answers the prompt it was
                actually given, and the score is of a question nobody meant to ask.
        """
        if not model_path:
            raise ValueError(
                "this endpoint holds no tokenizer (it was launched with "
                "--skip-tokenizer-init), so the evaluation has to tokenise locally and needs "
                "the model snapshot on disk. Pass --model-path, or set "
                "SELFEVO_PERIODIC_EVAL_MODEL_PATH, to the directory holding tokenizer.json."
            )
        if server_tokenizer_path:
            mine = os.path.realpath(str(model_path))
            theirs = os.path.realpath(server_tokenizer_path)
            if mine != theirs:
                raise ValueError(
                    f"the server tokenises with {theirs} and this evaluation would tokenise "
                    f"with {mine}. A different tokenizer builds a different prompt out of the "
                    f"same problem, so the model would answer a question nobody asked and the "
                    f"score would look entirely normal. Refusing."
                )
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(str(model_path)), model_path)

    def encode_chat(self, content: str) -> list:
        """The token ids a tokenising server would have built from this user message.

        The same chat template, applied on this side of the wire. It is the server's own
        template because :meth:`from_model_path` refuses a tokenizer that is not the one the
        server named.

        Args:
            content: The user message, i.e. exactly what :data:`PROMPT` produced.

        Returns:
            The prompt token ids.
        """
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids) -> str:
        """Text for a list of generated token ids.

        Args:
            ids: The completion's token ids.

        Returns:
            The completion as text, which is what the grader is handed -- unchanged, and
            unaware that this path exists.
        """
        return self.tok.decode(list(ids), skip_special_tokens=True)


#: Distinct refusals already printed, so a rejection is announced once rather than once per
#: problem. A run that is being refused by the server must SAY so: the failure mode this
#: replaces is 64 silent retries followed by a NaN that reads like a score.
_TOKEN_ID_PROBLEMS: set = set()


def _report_token_id_problem(url: str, status: int, body: str) -> None:
    """Print one distinct token-id transport problem, once.

    Args:
        url: The endpoint that answered.
        status: Its HTTP status.
        body: The first part of its body, which carries the server's own explanation.
    """
    key = (status, body[:120])
    if key in _TOKEN_ID_PROBLEMS:
        return
    _TOKEN_ID_PROBLEMS.add(key)
    print(f"TOKEN-ID PATH {url} HTTP {status}: {body}", file=sys.stderr, flush=True)


async def generate_ids(
    session,
    base_url: str,
    prompt_ids,
    params: dict,
    tio: "TokenIO",
    lora_path: str = "",
    return_logprob: bool = False,
) -> dict:
    """One completion from a tokenizer-less server, spoken and read in token ids.

    Returns exactly the shape :func:`generate` returns, so everything downstream -- the
    extraction, the grader, the counters -- is handed TEXT and is unchanged.

    Args:
        session: An open ``aiohttp.ClientSession``.
        base_url: The ``/v1`` base url; the native endpoint is derived with :func:`root_url`.
        prompt_ids: Prompt token ids from :meth:`TokenIO.encode_chat`.
        params: The resolved generation parameters.
        tio: The tokenizer that produced ``prompt_ids``, used to decode the reply.
        lora_path: The adapter id to route to, or empty for the base model.
        return_logprob: Whether to ask for per-token logprobs, which the liveness probe needs.

    Returns:
        ``{"text", "finish_reason", "status"}``, plus ``"logprobs"`` when asked. ``status`` is
        ``ok`` when the server answered and the reply decoded, else ``failed``. A reply with
        no ``output_ids``, a decode that raises, and a generation the server ABORTED are all
        ``failed`` and NOT an empty ``ok``: an empty completion is a WRONG ANSWER that counts
        in the denominator, and charging a harness fault to the model as a wrong answer is a
        silent zero.
    """
    url = root_url(base_url) + GENERATE_PATH
    sampling = {
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_new_tokens": params["max_tokens"],
    }
    if params.get("seed") is not None:
        sampling["sampling_seed"] = params["seed"]
    payload = {
        "input_ids": list(prompt_ids),
        "sampling_params": sampling,
        "stream": False,
    }
    if lora_path:
        payload["lora_path"] = lora_path
    if return_logprob:
        payload["return_logprob"] = True
    failed = {"text": "", "finish_reason": None, "status": "failed"}
    if return_logprob:
        failed["logprobs"] = []
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=params["timeout"]) as r:
                if r.status != 200:
                    _report_token_id_problem(url, r.status, (await r.text())[:400])
                    await asyncio.sleep(1 + attempt)
                    continue
                d = await r.json()
        except Exception:
            await asyncio.sleep(1 + attempt)
            continue
        ids = d.get("output_ids")
        if not isinstance(ids, list):
            _report_token_id_problem(url, 200, f"reply carried no output_ids: {str(d)[:200]}")
            return dict(failed)
        try:
            text = tio.decode(ids)
        except Exception as exc:
            _report_token_id_problem(
                url, 200, f"could not decode {len(ids)} output ids: {type(exc).__name__}: {exc}"
            )
            return dict(failed)
        meta = d.get("meta_info") or {}
        reason = (meta.get("finish_reason") or {}).get("type")
        if reason == "abort":
            # THE SERVER THREW THE REQUEST AWAY. `pause_generation`, which AReaL sends around
            # every weight update, does exactly this: it drops the requests in flight and
            # waits until they are gone. Graded as an answer, an abort is a WRONG one -- and
            # measured against A0 on 2026-09-02, six of eight generations came back aborted
            # after as few as 75 characters and every one of them scored zero. A curve built
            # from that reads "the model gets everything wrong" when what happened is that
            # the run interrupted its own evaluation.
            #
            # So it is a FAILED request, exactly like a transport error: excluded from the
            # denominator, counted in n_failed, and refused outright by
            # `assert_scored_something` once it happens to more than half the problems. The
            # record keeps finish_reason="abort", so the artifact says how many and which.
            _report_token_id_problem(
                url,
                200,
                "the server ABORTED this generation (finish_reason=abort). Requests in "
                "flight are dropped by pause_generation, which AReaL sends around every "
                "weight update. Counted as a failed request, NOT graded as a wrong answer.",
            )
            aborted = dict(failed)
            aborted["finish_reason"] = "abort"
            return aborted
        out = {
            "text": text,
            "finish_reason": reason,
            "status": "ok",
        }
        if return_logprob:
            out["logprobs"] = [
                float(t[0])
                for t in (meta.get("output_token_logprobs") or [])
                if isinstance(t, (list, tuple)) and t and t[0] is not None
            ]
        return out
    return dict(failed)


def apply_server_context_limit(params: dict, caps: "ServerCapabilities", bench: str) -> bool:
    """Clamp the token budget against the limit the SERVER was launched with.

    THE GUARD THIS FIXES, stated plainly because the existing one looks like it covers this
    and does not. :func:`resolve_params` clamps against :func:`model_context_limit`, which
    reads the model's own ``config.json``. A ``config.json`` cannot see ``--context-length``,
    because that is a SERVER flag: A0's Qwen2.5-32B declares ``max_position_embeddings:
    32768`` while its rollout server was launched with ``--context-length 4096``. So the
    existing guard passes, and then every request is rejected with HTTP 400 and the run
    reports ``acc=nan`` with ``fail=N/N`` -- exactly the shape that guard exists to prevent.
    The number the server will honour is the one the server publishes, and that is this one.

    Args:
        params: The resolved generation parameters, modified in place.
        caps: What the server said about itself.
        bench: Benchmark name, for the note.

    Returns:
        True when the budget was reduced. Nothing is written into ``params`` when it was not,
        so a run against a server that never needed clamping produces the row it always did.
    """
    eff, why = clamp_max_tokens(int(params.get("max_tokens") or 0), caps.context_limit)
    if not why:
        return False
    print(
        f"NOTE {bench}: {why} This limit is the SERVER's, not the model's ({caps.source}); "
        f"the model's own config.json cannot see it, which is why the existing clamp passed.",
        file=sys.stderr,
        flush=True,
    )
    params["max_tokens_requested"] = params["max_tokens"]
    params["max_tokens"] = eff
    params["server_context_limit"] = caps.context_limit
    return True


def build_generator(args, params: dict, caps: "ServerCapabilities"):
    """The single function every completion in one benchmark is produced by.

    THE ADDITIVE GUARANTEE lives here and nowhere else. When the server has a tokenizer this
    returns a closure over the unchanged :func:`generate` and writes nothing into ``params``,
    so the standalone path -- which produced the numbers in the paper -- is bit for bit what
    it was. The token-id path is reachable only when the SERVER said it has no tokenizer.

    Args:
        args: The parsed namespace, for ``base_url``, ``model`` and ``model_path``.
        params: The resolved generation parameters, modified in place on the token-id path so
            the results row records that it was taken and what it tokenised with.
        caps: What the server said about itself.

    Returns:
        ``async (session, prompt_text) -> {"text", "finish_reason", "status"}``.

    Raises:
        ValueError: From :meth:`TokenIO.from_model_path`, when the server has no tokenizer and
            this side cannot supply one.
    """
    model = getattr(args, "model", "") or ""
    if caps.has_tokenizer:
        url = chat_url(args.base_url)

        async def gen_text(session, prompt):
            """One completion over the OpenAI chat endpoint, unchanged.

            Args:
                session: An open ``aiohttp.ClientSession``.
                prompt: The user message.

            Returns:
                The generation record.
            """
            return await generate(session, url, model, prompt, params)

        return gen_text

    tio = TokenIO.from_model_path(getattr(args, "model_path", "") or "", caps.tokenizer_path)
    lora = adapter_route(model, caps)
    params["token_id_path"] = True
    params["tokenizer_path"] = tio.path
    params["lora_path"] = lora
    params["server_context_limit"] = caps.context_limit
    params["server_capabilities"] = caps.source

    async def gen_ids(session, prompt):
        """One completion over the native token-id endpoint.

        Args:
            session: An open ``aiohttp.ClientSession``.
            prompt: The user message, tokenised here and decoded on the way back.

        Returns:
            The generation record, in the same shape the text path returns.
        """
        return await generate_ids(
            session, args.base_url, tio.encode_chat(prompt), params, tio, lora_path=lora
        )

    return gen_ids


def adapter_route(model: str, caps: "ServerCapabilities") -> str:
    """The ``lora_path`` one served id must be sent under, or the empty string for the base.

    ``/generate`` has no model field: an adapter is reached by ``lora_path`` or not at all,
    and sglang REFUSES a ``lora_path`` it has never loaded -- so sending one for the base
    model fails every request, and omitting one for an adapter silently scores the base. Both
    the benchmark and the liveness probe have to make this decision, so it is made once here
    rather than twice; two copies of a routing rule that must agree is how one of them drifts.

    Decided from what the SERVER called its base model, so it needs no second ``/v1/models``
    read of a list that moves between requests.

    Args:
        model: The served id to route to.
        caps: What the server said about itself.

    Returns:
        The ``lora_path`` to send, or ``""`` when the id is the base model (or unknown, in
        which case ``/generate``'s own default -- the base -- is the only safe choice).
    """
    base = getattr(caps, "base_model", "") or ""
    if not model:
        return ""
    if base and os.path.realpath(model) == os.path.realpath(base):
        return ""
    return model


def headline_max_tokens(bench: str) -> int:
    """The token budget the HEADLINE evaluation of this benchmark ran at.

    Read from the same two places a headline run reads it -- :data:`BENCH_OVERRIDES` first,
    then the parser's own default -- rather than written down a second time, so "the headline
    budget" cannot drift away from the budget the headline runs actually used.

    Args:
        bench: Benchmark name.

    Returns:
        The cap in tokens.
    """
    v = BENCH_OVERRIDES.get(bench, {}).get("max_tokens")
    if isinstance(v, int) and v > 0:
        return v
    return int(build_parser().get_default("max_tokens"))


async def generate(session, url: str, model: str, prompt: str, params: dict) -> dict:
    """One completion.

    Returns:
        ``{"text", "finish_reason", "status"}`` where ``status`` is one of ``ok`` (the
        endpoint answered), or ``failed`` (no usable response after retries). An HTTP 200
        carrying empty content is ``ok`` with empty text, NOT ``failed``: an empty
        completion is a wrong answer, and excluding it from the denominator would inflate
        accuracy.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": params["max_tokens"],
    }
    if params["seed"] is not None:
        payload["seed"] = params["seed"]
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=params["timeout"]) as r:
                if r.status != 200:
                    await asyncio.sleep(1 + attempt)
                    continue
                d = await r.json()
                ch = (d.get("choices") or [{}])[0]
                return {
                    "text": (ch.get("message") or {}).get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "status": "ok",
                }
        except Exception:
            await asyncio.sleep(1 + attempt)
    return {"text": "", "finish_reason": None, "status": "failed"}


async def run_bench(bench: str, args, gen_fh=None, explicit=None) -> dict:
    """Score one benchmark against the served model and return its results row.

    Args:
        bench: Benchmark name.
        args: Parsed CLI namespace.
        gen_fh: Open file handle; every completion is written to it as JSONL.
        explicit: Generation keys the caller named on the command line, from
            :func:`explicit_gen_keys`. Passed rather than read off :data:`_EXPLICIT`
            so the precedence fix travels with the call: an importer that never runs
            main() would otherwise get an empty set, and the table would silently
            win again. None falls back to the global for backwards compatibility.

    Returns:
        The results row, including the parameters this benchmark actually ran with.
    """
    import aiohttp

    probs = load(bench, getattr(args, "split", "all"))
    if args.limit:
        probs = probs[: args.limit]
    # Resolved ONCE per benchmark and threaded through, so every completion in this
    # benchmark provably used the same parameters and the results row can report them.
    params = resolve_params(bench, args, _EXPLICIT if explicit is None else explicit)
    sem = asyncio.Semaphore(params["concurrency"])
    conn = aiohttp.TCPConnector(limit=params["concurrency"])

    async with aiohttp.ClientSession(connector=conn) as session:
        # BEFORE any generation: refuse an id this endpoint does not serve, and fold the
        # resolved id, the URL and the served list into the parameters this row reports.
        # Without the first half a run scores whatever weights happen to be loaded; without
        # the second half nobody can tell afterwards which ones those were.
        params.update(await verify_model(session, args.base_url,
                                         getattr(args, "model", None)))

        # What the SERVER says about itself, which is the only place its launch flags are
        # readable. Two things come out of it: the real context limit (the model's own
        # config.json cannot see --context-length, and on a training server the two differ by
        # 8x), and whether text requests can succeed at all. Both leave `params` untouched on
        # an ordinary tokenising server that needs no clamp, so the standalone path's results
        # row is unchanged.
        caps = await server_capabilities(session, args.base_url)
        apply_server_context_limit(params, caps, bench)
        gen_one = build_generator(args, params, caps)

        async def one(idx: int, p: dict, k: int) -> dict:
            async with sem:
                r = await gen_one(session, PROMPT.format(problem=p["problem"]))
            boxed = extract_boxed(r["text"])
            correct = grade(r["text"], p["answer"]) if r["status"] == "ok" else None
            return {
                # p["idx"] is the row's position in the SOURCE FILE; `idx` is only its
                # position in this (possibly split-filtered) run. Writing the latter made
                # a --split report run emit 0..249 while a full run emitted 0..499, so a
                # paired comparison silently matched different problems to each other and
                # reported a confident, meaningless gap.
                "benchmark": bench, "idx": p["idx"], "run_pos": idx, "sample": k,
                "gold": p["answer"], "boxed": boxed,
                "finish_reason": r["finish_reason"], "status": r["status"],
                "correct": correct, "text": r["text"],
            }

        # Progress, because a long benchmark otherwise prints nothing between "endpoint up"
        # and its final line -- for OlympiadBench that is close to an hour of silence, during
        # which a stalled run and a healthy one look identical from outside.
        _tasks = [one(i, p, k) for i, p in enumerate(probs) for k in range(params["n"])]
        _total = len(_tasks)
        _t0 = time.time()
        _done = 0
        _next_report = 0.0
        recs = []
        for _fut in asyncio.as_completed(_tasks):
            recs.append(await _fut)
            _done += 1
            _elapsed = time.time() - _t0
            # Report on a time cadence rather than a count, so a slow benchmark still speaks
            # early and a fast one does not spam. First line comes at PROGRESS_EVERY_S.
            if _elapsed >= _next_report or _done == _total:
                _next_report = _elapsed + PROGRESS_EVERY_S
                _rate = _done / _elapsed if _elapsed > 0 else 0.0
                _eta = (_total - _done) / _rate if _rate > 0 else float("nan")
                _bad = sum(1 for r in recs if r["status"] != "ok")
                _frac = _done / _total
                _fill = int(round(_PROGRESS_WIDTH * _frac))
                _bar = "#" * _fill + "-" * (_PROGRESS_WIDTH - _fill)
                _line = (
                    f"  {bench:<14s} [{_bar}] {100.0 * _frac:5.1f}%  "
                    f"{_done}/{_total}  {_rate:.1f}/s  eta {_eta / 60:.1f}m"
                    + (f"  FAILING {_bad}" if _bad else "")
                )
                # On a terminal, redraw one line in place. In a log file a carriage return
                # produces an unreadable single mega-line, so there we print whole lines and
                # accept the scrollback -- the file is read after the fact, not watched.
                if _PROGRESS_TTY:
                    print("\r" + _line + "\033[K", end="", file=sys.stderr, flush=True)
                    if _done == _total:
                        print(file=sys.stderr, flush=True)
                else:
                    print(_line, file=sys.stderr, flush=True)

    # as_completed yields in FINISH order, which would make generations.jsonl differ run to run
    # for identical inputs and defeat any diff between two scorings. Restore submission order.
    recs.sort(key=lambda r: (r["run_pos"], r["sample"]))

    if gen_fh is not None:
        for rec in recs:
            gen_fh.write(json.dumps(rec) + "\n")
        gen_fh.flush()

    n_failed = sum(1 for r in recs if r["status"] == "failed")
    n_trunc = sum(1 for r in recs if r["finish_reason"] == "length")
    n_nobox = sum(1 for r in recs if r["status"] == "ok" and r["boxed"] is None)

    # Per-problem mean over its OK samples, then mean over problems (avg@n; pass@1 at n=1).
    per: list[float] = []
    for i in range(len(probs)):
        nn = params["n"]
        chunk = [r["correct"] for r in recs[i * nn : (i + 1) * nn] if r["correct"] is not None]
        if chunk:
            per.append(sum(chunk) / len(chunk))
    acc = statistics.mean(per) if per else float("nan")
    lo, hi = wilson(round(acc * len(per)), len(per)) if per else (float("nan"),) * 2
    return {
        "benchmark": bench,
        "n_problems": len(probs),
        "n_graded": len(per),
        "n_failed": n_failed,
        "n_truncated": n_trunc,
        "n_no_box": n_nobox,
        "accuracy": acc,
        "wilson_lo": lo,
        "wilson_hi": hi,
        # Provenance: the parameters this benchmark ACTUALLY ran with, not the CLI defaults.
        # Two rows are comparable only if these match, and recording them is what lets a
        # comparison check that instead of assuming it.
        "params": dict(params),
        "truncation_rate": (n_trunc / len(recs)) if recs else float("nan"),
        # True when the token budget, not the model, is plausibly setting the score.
        "cap_limited": bool(recs) and (n_trunc / len(recs)) > CAP_LIMITED_RATE,
        "seed": params["seed"],
        "temperature": params["temperature"],
    }


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser this script actually runs with.

    Kept as a function so :func:`explicit_gen_keys` and the tests exercise the SAME
    parser main() uses. A test that rebuilt an equivalent parser of its own would go
    on passing while the real one drifted -- and the precedence rule under test is
    defined entirely by which flags this parser saw.

    Returns:
        The configured parser.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8404/v1")
    ap.add_argument("--model", default=None,
                    help="model id to score, as listed by <base-url>/models. REQUIRED and "
                         "deliberately without a default: an unregistered id is answered "
                         "HTTP 200 by the BASE model, so a default silently scores the "
                         "wrong weights. A LoRA adapter registered with "
                         "--lora-paths NAME=path is addressed as NAME.")
    ap.add_argument("--benchmarks", default=",".join(SUITE))
    ap.add_argument("--limit", type=int, default=0, help="problems per benchmark, 0 = all")
    ap.add_argument("--split", default="all", choices=["all", "search", "report"],
                    help="committed half to score; 'all' is the whole benchmark. Evolution "
                         "claims must search on one half and report on the other.")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0, help="passed to the server; None to omit")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--gen-out", type=str, default="", help="JSONL of every completion")
    return ap


def main() -> int:
    """Run the requested benchmarks and return a process exit status."""
    ap = build_parser()
    args = ap.parse_args()
    # Explicit CLI generation flags outrank BENCH_OVERRIDES; see resolve_params.
    global _EXPLICIT
    _EXPLICIT = explicit_gen_keys(ap, sys.argv[1:])


    # avg@n at temperature 0 measures batching nondeterminism, not model uncertainty:
    # measured on amc23 n=4, only 11/40 problems produced 4 identical samples.
    if args.n > 1 and args.temperature == 0.0:
        print("ERROR: --n > 1 at temperature 0 measures nondeterminism, not uncertainty. "
              "Raise --temperature or set --n 1.", file=sys.stderr)
        raise SystemExit(2)

    # Refused here as well as in run_bench, and before any dataset or endpoint work: the
    # message a user sees for a missing flag should name the flag, not arrive as a failed
    # lookup against a model list.
    if not args.model:
        print("ERROR: --model is required. It names which weights answer: an id the "
              "server does not serve is answered HTTP 200 by the BASE model, so there is "
              "no safe default. Pass an id from <base-url>/models.", file=sys.stderr)
        raise SystemExit(2)

    gen_fh = open(args.gen_out, "w") if args.gen_out else None
    rows = []
    try:
        for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
            t0 = time.time()
            r = asyncio.run(run_bench(b, args, gen_fh, _EXPLICIT))
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(
                f"{r['benchmark']:<14} acc={r['accuracy']:.4f} "
                f"wilson=[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
                f"n={r['n_graded']}/{r['n_problems']} "
                f"fail={r['n_failed']} trunc={r['n_truncated']} nobox={r['n_no_box']} "
                f"({r['seconds']}s)",
                flush=True,
            )
    finally:
        if gen_fh:
            gen_fh.close()

    if args.out:
        # allow_nan=False: NaN is not valid JSON and silently breaks downstream readers.
        Path(args.out).write_text(
            json.dumps(
                [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                  for k, v in r.items()} for r in rows],
                indent=2, allow_nan=False,
            )
        )
        print(f"wrote {args.out}")

    for r in rows:
        if r["n_graded"] == 0:
            print(f"WARNING {r['benchmark']}: graded nothing; accuracy is meaningless",
                  file=sys.stderr)
        elif r["n_graded"] < r["n_problems"]:
            # A partial outage shrinks the denominator while still printing a
            # normal-looking score, and timeouts correlate with long/hard generations.
            print(f"WARNING {r['benchmark']}: only {r['n_graded']}/{r['n_problems']} "
                  f"graded ({r['n_failed']} failed); accuracy is over survivors and is "
                  "biased upward", file=sys.stderr)
        # A run that answered nothing is a broken run, not a zero score.
        _nfail = int(r.get("n_failed", 0) or 0)
        _ntot = int(r.get("n_problems", 0) or 0) or 1
        if _nfail / _ntot > FAILED_RATE_ABORT:
            raise SystemExit(
                f"ABORT {r['benchmark']}: {_nfail}/{_ntot} requests FAILED "
                f"({100.0 * _nfail / _ntot:.1f}%). This is not a score. The usual cause is "
                f"max_tokens={(r.get('params') or {}).get('max_tokens')} exceeding the "
                f"model's context window, "
                f"which makes the server reject every request; check config.json "
                f"max_position_embeddings and leave room for the prompt. Other causes: the "
                f"endpoint died, or the served model name is wrong."
            )
        if r.get("cap_limited"):
            print(f"CAP-LIMITED {r['benchmark']}: {r['n_truncated']}/{r['n_problems']} "
                  f"({r['truncation_rate']:.1%}) hit max_tokens="
                  f"{r['params']['max_tokens']} and were graded wrong. This score is "
                  "partly a property of the token budget; do NOT compare it against a run "
                  "at a different cap.", file=sys.stderr)
        elif r["n_truncated"]:
            print(f"NOTE {r['benchmark']}: {r['n_truncated']} generation(s) hit the token "
                  f"cap (max_tokens={r['params']['max_tokens']}) and were graded as wrong",
                  file=sys.stderr)

    # A benchmark that graded nothing is a FAILURE, not a score of nan. The warning above
    # already said so, but main() returned None and was called bare, so the process exited
    # 0 and every caller recorded success. A headroom sweep did exactly that: 1100 problems
    # errored instantly because max_tokens equalled the model's context length, and the
    # driver logged "0 qwen2.5-1.5b" as a clean run.
    dead = [r for r in rows if r["n_graded"] == 0]
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
