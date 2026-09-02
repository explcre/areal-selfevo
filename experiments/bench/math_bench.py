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
    """
    if n == 0:
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
    ids = [m.get("id") for m in data
           if isinstance(m, dict) and isinstance(m.get("id"), str)]
    if not ids:
        raise RuntimeError(f"{url} listed no model ids: {str(payload)[:200]!r}")
    return ids


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
    url = chat_url(args.base_url)
    sem = asyncio.Semaphore(params["concurrency"])
    conn = aiohttp.TCPConnector(limit=params["concurrency"])

    async with aiohttp.ClientSession(connector=conn) as session:
        # BEFORE any generation: refuse an id this endpoint does not serve, and fold the
        # resolved id, the URL and the served list into the parameters this row reports.
        # Without the first half a run scores whatever weights happen to be loaded; without
        # the second half nobody can tell afterwards which ones those were.
        params.update(await verify_model(session, args.base_url,
                                         getattr(args, "model", None)))

        async def one(idx: int, p: dict, k: int) -> dict:
            async with sem:
                r = await generate(session, url, args.model, PROMPT.format(problem=p["problem"]), params)
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
