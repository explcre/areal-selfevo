"""Periodic benchmark evaluation during training, on the committed ``search`` split.

WHY THIS EXISTS. Until now the only way to get a benchmark number out of a run was to stop
training, serve the checkpoint by hand and score it. That happened once, at A0's
``globalstep149``, and the incident is the whole motivation for this module: the manual eval
produced 0.4696 on OlympiadBench while its own adapter-liveness probe printed *"0 of 3 greedy
outputs differ between adapter and base"* -- and the eval ran on anyway. Without a curve, a
12853-step run can train to completion before anyone discovers the adapter never moved.

FIVE THINGS THIS MODULE REFUSES TO DO, each because the tree already records the incident.

1. **It does not implement a second scorer.** Generation and grading go through
   ``experiments/bench/math_bench.run_bench`` -- the same function that produced the 0.4696 --
   so the periodic curve and the final number cannot disagree. A periodic eval that could
   disagree with the final eval is worse than no periodic eval, because it would be believed.

2. **It cannot read the ``report`` half.** ``EVAL_SPLIT`` is a module constant, not a
   parameter, and it is enforced twice: :func:`require_committed_split` refuses a benchmark
   with no committed split *before* a token is generated (rather than falling through to
   ``split="all"``, which contains ``report`` -- the same silent-fallthrough shape as
   ``partition.py``'s retracted control bug), and :func:`assert_search_only` re-checks the
   ``idx`` of every generation that came back against the committed file afterwards. The
   second check is not derived from the first: it compares generations returned by the
   harness against indices read from ``olympiadbench_split.json`` on disk.

3. **It does not report a liveness verdict from greedy text alone.** The step-149 probe did
   exactly that and raised a FALSE ALARM: comparing logprobs on the same prompts showed the
   adapter and base differing by max |dlogprob| 0.041, i.e. the adapter *was* live and a small
   LoRA delta had moved the distribution without moving the argmax. ``greedy_differ_frac`` is
   still emitted, because it is what a reader intuitively looks for, but the verdict series
   ``liveness/is_live`` is decided on logprobs. A zero in ``greedy_differ_frac`` is a prompt
   that is too easy, not a dead adapter.

4. **It never reports a score for zero problems.** :func:`assert_scored_something` raises
   rather than let ``n_graded == 0`` reach W&B as a confident 0.0, and every failure path
   emits NaN plus a non-zero ``status_code`` series -- never a zero that reads as "the model
   got everything wrong".

5. **It does not pass a configured model id straight through.** The rollout server publishes
   the adapter as ``<lora_name>-v<version>`` and keeps only a rolling window of them; on A0
   the window advances every 10 to 20 seconds as training publishes weights, so an id written
   into the environment at launch is evicted long before the first evaluation at step 50 and
   every point on the curve would fail. The id is instead RESOLVED from ``/v1/models`` at each
   evaluation (:func:`resolve_adapter`), PINNED for the whole of that evaluation, and checked
   again afterwards (:func:`assert_still_served`): an evaluation that spanned an eviction is
   refused rather than reported as one number over two sets of weights. The resolved id and
   its version are recorded with the score, so a curve point can always be traced to the
   weights that produced it. It never falls back to the base model id -- scoring the base and
   labelling it with the arm's name is the exact failure this project has already recorded --
   and a resolution failure emits a status code and NaN, never a zero.

READING THE CURVE. Two series answer two different questions and both are needed:
``liveness/is_live`` says whether the adapter is doing anything at all, and
``<bench>/accuracy`` says whether what it is doing helps. :func:`diagnose` combines them into
``periodic_eval/diagnosis``, coded by :data:`DIAGNOSIS`, so "not learning yet" (0) is a
different number from "learning but not helping" (1). The accuracy comparison is made against
the interval of the run's OWN first measurement, stored in the state file, not against a
threshold chosen after the fact.

COST. At the default cadence this is a measured 3.9% of training throughput; see
``EXPERIMENTS.md`` and :func:`throughput_fraction`. It is small because ``limit`` is small,
and a small ``limit`` buys a wide Wilson interval: at 64 problems the interval is roughly
+/-12 points, which cannot resolve a small gain. That is the intended trade. The accuracy
curve is here to show gross trend and catch collapse; the liveness curve, which costs
seconds, is the sensitive early warning.

THE ENDPOINT is not configuration. AReaL's launcher allocates the sglang port and the server
binds the host interface rather than localhost, so no constant written before the run can be
right. :func:`base_url_from_rollout` reads it off the inference engine the trainer is itself
generating against; ``SELFEVO_PERIODIC_EVAL_BASE_URL`` remains only as an explicit override
for use outside a trainer, and when it is set the engine is not consulted at all.

CONFIGURATION is by environment variable, following ``SELFEVO_CLUSTER_LORA`` (``actor.py``)
and ``SELFEVO_CLUSTER_LORA_ADAPTERS`` (``fsdp_engine.py``), the established shape in this tree
for a selfevo feature on the live path. Unset means off, and off means the trainer does one
``os.environ`` lookup and nothing else.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "DIAGNOSIS",
    "EVAL_SPLIT",
    "METRIC_KEYS",
    "REPORT_SPLIT",
    "AdapterEvicted",
    "AdapterUnresolved",
    "BestValDecision",
    "BestValTracker",
    "EmptyEvaluation",
    "LivenessReport",
    "LivenessUnavailable",
    "PeriodicEvalConfig",
    "PeriodicEvalHook",
    "ReportSplitTouched",
    "ResolvedAdapter",
    "assert_scored_something",
    "assert_search_only",
    "assert_still_served",
    "base_url_from_rollout",
    "bench_namespace",
    "diagnose",
    "metrics_from",
    "require_committed_split",
    "resolve_adapter",
    "run_periodic_eval",
    "select_newest_adapter",
    "split_adapter_version",
    "throughput_fraction",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "experiments" / "bench"

#: The only half a periodic, decision-driving evaluation may ever read. Deliberately a
#: module constant and not a config field: every arm decision in this project quotes
#: ``search`` and ``report`` is quoted once in the final write-up, so a knob that could
#: select ``report`` is a knob that can silently spend the reporting half.
EVAL_SPLIT = "search"

#: The half this module must never touch. Named so the guards can say what they refused.
REPORT_SPLIT = "report"

ENV_ENABLE = "SELFEVO_PERIODIC_EVAL"
ENV_PREFIX = "SELFEVO_PERIODIC_EVAL_"

#: Integer codes for ``periodic_eval/diagnosis``. W&B plots numbers, so the reading of the
#: curve is pinned here rather than left to whoever looks at it.
DIAGNOSIS = {
    "adapter_inert": 0,  # liveness says the adapter changes nothing: NOT LEARNING YET
    "live_no_gain": 1,  # live, accuracy inside the baseline interval: learning, not helping
    "live_better": 2,  # live, accuracy above the baseline interval: learning and helping
    "live_worse": 3,  # live, accuracy below the baseline interval: learning and hurting
    "unknown": -1,  # liveness or accuracy unavailable; never fold this into 0
}

#: Status codes for ``periodic_eval/status_code``. Zero is success and every other value
#: names a specific failure, so a gap in the accuracy curve has a reason attached.
STATUS = {
    "ok": 0,
    "config_error": 1,
    "split_error": 2,
    "empty_evaluation": 3,
    "report_touched": 4,
    "liveness_unavailable": 5,
    "endpoint_error": 6,
    "adapter_unresolved": 7,  # /v1/models lists no version of the configured adapter
    "adapter_evicted": 8,  # the pinned version disappeared while this evaluation ran
}

#: Every key this module can emit. Declared, and asserted against the real emission in
#: ``test_periodic_eval.py``, following ``SELECTOR_METRIC_KEYS`` in ``harness/selectors.py``
#: -- the one namespace in this tree that declares its keys. A routed run once shipped with
#: an entirely empty ``route/`` namespace and nobody noticed until the config was read back.
METRIC_KEYS = frozenset(
    {
        "periodic_eval/step",
        "periodic_eval/status_code",
        "periodic_eval/diagnosis",
        "periodic_eval/cost/seconds",
        "periodic_eval/cost/throughput_frac",
        "periodic_eval/liveness/n_probes",
        "periodic_eval/liveness/greedy_differ_frac",
        "periodic_eval/liveness/max_abs_dlogprob",
        "periodic_eval/liveness/mean_abs_dlogprob",
        "periodic_eval/liveness/is_live",
        "periodic_eval/adapter/version",
        "periodic_eval/adapter/n_served",
        "periodic_eval/best_val/score",
        "periodic_eval/best_val/step",
        "periodic_eval/best_val/steps_since_best",
        "periodic_eval/best_val/is_best",
        "periodic_eval/best_val/should_stop",
    }
)

#: The feedback- and inference-budget namespace, emitted EVERY step (not only on evaluation
#: steps), because a matched-budget claim is about the whole run. ``GOAL.md`` section 4 lists
#: "Matched feedback budget -- query counts not logged" as NOT MET; these are the counts.
#: ``budget/`` rather than ``feedback/``: the latter already belongs to ``routing/outcomes.py``.
BUDGET_KEYS = frozenset(
    {
        "budget/verifier_calls_total",
        "budget/verifier_calls_step",
        "budget/verifier_retries_total",
        "budget/verifier_refusals_total",
        "budget/cache_hits_total",
        "budget/cache_enabled",
        "budget/generated_tokens_total",
        "budget/generated_tokens_step",
        "budget/counter_visible",
    }
)

#: Grader invocations the periodic evaluation itself consumed. Emitted only on evaluation
#: steps, and separate from the training-time verifier counters: they are different budgets
#: and adding them together would answer neither question.
EVAL_GRADER_KEY = "periodic_eval/eval_grader_calls"

#: Per-benchmark key suffixes, formatted as ``periodic_eval/<bench>/<suffix>``. Kept apart
#: from :data:`METRIC_KEYS` because the benchmark set is configurable; :func:`metric_keys_for`
#: is the single place either shape is spelled.
BENCH_METRIC_SUFFIXES = (
    "accuracy",
    "wilson_lo",
    "wilson_hi",
    "n_truncated",
    "n_graded",
    "n_problems",
    "seconds",
)

#: Probe prompts for the liveness measurement. Deliberately NOT the trivial arithmetic the
#: step-149 probe used ("Compute 12*13"): those are prompts where base and adapter agree to
#: the bit, which is what produced the false alarm. These are long-form and open-ended, the
#: regime where a small LoRA delta actually shows up in the distribution.
DEFAULT_PROBE_PROMPTS = (
    "Solve step by step: let x satisfy x^3 - 6x^2 + 11x - 6 = 0. Find every real root.",
    "Prove that the sum of the first n odd positive integers equals n^2.",
    "A fair coin is flipped 10 times. What is the probability of exactly 6 heads?",
    "Explain, step by step, why the derivative of x^x is x^x(1 + ln x).",
    "Find all integers n such that n^2 + 3n + 2 is prime, and justify the answer.",
    "Compute the area enclosed by y = x^2 and y = 2x, showing the integration.",
)

#: Below this maximum absolute per-token logprob difference the adapter is treated as inert.
#: Chosen against the step-149 measurement: a genuinely live adapter 149 steps in produced
#: max |dlogprob| of 0.041 and 0.012, so a threshold three orders of magnitude below the
#: smaller of those separates "live but barely moved" from "bit-identical".
DEFAULT_LIVE_EPS = 1e-4


class ReportSplitTouched(RuntimeError):
    """A periodic evaluation graded a problem from the reserved ``report`` half.

    Raised rather than logged. The reporting half is spent the moment it is read for a
    decision, and a run that keeps going after this has already spent it.
    """


class EmptyEvaluation(RuntimeError):
    """An evaluation graded no problems, or far fewer than it asked for.

    The failure this exists for does not crash: a benchmark that grades nothing still
    returns a results row, and ``accuracy`` of NaN or 0.0 plotted on a curve reads as a model
    that got everything wrong rather than a harness that asked nothing.
    """


class LivenessUnavailable(RuntimeError):
    """The endpoint would not return the logprobs the liveness verdict is decided on.

    Never downgraded to "assume live" or "assume inert": both are answers, and neither was
    measured. The caller records ``status_code`` and emits NaN.
    """


class AdapterUnresolved(RuntimeError):
    """``/v1/models`` lists no version of the configured adapter.

    A refusal, never a substitution. The one substitution available is the base model id,
    which the server *does* serve, and scoring the base under the arm's name is the failure
    this project has already recorded once. The caller emits ``adapter_unresolved`` and NaN,
    which is distinguishable in the series from an accuracy of zero.
    """


class AdapterEvicted(RuntimeError):
    """The pinned adapter version stopped being served while the evaluation was running.

    The server keeps a rolling window of published versions, so a version valid when it was
    resolved can be evicted mid-evaluation. The alternative to refusing is to re-resolve and
    keep generating, which produces one accuracy number over two different sets of weights
    and reports it as a point on a curve. This module pins one version and refuses instead.
    """


def _math_bench():
    """Import ``experiments/bench/math_bench.py``, the harness that scores our arms.

    Imported lazily and by path because ``experiments/`` is not a package and this module
    must stay importable (and cheap) on a box with no benchmark data at all.

    Returns:
        The imported ``math_bench`` module.

    Raises:
        ImportError: If the harness is not on this checkout, naming the path it looked in.
    """
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    try:
        import math_bench  # type: ignore
    except ImportError as exc:  # pragma: no cover - only on a broken checkout
        raise ImportError(
            f"could not import the benchmark harness from {BENCH_DIR}: {exc}. The periodic "
            f"eval deliberately has no scorer of its own; there is nothing to fall back to."
        ) from exc
    return math_bench


#: Adapter ids are ``<name>-v<version>``. That shape has exactly one writer,
#: ``areal.api.io_struct.get_versioned_lora_name``, and ``test_periodic_eval.py`` derives its
#: expectation from that function rather than from this pattern, so the two cannot drift
#: apart silently. Anchored at both ends: ``a0_math_code-v3`` must not match ``a0_math``.
_ADAPTER_VERSION_RE = re.compile(r"^(?P<name>.+)-v(?P<version>\d+)$")


def split_adapter_version(model_id: str) -> tuple[str, int | None]:
    """Split a served model id into its adapter name and its version number.

    Args:
        model_id: A served id, e.g. ``"a0_math-v13"``.

    Returns:
        ``(name, version)``. ``version`` is None when the id carries no ``-v<N>`` suffix,
        which is what a base snapshot path or a stable alias looks like.
    """
    m = _ADAPTER_VERSION_RE.match(model_id or "")
    if m is None:
        return (model_id or ""), None
    return m.group("name"), int(m.group("version"))


def select_newest_adapter(served, configured: str, base_model: str = "") -> str:
    """The newest served version of the configured adapter, or a refusal.

    The server publishes a new version of the adapter every training step and keeps only a
    rolling window of them, so "the adapter" is a moving id and the configured string names
    the FAMILY, not the version. Versions are compared as integers: ``-v10`` is newer than
    ``-v9``, which a lexicographic comparison gets backwards.

    Args:
        served: Model ids the endpoint reported, in any order.
        configured: The configured adapter id. A version suffix on it is ignored -- what is
            written in the environment at launch is stale by construction.
        base_model: The base model id, refused explicitly below.

    Returns:
        The id to generate against.

    Raises:
        AdapterUnresolved: If no served id belongs to the configured adapter. The endpoint
            always serves the base model, so "nothing matched" and "use the base" are one
            keystroke apart, and taking the second is how a base-model score acquires an
            arm's name.
    """
    name, _ = split_adapter_version(configured)
    if not name:
        raise AdapterUnresolved(
            "no adapter id was configured, so there is nothing to resolve against the "
            "served list. An unregistered id is answered by the BASE model."
        )
    candidates = []
    for mid in served:
        n, v = split_adapter_version(mid)
        if n == name and v is not None:
            candidates.append((v, mid))
    if candidates:
        chosen = max(candidates, key=lambda t: t[0])[1]
    elif configured in served:
        # No versioned member: the endpoint serves the name itself, which is what a stable
        # alias (or a hand-launched single-adapter server) looks like. Not a fallback to
        # something else -- it is the configured id, verbatim, and only when it is served.
        chosen = configured
    else:
        raise AdapterUnresolved(
            f"{sorted(served)} contains no version of adapter {name!r}. Refusing to "
            f"substitute: the base model IS served here, and scoring it under this arm's "
            f"name is the failure this check exists for."
        )
    if base_model and chosen == base_model:
        raise AdapterUnresolved(
            f"resolving adapter {name!r} selected the BASE model id {chosen!r}. Refusing: "
            f"an evaluation of the base labelled with the arm's name is worse than no point."
        )
    return chosen


@dataclass(frozen=True)
class ResolvedAdapter:
    """Which weights one evaluation was pinned to, recorded alongside its score.

    Attributes:
        model: The served id every generation in this evaluation used.
        version: Its integer version, or None for an unversioned id.
        n_served: How many ids the endpoint listed, so the size of the rolling window an
            eviction came out of is visible rather than assumed.
    """

    model: str
    version: int | None
    n_served: int


async def resolve_adapter(session, cfg) -> ResolvedAdapter:
    """Ask the endpoint what it serves and pin the newest version of the configured adapter.

    Called at every evaluation, never cached: the window advances every 10 to 20 seconds, so
    an id resolved at the previous evaluation is gone by this one.

    Args:
        session: An open ``aiohttp.ClientSession``.
        cfg: The resolved configuration.

    Returns:
        The pinned adapter.

    Raises:
        AdapterUnresolved: If no served id belongs to the configured adapter.
        RuntimeError: From the harness' own model listing, if the endpoint cannot be read.
    """
    served = await _math_bench().list_served_models(session, cfg.base_url)
    model = select_newest_adapter(served, cfg.model, cfg.base_model)
    return ResolvedAdapter(
        model=model, version=split_adapter_version(model)[1], n_served=len(served)
    )


async def assert_still_served(session, cfg, pinned: ResolvedAdapter) -> None:
    """Refuse an evaluation that outlived the version it was pinned to.

    The generations already happened; this decides whether the number they produced may be
    recorded. It may not, because a window that moved during the run means some completions
    could have been served by weights this point does not name -- and a curve point that
    cannot be traced to one set of weights is not a measurement.

    Args:
        session: An open ``aiohttp.ClientSession``.
        cfg: The resolved configuration.
        pinned: What :func:`resolve_adapter` selected before any generation.

    Raises:
        AdapterEvicted: If the pinned id is no longer listed.
    """
    served = await _math_bench().list_served_models(session, cfg.base_url)
    if pinned.model not in served:
        raise AdapterEvicted(
            f"{pinned.model!r} was served when this evaluation started and is gone now "
            f"(now serving {sorted(served)}). The rolling window advanced mid-evaluation, so "
            f"this score cannot be attributed to one set of weights and is not recorded."
        )


def base_url_from_rollout(rollout) -> str:
    """The ``/v1`` endpoint of the inference server the TRAINER is generating against.

    The address cannot be configured ahead of the run: AReaL's launcher allocates the sglang
    port, and on A0 the server binds the host interface (``172.28.127.18:32735``) rather than
    localhost, so a probe of localhost finds nothing and a constant is a guess. The trainer
    already holds the engine that knows, so this reads it there instead of adding a second
    place the address is written down.

    ``RemoteSGLangEngine`` composes ``RemoteInfEngine`` and does not re-export its
    ``addresses``, so the composed engine is read as a fallback. Narrow and deliberate: the
    alternative is an environment variable that nobody can fill in correctly.

    Args:
        rollout: The trainer's inference engine.

    Returns:
        ``http://<host>:<port>/v1`` for the first server the engine is using.

    Raises:
        RuntimeError: If the engine holds no address. There is deliberately no default: an
            evaluation against a guessed endpoint is an evaluation of unknown weights.
    """
    addrs = getattr(rollout, "addresses", None) or getattr(
        getattr(rollout, "_engine", None), "addresses", None
    )
    if not addrs:
        raise RuntimeError(
            f"{type(rollout).__name__} exposes no inference server address, so there is no "
            f"endpoint to evaluate against. Set {ENV_PREFIX}BASE_URL explicitly if this "
            f"deployment keeps the address somewhere else."
        )
    addr = str(addrs[0]).strip()
    if "://" not in addr:
        addr = "http://" + addr
    addr = addr.rstrip("/")
    return addr if addr.endswith("/v1") else addr + "/v1"


def split_path(bench: str) -> Path:
    """Path to a benchmark's committed search/report split file.

    Args:
        bench: Benchmark name, e.g. ``"olympiadbench"``.

    Returns:
        The path ``math_bench`` itself resolves for ``--split``, so the two cannot drift.
    """
    return BENCH_DIR / f"{bench}_split.json"


def require_committed_split(bench: str) -> tuple[frozenset[int], frozenset[int]]:
    """The committed ``search`` and ``report`` indices, or a refusal.

    This is the structural half of "the periodic path cannot read ``report``". A benchmark
    with no committed split has no ``search`` half to evaluate on, and the only other thing
    the harness could do is score ``all`` -- which *contains* ``report``. Falling through to
    a superset is exactly the shape of the retracted ``partition_from_config`` bug, where an
    unlisted mode silently ran the control's mechanism under a new label. So this refuses,
    before any generation is paid for, rather than degrading.

    Args:
        bench: Benchmark name.

    Returns:
        ``(search_indices, report_indices)`` as frozensets of row indices.

    Raises:
        FileNotFoundError: If no split has been committed for this benchmark.
        ValueError: If the file lacks either half, or the halves overlap.
    """
    sf = split_path(bench)
    if not sf.exists():
        raise FileNotFoundError(
            f"{bench}: no committed split at {sf}. A periodic evaluation may only read the "
            f"'{EVAL_SPLIT}' half, and scoring the whole benchmark instead would read "
            f"'{REPORT_SPLIT}' as well. Commit a split with make_{bench}_split.py --write, "
            f"or drop this benchmark from the periodic set."
        )
    d = json.loads(sf.read_text())
    for half in (EVAL_SPLIT, REPORT_SPLIT):
        if half not in d or not isinstance(d[half], list):
            raise ValueError(f"{sf.name} has no '{half}' half; keys are {sorted(d)}")
    search, report = frozenset(d[EVAL_SPLIT]), frozenset(d[REPORT_SPLIT])
    if search & report:
        raise ValueError(
            f"{sf.name}: {len(search & report)} indices are in BOTH halves, so no half is "
            f"held out and the split file itself is wrong."
        )
    return search, report


def assert_search_only(bench: str, records) -> int:
    """Refuse a set of generations that touched anything outside the ``search`` half.

    The post-condition half of the ``report`` guard, and deliberately independent of the
    pre-condition: it reads the committed indices off disk and compares them against the
    ``idx`` the harness actually returned for each generation. Nothing here is derived from
    the value being checked, so this fires if ``--split`` is ever wired to the wrong half,
    if the split file changes under a running job, or if the harness's own filtering breaks.

    Args:
        bench: Benchmark name.
        records: Generation records as ``run_bench`` writes them, each carrying ``idx``, the
            row's position in the SOURCE file (not its position in the filtered run).

    Returns:
        The number of distinct problem indices seen, so a caller can check it is not zero.

    Raises:
        ReportSplitTouched: If any generation came from ``report``, or from an index in
            neither half.
    """
    search, report = require_committed_split(bench)
    seen = {int(r["idx"]) for r in records}
    leaked = sorted(seen & report)
    if leaked:
        raise ReportSplitTouched(
            f"{bench}: {len(leaked)} of {len(seen)} evaluated problems are in the reserved "
            f"'{REPORT_SPLIT}' half (first: {leaked[:5]}). The reporting half is spent the "
            f"moment it drives a decision; refusing to record this evaluation."
        )
    stray = sorted(seen - search)
    if stray:
        raise ReportSplitTouched(
            f"{bench}: {len(stray)} evaluated problems are in NEITHER committed half "
            f"(first: {stray[:5]}), so the split file does not describe what was scored."
        )
    return len(seen)


def assert_scored_something(row: dict, requested: int) -> None:
    """Refuse a results row that graded nothing, or a small fraction of what it asked for.

    A silent-zero path this project has been bitten by more than once: a benchmark that
    grades nothing returns a normal-looking row, and its accuracy plotted on a curve is
    indistinguishable from a model that answered everything wrong.

    Args:
        row: A ``run_bench`` results row.
        requested: How many problems the evaluation asked for.

    Raises:
        EmptyEvaluation: If nothing was graded, or fewer than half the requested problems
            were, or the accuracy is not a real number despite problems having been graded.
    """
    graded = int(row.get("n_graded") or 0)
    if requested <= 0:
        raise EmptyEvaluation(
            f"{row.get('benchmark')}: asked for {requested} problems. An evaluation of "
            f"nothing is not a score of zero."
        )
    if graded == 0:
        raise EmptyEvaluation(
            f"{row.get('benchmark')}: graded 0 of {requested} problems "
            f"({row.get('n_failed')} failed). This is a broken evaluation, not a score."
        )
    if graded * 2 < requested:
        raise EmptyEvaluation(
            f"{row.get('benchmark')}: graded only {graded} of {requested} problems "
            f"({row.get('n_failed')} failed). Accuracy over survivors is biased upward and "
            f"is not comparable with the rest of the curve."
        )
    acc = row.get("accuracy")
    if acc is None or (isinstance(acc, float) and math.isnan(acc)):
        raise EmptyEvaluation(
            f"{row.get('benchmark')}: graded {graded} problems but accuracy is {acc!r}."
        )


def _env_int(env, name: str, default: int) -> int:
    """One integer environment setting, refusing a value it cannot parse.

    Args:
        env: The environment mapping.
        name: Variable name without :data:`ENV_PREFIX`.
        default: Value when the variable is unset.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the variable is set but not an integer. A typo must not silently
            fall back to the default and run at a cadence nobody asked for.
    """
    raw = env.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name}={raw!r} is not an integer") from exc


def _env_float(env, name: str, default: float) -> float:
    """One float environment setting, refusing a value it cannot parse.

    Args:
        env: The environment mapping.
        name: Variable name without :data:`ENV_PREFIX`.
        default: Value when the variable is unset.

    Returns:
        The parsed float.

    Raises:
        ValueError: If the variable is set but not a float.
    """
    raw = env.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name}={raw!r} is not a number") from exc


@dataclass(frozen=True)
class PeriodicEvalConfig:
    """Everything the periodic evaluation needs, resolved once and never re-read.

    Frozen because the cadence and the problem set must be identical at every point on the
    curve: two points generated with different parameters are not comparable, and this tree
    already carries the incident where a table silently overrode a cap mid-sweep.

    Attributes:
        enabled: Master switch. False means the trainer hook returns immediately.
        freq_steps: Evaluate when ``global_step`` is a positive multiple of this.
        benchmarks: Benchmark names, each of which must have a committed split.
        limit: Problems per benchmark, taken from the front of the ``search`` half. Fixed,
            so every point on the curve scores the SAME problems.
        max_tokens, temperature, top_p, n, concurrency, timeout, seed: Generation
            parameters, passed to the harness exactly as the CLI would.
        base_url: OpenAI-compatible ``/v1`` endpoint. Empty under a trainer, where
            :func:`base_url_from_rollout` supplies it from the engine the run is already
            generating against; set explicitly only outside one.
        model: The configured adapter, which names a FAMILY rather than a version. The
            served id is ``<model>-v<N>`` and only a rolling window of ``N`` exists at any
            moment, so :func:`resolve_adapter` picks the newest at each evaluation. A version
            suffix written here is ignored: it is stale by the first evaluation.
        base_model: The base model id on the same endpoint, used only by the liveness probe.
        probe_prompts: Prompts for the liveness probe.
        probe_max_tokens: Tokens generated per probe. Small: this is a distribution
            measurement, not a capability measurement.
        live_eps: Maximum absolute per-token logprob difference below which the adapter is
            called inert.
        patience: Evaluations without a new best before ``should_stop`` is raised.
        state_path: Where best-validation state is persisted, so a resumed run does not
            forget which checkpoint was best.
        out_dir: Optional directory for the full results row and generations of every point.
        explicit_gen_keys: Generation keys the operator set explicitly, which outrank the
            harness's per-benchmark override table exactly as a CLI flag would.
    """

    enabled: bool = False
    freq_steps: int = 50
    benchmarks: tuple[str, ...] = ("olympiadbench",)
    limit: int = 64
    max_tokens: int = 16384
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    concurrency: int = 32
    timeout: int = 1800
    seed: int = 0
    base_url: str = ""
    model: str = ""
    base_model: str = ""
    model_path: str = ""
    probe_prompts: tuple[str, ...] = DEFAULT_PROBE_PROMPTS
    probe_max_tokens: int = 32
    live_eps: float = DEFAULT_LIVE_EPS
    patience: int = 10
    state_path: str = ""
    out_dir: str = ""
    explicit_gen_keys: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls, env=None, require_base_url: bool = True) -> "PeriodicEvalConfig":
        """Build the configuration from the environment, validating eagerly.

        Every problem that can be detected without a GPU is detected here, at trainer
        start, rather than at the first evaluation an hour into the run: a benchmark with no
        committed split, a missing model id, a cadence that is not a positive integer. The
        ``harness_variants`` field in ``cli_args.py`` is the pattern -- resolve against the
        registry before any GPU is booked.

        Args:
            env: Environment mapping; defaults to ``os.environ``.
            require_base_url: Whether an endpoint must be named here. False when the caller
                holds the trainer's inference engine, which knows the address that no
                constant can: AReaL allocates the port and the server binds the host
                interface. Never a licence to guess -- :func:`base_url_from_rollout` refuses
                rather than defaulting, and :func:`_evaluate_async` refuses an empty one.

        Returns:
            A configuration. When the master switch is unset, a disabled default whose other
            fields are never read.

        Raises:
            ValueError: If the feature is enabled but configured with something that cannot
                produce a comparable curve.
            FileNotFoundError: If an enabled benchmark has no committed split.
        """
        env = os.environ if env is None else env
        if env.get(ENV_ENABLE, "") not in ("1", "true", "True", "yes"):
            return cls()

        benchmarks = tuple(
            b.strip()
            for b in env.get(ENV_PREFIX + "BENCHMARKS", "olympiadbench").split(",")
            if b.strip()
        )
        if not benchmarks:
            raise ValueError(f"{ENV_PREFIX}BENCHMARKS resolved to no benchmarks")
        for b in benchmarks:
            require_committed_split(b)  # refuse now, not after an hour of generation

        freq = _env_int(env, "FREQ_STEPS", 50)
        if freq < 1:
            raise ValueError(f"{ENV_PREFIX}FREQ_STEPS={freq} must be >= 1")
        limit = _env_int(env, "LIMIT", 64)
        if limit < 1:
            raise ValueError(
                f"{ENV_PREFIX}LIMIT={limit} must be >= 1. Zero means 'no limit' to the "
                f"harness, which is a very different and much more expensive run."
            )

        model = env.get(ENV_PREFIX + "MODEL", "").strip()
        base_model = env.get(ENV_PREFIX + "BASE_MODEL", "").strip()
        base_url = env.get(ENV_PREFIX + "BASE_URL", "").strip()
        if not model:
            raise ValueError(
                f"{ENV_PREFIX}MODEL is required: it names which weights answer, and an id "
                f"the server does not serve is answered HTTP 200 by the BASE model."
            )
        if not base_model:
            raise ValueError(
                f"{ENV_PREFIX}BASE_MODEL is required: the liveness metric is the difference "
                f"between the adapter and the base, and there is no default for 'the base'."
            )
        if base_model == model:
            raise ValueError(
                f"{ENV_PREFIX}BASE_MODEL and {ENV_PREFIX}MODEL are both {model!r}. The "
                f"liveness probe would compare the adapter with itself and report a "
                f"difference of exactly zero forever."
            )
        if require_base_url and not base_url:
            raise ValueError(
                f"{ENV_PREFIX}BASE_URL is required: the /v1 endpoint serving this run. "
                f"Under the trainer it is read from the rollout engine instead and this "
                f"variable should be left unset."
            )

        explicit = frozenset(
            k
            for k in ("max_tokens", "temperature", "top_p", "n", "concurrency", "timeout", "seed")
            if env.get(ENV_PREFIX + k.upper())
        )
        return cls(
            enabled=True,
            freq_steps=freq,
            benchmarks=benchmarks,
            limit=limit,
            max_tokens=_env_int(env, "MAX_TOKENS", 16384),
            temperature=_env_float(env, "TEMPERATURE", 0.0),
            top_p=_env_float(env, "TOP_P", 1.0),
            n=_env_int(env, "N", 1),
            concurrency=_env_int(env, "CONCURRENCY", 32),
            timeout=_env_int(env, "TIMEOUT", 1800),
            seed=_env_int(env, "SEED", 0),
            base_url=base_url,
            model=model,
            base_model=base_model,
            model_path=env.get(ENV_PREFIX + "MODEL_PATH", "").strip(),
            probe_max_tokens=_env_int(env, "PROBE_MAX_TOKENS", 32),
            live_eps=_env_float(env, "LIVE_EPS", DEFAULT_LIVE_EPS),
            patience=_env_int(env, "PATIENCE", 10),
            state_path=env.get(ENV_PREFIX + "STATE", "").strip(),
            out_dir=env.get(ENV_PREFIX + "OUT_DIR", "").strip(),
            explicit_gen_keys=explicit,
        )

    def should_run(self, global_step: int) -> bool:
        """Whether an evaluation is due at this step.

        Step 0 is excluded: the weights there are the initialisation, the adapter is zero by
        construction, and a liveness verdict of "inert" at step 0 is true and uninformative.

        Args:
            global_step: The trainer's global step.

        Returns:
            True when an evaluation should run.
        """
        return bool(self.enabled) and global_step > 0 and global_step % self.freq_steps == 0

    def metric_keys(self) -> frozenset[str]:
        """Every metric key this configuration can emit, benchmark keys included.

        Returns:
            The union of :data:`METRIC_KEYS`, :data:`EVAL_GRADER_KEY` and the per-benchmark
            keys. :data:`BUDGET_KEYS` are deliberately NOT included: they are emitted on
            every step, not only on evaluation steps, so they are not part of the set an
            evaluation must fill in with NaN.
        """
        return (
            METRIC_KEYS
            | {EVAL_GRADER_KEY}
            | {f"periodic_eval/{b}/{s}" for b in self.benchmarks for s in BENCH_METRIC_SUFFIXES}
        )


def bench_namespace(cfg: PeriodicEvalConfig, bench: str, model: str):
    """The argument namespace ``run_bench`` is called with, with the split forced.

    Built here rather than by the caller so ``split`` has exactly one assignment in this
    module and it is the constant :data:`EVAL_SPLIT`. ``benchmarks`` is set to the single
    benchmark being run so the namespace cannot be reused to sweep something else.

    Args:
        cfg: The resolved configuration.
        bench: The benchmark this namespace is for.
        model: The RESOLVED served id, required rather than defaulted to ``cfg.model``: the
            configured id is a family name whose version is stale by the first evaluation,
            and a default here would let a caller skip resolution without saying so.

    Returns:
        An ``argparse.Namespace`` with every attribute ``run_bench`` and ``resolve_params``
        read from it.
    """
    import argparse

    return argparse.Namespace(
        base_url=cfg.base_url,
        model=model,
        model_path=cfg.model_path,
        benchmarks=bench,
        limit=cfg.limit,
        split=EVAL_SPLIT,
        n=cfg.n,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        concurrency=cfg.concurrency,
        timeout=cfg.timeout,
        seed=cfg.seed,
        out="",
        gen_out="",
    )


class _GraderCounter:
    """Count real calls to the harness's grader while an evaluation runs.

    The grader lives in ``experiments/bench/math_bench.py``, which another agent owns and
    this module treats as read-only. Counting is therefore done by temporarily rebinding
    ``math_bench.grade`` to a wrapper *around the real function* for the duration of one
    evaluation. The wrapper IS the call site, so this counts actual invocations rather than
    inferring them from ``n_problems x n``, which would be wrong wherever a generation
    failed and was never graded at all.

    Used as a context manager so the rebinding cannot outlive the evaluation.
    """

    def __init__(self, mb):
        """Wrap one harness module.

        Args:
            mb: The imported ``math_bench`` module.
        """
        self.mb = mb
        self.calls = 0
        self._original = None

    def __enter__(self) -> "_GraderCounter":
        """Install the counting wrapper.

        Returns:
            Self, so the caller can read ``calls`` afterwards.
        """
        self._original = self.mb.grade
        original = self._original

        def counting_grade(pred_text, gold):
            """Grade one completion and count the call.

            Args:
                pred_text: The model's completion.
                gold: The gold answer.

            Returns:
                Exactly what the real grader returns.
            """
            self.calls += 1
            return original(pred_text, gold)

        self.mb.grade = counting_grade
        return self

    def __exit__(self, *exc) -> None:
        """Restore the real grader, whatever happened."""
        if self._original is not None:
            self.mb.grade = self._original
        self._original = None


async def run_one_benchmark(
    cfg: PeriodicEvalConfig, bench: str, model: str
) -> tuple[dict, list[dict], int]:
    """Score one benchmark's ``search`` half through the production harness.

    Args:
        cfg: The resolved configuration.
        bench: Benchmark name.
        model: The pinned served id, from :func:`resolve_adapter`.

    Returns:
        ``(row, records, grader_calls)``: the harness's own results row, every generation it
        produced, and the number of real grader invocations it consumed.

    Raises:
        ReportSplitTouched: If any graded problem was outside the ``search`` half.
        EmptyEvaluation: If the evaluation graded nothing usable.
    """
    mb = _math_bench()
    require_committed_split(bench)  # before any generation is paid for
    args = bench_namespace(cfg, bench, model)
    buf = io.StringIO()
    t0 = time.time()
    with _GraderCounter(mb) as counter:
        row = await mb.run_bench(bench, args, buf, cfg.explicit_gen_keys)
    row["seconds"] = round(time.time() - t0, 1)
    records = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    # Order matters: refuse a leak before refusing an empty run, because a leaked evaluation
    # has already spent the reporting half whether or not it also graded enough problems.
    assert_search_only(bench, records)
    assert_scored_something(row, int(row.get("n_problems") or 0))
    return row, records, counter.calls


@dataclass(frozen=True)
class LivenessReport:
    """What the adapter-versus-base probe measured.

    Attributes:
        n_probes: Prompts compared.
        greedy_differ_frac: Fraction whose greedy TEXT differs. Emitted because it is what a
            reader looks for first, but NOT the verdict: at A0's step 149 this was 0.0 on a
            demonstrably live adapter.
        max_abs_dlogprob: Largest absolute per-token logprob difference over all probes. This
            is the verdict signal, because a small LoRA delta moves the distribution long
            before it moves the argmax.
        mean_abs_dlogprob: Mean of the same quantity, for a smoother curve.
        is_live: 1 when ``max_abs_dlogprob`` exceeds the configured epsilon, else 0.
        n_tokens_compared: Token positions that could be compared, so a verdict resting on
            almost no evidence is visible rather than implied.
    """

    n_probes: int
    greedy_differ_frac: float
    max_abs_dlogprob: float
    mean_abs_dlogprob: float
    is_live: int
    n_tokens_compared: int


async def _greedy_with_logprobs(session, url: str, model: str, prompt: str, cfg) -> tuple[str, list[float]]:
    """One greedy completion together with its per-token logprobs.

    Args:
        session: An open ``aiohttp.ClientSession``.
        url: The chat-completions URL.
        model: The model id to route to.
        prompt: The probe prompt.
        cfg: The resolved configuration, for ``probe_max_tokens`` and ``timeout``.

    Returns:
        ``(text, logprobs)`` where ``logprobs`` holds one value per generated token.

    Raises:
        LivenessUnavailable: If the endpoint fails, or answers without logprobs. Both are
            refusals rather than a defaulted verdict: "assume live" and "assume inert" are
            each an answer nobody measured.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": cfg.probe_max_tokens,
        "logprobs": True,
        "seed": cfg.seed,
    }
    try:
        async with session.post(url, json=payload, timeout=cfg.timeout) as r:
            if r.status != 200:
                raise LivenessUnavailable(f"{url} answered HTTP {r.status} for model {model!r}")
            d = await r.json()
    except LivenessUnavailable:
        raise
    except Exception as exc:
        raise LivenessUnavailable(f"{url} could not be queried: {type(exc).__name__}: {exc}") from exc

    ch = (d.get("choices") or [{}])[0]
    text = (ch.get("message") or {}).get("content") or ""
    content = ((ch.get("logprobs") or {}).get("content")) or []
    lps = [t.get("logprob") for t in content if isinstance(t, dict) and t.get("logprob") is not None]
    if not lps:
        raise LivenessUnavailable(
            f"{url} returned no per-token logprobs for model {model!r}. The liveness verdict "
            f"is decided on logprobs because greedy text alone raised a FALSE ALARM on a live "
            f"adapter at A0 step 149; without them there is no verdict to report."
        )
    return text, [float(x) for x in lps]


async def measure_liveness(session, cfg: PeriodicEvalConfig, model: str) -> LivenessReport:
    """Compare the adapter against the base model on a fixed probe set.

    Both ids are served by the same endpoint -- sglang routes a LoRA adapter by model id --
    so this is two requests per prompt against one server and costs seconds, not minutes.

    Args:
        session: An open ``aiohttp.ClientSession``.
        cfg: The resolved configuration.
        model: The pinned served id. The probe must measure the SAME weights the benchmark
            scored, or the liveness verdict belongs to a different point than the accuracy.

    Returns:
        The measurement.

    Raises:
        LivenessUnavailable: If the endpoint would not answer, or answered without logprobs,
            or no token position could be compared at all.
    """
    url = _math_bench().chat_url(cfg.base_url)
    differ = 0
    max_d = 0.0
    sum_d = 0.0
    n_tok = 0
    for prompt in cfg.probe_prompts:
        a_text, a_lp = await _greedy_with_logprobs(session, url, model, prompt, cfg)
        b_text, b_lp = await _greedy_with_logprobs(session, url, cfg.base_model, prompt, cfg)
        if a_text != b_text:
            differ += 1
        # Compare only positions both sides produced. When the argmax paths diverge the
        # shared prefix is still a valid comparison, and the divergence itself is already
        # counted by `differ`.
        for x, y in zip(a_lp, b_lp):
            d = abs(x - y)
            max_d = max(max_d, d)
            sum_d += d
            n_tok += 1
    if n_tok == 0:
        raise LivenessUnavailable(
            "no token position could be compared between adapter and base; the probe "
            "measured nothing and must not report a verdict."
        )
    return LivenessReport(
        n_probes=len(cfg.probe_prompts),
        greedy_differ_frac=differ / len(cfg.probe_prompts),
        max_abs_dlogprob=max_d,
        mean_abs_dlogprob=sum_d / n_tok,
        is_live=int(max_d > cfg.live_eps),
        n_tokens_compared=n_tok,
    )


@dataclass(frozen=True)
class BestValDecision:
    """The outcome of one best-validation update.

    Attributes:
        is_best: Whether this evaluation set a new best.
        should_stop: Whether patience has been exhausted.
        best_step: The step holding the best score so far.
        best_score: That score.
        steps_since_best: Evaluations since the best, the quantity patience counts.
        checkpoint: The checkpoint recorded for ``best_step``, or the empty string.
    """

    is_best: bool
    should_stop: bool
    best_step: int
    best_score: float
    steps_since_best: int
    checkpoint: str


class BestValTracker:
    """Best-validation checkpoint selection with patience, on the ``search``-split signal.

    The project standard is to train to a ceiling with patience and select the best
    validation checkpoint against **the same signal used for decisions**. That signal is the
    ``search`` half, which is precisely what the periodic evaluation measures, so selection
    reads this curve and nothing else.

    State is persisted as JSON so a resumed run does not silently restart its patience
    counter or forget which checkpoint was best -- the run is 12853 steps per epoch and
    will be resumed.
    """

    def __init__(self, patience: int = 10, state_path: str | Path = ""):
        """Create a tracker, loading persisted state when a path is given and exists.

        Args:
            patience: Evaluations without improvement before ``should_stop`` is raised.
            state_path: Where to persist state. Empty disables persistence.
        """
        self.patience = int(patience)
        self.state_path = Path(state_path) if state_path else None
        self.best_step = -1
        self.best_score = -math.inf
        self.best_checkpoint = ""
        self.n_since_best = 0
        self.first_score: float | None = None
        self.first_wilson: tuple[float, float] | None = None
        self.history: list[dict] = []
        if self.state_path is not None and self.state_path.exists():
            self.load_state_dict(json.loads(self.state_path.read_text()))

    def state_dict(self) -> dict:
        """Serialisable state.

        Returns:
            A JSON-safe dict holding the best point, the patience counter and the first
            measurement's interval.
        """
        return {
            "best_step": self.best_step,
            "best_score": None if self.best_score == -math.inf else self.best_score,
            "best_checkpoint": self.best_checkpoint,
            "n_since_best": self.n_since_best,
            "first_score": self.first_score,
            "first_wilson": list(self.first_wilson) if self.first_wilson else None,
            "history": self.history,
            "patience": self.patience,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore state written by :meth:`state_dict`.

        Args:
            d: A previously serialised state.
        """
        self.best_step = int(d.get("best_step", -1))
        bs = d.get("best_score")
        self.best_score = -math.inf if bs is None else float(bs)
        self.best_checkpoint = d.get("best_checkpoint", "") or ""
        self.n_since_best = int(d.get("n_since_best", 0))
        fs = d.get("first_score")
        self.first_score = None if fs is None else float(fs)
        fw = d.get("first_wilson")
        self.first_wilson = (float(fw[0]), float(fw[1])) if fw else None
        self.history = list(d.get("history") or [])

    def update(
        self,
        step: int,
        score: float,
        checkpoint: str = "",
        wilson: tuple[float, float] | None = None,
        model: str = "",
    ) -> BestValDecision:
        """Record one evaluation and decide whether it is the new best.

        Strictly greater than, not greater-or-equal: on a tie the EARLIER checkpoint wins.
        A run whose score plateaus would otherwise keep advancing "best" to the latest
        checkpoint, which is the no-op this selection exists to avoid -- selecting the last
        checkpoint is exactly what happens with no selection at all.

        Args:
            step: The global step this score was measured at.
            score: The decision signal, i.e. accuracy on the ``search`` half.
            checkpoint: The checkpoint this score belongs to, recorded so the best one can be
                found afterwards without re-deriving the cadence.
            wilson: The measurement's Wilson bounds, stored for the FIRST evaluation only and
                used later as the baseline interval by :func:`diagnose`.
            model: The RESOLVED served id these weights answered as, written into the history
                so a point on the curve can be traced back to the weights that produced it.
                The checkpoint path alone cannot do that here: the adapter id the server
                answered is a different name from the checkpoint the trainer wrote.

        Returns:
            The decision.
        """
        if self.first_score is None and not (isinstance(score, float) and math.isnan(score)):
            self.first_score = float(score)
            if wilson is not None:
                self.first_wilson = (float(wilson[0]), float(wilson[1]))
        is_best = score > self.best_score
        if is_best:
            self.best_step = int(step)
            self.best_score = float(score)
            self.best_checkpoint = checkpoint or ""
            self.n_since_best = 0
        else:
            self.n_since_best += 1
        self.history.append(
            {
                "step": int(step),
                "score": float(score),
                "checkpoint": checkpoint,
                "model": model,
            }
        )
        decision = BestValDecision(
            is_best=is_best,
            should_stop=self.n_since_best >= self.patience,
            best_step=self.best_step,
            best_score=self.best_score,
            steps_since_best=self.n_since_best,
            checkpoint=self.best_checkpoint,
        )
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state_dict(), indent=1) + "\n")
        return decision


def diagnose(liveness: LivenessReport | None, accuracy: float, baseline: tuple[float, float] | None) -> int:
    """Read the curve: is the adapter inert, or live but not helping, or actually helping?

    This is the question requirement 5 exists for, and the two inputs answer different halves
    of it. Liveness comes first because an inert adapter makes the accuracy number
    meaningless -- it is the base model's score wearing the arm's label. Only once the
    adapter is known to be doing something does the accuracy comparison mean anything.

    The comparison is against the run's OWN first measurement's Wilson interval, not a
    threshold: a difference inside the interval of the point it is compared with is not a
    difference, and picking a fixed threshold afterwards is how a noise band becomes a claim.

    Args:
        liveness: The probe result, or None when it could not be measured.
        accuracy: This evaluation's accuracy on the ``search`` half.
        baseline: The Wilson bounds of the run's first evaluation, or None on the first one.

    Returns:
        One of the values of :data:`DIAGNOSIS`.
    """
    if liveness is None:
        return DIAGNOSIS["unknown"]
    if not liveness.is_live:
        return DIAGNOSIS["adapter_inert"]
    if baseline is None or accuracy is None or (isinstance(accuracy, float) and math.isnan(accuracy)):
        return DIAGNOSIS["unknown"]
    lo, hi = baseline
    if accuracy > hi:
        return DIAGNOSIS["live_better"]
    if accuracy < lo:
        return DIAGNOSIS["live_worse"]
    return DIAGNOSIS["live_no_gain"]


def throughput_fraction(eval_seconds: float, freq_steps: int, step_seconds: float) -> float:
    """What share of training wall time this evaluation cadence costs.

    Reported rather than assumed, and computed from the run's own measured step time so it
    stays true when the step time changes.

    Args:
        eval_seconds: Wall seconds one evaluation took.
        freq_steps: Steps between evaluations.
        step_seconds: Measured seconds per training step.

    Returns:
        The fraction of training wall time spent evaluating, or NaN when the step time is
        unknown. NaN rather than 0.0: an unknown cost is not a free one.
    """
    if not step_seconds or step_seconds <= 0 or freq_steps < 1:
        return float("nan")
    return eval_seconds / (freq_steps * step_seconds)


def metrics_from(
    global_step: int,
    rows: dict[str, dict],
    liveness: LivenessReport | None,
    decision: BestValDecision | None,
    seconds: float,
    throughput_frac: float,
    status_code: int,
    diagnosis: int,
    adapter: "ResolvedAdapter | None" = None,
) -> dict[str, float]:
    """Flatten one evaluation into the metric dict the trainer's own logger commits.

    Every value is a float, because that is what ``StatsLogger.commit`` forwards to W&B, and
    a missing measurement is NaN rather than 0.0 -- a zero accuracy reads as a model that got
    everything wrong.

    Args:
        global_step: The step evaluated.
        rows: Benchmark name to its ``run_bench`` results row.
        liveness: The probe result, or None if unavailable.
        decision: The best-validation decision, or None if selection did not run.
        seconds: Wall seconds this evaluation took.
        throughput_frac: Share of training throughput it cost.
        status_code: One of :data:`STATUS`.
        diagnosis: One of :data:`DIAGNOSIS`.
        adapter: The pinned adapter, or None when resolution never succeeded. Its version is
            emitted as its own series so every point on the accuracy curve carries the
            version of the weights that produced it.

    Returns:
        A flat mapping of metric key to float.
    """
    nan = float("nan")
    out: dict[str, float] = {
        "periodic_eval/step": float(global_step),
        "periodic_eval/status_code": float(status_code),
        "periodic_eval/diagnosis": float(diagnosis),
        "periodic_eval/cost/seconds": float(seconds),
        "periodic_eval/cost/throughput_frac": float(throughput_frac),
        "periodic_eval/liveness/n_probes": float(liveness.n_probes) if liveness else nan,
        "periodic_eval/liveness/greedy_differ_frac": float(liveness.greedy_differ_frac) if liveness else nan,
        "periodic_eval/liveness/max_abs_dlogprob": float(liveness.max_abs_dlogprob) if liveness else nan,
        "periodic_eval/liveness/mean_abs_dlogprob": float(liveness.mean_abs_dlogprob) if liveness else nan,
        "periodic_eval/liveness/is_live": float(liveness.is_live) if liveness else nan,
        "periodic_eval/adapter/version": (
            float(adapter.version) if adapter is not None and adapter.version is not None else nan
        ),
        "periodic_eval/adapter/n_served": float(adapter.n_served) if adapter is not None else nan,
        "periodic_eval/best_val/score": float(decision.best_score) if decision else nan,
        "periodic_eval/best_val/step": float(decision.best_step) if decision else nan,
        "periodic_eval/best_val/steps_since_best": float(decision.steps_since_best) if decision else nan,
        "periodic_eval/best_val/is_best": float(decision.is_best) if decision else nan,
        "periodic_eval/best_val/should_stop": float(decision.should_stop) if decision else nan,
    }
    for bench, row in rows.items():
        for suffix in BENCH_METRIC_SUFFIXES:
            v = row.get(suffix)
            out[f"periodic_eval/{bench}/{suffix}"] = nan if v is None else float(v)
    return out


async def _evaluate_async(
    cfg: PeriodicEvalConfig,
) -> tuple[dict[str, dict], dict[str, list], LivenessReport, int, ResolvedAdapter]:
    """Pin one adapter version, run every benchmark and the probe against it, then re-check.

    THE PINNING DECISION, stated here because the alternative is defensible and this is not
    the one taken: the served id is resolved ONCE, before any generation, and every request
    in this evaluation uses it. Re-resolving mid-evaluation would keep the evaluation alive
    across an eviction at the cost of averaging two sets of weights into one number, and a
    curve point that cannot be attributed to one adapter version is not a measurement. So the
    version is pinned and :func:`assert_still_served` refuses the point if it did not survive.

    In the trainer this hook runs inside the paused-rollout window, and the window advances
    only when training publishes a new version -- which it cannot do while this is running --
    so the pin normally holds. The check is for every other deployment, and for the day that
    stops being true.

    Args:
        cfg: The resolved configuration.

    Returns:
        ``(rows, records, liveness, grader_calls, pinned)``.

    Raises:
        ValueError: If no endpoint was resolved at all.
        AdapterUnresolved: If the endpoint serves no version of the configured adapter.
        AdapterEvicted: If the pinned version was evicted before the evaluation finished.
        LivenessUnavailable: Propagated from the probe.
        ReportSplitTouched, EmptyEvaluation: Propagated from the benchmark run.
    """
    import aiohttp

    if not cfg.base_url:
        raise ValueError(
            "no /v1 endpoint: neither the trainer's rollout engine nor "
            f"{ENV_PREFIX}BASE_URL supplied one, and there is no default to guess."
        )
    rows: dict[str, dict] = {}
    records: dict[str, list] = {}
    grader_calls = 0
    conn = aiohttp.TCPConnector(limit=max(2, min(8, cfg.concurrency)))
    async with aiohttp.ClientSession(connector=conn) as session:
        pinned = await resolve_adapter(session, cfg)
        for bench in cfg.benchmarks:
            row, recs, calls = await run_one_benchmark(cfg, bench, pinned.model)
            rows[bench], records[bench] = row, recs
            grader_calls += calls
        liveness = await measure_liveness(session, cfg, pinned.model)
        await assert_still_served(session, cfg, pinned)
    return rows, records, liveness, grader_calls, pinned


def run_periodic_eval(
    cfg: PeriodicEvalConfig,
    global_step: int,
    tracker: BestValTracker,
    checkpoint: str = "",
    step_seconds: float = 0.0,
    logger=None,
) -> dict[str, float]:
    """Evaluate the current weights and return the metrics for the trainer to commit.

    The single entry point the trainer calls. It never raises: a run of many days must not
    die because an endpoint hiccupped during an evaluation. What it does instead is emit a
    non-zero ``periodic_eval/status_code`` and NaN for every measurement it could not make,
    so a gap in the curve carries its own reason and cannot be misread as a score.

    Args:
        cfg: The resolved configuration.
        global_step: The step being evaluated.
        tracker: Best-validation state, carried across calls.
        checkpoint: Path of the checkpoint holding these weights, recorded with the best
            score so the selected checkpoint can be found afterwards.
        step_seconds: Measured seconds per training step, for the throughput cost.
        logger: Optional logger for the human-readable line.

    Returns:
        The metric mapping, always containing every key in ``cfg.metric_keys()``.
    """
    t0 = time.time()
    rows: dict[str, dict] = {}
    liveness: LivenessReport | None = None
    decision: BestValDecision | None = None
    resolved: ResolvedAdapter | None = None
    status = STATUS["ok"]
    records: dict[str, list] = {}
    grader_calls = 0
    try:
        rows, records, liveness, grader_calls, resolved = asyncio.run(_evaluate_async(cfg))
    except AdapterUnresolved as exc:
        status = STATUS["adapter_unresolved"]
        _log(logger, "error", f"periodic eval could not resolve an adapter at step {global_step}: {exc}")
    except AdapterEvicted as exc:
        status = STATUS["adapter_evicted"]
        _log(logger, "error", f"periodic eval outlived its adapter at step {global_step}: {exc}")
    except SystemExit as exc:
        # `math_bench.verify_model` refuses an unserved id with SystemExit, which is a
        # BaseException: an `except Exception` below would let it past and kill a run of many
        # days. It is reported as an eviction because the id was resolved from this same
        # endpoint seconds earlier, so the served set moved between resolving and using it;
        # the message carries the harness' own text, which names the other possibilities.
        status = STATUS["adapter_evicted"]
        _log(logger, "error", f"periodic eval: the harness refused the pinned adapter at step {global_step}: {exc}")
    except ReportSplitTouched as exc:
        status = STATUS["report_touched"]
        _log(logger, "error", f"periodic eval REFUSED at step {global_step}: {exc}")
    except EmptyEvaluation as exc:
        status = STATUS["empty_evaluation"]
        _log(logger, "error", f"periodic eval graded nothing at step {global_step}: {exc}")
    except LivenessUnavailable as exc:
        status = STATUS["liveness_unavailable"]
        _log(logger, "error", f"periodic eval liveness unavailable at step {global_step}: {exc}")
    except FileNotFoundError as exc:
        status = STATUS["split_error"]
        _log(logger, "error", f"periodic eval split error at step {global_step}: {exc}")
    except Exception as exc:  # endpoint down, transport error, harness abort
        status = STATUS["endpoint_error"]
        _log(logger, "error", f"periodic eval failed at step {global_step}: {type(exc).__name__}: {exc}")

    primary = cfg.benchmarks[0]
    accuracy = float(rows.get(primary, {}).get("accuracy", float("nan")))
    if status == STATUS["ok"]:
        wilson = (rows[primary].get("wilson_lo"), rows[primary].get("wilson_hi"))
        decision = tracker.update(
            global_step,
            accuracy,
            checkpoint,
            wilson,
            model="" if resolved is None else resolved.model,
        )
    diagnosis = (
        diagnose(liveness, accuracy, tracker.first_wilson)
        if status == STATUS["ok"]
        else DIAGNOSIS["unknown"]
    )
    seconds = time.time() - t0
    frac = throughput_fraction(seconds, cfg.freq_steps, step_seconds)
    metrics = metrics_from(
        global_step, rows, liveness, decision, seconds, frac, status, diagnosis, resolved
    )
    metrics[EVAL_GRADER_KEY] = float(grader_calls)
    # Every key the configuration declares is emitted every time, NaN included. A key that
    # appears only on the steps where it succeeded produces a W&B series with invisible gaps.
    for key in cfg.metric_keys():
        metrics.setdefault(key, float("nan"))
    if cfg.out_dir:
        _persist(cfg, global_step, rows, records, liveness, decision, metrics, resolved)
    _log(
        logger,
        "info",
        f"periodic eval step={global_step} status={status} diagnosis={diagnosis} "
        f"model={'UNRESOLVED' if resolved is None else resolved.model} "
        f"acc={accuracy:.4f} live={getattr(liveness, 'is_live', 'na')} "
        f"maxdlogp={getattr(liveness, 'max_abs_dlogprob', float('nan')):.5f} "
        f"best={getattr(decision, 'best_score', float('nan'))}@{getattr(decision, 'best_step', -1)} "
        f"{seconds:.1f}s ({frac:.1%} of throughput)",
    )
    return metrics


def _persist(cfg, global_step, rows, records, liveness, decision, metrics, resolved=None) -> None:
    """Write the full artifact for one evaluation point.

    Every number this project reports must be re-auditable without regenerating it, so the
    results row, the generations and the metrics are written whole.

    Args:
        cfg: The resolved configuration.
        global_step: The step evaluated.
        rows: Benchmark results rows.
        records: Generations per benchmark.
        liveness: The probe result, or None.
        decision: The best-validation decision, or None.
        metrics: The emitted metric mapping.
        resolved: The pinned adapter, or None when resolution failed. Written whole, so the
            artifact says which served id produced these generations rather than which id
            was configured months earlier.
    """
    d = Path(cfg.out_dir) / f"step{global_step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "split": EVAL_SPLIT,
                "configured_model": cfg.model,
                "resolved_adapter": None if resolved is None else resolved.__dict__,
                "rows": rows,
                "liveness": None if liveness is None else liveness.__dict__,
                "best_val": None if decision is None else decision.__dict__,
                "metrics": {k: (None if math.isnan(v) else v) for k, v in metrics.items()},
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    for bench, recs in records.items():
        with open(d / f"{bench}_generations.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")


class PeriodicEvalHook:
    """The trainer-side handle: one object, called once per step, that decides everything.

    Kept here rather than in the trainer so the vendor edit is two lines and the whole
    decision -- cadence, rank gating, cost accounting, budget counters -- is testable on CPU
    without booting AReaL.

    **This deliberately does not go through AReaL's own ``Evaluator``.** A0's preflight
    records *"in-training evaluator disabled ... it deadlocks this stack"*, so reusing
    ``_evaluate`` would inherit a known deadlock. The hook instead runs in
    ``_export_and_commit_stats``, which sits inside the same paused-rollout window (after
    ``rollout.pause()``, before ``rollout.resume()``), just after the checkpoint for this
    step has been written, and immediately before the stats for this step are committed --
    so the evaluation scores the checkpoint it names and its metrics ride the existing W&B
    commit at the correct step.

    One consequence worth stating: on a non-single-controller deployment
    ``_export_and_commit_stats`` ends in a ``dist.barrier``, and only the primary rank runs
    the evaluation, so the other ranks wait for it. The default cadence is sized so that wait
    is under a minute; a much larger ``LIMIT`` would need the barrier timeout checked first.
    """

    #: How many recent step durations to keep for the median used as the throughput
    #: denominator. A median, not a mean: checkpoint steps are much slower than plain ones
    #: and would drag a mean upward, making the evaluation look cheaper than it is.
    STEP_WINDOW = 20

    def __init__(self, env=None, logger=None, rollout=None):
        """Resolve configuration once, at trainer start.

        A misconfiguration raises here -- before any GPU work -- rather than an hour into
        the run, following the ``harness_variants`` pattern in ``cli_args.__post_init__``.

        Args:
            env: Environment mapping; defaults to ``os.environ``.
            logger: Optional logger for the human-readable evaluation line.
            rollout: The trainer's inference engine, read for the endpoint address at each
                evaluation. None outside a trainer, where the address must be configured.
        """
        self.logger = logger
        self._rollout = rollout
        self.config = PeriodicEvalConfig.from_env(env, require_base_url=rollout is None)
        self.tracker = (
            BestValTracker(self.config.patience, self.config.state_path)
            if self.config.enabled
            else None
        )
        self._step_times: list[float] = []
        self._last_step_t: float | None = None
        self._last_counts = None

    @property
    def enabled(self) -> bool:
        """Whether any evaluation will ever run.

        Returns:
            True when the feature is switched on.
        """
        return bool(self.config.enabled)

    def _eval_config(self) -> PeriodicEvalConfig:
        """The configuration for one evaluation, with the endpoint filled in.

        PRECEDENCE, written down once because two places to look for an address is how a run
        scores the wrong server: an explicitly set ``SELFEVO_PERIODIC_EVAL_BASE_URL`` is used
        verbatim and the engine is NOT consulted; otherwise the address comes from the
        rollout engine the trainer is itself generating against. There is no third case and
        no localhost default -- A0's server binds the host interface on a launcher-allocated
        port, so a guess is simply wrong, and a wrong endpoint either fails to connect or,
        worse, scores some other run's weights.

        Returns:
            The configuration to evaluate with.

        Raises:
            RuntimeError: From :func:`base_url_from_rollout`, if the engine holds no address.
        """
        if self.config.base_url or self._rollout is None:
            return self.config
        return replace(self.config, base_url=base_url_from_rollout(self._rollout))

    def step_seconds(self) -> float:
        """Median recent training step duration, measured from this hook's own clock.

        Returns:
            Seconds per step, or 0.0 before enough steps have been seen. Zero propagates to
            a NaN throughput fraction rather than a fabricated one.
        """
        if len(self._step_times) < 3:
            return 0.0
        s = sorted(self._step_times)
        return s[len(s) // 2]

    def _observe_step(self) -> None:
        """Record the wall time of the step that just finished.

        The evaluation's own duration is excluded by resetting the clock after an
        evaluation, so the throughput denominator stays the cost of TRAINING.
        """
        now = time.time()
        if self._last_step_t is not None:
            self._step_times.append(now - self._last_step_t)
            del self._step_times[: -self.STEP_WINDOW]
        self._last_step_t = now

    def budget_metrics(self) -> dict[str, float]:
        """Verifier and inference budget counters, cumulative and per step.

        Emitted on EVERY step, because a matched-budget claim is about the whole run rather
        than about the steps an evaluation happened to land on.

        Returns:
            A mapping over :data:`BUDGET_KEYS`. When the counter has never been touched in
            this process the amounts are NaN and ``budget/counter_visible`` is 0: a reader
            must be able to tell "no verifier calls happened" from "this process cannot see
            them", and a silent zero would conflate them.
        """
        from selfevo import feedback_budget

        counts = feedback_budget.snapshot()
        prev = self._last_counts
        self._last_counts = counts
        nan = float("nan")
        if not counts.visible:
            return {
                "budget/verifier_calls_total": nan,
                "budget/verifier_calls_step": nan,
                "budget/verifier_retries_total": nan,
                "budget/verifier_refusals_total": nan,
                "budget/cache_hits_total": nan,
                "budget/cache_enabled": float(counts.cache_enabled),
                "budget/generated_tokens_total": nan,
                "budget/generated_tokens_step": nan,
                "budget/counter_visible": 0.0,
            }
        delta = counts - prev if prev is not None else counts
        return {
            "budget/verifier_calls_total": float(counts.calls),
            "budget/verifier_calls_step": float(delta.calls),
            "budget/verifier_retries_total": float(counts.retries),
            "budget/verifier_refusals_total": float(counts.refusals),
            "budget/cache_hits_total": float(counts.cache_hits),
            "budget/cache_enabled": float(counts.cache_enabled),
            "budget/generated_tokens_total": float(counts.generated_tokens),
            "budget/generated_tokens_step": float(delta.generated_tokens),
            "budget/counter_visible": 1.0,
        }

    def maybe_run(
        self, global_step: int, checkpoint: str = "", is_primary: bool = True
    ) -> dict[str, float]:
        """Everything the trainer needs, in one call per step.

        Args:
            global_step: The trainer's global step.
            checkpoint: Path of the checkpoint just written for this step, recorded with the
                best score so the selected checkpoint can be found afterwards.
            is_primary: Whether this process is the one that logs. Non-primary ranks must not
                evaluate: several ranks each running an hour-long evaluation before a shared
                barrier is a deadlock, which is exactly how AReaL's own evaluator behaves on
                this stack.

        Returns:
            Metrics to merge into the step's stats dict. Empty when this rank does not log.
        """
        if not is_primary:
            return {}
        self._observe_step()
        metrics = self.budget_metrics()
        if not self.config.should_run(global_step):
            return metrics
        try:
            cfg = self._eval_config()
        except Exception as exc:
            # The endpoint could not be read off the engine. Reported here, where the reason
            # is known, and then handed on with an empty base_url so the evaluation fails
            # through the one path that emits a status code and NaN rather than raising into
            # the training loop.
            _log(
                logger=self.logger,
                level="error",
                msg=(
                    f"periodic eval cannot read the rollout endpoint at step {global_step}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            cfg = self.config
        metrics.update(
            run_periodic_eval(
                cfg,
                global_step,
                self.tracker,
                checkpoint=checkpoint,
                step_seconds=self.step_seconds(),
                logger=self.logger,
            )
        )
        # Exclude the evaluation's own wall time from the training step-time window.
        self._last_step_t = time.time()
        return metrics


def _log(logger, level: str, msg: str) -> None:
    """Emit one line through the trainer's logger, or stderr when there is none.

    Args:
        logger: A logger, or None.
        level: ``"info"`` or ``"error"``.
        msg: The message.
    """
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg)
    else:
        print(msg, file=sys.stderr, flush=True)
