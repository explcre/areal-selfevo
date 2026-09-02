"""The adapter-liveness guard must be able to FAIL. These are its two controls.

WHY THIS FILE EXISTS. On 2026-09-02 the liveness probe was pointed at the BASE MODEL with no
adapter, so the two things being compared were the same weights. It answered ``is_live=1``
with ``max_abs_dlogprob`` 0.978, against 1.007 for a real adapter. The guard could not fail,
and the two A0 evaluation points it had already decided -- 1.04127 at step 50 and 1.04126 at
step 100, fifty steps and fifty adapter versions apart -- were that saturation rather than a
measurement of anything.

The cause was that the probe GENERATED greedily from each side and then subtracted the two
logprob streams by position. Greedy decoding on a batching server is not reproducible: the
same weights produced different text on 1 of 6 probe prompts, and after the argmax paths
diverge, position *i* holds DIFFERENT TOKENS on the two sides. Divergence only happens at
near-ties, where one branch sits near log(0.5) and the other near 0, so the subtraction
lands near 1.0 no matter what the adapter is doing. A0's own record shows the metric
following the coin flip and not the weights: ``greedy_differ_frac`` was 0.167 with
``max_abs_dlogprob`` 1.0413 at steps 50 AND 100, then 0.0 with 0.1167 at step 150 -- a NINE
FOLD drop in the "difference" while the adapter had only trained further.

So the tests below are the two controls, and the negative one is the load-bearing half.
`test_the_base_model_compared_with_itself_is_inert` is the test the old implementation could
never have passed, and any future implementation that cannot pass it must not ship.

HOUSE STYLE, following ``test_periodic_eval.py``: a real HTTP endpoint on loopback served by
the standard library, driving the real production ``measure_liveness`` over real aiohttp. The
stub speaks sglang's native ``/generate``, because that is where per-position logprobs for a
FIXED sequence come from and the repaired probe scores rather than generates.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "bench"))

pytest.importorskip("aiohttp")
pe = pytest.importorskip("selfevo.periodic_eval")

import aiohttp  # noqa: E402

BASE = "/models/qwen2.5-32b-instruct"
ADAPTER = "a0_math-v51"

#: The largest pooled mean |dlogprob| this project has MEASURED between one set of weights
#: and ITSELF. From 39 negative controls run against A0's own rollout server on 2026-09-02:
#: the server has a rare second numerical path -- 2 of 24 scorings of one fixed sequence came
#: back on it, differing by 0.605 nats at a single position -- and pooled over the probe set
#: it is worth this much. A verdict threshold at or below this number cannot separate a live
#: adapter from the base model compared with itself; one of those 39 controls did in fact
#: come back LIVE at the shipped default of 1e-4.
MEASURED_SAME_WEIGHTS_FLOOR = 0.00952

#: The smallest pooled mean |dlogprob| measured against a REAL adapter in the same session
#: (A0 around step 150, six repeats). The usable epsilon lies between the two.
MEASURED_ADAPTER_SIGNAL = 0.0446


class _Policy:
    """What the stub scores, mutable per test.

    Attributes:
        vectors: ``route -> list[float]`` of per-position logprobs, where ``route`` is the
            ``lora_path`` sent, or ``""`` for the base model.
        deviant: ``route -> (call_index, list[float])``. On that route's Nth scoring the stub
            answers the deviant vector instead, which is how the server's rare second
            numerical path is reproduced without waiting for it.
        calls: Every ``(route, max_new_tokens)`` served, in order, so a test can prove the
            probe scored rather than generated.
        status: HTTP status to answer with.
        omit_logprobs: Answer without the logprob block, as a server that cannot do it would.
        empty: Answer with the logprob block present but carrying no scored position, which
            is what a one-token sequence returns: the first position has nothing before it.
        abort: Answer with ``finish_reason.type == "abort"``, as a weight sync causes.
    """

    def __init__(self):
        """Start from a healthy endpoint whose base and adapter differ."""
        base = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
        self.vectors = {
            "": list(base),
            ADAPTER: [x - 0.05 for x in base],
        }
        self.deviant = {}
        self.calls = []
        self.status = 200
        self.omit_logprobs = False
        self.empty = False
        self.abort = False


class _Handler(BaseHTTPRequestHandler):
    """Minimal sglang-native endpoint: ``/generate`` with ``return_logprob``."""

    def log_message(self, *a):
        """Silence per-request logging, which floods pytest output."""

    def do_POST(self):
        """Score one fixed sequence under the requested route."""
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        p = self.server.policy
        route = body.get("lora_path", "") or ""
        p.calls.append((route, (body.get("sampling_params") or {}).get("max_new_tokens")))
        if p.status != 200:
            self._send(p.status, {"error": "nope"})
            return
        if p.abort:
            self._send(200, {"meta_info": {"finish_reason": {"type": "abort"}}})
            return
        vec = p.vectors.get(route, p.vectors[""])
        want = p.deviant.get(route)
        if want is not None:
            idx, dev = want
            if sum(1 for r, _ in p.calls if r == route) - 1 == idx:
                vec = dev
        meta = {"finish_reason": {"type": "stop"}}
        if not p.omit_logprobs:
            scored = [] if p.empty else [[v, i + 2, None] for i, v in enumerate(vec)]
            meta["input_token_logprobs"] = [[None, 1, None]] + scored
        self._send(200, {"text": "", "meta_info": meta})

    def _send(self, code, payload):
        """Write one JSON response.

        Args:
            code: HTTP status.
            payload: Body.
        """
        b = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture()
def endpoint():
    """A loopback ``/generate`` endpoint and the policy that drives it.

    Yields:
        ``(base_url, policy)``.
    """
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.policy = _Policy()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", srv.policy
    finally:
        srv.shutdown()
        srv.server_close()


def _config(url, **kw):
    """A configuration pointed at the stub.

    Args:
        url: The ``/v1`` base url.
        **kw: Overrides.

    Returns:
        The configuration.
    """
    kw.setdefault("probe_prompts", ("one probe prompt",))
    return pe.PeriodicEvalConfig(
        enabled=True,
        benchmarks=("olympiadbench",),
        base_url=url,
        model=ADAPTER,
        base_model=BASE,
        model_path=BASE,
        timeout=10,
        **kw,
    )


def _liveness(cfg, model):
    """Run the real probe against the stub.

    Args:
        cfg: The configuration.
        model: The served id to probe.

    Returns:
        The :class:`LivenessReport`.
    """

    async def go():
        """Open a session and take one measurement.

        Returns:
            The :class:`LivenessReport`.
        """
        async with aiohttp.ClientSession() as s:
            return await pe.measure_liveness(s, cfg, model)

    return asyncio.run(go())


# ------------------------------------------------------------------- the two controls ----


def test_the_base_model_compared_with_itself_is_inert(endpoint):
    """THE NEGATIVE CONTROL, and the reason this file exists.

    Both sides are the base model. There is no adapter, no difference, and nothing to
    report. A guard that answers anything but "inert" here cannot fail and its verdicts mean
    nothing -- which is exactly what the shipped guard did on 2026-09-02, answering LIVE at
    0.978.
    """
    url, _ = endpoint
    cfg = _config(url)
    rep = _liveness(cfg, cfg.base_model)
    assert rep.mean_abs_dlogprob == 0.0, "identical weights produced a non-zero difference"
    assert rep.max_abs_dlogprob == 0.0
    assert rep.is_live == 0, "the guard called the BASE MODEL live against itself"


def test_a_real_adapter_is_live(endpoint):
    """THE POSITIVE CONTROL. Without it the negative one is satisfied by always saying inert."""
    url, _ = endpoint
    rep = _liveness(_config(url), ADAPTER)
    assert rep.mean_abs_dlogprob > 0.0
    assert rep.is_live == 1, "a demonstrably different adapter was called inert"


def test_the_two_controls_separate(endpoint):
    """Stated as one assertion, because a guard is only as good as the gap between them."""
    url, _ = endpoint
    cfg = _config(url)
    dead = _liveness(cfg, cfg.base_model)
    live = _liveness(cfg, ADAPTER)
    assert dead.is_live == 0 and live.is_live == 1
    assert live.mean_abs_dlogprob > dead.mean_abs_dlogprob + cfg.live_eps


# --------------------------------------------------------- what made the old one break ----


def test_the_probe_generates_nothing(endpoint):
    """The structural pin. Every request must ask for ZERO new tokens.

    The old probe generated from each side and zipped the two streams, which stops being a
    comparison the moment greedy diverges -- and on a batching server it diverges on the same
    weights. A probe that scores a fixed sequence cannot have that bug; one that generates
    can, however the arithmetic afterwards is written. So this asserts the shape, not the
    number.
    """
    url, policy = endpoint
    _liveness(_config(url), ADAPTER)
    assert policy.calls, "the probe issued no requests at all"
    assert all(n == 0 for _, n in policy.calls), f"the probe generated tokens: {policy.calls}"


def test_identical_weights_stay_inert_when_the_server_is_nondeterministic(endpoint):
    """The exact 2026-09-02 failure, reproduced against a server that flips.

    The stub answers the base route's SECOND scoring with a different vector, which is what
    A0's rollout server does about 8% of the time. The verdict must still be inert: whatever
    the server's nondeterminism is worth, it is worth that to the in-band control too.
    """
    url, policy = endpoint
    cfg = _config(url, live_eps=0.02)
    policy.deviant[""] = (1, [x - 0.6 for x in policy.vectors[""]])
    rep = _liveness(cfg, cfg.base_model)
    assert rep.noise_mean_abs_dlogprob > 0.0, "premise: the control must have SEEN the flip"
    assert rep.is_live == 0, "server nondeterminism was reported as a live adapter"


def test_the_in_band_control_is_actually_measured(endpoint):
    """The noise series must come from the endpoint, not from a zero nobody measured."""
    url, policy = endpoint
    cfg = _config(url, live_eps=0.02)
    policy.deviant[""] = (1, [x - 0.6 for x in policy.vectors[""]])
    rep = _liveness(cfg, ADAPTER)
    assert rep.noise_max_abs_dlogprob == pytest.approx(0.6, abs=1e-6)


def test_a_verdict_carried_by_one_position_is_not_a_verdict(endpoint):
    """The statistic must be the MEAN. A single near-tie position can carry a maximum.

    One position moved by 0.6 in eight is worth 0.075 on the mean; the same 0.6 as a maximum
    is indistinguishable from the divergence artefact that broke the old probe.
    """
    url, policy = endpoint
    cfg = _config(url, live_eps=0.2)
    spike = list(policy.vectors[""])
    spike[0] -= 0.6
    policy.vectors[ADAPTER] = spike
    rep = _liveness(cfg, ADAPTER)
    assert rep.max_abs_dlogprob == pytest.approx(0.6, abs=1e-6)
    assert rep.is_live == 0, "a single position carried the verdict"


# ----------------------------------------------------------------------- the refusals ----


def test_a_dead_endpoint_has_no_verdict(endpoint):
    """"Assume live" and "assume inert" are each an answer nobody measured."""
    url, policy = endpoint
    policy.status = 500
    with pytest.raises(pe.LivenessUnavailable):
        _liveness(_config(url), ADAPTER)


def test_a_server_without_logprobs_has_no_verdict(endpoint):
    """The verdict is decided on logprobs; without them there is nothing to decide it with."""
    url, policy = endpoint
    policy.omit_logprobs = True
    with pytest.raises(pe.LivenessUnavailable):
        _liveness(_config(url), ADAPTER)


def test_an_aborted_scoring_has_no_verdict(endpoint):
    """AReaL's weight sync drops requests in flight. A dropped request is not a measurement."""
    url, policy = endpoint
    policy.abort = True
    with pytest.raises(pe.LivenessUnavailable):
        _liveness(_config(url), ADAPTER)


def test_the_probe_reports_how_much_evidence_it_had(endpoint):
    """A verdict resting on almost no evidence must be visible as such, not implied."""
    url, _ = endpoint
    rep = _liveness(_config(url, probe_prompts=("a", "b", "c")), ADAPTER)
    assert rep.n_probes == 3
    assert rep.n_tokens_compared == 24


def test_a_signal_smaller_than_the_servers_own_noise_is_not_a_verdict(endpoint):
    """The in-band control has to be able to VETO, not merely be reported beside the signal.

    The adapter really does differ here, by 0.05. The server also flips one of its two base
    scorings by 0.6. A difference smaller than what the same weights produced twice in a row
    is not evidence about the adapter, and a guard that reports it as one is quoting the
    server's nondeterminism as a finding. This is the assertion a verdict that ignores the
    control passes and this one does not.
    """
    url, policy = endpoint
    cfg = _config(url, live_eps=0.02)
    policy.deviant[""] = (1, [x - 0.6 for x in policy.vectors[""]])
    rep = _liveness(cfg, ADAPTER)
    assert rep.mean_abs_dlogprob > cfg.live_eps, "premise: the raw signal must clear epsilon"
    assert rep.noise_mean_abs_dlogprob > rep.mean_abs_dlogprob, "premise: noise must exceed it"
    assert rep.is_live == 0, "the server's own nondeterminism was reported as a live adapter"


def test_a_probe_that_compared_nothing_refuses(endpoint):
    """Nothing to compare is not a verdict of inert, and not a verdict of live either.

    A maximum over an empty set is whatever default the code happens to carry, and both
    available defaults are answers nobody measured. The probe must raise.
    """
    url, policy = endpoint
    policy.empty = True
    with pytest.raises(pe.LivenessUnavailable):
        _liveness(_config(url), ADAPTER)


# ------------------------------------------------------------------------ calibration ----


def test_the_default_epsilon_clears_the_measured_same_weights_noise_floor():
    """The threshold has to sit ABOVE what identical weights can produce on this server.

    Not a style point. At the shipped default of 1e-4, one of 39 negative controls run
    against A0's live rollout server came back LIVE with a pooled mean of 0.00952 -- the base
    model, called live, by the repaired guard. The usable window is bounded below by
    :data:`MEASURED_SAME_WEIGHTS_FLOOR` and above by :data:`MEASURED_ADAPTER_SIGNAL`; 0.02
    sits near its geometric middle.
    """
    assert pe.DEFAULT_LIVE_EPS > MEASURED_SAME_WEIGHTS_FLOOR, (
        f"DEFAULT_LIVE_EPS={pe.DEFAULT_LIVE_EPS} is at or below the measured same-weights "
        f"floor {MEASURED_SAME_WEIGHTS_FLOOR}, so the negative control can still fail"
    )
    assert pe.DEFAULT_LIVE_EPS < MEASURED_ADAPTER_SIGNAL, (
        f"DEFAULT_LIVE_EPS={pe.DEFAULT_LIVE_EPS} is at or above the weakest real adapter "
        f"signal {MEASURED_ADAPTER_SIGNAL}, so a live adapter would be called inert"
    )
