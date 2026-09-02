"""The evaluation must be able to talk to a TRAINING rollout server, and must never be
mistaken for the headline evaluation when it does.

Two failures this file exists to catch, and they pull in opposite directions.

1. **It cannot talk to the server at all.** A0's rollout server runs sglang with
   `--skip-tokenizer-init`, which is right for the trainer because the trainer speaks token
   ids. The consequence is that the server holds NO tokenizer, so every text request fails for
   every model id it serves -- chat completions answer HTTP 500 with a null-tokenizer error
   and text completions answer HTTP 400 saying the engine cannot accept text prompts. The
   evaluation was loud about it (a non-numeric accuracy and a status code, never a silent
   zero) and produced no curve at all, which is why the run had no validation signal. The fix
   is for the evaluation to tokenise locally and speak token ids, and the tests below drive
   the REAL production path against a fake server that behaves exactly as the real one was
   measured to behave.

2. **It talks to the server and the number is then believed.** This is the worse failure. The
   training server is configured for training: it caps a request at `--context-length 4096`
   against a headline evaluation budget of 16384, and the run generates short completions by
   design. A benchmark score is not interpretable without the budget that produced it --
   measured on 2026-09-02, the same model on the same problems through the same grader moved
   FIFTY POINTS when the cap changed by under four times. So every point carries the budget
   that produced it, the comparability flag, and a series name (`trend_score`) that is not an
   accuracy and cannot be lined up against one.

THE ADDITIVE GUARANTEE is the third thing here and it is proved rather than asserted. The
same `math_bench.run_bench` produced every headline number in the paper against an ordinary
tokenising server. `test_the_standalone_row_is_byte_identical_to_the_pre_change_baseline`
compares the whole results row and every generation against a golden recorded from the
PRE-CHANGE code on a fixed input, so "unchanged" is a measurement and not a claim.

HOUSE STYLE, following `test_periodic_eval.py`. Every test drives the real production entry
point through a real HTTP endpoint on loopback, so `run_bench` runs unmodified: real aiohttp,
real `verify_model`, real grading, real split filtering. The tokenizer is real too -- a
byte-level one built in `tmp_path`, which round-trips arbitrary text exactly and carries a
real chat template, so the encode and the decode under test are the library's and not a stub's.
No GPU, no network beyond loopback, no second scorer anywhere in the file.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "experiments" / "bench"
sys.path.insert(0, str(BENCH))

# math_bench resolves its data root at import time, so this must precede the import.
if not os.environ.get("MATH_EVAL_DATA"):
    _guess = Path(os.path.expanduser("~/evaldata"))
    if (_guess / "olympiadbench" / "test.jsonl").exists():
        os.environ["MATH_EVAL_DATA"] = str(_guess)

mb = pytest.importorskip("math_bench")
pe = pytest.importorskip("selfevo.periodic_eval")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

ADAPTER = "a0_math-v7"
BASE = "/models/qwen2.5-32b-instruct"

#: What the real server was measured to cap at, and what the headline evaluation ran at.
SERVER_CONTEXT = 4096
HEADLINE_CAP = mb.BENCH_OVERRIDES["olympiadbench"]["max_tokens"]

_HAS_DATA = (mb.DATA / "olympiadbench" / "test.jsonl").exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="olympiadbench data not on this box")

#: The golden row recorded from the PRE-CHANGE code. See `make_standalone_golden.py`.
GOLDEN = Path(__file__).resolve().parent / "baselines" / "standalone_row_pre_token_id.json"


# ------------------------------------------------------------------- the tokenizer ----
#
# A real tokenizer, not a stub, because two of the mutations this file must kill live inside
# the encode and the decode. Byte-level with an empty merge table: every one of the 256 bytes
# is a token, so any string round-trips exactly, and a chat template is supplied so
# `apply_chat_template` is the library's own code path rather than string concatenation.


def build_tokenizer_dir(d: Path) -> Path:
    """Write a small but real HuggingFace tokenizer into a directory.

    Args:
        d: Directory to write into.

    Returns:
        The same directory, now loadable by ``AutoTokenizer.from_pretrained``.
    """
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel

    d.mkdir(parents=True, exist_ok=True)
    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": (
                    '{% for m in messages %}<|im_start|>{{ m["role"] }}\n'
                    '{{ m["content"] }}<|im_end|>\n{% endfor %}'
                    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
                ),
                "clean_up_tokenization_spaces": False,
            }
        )
    )
    return d


@pytest.fixture
def tokenizer_dir(tmp_path):
    """A directory holding a real, tiny tokenizer.

    Yields:
        The path.
    """
    return build_tokenizer_dir(tmp_path / "tok")


@pytest.fixture
def tio(tokenizer_dir):
    """The production tokenizer wrapper, loaded from :func:`tokenizer_dir`.

    Yields:
        A ``math_bench.TokenIO``.
    """
    return mb.TokenIO.from_model_path(str(tokenizer_dir))


# ---------------------------------------------------------------- the stub endpoint ----


class _Policy:
    """What the stub endpoint answers, mutated per test.

    Attributes:
        has_tokenizer: What ``/get_server_info`` reports for ``skip_tokenizer_init``.
        publish_info: When False, ``/get_server_info`` 404s, which is how a server that does
            not publish its launch flags behaves.
        context_limit: The total the stub will accept, enforced on ``/generate`` with the real
            server's own refusal message.
        tokenizer_path: What the server says it tokenises with.
        base_model: What the server calls its base model.
        models: Ids reported by ``/v1/models``.
        output_ids_for: ``(lora, input_ids) -> list[int] | None``; None omits ``output_ids``
            from the reply, which is how a broken reply looks.
        logprobs_for: ``lora -> list[float] | None``.
        finish_type: The ``finish_reason.type`` reported by ``/generate``.
        text_for: ``(model, prompt) -> str`` for the chat endpoint.
        chat_finish_reason: The chat endpoint's finish reason.
        n_chat, n_generate, n_info, n_models: Requests served, per endpoint.
        generate_loras: The ``lora_path`` of every ``/generate`` request, in order.
        generate_caps: The ``max_new_tokens`` of every ``/generate`` request, in order.
        generate_inputs: The ``input_ids`` of every ``/generate`` request, in order, so a
            test can decode them and check WHAT was asked rather than only that something was.
    """

    def __init__(self):
        """Start from a healthy TOKENIZER-LESS server, which is the case under test."""
        self.has_tokenizer = False
        self.publish_info = True
        self.context_limit = SERVER_CONTEXT
        self.tokenizer_path = ""
        self.base_model = BASE
        self.models = [BASE, ADAPTER]
        self.output_ids_for = None
        self.logprobs_for = lambda lora: [-0.10, -0.20, -0.30]
        self.finish_type = "stop"
        self.text_for = lambda model, prompt: "Reasoning. The answer is \\boxed{42}"
        self.chat_finish_reason = "stop"
        self.n_chat = 0
        self.n_generate = 0
        self.n_info = 0
        self.n_models = 0
        self.generate_loras = []
        self.generate_caps = []
        self.generate_inputs = []


class _Handler(BaseHTTPRequestHandler):
    """A fake sglang: ``/v1/models``, ``/get_server_info``, ``/v1/chat/completions``, ``/generate``."""

    def log_message(self, *a):
        """Silence the default per-request logging, which floods pytest output."""

    def _send(self, code: int, payload: dict) -> None:
        """Write one JSON response.

        Args:
            code: HTTP status.
            payload: Body.
        """
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Answer the model list and the server-info block."""
        p = self.server.policy
        path = self.path.rstrip("/")
        if path.endswith("/models"):
            p.n_models += 1
            self._send(200, {"object": "list", "data": [{"id": m} for m in p.models]})
        elif path.endswith("/get_server_info"):
            p.n_info += 1
            if not p.publish_info:
                self._send(404, {"error": "not found"})
                return
            self._send(
                200,
                {
                    "skip_tokenizer_init": not p.has_tokenizer,
                    "context_length": p.context_limit,
                    "tokenizer_path": p.tokenizer_path,
                    "model_path": p.base_model,
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        """Answer a chat completion or a native token-id generation."""
        p = self.server.policy
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path.rstrip("/").endswith("/generate"):
            self._generate(p, req)
        else:
            self._chat(p, req)

    def _chat(self, p, req):
        """The OpenAI chat endpoint, which a tokenizer-less server cannot serve.

        Args:
            p: The policy.
            req: The parsed request body.
        """
        p.n_chat += 1
        if not p.has_tokenizer:
            # Verbatim from A0's live server on 2026-09-02.
            self._send(
                500,
                {
                    "object": "error",
                    "message": "Internal server error: 'NoneType' object has no attribute "
                    "'apply_chat_template'",
                    "code": 500,
                },
            )
            return
        model = req.get("model", "")
        prompt = (req.get("messages") or [{}])[0].get("content", "")
        choice = {
            "message": {"content": p.text_for(model, prompt)},
            "finish_reason": p.chat_finish_reason,
        }
        lps = p.logprobs_for(model)
        if lps is not None:
            choice["logprobs"] = {"content": [{"token": "t", "logprob": x} for x in lps]}
        self._send(200, {"choices": [choice]})

    def _generate(self, p, req):
        """The native token-id endpoint.

        Args:
            p: The policy.
            req: The parsed request body.
        """
        p.n_generate += 1
        ids = req.get("input_ids") or []
        sp = req.get("sampling_params") or {}
        cap = int(sp.get("max_new_tokens") or 0)
        p.generate_loras.append(req.get("lora_path", ""))
        p.generate_caps.append(cap)
        p.generate_inputs.append(list(ids))
        if p.context_limit and len(ids) + cap > p.context_limit:
            # Verbatim shape from A0's live server.
            self._send(
                400,
                {
                    "object": "error",
                    "message": (
                        f"Requested token count exceeds the model's maximum context length of "
                        f"{p.context_limit} tokens. You requested a total of {len(ids) + cap} "
                        f"tokens: {len(ids)} tokens from the input messages and {cap} tokens "
                        f"for the completion."
                    ),
                    "code": 400,
                },
            )
            return
        out = p.output_ids_for(req.get("lora_path", ""), ids) if p.output_ids_for else []
        body = {"meta_info": {"finish_reason": {"type": p.finish_type}}}
        if out is not None:
            body["output_ids"] = list(out)
        if req.get("return_logprob"):
            lps = p.logprobs_for(req.get("lora_path", ""))
            if lps is not None:
                body["meta_info"]["output_token_logprobs"] = [[x, 1, None] for x in lps]
        self._send(200, body)


@pytest.fixture
def endpoint():
    """A running fake sglang.

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


def _run(coro):
    """Drive one coroutine to completion.

    Args:
        coro: The coroutine.

    Returns:
        Its result.
    """
    return asyncio.run(coro)


def _args(url, tokenizer_dir=None, **kw):
    """A namespace shaped exactly as ``run_bench`` reads it.

    Args:
        url: The stub's ``/v1`` url.
        tokenizer_dir: Where the local tokenizer lives, or None.
        **kw: Field overrides.

    Returns:
        An ``argparse.Namespace``.
    """
    import argparse

    d = dict(
        base_url=url,
        model=ADAPTER,
        model_path=str(tokenizer_dir) if tokenizer_dir else "",
        benchmarks="olympiadbench",
        limit=4,
        split="search",
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        concurrency=4,
        timeout=30,
        seed=0,
        out="",
        gen_out="",
    )
    d.update(kw)
    return argparse.Namespace(**d)


def _answer_ids(tio, text):
    """Token ids whose decoding is exactly ``text``.

    Args:
        tio: The tokenizer wrapper.
        text: The completion the server should appear to have produced.

    Returns:
        A list of token ids.
    """
    return list(tio.tok(text, add_special_tokens=False)["input_ids"])


def _config(url, tokenizer_dir, **kw):
    """A working periodic-eval configuration pointed at the stub.

    Args:
        url: The stub's ``/v1`` url.
        tokenizer_dir: Where the local tokenizer lives.
        **kw: Field overrides.

    Returns:
        The configuration.
    """
    d = dict(
        enabled=True,
        freq_steps=5,
        benchmarks=("olympiadbench",),
        limit=4,
        max_tokens=64,
        concurrency=4,
        timeout=30,
        base_url=url,
        model=ADAPTER,
        base_model=BASE,
        model_path=str(tokenizer_dir),
        probe_prompts=("probe one", "probe two"),
        probe_max_tokens=4,
        patience=3,
        explicit_gen_keys=frozenset({"max_tokens"}),
    )
    d.update(kw)
    return pe.PeriodicEvalConfig(**d)


# ------------------------------------------------------------------------- premise ----


def test_premise_the_fake_tokenizer_round_trips_arbitrary_text_exactly(tio):
    """If the tokenizer loses bytes, every decode test below tests the tokenizer, not the code."""
    for s in ("Reasoning. The answer is \\boxed{42}", "\\boxed{\\frac{1}{2}}", "éè <tag> 3.14"):
        assert tio.decode(_answer_ids(tio, s)) == s


def test_premise_the_stub_refuses_text_exactly_as_the_real_server_does(endpoint, tio):
    """The whole blocker, reproduced. If this passes trivially the token-id tests prove nothing.

    A tokenizer-less server answers a chat completion with HTTP 500 and a null-tokenizer
    message, which is what was measured against A0 on 2026-09-02.
    """
    url, policy = endpoint
    params = dict(temperature=0.0, top_p=1.0, max_tokens=8, seed=0, timeout=5)
    r = _run(_chat_once(url, params))
    assert r["status"] == "failed", "the stub answered a text request it should have refused"
    assert policy.n_chat == 3, "the harness did not really try, so nothing was refused"


async def _chat_once(url, params):
    """One completion through the unchanged text path.

    Args:
        url: The stub's ``/v1`` url.
        params: Generation parameters.

    Returns:
        The generation record.
    """
    import aiohttp

    async with aiohttp.ClientSession() as s:
        return await mb.generate(s, mb.chat_url(url), ADAPTER, "hello", params)


# ------------------------------------------------- a server that HAS a tokenizer ----


def test_a_tokenising_server_is_reported_as_having_a_tokenizer(endpoint):
    """The detection is read off the server, never guessed at."""
    url, policy = endpoint
    policy.has_tokenizer = True
    caps = _run(_caps(url))
    assert caps.has_tokenizer is True
    assert "skip_tokenizer_init=False" in caps.source


def test_an_endpoint_that_publishes_nothing_is_treated_as_tokenising(endpoint):
    """An unknown server must behave exactly as it did before this code existed."""
    url, policy = endpoint
    policy.publish_info = False
    caps = _run(_caps(url))
    assert caps.has_tokenizer is True
    assert caps.context_limit is None


async def _caps(url):
    """Read the stub's capabilities through the production function.

    Args:
        url: The stub's ``/v1`` url.

    Returns:
        The capabilities.
    """
    import aiohttp

    async with aiohttp.ClientSession() as s:
        return await mb.server_capabilities(s, url)


@needs_data
def test_a_tokenising_server_never_touches_the_token_id_endpoint(endpoint, tokenizer_dir):
    """THE ADDITIVE GUARANTEE, in its most direct form.

    A server with a tokenizer must be generated against exactly as before: through
    ``/v1/chat/completions`` and never through ``/generate``, even when a local tokenizer is
    sitting right there and would work.
    """
    url, policy = endpoint
    policy.has_tokenizer = True
    policy.publish_info = True
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir), None, frozenset({"max_tokens"})))
    assert policy.n_chat == 4, "the chat endpoint was not the one used"
    assert policy.n_generate == 0, "the token-id path engaged against a TOKENISING server"
    assert row["n_graded"] == 4


@needs_data
def test_a_tokenising_server_leaves_no_token_id_provenance_in_the_row(endpoint, tokenizer_dir):
    """No new keys on the old path, which is what makes the golden comparison possible."""
    url, policy = endpoint
    policy.has_tokenizer = True
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir), None, frozenset({"max_tokens"})))
    for key in ("token_id_path", "tokenizer_path", "lora_path", "server_capabilities"):
        assert key not in row["params"], f"{key} appeared on the standalone path"


@needs_data
@pytest.mark.parametrize("publishes_info", [True, False])
def test_the_standalone_row_is_byte_identical_to_the_pre_change_baseline(
    endpoint, tokenizer_dir, publishes_info
):
    """The proof that the headline path is unchanged, on a fixed input.

    The golden was recorded by running this same fixture against the PRE-CHANGE
    ``math_bench.py`` (see ``make_standalone_golden.py``). Both shapes of tokenising server
    are checked: one that publishes its launch flags and one that 404s, because the new code
    asks a question the old code never asked and the answer must not reach the row either way.
    """
    if not GOLDEN.exists():
        pytest.skip(f"no recorded pre-change baseline at {GOLDEN}")
    url, policy = endpoint
    policy.has_tokenizer = True
    policy.publish_info = publishes_info
    policy.context_limit = 32768
    import io

    buf = io.StringIO()
    row = _run(
        mb.run_bench("olympiadbench", _args(url, tokenizer_dir), buf, frozenset({"max_tokens"}))
    )
    got = {
        "row": _normalise(row),
        "generations": [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()],
    }
    want = json.loads(GOLDEN.read_text())
    assert got == want, "the standalone path's output moved"


def _normalise(row):
    """Strip the parts of a results row that change between runs of the same input.

    Args:
        row: A ``run_bench`` results row.

    Returns:
        A copy with the loopback port replaced, so two runs of the same input compare equal.
    """
    out = copy.deepcopy(row)
    p = out.get("params") or {}
    if isinstance(p.get("endpoint"), str):
        p["endpoint"] = re.sub(r":\d+/", ":PORT/", p["endpoint"])
    return out


# ------------------------------------------ a server WITHOUT a tokenizer: the fix ----


def test_a_tokenizerless_server_is_detected_from_what_it_says(endpoint):
    """The flag the SERVER was launched with, which no model config can see."""
    url, _policy = endpoint
    caps = _run(_caps(url))
    assert caps.has_tokenizer is False
    assert caps.context_limit == SERVER_CONTEXT
    assert caps.base_model == BASE


@needs_data
def test_the_token_id_path_scores_the_search_half_through_the_unchanged_grader(
    endpoint, tokenizer_dir, tio
):
    """The fix, end to end: no tokenizer on the server, a full graded row out of the harness."""
    url, policy = endpoint
    right = _answer_ids(tio, "Reasoning. The answer is \\boxed{42}")
    policy.output_ids_for = lambda lora, ids: right
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir), None, frozenset({"max_tokens"})))
    assert policy.n_generate == 4, "the token-id endpoint was not the one used"
    assert policy.n_chat == 0, "a text request was made to a server that cannot serve one"
    assert row["n_problems"] == 4 and row["n_graded"] == 4
    assert row["n_failed"] == 0
    assert row["params"]["token_id_path"] is True


@needs_data
def test_the_reply_decodes_to_text_the_grader_then_scores(endpoint, tokenizer_dir, tio):
    """Token ids in, a graded answer out -- and the grading is the harness's own.

    The gold answer of the first search-half problem is read off disk and handed back as token
    ids, so a correct decode is the only way this scores 1.0. A decoder that loses the final
    token breaks the closing brace of ``\\boxed{...}`` and the score collapses.
    """
    url, policy = endpoint
    probs = mb.load("olympiadbench", "search")[:4]
    golds = {i: p["answer"] for i, p in enumerate(probs)}
    policy.output_ids_for = lambda lora, ids: _answer_ids(
        tio, "Reasoning. The answer is \\boxed{" + golds[0] + "}"
    )
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=1), None, frozenset({"max_tokens"})))
    assert row["n_graded"] == 1
    assert row["accuracy"] == 1.0, "the decoded text did not reach the grader intact"
    assert row["n_no_box"] == 0, "the decoded text carried no balanced \\boxed{}"


@needs_data
def test_a_wrong_decoded_answer_scores_zero_and_is_still_graded(endpoint, tokenizer_dir, tio):
    """The grader must be able to say NO through this path too, or the test above is vacuous."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(
        tio, "Reasoning. The answer is \\boxed{-999999}"
    )
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=2), None, frozenset({"max_tokens"})))
    assert row["n_graded"] == 2 and row["accuracy"] == 0.0


@needs_data
def test_the_adapter_is_routed_by_lora_path_and_the_base_model_is_not(endpoint, tokenizer_dir, tio):
    """`/generate` has no model field: an adapter is reached by `lora_path` or not at all.

    And the BASE model must NOT be sent one -- sglang refuses a `lora_path` it never loaded,
    so routing the base as an adapter would fail every liveness probe.
    """
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=1), None, frozenset({"max_tokens"})))
    assert policy.generate_loras == [ADAPTER]
    policy.generate_loras.clear()
    _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=1, model=BASE), None, frozenset({"max_tokens"})))
    assert policy.generate_loras == [""], "the base model was routed as a LoRA adapter"


@needs_data
def test_the_ids_sent_are_the_prompt_a_tokenising_server_would_have_built(
    endpoint, tokenizer_dir, tio
):
    """The prompt has to be the SAME prompt, or the score is of a question nobody asked.

    On a tokenising server the chat template is applied server side and the harness never sees
    it. Here it is applied on this side, so what is actually on the wire is decoded back and
    compared against the template the tokenizer itself produces from the harness's own PROMPT.
    A path that skipped the template, or applied it to the wrong text, would still generate,
    still decode and still score -- just against a different question.
    """
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    probs = mb.load("olympiadbench", "search")[:1]
    _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=1), None, frozenset({"max_tokens"})))
    assert len(policy.generate_inputs) == 1
    sent = tio.decode(policy.generate_inputs[0])
    want = tio.tok.apply_chat_template(
        [{"role": "user", "content": mb.PROMPT.format(problem=probs[0]["problem"])}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert sent == want
    assert probs[0]["problem"] in sent, "the problem itself did not reach the server"
    assert sent.startswith("<|im_start|>"), "the chat template was not applied"


@needs_data
def test_the_recorded_budget_is_the_one_that_ran_not_the_one_that_was_asked_for(
    endpoint, tokenizer_dir, tio
):
    """A point that named a budget which never ran would be worse than one with no budget.

    The configuration asks for the headline cap; this server will not accept it, so the
    harness clamps. The number recorded beside the score must be the clamped one.
    """
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    cfg = _config(url, tokenizer_dir, max_tokens=HEADLINE_CAP, limit=2)
    m = pe.run_periodic_eval(cfg, 50, pe.BestValTracker(cfg.patience))
    assert m["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert m["periodic_eval/olympiadbench/max_tokens"] == SERVER_CONTEXT - mb.PROMPT_HEADROOM
    assert m["periodic_eval/olympiadbench/max_tokens"] != HEADLINE_CAP
    assert m["periodic_eval/olympiadbench/budget_matches_headline"] == 0.0


def test_the_benchmark_and_the_liveness_probe_route_an_id_the_same_way(endpoint):
    """One definition of "adapter or base", used by both, so the two cannot drift apart.

    They must agree: sglang refuses a `lora_path` it never loaded, so a benchmark that routes
    the adapter and a probe that routes the base would compare weights that never answered.
    """
    url, _policy = endpoint
    caps = _run(_caps(url))
    assert mb.adapter_route(ADAPTER, caps) == ADAPTER
    assert mb.adapter_route(BASE, caps) == ""
    assert mb.adapter_route("", caps) == ""


def test_a_tokenizer_that_is_not_the_servers_is_refused(endpoint, tokenizer_dir, tmp_path):
    """A different tokenizer builds a different prompt, and the score would look normal."""
    other = build_tokenizer_dir(tmp_path / "other")
    with pytest.raises(ValueError) as exc:
        mb.TokenIO.from_model_path(str(other), str(tokenizer_dir))
    assert "tokenise" in str(exc.value)


def test_a_matching_tokenizer_is_accepted(tokenizer_dir):
    """A guard that refused everything would guard nothing."""
    t = mb.TokenIO.from_model_path(str(tokenizer_dir), str(tokenizer_dir))
    assert t.decode(_answer_ids(t, "ok")) == "ok"


def test_no_local_tokenizer_is_refused_with_a_message_naming_the_variable():
    """The operator must be told what to set, not left with a transport error."""
    with pytest.raises(ValueError) as exc:
        mb.TokenIO.from_model_path("")
    assert "SELFEVO_PERIODIC_EVAL_MODEL_PATH" in str(exc.value)


# ------------------------------------------------------------------ silent zeros ----


@needs_data
def test_a_reply_with_no_token_ids_is_a_failed_request_not_an_empty_wrong_answer(
    endpoint, tokenizer_dir
):
    """An empty completion is a WRONG ANSWER and counts in the denominator; this is not one."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: None
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=2), None, frozenset({"max_tokens"})))
    assert row["n_failed"] == 2
    assert row["n_graded"] == 0, "a harness fault was charged to the model as a wrong answer"


@needs_data
def test_a_decode_failure_is_a_failed_request_not_an_empty_string_scored_wrong(
    endpoint, tokenizer_dir, tio, monkeypatch
):
    """THE MUTATION THIS EXISTS FOR: a decode that returns "" is graded as a wrong answer.

    The decoder is made to raise, which is what a corrupt id stream does, and the result must
    be a FAILED request rather than a zero attributed to the model.
    """
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")

    def boom(self, ids):
        """Fail every decode.

        Args:
            self: The TokenIO.
            ids: The ids that would have been decoded.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("corrupt id stream")

    monkeypatch.setattr(mb.TokenIO, "decode", boom)
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=2), None, frozenset({"max_tokens"})))
    assert row["n_failed"] == 2
    assert row["n_graded"] == 0, "a decode failure was scored as the model getting it wrong"


@needs_data
def test_an_aborted_generation_is_not_graded_as_a_wrong_answer(endpoint, tokenizer_dir, tio):
    """Measured on A0: six of eight generations came back aborted and every one scored zero.

    `pause_generation`, which AReaL sends around every weight update, drops the requests in
    flight. Graded, that is a curve of zeros that reads as a model getting everything wrong
    when what happened is that the run interrupted its own evaluation.
    """
    url, policy = endpoint
    policy.finish_type = "abort"
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "Reasoning cut off here")
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=4), None, frozenset({"max_tokens"})))
    assert row["n_failed"] == 4
    assert row["n_graded"] == 0
    assert math.isnan(row["accuracy"]), "an aborted run reported a score"


@needs_data
def test_an_evaluation_of_aborted_generations_is_refused_by_the_periodic_guard(
    endpoint, tokenizer_dir, tio
):
    """And the refusal reaches the caller as EmptyEvaluation, not as a low number."""
    url, policy = endpoint
    policy.finish_type = "abort"
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "cut off")
    cfg = _config(url, tokenizer_dir)
    with pytest.raises(pe.EmptyEvaluation):
        _run(pe.run_one_benchmark(cfg, "olympiadbench", ADAPTER))


# ------------------------------------------------------------- the budget the server takes ----


def test_the_model_config_guard_cannot_see_the_servers_limit(tmp_path):
    """WHY THE EXISTING GUARD PASSES WHILE THE REQUEST FAILS.

    `model_context_limit` reads the model's own config.json. A0's Qwen2.5-32B declares 32768
    there while its rollout server was launched with `--context-length 4096`. This is the
    measurement that says the two disagree, so the fix cannot be "use the existing clamp".
    """
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert mb.model_context_limit(str(tmp_path)) == 32768
    assert mb.clamp_max_tokens(HEADLINE_CAP, 32768) == (HEADLINE_CAP, None)
    eff, why = mb.clamp_max_tokens(HEADLINE_CAP, SERVER_CONTEXT)
    assert eff < HEADLINE_CAP and why


@needs_data
def test_an_over_budget_cap_is_rejected_by_the_server_when_it_is_not_clamped(
    endpoint, tokenizer_dir, tio
):
    """The premise for the clamp test: the stub really refuses an over-budget request."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    params = dict(temperature=0.0, top_p=1.0, max_tokens=HEADLINE_CAP, seed=0, timeout=5)
    r = _run(_generate_once(url, tio, params))
    assert r["status"] == "failed"
    assert policy.n_generate == 3, "the request was not actually sent three times"


async def _generate_once(url, tio, params, lora=ADAPTER):
    """One completion through the token-id path, with no clamp in the way.

    Args:
        url: The stub's ``/v1`` url.
        tio: The tokenizer wrapper.
        params: Generation parameters.
        lora: The adapter to route to.

    Returns:
        The generation record.
    """
    import aiohttp

    async with aiohttp.ClientSession() as s:
        return await mb.generate_ids(s, url, tio.encode_chat("hello"), params, tio, lora_path=lora)


@needs_data
def test_the_cap_is_clamped_to_what_the_server_published_and_the_row_says_so(
    endpoint, tokenizer_dir, tio
):
    """The guard fix. The headline cap is refused by this server, so it is reduced, loudly."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    row = _run(
        mb.run_bench(
            "olympiadbench",
            _args(url, tokenizer_dir, limit=2, max_tokens=HEADLINE_CAP),
            None,
            frozenset({"max_tokens"}),
        )
    )
    assert row["n_graded"] == 2, "the clamp did not make the requests acceptable"
    assert row["params"]["max_tokens"] == SERVER_CONTEXT - mb.PROMPT_HEADROOM
    assert row["params"]["max_tokens_requested"] == HEADLINE_CAP
    assert row["params"]["server_context_limit"] == SERVER_CONTEXT
    assert all(c == SERVER_CONTEXT - mb.PROMPT_HEADROOM for c in policy.generate_caps)


@needs_data
def test_a_cap_the_server_accepts_is_left_alone(endpoint, tokenizer_dir, tio):
    """A clamp that always fired would silently change every budget it touched."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    row = _run(mb.run_bench("olympiadbench", _args(url, tokenizer_dir, limit=1, max_tokens=512), None, frozenset({"max_tokens"})))
    assert row["params"]["max_tokens"] == 512
    assert "max_tokens_requested" not in row["params"]


# ---------------------------------------------------- the budget is recorded, always ----


@needs_data
def test_every_emitted_point_carries_the_budget_that_produced_it(endpoint, tokenizer_dir, tio):
    """The headline requirement: a number and its budget travel together or not at all."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    cfg = _config(url, tokenizer_dir, max_tokens=256)
    tracker = pe.BestValTracker(cfg.patience)
    m = pe.run_periodic_eval(cfg, 50, tracker)
    assert m["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert not math.isnan(m["periodic_eval/olympiadbench/trend_score"])
    assert m["periodic_eval/olympiadbench/max_tokens"] == 256
    assert m["periodic_eval/olympiadbench/headline_max_tokens"] == HEADLINE_CAP
    assert m["periodic_eval/olympiadbench/budget_matches_headline"] == 0.0


def test_a_point_cannot_be_emitted_without_its_budget():
    """THE REQUIRED TEST. A score with no budget beside it must not reach the logger.

    Driven through the one function every emission goes through, with a row that carries a
    real accuracy and no `params.max_tokens` -- which is exactly the shape a future caller
    would produce by building a row by hand.
    """
    row = {"benchmark": "olympiadbench", "accuracy": 0.5, "n_graded": 4, "n_problems": 4}
    with pytest.raises(pe.BudgetUnrecorded) as exc:
        pe.metrics_from(50, {"olympiadbench": row}, None, None, 1.0, 0.0, pe.STATUS["ok"], 1)
    assert "max_tokens" in str(exc.value)


def test_the_row_guard_refuses_a_budgetless_row_before_it_can_be_emitted():
    """The pre-condition half, independent of the post-condition half above."""
    with pytest.raises(pe.BudgetUnrecorded):
        pe.assert_row_records_budget("olympiadbench", {"accuracy": 0.5, "params": {}})


def test_the_row_guard_accepts_a_row_that_records_its_budget():
    """A guard that always fires guards nothing."""
    pe.assert_row_records_budget("olympiadbench", {"accuracy": 0.5, "params": {"max_tokens": 8}})


def test_a_failed_evaluation_emits_no_score_and_is_not_refused_for_a_missing_budget():
    """A point with no score has nothing to misread, so the budget guard must stay quiet."""
    m = pe.metrics_from(50, {}, None, None, 1.0, 0.0, pe.STATUS["endpoint_error"], -1)
    assert m["periodic_eval/status_code"] == pe.STATUS["endpoint_error"]


@pytest.mark.parametrize(
    "cap,expected", [(HEADLINE_CAP, 1.0), (HEADLINE_CAP - 1, 0.0), (256, 0.0)]
)
def test_the_comparability_flag_follows_the_budget_in_both_directions(cap, expected):
    """A flag stuck at either value is useless; both readings are pinned here."""
    row = {"benchmark": "olympiadbench", "accuracy": 0.5, "params": {"max_tokens": cap}}
    m = pe.metrics_from(50, {"olympiadbench": row}, None, None, 1.0, 0.0, pe.STATUS["ok"], 1)
    assert m["periodic_eval/olympiadbench/budget_matches_headline"] == expected


def test_the_headline_budget_is_read_from_the_table_the_headline_runs_used():
    """Derived from `BENCH_OVERRIDES`, not written down a second time where it can drift."""
    assert mb.headline_max_tokens("olympiadbench") == mb.BENCH_OVERRIDES["olympiadbench"]["max_tokens"]
    assert mb.headline_max_tokens("math500") == mb.build_parser().get_default("max_tokens")


def test_this_namespace_publishes_no_accuracy_series_at_all():
    """The naming constraint, structurally: there is nothing here to line up with a headline.

    A series called `periodic_eval/<bench>/accuracy` sits on the same axis as a headline
    accuracy and nothing in the plot says the two were measured at different budgets.
    """
    cfg = _config("http://x/v1", "/nowhere")
    keys = cfg.metric_keys()
    assert not [k for k in keys if k.endswith("/accuracy")]
    assert "periodic_eval/olympiadbench/trend_score" in keys
    for suffix in ("max_tokens", "headline_max_tokens", "budget_matches_headline"):
        assert f"periodic_eval/olympiadbench/{suffix}" in keys


@needs_data
def test_the_results_file_records_the_budget_next_to_the_score(endpoint, tokenizer_dir, tio, tmp_path):
    """The artifact is what gets re-read months later, so it must say it too."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    out = tmp_path / "out"
    cfg = _config(url, tokenizer_dir, max_tokens=256, out_dir=str(out))
    pe.run_periodic_eval(cfg, 50, pe.BestValTracker(cfg.patience))
    d = json.loads((out / "step50" / "results.json").read_text())
    b = d["token_budget"]["olympiadbench"]
    assert b["max_tokens"] == 256
    assert b["headline_max_tokens"] == HEADLINE_CAP
    assert b["not_comparable_with_headline"] is True
    assert "must not be plotted beside" in b["note"]


@needs_data
def test_the_log_line_says_the_number_is_not_comparable(endpoint, tokenizer_dir, tio):
    """The human-readable line is what anyone watching a run actually reads."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    lines = []

    class _Logger:
        """Collect the lines the evaluation emits."""

        def info(self, msg):
            """Record one info line.

            Args:
                msg: The message.
            """
            lines.append(msg)

        def error(self, msg):
            """Record one error line.

            Args:
                msg: The message.
            """
            lines.append(msg)

    cfg = _config(url, tokenizer_dir, max_tokens=256)
    pe.run_periodic_eval(cfg, 50, pe.BestValTracker(cfg.patience), logger=_Logger())
    joined = "\n".join(lines)
    assert "NOT-COMPARABLE-WITH-HEADLINE" in joined
    assert "max_tokens=256" in joined
    assert "acc=" not in joined, "the line still calls this an accuracy"


# ------------------------------------------------------------------------ liveness ----


@needs_data
def test_liveness_is_measured_through_token_ids_when_the_server_has_no_tokenizer(
    endpoint, tokenizer_dir, tio
):
    """Without this the whole evaluation still fails: liveness runs after the benchmark and a
    LivenessUnavailable makes the point NaN however well the benchmark went."""
    url, policy = endpoint
    policy.output_ids_for = lambda lora, ids: _answer_ids(tio, "\\boxed{42}")
    policy.logprobs_for = lambda lora: ([-0.1, -0.2] if lora else [-0.9, -0.2])
    cfg = _config(url, tokenizer_dir)
    rep = _run(_liveness(cfg))
    assert policy.n_chat == 0, "a text request was made to a server that cannot serve one"
    assert rep.n_tokens_compared == 4
    assert rep.is_live == 1
    assert rep.max_abs_dlogprob == pytest.approx(0.8)


def test_liveness_still_uses_the_chat_endpoint_when_the_server_has_one(endpoint, tokenizer_dir):
    """Additive here too: the path that produced every previous liveness verdict is unchanged."""
    url, policy = endpoint
    policy.has_tokenizer = True
    policy.logprobs_for = lambda model: ([-0.1, -0.2] if model == ADAPTER else [-0.9, -0.2])
    cfg = _config(url, tokenizer_dir)
    rep = _run(_liveness(cfg))
    assert policy.n_generate == 0, "the token-id path engaged against a TOKENISING server"
    assert policy.n_chat == 4
    assert rep.is_live == 1


async def _liveness(cfg):
    """Run the production liveness probe against the stub.

    Args:
        cfg: The configuration.

    Returns:
        The liveness report.
    """
    import aiohttp

    async with aiohttp.ClientSession() as s:
        return await pe.measure_liveness(s, cfg, ADAPTER)


def test_a_probe_with_no_logprobs_is_still_refused_on_the_token_id_path(endpoint, tokenizer_dir):
    """"Assume live" and "assume inert" are each an answer nobody measured, on either path."""
    url, policy = endpoint
    policy.logprobs_for = lambda lora: None
    cfg = _config(url, tokenizer_dir)
    with pytest.raises(pe.LivenessUnavailable):
        _run(_liveness(cfg))
