"""The GPU half of the probe, driven end to end on CPU with a tiny model.

What is proved here is the ARITHMETIC, not the science: that the dump computes the gradients
it says it computes, under the denominator that makes the analysis's central identity true,
and that it refuses rather than guesses when its inputs are wrong. The science needs a real
checkpoint and a real batch.

The identity under test is the one everything downstream rests on::

    sum over groups of L_group  ==  L_batch

which holds only because every group's loss carries the SAME global denominator. If a group
were normalised by its own token count, the per-group gradients would each be on a different
scale, their sums would not be the cluster gradient, and the four-partition comparison would
be comparing rescaled noise. That is invisible in every output, so it is a test.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from selfevo.cluster_lora.interference_dump import (  # noqa: E402
    LOGIT_PEAK_BYTES_PER_ELEMENT,
    DumpConfig,
    Group,
    LogitsBudgetExceeded,
    RolloutSchemaError,
    assert_logits_fit,
    chunk_tokens_for_budget,
    group_losses,
    load_rollouts,
    logits_peak_bytes,
    run_dump,
)

VOCAB = 128


@pytest.fixture(autouse=True)
def _single_threaded():
    """Pin torch to one CPU thread for this file, and put it back afterwards.

    MEASURED on this box: a 32x32 ``nn.Linear`` took ~10 ms, and one dump of six tiny groups
    took 27 s, almost all of it inside ``torch._C._nn.linear``. That is thread thrashing
    against the training job's host processes, not arithmetic -- one thread brings the same
    dump to a couple of seconds. It says nothing about the cost on a GPU, where these
    matmuls are real work; see FINDINGS_cluster_lora.md before quoting any timing from here.
    """
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


class CharTokenizer:
    """A character-level stand-in, so the test needs no downloaded tokenizer.

    Deliberately a stub rather than a cached real tokenizer: a test that silently skips when
    a model cache is cold is a test that stops constraining the moment it matters.
    """

    def encode(self, text, add_special_tokens=False):
        """Map each character to its ordinal, modulo the tiny model's vocabulary."""
        return [ord(c) % VOCAB for c in text]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A four-layer causal LM saved to disk, which is what the dump loads."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    d = tmp_path_factory.mktemp("ckpt")
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture
def rollouts(tmp_path):
    """Six groups of four samples over two tasks, with boxed answers and mixed rewards."""
    path = tmp_path / "rollouts.jsonl"
    rows = []
    for g in range(6):
        for s in range(4):
            rows.append({
                "group_id": f"p{g}",
                "task": "math" if g < 3 else "code",
                "prompt": f"solve problem number {g} carefully ",
                "response": f"think {s} then \\boxed{{{s}}}",
                # Mixed within every group, so no group is RL-silent by accident and the
                # advantages are non-zero.
                "reward": float(s % 2),
            })
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


@pytest.fixture
def patched_tokenizer(monkeypatch):
    """Point the dump at the stub tokenizer."""
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *a, **k: CharTokenizer()),
    )


def cfg_for(ckpt, rollouts, out, **kw):
    """A CPU dump configuration."""
    base = dict(
        model=ckpt, rollouts=rollouts, out=str(out), device="cpu", dtype="float32",
        sketch_dim=128, full_grad_groups=2, lora_rank=4, lora_alpha=8,
        target_modules=("q_proj", "v_proj"), max_len=128, last_n_layers=2,
    )
    base.update(kw)
    return DumpConfig(**base)


# ------------------------------------------------------------- the loss and its scale ---


def test_the_group_losses_sum_to_the_batch_loss():
    """The global-denominator identity, which makes cluster gradients exact.

    Computed against an independent reference rather than against the function's own
    arithmetic: a test that re-derived ``group_losses`` would pin a copy of the code and
    could not notice the copy drifting.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    conf = AutoConfig.for_model(
        "qwen2", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(conf).eval()
    groups = [
        Group(f"p{g}", "math", [1, 2, 3], [[4, 5, 6, 7], [8, 9, 10, 11]], [1.0, 0.0])
        for g in range(3)
    ]
    resp = sum(len(r) for g in groups for r in g.response_ids)
    prompt = sum(len(g.prompt_ids) * g.size for g in groups)

    total = 0.0
    for g in groups:
        loss, _ = group_losses(
            model, g, device="cpu", token_denominator=resp, prompt_denominator=prompt
        )
        total += float(loss.detach())

    # Independent reference: every row of every group in ONE forward, one denominator.
    ref = 0.0
    for g in groups:
        adv = g.advantages()
        for i, r in enumerate(g.response_ids):
            seq = torch.tensor([g.prompt_ids + r])
            lp = torch.log_softmax(model(input_ids=seq).logits[:, :-1, :], -1)
            picked = lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
            n_p = len(g.prompt_ids)
            ref += -float(adv[i]) * float(picked[n_p - 1:].sum().detach())
    ref /= resp
    assert total == pytest.approx(ref, abs=1e-5)


def test_a_unanimous_group_carries_no_gradient_and_is_not_rescued():
    """29-44% of groups are RL-silent, and the probe must show that, not paper over it.

    Group-level reward normalisation makes a unanimous group's advantages identically zero.
    A probe that renormalised to give it a direction would invent conflict out of groups that
    contribute nothing to any update.
    """
    g = Group("p", "math", [1, 2], [[3, 4], [5, 6]], [1.0, 1.0])
    assert np.allclose(g.advantages(), 0.0)
    mixed = Group("p", "math", [1, 2], [[3, 4], [5, 6]], [1.0, 0.0])
    assert not np.allclose(mixed.advantages(), 0.0)
    assert mixed.advantages().mean() == pytest.approx(0.0)


def test_a_zero_denominator_is_refused():
    """It would divide the whole measurement by nothing rather than by the batch."""
    g = Group("p", "math", [1, 2], [[3, 4]], [1.0])
    with pytest.raises(ValueError, match="denominators must be positive"):
        group_losses(None, g, device="cpu", token_denominator=0, prompt_denominator=1)


# ------------------------------------------------------------------- rollout loading ----


def test_groups_are_formed_by_prompt_identity(rollouts):
    gs = load_rollouts(rollouts, tokenizer=CharTokenizer())
    assert len(gs) == 6 and all(g.size == 4 for g in gs)
    assert [g.group_id for g in gs] == [f"p{i}" for i in range(6)]
    assert {g.task for g in gs} == {"math", "code"}


def test_token_ids_are_accepted_as_well_as_text(tmp_path):
    """Both shapes occur in this project's dumps; requiring one means a lossy conversion."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(
        {"group_id": "a", "prompt_ids": [1, 2], "response_ids": [3, 4], "reward": 1.0}
    ))
    gs = load_rollouts(str(p), tokenizer=CharTokenizer())
    assert gs[0].prompt_ids == [1, 2] and gs[0].response_ids == [[3, 4]]


def test_a_record_with_no_reward_is_refused_and_the_error_names_the_fields(tmp_path):
    """A default reward would fabricate the very signal under test.

    The message lists the field names that were looked for and the ones the record has, so
    a schema mismatch on another box is fixed by reading the error rather than by guessing.
    """
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"group_id": "a", "prompt": "x", "response": "y"}))
    with pytest.raises(RolloutSchemaError, match="no reward field") as e:
        load_rollouts(str(p), tokenizer=CharTokenizer())
    assert "rewards" in str(e.value) and "group_id" in str(e.value)


def test_a_record_with_no_response_is_refused(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"group_id": "a", "prompt": "x", "reward": 1.0}))
    with pytest.raises(RolloutSchemaError, match="no response field"):
        load_rollouts(str(p), tokenizer=CharTokenizer())


def test_a_record_with_no_prompt_is_refused(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"group_id": "a", "response": "y", "reward": 1.0}))
    with pytest.raises(RolloutSchemaError, match="no prompt field"):
        load_rollouts(str(p), tokenizer=CharTokenizer())


# ------------------------------------------------- the harness group-shaped record ------


def test_one_line_per_GROUP_with_a_list_of_responses_is_accepted(tmp_path):
    """The shape the harness writes: prompt, responses, per-response rewards, one line.

    Supported alongside the per-sample shape because requiring one means a conversion step,
    and a conversion step is another place rows are dropped without anything saying so.
    """
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"group_id": "q0", "task": "math", "prompt": "a?",
         "responses": ["r0", "r1", "r2"], "rewards": [1.0, 0.0, 1.0],
         "lengths": [4, 5, 6], "truncated": [False, False, True]},
        {"group_id": "q1", "task": "math", "prompt": "b?",
         "responses": ["s0", "s1"], "rewards": [0.0, 0.0]},
    ]))
    gs = load_rollouts(str(p), tokenizer=CharTokenizer())
    assert [g.group_id for g in gs] == ["q0", "q1"]
    assert [g.size for g in gs] == [3, 2]
    assert gs[0].rewards == [1.0, 0.0, 1.0]
    # Unknown extra fields are ignored, not rejected.
    assert gs[1].task == "math"


def test_a_reward_list_that_does_not_match_the_responses_is_refused(tmp_path):
    """It would score one rollout with another rollout's reward, silently."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(
        {"group_id": "q", "prompt": "a", "responses": ["x", "y", "z"], "rewards": [1.0, 0.0]}
    ))
    with pytest.raises(RolloutSchemaError, match="2 rewards for 3 responses"):
        load_rollouts(str(p), tokenizer=CharTokenizer())


@pytest.mark.parametrize(
    "rec,gid,task",
    [
        ({"uid": "u7", "question": "a", "completions": ["x"], "scores": [1.0]}, "u7", "unknown"),
        ({"prompt_id": 3, "prompt": "a", "response": "x", "acc": [0.5],
          "data_source": "gsm8k"}, "3", "gsm8k"),
    ],
    ids=["uid-question-completions-scores", "prompt_id-acc-data_source"],
)
def test_alternate_field_names_are_accepted(tmp_path, rec, gid, task):
    """Rollout dumps here have used several names; a loader fixed to one needs a converter."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec))
    g = load_rollouts(str(p), tokenizer=CharTokenizer())[0]
    assert g.group_id == gid and g.task == task and g.size == 1


def test_an_explicit_key_overrides_the_search(tmp_path):
    """So a batch whose fields collide with the defaults can still be read."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(
        {"group_id": "a", "mine": "b", "prompt": "x", "response": "y", "reward": 0.0,
         "score": 1.0}
    ))
    g = load_rollouts(str(p), tokenizer=CharTokenizer(), group_key="mine",
                      reward_key="score")[0]
    assert g.group_id == "b" and g.rewards == [1.0]


def test_a_batch_with_no_task_labels_stays_unknown_so_the_calibration_refuses(tmp_path):
    """"unknown" for every group makes task_partition raise, which is the honest outcome."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"group_id": "a", "prompt": "x", "response": "y", "reward": 1.0}))
    assert load_rollouts(str(p), tokenizer=CharTokenizer())[0].task == "unknown"


def test_an_empty_file_is_refused_rather_than_dumping_nothing(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("")
    with pytest.raises(RolloutSchemaError, match="no groups"):
        load_rollouts(str(p), tokenizer=CharTokenizer())


# ----------------------------------------------------------------------- the dump -------


def test_the_dump_writes_every_field_the_analysis_reads(ckpt, rollouts, tmp_path,
                                                        patched_tokenizer):
    """The contract between the two halves of the probe, checked at the file."""
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rollouts, out))
    d = np.load(out, allow_pickle=True)
    assert d["sketch"].shape == (6, 128)
    assert d["prompt_sketch"].shape == (6, 128)
    assert d["meds_feature"].shape == (6, 2)      # last_n_layers=2 of a 4-layer model
    assert d["full_grad"].shape[0] == 2
    assert [str(x) for x in d["group_id"]] == [f"p{i}" for i in range(6)]
    assert set(str(x) for x in d["task"]) == {"math", "code"}
    assert meta["n_groups"] == 6 and meta["n_lora_params"] > 0
    assert "global" in meta["denominator"]


def test_the_stored_full_gradients_reproduce_the_sketched_cosines(ckpt, rollouts, tmp_path,
                                                                  patched_tokenizer):
    """The sketch validation, on real gradients rather than on planted vectors.

    This is the check that licenses reading any cosine in the analysis at all.
    """
    out = tmp_path / "dump.npz"
    run_dump(cfg_for(ckpt, rollouts, out, sketch_dim=8192, full_grad_groups=4))
    d = np.load(out, allow_pickle=True)
    full, sk = d["full_grad"], d["sketch"]
    exact, approx = [], []
    for i in range(full.shape[0]):
        for j in range(i + 1, full.shape[0]):
            exact.append(float(np.dot(full[i], full[j])
                               / (np.linalg.norm(full[i]) * np.linalg.norm(full[j]))))
            approx.append(float(np.dot(sk[i], sk[j])
                                / (np.linalg.norm(sk[i]) * np.linalg.norm(sk[j]))))
    err = np.abs(np.array(exact) - np.array(approx))
    assert err.max() < 0.15, (exact, approx)


def test_a_full_gradient_dump_over_the_size_limit_is_refused(ckpt, rollouts, tmp_path,
                                                             patched_tokenizer):
    """Named up front, so the caller lowers the count on purpose.

    Discovering the size after an hour on eight GPUs is the alternative.
    """
    with pytest.raises(RuntimeError, match="over the"):
        run_dump(cfg_for(ckpt, rollouts, tmp_path / "d.npz",
                         full_grad_groups=6, max_full_grad_gb=1e-9))


def test_the_prompt_nll_covers_exactly_the_prompt_positions():
    """The ELREA feature is a different LOSS, not the RL loss under another mask.

    Checked against an independent reference over the prompt positions, because a mask that
    quietly covered the response instead would still produce a plausible gradient, a
    plausible sketch and a plausible clustering -- and the ELREA ablation would then be
    comparing the method against itself. Measured: the mutation replacing the prompt mask
    with the response mask SURVIVED a test that only checked the two sketches were not
    parallel.

    Position t predicts token t+1, so the prompt region in emitter coordinates is
    ``t < n_prompt - 1``: the position at ``n_prompt - 1`` emits the response's first token
    and belongs to the response.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    conf = AutoConfig.for_model(
        "qwen2", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(conf).eval()
    g = Group("p", "math", [1, 2, 3, 4], [[5, 6, 7], [8, 9, 10]], [1.0, 0.0])
    resp = sum(len(r) for r in g.response_ids)
    prompt = len(g.prompt_ids) * g.size
    _grpo, pnll = group_losses(
        model, g, device="cpu", token_denominator=resp, prompt_denominator=prompt
    )

    ref = 0.0
    n_p = len(g.prompt_ids)
    for r in g.response_ids:
        seq = torch.tensor([g.prompt_ids + r])
        lp = torch.log_softmax(model(input_ids=seq).logits[:, :-1, :], -1)
        picked = lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        ref += -float(picked[: n_p - 1].sum())
    assert float(pnll.detach()) == pytest.approx(ref / prompt, abs=1e-5)


def test_the_two_gradients_are_different_gradients(ckpt, rollouts, tmp_path,
                                                   patched_tokenizer):
    """The ELREA feature must not be the GRPO gradient under another name.

    The GRPO loss is masked to RESPONSE tokens, so its gradient restricted to prompt
    positions is identically zero; the prompt feature is a different loss entirely, computed
    in this script because the trainer's path cannot produce it. If the two sketches agreed,
    the ELREA ablation would be comparing the method against itself.
    """
    out = tmp_path / "dump.npz"
    run_dump(cfg_for(ckpt, rollouts, out))
    d = np.load(out, allow_pickle=True)
    a, b = d["sketch"], d["prompt_sketch"]
    cos = float(np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))
    assert abs(cos) < 0.9, cos


def test_the_dump_records_how_much_of_the_gradient_was_zero(ckpt, rollouts, tmp_path,
                                                            patched_tokenizer):
    """At a fresh LoRA init B=0, so dL/dA vanishes and half of every gradient is zero.

    That is still a real gradient but not a mid-training one, and a probe run that way must
    say so rather than have its cosines read as if they came from a trained checkpoint.
    """
    out = tmp_path / "dump.npz"
    run_dump(cfg_for(ckpt, rollouts, out))
    z = np.load(out, allow_pickle=True)["zero_block_fraction"]
    assert z.shape == (6,)
    assert float(z.mean()) == pytest.approx(0.5, abs=0.01)


def test_a_model_with_no_lora_parameters_is_refused(ckpt, rollouts, tmp_path,
                                                    patched_tokenizer, monkeypatch):
    """It would sketch an empty gradient and report cosines over nothing."""
    import selfevo.cluster_lora.interference_dump as dump

    monkeypatch.setattr(dump, "_lora_blocks", lambda model: iter(()))
    with pytest.raises(RuntimeError, match="no trainable LoRA parameters"):
        run_dump(cfg_for(ckpt, rollouts, tmp_path / "d.npz"))


def test_dump_then_analyse_runs_end_to_end(ckpt, rollouts, tmp_path, patched_tokenizer):
    """Both halves against each other, on a real model, with all four partitions attempted.

    The values are not asserted -- a randomly initialised four-layer model has no behaviour
    to cluster -- but the pipeline either produces four blocks from one file or it does not,
    and that is the thing this test can settle.
    """
    pytest.importorskip("sklearn")
    from selfevo.cluster_lora.interference_analyze import analyse_dump

    out = tmp_path / "dump.npz"
    run_dump(cfg_for(ckpt, rollouts, out))
    res = analyse_dump(str(out), n_boot=20, seed=0)
    assert [b["partition"] for b in res["partitions"]] == [
        "meds", "random_matched", "elrea", "task"
    ]
    assert res["n_groups"] == 6
    assert res["sketch_validation"]["n_groups"] == 2


PROBE_VENV = "/home/ubuntu/venv_probe/bin/python"
# The repo this test file belongs to, NOT a hardcoded checkout. The mutation harness runs
# against a copy, and a hardcoded path would make the subprocess import the live tree while
# the rest of the run used the copy -- an isolation hole that reports every mutation as
# killed for the wrong reason.
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])


@pytest.mark.skipif(not os.path.exists(PROBE_VENV), reason="no ~/venv_probe on this box")
def test_the_dump_and_the_analysis_run_in_their_two_SEPARATE_venvs(ckpt, rollouts, tmp_path,
                                                                   patched_tokenizer):
    """The split that makes the probe usable at all, exercised as two real processes.

    The GPU box carries torch, peft and transformers and must NOT acquire scikit-learn or
    hdbscan -- installing them under a live job is the dependency risk this project refuses.
    So the dump imports none of them and the analysis imports no torch. A test that ran both
    halves in one interpreter would prove nothing about that, because the interpreter it ran
    in would have everything.

    This is also exactly the invocation the H100 run uses, so a broken CLI fails here rather
    than after the GPUs are allocated.
    """
    out = tmp_path / "dump.npz"
    run_dump(cfg_for(ckpt, rollouts, out))

    # The analysis venv must NOT be able to run the dump: it has no torch.
    probe_has_torch = subprocess.run(
        [PROBE_VENV, "-c", "import torch"], capture_output=True
    ).returncode == 0
    assert not probe_has_torch, "the analysis venv has torch, so the split is untested here"
    # And the training venv must NOT have the clustering deps.
    assert subprocess.run(
        [sys.executable, "-c", "import hdbscan"], capture_output=True
    ).returncode != 0, "the training venv has hdbscan; it was supposed to stay clean"

    r = subprocess.run(
        [PROBE_VENV, "-m", "selfevo.cluster_lora.interference_analyze",
         "--dump", str(out), "--out", str(tmp_path / "res.json"),
         "--bootstrap", "20", "--min-cluster-size", "2"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT), timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "res.json").read_text())
    assert [b["partition"] for b in res["partitions"]] == [
        "meds", "random_matched", "elrea", "task"
    ]
    assert res["n_groups"] == 6


# =========================================================================================
# The OOM fix: the unembedding is chunked, and the budget is a guard rather than a stack
# trace. Measured on an 80 GiB H100 at 32B / vocab 152,064: the old path built every
# response in a group as ONE padded batch, and the worst group of the probe batch is
# 8 x 2,089 = 16,712 padded tokens -- 4.73 GB of bf16 logits plus a 9.47 GB fp32 copy with
# the bf16 node still alive, 14.20 GB, against 61.02 GB of resident weights. It died inside
# the LM head in the first group, where no guard could see it.
# =========================================================================================


def tiny_lm(vocab=VOCAB, seed=0, lora=True):
    """A small causal LM, optionally LoRA-wrapped, for the equivalence checks."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    conf = AutoConfig.for_model(
        "qwen2", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=vocab,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(conf)
    if not lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    m = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                   target_modules=["q_proj", "v_proj"], bias="none"),
        autocast_adapter_dtype=False,
    )
    # B is zero at init, so dL/dA vanishes and half the gradient would be trivially equal
    # under any refactor. Randomise so both halves of every adapter carry signal.
    with torch.no_grad():
        for n, prm in m.named_parameters():
            if prm.requires_grad and "lora_" in n:
                prm.normal_(0.0, 0.05, generator=torch.Generator().manual_seed(seed + 1))
    return m


def reference_losses(model, group, *, token_denominator, prompt_denominator):
    """The ORIGINAL unchunked path, kept as the thing the refactor has to reproduce.

    This is the code that OOMed: one forward through the causal-LM wrapper, ``.float()`` on
    the full-vocabulary logits, a log-softmax over all of them, then a gather. It is fine at
    this size and is the only honest reference for "the measured gradient did not change".
    """
    adv = group.advantages()
    n_prompt = len(group.prompt_ids)
    seqs = [group.prompt_ids + r for r in group.response_ids]
    width = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, sq in enumerate(seqs):
        ids[i, : len(sq)] = torch.tensor(sq, dtype=torch.long)
        attn[i, : len(sq)] = 1
    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits.float()
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = ids[:, 1:]
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    valid = attn[:, 1:].bool()
    pos = torch.arange(width - 1).unsqueeze(0)
    resp_mask = valid & (pos >= n_prompt - 1)
    prompt_mask = valid & (pos < n_prompt - 1)
    a = torch.tensor(adv, dtype=torch.float32).unsqueeze(1)
    return (
        -(a * picked * resp_mask).sum() / token_denominator,
        -(picked * prompt_mask).sum() / prompt_denominator,
    )


def lora_grads(model):
    """LoRA gradients by name, so two paths can be compared parameter by parameter."""
    return {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    }


def probe_group(n_seq=4, n_prompt=9, n_resp=7):
    """A group shaped like a real one: one shared prompt, several sampled responses."""
    rng = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, VOCAB, (n_prompt,), generator=rng).tolist()
    resp = [torch.randint(0, VOCAB, (n_resp,), generator=rng).tolist() for _ in range(n_seq)]
    return Group("g", "math", prompt, resp, [float(i % 2) for i in range(n_seq)])


@pytest.mark.parametrize("chunk", [1, 3, 7, 10_000])
def test_the_chunked_path_is_numerically_identical_to_the_unchunked_one(chunk):
    """The hard requirement: the fix must not change WHAT IS MEASURED.

    Both the loss and every LoRA gradient are compared against the original full-vocabulary
    path, at four chunk sizes including one token per chunk and one chunk for everything. If
    any of them differed, the whole four-partition comparison would be measuring a different
    gradient than the one the paper claims.

    Chunking is a reassociation of a sum over positions, so this is exact arithmetic rather
    than an approximation, and the tolerance is set accordingly.
    """
    g = probe_group()
    resp = sum(len(r) for r in g.response_ids)
    prompt = len(g.prompt_ids) * g.size

    ref_model = tiny_lm(seed=5)
    ref_model.zero_grad(set_to_none=True)
    ref_grpo, _ = reference_losses(
        ref_model, g, token_denominator=resp, prompt_denominator=prompt
    )
    ref_grpo.backward()
    ref = lora_grads(ref_model)

    got_model = tiny_lm(seed=5)
    got_model.zero_grad(set_to_none=True)
    got_grpo, _ = group_losses(
        got_model, g, device="cpu", token_denominator=resp, prompt_denominator=prompt,
        chunk_tokens=chunk,
    )
    got_grpo.backward()
    got = lora_grads(got_model)

    assert float(got_grpo.detach()) == pytest.approx(float(ref_grpo.detach()), abs=1e-6)
    assert set(got) == set(ref) and ref
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6, rtol=1e-5), (
            k, chunk, (got[k] - ref[k]).abs().max()
        )
    # And the gradient is not trivially zero, or the comparison above is vacuous.
    assert max(float(v.abs().max()) for v in ref.values()) > 0


def test_the_prompt_loss_is_identical_under_chunking_too():
    """The ELREA feature is the other half of the measurement and gets the same check."""
    g = probe_group()
    resp = sum(len(r) for r in g.response_ids)
    prompt = len(g.prompt_ids) * g.size

    ref_model = tiny_lm(seed=6)
    ref_model.zero_grad(set_to_none=True)
    _, ref_nll = reference_losses(
        ref_model, g, token_denominator=resp, prompt_denominator=prompt
    )
    ref_nll.backward()
    ref = lora_grads(ref_model)

    got_model = tiny_lm(seed=6)
    got_model.zero_grad(set_to_none=True)
    _, got_nll = group_losses(
        got_model, g, device="cpu", token_denominator=resp, prompt_denominator=prompt,
        chunk_tokens=4,
    )
    got_nll.backward()
    got = lora_grads(got_model)
    assert float(got_nll.detach()) == pytest.approx(float(ref_nll.detach()), abs=1e-6)
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6, rtol=1e-5)


def test_checkpointing_the_chunks_does_not_change_the_gradient():
    """Recomputing a chunk's logits in the backward must be exact, not merely close.

    It is only exact for a deterministic forward, which is why the dump refuses to run with
    dropout active rather than trusting it.
    """
    g = probe_group()
    resp = sum(len(r) for r in g.response_ids)
    prompt = len(g.prompt_ids) * g.size
    out = {}
    for flag in (True, False):
        m = tiny_lm(seed=7)
        m.zero_grad(set_to_none=True)
        loss, _ = group_losses(
            m, g, device="cpu", token_denominator=resp, prompt_denominator=prompt,
            chunk_tokens=5, use_checkpoint=flag,
        )
        loss.backward()
        out[flag] = lora_grads(m)
    for k in out[True]:
        assert torch.equal(out[True][k], out[False][k]), k


def test_the_chunked_path_still_reaches_every_lora_parameter():
    """Calling the trunk directly bypasses the causal-LM wrapper's forward.

    PEFT replaces the Linear MODULES in place, so LoRA is inside the trunk and is reached
    whichever forward calls them -- but that is PEFT's design, not this project's, and a
    prompt-learning method would live on the wrapper instead. Asserted rather than assumed.
    """
    g = probe_group()
    m = tiny_lm(seed=8)
    m.zero_grad(set_to_none=True)
    loss, _ = group_losses(
        m, g, device="cpu", token_denominator=sum(len(r) for r in g.response_ids),
        prompt_denominator=len(g.prompt_ids) * g.size, chunk_tokens=6,
    )
    loss.backward()
    grads = lora_grads(m)
    assert grads, "no LoRA parameters at all"
    reached = [k for k, v in grads.items() if float(v.abs().sum()) > 0]
    assert len(reached) == len(grads), sorted(set(grads) - set(reached))


# ------------------------------------------------------------------ the budget arithmetic --


def test_the_budget_prices_the_measured_failure():
    """The estimate must reproduce the allocation that actually died.

    32B, vocab 152,064, the probe batch's worst group at 8 x 2,089 = 16,712 padded tokens.
    The old path held bf16 logits, an fp32 copy and the log-softmax output at 10 bytes per
    element over the WHOLE group; this is the number that has to be recognisable as 14.20 GB.
    """
    v, tokens = 152_064, 16_712
    assert logits_peak_bytes(tokens, v, peak_bytes=10) / 1e9 == pytest.approx(25.4, abs=0.1)
    # bf16 alone, which is the figure the traceback's allocator was working against.
    assert logits_peak_bytes(tokens, v, peak_bytes=2) / 1e9 == pytest.approx(5.08, abs=0.05)


def test_the_default_budget_lands_in_the_measured_safe_band():
    """2,048-4,096 padded tokens per chunk is what the length distribution calls for.

    p90 is 9,320 padded tokens per group and the max 16,712, so a chunk sized near p90 would
    leave roughly the ~10 GB headroom at which the run died. Deriving the chunk from a memory
    BUDGET rather than fixing a token count is what makes the same setting mean the same
    memory at another vocabulary.
    """
    n = chunk_tokens_for_budget(152_064, 4 * 1024**3)
    assert 2048 <= n <= 4096, n
    # Half the budget halves the chunk; a smaller vocabulary buys a larger one.
    assert chunk_tokens_for_budget(152_064, 2 * 1024**3) == n // 2
    assert chunk_tokens_for_budget(32_000, 4 * 1024**3) > n


def test_the_chunk_is_capped_at_the_work_available():
    """A 200-token group must not plan a 2,000-token chunk and pretend to price it."""
    assert chunk_tokens_for_budget(152_064, 4 * 1024**3, cap=200) == 200


def test_a_budget_too_small_for_one_token_is_refused_not_rounded_up():
    """No chunk size can fix it, so silently returning 1 would guarantee the OOM anyway."""
    with pytest.raises(LogitsBudgetExceeded, match="no chunk size"):
        chunk_tokens_for_budget(152_064, 1024)


def test_the_guard_refuses_the_worst_real_group_on_a_full_card():
    """Group id 11 of the probe batch: a 1,330-token prompt multiplied across eight samples.

    It is the only group far beyond p99 and it is 11th of 128 in file order, so a guard that
    fires does so in the first minute rather than after a hundred groups of work. Priced here
    against the memory that was actually free when the run died.
    """
    with pytest.raises(LogitsBudgetExceeded, match="id=11") as e:
        assert_logits_fit(
            group_id="id=11", n_tokens=16_712, vocab=152_064, chunk_tokens=2192,
            budget_bytes=4 * 1024**3, free_bytes=int(0.05 * 1e9),
        )
    msg = str(e.value)
    assert "16712 padded tokens" in msg and "Unchunked" in msg


def test_the_guard_passes_the_same_group_when_the_memory_is_there():
    """The refusal has to be about memory, not about the group being large."""
    rec = assert_logits_fit(
        group_id="id=11", n_tokens=16_712, vocab=152_064, chunk_tokens=2192,
        budget_bytes=4 * 1024**3, free_bytes=int(18 * 1e9),
    )
    assert rec["n_tokens"] == 16_712 and rec["chunk_tokens"] == 2192
    assert rec["unchunked_peak_bytes"] > rec["chunk_peak_bytes"] * 7


def test_a_chunk_over_the_budget_is_refused_even_with_memory_free():
    """The budget is the declared plan; exceeding it silently would make it decoration."""
    with pytest.raises(LogitsBudgetExceeded, match="over the"):
        assert_logits_fit(
            group_id="g", n_tokens=1000, vocab=152_064, chunk_tokens=100_000,
            budget_bytes=4 * 1024**3, free_bytes=int(80 * 1e9),
        )


def test_the_guard_fires_BEFORE_the_forward_runs():
    """The entire point. The old failure happened inside the LM head, unguardable.

    A model that raises if it is called at all makes "before" checkable rather than implied
    by argument order.
    """
    class Exploding:
        """Reports a decoder and an unembedding, and refuses to be run."""

        def __init__(self):
            self.calls = 0
            self.weight = torch.zeros(152_064, 8)

        def get_decoder(self):
            """Return a callable that must never be reached."""
            def _boom(**_kw):
                self.calls += 1
                raise AssertionError("the forward ran despite the guard")
            return _boom

        def get_output_embeddings(self):
            """An unembedding of the real vocabulary, so the estimate is realistic."""
            return self

    m = Exploding()
    with pytest.raises(LogitsBudgetExceeded):
        group_losses(
            m, probe_group(n_seq=8, n_prompt=1330, n_resp=759), device="cpu",
            token_denominator=100, prompt_denominator=100,
            chunk_tokens=2192, free_bytes=int(0.05 * 1e9),
        )
    assert m.calls == 0


def test_active_dropout_is_refused_because_the_forward_is_recomputed():
    """Checkpointing differentiates the SECOND forward, which draws a different mask.

    The result would be a plausible gradient with no error anywhere -- the failure class this
    project distrusts most, and one it has already recorded once for checkpointing.
    """
    from selfevo.cluster_lora.interference_dump import _assert_deterministic_forward

    m = tiny_lm(seed=9)
    _assert_deterministic_forward(m)  # the shipped config has no active dropout
    m.add_module("noisy", torch.nn.Dropout(p=0.1))
    with pytest.raises(RuntimeError, match="dropout is active"):
        _assert_deterministic_forward(m)


def test_the_dump_records_the_budget_it_ran_under(ckpt, rollouts, tmp_path,
                                                  patched_tokenizer):
    """A completed dump must be readable against its plan, not against assumed defaults."""
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rollouts, out))
    assert meta["vocab"] == VOCAB
    assert meta["chunk_tokens"] >= 1
    assert meta["gradient_checkpointing"] is True and meta["use_checkpoint"] is True
    assert meta["logits_budget"]["n_tokens"] > 0
    assert meta["logits_budget"]["unchunked_peak_bytes"] >= \
        meta["logits_budget"]["chunk_peak_bytes"]


def test_a_dump_whose_worst_group_will_not_fit_refuses_up_front(ckpt, rollouts, tmp_path,
                                                                patched_tokenizer):
    """Priced once before the loop, so the refusal costs seconds rather than an hour."""
    with pytest.raises(LogitsBudgetExceeded):
        run_dump(cfg_for(ckpt, rollouts, tmp_path / "d.npz", chunk_tokens=10**7))


def test_no_full_vocabulary_tensor_is_ever_larger_than_one_chunk():
    """The guarantee the fix exists for, asserted on the allocations rather than inferred.

    Every other test here checks that the ANSWER is unchanged. This one checks the thing that
    changed: the widest full-vocabulary tensor the forward builds. Under the old path it was
    ``B * T x V`` -- 16,712 x 152,064 at 32B, which is what did not fit; under the new one no
    single unembedding output exceeds ``chunk x V``, whatever the group.

    Measured by recording the size of every ``F.linear`` output whose last dimension is the
    vocabulary, which is exactly the unembedding and nothing else.
    """
    import torch.nn.functional as F

    g = probe_group(n_seq=6, n_prompt=11, n_resp=9)
    m = tiny_lm(seed=11)
    seen = []
    real = F.linear

    def recording(inp, weight, bias=None):
        """Record the unembedding outputs, pass everything else through untouched."""
        out = real(inp, weight, bias)
        if out.shape[-1] == VOCAB:
            seen.append(int(out.numel()))
        return out

    chunk = 7
    positions = g.size * (len(g.prompt_ids) + len(g.response_ids[0]) - 1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "linear", recording)
        loss, _ = group_losses(
            m, g, device="cpu", token_denominator=sum(len(r) for r in g.response_ids),
            prompt_denominator=len(g.prompt_ids) * g.size, chunk_tokens=chunk,
            use_checkpoint=False,
        )
        loss.backward()

    assert seen, "no unembedding output was observed at all"
    assert max(seen) <= chunk * VOCAB, (max(seen), chunk * VOCAB)
    # And the group genuinely needed several chunks, or the bound is met trivially.
    assert positions > chunk * 2
    assert len(seen) >= positions // chunk
    # The old path would have built one tensor of positions * VOCAB; nothing here is close.
    assert max(seen) < positions * VOCAB


def test_the_reduction_is_done_in_fp32_without_materialising_an_fp32_copy():
    """Half the old peak was an fp32 COPY of the logits, and it was not buying accuracy.

    ``log_softmax(x.float())`` allocates a second full-vocabulary tensor and keeps the bf16
    node alive beside it; ``log_softmax(x, dtype=torch.float32)`` performs the same reduction
    in fp32 with no such copy. MEASURED on CPU at bf16 over a 19k vocabulary: the new form is
    BIT-IDENTICAL to the old one, while dropping the ``dtype`` argument and reducing in bf16
    costs 0.044 in log-probability -- which at 152k vocabulary would move every cosine in the
    analysis for no reason anyone could see.
    """
    from selfevo.cluster_lora.interference_dump import _chunk_weighted_logp

    torch.manual_seed(0)
    v, c, h = 19_008, 4, 32
    hidden = (torch.randn(c, h) * 2).to(torch.bfloat16)
    unembed = (torch.randn(v, h) * 0.5).to(torch.bfloat16)
    targets = torch.randint(0, v, (c,))
    w = torch.ones(c)

    got = _chunk_weighted_logp(hidden, targets, (w,), unembed)
    logits = torch.nn.functional.linear(hidden, unembed)
    ref = (torch.log_softmax(logits.float(), dim=-1)
           .gather(1, targets.unsqueeze(1)).squeeze(1) * w).sum()
    assert torch.equal(got[0], ref), (float(got[0]), float(ref))

    bf16_only = (torch.log_softmax(logits, dim=-1).float()
                 .gather(1, targets.unsqueeze(1)).squeeze(1) * w).sum()
    assert not torch.equal(bf16_only, ref), "bf16 reduction happens to be exact here"


def test_a_model_with_no_decoder_trunk_is_refused():
    """Falling back to the wrapper's forward would silently restore the OOM.

    The chunked path exists only because the trunk can be called without the LM head; a model
    where it cannot be found has to say so rather than quietly take the path that does not
    fit.
    """
    from selfevo.cluster_lora.interference_dump import _decoder_and_unembedding

    with pytest.raises(RuntimeError, match="could not locate the decoder trunk"):
        _decoder_and_unembedding(torch.nn.Linear(4, 4))


@pytest.mark.parametrize("seq_chunk", [1, 2, 3, 8])
def test_sub_batching_the_trunk_gives_the_same_gradient_as_one_backward(seq_chunk):
    """The other half of the OOM fix, and it has the same hard requirement.

    Sequences in a GRPO group are independent -- causal attention with right padding gives no
    path between them -- and the loss is a sum over them under a denominator that does not
    depend on the split, so backwarding each sub-batch accumulates exactly the gradient one
    backward over the whole group would. Checked at every sub-batch size including one
    sequence at a time and the whole group at once.
    """
    from selfevo.cluster_lora.interference_dump import group_backward

    g = probe_group(n_seq=4, n_prompt=8, n_resp=6)
    resp = sum(len(r) for r in g.response_ids)
    prompt = len(g.prompt_ids) * g.size

    whole = tiny_lm(seed=12)
    whole.zero_grad(set_to_none=True)
    ref_loss, _ = group_losses(
        whole, g, device="cpu", token_denominator=resp, prompt_denominator=prompt
    )
    ref_loss.backward()
    ref = lora_grads(whole)

    split = tiny_lm(seed=12)
    split.zero_grad(set_to_none=True)
    got_value = group_backward(
        split, g, which="grpo", device="cpu", token_denominator=resp,
        prompt_denominator=prompt, seq_chunk=seq_chunk,
    )
    got = lora_grads(split)

    assert got_value == pytest.approx(float(ref_loss.detach()), abs=1e-6)
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6, rtol=1e-5), (
            k, seq_chunk, (got[k] - ref[k]).abs().max()
        )
    assert max(float(v.abs().max()) for v in ref.values()) > 0


def test_a_sub_batch_keeps_the_PARENT_groups_advantages():
    """The subtle way sub-batching could silently change the measurement.

    GRPO advantages are the group's rewards centred and scaled WITHIN the group. If a
    sub-batch recomputed them from its own slice it would centre a subset -- a different, and
    for a two-sample slice a nearly meaningless, quantity -- and the run would complete with
    plausible numbers. Here the rewards are chosen so that slicing changes the centring:
    every slice of two has a different mean from the whole.
    """
    from selfevo.cluster_lora.interference_dump import group_backward

    rng = torch.Generator().manual_seed(4)
    prompt = torch.randint(0, VOCAB, (7,), generator=rng).tolist()
    resp = [torch.randint(0, VOCAB, (5,), generator=rng).tolist() for _ in range(4)]
    g = Group("g", "math", prompt, resp, [0.0, 0.0, 0.0, 1.0])
    # Sanity: the whole group's advantages are not the concatenation of its slices' own.
    from numpy import allclose as np_allclose

    slice_own = Group("g", "math", prompt, resp[:2], [0.0, 0.0]).advantages()
    assert not np_allclose(slice_own, g.advantages()[:2])

    resp_d = sum(len(r) for r in g.response_ids)
    prompt_d = len(g.prompt_ids) * g.size

    whole = tiny_lm(seed=13)
    whole.zero_grad(set_to_none=True)
    ref_loss, _ = group_losses(
        whole, g, device="cpu", token_denominator=resp_d, prompt_denominator=prompt_d
    )
    ref_loss.backward()
    ref = lora_grads(whole)

    split = tiny_lm(seed=13)
    split.zero_grad(set_to_none=True)
    group_backward(
        split, g, which="grpo", device="cpu", token_denominator=resp_d,
        prompt_denominator=prompt_d, seq_chunk=2,
    )
    for k in ref:
        assert torch.allclose(lora_grads(split)[k], ref[k], atol=1e-6, rtol=1e-5), k


def test_the_forward_token_budget_prices_the_measured_activation_cost():
    """64 layers x 5120 wide is 655 KB of retained activations per token.

    The probe batch's worst group at 16,712 padded tokens therefore retains 10.95 GB under
    gradient checkpointing, on top of 61.02 GB of weights -- which is why the trunk is
    sub-batched and not only the head.
    """
    from selfevo.cluster_lora.interference_dump import forward_tokens_for_budget

    per_token = 64 * 5120 * 2
    assert per_token / 1024 == pytest.approx(640.0)
    assert 16_712 * per_token / 1e9 == pytest.approx(10.95, abs=0.05)
    # The 6 GB default, and what it buys on the two groups that matter.
    n = forward_tokens_for_budget(64, 5120, 6 * 1024**3)
    assert n == 9830, n
    # id=11 at T=2089: four of its eight sequences per forward, so two forwards, retaining
    # 8,356 x 640 KB = 5.48 GB instead of 10.95. With 61.02 GB of weights and a 4 GB head
    # chunk that leaves roughly 7 GB spare on an 80 GiB card.
    assert n // 2089 == 4
    assert 4 * 2089 * 64 * 5120 * 2 / 1e9 == pytest.approx(5.48, abs=0.02)
    # The median group (T=875) still runs as ONE forward, so only the tail pays for the split.
    assert n // 875 >= 8


def test_an_unknown_loss_name_is_refused():
    from selfevo.cluster_lora.interference_dump import group_backward

    with pytest.raises(ValueError, match="unknown loss"):
        group_backward(None, probe_group(), which="both", device="cpu",
                       token_denominator=1, prompt_denominator=1)


@pytest.mark.parametrize("seq_chunk,expected", [(1, 4), (2, 2), (3, 2), (4, 1), (99, 1)])
def test_the_trunk_really_is_split_into_that_many_forwards(seq_chunk, expected):
    """Sub-batching that produces the right ANSWER while never splitting saves nothing.

    Every other test of ``group_backward`` asserts the gradient is unchanged -- and it is
    unchanged when the split never happens, so those tests pass on a version that runs one
    forward over the whole group and OOMs at 32B exactly as before. Measured: the mutation
    forcing ``seq_chunk = group.size`` SURVIVED all of them. What has to be checked is the
    number of trunk forwards.
    """
    import selfevo.cluster_lora.interference_dump as dump

    g = probe_group(n_seq=4, n_prompt=8, n_resp=6)
    m = tiny_lm(seed=14)
    calls = []
    real = dump.group_losses

    def counting(model, group, **kw):
        """Record each sub-batch's size, then delegate untouched."""
        calls.append(group.size)
        return real(model, group, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dump, "group_losses", counting)
        dump.group_backward(
            m, g, which="grpo", device="cpu",
            token_denominator=sum(len(r) for r in g.response_ids),
            prompt_denominator=len(g.prompt_ids) * g.size, seq_chunk=seq_chunk,
        )
    assert len(calls) == expected, calls
    assert sum(calls) == g.size, calls
    assert max(calls) <= min(seq_chunk, g.size)


def test_the_default_sub_batch_follows_the_activation_budget():
    """With no explicit seq_chunk the split must come from the budget, not from the group.

    A default that silently ran the whole group would be the OOM again, and would pass every
    gradient-equality test.
    """
    import selfevo.cluster_lora.interference_dump as dump

    g = probe_group(n_seq=4, n_prompt=8, n_resp=6)
    m = tiny_lm(seed=15)
    width = len(g.prompt_ids) + len(g.response_ids[0])
    per_token = m.config.num_hidden_layers * m.config.hidden_size * 2
    # A budget deliberately small enough that only two sequences fit per forward.
    budget = 2 * width * per_token
    calls = []
    real = dump.group_losses

    def counting(model, group, **kw):
        """Record each sub-batch's size, then delegate untouched."""
        calls.append(group.size)
        return real(model, group, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dump, "group_losses", counting)
        dump.group_backward(
            m, g, which="grpo", device="cpu",
            token_denominator=sum(len(r) for r in g.response_ids),
            prompt_denominator=len(g.prompt_ids) * g.size,
            activation_budget_bytes=budget,
        )
    assert calls == [2, 2], calls


# =========================================================================================
# The guard's ADVICE. Found by running it on the box: the free-memory branch reused the
# budget branch's clause and told the reader to raise --logits-budget-gb, which is strictly
# counterproductive there -- the chunk is derived from the budget, so a larger budget makes a
# larger chunk and a larger requirement. A guard that refuses correctly and then misdirects
# is worse than one that says nothing, because the reader is stuck at that moment.
# =========================================================================================


REAL_FREE = int(4.60e9)   # measured free memory on the 80 GB card holding 61 GB of weights
REAL_VOCAB = 152_064


def test_the_free_memory_branch_does_not_advise_raising_the_budget():
    """The bug. Raising the budget enlarges the very quantity being compared."""
    with pytest.raises(LogitsBudgetExceeded) as e:
        assert_logits_fit(
            group_id="2", n_tokens=8824, vocab=REAL_VOCAB, chunk_tokens=2192,
            budget_bytes=4 * 1024**3, free_bytes=REAL_FREE,
        )
    msg = str(e.value)
    # Every mention of raising the budget must be a NEGATED one. Asserted as a count rather
    # than as absence, because the correct message necessarily contains the phrase inside
    # "Do NOT raise ...", and a bare `not in` check fails on the fixed version.
    assert msg.count("raise --logits-budget-gb") == msg.count(
        "Do NOT raise --logits-budget-gb"
    ), msg
    assert "Do NOT raise --logits-budget-gb" in msg
    assert "LOWER the budget" in msg
    assert "--chunk-tokens" in msg


def test_the_budget_branch_DOES_advise_raising_the_budget():
    """The other branch is the one where raising it is the fix, and it must still say so."""
    with pytest.raises(LogitsBudgetExceeded) as e:
        assert_logits_fit(
            group_id="g", n_tokens=1000, vocab=REAL_VOCAB, chunk_tokens=100_000,
            budget_bytes=4 * 1024**3, free_bytes=int(80e9),
        )
    assert "raise --logits-budget-gb to at least" in str(e.value)


def test_the_refusal_names_a_value_that_actually_satisfies_it():
    """A round trip, which is the only way to know the advice is arithmetic and not prose.

    The run that first hit this guard had to derive the working value by hand from the
    message. The message now computes it, and this test takes the number the message would
    print, feeds it back in, and requires the guard to pass -- so the suggestion cannot drift
    from the check it is offered to satisfy.
    """
    from selfevo.cluster_lora.interference_dump import satisfying_plan

    budget, chunk = satisfying_plan(REAL_FREE, vocab=REAL_VOCAB)
    assert chunk >= 1
    # The suggested chunk must pass, at the suggested budget.
    rec = assert_logits_fit(
        group_id="2", n_tokens=8824, vocab=REAL_VOCAB, chunk_tokens=chunk,
        budget_bytes=budget, free_bytes=REAL_FREE,
    )
    assert rec["chunk_tokens"] == chunk
    # And one token more must NOT, or the suggestion is not the largest satisfying value.
    with pytest.raises(LogitsBudgetExceeded):
        assert_logits_fit(
            group_id="2", n_tokens=8824, vocab=REAL_VOCAB, chunk_tokens=chunk + 1,
            budget_bytes=budget + REAL_VOCAB * 12, free_bytes=REAL_FREE,
        )


def test_a_card_too_full_for_even_one_token_says_no_setting_can_fix_it():
    """Suggesting a chunk of zero would be advice that cannot be followed."""
    with pytest.raises(LogitsBudgetExceeded, match="no setting can repair"):
        assert_logits_fit(
            group_id="g", n_tokens=8824, vocab=REAL_VOCAB, chunk_tokens=2192,
            budget_bytes=4 * 1024**3, free_bytes=1024,
        )


def test_the_suggested_units_match_the_flag_that_consumes_them():
    """The flag is GiB (``logits_budget_gb * 1024**3``), so the message must print GiB.

    A suggestion printed in decimal GB would be ~7% too large at this scale and would not
    satisfy the check it was offered for.
    """
    from selfevo.cluster_lora.interference_dump import _gib

    assert _gib(4 * 1024**3) == 4.0
    # Floored, never rounded up: a rounded-up suggestion does not satisfy.
    assert _gib(int(2.99 * 1024**3)) == 2.9
    with pytest.raises(LogitsBudgetExceeded) as e:
        assert_logits_fit(group_id="2", n_tokens=8824, vocab=REAL_VOCAB,
                          chunk_tokens=2192, budget_bytes=4 * 1024**3, free_bytes=REAL_FREE)
    assert "GiB" in str(e.value) and " GB" not in str(e.value)


# ------------------------------------------------- the budget derived from free memory ----


def test_the_default_budget_follows_MEASURED_free_memory():
    """A fixed default was measured refusing the majority of a real batch.

    Group 2 has 8,824 padded tokens -- p75 of the batch, not the 16,712-token outlier -- and
    the per-chunk requirement is near-constant across groups, so a fixed 4 GiB on a card with
    4.60 GB free refuses most of the 128 groups. The derived budget has to fix that.
    """
    from selfevo.cluster_lora.interference_dump import resolve_logits_budget

    budget, why = resolve_logits_budget(None, REAL_FREE, vocab=REAL_VOCAB)
    assert "derived from" in why
    assert budget < 4 * 1024**3
    # The whole point: the group that WAS refused now passes.
    chunk = budget // (REAL_VOCAB * 12)
    assert_logits_fit(group_id="2", n_tokens=8824, vocab=REAL_VOCAB, chunk_tokens=chunk,
                      budget_bytes=budget, free_bytes=REAL_FREE)


def test_an_explicit_budget_is_honoured_even_when_it_will_not_fit():
    """A flag the caller set is a statement of intent.

    Silently overriding it would make the refusal that follows unattributable to anything the
    caller did, which is worse than refusing.
    """
    from selfevo.cluster_lora.interference_dump import resolve_logits_budget

    budget, why = resolve_logits_budget(4.0, REAL_FREE, vocab=REAL_VOCAB)
    assert budget == 4 * 1024**3 and "explicit" in why


def test_the_derived_budget_is_capped_on_an_empty_card():
    """Free memory buys a bigger chunk only up to the ceiling; past that it buys nothing."""
    from selfevo.cluster_lora.interference_dump import resolve_logits_budget

    budget, why = resolve_logits_budget(None, int(78e9), vocab=REAL_VOCAB)
    assert budget == 4 * 1024**3 and "capped" in why


def test_an_unmeasurable_card_falls_back_to_the_ceiling_and_says_so():
    from selfevo.cluster_lora.interference_dump import resolve_logits_budget

    budget, why = resolve_logits_budget(None, None, vocab=REAL_VOCAB)
    assert budget == 4 * 1024**3 and "could not be measured" in why


def test_a_non_positive_explicit_budget_is_refused():
    from selfevo.cluster_lora.interference_dump import resolve_logits_budget

    with pytest.raises(ValueError, match="must be positive"):
        resolve_logits_budget(0.0, REAL_FREE, vocab=REAL_VOCAB)


def test_the_dump_records_which_budget_branch_it_took(ckpt, rollouts, tmp_path,
                                                      patched_tokenizer):
    """A completed run must say how its budget was chosen, not leave it to be inferred."""
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rollouts, out))
    assert "logits_budget_why" in meta and meta["logits_budget_bytes"] > 0
    meta2 = run_dump(cfg_for(ckpt, rollouts, out, logits_budget_gb=1.0))
    assert "explicit" in meta2["logits_budget_why"]
    assert meta2["logits_budget_bytes"] == 1024**3


# =========================================================================================
# The full-gradient store. Found by running the probe on the real batch: 90 of its 128 groups
# are unanimous, so taking the FIRST N groups stored seven exactly-zero gradients and one of
# norm 1.28e-4. The validation needs two non-zero to form a pair, found one, and reported
# that it could not validate -- correctly, but the run had no way to have validated it.
# =========================================================================================


def silent_rollouts(tmp_path, n_groups=10, n_informative=3):
    """A batch shaped like the real one: mostly unanimous, a few informative, silent first.

    The informative groups are placed LAST so a store that takes the first N gets nothing but
    zeros, which is exactly what happened on the box.
    """
    path = tmp_path / "silent.jsonl"
    rows = []
    for g in range(n_groups):
        informative = g >= n_groups - n_informative
        for s in range(4):
            rows.append({
                "group_id": f"p{g}", "task": "math",
                "prompt": f"solve problem number {g} carefully ",
                "response": f"think {s} then \\boxed{{{s}}}",
                # Unanimous unless informative: all 1.0 (k=G) or all 0.0 (k=0).
                "reward": float(s % 2) if informative else float(g % 2),
            })
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_a_unanimous_group_is_reported_as_silent():
    """The predicate the store selects on: reward spread, not a k threshold.

    Group-level normalisation centres the rewards, so a unanimous group has advantages
    identically zero whatever the rewards were -- k=0 and k=G are the same case.
    """
    assert Group("g", "m", [1], [[2], [3]], [0.0, 0.0]).is_silent
    assert Group("g", "m", [1], [[2], [3]], [1.0, 1.0]).is_silent
    assert not Group("g", "m", [1], [[2], [3]], [1.0, 0.0]).is_silent


def test_the_store_selects_INFORMATIVE_groups_not_the_first_N(ckpt, tmp_path,
                                                              patched_tokenizer):
    """The defect. On a 70%-silent batch, "the first N" is a store of zeros.

    The fixture puts every informative group last, so a selection that ignores silence stores
    nothing usable and a selection that respects it stores exactly the right groups.
    """
    rj = silent_rollouts(tmp_path, n_groups=10, n_informative=3)
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rj, out, full_grad_groups=2))
    assert meta["n_groups_informative"] == 3
    # The informative groups are p7, p8, p9 -- never p0.
    assert meta["full_grad_group_ids"] == ["p7", "p8"], meta["full_grad_group_ids"]
    assert "informative" in meta["full_grad_selection"]


def test_two_stored_gradients_are_enough_once_they_are_the_right_two(ckpt, tmp_path,
                                                                     patched_tokenizer):
    """With the selection fixed, N=2 validates -- which is why the store stopped being a
    memory decision and became a statistical one."""
    rj = silent_rollouts(tmp_path, n_groups=10, n_informative=3)
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rj, out, full_grad_groups=2))
    assert meta["full_grad_nonzero"] == 2
    assert meta["sketch_validation_status"].startswith("ok:")
    d = np.load(out, allow_pickle=True)
    assert d["full_grad"].shape[0] == 2
    assert float(np.abs(d["full_grad"]).max()) > 0


def test_a_batch_with_too_few_informative_groups_names_the_condition_loudly(ckpt, tmp_path,
                                                                            patched_tokenizer):
    """The failure must be a named status, not a sentence the reader has to notice.

    On the real run the analysis said "every stored full gradient is zero" and the sketch went
    unvalidated. A status string in the metadata is something a script can refuse on.
    """
    rj = silent_rollouts(tmp_path, n_groups=6, n_informative=1)
    out = tmp_path / "dump.npz"
    meta = run_dump(cfg_for(ckpt, rj, out, full_grad_groups=4))
    assert meta["sketch_validation_status"].startswith("IMPOSSIBLE")
    assert "UNVALIDATED" in meta["sketch_validation_status"]
    assert meta["n_groups_informative"] == 1
    assert meta["full_grad_nonzero"] < 2


def test_the_dump_records_how_much_of_the_batch_carries_no_gradient_at_all(ckpt, tmp_path,
                                                                           patched_tokenizer):
    """The measurement rests on the informative groups only, so the count has to be visible.

    On the real batch that is 38 of 128, and a reader who does not know that will over-read
    the result.
    """
    rj = silent_rollouts(tmp_path, n_groups=10, n_informative=3)
    meta = run_dump(cfg_for(ckpt, rj, tmp_path / "d.npz", full_grad_groups=2))
    assert meta["n_zero_grpo_sketches"] == 7
    assert meta["n_groups"] - meta["n_zero_grpo_sketches"] == 3


def test_the_store_is_moved_to_the_host_BEFORE_it_is_upcast(ckpt, rollouts, tmp_path,
                                                            patched_tokenizer):
    """Order matters, and dtype makes it observable without a GPU.

    ``.float().cpu()`` builds a full fp32 copy on the accelerator first -- twice the transfer,
    and 134 MB of device memory per stored group at 33.5M LoRA parameters, which is what ate
    the run's headroom. ``.cpu().float()`` transfers in the gradient's own dtype and upcasts
    on the host. On a CPU box the DEVICE cannot distinguish them, but the DTYPE crossing
    ``.cpu()`` can: under the correct order ``.cpu()`` is called on a bfloat16 tensor, under
    the wrong one it only ever sees float32.
    """
    seen = []
    real_cpu = torch.Tensor.cpu

    def recording(self, *a, **kw):
        """Record the dtype at the moment of the host transfer."""
        seen.append(self.dtype)
        return real_cpu(self, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "cpu", recording)
        run_dump(cfg_for(ckpt, rollouts, tmp_path / "d.npz", dtype="bfloat16",
                         full_grad_groups=2))
    assert torch.bfloat16 in seen, (
        "every host transfer saw float32, so the gradient was upcast before it was moved"
    )


# =========================================================================================
# The guard's INPUT. The fourth projection attempt failed with the guard firing on exactly
# the condition it was built for -- but the number it was fed was wrong. PyTorch's caching
# allocator keeps freed blocks RESERVED, so `mem_get_info` reports driver-free, which shrinks
# monotonically as a run proceeds even though nothing leaks. At a 4.0 GiB budget it refused
# at group 5 of 153, at 1.5 GiB at group 25: a smaller chunk only buys more groups before the
# window closes. Every existing test checked WHEN the guard fired, none checked what it read.
# =========================================================================================


class FakePool:
    """A caching allocator that reserves what it frees, like the real one.

    ``mem_get_info`` returns driver-free, which is total minus what the pool holds. Each read
    models a group's activations entering the pool. ``empty_cache`` returns them. This is the
    smallest thing that reproduces the failure, and it makes the fix checkable on a box with
    no accelerator at all.
    """

    def __init__(self, total=80 * 1024**3, per_group=2 * 1024**3):
        self.total = total
        self.per_group = per_group
        self.reserved = 0
        self.releases = 0
        self.calls: list[str] = []

    def mem_get_info(self, _device=None):
        """Report driver-free, then grow the pool as the next group would."""
        self.calls.append("read")
        free = self.total - self.reserved
        self.reserved += self.per_group
        return (free, self.total)

    def empty_cache(self):
        """Return every reserved block to the driver."""
        self.calls.append("release")
        self.releases += 1
        self.reserved = 0


@pytest.fixture
def fake_cuda(monkeypatch):
    """Present a CUDA device backed by :class:`FakePool`."""
    pool = FakePool()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", pool.mem_get_info)
    monkeypatch.setattr(torch.cuda, "empty_cache", pool.empty_cache)
    return pool


def test_the_free_memory_probe_releases_the_cache_BEFORE_it_reads(fake_cuda):
    """Order is the fix. Reading first would report the pool's history, not availability."""
    from selfevo.cluster_lora.interference_dump import _free_bytes

    _free_bytes("cuda")
    assert fake_cuda.calls == ["release", "read"], fake_cuda.calls


def test_the_guards_input_does_not_drift_across_groups(fake_cuda):
    """The failure, reproduced and then removed.

    Twenty reads standing in for twenty groups. With the cache released each time the reading
    is the same every time; the guard is comparing its estimate against what is actually
    available rather than against how long the run has been going.
    """
    from selfevo.cluster_lora.interference_dump import _free_bytes

    seen = [_free_bytes("cuda") for _ in range(20)]
    assert len(set(seen)) == 1, seen
    assert seen[0] == 80 * 1024**3


def test_and_without_the_release_it_really_does_drift(fake_cuda):
    """Non-vacuity. If the pool did not shrink the reading, the test above proves nothing.

    This is the old behaviour: monotonically falling, and it crosses any fixed budget
    eventually -- which is why no chunk size finishes the batch.
    """
    from selfevo.cluster_lora.interference_dump import _free_bytes

    # Forty reads: enough for a 2 GiB-per-group pool to walk an 80 GiB card down past any
    # fixed budget, which is the point -- no chunk size finishes, each value just fails later.
    seen = [_free_bytes("cuda", release_cache=False) for _ in range(40)]
    assert seen == sorted(seen, reverse=True) and seen[0] > seen[-1]
    assert min(seen) < 4 * 1024**3, seen[-3:]
    # The reading falls with the number of groups processed, not with anything real.
    assert seen[0] - seen[1] == fake_cuda.per_group


def test_a_cpu_device_still_reports_unknown_rather_than_guessing():
    """The budget check must still run where the device check cannot."""
    from selfevo.cluster_lora.interference_dump import _free_bytes

    assert _free_bytes("cpu") is None


def test_the_dump_re_measures_free_memory_for_every_group(ckpt, rollouts, tmp_path,
                                                          patched_tokenizer):
    """A value read once and reused would be stale by the group it mattered for.

    Recorded per reading so a finished run shows the guard's input rather than only its
    verdict: first and last far apart with min == last is a pool that was never released.
    """
    meta = run_dump(cfg_for(ckpt, rollouts, tmp_path / "d.npz"))
    # CPU reports None, so nothing is recorded -- the keys must still exist and say so.
    assert meta["free_bytes_readings"] == 0
    assert meta["free_bytes_first"] is None and meta["free_bytes_min"] is None


# ---------------------------------------------------- the derived branch must engage ------


def test_omitting_the_budget_flag_reaches_the_DERIVED_branch(monkeypatch, tmp_path):
    """The flag defaulted to 4.0 in argparse while the dataclass defaulted to None.

    So omitting it logged `explicit --logits-budget-gb 4.0` and the derivation never ran. The
    dataclass default was right and the CLI silently overrode it -- and the patch that was
    supposed to fix the CLI had no assertion on its own replacement, so it did nothing and
    said nothing. That is the failure mode this project distrusts, committed while fixing it.
    """
    import selfevo.cluster_lora.interference_dump as dump

    seen = []

    def capture(cfg):
        """Record the config the CLI built and return metadata main() can serialise."""
        seen.append(cfg)
        return {"ok": True}

    monkeypatch.setattr(dump, "run_dump", capture)
    base = ["--model", "m", "--rollouts", "r", "--out", str(tmp_path / "o.npz")]
    dump.main(base)
    assert seen[-1].logits_budget_gb is None, (
        "omitting the flag must leave it None so resolve_logits_budget derives it"
    )
    # And an explicit value still arrives intact, so the fix did not simply ignore the flag.
    dump.main(base + ["--logits-budget-gb", "2.5"])
    assert seen[-1].logits_budget_gb == 2.5
