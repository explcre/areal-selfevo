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
    DumpConfig,
    Group,
    RolloutSchemaError,
    group_losses,
    load_rollouts,
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
