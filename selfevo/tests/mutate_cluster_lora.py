#!/usr/bin/env python3
"""Mutation-test the per-cluster LoRA guards against a COPY of the repo.

A green suite proves the tests run, not that they constrain anything. This project has
already shipped five defects past 350 passing tests, so every guard in
``selfevo/cluster_lora`` is checked by breaking it on purpose and requiring the suite to
notice.

**A copy, not the live checkout.** A training supervisor relaunches on process exit here, so
a mutated file sitting on disk for even a few seconds could be imported by a real run. The
harness refuses to start unless imports resolve inside the copy, and it re-checks each
target's sha256 after every restore -- at the start and at the end -- so a crashed run cannot
leave a mutation behind.

Two interpreters are needed and that is not an accident: the dump half imports torch and must
NOT import scikit-learn or hdbscan, the analysis half is the reverse. Run it twice::

    ~/venv312b/bin/python selfevo/tests/mutate_cluster_lora.py COPY torch
    ~/venv_probe/bin/python selfevo/tests/mutate_cluster_lora.py COPY cluster

Mutations whose tests cannot run under the chosen interpreter are reported as NOT APPLICABLE
rather than as killed, because a mutation that was never exercised is not a passing one.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
ARM = sys.argv[2] if len(sys.argv) > 2 else "torch"

ADAPTERS = REPO / "selfevo/cluster_lora/adapters.py"
MERGE = REPO / "selfevo/cluster_lora/merge.py"
PARTITION = REPO / "selfevo/cluster_lora/partition.py"
SKETCH = REPO / "selfevo/cluster_lora/sketch.py"
DUMP = REPO / "selfevo/cluster_lora/interference_dump.py"
ANALYZE = REPO / "selfevo/cluster_lora/interference_analyze.py"

T_ROUTING = "selfevo/tests/test_cluster_lora_routing.py"
T_MERGE = "selfevo/tests/test_cluster_lora_merge.py"
T_PART = "selfevo/tests/test_cluster_lora_partition.py"
T_FEAT = "selfevo/tests/test_cluster_lora_features.py"
T_CLUST = "selfevo/tests/test_cluster_lora_clustering.py"
T_SKETCH = "selfevo/tests/test_cluster_lora_sketch.py"
T_PROBE = "selfevo/tests/test_cluster_lora_probe.py"
T_ANALYZE = "selfevo/tests/test_cluster_lora_analyze.py"

# Which test files each arm can actually run. The "torch" arm has no scikit-learn, so the
# clustering and analysis suites would SKIP there and report a survivor that is really an
# un-run test.
ARMS = {
    "torch": [T_ROUTING, T_MERGE, T_PART, T_FEAT, T_SKETCH, T_PROBE],
    "cluster": [T_PART, T_SKETCH, T_CLUST, T_ANALYZE],
}

# (target, tests, label, find, replace) -- each a one-line defect a careless edit could make.
MUTATIONS = [
    # ---------------------------------------------------------------- adapter isolation --
    (ADAPTERS, [T_ROUTING], "gradients zeroed into tensors, so weight decay moves every idle expert",
     "optimizer.zero_grad(set_to_none=True)", "optimizer.zero_grad(set_to_none=False)"),
    (ADAPTERS, [T_ROUTING], "the adapter is never activated, so every expert stays in the graph",
     "        self.model.set_adapter(name)\n        try:", "        try:"),
    (ADAPTERS, [T_ROUTING], "the requires_grad leak check never fires",
     "                    if param.requires_grad:", "                    if False:"),
    (ADAPTERS, [T_ROUTING], "the previous adapter is never restored after a cluster",
     "            if previous:\n                self.model.set_adapter(previous[0] if len(previous) == 1 else previous)",
     "            if previous:\n                pass"),
    (ADAPTERS, [T_ROUTING], "a cluster whose loss reaches no adapter is accepted silently",
     "            if not grads:", "            if False and not grads:"),
    (ADAPTERS, [T_ROUTING], "a batch naming an unmanaged adapter is accepted",
     "        if unknown:", "        if False:"),
    (ADAPTERS, [T_ROUTING], "parameters() ignores the adapter name and returns every expert's tensors",
     "                if table is None or name not in table:", "                if table is None:"),
    (ADAPTERS, [T_ROUTING], "unchanged() weakens to a tolerance, so a small leak passes",
     "        return all(torch.equal(now[k], before[k]) for k in before)",
     "        return all(torch.allclose(now[k], before[k]) for k in before)"),
    (ADAPTERS, [T_ROUTING], "a group's rows are truncated to one, so most of the batch is dropped",
     "        out.setdefault(key, []).extend(range(row, row + size))",
     "        out.setdefault(key, []).extend(range(row, row + 1))"),
    (ADAPTERS, [T_ROUTING], "a zero-size group is accepted",
     "        if size <= 0:", "        if False:"),
    (ADAPTERS, [T_ROUTING], "a partition that does not describe the batch is accepted",
     "    if len(keys) != len(group_sizes):", "    if False:"),
    # ------------------------------------------------------------------------- the merge --
    (MERGE, [T_MERGE], "the LoRA scaling is dropped from the delta",
     "        out[mod_name] = float(mod.scaling[name]) * (B @ A)",
     "        out[mod_name] = (B @ A)"),
    (MERGE, [T_MERGE], "the merge verification never fires",
     "        if err > atol + rtol * scale and err > worst:", "        if False:"),
    (MERGE, [T_MERGE], "cat becomes linear, which is a different merge at these ranks",
     'adapter_name=target, combination_type="cat"',
     'adapter_name=target, combination_type="linear"'),
    (MERGE, [T_MERGE], "the ragged-module refusal never fires",
     "        elif here != modules:", "        elif False:"),
    (MERGE, [T_MERGE], "weights are ignored, so a weighted merge is silently a plain sum",
     "            total[mod_name] = total.get(mod_name, torch.zeros_like(dW)) + wi * dW",
     "            total[mod_name] = total.get(mod_name, torch.zeros_like(dW)) + dW"),
    # ------------------------------------------------------- the partition and the control --
    (PARTITION, [T_PART], "the control stops permuting, so it reproduces the method exactly",
     "    permuted = tuple(int(v) for v in rng.permutation(labels))",
     "    permuted = tuple(int(v) for v in labels)"),
    (PARTITION, [T_PART], "the control samples labels instead of permuting them, losing the size match",
     "    permuted = tuple(int(v) for v in rng.permutation(labels))",
     "    permuted = tuple(int(v) for v in rng.choice(labels, size=len(labels)))"),
    (PARTITION, [T_PART], "the control's own size-match assertion never fires",
     "    if out.size_multiset() != reference.size_multiset():", "    if False:"),
    (PARTITION, [T_PART], "a single-task batch is reported as a cross-task calibration",
     "    if len(distinct) < 2:", "    if False:"),
    (PARTITION, [T_PART], "churn is keyed on batch position, which is reshuffled every step",
     "    before = dict(zip(previous.group_ids, previous.keys))\n    after = dict(zip(current.group_ids, current.keys))",
     "    before = dict(enumerate(previous.keys))\n    after = dict(enumerate(current.keys))"),
    (PARTITION, [T_PART], "no overlap is reported as full overlap with zero churn",
     "    if not overlap:\n        return 0.0, 0, 0", "    if not overlap:\n        return 0.0, 0, len(after)"),
    (PARTITION, [T_PART], "churn against a partition with no ids is accepted",
     "        if p.n_groups and not p.group_ids:", "        if False:"),
    (PARTITION, [T_PART], "adapters sort lexically, so cluster_10 lands between 1 and 2",
     "        named = sorted(\n            {v for v in self.labels if v != -1}\n        )",
     "        named = sorted(\n            {v for v in self.labels if v != -1}, key=str\n        )"),
    (PARTITION, [T_PART], "a second noise label is accepted and gets a private adapter",
     "    if label < -1:", "    if False:"),
    (PARTITION, [T_PART], "an empty batch is partitioned instead of refused",
     "    if n_groups <= 0:", "    if False:"),
    (PARTITION, [T_PART], "a partition with no features silently becomes one adapter for everything",
     "    if features is None or partitioner is None:", "    if False and features is None:"),
    (PARTITION, [T_PART], "capacities that do not partition the batch are accepted",
     "    if sum(cap) != n_rows:", "    if False:"),
    (PARTITION, [T_CLUST], "the expert-identity resync is disabled, the naive MEDS behaviour",
     "        state = getattr(self.clusterer, \"_state\", None)",
     "        return\n        state = getattr(self.clusterer, \"_state\", None)"),
    (PARTITION, [T_CLUST], "overlap matching takes the SMALLEST overlap first",
     "        for (raw, stable), _n in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0])):",
     "        for (raw, stable), _n in sorted(overlap.items(), key=lambda kv: (kv[1], kv[0])):"),
    (PARTITION, [T_ANALYZE], "the feature partition ignores its size targets",
     "        d = np.linalg.norm(unit[:, None, :] - km.cluster_centers_[None, :, :], axis=2)\n        labels = tuple(int(v) for v in balanced_assign(d, caps))",
     "        labels = tuple(int(v) for v in km.labels_)"),
    # ------------------------------------------------------------------------ the sketch --
    (SKETCH, [T_SKETCH], "every parameter shares one hash, so different tensors alias",
     "    rng = np.random.default_rng((self.seed ^ _SEED_SALT ^ digest) & ((1 << 63) - 1))",
     "    rng = np.random.default_rng((self.seed ^ _SEED_SALT) & ((1 << 63) - 1))"),
    (SKETCH, [T_SKETCH], "the random signs are dropped, so the sketch is a sum not a projection",
     "        np.add.at(out, idx, sign * flat.astype(dtype, copy=False))",
     "        np.add.at(out, idx, flat.astype(dtype, copy=False))"),
    # T_PROBE as well as T_SKETCH: the agreement test inside T_SKETCH importorskips torch,
    # so on the analysis interpreter it SKIPS and the mutation would be reported as surviving
    # a test that never ran. Naming a torch-only file makes the mutation not-applicable there
    # instead, which is the honest label.
    (SKETCH, [T_SKETCH, T_PROBE], "the torch path drops the signs, diverging from the numpy one",
     "        contrib.index_add_(0, idx_t, sign_t * flat.to(torch.float64))",
     "        contrib.index_add_(0, idx_t, flat.to(torch.float64))"),
    (SKETCH, [T_SKETCH], "a layout change between groups is accepted, so sketches stop being comparable",
     "            if cached[0].shape[0] != length:", "            if False:"),
    (SKETCH, [T_SKETCH], "a NaN gradient is sketched instead of refused",
     "        if not np.isfinite(flat).all():", "        if False:"),
    (SKETCH, [T_SKETCH], "the resolution floor shrinks by a factor of the dimension",
     "    return float(n_sigma / math.sqrt(dim))", "    return float(n_sigma / dim)"),
    # -------------------------------------------------------------------------- the dump --
    (DUMP, [T_PROBE], "the response mask is off by one, mixing the prompt into the RL loss",
     "    resp_mask = valid & (pos >= n_prompt - 1)", "    resp_mask = valid & (pos >= n_prompt)"),
    (DUMP, [T_PROBE], "the GRPO loss is normalised per group, breaking the sum identity",
     "    grpo = -sums[0] / token_denominator",
     "    grpo = -sums[0] / float(resp_mask.sum().clamp(min=1))"),
    (DUMP, [T_PROBE], "the prompt loss is normalised per group, breaking the same identity",
     "    prompt_nll = -sums[1] / prompt_denominator",
     "    prompt_nll = -sums[1] / float(prompt_mask.sum().clamp(min=1))"),
    (DUMP, [T_PROBE], "advantages are raw rewards, so a unanimous group gets a direction",
     "        return centred / sd if sd > 0 else centred", "        return r"),
    (DUMP, [T_PROBE], "a zero denominator is accepted",
     "    if token_denominator <= 0 or prompt_denominator <= 0:", "    if False:"),
    (DUMP, [T_PROBE], "the full-gradient size limit never fires",
     "    if gb > cfg.max_full_grad_gb:", "    if False:"),
    (DUMP, [T_PROBE], "a model with no LoRA parameters is accepted",
     "    if not blocks:", "    if False:"),
    (DUMP, [T_PROBE], "a reward list of the wrong length is accepted",
     "            if len(rewards) != len(resp_list):", "            if False:"),
    (DUMP, [T_PROBE], "a record with no reward silently gets one",
     "            if rk is None or rk not in rec:", "            if False:"),
    (DUMP, [T_PROBE], "the prompt NLL is masked to the response, so ELREA reads the RL gradient",
     "    prompt_mask = valid & (pos < n_prompt - 1)", "    prompt_mask = resp_mask"),
    # ------------------------------------------- the OOM fix: chunking and its budget guard --
    (DUMP, [T_PROBE], "chunking is disabled, restoring the full-vocabulary allocation that OOMed",
     "    chunk = max(1, min(int(chunk_tokens), n))", "    chunk = n"),
    (DUMP, [T_PROBE], "the budget guard is removed, so the run dies inside the LM head again",
     """    assert_logits_fit(
        group_id=group.group_id, n_tokens=len(seqs) * width, vocab=vocab,
        chunk_tokens=plan, budget_bytes=logits_budget_bytes, free_bytes=free_bytes,
        peak_bytes=peak_bytes_per_element,
    )""",
     "    pass"),
    (DUMP, [T_PROBE], "the budget ceiling never fires",
     "    if peak > budget_bytes:", "    if False:"),
    (DUMP, [T_PROBE], "the free-memory check never fires",
     "    if free_bytes is not None and peak * headroom > free_bytes:", "    if False:"),
    (DUMP, [T_PROBE], "the chunk ignores the vocabulary, so the same setting means a different memory",
     "    n = int(budget_bytes) // per_token", "    n = int(budget_bytes)"),
    (DUMP, [T_PROBE], "a budget too small for one token is rounded up instead of refused",
     "    if n < 1:", "    if False:"),
    (DUMP, [T_PROBE], "the chunk is not capped at the work available, so the recorded plan is fiction",
     "    if cap is not None:", "    if False:"),
    (DUMP, [T_PROBE], "the peak estimate drops the vocabulary factor",
     "    return int(chunk_tokens) * int(vocab) * int(peak_bytes)",
     "    return int(chunk_tokens) * int(peak_bytes)"),
    (DUMP, [T_PROBE], "active dropout is accepted, so a recomputed forward is differentiated",
     "    if active:", "    if False:"),
    (DUMP, [T_PROBE], "the reduction falls back to bf16, moving every log-probability",
     "    logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)",
     "    logp = torch.log_softmax(logits, dim=-1)"),
    (DUMP, [T_PROBE], "a model with no decoder trunk is accepted instead of refused",
     "    if decoder is None or unembed is None:", "    if False:"),
    (DUMP, [T_PROBE], "the emitting positions are off by one against the targets",
     "        hidden[:, :-1, :].reshape(-1, hidden.shape[-1]),",
     "        hidden[:, 1:, :].reshape(-1, hidden.shape[-1]),"),
    (DUMP, [T_PROBE], "the two losses share one weight vector, so ELREA reads the RL weights",
     "        (w_grpo, w_nll),", "        (w_grpo, w_grpo),"),
    (DUMP, [T_PROBE], "a sub-batch recomputes its advantages, centring the rewards within the slice",
     "    adv = group.advantages() if _advantages is None else _advantages",
     "    adv = group.advantages()"),
    (DUMP, [T_PROBE], "sub-batching is disabled, restoring the trunk activation cost",
     "    seq_chunk = max(1, min(int(seq_chunk), group.size))",
     "    seq_chunk = group.size"),
    (DUMP, [T_PROBE], "only the first sub-batch is backwarded, so most of the group is dropped",
     "        total += float(loss.detach())", "        total += float(loss.detach()); break"),
    (DUMP, [T_PROBE], "the sub-batch keeps the whole group's responses, double counting them",
     "            response_ids=group.response_ids[start:stop], rewards=group.rewards[start:stop],",
     "            response_ids=group.response_ids, rewards=group.rewards,"),
    (DUMP, [T_PROBE], "the activation budget ignores depth, so a deep model is priced as a shallow one",
     "    per_token = int(n_layers) * int(hidden) * 2", "    per_token = int(hidden) * 2"),
    (DUMP, [T_PROBE], "an unknown loss name is accepted and silently backwards the RL loss",
     '    if which not in ("grpo", "nll"):', "    if False:"),
    # ------------------------------------ the guard advice, and the budget it derives -------
    (DUMP, [T_PROBE], "the free-memory branch tells the reader to RAISE the budget, the real bug",
     '                f"LOWER the budget: set --logits-budget-gb {_gib(fit_budget):.1f} or less "',
     '                f"Lower --chunk-tokens, raise --logits-budget-gb "'),
    (DUMP, [T_PROBE], "the satisfying plan multiplies by the headroom instead of dividing",
     "    max_peak = int(free_bytes / headroom)", "    max_peak = int(free_bytes * headroom)"),
    (DUMP, [T_PROBE], "the suggested chunk drops the vocabulary factor",
     "    return max_peak, int(max_peak // (int(vocab) * int(peak_bytes)))",
     "    return max_peak, int(max_peak // int(peak_bytes))"),
    (DUMP, [T_PROBE], "the suggestion is rounded rather than floored, so it does not satisfy",
     "    return math.floor(n / 1024**3 * 10) / 10", "    return round(n / 1024**3, 1)"),
    (DUMP, [T_PROBE], "the suggestion is printed in decimal GB but the flag reads GiB",
     "    return math.floor(n / 1024**3 * 10) / 10", "    return math.floor(n / 1e9 * 10) / 10"),
    (DUMP, [T_PROBE], "an explicit budget is silently overridden by the derived one",
     "    if requested_gb is not None:", "    if False:"),
    (DUMP, [T_PROBE], "the budget ignores free memory and always takes the ceiling",
     "    budget = min(int(ceiling_bytes), max(0, fit))", "    budget = int(ceiling_bytes)"),
    (DUMP, [T_PROBE], "the headroom is dropped, so the guard and the derivation both under-price",
     "DEFAULT_HEADROOM = 1.5", "DEFAULT_HEADROOM = 1.0"),
    (DUMP, [T_PROBE], "a card too full for one token is given a chunk of zero as advice",
     "        if fit_chunk < 1:", "        if False:"),
    # ---------------------------------------------------------------------- the analysis --
    (ANALYZE, [T_ANALYZE], "a cluster's gradient overwrites instead of summing its members",
     "        out[key] = out[key] + vec if key in out else vec.astype(np.float64).copy()",
     "        out[key] = vec.astype(np.float64).copy()"),
    (ANALYZE, [T_ANALYZE], "zero-gradient clusters are counted as orthogonal pairs",
     "    live = [n for n in names if norms[n] > 0.0]", "    live = list(names)"),
    (ANALYZE, [T_ANALYZE], "an empty conflict rate is reported as zero conflict",
     '        "conflict_rate": (\n            float(np.mean([c < 0 for c in cosines])) if cosines else None\n        ),',
     '        "conflict_rate": (\n            float(np.mean([c < 0 for c in cosines])) if cosines else 0.0\n        ),'),
    (ANALYZE, [T_ANALYZE], "an unresolvable cosine is reported as resolved",
     '        "resolved": (mean is not None and abs(mean) > floor),',
     '        "resolved": (mean is not None),'),
    (ANALYZE, [T_ANALYZE], "the bootstrap stops resampling, so every interval is degenerate",
     "        idx = rng.integers(0, n, size=n)", "        idx = np.arange(n)"),
    (ANALYZE, [T_ANALYZE], "the ELREA partition is no longer size-matched to MEDS",
     "            prompt_sketches, n_clusters=len(sizes), match_sizes=sizes,",
     "            prompt_sketches, n_clusters=len(sizes), match_sizes=None,"),
    (ANALYZE, [T_ANALYZE], "the mean cosine is reported without the pairs it came from",
     '        "mean_cosine": float(np.mean(cosines)) if cosines else None,',
     '        "mean_cosine": 0.0,'),
]


def run_tests(tests) -> bool:
    """True if every named test file passes inside the copy."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=2400, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless pytest imports the COPY, not the live checkout."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.cluster_lora.adapters as m; print(m.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=300,
    )
    got = pathlib.Path(r.stdout.strip()).resolve()
    if got != ADAPTERS:
        raise SystemExit(f"ISOLATION FAILED: imports resolve to {got}, not {ADAPTERS}")
    print(f"isolated: imports resolve to {got}")


def main() -> int:
    """Apply each applicable mutation, run its tests, restore, and report the kill table."""
    if ARM not in ARMS:
        raise SystemExit(f"unknown arm {ARM!r}; expected one of {sorted(ARMS)}")
    runnable = set(ARMS[ARM])
    _assert_isolated()

    targets = {ADAPTERS, MERGE, PARTITION, SKETCH, DUMP, ANALYZE}
    original = {f: f.read_text() for f in targets}
    digests = {f: hashlib.sha256(t.encode()).hexdigest() for f, t in original.items()}

    applicable = [m for m in MUTATIONS if set(m[1]) <= runnable]
    skipped_arm = len(MUTATIONS) - len(applicable)
    baseline_tests = sorted({t for m in applicable for t in m[1]})
    if not run_tests(baseline_tests):
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"arm {ARM}: baseline green; {len(applicable)} mutations "
          f"({skipped_arm} not applicable to this interpreter)\n")

    survivors = []
    for target, tests, label, find, repl in applicable:
        src = original[target]
        if src.count(find) != 1:
            print(f"SKIP      {label}: anchor appears {src.count(find)}x")
            survivors.append((label, f"anchor appears {src.count(find)}x -- NOT a kill"))
            continue
        target.write_text(src.replace(find, repl, 1))
        try:
            passed = run_tests(tests)
        finally:
            target.write_text(src)
            assert hashlib.sha256(target.read_text().encode()).hexdigest() == digests[target], \
                f"restore of {target} failed"
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    for f in targets:
        assert hashlib.sha256(f.read_text().encode()).hexdigest() == digests[f], \
            f"{f} was left modified"
    print(f"\n{len(applicable) - len(survivors)}/{len(applicable)} killed on arm {ARM}")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
