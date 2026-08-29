# Benchmark and baseline inventory

Everything below was read off the pinned clones in `~/baselines/` on the A100 host, not
from papers or memory. Commits are pinned in `~/LAYOUT.md`.

## Harnesses that already exist (do not rebuild)

| Repo | Commit | License | Eval entry point |
|---|---|---|---|
| BigBang-v1 | 5128884 | LICENSE present | `scripts/run_eval.py`, `evaluation/benchmarks/registry.py` |
| RAGEN | d97bb32 | LICENSE present | `ragen/eval.py`, `scripts/eval_batch.sh`, `docs/eval.md` |
| Absolute-Zero-Reasoner | 484afa4 | LICENSE present | `evaluation/math_eval/` (`run.sh`, `eval_math_nodes.sh`), `evaluation/code_eval/` |
| R-Zero | 5699329 | **NO LICENSE** | `evaluation/evaluate.bash`, `eval_bbeh.py`, `eval_supergpqa.py` |
| MEDS | 4f46841 | LICENSE present | `recipe/` (verl-based) |
| autoresearch | 228791f | **NO LICENSE** | `train.py`, `prepare.py`, `analysis.ipynb` |

**Licensing rule.** The two NO-LICENSE repos (R-Zero, autoresearch) are cloned to *run and
cite*. Their code is never copied into `~/areal-selfevo` or anything we publish. Default
copyright means no redistribution right, regardless of public visibility on GitHub.

## BigBang-v1's own benchmark set (from `evaluation/benchmarks/registry.py`)

Registered keys, verified by grep of the `BENCHMARKS` dict:

- `BrowseComp` / `browsecomp`
- `FrontierScience-Research` / `frontierscience`
- `SciCode-Verified` / `scicode-verified` / `scicode_verified`
- `hle` (Humanity's Last Exam)
- `swebench-pro`
- `xbench-DeepSearch-2510` / `xbench`

Note this set is **agentic search + science + SWE**, not the frontier *math* suite. Any
claim that we "match BigBang's benchmarks" has to mean this list. A separate math suite
(AIME/MATH-500 style) comes from the AZR `math_eval` harness, not from BigBang.

The registry is dispatched through `BenchmarkSpec` with a `harness` field;
`BIGBANG_HARNESS_BENCHMARKS` filters to those with `harness == "bigbang"`, so some entries
are delegated to external harnesses. Confirm per-benchmark before claiming coverage.

## Still to verify before any of this enters a paper table

- Which registry entries actually run end-to-end without private data or an API key.
  `BrowseComp`, `xbench-DeepSearch`, and `FrontierScience-Research` are search benchmarks
  and likely need live web access or a paid search API.
- `hle` and `swebench-pro` are gated/large; check `evaluation/benchmarks/download.py` for
  what it fetches and how big.
- `judges.py` implies LLM-as-judge scoring for at least some benchmarks — that is a paid
  API dependency and a confound. Identify which benchmarks are judge-scored and which are
  exact-match before reporting any number.

## Regimes the user asked for, and their status

| Regime | Source | Status |
|---|---|---|
| Frontier agentic/science/SWE | BigBang registry above | harness cloned, not yet run |
| Math | AZR `evaluation/math_eval` | harness cloned, not yet run |
| Code | AZR `evaluation/code_eval` | harness cloned, not yet run |
| Science QA | GeneBench-Pro, BioMysteryBench | local material is READ-ONLY; outputs go to a new folder |
| Enterprise SQL | Spider 2.0, DABstep, BIRD | not yet cloned |
| Search | Search-R1 suite | not yet cloned |
| Agent RL | RAGEN | cloned, MIT, runnable as a real baseline |

## VERIFIED: both science benchmarks can score a locally served model (no paid API)

Local material at `~/text-dna/genebench-pro/` is **READ-ONLY**. It holds two runnable
harnesses; all our outputs go to a separate folder via their `--out` flags.

- **BioMysteryBench** (`bio-mystery/bmb/`): `runner.py --backend served` with
  `--base-url` (OpenAI-compatible root, e.g. `http://host:8404/v1`). `served_backend.py`
  posts to `base_url + "/chat/completions"`. Backends: `codex|claude|served|biomni`.
  Subsets: `all|solvable|unsolvable`. `--max-steps` default 25.
- **GeneBench-Pro** (`genebench-pro/gbp/`): backends are `openrouter|codex|biomni`, but the
  `openrouter` path takes `--base-url` and `runner.py:190` sets
  `needs_key = args.backend == "openrouter" and not args.base_url` — so a local endpoint
  needs no key. `agent.py:89` documents the intent: *"Defaults to OpenRouter; point it at a
  local vLLM."* `runner.py` passes `base_url=args.base_url` into `run_attempt`.

Consequence: the science regime runs entirely on our 8xA100 against our own sglang/vLLM
endpoint. No OpenRouter spend, and no dependence on a third-party model's availability for
the numbers that go in the paper.

Caveat to check before reporting: BioMysteryBench has a `judge.py` (LLM-as-judge). Confirm
whether the judge can also be served locally, or whether judged scores carry an API
dependency and a confound. GeneBench-Pro has `grading.py`, which looks deterministic.

### Resolved: the BioMysteryBench judge is NOT locally servable
`bmb/judge.py:233` shells out to the **Codex CLI** (`-m <judge_model> -c
model_reasoning_effort=...`), not to an OpenAI-compatible HTTP endpoint. So:

- **Solving** runs locally (`--backend served --base-url <our endpoint>`), free.
- **Judging** goes through Codex (ChatGPT Pro subscription, which is permitted; not
  metered OpenRouter spend). Default `--judge-model gpt-5.6-sol`, `--judge-effort medium`.

`judge.py:22-24` states it directly: *"Reported scores therefore depend on the judge model,
which must be stated alongside any number. Anthropic graded with their own models; we use
whatever `--judge-model` names. Numbers are not exchangeable across judges."* Every
BioMysteryBench number we publish must name its judge model and effort.

`runner.py` already excludes attempts with `judge_error` from the denominator (line 61,
tracked in `excluded`). That is the right behaviour, but it means a judge outage silently
shrinks n — check the `excluded` counts on every run rather than trusting the headline.
GeneBench-Pro's `grading.py` is deterministic and carries no such dependency.
