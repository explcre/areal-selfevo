# Frozen benchmark set

Frozen 2026-08-29. Nothing is added after this point without recording why, so the
evaluation suite cannot drift toward whatever our method happens to do well on.

Status is deliberately blunt. **RUNNABLE** means it has been shown to execute end to end
here; **CLONED** means the harness is on disk and has never been run; **EXTERNAL** means we
have neither harness nor data. Several rows that look impressive are CLONED.

## Where each set comes from

| source | how the list was obtained |
|---|---|
| BigBang-v1 | `evaluation/benchmarks/registry.py` + `assets/bigbang_main_figure.png` |
| Ornith-1.5 | HuggingFace model card for `ornith-ai/Ornith-1.5-35B-A3B` |
| EvoTrainer | arXiv 2606.03108 tables (PDF text extraction) |
| ours | already present locally from earlier projects |

## The set

### Science / research
| benchmark | from | license | status |
|---|---|---|---|
| **BioMysteryBench** (Human-Solvable, Human-Difficult) | BigBang, ours | local | **RUNNABLE** |
| **GeneBench-Pro** | ours | local, 10 public problems | **RUNNABLE** |
| FrontierScience-Research | BigBang | registry | CLONED |
| HLE (no tools / with tools) | BigBang, Ornith | registry | CLONED |
| GPQA Diamond | Ornith | — | EXTERNAL |

### Code / SWE
| benchmark | from | license | status |
|---|---|---|---|
| SWE-Bench Pro | BigBang, Ornith | registry | CLONED |
| SWE-bench Verified / Multilingual | Ornith | — | EXTERNAL |
| SciCode-Verified | BigBang | registry (Apache-2.0 data) | CLONED |
| Terminal-Bench 2.1 (Terminus-2, Claude Code) | Ornith | — | EXTERNAL |
| DeepSWE, NL2Repo, SWE Atlas QnA, Frontier-Bench v0.1, ClawEval | Ornith | — | EXTERNAL |
| LiveCodeBench-v6 | EvoTrainer | — | EXTERNAL |
| AZR `code_eval` | ours | MIT | CLONED |

### Math
| benchmark | from | license | status |
|---|---|---|---|
| AIME 2024, AIME 2025 | EvoTrainer | — | EXTERNAL |
| AZR `math_eval` | ours | MIT | CLONED |
| GSM8K | AReaL Step 0 | in-repo | **RUNNABLE** (and saturated -- see below) |

### Search / browsing
| benchmark | from | license | status |
|---|---|---|---|
| BrowseComp | BigBang, Ornith | registry | CLONED |
| xbench-DeepSearch-2510 | BigBang | registry | CLONED |
| WideSearch | Ornith | — | EXTERNAL |
| Search-R1 suite (NQ/TriviaQA/HotpotQA/…) | ours | LICENSE | CLONED |

### Enterprise SQL / data query
| benchmark | from | license | status |
|---|---|---|---|
| Spider 2.0 (`spider2-lite`, `-dbt`, `-snow`) | ours | LICENSE | CLONED |
| BIRD `mini_dev` | ours | **NO LICENSE** | CLONED, run+cite only |
| DAMO-ConvAI | ours | LICENSE | CLONED |
| DABstep | ours | — | **NOT OBTAINED** -- it is a HuggingFace dataset, not the GitHub path tried |

### Agentic / tool use
| benchmark | from | license | status |
|---|---|---|---|
| MCP-Atlas, Toolathlon-Verified | Ornith | — | EXTERNAL |
| MLE-Bench (Lite), PaperBench (Code-Dev) | BigBang | — | EXTERNAL |
| RAGEN (11 task configs) | ours | MIT | CLONED |

## Comparison points already published

BigBang-v1, from their figure -- these are the numbers to beat or contextualise:

| benchmark | BigBang V1 | best shown |
|---|---|---|
| BrowseComp | 76.5 | GPT 5.5 84.4 |
| MLE-Bench (Lite) | 59.1 | GLM 5.2 72.7 |
| BioMysteryBench (Human-Solvable) | 57.5 | GPT 5.5 76.7 |
| SWE-Bench Pro | 54.2 | GLM 5.2 62.1 |
| PaperBench (Code-Dev) | 53.6 | GPT 5.5 64.2 |
| HLE | 50.3 | GLM 5.2 54.7 |
| FrontierScience Research | 46.2 | GPT 5.5 58.3 |
| **BioMysteryBench (Human-Difficult)** | **15.7** | GPT 5.5 23.5 |

Ornith-1.5-397B: Terminal-Bench 2.1 **86.1**, DeepSWE **56.0** (vs Claude Opus 4.8 at 85.0 / 59.0).
Ornith-1.5-9B: Terminal-Bench 2.1 **46.2**, SWE-bench Verified **70.6**.

## Two cautions that belong with the set

**GSM8K is not an experimental substrate.** Qwen2.5-1.5B-Instruct already solves ~76% of
it, and our own Step 0 runs moved inside the step-to-step noise band. It stays in the set
only as a plumbing check.

**EvoTrainer's SWE numbers are not comparable to anything public.** Their evaluation is
"77 held-out Python instances" -- a custom set, not SWE-Bench. Their Math is AIME 2024 +
AIME 2025 + 78 competition problems, and Coding is LiveCodeBench-v6 on 175 held-out
problems. Cite their deltas, not their absolute scores, and never against a different set.

## Where headroom actually is

**BioMysteryBench (Human-Difficult)** is the strongest candidate for a real comparison:
it is in BigBang's published set, we hold it locally and it is RUNNABLE, and the scores are
low enough to move (Qwen3.6-35B 2.0, BigBang 15.7, GPT 5.5 23.5). Unlike GSM8K there is
room for a method to show something.
