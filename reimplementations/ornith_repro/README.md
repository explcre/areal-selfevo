# `ornith_repro` — a reimplementation of Ornith-1.5's self-improvement loop

**This is a REIMPLEMENTATION of a described method, not a reproduction of Ornith-1.5.**
Read `AMBIGUITIES.md` before reading any number produced here. Nothing in this directory
is part of our own method; it exists so that we have a baseline we can actually run, and
an honest statement of what of Ornith's description survives reconstruction.

## Why a reproduction is impossible

Ornith-1.5 publishes explicit reward formulas on its method page, so the *method* is
implementable. What is not available, established by a source-level audit of the release
on 2026-08-19 (`self-evo-agent/codex_docs/19-ornith-diagnosis--*`) and unchanged since:

* `ornith-ai/Ornith-1` contains a README, a LICENSE, a .gitignore and four images. No
  source file, config, log or training loop exists in any of its commits.
* No task buffer, per-task outcome, harness, rollout trace, reward, advantage, ablation,
  raw benchmark output, compute ledger, or trainable config was released.
* The README documents unpublished modifications to its serving stack and chat template,
  and two of its benchmarks are judged by a proprietary model.
* Weights are 9B, 35B-A3B and 397B. **There is no 32B model**, so scale matching was
  never possible. Their bases were Qwen3.5 and Gemma 4; ours is a deliberate substitution.

## No head-to-head exists, and we must not imply one

Everything Ornith reports is agentic or repository-scale coding: Terminal-Bench 2.1,
SWE-Bench Verified/Pro/Multilingual under OpenHands, DeepSWE and NL2Repo under a
proprietary harness, at 128K–400K context with four-hour timeouts. We do not have those
scaffolds. Ornith reports **no competition mathematics, no LiveCodeBench and no SQL**;
our project reports exactly those. **The benchmark intersection is empty.** Any table
placing our numbers beside theirs is one we constructed and must be labelled as such.

## The loop, as published

```
R_task        = V(q,s) * D(q,s,{tau_i}) * N(q)
D             = exp(-(p - p*)^2 / (2 sigma^2)),   p* = 0.2      <- the ONLY published constant
N(q)          = 1 - max_{q_j in B} sim(q, q_j)
R_harness     = C(q,h) * F(h,{tau_i}) * H(h)                    alignment, fidelity, hack resistance
R_rollout     = h(q, tau_i)
p             = (1/N) sum_i 1[s(q, tau_i) = success]
```

Three sequential stages per iteration — task generation, scaffold generation, rollout —
each optimised with GRPO on its own reward, with reward propagated across all three.
`sigma`, the rollout count, `sim`, and the definitions of `V, C, F, H` are **not
disclosed**; each is our choice, with an id, in `AMBIGUITIES.md`.

## Layout

| file | what |
|---|---|
| `rewards.py` | the published equations, verbatim, nothing added |
| `grpo.py` | group advantages; degenerate groups surfaced, not hidden |
| `guards.py` | the silent-zero guards (G1–G7), each mutation-tested |
| `loop.py` | one iteration: three stages, artifacts, provenance |
| `buffer.py` | the task buffer `B` |
| `judges.py` | V, C, F, H rubrics — **the largest reconstruction gap** |
| `controls.py` | size-matched random controls at measured proportions |
| `llm.py` | base model as a PARAMETER; deterministic stub for CPU work |
| `tests/` | 48 tests: one-sample E2E, guards firing, silent-zero paths |

## Running

```bash
python -m pytest ornith_repro/tests -q      # 48 tests, CPU only, no GPU, seconds
python mutate.py $(which python)            # 12 mutations, all must be caught
python gate_selection.py --k 8 --sigma 0.15 # the CPU measurement
```

## Base model

`OrnithConfig.base_model` is a parameter, never a constant. Target `Qwen/Qwen3.8-27B`;
fallback `Qwen/Qwen3-32B`; last resort `Qwen2.5-32B-Instruct`. Whether Qwen3.8-27B loads,
serves and trains in this stack is being determined separately; **nothing here depends on
that answer**, because every CPU-side assertion runs against the deterministic stub
client. Whichever base actually runs must be stated in the write-up: that choice is part
of the result.

## Correctness discipline

* Every guard has a **mutation test** that makes it fire. A guard that has never been seen
  to fail is not evidence.
* Every assertion is on **observable state** — a field in a record, or a value read back
  off disk — never on a function returning True.
* `mutate.py` applies 12 deliberate defects; the suite must go red for all 12. It found a
  real hole on first run (the *default* abort policy was untested because every test
  passed it explicitly), which is now closed.
