# Base-model policy for formal training runs

Decision (2026-08-29): **formal runs use a frontier base, not Qwen2.5-1.5B/7B.**

Everything so far ran on Qwen2.5-1.5B-Instruct because that is what AReaL's published GSM8K
config specifies, and reproducing it was the goal. It is the wrong substrate for a paper:
it solves ~76% of GSM8K before training, so any method effect lands inside the step-to-step
noise band.

## Preferred bases, in order

| model | HF id | params | why |
|---|---|---|---|
| Ornith-1.5-35B-A3B | `ornith-ai/Ornith-1.5-35B-A3B` | 35B total / ~3B active (MoE) | self-improving coding model; published numbers to compare against directly |
| BigBang-v1 | `endless-frontier/BigBang-v1` | 35B-A3B | the self-evolving-synthesis model this work positions against; its own benchmark table is in the frozen set |
| Ornith-1.5-9B | `ornith-ai/Ornith-1.5-9B` | 9B | cheapest frontier option; Terminal-Bench 2.1 46.2, SWE-bench Verified 70.6 |
| Qwen2.5-7B-Instruct | cached locally | 7B | fallback / smoke tests only |

A ~35B MoE with ~3B active fits comfortably on 8xA100-80GB; the 9B fits trivially. The
397B Ornith variant does not.

**Not verified yet:** none of these have been downloaded or served here. "Qwen3.8" was
mentioned as a candidate but I could not confirm such a release exists, so it is not listed
rather than guessed at.

## Why this matters for the claim

Beating SOTA requires starting from a model where SOTA is defined. BigBang-v1 and
Ornith-1.5 both publish numbers on benchmarks in our frozen set (BioMysteryBench,
SWE-Bench Pro, HLE, BrowseComp for BigBang; Terminal-Bench, SWE-bench, LiveCodeBench-adjacent
for Ornith), so training from one of them makes the comparison direct rather than
apples-to-oranges.

## Measured floor to beat, on the substrate we can already score

Qwen2.5-7B-Instruct, scored here with `experiments/bench/math_bench.py`:

| benchmark | accuracy |
|---|---|
| AIME 2024 | 10.0% +/- 5.6 |
| AIME 2025 | 3.3% +/- 3.3 |
| AMC 23 | 57.5% +/- 7.9 |
| MATH-500 | 75.0% +/- 1.9 |

AIME is the useful axis: at 3-10% there is real headroom, unlike GSM8K at 76%.
