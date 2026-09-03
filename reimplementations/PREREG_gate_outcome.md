# Pre-registration: does Ornith's difficulty gate improve self-improvement?

Written 2026-09-03, BEFORE any real-model run. Pre-registered because the two components
this project has retired (the learned router, the MEDS clustering) were both retired by a
stop rule written in advance, and both had reached exactly the stage the gate is at now:
**mechanism confirmed, outcome unmeasured.**

## The claim under test

Established already (Sec.~`sec:external`): the gate *selects what it says it selects*,
enriching tasks near the target difficulty at z=16.2 against a size-matched random control.
That is the first learned component here not to tie with its control.

NOT established: that this makes the model improve faster. "Selects harder tasks" and
"improves the model faster" are different claims and only the first is measured.

## The decomposition

    improvement per unit budget  =  (task quality)  x  (budget efficiency)

* **Budget efficiency** — settled CPU-side, see `gate_outcome.py` Part A. Under binary-reward
  GRPO, every non-degenerate group carries advantage energy exactly `G` (verified: mean
  7.999962 against G=8, max deviation 4.8e-05), and every degenerate group carries exactly
  0. So energy per rollout is exactly the non-degenerate fraction, with no learning
  assumptions at all.
* **Task quality** — NOT settleable CPU-side. See "why the simulation cannot answer this".

## Arms

| arm | selector |
|---|---|
| **T** treatment | full `R_task = V*D*N` gate |
| **C1** matched-difficulty random | random tasks resampled to T's **realised** difficulty distribution |
| **C2** band filter | keep `0.1 < p_hat < 0.9`, no target, no kernel — the cheap alternative |

C1 is the control that isolates the claim. C2 is included because the simulation suggests
much of the gate's benefit may be attributable to "avoid hopeless and trivial" rather than
to targeting 0.2, and a method that ties with a one-line filter is worth knowing about.

## Matching rules (these are what make the control unfoolable)

1. Match on **realised** difficulty, measured from T's own artifacts, **not** on the nominal
   `p* = 0.2`. The two differ: T's realised mean theta is 0.304, not 0.200.
2. Use the **fresh-block** difficulty as the reference (0.551 at k=8), **never** the
   selecting-block 0.946, or the control is matched to an inflated target.
3. Match **total token budget**, not step count — the arms differ in generation cost, and
   step-matching would silently hand one arm more compute.
4. Score on **held-out capability only**. No property of the selected tasks is an outcome.
   Train reward is not an outcome; this project has already shown it mis-orders checkpoints.

## Outcome and analysis

* Primary: held-out accuracy on the project's existing benchmarks, paired by item, McNemar.
* Report the **standard error on the DIFFERENCE** between arms, not per-arm error bars.
* Power: RLVR eval noise here is ~6 points at n=1, so a difference below that is not
  resolvable and will be reported as "not resolvable at this n", not as a null.
* Report the k-histogram of unanimous groups per arm. A batch that is mostly unanimous
  gives a false negative on any gradient statistic.
* Secondary (mechanistic, not a substitute): advantage energy actually delivered per token
  in each arm, which decomposes any outcome difference into "better tasks" vs "less waste".

## Stop rules, fixed in advance

* If **T vs C1** does not separate beyond the noise band, the gate **works and does not
  matter**, and that is the result. It is publishable given the router and the clustering,
  and it will be reported plainly rather than tuned until it separates.
* If **T vs C2** does not separate, the gate's benefit is attributable to a one-line filter
  and the kernel plus target is unsupported complexity. Report that too.
* No post-hoc changes to the difficulty target, sigma, or the control's matching key. Any
  change after seeing an outcome is a new experiment with a new pre-registration.

## Why the simulation cannot answer this, demonstrated rather than asserted

`gate_outcome.py` Part B runs five defensible learning rules against four controls.

1. **The specified control is vacuous in a difficulty-only simulation.** Against
   `matched_difficulty` the largest absolute difference across all five rules is **0.0115**,
   i.e. zero to within noise — necessarily, because theta is the entire task representation
   in the simulation, so matching it exhausts the task. A real run is not vacuous, because
   there the gate also selects on validity and novelty and the control does not.
2. **The sign flips with the assumed rule.** Under `learnability`, `frontier`, `zpd` and
   `lifetime` the gate beats uniform; under `success_teaches` (gain proportional to theta)
   it **loses**, by -0.0911 against uniform and -0.1547 against the band filter. Which rule
   holds is exactly what is unknown, so a cheap simulation would return the answer it was
   built to give.

Conclusion: the outcome question **must** be run on a real model. This file exists so that
when it is, the design is fixed beforehand.

## What the CPU-side work does establish (Part A, no learning assumptions)

| selector | mean theta | dead-group fraction | advantage energy / rollout |
|---|---|---|---|
| gate, realised | 0.304 | 0.1206 | **0.8794** |
| size-matched uniform | 0.481 | 0.2485 | 0.7515 |
| idealised at p*=0.2 | 0.200 | 0.1678 | 0.8322 |
| idealised at 0.5 | 0.500 | 0.0078 | 0.9922 |

Two corrections to how the waste figure has been quoted, including by us:

* The realised dead-group fraction is **0.1206, not 0.1678**. The 17% figure is the
  idealised value *at* p*=0.2, and the gate does not hit its target — realised mean theta is
  0.304. Quoting 17% overstates the waste by about 40% relative.
* Against **no selection at all** the gate is *more* budget-efficient (0.879 vs 0.752),
  because an unselected pool is full of near-0 and near-1 tasks that are far deader than
  anything near 0.2. The waste claim is only ever a comparison **between targets**
  (0.879 vs 0.992 at p*=0.5), never between gate and no gate.
