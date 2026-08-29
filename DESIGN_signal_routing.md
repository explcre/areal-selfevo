# Signal routing: choosing SFT / RL / distillation per unit

## The claim

Different data want different training signals, and the choice can be made from a quantity
we already compute for free. This is not a heuristic: for group-based RL there is an exact
condition under which the RL gradient is **identically zero**, and that condition is
measurable per prompt, per cluster, and (with one extra assumption) per token.

## 1. Unification: all three modes are per-token weights on log pi

Write every mode as

    L(theta) = - E_{(x,y) ~ q} [ sum_t  m_t * w_t * log pi_theta(y_t | x, y_<t) ]
               + beta * sum_t m_t * D_t(theta)

with `q` the proposal distribution the tokens came from, `m_t` the loss mask, `w_t` a
per-token scalar weight, and `D_t` a distributional term.

| mode | q | w_t | D_t |
|---|---|---|---|
| SFT | gold / teacher text | lambda_sft | 0 |
| GRPO / RLVR | pi_old (on-policy) | effective advantage (below) | 0 |
| hard distillation | teacher samples | lambda_kd | 0 |
| soft distillation | teacher or pi | 0 | KL(p_teacher \|\| pi_theta) |

**Stated precisely, because the losses are not literally identical.** PPO's clipped
objective is `min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)`, not `w_t log pi`. But its
*gradient* is `w_t * grad log pi` with `w_t = r_t * A_t` on the unclipped branch and
`w_t = 0` on the clipped branch. Since the gradient is what composes, the per-token weight
abstraction is exact at the level that matters. Hard distillation is SFT on teacher text --
they are the same estimator with a different `q`, and the framework should not pretend
otherwise. Soft distillation is genuinely different: it needs the full (or top-k) teacher
distribution, so it gets its own term rather than being forced into `w_t`.

Consequence for implementation: a router emits **per-token weights**, and AReaL already
carries per-token `advantages` and `loss_mask` tensors through `trainer/ppo/actor.py`. So
routing composes with the existing loss path instead of replacing it.

## 2. The routing criterion: when is RL provably silent?

For a group of `G` samples on prompt `x` with binary reward, GRPO's advantage is
`A_i = (r_i - rbar)` (optionally / sigma). If every `r_i` is equal, `A_i = 0` for all `i`
and the group contributes **exactly zero gradient**. Under `r_i ~ Bernoulli(p)`,

    P(silent group) = p^G + (1-p)^G

Define **RL informativeness**

    I_RL(p, G) = 1 - p^G - (1-p)^G

`I_RL` is 0 at `p in {0,1}` and maximal at `p = 1/2`. With `G = 4`: `I_RL(0.5) = 0.875`,
`I_RL(0.9) = 0.34`, `I_RL(0.99) = 0.039`. This is the same quantity behind the group-size
law already in the repo, `G >= 1/(8 eps p (1-p))`.

**The subtlety that makes this a design and not a threshold.** RL is silent at *both* ends,
but the two ends need opposite responses:

- `p ~ 0` -- the model cannot solve it. RL has nothing to push on. The unit needs an
  **external target**: SFT or distillation from a teacher. Value is high *if* a teacher
  target exists, and zero otherwise.
- `p ~ 1` -- the model already solves it. RL is silent because there is nothing left to
  learn. The correct action is to **spend less compute here**, not to add SFT (which would
  only sharpen an already-correct policy and burn entropy).

A router keyed on `I_RL` alone conflates these. So the decision is on the pair
`(I_RL, side)` where `side = sign(p - 1/2)`, plus whether a teacher target is available.

## 3. Granularity

`task | cluster | sample | token`, each estimating `p` at a different resolution:

- **task / cluster** -- `p` averaged over a dataset or a semantic cluster. Cheap, stable,
  coarse. Good default.
- **sample** (per prompt) -- `p_hat` is the group's mean reward, already computed. Free.
- **token** -- needs an extra assumption, stated below and marked as a hypothesis to be
  tested rather than an assertion.

**Token-level, honestly.** GRPO assigns one advantage per *sequence*, so every token in a
sequence shares `A_i`. Token-level routing therefore requires a claim about where within a
sequence the RL signal actually lives. The claim we will test:

> On a prefix shared by all group members, the net RL gradient is approximately zero,
> because each member contributes `A_i * grad log pi(shared token)` and `sum_i A_i = 0` by
> construction of the centered advantage.

If that holds, shared prefixes are RL-dead and are exactly where a teacher signal can be
added without fighting the RL gradient. This is checkable directly: compute the per-token
net gradient weight over a group and compare shared-prefix positions against post-divergence
positions. **It ships behind a flag and off by default until that test passes.** The
existing repo finding that 73.4% of step-level groups emit advantage exactly 0 is
consistent with it but is not the same measurement.

## 4. Module layout

    selfevo/
      signals/base.py       TrainingSignal, SignalSource protocol
      signals/rl.py         on-policy advantage source (wraps AReaL)
      signals/sft.py        gold / teacher-token source
      signals/distill.py    soft-target source
      routing/criteria.py   I_RL, silence side, group-size law  (pure, no torch)
      routing/routers.py    Static, SolveRate, Mixture routers
      combine.py            fold weighted signals into one loss
      config.py             dataclasses + registries

Extension points, so later modes need no edits to existing files:
- `SignalSource` is a Protocol; new modes register by name.
- `Router` is a Protocol returning a `RoutingDecision` (a mode -> weight mapping, so
  mixtures and soft routing are expressible, not just hard argmax).
- `Granularity` is an enum consumed by the router, not baked into call sites.
- A future **learned** router (an RL meta-policy over modes) implements the same Protocol;
  nothing else changes.

## 5. What is deliberately not built yet

- The learned meta-policy router. The static and solve-rate routers must be shown to beat
  a fixed-mode baseline first, or a learned router has nothing to improve on.
- Token-level routing is implemented but off by default, pending its own test.
- Soft distillation needs teacher logits plumbed through the rollout; the interface is
  defined, the transport is not.

## 6. How this gets falsified

The framework is only interesting if routing beats every fixed mode. The controls are:
1. all-RL, all-SFT, all-distill (fixed-mode baselines)
2. **random routing** at the same mode proportions as the learned/criterion router -- this
   is the control that catches "the gain came from mixing, not from choosing"
3. inverted routing (route to the mode the criterion says is worse) -- should be worse than
   random if the criterion carries signal

Without control 2 a routing gain is not attributable to the criterion. That control is
mandatory, not optional.
