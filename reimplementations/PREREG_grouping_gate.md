# Pre-registration: does finer grouping identify weakness at all?

Written 2026-09-04, **before computing any of the statistics below**, and before the probe
widening now in flight has finished. Nothing here has been seen.

Pre-registered because the two components this project has retired — a learned router and the
MEDS clustering — were both retired by a stop rule written in advance, and both failed at the
same point: a learned structure that made real distinctions turned out to make them no better
than a size-matched random structure of the same shape. Fine-grained weakness grouping is at
exactly that stage, and it is cheap to test one step before a training run rather than after.

## The claim under test

A curriculum that targets weakness needs groups whose weakness actually differs. If a
grouping's per-group accuracies are no more spread out than those of a random partition with
the same group sizes, **the grouping has identified nothing**, and any downstream targeting
gain would have to come from somewhere else.

This is NOT the claim MEDS was retired for. MEDS failed at "do behavioural clusters have less
gradient conflict than a size-matched random partition". This asks whether a grouping
identifies differential weakness. Different question, same machinery, and the write-up must
say so rather than read as a rehabilitation of a closed result.

## Statistic

Weighted between-group variance of per-group accuracy:

    V = sum_g (n_g / N) * (acc_g - acc_pool)^2

where `acc_g` is the mean per-problem success rate `p_hat` over the problems in group g,
`n_g` its size, `N` the total, and `acc_pool` the pooled mean. Per-problem `p_hat` comes from
the widened probe and is fixed across everything below.

## Null

Permutation. Each null draw keeps the per-problem `p_hat` values fixed and **reassigns
problems to groups uniformly at random, preserving the observed group-size multiset**. This
matters: small groups inflate observed variance through sampling noise, and a size-matched
permutation null carries exactly the same inflation, so the comparison is against noise of the
right shape rather than against zero.

* **R = 2000 permutations** per arm.
* One-sided p = fraction of null draws with `V_null >= V_observed`.
* Effect size reported as `V_observed / median(V_null)`.
* Groups with fewer than 5 problems are pooled into a single `other` group **before** any
  statistic is computed. Declared now so it cannot be tuned later.
* Each arm gets its OWN size-matched null, because the arms have different group-size
  multisets and a shared null would compare them against the wrong reference.

## Arms

1. **coarse** — the four OlympiadBench subfields we already have.
2. **fine-label** — finer topic labels assigned by the model (e.g. pigeonhole, generating
   functions, modular arithmetic).
3. **cluster** — embedding-based clusters, k chosen to match the fine-label group count.

## Threshold and decisions, fixed in advance

* An arm **passes** if p < 0.05.
* **fine passes and exceeds coarse** (higher V, and coarse's V outside fine's permutation
  interval): finer grouping helps; proceed to weakness-ranked generation on fine labels.
* **fine passes but does not exceed coarse**: the finer machinery is unnecessary complexity;
  proceed on the coarse labels we already have and say so.
* **coarse passes, fine does not**: the coarse labels carry the signal; proceed on coarse.
* **neither passes**: fine granularity identifies nothing on this pool. Do **not** build
  weakness-targeted generation here. Report it as a negative — it is cheap, it is publishable
  given the router and the clustering, and it would be the third component retired by this
  same test.
* **cluster passes but fine-label does not**: prefer the labels anyway unless cluster's effect
  size exceeds fine-label's by more than 2x, for the auditability reason below.

## Why labels are preferred over cluster indices at equal performance

Not interpretability, which is a convenience. **Auditability, which is a safety property.** If
the curriculum claims the model is weak at pigeonhole arguments, an auditor can verify that the
exemplars shown were pigeonhole problems drawn from the training half. A cluster index gives
an auditor nothing to check against, so a leak from the probe or the evaluation half into the
exemplar pool would be much harder to detect — and that leak is precisely what the audit
exists to catch.

## What this gate does not establish

Passing it means groups differ in weakness. It does **not** mean targeting the weak ones
improves held-out capability; that is the next experiment and it keeps its own control
(weakness-ranked generation against random-category generation at the treatment's own realised
proportions, matched on budget, scored on held-out capability). A grouping can be real and
still not matter, which is the outcome three components in this project have already produced.
