"""Gold grounding for RL-silent groups, by batch construction.

On MATH, 25.5% of all groups are UNSOLVED: every rollout was wrong, so nothing correct of the
model's own exists to reinforce, and ``selfevo/tests/test_gold_target_reachability.py`` pins
that no router and no fixed rule can reach them -- an advantage is a coefficient on tokens the
model EMITTED, and in such a group every emitted token is wrong. The cheapest supplier of a
correct target is the dataset's own gold solution, and the only altitude at which it can
receive gradient is the batch: put the gold in as a row and let the estimator that already
runs on every other row run on it too.

The path is three pure pieces, each with one seam, so every piece is testable on CPU:

* ``areal/dataset/competition_math.py`` keeps the gold TOKENISED as ``gold_ids`` behind
  ``keep_solution`` (default off, so every existing run is bit-for-bit unchanged).
* :mod:`selfevo.gold.attach` puts ``gold_ids``/``gold_mask`` into a trajectory at that
  trajectory's own width, and is called both from ``RLVRWorkflow`` and from the one place
  every workflow's trajectory converges -- ``WorkflowExecutor`` -- so the OpenAI-proxy path
  the live MATH runs use is served by the same tested function.
* :mod:`selfevo.gold.substitute` turns a gold-carrying batch into one where a rollout of each
  qualifying group has been replaced by the gold row, with the counts, the typed refusals and
  the off-policy handling the two baselines (DyME, LSPO) need. The actor's only job is to call
  it before ``compute_logp`` and to reconcile the recomputed log-probabilities after.
"""

from selfevo.gold.attach import (
    GOLD_KEYS,
    GoldAttachError,
    GoldError,
    attach_gold,
    attach_gold_from_data,
    prompt_lengths,
)
from selfevo.gold.substitute import (
    GOLD_LOGP_SENTINEL,
    GoldLogprobPolicy,
    GoldMissingError,
    GoldOrderingError,
    GoldPolicyError,
    GoldRule,
    GoldShapeError,
    GoldStats,
    assert_gold_logprobs_filled,
    reconcile_gold_logprobs,
    substitute_gold_rows,
    substitute_in_place,
)

__all__ = [
    "GOLD_KEYS",
    "GOLD_LOGP_SENTINEL",
    "GoldAttachError",
    "GoldError",
    "GoldLogprobPolicy",
    "GoldMissingError",
    "GoldOrderingError",
    "GoldPolicyError",
    "GoldRule",
    "GoldShapeError",
    "GoldStats",
    "assert_gold_logprobs_filled",
    "attach_gold",
    "attach_gold_from_data",
    "prompt_lengths",
    "reconcile_gold_logprobs",
    "substitute_gold_rows",
    "substitute_in_place",
]
