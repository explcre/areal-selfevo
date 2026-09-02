# Routing policy for arm r_covariate. COVARIATES ONLY.
#
# The experiment is the EXCLUSION. Of the seven observability features, two are functions of
# the pass count k -- solve_rate and reward_std -- and five are covariates:
# mean_response_len, len_dispersion, mean_logprob, logprob_dispersion, truncated_fraction.
# Under binary rewards k is everything the outcome statistics contain, and the 102/102
# rule-collapse result constrains nothing about covariates because four of the five were held
# CONSTANT across those contexts. So this policy reads NO function of k. If it beats its
# rate-matched control, covariates carry routing information beyond the pass count.
#
# `solve_rate` and `reward_std` MUST NOT appear below. Their absence is the hypothesis.
#
# Primary signal is truncated_fraction: it is the one covariate with a measurement behind it
# on this box (truncation is non-termination, n_truncated == n_no_box everywhere it was
# checked), and it is the feature the harness selector was already shown to act on.
# Tiebreak is mean_logprob, a confidence proxy that costs nothing because the sampler already
# returned it, and which is independent of length.
#
# Feature access is by SUBSCRIPT, not .get(): the policy allowlist permits ast.Subscript and
# rejects attribute calls, and every group carries all seven features from
# GroupFeatures.as_extra(), so a missing key would be a bug worth raising rather than
# defaulting past. A raise costs SKIP via the router fallback, which is the safe direction.
#
# Thresholds are fixed IN ADVANCE and are not tuned: 0.5 is the same "most of this batch never
# terminated" line the truncation harness selector uses, and -1.0 nats/token is a round
# separation between confident and unconfident generations rather than a fitted value.


def route(features):
    """Choose a training mode from covariates alone.

    Args:
        features: Observability features for one GRPO group. Reads only
            ``truncated_fraction`` and ``mean_logprob``; never ``solve_rate`` or
            ``reward_std``.

    Returns:
        One of ``"skip"``, ``"sft"``, ``"rl"``.
    """
    truncated = features["truncated_fraction"]
    confidence = features["mean_logprob"]

    # Mostly non-terminating: the group ran out of budget rather than reasoning badly, so its
    # rollouts are not evidence about the answer and training on them teaches length.
    if truncated >= 0.5:
        return "skip"

    # Terminating and confident: the model already commits to these, so reinforcing its own
    # sampled tokens is the cheap update that needs no teacher.
    if truncated <= 0.05 and confidence >= -1.0:
        return "sft"

    # Everything else: ordinary policy gradient.
    return "rl"
