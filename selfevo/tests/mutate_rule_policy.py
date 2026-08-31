"""Mutation test for the M9 rule evolve-policy: its branches, its guards, and its registration.

Modelled on ``mutate_contextual_cold_start.py`` / ``mutate_random_control.py``, and covering
BOTH files that decide what a training run gets: the rule itself, and the ``compose``
registry entries through which ``_route_groups`` builds it with no arguments.

**RETRACTED, 2026-08-31, by adversarial audit.** The first version of this harness excluded
"a ``reward_std`` threshold below the smallest attainable non-zero std of a BINARY group" as
provably equivalent. **It is not.** ``std > 0.05`` survived all 41 tests AND the full
961-test suite while flipping the shipped decision on realistic partial-credit groups
(``[1.0, 0.96, 1.0, 0.96]`` at 2.0e-02, ``[0.5, 0.55]`` at 2.5e-02, ``[1.0, 0.99]`` at
5.0e-03). The exclusion was self-refuting on its own terms, since 0.1 lies inside the class
it declared equivalent and was itself listed and killed. The tested boundary was 0.1 and
everything in (0, 0.1] was unconstrained. That gap is now closed from both sides -- by
behavioural cases at 5.0e-03 and by a property test that recomputes the float32 noise floor
and the smallest real dispersion and asserts the constant lies between them -- and the
threshold mutations below span the band.

One exclusion stands, and only one: ``ctx.has_target`` -> ``ctx.has_teacher`` on the unsolved
branch. That branch is reached only when the group is silent and ``solve_rate == 0``, and
``has_self_target`` is ``solve_rate > 0``, so the two expressions are equal there by
construction rather than by measurement.

The genuinely unconstrained residue, stated with its measured bounds rather than waved at:
any ``_UNANIMITY_EPS`` in roughly (4.8e-07, 5.0e-05) passes both the property test and every
behavioural case, because no group these graders can produce has a dispersion in that band --
float32 residue tops out at 1.192e-07 and the smallest real dispersion found is 5.0e-03.

Run it in the BACKGROUND. A harness killed mid-mutation leaves the target file mutated and
the next run adopts the corruption as its baseline (EXPERIMENTS.md, 2026-08-31). Verify the
tree afterwards with ``git diff --stat`` either way.
"""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TESTS = "selfevo/tests/test_rule_policy.py"
RULE = REPO / "selfevo/routing/rule_policy.py"
COMPOSE = REPO / "selfevo/compose.py"

MUTATIONS = [
    # ---- the silence test: the one split the whole rule rests on ---------------------
    (RULE, "silence test inverted, so silent and informative swap",
     "        if not _is_silent(std):", "        if _is_silent(std):"),
    (RULE, "silence test keyed on the OUTCOME split instead of the reward split",
     "        if not _is_silent(std):", "        if 0.0 < ctx.solve_rate < 1.0:"),
    (RULE, "float32 residue reads as signal again (the audit's F2, reintroduced)",
     "    return reward_std <= _UNANIMITY_EPS", "    return reward_std == 0.0"),
    (RULE, "silence comparison flipped inside the helper",
     "    return reward_std <= _UNANIMITY_EPS", "    return reward_std >= _UNANIMITY_EPS"),
    # ---- the tolerance itself, spanning the band the audit found unconstrained -------
    (RULE, "tolerance zeroed: back to the exact-zero test that F2 broke",
     "_UNANIMITY_EPS: float = 1e-6", "_UNANIMITY_EPS: float = 0.0"),
    (RULE, "tolerance below the measured float32 noise floor (1.192e-07)",
     "_UNANIMITY_EPS: float = 1e-6", "_UNANIMITY_EPS: float = 1e-8"),
    (RULE, "tolerance raised into the band where real dispersion starts",
     "_UNANIMITY_EPS: float = 1e-6", "_UNANIMITY_EPS: float = 1e-4"),
    (RULE, "tolerance at 0.05 -- the mutant that survived the first harness",
     "_UNANIMITY_EPS: float = 1e-6", "_UNANIMITY_EPS: float = 0.05"),
    (RULE, "tolerance at 0.1, deleting a partial-credit gradient",
     "_UNANIMITY_EPS: float = 1e-6", "_UNANIMITY_EPS: float = 0.1"),
    # ---- the two silent branches -----------------------------------------------------
    (RULE, "solved and unsolved branches swapped",
     "        if ctx.has_self_target:", "        if not ctx.has_self_target:"),
    (RULE, "solved branch defaults to SFT despite being measured inert at 0.5, harmful at 2.0",
     "    solved_mode: str = TrainingMode.SKIP", "    solved_mode: str = TrainingMode.SFT"),
    (RULE, "solved branch hardcodes SKIP, so the A/B knob decides nothing",
     "                {self.solved_mode: 1.0},", "                {TrainingMode.SKIP: 1.0},"),
    (RULE, "unsolved branch takes a teacher mode when no target exists",
     "        if ctx.has_target:", "        if True:"),
    # ---- the harness axis ------------------------------------------------------------
    (RULE, "truncation gate removed, so every unsolved group proposes",
     "        propose = trunc >= self.truncated_threshold", "        propose = True"),
    (RULE, "truncation comparison flipped",
     "        propose = trunc >= self.truncated_threshold",
     "        propose = trunc <= self.truncated_threshold"),
    (RULE, "truncation threshold lowered to zero, so it gates nothing",
     "    truncated_threshold: float = 1.0", "    truncated_threshold: float = 0.0"),
    (RULE, "harness action emitted in a run with no harness arm",
     "            if ctx.can_evolve_harness:", "            if True:"),
    # ---- validation, which must be loud ----------------------------------------------
    (RULE, "missing feature no longer raises MissingFeatures",
     "            if name not in ctx.extra:", "            if False:"),
    (RULE, "non-finite feature accepted, so NaN >= threshold reads as 'terminated'",
     "            if not math.isfinite(value):", "            if False:"),
    (RULE, "features describing a different unit accepted",
     '        if abs(out["solve_rate"] - ctx.solve_rate) > _SOLVE_RATE_TOLERANCE:',
     "        if False:"),
    (RULE, "solve-rate tolerance widened until any disagreement passes",
     "_SOLVE_RATE_TOLERANCE: float = 1e-6", "_SOLVE_RATE_TOLERANCE: float = 1.0"),
    (RULE, "arithmetically impossible group routed instead of refused",
     '        if _is_silent(out["reward_std"]) and out["solve_rate"] not in (0.0, 1.0):',
     "        if False:"),
    (RULE, "TOKEN granularity accepted, so a solved token takes the unsolved branch",
     "        if ctx.granularity is Granularity.TOKEN:", "        if False:"),
    (RULE, "teacher_mode no longer has to require a teacher",
     "        if not modes[self.teacher_mode]:", "        if False:"),
    (RULE, "solved_mode='rl' accepted: a logged rl group that changes no weight",
     "        if self.solved_mode == TrainingMode.RL:", "        if False:"),
    (RULE, "feature_names no longer has to cover what the rule reads",
     "        missing = [f for f in READ_FEATURES if f not in self.feature_names]",
     "        missing = []"),
    (RULE, "truncated_threshold range check removed",
     "        if not 0.0 <= self.truncated_threshold <= 1.0:", "        if False:"),
    (RULE, "decision reason no longer names the branch that fired",
     '                reason=f"informative: reward_std={std:.3e} > {_UNANIMITY_EPS:.0e}",',
     '                reason="routed",'),
    # ---- the registry: what a config can actually select ------------------------------
    (COMPOSE, "rule not registered as a router, so no config can select the baseline",
     '    "rule": _rule_router,                # M9: hand-written, deterministic, same 7 features\n',
     ""),
    (COMPOSE, "evolve_policy=rule reverts to the one-scalar router it replaced",
     '    "rule": _rule_router,\n    "learned_weights": _contextual_router,',
     '    "rule": _solve_rate_router,\n    "learned_weights": _contextual_router,'),
    (COMPOSE, "factory injects a default, so the dataclass default stops being the arm",
     "    return RulePolicyRouter(**kw)  # type: ignore[arg-type]",
     '    kw.setdefault("solved_mode", "sft")\n    return RulePolicyRouter(**kw)  # type: ignore[arg-type]'),
]


def run_tests() -> bool:
    """Whether the rule-policy suite is green on whatever is currently on disk."""
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=900)
    return r.returncode == 0


def main() -> int:
    """Apply each mutation, run the suite, restore, and report the score."""
    originals = {f: f.read_text() for f in {RULE, COMPOSE}}
    digests = {f: hashlib.sha256(t.encode()).hexdigest() for f, t in originals.items()}
    if not run_tests():
        print("BASELINE IS RED"); return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")
    survivors = []
    for target, label, find, repl in MUTATIONS:
        orig = originals[target]
        if orig.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {orig.count(find)}x")
            survivors.append(label); continue
        target.write_text(orig.replace(find, repl, 1))
        try:
            survived = run_tests()
        finally:
            target.write_text(orig)
            assert hashlib.sha256(target.read_text().encode()).hexdigest() == digests[target]
        print(f"{'SURVIVED' if survived else 'killed  '}  {label}")
        if survived:
            survivors.append(label)
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    for x in survivors:
        print(f"  SURVIVOR: {x}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
