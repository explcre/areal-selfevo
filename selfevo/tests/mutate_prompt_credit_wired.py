"""Mutation test for the prompt-credit wiring in the actor."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "areal/trainer/ppo/actor.py"
TESTS = [
    "selfevo/tests/test_prompt_credit_wired.py",
    "selfevo/tests/test_m8_config_gap.py",
]

MUTATIONS = [
    ("prompt credit never runs, so credit='prompt' silently means 'batch'",
     '        if use_prompt_credit and hasattr(router, "observe"):',
     '        if False and hasattr(router, "observe"):'),
    ("credit mode ignored, so both arms take the prompt path",
     '        use_prompt_credit = _credit in (\n            "prompt",\n            "prompt_centered",\n            "prompt_self_baseline",\n        )',
     '        use_prompt_credit = True'),
    ("credit mode ignored the other way, so the arm never exists",
     '        use_prompt_credit = _credit in (\n            "prompt",\n            "prompt_centered",\n            "prompt_self_baseline",\n        )',
     '        use_prompt_credit = False'),
    ("the self-baseline arm silently runs the 'last' baseline, i.e. plain per-prompt credit",
     '                    baseline="self_mean"\n                    if _credit == "prompt_self_baseline"\n                    else "last"',
     '                    baseline="last"'),
    ("every per-prompt arm gets the self baseline, erasing the ablation rung",
     '                    baseline="self_mean"\n                    if _credit == "prompt_self_baseline"\n                    else "last"',
     '                    baseline="self_mean"'),
    ("the correspondence control never permutes, so it is the treatment run twice",
     '            if _shuffle_seed is not None:',
     '            if False:'),
    ("the shuffle fires on every arm, so the treatment is its own control",
     '            if _shuffle_seed is not None:',
     '            if True:'),
    ("ledger rebuilt every batch, so no prompt ever pairs",
     '            ledger = getattr(self, "_selfevo_ledger", None)',
     '            ledger = None'),
    ("outcomes keyed by batch instead of by the prior unit, so credit lands on no arm",
     '                outcomes[prior.unit_id] = DecisionOutcome(',
     '                outcomes[str(prior.step)] = DecisionOutcome('),
    ("the prompt key uses the whole row, so a prompt never matches itself",
     '                    key = prompt_key(ids_cpu[row], mask_cpu[row])',
     '                    key = str(ids_cpu[row])'),
    ("centring never applied, so the common training trend stays in every credit",
     '            if _credit == "prompt_centered" and pairs:',
     '            if False:'),
    ("centring applied to the raw prompt arm too, erasing the ablation",
     '            if _credit == "prompt_centered" and pairs:',
     '            if pairs:'),
    ("shift uses the max instead of the mean, so credits are all non-positive",
     '                shift = sum(d for _, d in pairs) / len(pairs)',
     '                shift = max(d for _, d in pairs)'),
    ("shift added instead of subtracted, doubling the common trend",
     '                    mode=prior.mode, value=delta - shift, batch_id=str(prior.step)',
     '                    mode=prior.mode, value=delta + shift, batch_id=str(prior.step)'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)
    return r.returncode == 0


def main() -> int:
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()
    if not run_tests():
        print("BASELINE IS RED"); return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")
    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append(label); continue
        TARGET.write_text(original.replace(find, repl, 1))
        try:
            survived = run_tests()
        finally:
            TARGET.write_text(original)
            assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest
        print(f"{'SURVIVED' if survived else 'killed  '}  {label}")
        if survived:
            survivors.append(label)
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    for x in survivors:
        print(f"  SURVIVOR: {x}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
