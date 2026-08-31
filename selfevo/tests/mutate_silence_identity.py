"""Mutation test for the silent-channel decomposition identity.

KNOWN EQUIVALENT MUTANT (survives, and should): "silent computed without the loss mask".
Under the shipped configuration (reward_norm at group level, adv_norm off, gae_lambda 1.0)
the advantage is CONSTANT across a sequence's tokens, so the masked and unmasked sums are
proportional and are zero together -- the mask cannot change which groups read as silent.
Measured directly: a row whose loss_mask is zeroed comes out of _compute_advantages with
advantages already all 0.0, and a row with an intact mask carries the SAME value on its
prompt columns (-0.866) as on its response columns.

The mask is NOT redundant in general -- prompt positions do carry non-zero advantages, which
is why group_apply masks its writes -- but making this mutant fail needs token-level
advantages (gae_lambda < 1 with a value model), which no config here uses. Recorded rather
than papered over: 4/5 killed, 1 equivalent under the shipped config.
"""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "areal/trainer/ppo/actor.py"
TESTS = "selfevo/tests/test_silence_identity.py"

MUTATIONS = [
    ("third bucket dropped, so truncated groups vanish from the decomposition",
     '                        unclassified_group_fraction=float(other.mean()),\n', ''),
    ("unclassified hardcoded to zero",
     '                    other = silent * (1.0 - solved) * (1.0 - unsolved)',
     '                    other = silent * 0.0'),
    ("unclassified drops the silent factor, counting non-silent groups",
     '                    other = silent * (1.0 - solved) * (1.0 - unsolved)',
     '                    other = (1.0 - solved) * (1.0 - unsolved)'),
    ("solved and unsolved complements swapped into an OR",
     '                    other = silent * (1.0 - solved) * (1.0 - unsolved)',
     '                    other = silent * ((1.0 - solved) + (1.0 - unsolved))'),
    ("silent computed without the loss mask, changing what silence means",
     '                    seq_adv = (advantages * data["loss_mask"]).sum(-1)',
     '                    seq_adv = advantages.sum(-1)'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=1200)
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
