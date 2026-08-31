"""Mutation test for the per-prompt credit ledger."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "selfevo/routing/prompt_credit.py"
TESTS = "selfevo/tests/test_prompt_credit.py"

MUTATIONS = [
    ("identity hashes the whole row, so a prompt never matches itself",
     "    payload = \",\".join(str(int(t)) for t in input_ids[:n_prompt]).encode()",
     "    payload = \",\".join(str(int(t)) for t in input_ids).encode()"),
    ("identity hashes only the first token, merging unrelated prompts",
     "    payload = \",\".join(str(int(t)) for t in input_ids[:n_prompt]).encode()",
     "    payload = str(int(input_ids[0])).encode()"),
    ("a row with no prompt region returns a constant instead of raising",
     "    if n_prompt == 0:", "    if False:"),
    ("length mismatch accepted",
     "    if len(input_ids) != len(loss_mask):", "    if False:"),
    ("records before crediting, so a sighting pairs with itself",
     "        prior = self._pending.get(key)",
     "        self._pending[key] = PriorDecision(unit_id=unit_id, mode=mode, value=value, step=step)\n        prior = self._pending.get(key)"),
    ("eviction takes the newest instead of the oldest",
     "            self._pending.popitem(last=False)",
     "            self._pending.popitem(last=True)"),
    ("evictions not counted, so starvation is invisible",
     "            self.evicted += 1", "            pass"),
    ("delta sign flipped, rewarding harmful modes",
     "            out = (prior, value - prior.value)",
     "            out = (prior, prior.value - value)"),
    ("same-batch guard removed, so duplicate prompts pair within one batch",
     "        if prior is not None and prior.step == step:", "        if False:"),
    ("same-batch guard fires on every sighting, so nothing ever pairs",
     "        if prior is not None and prior.step == step:", "        if prior is not None:"),
    ("same-batch skip overwrites the earlier record",
     "            self.same_batch_skips += 1\n            return None",
     "            self.same_batch_skips += 1\n            self._pending[key] = PriorDecision(unit_id=unit_id, mode=mode, value=value, step=step)\n            return None"),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=900)
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
