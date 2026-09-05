#!/usr/bin/env python3
"""Mutation test: break each property on purpose and require the suite to NOTICE.

A green suite is not evidence that a property holds; it is evidence that the suite ran. This
project has recorded five defects that survived 350 passing tests. So each of the four
properties an auditor would attack is broken here, one at a time, in a COPY of the package,
and the corresponding test must fail. A mutant that survives means the test does not constrain
the property it claims to.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

SRC = "/mnt/localssd/gate/code"
MUTANTS = [
    ("shared buffer becomes one buffer per source",
     "tasksource/pipeline.py",
     "            res: SourceResult = src.fetch(n_per_source, rng)",
     "            self.buffer = SharedNoveltyBuffer(self.buffer.threshold)\n"
     "            res: SourceResult = src.fetch(n_per_source, rng)",
     "test_1_novelty_buffer_is_shared_across_sources"),
    ("contamination check degrades to exact match only",
     "tasksource/registry.py",
     "        sim, at = most_similar(text, self.held_out)\n        return sim < self.threshold, sim, at",
     "        sim = 1.0 if text in self.held_out else 0.0\n        return sim < self.threshold, sim, -1",
     "test_2_contamination_catches_a_paraphrase_not_only_an_exact_match"),
    ("provenance dropped on the way to the artifact",
     "tasksource/base.py",
     '        d["provenance"] = asdict(self.provenance)\n        return d',
     '        d.pop("provenance", None)\n        return d',
     "test_3_provenance_survives_to_the_artifact"),
    ("a source returning ok with zero tasks is treated as a success",
     "tasksource/pipeline.py",
     '            if st.candidates == 0:\n                st.ok, st.failure_reason = False, "source returned ok with zero tasks"\n                continue',
     "            if st.candidates == 0:\n                continue",
     "test_4_a_source_that_produces_nothing_is_a_reported_failure"),
    ("a refuted key is accepted anyway",
     "tasksource/pipeline.py",
     "                        st.refuted += 1",
     "                        st.refuted += 1\n                        pass  # fall through",
     "test_5_a_refuted_key_is_never_accepted"),
]


def run_one(name, relpath, old, new, expect_fail):
    """Apply one mutation in a temporary copy and report whether the suite caught it."""
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(SRC, "tasksource"), os.path.join(d, "tasksource"))
        shutil.copy(os.path.join(SRC, "test_tasksource.py"), d)
        path = os.path.join(d, relpath)
        src = open(path).read()
        if old not in src:
            return name, "COULD NOT APPLY", "anchor text not found in %s" % relpath
        # The refuted mutant needs the `continue` removed, not a line added after it.
        if "fall through" in new:
            src = src.replace(
                "                        st.refuted += 1\n"
                "                        # A refuted key is not accepted: it would train the "
                "solver against a\n"
                "                        # wrong answer and, worse, score p_hat low and look "
                "ideally difficult.\n"
                "                        continue\n",
                "                        st.refuted += 1\n")
        else:
            src = src.replace(old, new, 1)
        open(path, "w").write(src)
        r = subprocess.run([sys.executable, "test_tasksource.py"], cwd=d,
                           capture_output=True, text=True)
        out = r.stdout
        caught = any(line.startswith(("FAIL " + expect_fail, "ERROR " + expect_fail))
                     for line in out.splitlines())
        other = [l for l in out.splitlines() if l.startswith(("FAIL", "ERROR"))
                 and expect_fail not in l]
        return name, ("CAUGHT" if caught else "SURVIVED"), \
            ("also failed: %s" % [o.split(":")[0] for o in other] if other else "")


if __name__ == "__main__":
    print("%-58s %-10s %s" % ("mutation", "verdict", "notes"))
    survived = 0
    for m in MUTANTS:
        name, verdict, note = run_one(*m)
        if verdict != "CAUGHT":
            survived += 1
        print("%-58s %-10s %s" % (name[:58], verdict, note))
    print()
    if survived:
        print("%d mutant(s) SURVIVED: the suite does not constrain those properties" % survived)
    else:
        print("all %d mutants caught: each property is genuinely constrained by a test"
              % len(MUTANTS))
    raise SystemExit(1 if survived else 0)
