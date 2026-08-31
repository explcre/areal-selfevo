"""Tests for the self-consistency baseline.

Built around cases where the right answer is known by construction, because a voting bug
produces a plausible curve rather than an error.
"""
from __future__ import annotations
import json, subprocess, sys, tempfile, pathlib

BENCH = pathlib.Path(__file__).resolve().parent
SC = BENCH / "selfconsistency.py"


def write(rows, path):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def gen(idx, sample, boxed, gold):
    """A generation whose grade() outcome is decided by the boxed answer we embed."""
    return {"benchmark": "math500", "idx": idx, "run_pos": idx, "sample": sample,
            "gold": gold, "boxed": boxed, "finish_reason": "stop", "status": "ok",
            "correct": boxed == gold, "text": f"reasoning \\boxed{{{boxed}}}"}


def run(path, extra=()):
    r = subprocess.run([sys.executable, str(SC), str(path), "--trials", "0", *extra],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout


def parse(out):
    d = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) == 3 and p[0].strip().isdigit():
            d[int(p[0])] = (float(p[1]), float(p[2]))
    return d


def test_majority_recovers_a_correct_answer_a_single_sample_misses():
    """3 samples, 2 correct: maj@3 must be 1.0 while maj@1 on a wrong first sample is 0."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        # first sample wrong, next two right -> maj@1 (first only) = 0, maj@3 = 1
        write([gen(0, 0, "9", "7"), gen(0, 1, "7", "7"), gen(0, 2, "7", "7")], p)
        d = parse(run(p))
        assert d[1][0] == 0.0, d
        assert d[3][0] == 1.0, d


def test_majority_can_be_wrong_when_the_model_is_confidently_wrong():
    """Voting is not an oracle: a 2/3 wrong majority must score 0, and pass@3 must be 1."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        write([gen(0, 0, "9", "7"), gen(0, 1, "9", "7"), gen(0, 2, "7", "7")], p)
        d = parse(run(p))
        assert d[3][0] == 0.0, "majority was wrong; maj@3 must not be rescued"
        assert d[3][1] == 1.0, "pass@3 sees the correct sample"


def test_pass_at_k_dominates_maj_at_k_everywhere():
    """An oracle can never do worse than a vote."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        rows = []
        for i in range(6):
            for s in range(4):
                rows.append(gen(i, s, "7" if (i + s) % 3 else "9", "7"))
        write(rows, p)
        d = parse(run(p))
        for k, (m, q) in d.items():
            assert q >= m - 1e-9, f"pass@{k}={q} < maj@{k}={m}"


def test_answer_normalisation_merges_only_equivalent_spellings():
    r"""\dfrac{1}{2} and \frac{1}{2} are one vote; 1/2 stays separate (conservative)."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        write([gen(0, 0, r"\dfrac{1}{2}", r"\frac{1}{2}"),
               gen(0, 1, r"\frac{1}{2}", r"\frac{1}{2}"),
               gen(0, 2, "9", r"\frac{1}{2}")], p)
        d = parse(run(p))
        assert d[3][0] == 1.0, "the two spellings should have formed a 2-vote majority"


def test_unparseable_samples_do_not_vote():
    """A sample with no boxed answer must abstain rather than form a null majority."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        rows = [gen(0, 0, "7", "7")]
        for s in (1, 2):
            r = gen(0, s, "7", "7"); r["boxed"] = None; r["text"] = "no box"; r["correct"] = False
            rows.append(r)
        write(rows, p)
        d = parse(run(p))
        assert d[3][0] == 1.0, "the single parseable (correct) vote should win"


def test_refuses_a_single_sample_file():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        write([gen(0, 0, "7", "7")], p)
        r = subprocess.run([sys.executable, str(SC), str(p)], capture_output=True, text=True)
        assert r.returncode != 0
        assert "not a scaling baseline" in (r.stdout + r.stderr)


def test_a_tie_is_not_broken_in_the_baselines_favour():
    """Two mutants survived by letting a tie resolve to the correct answer.

    One broke ties by consulting `correct`, which is an oracle the baseline does not have.
    The other disabled \\dfrac/\\frac merging, turning a real 2-vote majority into a
    three-way tie that happened to resolve correctly anyway. Both inflate the baseline.

    Here the tie is 1-1 with the WRONG answer first, so any tie-break that peeks at
    correctness, or any normalisation failure that manufactures a tie, changes the score.
    """
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        write([gen(0, 0, "9", "7"), gen(0, 1, "7", "7")], p)
        d = parse(run(p))
        assert d[2][0] == 0.0, "a 1-1 tie must resolve to the first sample, which is wrong"
        assert d[2][1] == 1.0, "pass@2 still sees the correct sample"


def test_equivalent_spellings_merge_rather_than_forming_a_tie():
    r"""\dfrac{1}{2} twice must OUTVOTE a single wrong answer that comes first."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        write([gen(0, 0, "9", r"\frac{1}{2}"),
               gen(0, 1, r"\dfrac{1}{2}", r"\frac{1}{2}"),
               gen(0, 2, r"\frac{1}{2}", r"\frac{1}{2}")], p)
        d = parse(run(p))
        # Without dfrac/frac merging this is a 1-1-1 tie won by the wrong first sample.
        assert d[3][0] == 1.0, "the two spellings must merge into a winning 2-vote block"


def test_refuses_an_empty_generations_file():
    """An empty artifact must not report a 0.000 baseline that looks like a measurement."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "g.jsonl"
        p.write_text("")
        r = subprocess.run([sys.executable, str(SC), str(p)], capture_output=True, text=True)
        assert r.returncode != 0
        assert "empty generations" in (r.stdout + r.stderr)
