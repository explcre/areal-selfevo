"""The three task sources, each emitting the same record through the same interface.

What each one threatens, stated where the code is rather than in a report:

* **generated** -- the model writes both the problem and its key, so the key inherits the
  model's errors. Measured on this project at 6.56% refuted among decided cases, against
  2.55% for curated problems, with two refutations being literally the same problem generated
  twice carrying different asserted answers.
* **retrieved** -- the key is written by someone other than the model being trained, so it
  should NOT show that failure; but a retrieved problem may be memorised, which reads as easy
  and corrupts difficulty in the opposite direction, and one overlapping the held-out half
  destroys the evaluation. Provenance and licence are recorded per item.
* **distilled** -- a teacher writes the task, so it inherits the TEACHER's errors and the
  verifier applies unchanged. A teacher from the solver's own family is the collusion risk
  already logged for the judges, and is reported rather than assumed away.
"""
from __future__ import annotations

import json
import os
import re

from .base import Provenance, SourceResult, TaskRecord

#: Licences of the corpora shipped in the pinned Absolute-Zero-Reasoner clone. Recorded per
#: item rather than assumed: an item whose licence we cannot name says so.
CORPUS_LICENCE = {
    "math500": "MIT (AZR redistribution of MATH)",
    "math": "MIT (AZR redistribution of MATH)",
    "gsm8k": "MIT (AZR redistribution of GSM8K)",
    "amc23": "MIT (AZR redistribution)",
    "aime24": "MIT (AZR redistribution)",
    "aime25": "MIT (AZR redistribution)",
    "minerva_math": "MIT (AZR redistribution)",
    "college_math": "MIT (AZR redistribution)",
    "olympiadbench": "MIT (AZR redistribution of OlympiadBench)",
}


def _norm_answer(row: dict) -> str | None:
    """Pull the asserted key out of a corpus row, whatever the corpus calls it."""
    for k in ("answer", "final_answer", "Answer", "solution"):
        if k in row and row[k] not in (None, ""):
            v = row[k]
            if isinstance(v, list):
                v = v[0] if v else None
            return None if v is None else str(v).strip()
    return None


class RetrievedSource:
    """Curated public problems already on disk, with provenance and licence per item.

    This is retrieval from PINNED CORPORA, not live scraping: nothing is fetched from the
    open web, so nothing is laundered, and the property under test -- that the key was
    written by someone other than the model being trained -- holds exactly as it would for a
    scraped problem. The corpora excluded are the ones our own evaluation uses.
    """

    name = "retrieved"

    def __init__(self, root: str, corpora, exclude_corpora=("olympiadbench",)):
        self.root = root
        self.corpora = [c for c in corpora if c not in exclude_corpora]
        self._rows: list[tuple[str, int, dict]] | None = None

    def _load(self) -> list[tuple[str, int, dict]]:
        """Read every usable row once, keeping its corpus and row index for provenance."""
        if self._rows is not None:
            return self._rows
        out = []
        for c in self.corpora:
            path = os.path.join(self.root, c, "test.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("problem") or row.get("question")
                    ans = _norm_answer(row)
                    if text and ans and len(str(ans)) <= 120:
                        out.append((c, i, {"text": str(text), "answer": str(ans)}))
        self._rows = out
        return out

    def fetch(self, n: int, rng) -> SourceResult:
        """Draw `n` distinct rows at random across the permitted corpora."""
        rows = self._load()
        if not rows:
            return SourceResult.failure(
                self.name, n, "no usable rows found under %s for corpora %s"
                % (self.root, self.corpora))
        picked = rng.sample(rows, min(n, len(rows)))
        tasks = []
        for corpus, idx, r in picked:
            tasks.append(TaskRecord(
                text=r["text"], answer=r["answer"],
                provenance=Provenance(
                    source=self.name, origin="%s#%d" % (corpus, idx),
                    licence=CORPUS_LICENCE.get(corpus, "unknown"),
                    detail={"corpus": corpus, "row": idx, "retrieved_from": "local pinned "
                            "Absolute-Zero-Reasoner clone, not the open web"})))
        return SourceResult(self.name, tasks, attempted=n, ok=True, cost_tokens=0)


PROPOSER_FORMAT = ("Reply in exactly this format and nothing else:\n"
                   "PROBLEM: <statement on one line>\nANSWER: <final answer only>\n")

GENERATE_PROMPT = ("You are proposing a new training problem for a model.\n\n"
                   "{context}\n"
                   "Propose ONE new problem that is fully self-contained, has exactly ONE "
                   "correct final answer as a closed form, and that you are certain of.\n\n"
                   + PROPOSER_FORMAT)

DISTIL_PROMPT = ("You are a teacher writing an exam problem for a student model.\n\n"
                 "{context}\n"
                 "Write ONE problem that is fully self-contained, has exactly ONE correct "
                 "final answer as a closed form, and that you have solved yourself.\n\n"
                 + PROPOSER_FORMAT)


def parse_problem_answer(text: str) -> tuple[str | None, str | None, str]:
    """Split a PROBLEM/ANSWER reply, or say why it is unusable.

    The reasoning block is stripped first: this base is a thinking model and emits one
    before the fields whatever the prompt says.
    """
    if not text:
        return None, None, "empty"
    body = text.split("</think>")[-1]
    mp = re.search(r"PROBLEM:\s*(.+?)(?:\n\s*ANSWER:|\Z)", body, re.S)
    ma = re.search(r"ANSWER:\s*(.+?)\s*\Z", body, re.S)
    if not mp:
        return None, None, "no PROBLEM field"
    if not ma:
        return None, None, "no ANSWER field"
    problem = " ".join(mp.group(1).split())
    answer = " ".join(ma.group(1).split())
    if len(problem) < 40:
        return None, None, "problem too short"
    if len(answer) > 120:
        return None, None, "answer not closed form"
    return problem, answer, "ok"


class ModelWrittenSource:
    """Shared machinery for the two sources where a model writes the task and its key.

    `generated` and `distilled` differ in WHICH model writes and in what the prompt says, not
    in how the reply is parsed or how failure is reported, so they share this.
    """

    def __init__(self, name, backend, prompt, context="", licence="model-generated",
                 oversample=4, detail=None):
        self.name = name
        self.backend = backend
        self.prompt = prompt
        self.context = context
        self.licence = licence
        self.oversample = oversample
        self.detail = detail or {}

    def fetch(self, n: int, rng) -> SourceResult:
        """Generate candidates and parse them; an empty yield is an explicit failure."""
        want = n * self.oversample
        try:
            replies, cost = self.backend.generate(
                [self.prompt.format(context=self.context)] * want)
        except Exception as exc:  # a backend that cannot run is a failure, not zero tasks
            return SourceResult.failure(self.name, want,
                                        "backend %r raised: %r" % (self.backend.name, exc))
        tasks, reasons = [], {}
        for text in replies:
            problem, answer, why = parse_problem_answer(text)
            if why != "ok":
                reasons[why] = reasons.get(why, 0) + 1
                continue
            tasks.append(TaskRecord(
                text=problem, answer=answer,
                provenance=Provenance(source=self.name, origin=self.backend.name,
                                      licence=self.licence,
                                      detail=dict(self.detail, parse="PROBLEM/ANSWER"))))
            if len(tasks) >= n:
                break
        if not tasks:
            return SourceResult.failure(
                self.name, want,
                "produced %d replies and none parsed as a task; reasons=%s"
                % (len(replies), reasons), cost_tokens=cost)
        return SourceResult(self.name, tasks, attempted=want, ok=True, cost_tokens=cost,
                            detail={"parse_rejections": reasons, "replies": len(replies)})
