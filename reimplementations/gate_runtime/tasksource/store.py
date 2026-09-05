"""Permanent, append-only task store. Every field exists because something went wrong without it.

The record is deliberately opinionated: it REFUSES to be constructed without the things this
project has learned it needs, rather than defaulting them. Each refusal below traces to a
specific failure already in the record:

* a positional index caused a leak no set-level guard could see, so identity is a CONTENT hash
  as well as an id;
* a teacher from the solver's own family is a collusion risk, so a distilled task must name its
  teacher and version or it cannot be stored;
* a refutation from string matching and one from exact symbolic comparison differ by 0.43 on
  the same problems, so a verdict must name its comparator, and a refutation must carry its
  witness;
* a task's difficulty under a scaffold differs from its difficulty bare by 0.2 to 0.36 in an
  unpredictable direction, so a success rate without the prompt it was measured under is not a
  fact about the task and is refused;
* a similarity score has a false-positive floor of 0.14 on short statements, so a dedup score
  without its threshold and its length-band floor is uninterpretable and is refused.

Schema v1 -> v2: v1 carried a single dedup boolean named `above_floor` whose value was
in fact `score >= threshold` -- the action test wearing the floor test's name. The two
answer different questions: whether an overlap triggers the reference set's action, and
whether the overlap is distinguishable from the chance overlap between unrelated problems
of that length. v2 records both, `above_threshold` and `above_floor`, and v1 rows are
refused rather than reread, because a v1 `above_floor` does not mean what a v2 one means.

Schema v2 -> v3: v2's `cost.score_tokens` held the batch mean over every task scored in
the run, written into each record as though it were that record's own cost. It read as
a per-task measurement and was not one, and a cost ranking built on it compared three
identical numbers. v3 attributes verification cost to the task's own samples and names
the basis of each figure in `produce_basis` and `score_basis`. v2 rows are refused.

Schema changes are versioned. An unknown version is REFUSED LOUDLY rather than read with
today's assumptions, because silently reinterpreting an old row is how a store stops being a
record of what happened.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

#: Bump when the meaning of any field changes. Readers refuse versions they do not know.
SCHEMA_VERSION = 3

#: Statement-length bands and the measured false-positive floor of the similarity measure in
#: each, per reference set. Measured, not assumed: see `out/dedup_floors.json`.
LENGTH_BANDS = ((0, 60), (60, 100), (100, 160), (160, 260), (260, 10**9))
FLOORS = {
    "held_out":      {0.45: {(0, 60): 0.0917, (60, 100): 0.0809, (100, 160): 0.0238,
                             (160, 260): 0.0000, (260, 10**9): 0.0000}},
    "training_pool": {0.60: {(0, 60): 0.0000, (60, 100): 0.0000, (100, 160): 0.0000,
                             (160, 260): 0.0000, (260, 10**9): 0.0000}},
    "run_buffer":    {0.60: {(0, 60): 0.2917, (60, 100): 0.1397, (100, 160): 0.0635,
                             (160, 260): 0.0141, (260, 10**9): 0.0213}},
}

REQUIRED_PROVENANCE = {
    "generated": ("prompt_version", "exemplars"),
    "retrieved": ("corpus", "row", "licence"),
    "distilled": ("teacher_model", "teacher_version", "prompt_version"),
}


def band_for(text: str) -> tuple[int, int]:
    """The statement-length band a text falls in, for floor lookup."""
    n = len(text)
    for lo, hi in LENGTH_BANDS:
        if lo <= n < hi:
            return (lo, hi)
    return LENGTH_BANDS[-1]


def floor_for(reference: str, threshold: float, text: str) -> float | None:
    """Measured false-positive floor for this reference set, threshold and statement length.

    Returns None when no floor has been measured for that combination, which callers must
    store as None rather than as zero: an unmeasured floor is not a floor of zero.
    """
    by_th = FLOORS.get(reference)
    if not by_th:
        return None
    exact = by_th.get(threshold)
    if exact is None:
        return None
    return exact.get(band_for(text))


def content_hash(text: str) -> str:
    """SHA-256 of the whitespace-normalised statement.

    Content-addressed rather than positional. A row index is a property of a file, and using
    one as identity is what let a leak through a set-level guard: the guard compared sets of
    indices while the underlying file had changed.
    """
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def make_dedup(reference: str, score: float, threshold: float, text: str) -> dict:
    """One deduplication reading, with everything needed to interpret it.

    A bare score is uninterpretable: the same 0.5 means contamination against a 337-problem
    held-out set and noise against a 30-item run buffer, where the floor is 0.29 for short
    statements.
    """
    lo, hi = band_for(text)
    floor = floor_for(reference, threshold, text)
    return {"score": round(float(score), 4), "threshold": threshold,
            "length_band": "%d-%s" % (lo, "inf" if hi >= 10**9 else hi),
            "false_positive_floor": floor,
            "above_threshold": bool(score >= threshold),
            "above_floor": None if floor is None else bool(score >= floor)}


@dataclass
class StoredTask:
    """One task, permanently. Construction fails rather than storing an ambiguous record."""

    text: str
    answer: str
    source_type: str
    provenance: dict
    verification: dict
    difficulty: dict
    dedup: dict
    cost: dict
    task_id: str = ""
    content_hash: str = ""
    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""

    def __post_init__(self) -> None:
        """Refuse every ambiguity this project has already been bitten by."""
        if self.source_type not in REQUIRED_PROVENANCE:
            raise ValueError("unknown source_type %r; expected one of %s"
                             % (self.source_type, sorted(REQUIRED_PROVENANCE)))
        missing = [k for k in REQUIRED_PROVENANCE[self.source_type]
                   if k not in self.provenance or self.provenance[k] in (None, "")]
        if missing:
            raise ValueError(
                "source_type %r requires provenance fields %s; missing %s. A distilled task "
                "without its teacher cannot be audited for collusion with the solver, and a "
                "retrieved one without its licence cannot be redistributed."
                % (self.source_type, list(REQUIRED_PROVENANCE[self.source_type]), missing))
        for k in ("verdict", "backend", "comparator"):
            if not self.verification.get(k):
                raise ValueError(
                    "verification needs %r. A verdict without its comparator is unreadable: "
                    "string matching and exact symbolic comparison differ by 0.43 on the same "
                    "problems." % k)
        if self.verification["verdict"] == "refuted" and not self.verification.get("witness"):
            raise ValueError(
                "a REFUTED verdict must carry the witness that refuted it (the independently "
                "computed answer); without it the claim cannot be rechecked and 16 of 16 such "
                "claims turned out to be formatting.")
        if self.difficulty is not None and self.difficulty != {}:
            if self.difficulty.get("success_rate") is not None and not self.difficulty.get(
                    "measured_under_prompt"):
                raise ValueError(
                    "a success rate must name the prompt it was measured under. Difficulty "
                    "under a scaffold differs from difficulty bare by 0.2-0.36 in an "
                    "unpredictable direction, so a bare number is not a fact about the task.")
        for ref in ("held_out", "training_pool", "run_buffer"):
            d = self.dedup.get(ref)
            if not isinstance(d, dict):
                raise ValueError("dedup must carry all three reference sets; missing %r. They "
                                 "mean different things and must not be collapsed." % ref)
            for k in ("score", "threshold", "length_band", "above_threshold"):
                if k not in d:
                    raise ValueError("dedup[%r] needs %r" % (ref, k))
            if "false_positive_floor" not in d:
                raise ValueError(
                    "dedup[%r] needs false_positive_floor (None if unmeasured). A score "
                    "without its floor is what produced two withdrawn findings." % ref)
        for k in ("produce_tokens", "score_tokens"):
            if k not in self.cost:
                raise ValueError("cost needs %r" % k)
        if not self.content_hash:
            self.content_hash = content_hash(self.text)
        if not self.task_id:
            self.task_id = self.content_hash[:16]
        if not self.created_utc:
            self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_json(self) -> dict:
        """Serialisable record."""
        return asdict(self)


class TaskStore:
    """Append-only JSONL store. Never rewrites, never reinterprets."""

    def __init__(self, root: str, name: str = "tasks"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "%s.v%d.jsonl" % (name, SCHEMA_VERSION))

    def append(self, rec: StoredTask) -> str:
        """Append one record and return its task id."""
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\n")
        return rec.task_id

    def append_many(self, recs) -> int:
        """Append several records; returns how many were written."""
        n = 0
        with open(self.path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r.to_json(), sort_keys=True) + "\n")
                n += 1
        return n

    @staticmethod
    def read(path: str):
        """Read records, refusing any schema version this code does not know.

        Refusing loudly is the whole point. A future reader that silently applied today's
        field meanings to an older row would turn the store from a record of what happened
        into a record of what we currently assume.
        """
        out = []
        for i, line in enumerate(open(path)):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            v = row.get("schema_version")
            if v is None:
                raise ValueError("record %d in %s has no schema_version; refusing to guess"
                                 % (i, path))
            if v != SCHEMA_VERSION:
                raise ValueError(
                    "record %d in %s has schema_version %r but this code understands %r. "
                    "Refusing to read it: no migration is registered, and reinterpreting an "
                    "older row with today's field meanings would silently corrupt the record."
                    % (i, path, v, SCHEMA_VERSION))
            out.append(row)
        return out
