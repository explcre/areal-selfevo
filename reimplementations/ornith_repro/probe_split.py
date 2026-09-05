"""Carve a weakness PROBE out of the search half, provably disjoint from the report half.

WHY THIS IS ITS OWN MODULE WITH ITS OWN GUARDS. The curriculum ranks categories by how weak
the model is on them, and then a training gain is reported on held-out data. If the set used
to RANK overlaps the set used to REPORT, the ranking is selecting on the outcome and the gain
is circular -- the result would be void rather than merely weak. That failure is invisible in
any output: the numbers look normal and the split is wrong.

So disjointness is asserted rather than assumed, on every load, and the assertion is what the
audit is invited to attack. Three sets are involved and all three relations are checked:

    probe  is a subset of search
    probe  is disjoint from report        <-- the one that voids the experiment
    probe  and train partition search      <-- so nothing is silently dropped

The split is stratified by subfield because the whole point is per-category measurement, and
an unstratified draw can leave a category with two problems and an interval spanning the mean.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict


class SplitError(RuntimeError):
    """Raised when a split relation does not hold. Never downgraded to a warning."""


def load_committed(split_path: str, source_path: str) -> tuple[list[int], list[int], str]:
    """Read the committed search/report split and verify it against its own dataset.

    Args:
        split_path: The committed split JSON.
        source_path: The dataset the split's indices address.

    Returns:
        `(search, report, md5)`.

    Raises:
        SplitError: if the dataset md5 differs from the one the split was built against, or
            if search and report are not disjoint.
    """
    raw = open(source_path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    spec = json.load(open(split_path))
    if md5 != spec["dataset_md5"]:
        raise SplitError(
            "dataset md5 %s does not match the split's %s; the indices address different "
            "problems and every number computed from them would be wrong"
            % (md5, spec["dataset_md5"]))
    search, report = list(spec["search"]), list(spec["report"])
    if set(search) & set(report):
        raise SplitError("committed search and report halves overlap")
    return search, report, md5


def make_probe_split(search: list[int], report: list[int], subfields: dict,
                     probe_frac: float = 0.35, seed: int = 20260904
                     ) -> tuple[list[int], list[int]]:
    """Split the search half into a probe and a training remainder, stratified by subfield.

    Args:
        search: Indices of the search half.
        report: Indices of the report half, used only to assert disjointness.
        subfields: Mapping index -> subfield label.
        probe_frac: Share of each subfield to place in the probe.
        seed: RNG seed, recorded so the split is reproducible.

    Returns:
        `(probe, train)`.

    Raises:
        SplitError: if any required relation fails.
    """
    rng = random.Random(seed)
    by_field = defaultdict(list)
    for i in search:
        by_field[subfields.get(i, "unknown")].append(i)

    probe: list[int] = []
    for field in sorted(by_field):
        items = sorted(by_field[field])
        rng.shuffle(items)
        k = max(1, int(round(probe_frac * len(items)))) if len(items) >= 2 else 0
        probe.extend(items[:k])
    probe = sorted(probe)
    train = sorted(set(search) - set(probe))

    assert_probe_disjoint(probe, train, report, search)
    return probe, train


def assert_probe_disjoint(probe, train, report, search) -> None:
    """Refuse any split that could make a later capability claim circular.

    Args:
        probe: Probe indices.
        train: Training indices.
        report: Held-out evaluation indices.
        search: The half both probe and train are drawn from.

    Raises:
        SplitError: on any violated relation, naming which one.
    """
    p, t, r, s = set(probe), set(train), set(report), set(search)
    if p & r:
        raise SplitError(
            "probe and report overlap on %d problems (e.g. %s); ranking categories on the "
            "evaluation set makes any reported gain circular"
            % (len(p & r), sorted(p & r)[:5]))
    if t & r:
        raise SplitError("train and report overlap on %d problems" % len(t & r))
    if p & t:
        raise SplitError("probe and train overlap on %d problems" % len(p & t))
    if p | t != s:
        raise SplitError("probe and train do not partition the search half (%d vs %d)"
                         % (len(p | t), len(s)))
    if not p:
        raise SplitError("probe is empty")


def write_split(path: str, probe, train, report, md5: str, seed: int,
                probe_frac: float) -> None:
    """Persist the split with the provenance needed to re-verify it.

    Args:
        path: Destination JSON.
        probe: Probe indices.
        train: Training indices.
        report: Report indices.
        md5: Dataset md5 the indices address.
        seed: RNG seed used.
        probe_frac: Requested probe share.
    """
    json.dump({"dataset_md5": md5, "seed": seed, "probe_frac": probe_frac,
               "n_probe": len(probe), "n_train": len(train), "n_report": len(report),
               "probe": list(probe), "train": list(train), "report": list(report),
               "note": "probe is carved from the SEARCH half only; it never intersects "
                       "report, which is the evaluation set."},
              open(path, "w"), indent=2)
