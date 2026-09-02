"""Sources of a correct row for a group that has none, behind one batch-construction seam.

READ ``selfevo/gold/__init__.py`` FIRST; it states the argument that makes this package
possible, and it is not restated here. In one line: an advantage is a coefficient on tokens the
model EMITTED, so a group in which every rollout is wrong cannot be reached through the
advantage by any router or any fixed rule, and the only altitude at which a correct row can
receive gradient is the BATCH. ``selfevo/gold/substitute.py`` is that seam. This package
generalises WHAT it splices in.

Four sources, one interface (:mod:`selfevo.supply.base`), one writer, one set of counters:

``gold``
    :mod:`selfevo.supply.gold` -- the dataset's own solution, previously inline in
    ``substitute_gold_rows`` and now the first implementation of the interface rather than a
    special case. The default path is bit-identical to before the refactor, and
    ``test_supply_sources.py`` pins that against a digest measured beforehand.
``self``
    :mod:`selfevo.supply.self_gen` -- a correct rollout of the SAME prompt from outside this
    group: an earlier step that solved it (:mod:`selfevo.supply.store`) or a
    higher-temperature resample (a declared seam, not implemented, because it needs an engine).
``corpus``
    :mod:`selfevo.supply.corpus` -- a solved example for this prompt from an offline pool.
``teacher``
    :mod:`selfevo.supply.teacher` -- a stronger model's VERIFIED completion. Interface plus an
    offline replay client only; nothing here serves a model or touches a GPU.

THE REGISTRY IS ROUTERS-SHAPED ON PURPOSE. ``selfevo/CONTRIBUTING.md`` §4.6 counts six
extension seams in this tree wearing five different shapes and says new axes should copy
``ROUTERS``: a dict of lazy factory wrappers, where the WRAPPER is the seam and an
experiment-deciding default is baked into it rather than left in a class default that
``factory()`` cannot reach. Two arms in this repo ran bit-identical to the off arm because a
default lived somewhere the factory could not see, and both retractions are recorded in
``compose.py``. So ``teacher`` has no default client and no default verifier, and asking for it
without one is a construction-time refusal rather than an arm that silently supplies nothing.

GATING. The capability is off unless a caller passes suppliers to the seam:
``substitute_gold_rows`` and ``substitute_in_place`` default to ``suppliers=None``, which means
the gold-only path exactly as it was, and ``GoldRule.NONE`` remains a true no-op above that.
Nothing in this package runs unless an arm asks for it by name.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from selfevo.supply.base import (
    NO_SOURCE,
    SUPPLY_SOURCES,
    Refusal,
    Supplier,
    SupplierRefused,
    SupplyConfigError,
    SupplyError,
    SupplyOffer,
    SupplyRequest,
    key_for_prompt,
    source_code,
)
from selfevo.supply.policy import (
    FixedSourcePolicy,
    ForcedSourcePolicy,
    MatchedSourceControl,
    SourcePolicy,
    source_proportions,
)

__all__ = [
    "NO_SOURCE",
    "SUPPLIERS",
    "SUPPLY_SOURCES",
    "FixedSourcePolicy",
    "ForcedSourcePolicy",
    "MatchedSourceControl",
    "Refusal",
    "SourcePolicy",
    "Supplier",
    "SupplierRefused",
    "SupplyConfigError",
    "SupplyError",
    "SupplyOffer",
    "SupplyRequest",
    "build_supplier",
    "build_suppliers",
    "default_suppliers",
    "key_for_prompt",
    "source_code",
    "source_proportions",
]


def _gold_supplier(**kw: Any) -> Supplier:
    """Factory for :class:`selfevo.supply.gold.GoldSupplier`.

    Stateless and dependency-free: everything it reads is already in the batch. Imported
    lazily, like ``compose.ROUTERS``' factories, so this package stays importable for
    configuration validation without pulling torch through every submodule.
    """
    from selfevo.supply.gold import GoldSupplier

    return GoldSupplier(**kw)


def _self_supplier(**kw: Any) -> Supplier:
    """Factory for :class:`selfevo.supply.self_gen.SelfGeneratedSupplier`.

    Requires ``store``. There is deliberately no default store: a supplier constructed with a
    fresh empty one would serve nothing and report as a self-generated arm, which is the exact
    shape of the two arms this repo has already retracted for running bit-identical to the off
    arm. The trainer owns the store, because it must survive across steps.

    Raises:
        SupplyConfigError: If no store is supplied.
    """
    from selfevo.supply.self_gen import SelfGeneratedSupplier

    if "store" not in kw:
        raise SupplyConfigError(
            "source=self needs store=<SolvedRolloutStore>. A fresh empty store would serve "
            "nothing while reporting as a self-generated arm, and the store has to outlive the "
            "batch anyway: its whole premise is a prompt solved at an EARLIER step."
        )
    return SelfGeneratedSupplier(**kw)


def _corpus_supplier(**kw: Any) -> Supplier:
    """Factory for :class:`selfevo.supply.corpus.CorpusSupplier`.

    Requires ``pool``; ``load_corpus_jsonl`` builds one from a file. No default pool, for the
    reason ``_code_policy_router`` gives for having no default policy: a default would silently
    decide the experiment.

    Raises:
        SupplyConfigError: If no pool is supplied.
    """
    from selfevo.supply.corpus import CorpusSupplier

    if "pool" not in kw:
        raise SupplyConfigError(
            "source=corpus needs pool=<mapping>; use selfevo.supply.corpus.load_corpus_jsonl. "
            "There is no default pool because a default would silently decide the experiment."
        )
    return CorpusSupplier(**kw)


def _teacher_supplier(**kw: Any) -> Supplier:
    """Factory for :class:`selfevo.supply.teacher.TeacherSupplier`.

    Requires ``verify``; see that module's docstring for why an unverified teacher completion is
    a wrong target that still looks like a target. ``client`` defaults to None, which is an
    honest teacherless arm: every request is refused with ``Refusal.UNAVAILABLE`` and counted.

    Raises:
        SupplyConfigError: If no verifier is supplied.
    """
    from selfevo.supply.teacher import TeacherSupplier

    kw.setdefault("client", None)
    kw.setdefault("verify", None)
    return TeacherSupplier(kw.pop("client"), **kw)


# The closed registry, one entry per member of SUPPLY_SOURCES. Closed and cross-checked below,
# because `partition_from_config` records what an open one costs: a name with no branch ran the
# CONTROL's mechanism under the new arm's label, and the artifact afterwards could not say
# which had produced the table.
SUPPLIERS: dict[str, Callable[..., Supplier]] = {
    "gold": _gold_supplier,
    "self": _self_supplier,
    "corpus": _corpus_supplier,
    "teacher": _teacher_supplier,
}

if tuple(SUPPLIERS) != SUPPLY_SOURCES:  # pragma: no cover - import-time consistency check
    raise RuntimeError(
        f"SUPPLIERS {tuple(SUPPLIERS)} and SUPPLY_SOURCES {SUPPLY_SOURCES} disagree; a name in "
        "one and not the other is a source that either cannot be built or cannot be named"
    )


def build_supplier(name: str, **kwargs: Any) -> Supplier:
    """Build one supplier by name.

    Args:
        name: A member of :data:`SUPPLY_SOURCES`.
        **kwargs: Forwarded to the factory.

    Returns:
        The supplier.

    Raises:
        SupplyConfigError: For an unknown name, resolved here rather than after model load.
    """
    if name not in SUPPLIERS:
        raise SupplyConfigError(
            f"unknown supply source {name!r}; expected one of {list(SUPPLIERS)}"
        )
    return SUPPLIERS[name](**kwargs)


def build_suppliers(
    spec: Mapping[str, Mapping[str, Any]] | Sequence[str],
) -> dict[str, Supplier]:
    """Build a name-to-supplier mapping for one arm.

    Args:
        spec: Either a sequence of source names, each built with no arguments, or a mapping
            from source name to that factory's keyword arguments.

    Returns:
        ``{name: supplier}`` in :data:`SUPPLY_SOURCES` order, which is the order the counters
        and the ``source_ids`` codes use.

    Raises:
        SupplyConfigError: For an unknown name or a factory that refuses its arguments.
    """
    if not isinstance(spec, Mapping):
        spec = {name: {} for name in spec}
    return {
        name: build_supplier(name, **dict(spec[name]))
        for name in SUPPLY_SOURCES
        if name in spec
    }


def default_suppliers() -> dict[str, Supplier]:
    """The gold-only mapping, which is what the seam uses when no arm asked for more.

    Returns:
        ``{"gold": GoldSupplier()}``. This is the OFF state of this package: the seam behaves
        exactly as it did before suppliers existed, byte for byte, and
        ``test_supply_sources.py`` asserts that against a digest recorded beforehand.
    """
    return {"gold": _gold_supplier()}
