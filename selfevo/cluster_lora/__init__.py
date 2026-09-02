"""Per-cluster LoRA experts, routed by discovered behavioural clusters during GRPO.

The method in one line: cluster prompt GROUPS by the model's own behaviour (MEDS-style
HDBSCAN over the latter-half layer-wise logits at the final answer token), keep one LoRA
adapter per cluster plus a SHARED adapter for HDBSCAN noise, let each group's GRPO loss
update only its own cluster's adapter, and merge the adapters at inference.

The hypothesis the paper rests on is mechanical, not aesthetic: behavioural subpopulations
inside one task want CONFLICTING parameter updates, a single shared adapter averages them,
and the average is worth less than either. That is a claim about gradients, so it is
measurable before any training run -- :mod:`selfevo.cluster_lora.interference_dump` and
:mod:`selfevo.cluster_lora.interference_analyze` measure it, and the size-matched random
partition in :mod:`selfevo.cluster_lora.partition` is what separates "clustering helped"
from "more adapters helped".

Nothing here imports torch at module import time except the modules that genuinely need it,
so the partition and reporting layers stay testable on a CPU box with no model present.
"""

from __future__ import annotations

__all__ = [
    "BehaviourFeatureUnavailable",
    "ClusterLoRAKeyFn",
    "Partition",
    "PartitionUnavailable",
    "ReachReport",
    "SHARED_CLUSTER",
    "cluster_key",
    "meds_partition",
    "no_partition",
    "partition_from_config",
    "random_matched_partition",
    "reach_report",
    "sketch_dim_resolution",
    "sketch_vector",
]

from .partition import (
    SHARED_CLUSTER,
    Partition,
    PartitionUnavailable,
    cluster_key,
    meds_partition,
    no_partition,
    partition_from_config,
    random_matched_partition,
)
from .reach import ReachReport, reach_report
from .sketch import sketch_dim_resolution, sketch_vector


def __getattr__(name: str):
    """Defer the torch-dependent names so importing this package needs no torch.

    The partition, reach and sketch layers are pure numpy and are exercised on boxes that
    have no model at all; making the package import ``torch`` eagerly would make those
    tests impossible to run where they are cheapest.
    """
    if name in ("BehaviourFeatureUnavailable", "ClusterLoRAKeyFn"):
        from . import features

        return getattr(features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
