"""A linear random-projection sketch of a gradient, so any partition is free after one dump.

The measurement the method rests on is the pairwise angle between PER-CLUSTER LoRA
gradients. A cluster's gradient is the SUM of its member groups' gradients (exactly, when
every group's loss carries the same denominator -- see
:func:`selfevo.cluster_lora.interference_dump.group_losses`), so if we store one vector per
GROUP we can form any partition afterwards without touching a GPU again. That is the whole
reason the probe is split into a dump and an analysis: four partitions, one forward-backward
pass over the batch.

Storing the full gradient per group is not affordable -- LoRA r=16 over all-linear on a 32B
model is order 1e8 parameters, so 64 groups would be ~25 GB in fp32. A **linear** sketch is,
and linearity is not a convenience here, it is the correctness condition:

    sketch(g_a + g_b) == sketch(g_a) + sketch(g_b)

is what makes "sum the member groups' sketches" equal "sketch the cluster's gradient". A
non-linear compression (top-k, quantisation, normalisation) would break that silently, and
the resulting cosines would describe nothing.

The sketch is a **CountSketch**: each parameter coordinate is hashed to one of ``dim``
buckets with a random sign, and buckets accumulate. It is linear by construction, unbiased
on inner products, and costs one pass with no dense projection matrix -- a dense Gaussian
projection of 1e8 x 8192 is 3 TB and cannot be formed at all.

**The resolution floor is the honest limit of this instrument and is reported, not hidden.**
The sketched inner product of two unit vectors has standard error about ``1/sqrt(dim)``, so
at the default ``dim=8192`` a cosine below roughly 0.033 is indistinguishable from zero.
A published cross-task figure of ~1e-5 therefore CANNOT be confirmed from sketches at this
dimension; it can only be reported as "below the floor". That is why
:mod:`selfevo.cluster_lora.interference_dump` also writes the full unprojected gradient for
the first few groups: those pairs get an exact cosine, and the analysis reports the sketch
error against them rather than assuming the sketch was fine.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "SketchPlan",
    "sketch_dim_resolution",
    "sketch_vector",
    "sketch_torch",
]

# Mixed into every per-block seed so a caller's ``seed=0`` is not the same stream as numpy's
# default, which would make an accidentally-unseeded run look reproducible.
_SEED_SALT = 0x5C1E7C4


def sketch_dim_resolution(dim: int, n_sigma: float = 3.0) -> float:
    """Smallest cosine this sketch dimension can tell apart from zero.

    Args:
        dim: Sketch dimension.
        n_sigma: How many standard errors count as "resolved". 3.0 is the default so a
            reported non-zero is not a one-in-twenty fluctuation.

    Returns:
        The resolution floor on a cosine. A measured ``|cos|`` below this is noise and must
        be reported as such rather than as a small number.

    Raises:
        ValueError: If ``dim`` is not positive.
    """
    if dim <= 0:
        raise ValueError(f"sketch dim must be positive, got {dim}")
    return float(n_sigma / math.sqrt(dim))


class SketchPlan:
    """The hashing decided once, so every group is sketched by the SAME projection.

    Two groups sketched under different hashes have uncorrelated sketches and their cosine
    is zero whatever their gradients did -- the single most dangerous silent failure this
    file can have, because the answer it produces (no conflict anywhere) is a publishable
    looking result. The plan is therefore keyed on the parameter NAME and its length, and
    :meth:`block` raises if the same name arrives with a different length.

    Args:
        dim: Sketch dimension.
        seed: Base seed. The same ``(name, length, dim, seed)`` always yields the same
            hash, on any machine and in any process, because the streams come from
            ``np.random.default_rng`` seeded per block rather than from global state.

    Raises:
        ValueError: If ``dim`` is not positive.
    """

    def __init__(self, dim: int = 8192, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError(f"sketch dim must be positive, got {dim}")
        self.dim = int(dim)
        self.seed = int(seed)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def block(self, name: str, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Bucket indices and signs for one parameter tensor.

        Cached, because the same parameter is sketched once per group and regenerating the
        stream would dominate the cost for a batch of 64 groups.

        Args:
            name: Parameter name. Part of the hash key, so two tensors of the same length
                do not share a hash and collide systematically.
            length: Number of coordinates in the tensor.

        Returns:
            ``(indices, signs)``, both of length ``length``.

        Raises:
            ValueError: If ``name`` was previously used with a different length, which
                means the caller changed the parameter layout between groups and the
                sketches are no longer comparable.
        """
        cached = self._cache.get(name)
        if cached is not None:
            if cached[0].shape[0] != length:
                raise ValueError(
                    f"parameter {name!r} was sketched with length {cached[0].shape[0]} and "
                    f"is now {length}; sketches taken under different layouts are not "
                    "comparable and their cosine would be meaningless"
                )
            return cached
        # A per-name seed derived deterministically from the base seed. Python's hash() is
        # salted per process and would make the projection differ between the dump and any
        # re-dump, so the name is folded in with a stable digest instead.
        digest = int.from_bytes(
            __import__("hashlib").blake2b(name.encode(), digest_size=8).digest(), "little"
        )
        rng = np.random.default_rng((self.seed ^ _SEED_SALT ^ digest) & ((1 << 63) - 1))
        idx = rng.integers(0, self.dim, size=length, dtype=np.int64)
        sign = rng.integers(0, 2, size=length, dtype=np.int8) * 2 - 1
        self._cache[name] = (idx, sign)
        return idx, sign


def sketch_vector(
    blocks: Iterable[Tuple[str, np.ndarray]],
    plan: SketchPlan,
    *,
    dtype: np.dtype | str = np.float64,
) -> np.ndarray:
    """CountSketch of a named collection of arrays into one vector. The reference path.

    Args:
        blocks: ``(name, array)`` pairs. Arrays are flattened; only their contents matter.
        plan: The shared hashing. Passing a fresh plan per group is the failure this type
            exists to make visible.
        dtype: Accumulator dtype. float64 by default: the sketch sums order 1e8 terms into
            8192 buckets, and float32 accumulation there loses about three digits.

    Returns:
        A ``(plan.dim,)`` array.

    Raises:
        ValueError: If a block is empty, or contains a non-finite value -- a NaN gradient
            propagates into every downstream cosine and would be reported as a number.
    """
    out = np.zeros(plan.dim, dtype=dtype)
    seen = False
    for name, arr in blocks:
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 0:
            raise ValueError(f"block {name!r} is empty; an empty gradient block means the "
                             "parameter was not in the graph and the sketch would silently "
                             "omit it")
        if not np.isfinite(flat).all():
            raise ValueError(
                f"block {name!r} holds a non-finite value; a NaN here becomes a NaN cosine "
                "and would be reported as a measurement"
            )
        idx, sign = plan.block(name, flat.size)
        np.add.at(out, idx, sign * flat.astype(dtype, copy=False))
        seen = True
    if not seen:
        raise ValueError("no blocks were sketched; an all-zero sketch is not a gradient")
    return out


def sketch_torch(blocks, plan: SketchPlan):
    """The same sketch on torch tensors, so a GPU gradient is never copied to host in full.

    Uses the SAME ``plan`` -- the indices and signs are generated in numpy and moved to the
    tensor's device -- so this is the identical projection, not a parallel implementation
    that has to be trusted to agree. ``test_cluster_lora_sketch.py`` asserts the agreement
    anyway, because "identical by construction" is exactly the claim that has been wrong
    here before.

    Args:
        blocks: ``(name, tensor)`` pairs.
        plan: The shared hashing.

    Returns:
        A ``(plan.dim,)`` float64 CPU numpy array, so the dump and the analysis compare
        like with like regardless of the compute dtype.

    Raises:
        ValueError: On an empty or non-finite block, for the same reasons as
            :func:`sketch_vector`.
    """
    import torch

    out = torch.zeros(plan.dim, dtype=torch.float64)
    seen = False
    for name, tensor in blocks:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            raise ValueError(f"block {name!r} is empty; the parameter was not in the graph")
        if not torch.isfinite(flat).all():
            raise ValueError(f"block {name!r} holds a non-finite value")
        idx, sign = plan.block(name, int(flat.numel()))
        idx_t = torch.from_numpy(idx).to(flat.device)
        sign_t = torch.from_numpy(sign).to(flat.device, dtype=torch.float64)
        contrib = torch.zeros(plan.dim, dtype=torch.float64, device=flat.device)
        contrib.index_add_(0, idx_t, sign_t * flat.to(torch.float64))
        out += contrib.cpu()
        seen = True
    if not seen:
        raise ValueError("no blocks were sketched; an all-zero sketch is not a gradient")
    return out.numpy()
