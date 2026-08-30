"""Packed-sequence support for token-level routing.

AReaL's FSDP engine hands the loss a PACKED microbatch: `loss_mask` and friends are 1-D
`[total_length]`, with `cu_seqlens` `[n_seq + 1]` marking sequence boundaries. The routing
functions were written for `(B, T)` and, given a packed tensor reshaped to `(1, total)`,
read the entire microbatch as a single group and declared 100% of tokens RL-dead. That is
the worst possible failure: not a crash, a confident wrong answer.

This module converts between the two layouts and refuses anything it cannot interpret.
"""

from __future__ import annotations

import torch

__all__ = ["is_packed", "unpack", "repack", "PackedLayoutError"]


class PackedLayoutError(ValueError):
    """Raised when a packed batch cannot be interpreted."""


def is_packed(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> bool:
    """True when `x` is a packed 1-D batch that `cu_seqlens` describes.

    A 1-D tensor WITHOUT cu_seqlens is not packed-but-unknown, it is unusable: there is no
    way to recover sequence boundaries, and guessing produces a single enormous "group".
    """
    return x.ndim == 1 and cu_seqlens is not None


def unpack(x: torch.Tensor, cu_seqlens: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """Convert a packed `[total]` tensor to padded `(n_seq, max_len)`.

    Args:
        x: Packed 1-D tensor.
        cu_seqlens: `[n_seq + 1]` cumulative lengths, starting at 0.
        pad_value: Value written at padding positions.

    Returns:
        `(n_seq, max_len)` tensor, rows right-padded.

    Raises:
        PackedLayoutError: If the layout is inconsistent with the tensor, rather than
            silently producing a batch of the wrong shape.
    """
    if x.ndim != 1:
        raise PackedLayoutError(f"expected a 1-D packed tensor, got shape {tuple(x.shape)}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise PackedLayoutError(f"cu_seqlens must be 1-D with >= 2 entries, got "
                                f"{tuple(cu_seqlens.shape)}")
    cs = cu_seqlens.to(torch.long).tolist()
    if cs[0] != 0:
        raise PackedLayoutError(f"cu_seqlens must start at 0, got {cs[0]}")
    if any(b < a for a, b in zip(cs, cs[1:])):
        raise PackedLayoutError("cu_seqlens is not non-decreasing")
    if cs[-1] != x.numel():
        raise PackedLayoutError(
            f"cu_seqlens ends at {cs[-1]} but the packed tensor holds {x.numel()} elements; "
            "unpacking would silently drop or invent tokens"
        )
    lens = [b - a for a, b in zip(cs, cs[1:])]
    n, m = len(lens), max(lens) if lens else 0
    out = torch.full((n, m), pad_value, dtype=x.dtype, device=x.device)
    for i, (a, b) in enumerate(zip(cs, cs[1:])):
        out[i, : b - a] = x[a:b]
    return out


def repack(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unpack`: `(n_seq, max_len)` back to packed `[total]`.

    Raises:
        PackedLayoutError: On a shape that cannot correspond to the given layout.
    """
    cs = cu_seqlens.to(torch.long).tolist()
    lens = [b - a for a, b in zip(cs, cs[1:])]
    if x.ndim != 2 or x.shape[0] != len(lens):
        raise PackedLayoutError(
            f"expected ({len(lens)}, max_len) to repack, got {tuple(x.shape)}")
    if lens and x.shape[1] < max(lens):
        raise PackedLayoutError(
            f"padded width {x.shape[1]} is shorter than the longest sequence {max(lens)}")
    out = torch.empty(cs[-1], dtype=x.dtype, device=x.device)
    for i, (a, b) in enumerate(zip(cs, cs[1:])):
        out[a:b] = x[i, : b - a]
    return out
