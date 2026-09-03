"""Put the dataset's gold solution into a trajectory, at the trajectory's own width.

Two facts decide everything in this module, and both are measured rather than assumed.

THE COLLATION TRAP. ``check_trajectory_format`` only WARNS when a tensor's second dim
differs from ``input_ids``' (``workflow_executor.py``, "The first dim of tensor ... rather
than the batch size"), and ``concat_padded_tensors`` pads dims 1..N-1 to the per-KEY maximum
across trajectories. So a gold tensor of its own natural length does not fail at collation:
it survives as ``(B, max_gold_len)`` beside an ``(B, T)`` batch, and then
``pack_tensor_dict`` -- which packs only tensors whose ``shape[1] == seq_len`` -- silently
leaves it 2-D and unpacked while every sibling becomes 1-D, so the break lands in the engine,
one stage after the mistake, wearing a shape error that names neither gold nor collation.
The fix is to pad the gold to the TRAJECTORY's own width here, at construction, so that
after collation ``gold_ids`` has exactly ``input_ids``' shape and is packed, split and
unpacked by the same code paths as the tensors beside it.

THE PER-ROW TENSOR TRAP, already paid for once in this repo. ``_compute_advantages`` stores
``group_ids`` per TOKEN and not per sequence, with the comment "a (B,) tensor does not
survive the pipeline ... which is exactly how the first routed run died". Gold is a per-ROW
fact, so the same rule applies to it: everything this module emits is ``(B, T)``.

WHERE THE GOLD IS TOKENISED: at DATASET ADAPTATION, in
``areal/dataset/competition_math.py``, not here. Measured on this box with the live 30B
tokenizer, all 7500 MATH training solutions tokenise in 3.29s ONCE and are then cached by
``datasets`` fingerprinting, whereas encoding at workflow time re-encodes the same solution
for every rollout of every epoch -- 80 times per prompt at the live ``n_samples=8`` over 10
epochs -- inside the async rollout loop of every rollout worker. This module therefore only
places and pads integers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

__all__ = [
    "GOLD_KEYS",
    "GoldError",
    "GoldAttachError",
    "attach_gold",
    "attach_gold_from_data",
    "prompt_lengths",
]

# The keys a gold-carrying trajectory adds. Named once so the workflow, the executor, the
# substitution seam and the tests cannot drift apart on a spelling.
GOLD_KEYS = ("gold_ids", "gold_mask")


class GoldError(Exception):
    """Base for every refusal on the gold path.

    Typed, and every subclass is raised rather than warned, because the failure this whole
    path exists to avoid is the SILENT one: an arm that is configured, logs as a gold arm and
    applies no gold. A warning at step 3 of 900 in a training log is not read.
    """


class GoldAttachError(GoldError):
    """The gold cannot be placed into this trajectory.

    Raised for a malformed trajectory or a gold too long for the row, never for the ordinary
    case of a dataset row that simply has no gold -- that one yields an all-zero
    ``gold_mask``, which keeps the key set uniform across trajectories
    (``concat_padded_tensors`` refuses a batch whose dicts disagree on keys) and is counted
    downstream as a group whose gold was unusable.
    """


def prompt_lengths(
    loss_mask: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-row index of the first RESPONSE token, in TOKEN coordinates.

    Args:
        loss_mask: ``(B, T)``, non-zero on response tokens, as a rollout emits it -- i.e.
            BEFORE ``_compute_advantages`` rolls it left by one. Passing the rolled mask
            gives an answer one position short, which is the defect ``_infer_prompt_lens``
            exists to undo for the other coordinate convention.
        attention_mask: ``(B, T)`` or None. Used only for a row with no response at all,
            where the "first response token" does not exist and the honest answer is the
            row's real length rather than the 0 that ``argmax`` returns on an all-zero row.

    Returns:
        ``(B,)`` long.

    Why this is not read off ``resp.input_len``: by the time a batch reaches the seam that
    spends a gold, the rollout objects are gone and the batch is all that is left. The mask
    is the only surviving statement of where a row's prompt ended.
    """
    lm = loss_mask.long()
    first = lm.argmax(dim=-1)
    has_response = lm.any(dim=-1)
    if attention_mask is None:
        fallback = torch.full_like(first, lm.shape[-1])
    else:
        fallback = attention_mask.long().sum(-1)
    return torch.where(has_response, first, fallback)


def attach_gold(
    traj: dict[str, Any],
    gold_ids: Sequence[int] | torch.Tensor | None,
) -> dict[str, Any]:
    """Add ``gold_ids``/``gold_mask`` to a trajectory, padded to the trajectory's width.

    Args:
        traj: A trajectory dict as a workflow returns it: every tensor ``(n, T)`` except the
            per-row scalars, and ``input_ids`` present. Not mutated.
        gold_ids: The gold solution's token ids, already tokenised by the dataset adapter.
            ``None`` or empty means this row has no usable gold, which is a legitimate state
            and produces an all-zero ``gold_mask`` rather than a missing key.

    Returns:
        A new dict with two extra ``(n, T)`` tensors:

        * ``gold_ids`` -- the gold token ids, LEFT-ALIGNED at column 0 and right-padded with
          zeros. Left-aligned rather than placed after the prompt because a gold is a
          property of the dataset ROW, shared by every rollout of the group, while a prompt
          offset is a property of an individual rollout; keeping the two apart means the same
          gold tensor is identical across the group's rows and the splice happens once, at
          the substitution seam, where the victim row's prompt is known.
        * ``gold_mask`` -- 1 on the real gold tokens, 0 on padding, so ``gold_mask.sum(-1)``
          is the gold's true length and no consumer has to guess whether a 0 is a pad or the
          token id 0.

    Raises:
        GoldAttachError: If ``input_ids`` is missing or not 2-D, or if the gold is longer
            than the trajectory's width. The second case is a REFUSAL and not a truncation on
            purpose: a truncated derivation is a wrong target that still looks like a target,
            and training on it is worse than not training on it. Callers that want the run to
            proceed catch this and attach an empty gold; the substitution seam then counts
            the group as one whose gold was unusable and says so in its stats.
    """
    ids = traj.get("input_ids")
    if not torch.is_tensor(ids) or ids.ndim != 2:
        raise GoldAttachError(
            "attach_gold needs a trajectory with a 2-D input_ids tensor; got "
            f"{type(ids).__name__} with shape {getattr(ids, 'shape', None)}"
        )
    n, width = int(ids.shape[0]), int(ids.shape[1])

    if gold_ids is None:
        flat: list[int] = []
    elif torch.is_tensor(gold_ids):
        flat = [int(v) for v in gold_ids.reshape(-1).tolist()]
    else:
        flat = [int(v) for v in gold_ids]

    if len(flat) > width:
        raise GoldAttachError(
            f"gold is {len(flat)} tokens but the trajectory is {width} wide. Refusing to "
            "truncate: a cut-off derivation is a wrong target that still looks like a "
            "target. Attach an empty gold for this row instead."
        )

    row_ids = torch.zeros(width, dtype=ids.dtype)
    row_mask = torch.zeros(width, dtype=torch.int32)
    if flat:
        row_ids[: len(flat)] = torch.tensor(flat, dtype=ids.dtype)
        row_mask[: len(flat)] = 1

    out = dict(traj)
    out["gold_ids"] = row_ids.unsqueeze(0).expand(n, width).contiguous()
    out["gold_mask"] = row_mask.unsqueeze(0).expand(n, width).contiguous()
    return out


def attach_gold_from_data(
    traj: dict[str, Any] | None,
    data: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach the gold a dataset row carries, or return the trajectory untouched.

    This is the whole of the workflow-side seam, so that the RLVR workflow and the
    OpenAI-proxy workflow -- which assemble their tensors in different files, and which no
    single edit can serve -- both reach the same tested function instead of two copies of the
    same padding arithmetic.

    Args:
        traj: The trajectory a workflow produced, or None for a rejected one.
        data: The dataset row the rollout was run on. ``gold_ids`` is present only when the
            dataset adapter was asked for it (``keep_solution=True``), so a run that did not
            ask for gold takes the early return and is bit-identical to one built before this
            function existed.

    Returns:
        The trajectory, with gold attached when there was gold to attach.

    Raises:
        GoldAttachError: Propagated from :func:`attach_gold` for a malformed trajectory. A
            gold too long for the row is NOT propagated: it is the ordinary consequence of a
            long derivation meeting a short rollout, and it is recorded as an empty gold so
            the key set stays uniform across the batch and the loss of reach is counted at
            the substitution seam rather than killing a rollout worker.
    """
    if traj is None or not isinstance(traj, dict) or not data:
        return traj
    if "gold_ids" not in data:
        return traj
    if not torch.is_tensor(traj.get("input_ids")):
        return traj
    try:
        return attach_gold(traj, data.get("gold_ids"))
    except GoldAttachError as exc:
        if "Refusing to truncate" not in str(exc):
            raise
        return attach_gold(traj, None)
