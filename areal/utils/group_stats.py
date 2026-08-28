# SPDX-License-Identifier: Apache-2.0

"""Pure-observer instrumentation for per-rollout-group reward/advantage stats.

The point of this module is to answer "how many groups carried any learning
signal at all?".  Under group-relative normalization (GRPO and friends) a group
whose members all share the same reward is *silent*: centering maps it to zero
and it contributes nothing to the gradient.  :class:`GroupStatsRecorder` snaps
a picture of every group right before normalization happens so that silent /
all-zero / all-one rates and the between-vs-within variance split can be read
off after the fact.

Everything here is read-only with respect to training.  The recorder never
writes to the tensors it is handed and nothing it computes is fed back into
:class:`areal.utils.data.Normalization`; attaching one must leave the
normalized output bitwise unchanged.
"""

import json
import os
from dataclasses import asdict, dataclass

import torch

#: Number of equal-width bins used by :meth:`GroupStatsRecorder.summary` for the
#: ``p_hat`` histogram.  Bin ``i`` covers ``[i / B, (i + 1) / B)``; the final bin
#: is closed so that ``p_hat == 1.0`` lands in it.
P_HAT_HIST_BINS = 10


@dataclass
class GroupStats:
    """Statistics of a single rollout group, measured before normalization.

    All fields describe the group's *raw* values (rewards or advantages) after
    reducing any trailing dimensions to one scalar per sample; see
    :meth:`GroupStatsRecorder.record` for that reduction.

    Attributes:
        step: Training step the group was observed at, or ``None`` when the
            caller did not supply one.  Purely a label; nothing keys off it.
        group_index: Position of the group in the ``group_slices`` list it came
            from.  Not a stable identifier across steps.
        size: Number of samples the group's slice actually selected.  May be 0
            for a degenerate slice.
        n_positive: Number of samples with a strictly positive value.
        p_hat: ``n_positive / size``, i.e. the empirical success rate for a
            binary reward.  Defined as ``0.0`` for an empty group.
        is_silent: True when every member holds the same value, so group
            centering zeroes the whole group.  Compared *exactly*, with no
            tolerance, so float noise counts as non-silent and any NaN member
            makes the group non-silent.  Groups of size 0 or 1 are silent by
            convention: they have no peer to contrast against.
        reward_mean: Mean of the group's values (``0.0`` for an empty group).
        reward_std: Population standard deviation (``correction = 0``) of the
            group's values, so a singleton group reports ``0.0`` rather than
            NaN.  ``0.0`` for an empty group.

    This is a plain record.  It does not guarantee that the values are
    comparable across steps, that ``n_positive`` is meaningful for non-binary
    rewards, or that loss masking was taken into account -- it was not.
    """

    step: int | None
    group_index: int
    size: int
    n_positive: int
    p_hat: float
    is_silent: bool
    reward_mean: float
    reward_std: float


class GroupStatsRecorder:
    """Collects :class:`GroupStats` for every group seen during normalization.

    Typical use is to attach one to :class:`areal.utils.data.Normalization`,
    which calls :meth:`record` once per ``__call__`` with the group slices it
    was about to normalize over.

    What it guarantees:

    * It never mutates the tensor passed to :meth:`record`, and never returns
      anything that could steer normalization.  Attaching a recorder leaves the
      normalized output bitwise identical.
    * A degenerate group (empty slice, singleton, all-NaN) is recorded rather
      than raised on.

    What it does *not* guarantee:

    * **Not thread-safe.**  Single writer only: the records list and the flush
      cursor are mutated without any lock.  Sharing one recorder across threads
      or across concurrently-running trainers will interleave or lose records.
    * It ignores ``loss_mask``.  For a ``(B, T)`` input the per-sample value is
      a plain mean over all trailing positions, padding included.
    * It holds every record in memory until :meth:`reset` is called; call
      :meth:`flush` periodically and :meth:`reset` to bound growth.
    * Moving statistics to host memory forces a device synchronization when
      ``x`` lives on an accelerator, so recording is not free.
    """

    def __init__(self, out_path: str | None = None, enabled: bool = False):
        """Create a recorder.

        Args:
            out_path: JSONL file that :meth:`flush` appends to.  ``None``
                (the default) makes :meth:`flush` a no-op; records are still
                accumulated in memory and readable via :attr:`records` and
                :meth:`summary`.
            enabled: Master switch.  While False, :meth:`record` returns
                immediately and nothing is collected.  Defaults to False so
                that an accidentally-attached recorder costs nothing.
        """
        self.out_path = out_path
        self.enabled = enabled
        self.records: list[GroupStats] = []
        # Index of the first record not yet written by ``flush``.
        self._flushed = 0

    @torch.no_grad()
    def record(
        self,
        x: torch.Tensor,
        group_slices: list[slice],
        step: int | None = None,
    ) -> None:
        """Append one :class:`GroupStats` per entry of ``group_slices``.

        ``x`` is treated as read-only.  Inputs of shape ``(B,)`` are used as
        is; inputs of shape ``(B, T)`` (or more dims) are reduced to one scalar
        per sample by taking the mean over the trailing dimensions, ignoring
        any loss mask.  Statistics are computed in float64.

        Args:
            x: The values about to be normalized, batch-major.
            group_slices: Slices into ``x``'s first dimension, one per group,
                as produced by ``Normalization._build_group_slices``.
            step: Optional training-step label stored on each record.  Typed as
                optional because ``Normalization.__call__`` defaults it to
                ``None``.

        Returns:
            None.  This is an observer; it has no effect on the caller.

        Does not raise on degenerate groups: an empty slice is recorded with
        ``size = 0``.  Does nothing at all when :attr:`enabled` is False.
        """
        if not self.enabled:
            return

        values = x.detach().to(torch.float64)
        if values.ndim > 1:
            values = values.mean(dim=tuple(range(1, values.ndim)))
        # Reshape covers the 0-d case; ``.cpu()`` is a no-op for host tensors.
        values = values.reshape(-1).cpu()

        for group_index, group_slice in enumerate(group_slices):
            v = values[group_slice]
            size = int(v.numel())
            if size == 0:
                self.records.append(
                    GroupStats(
                        step=step,
                        group_index=group_index,
                        size=0,
                        n_positive=0,
                        p_hat=0.0,
                        is_silent=True,
                        reward_mean=0.0,
                        reward_std=0.0,
                    )
                )
                continue
            n_positive = int((v > 0).sum().item())
            self.records.append(
                GroupStats(
                    step=step,
                    group_index=group_index,
                    size=size,
                    n_positive=n_positive,
                    p_hat=n_positive / size,
                    is_silent=size == 1 or bool((v == v[0]).all().item()),
                    reward_mean=float(v.mean().item()),
                    reward_std=float(v.std(unbiased=False).item()),
                )
            )

    def summary(self) -> dict:
        """Aggregate every record collected so far.

        Returns a dict with:

        * ``n_groups``: number of records, the denominator of every rate below.
        * ``silent_rate``: fraction of groups whose members are all equal.
        * ``all_zero_rate`` / ``all_one_rate``: fraction of groups with
          ``p_hat == 0`` / ``p_hat == 1``, i.e. no member positive / every
          member positive.
        * ``p_hat_hist``: counts of ``p_hat`` in :data:`P_HAT_HIST_BINS` equal
          bins over ``[0, 1]``, last bin closed.
        * ``between_group_var``: population variance of the per-group means.
        * ``within_group_var``: mean of the per-group population variances.

        The last two are the variance decomposition of the reward signal:
        ``between_group_var`` is what group-relative normalization throws away
        and ``within_group_var`` is what it keeps.

        ``between_group_var`` is **not bias-corrected for sampling noise**.
        Each group mean is itself estimated from finitely many samples, so its
        sampling variance (roughly ``within_group_var / size``) is baked into
        the number; it therefore *overstates* the true between-group variance,
        and the overstatement grows as groups get smaller.  Subtract
        ``within_group_var / size`` yourself if you need an unbiased figure.

        Degenerate groups are counted like any other: an empty group reports
        ``p_hat = 0.0`` and ``is_silent = True``, so it inflates
        ``silent_rate`` and ``all_zero_rate``.  With slices built by
        ``Normalization._build_group_slices`` empty groups cannot occur.

        With no records, every rate and variance is ``0.0`` and the histogram
        is all zeros; nothing is normalized by zero.
        """
        n_groups = len(self.records)
        hist = [0] * P_HAT_HIST_BINS
        if n_groups == 0:
            return {
                "n_groups": 0,
                "silent_rate": 0.0,
                "all_zero_rate": 0.0,
                "all_one_rate": 0.0,
                "p_hat_hist": hist,
                "between_group_var": 0.0,
                "within_group_var": 0.0,
            }

        n_silent = 0
        n_all_zero = 0
        n_all_one = 0
        sum_mean = 0.0
        sum_mean_sq = 0.0
        sum_var = 0.0
        for r in self.records:
            n_silent += int(r.is_silent)
            n_all_zero += int(r.p_hat == 0.0)
            n_all_one += int(r.p_hat == 1.0)
            sum_mean += r.reward_mean
            sum_mean_sq += r.reward_mean * r.reward_mean
            sum_var += r.reward_std * r.reward_std
            bin_index = min(int(r.p_hat * P_HAT_HIST_BINS), P_HAT_HIST_BINS - 1)
            hist[max(bin_index, 0)] += 1

        mean_of_means = sum_mean / n_groups
        between_group_var = max(sum_mean_sq / n_groups - mean_of_means**2, 0.0)
        return {
            "n_groups": n_groups,
            "silent_rate": n_silent / n_groups,
            "all_zero_rate": n_all_zero / n_groups,
            "all_one_rate": n_all_one / n_groups,
            "p_hat_hist": hist,
            "between_group_var": between_group_var,
            "within_group_var": sum_var / n_groups,
        }

    def flush(self) -> None:
        """Append not-yet-written records to :attr:`out_path` as JSONL.

        One JSON object per line, in the order recorded.  Creates the parent
        directory if needed and opens the file in append mode, so repeated
        flushes extend the same file and a re-run adds to it rather than
        replacing it.  Records already written are tracked and never written
        twice; the in-memory records are kept so :meth:`summary` still sees
        them (use :meth:`reset` to drop them).

        A no-op when :attr:`out_path` is ``None`` or when there is nothing new.
        Non-finite values are written as JSON's ``NaN``/``Infinity`` extension,
        which Python's :mod:`json` reads back but a strict parser will reject.
        Not atomic and not safe against concurrent writers.
        """
        if self.out_path is None or self._flushed >= len(self.records):
            return
        directory = os.path.dirname(self.out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        pending = self.records[self._flushed :]
        with open(self.out_path, "a") as f:
            for r in pending:
                f.write(json.dumps(asdict(r)) + "\n")
        self._flushed = len(self.records)

    def reset(self) -> None:
        """Drop all in-memory records and the flush cursor.

        Does not touch :attr:`out_path`: anything already flushed stays on
        disk, and records dropped before a :meth:`flush` are lost.
        """
        self.records = []
        self._flushed = 0
