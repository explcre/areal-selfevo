# SPDX-License-Identifier: Apache-2.0

"""Dataset-source routing for multi-teacher on-policy distillation."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from typing import Any

from areal.api.cli_args import _DatasetConfig

MOPD_ROUTE_METADATA_KEY = "__areal_mopd_route"


class MOPDDataset:
    """Concatenate routed sources without modifying their stored samples."""

    def __init__(self, sources: list[tuple[Any, str]]) -> None:
        if not sources:
            raise ValueError("MOPD dataset sources must not be empty")
        self._datasets = [dataset for dataset, _ in sources]
        self._routes = [route for _, route in sources]
        self._offsets: list[int] = []

        from areal.infra.data_service.rdataset import RDataset

        remote = [isinstance(dataset, RDataset) for dataset in self._datasets]
        if any(remote) and not all(remote):
            raise ValueError("MOPD dataset cannot mix local and remote sources")
        self._is_remote = all(remote)
        if not self._is_remote:
            self._refresh_offsets()

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    def _refresh_offsets(self) -> None:
        total = 0
        offsets: list[int] = []
        for dataset in self._datasets:
            total += len(dataset)
            offsets.append(total)
        self._offsets = offsets

    def __len__(self) -> int:
        if not self._offsets:
            self._refresh_offsets()
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)

        source_index = bisect_right(self._offsets, index)
        source_start = 0 if source_index == 0 else self._offsets[source_index - 1]
        sample = self._datasets[source_index][index - source_start]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"MOPD dataset samples must be mappings, got {type(sample).__name__}"
            )
        if MOPD_ROUTE_METADATA_KEY in sample or "mopd_route" in sample:
            raise ValueError("MOPD route must be configured on the dataset source")

        routed_sample = dict(sample)
        routed_sample[MOPD_ROUTE_METADATA_KEY] = self._routes[source_index]
        return routed_sample

    def connect(
        self,
        controller: Any,
        dataset_id: str,
        tokenizer_or_processor_path: str = "",
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        """Connect every remote source to one shared data controller."""
        if not self._is_remote:
            raise RuntimeError("Only remote MOPD datasets require connect()")
        for index, dataset in enumerate(self._datasets):
            dataset.connect(
                controller,
                dataset_id=f"{dataset_id}_source_{index}",
                tokenizer_or_processor_path=tokenizer_or_processor_path,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        self._refresh_offsets()

    def _start_prefetch(self, indices: list[int]) -> None:
        """Translate global sampler indices to each remote source."""
        if not self._is_remote:
            return
        source_indices: list[list[int]] = [[] for _ in self._datasets]
        for index in indices:
            source_index = bisect_right(self._offsets, index)
            source_start = 0 if source_index == 0 else self._offsets[source_index - 1]
            source_indices[source_index].append(index - source_start)
        for dataset, local_indices in zip(self._datasets, source_indices, strict=True):
            dataset._start_prefetch(local_indices)

    def close(self) -> None:
        """Close all remote source proxies."""
        if not self._is_remote:
            return
        for dataset in self._datasets:
            dataset.close()


def is_remote_dataset(dataset: Any) -> bool:
    """Return whether a dataset needs data-service connection and prefetching."""
    from areal.infra.data_service.rdataset import RDataset

    return isinstance(dataset, RDataset) or (
        isinstance(dataset, MOPDDataset) and dataset.is_remote
    )


def get_mopd_dataset(
    dataset_config: _DatasetConfig,
    tokenizer: Any = None,
    processor: Any = None,
    source_loader: Callable[..., Any] | None = None,
) -> MOPDDataset:
    """Load configured sources and attach their mandatory route as metadata."""
    if not dataset_config.sources:
        raise ValueError("MOPD dataset config must contain at least one source")

    if source_loader is None:
        from areal.dataset import get_custom_dataset

        def source_loader(**kwargs):
            source_config = kwargs["source_config"]
            split = kwargs["split"]
            return get_custom_dataset(
                split=split,
                dataset_config=source_config,
                tokenizer=kwargs["tokenizer"],
                processor=kwargs["processor"],
            )

    routed_sources: list[tuple[Any, str]] = []
    for source in dataset_config.sources:
        split = source.split or dataset_config.split
        source_config = _DatasetConfig(
            path=source.path,
            type=source.type,
            split=split,
            max_length=(
                source.max_length
                if source.max_length is not None
                else dataset_config.max_length
            ),
            dataset_kwargs=dataset_config.dataset_kwargs | source.dataset_kwargs,
            scheduling_spec=dataset_config.scheduling_spec,
        )
        dataset = source_loader(
            source=source,
            source_config=source_config,
            split=split,
            tokenizer=tokenizer,
            processor=processor,
        )
        routed_sources.append((dataset, source.route))
    return MOPDDataset(routed_sources)


__all__ = [
    "MOPDDataset",
    "MOPD_ROUTE_METADATA_KEY",
    "get_mopd_dataset",
    "is_remote_dataset",
]
