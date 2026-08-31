"""MEDS trajectory clustering, as a routing feature.

MEDS (arXiv 2604.11297) clusters reasoning behaviour by treating layer-wise logits from the
forward pass as lightweight representations and grouping similar error patterns with HDBSCAN.
That is the learned counterpart to the *derived* cluster key this project already has (the
silence side of a group), and the two together make "how to cluster" a controlled ablation
rather than a design choice.

**The clustering functions below are vendored VERBATIM from the authors' implementation**,
`recipe/meds/layer_logits_utils.py` in the MEDS repository -- `_cluster_with_hdbscan` and
`_classify_with_knn`, unchanged including their defaults (`min_cluster_size=2`,
`min_samples=1`, L2 normalisation before Euclidean HDBSCAN, cosine kNN with k=3 and a
majority vote). Reimplementing them would have made every difference in a comparison
ambiguous between "our clustering is different" and "our clustering is wrong".

**What is ours** is the wrapper: the two-phase fit/assign lifecycle, the guards, and the
adaptation to a routing feature rather than a reward shaper.

**Dependencies are imported lazily and are not installed on the training boxes.** ``hdbscan``
and ``scikit-learn`` are absent from the venvs that are currently running jobs, and
installing them there could pull a different numpy or scipy under a live process. The import
therefore happens inside the call, and the error names the exact install command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MEDSClusterer", "ClusteringUnavailable"]


class ClusteringUnavailable(ImportError):
    """Raised when hdbscan/scikit-learn are not installed, naming how to install them."""


def _require_deps():
    """Import the clustering dependencies, or explain precisely what is missing.

    Prefers the standalone ``hdbscan`` package, which is what MEDS imports. Falls back to
    ``sklearn.cluster.HDBSCAN``, which is the same algorithm upstreamed into scikit-learn --
    the fallback is used so the vendored logic can be exercised on a box without the
    standalone package, and :attr:`MEDSClusterer.backend` records which one ran, because
    "clustering differed" and "clustering library differed" must not be confusable in a
    comparison.

    Returns:
        ``(hdbscan_factory, NearestNeighbors, normalize, backend_name)``.

    Raises:
        ClusteringUnavailable: If neither backend nor scikit-learn is importable.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        raise ClusteringUnavailable(
            f"MEDS clustering needs scikit-learn ({exc}). Install with "
            "`pip install hdbscan scikit-learn`. They are deliberately NOT installed in the "
            "training venvs: resolving them can pull a different numpy or scipy underneath a "
            "running job."
        ) from exc

    try:
        import hdbscan as _hdb

        return _hdb.HDBSCAN, NearestNeighbors, normalize, "hdbscan"
    except ImportError:
        pass
    try:
        from sklearn.cluster import HDBSCAN as _SkHDBSCAN

        return _SkHDBSCAN, NearestNeighbors, normalize, "sklearn"
    except ImportError as exc:
        raise ClusteringUnavailable(
            f"neither hdbscan nor sklearn.cluster.HDBSCAN is available ({exc}); "
            "`pip install hdbscan` or upgrade scikit-learn to >= 1.3"
        ) from exc


# --------------------------------------------------------------------------------------
# Vendored from MEDS, recipe/meds/layer_logits_utils.py. Bodies unchanged except that the
# module-level imports they relied on are passed in, so this file has no hard dependency.
# --------------------------------------------------------------------------------------


def _cluster_with_hdbscan(
    state: dict,
    min_cluster_size: int = 2,
    use_l2_normalize: bool = True,
    metric: str = "euclidean",
) -> None:
    """Use HDBSCAN density-based clustering. Verbatim from MEDS.

    Args:
        state: KNN state dictionary with 'vectors' key
        min_cluster_size: Minimum cluster size for HDBSCAN
        use_l2_normalize: Kept for signature fidelity with the original.
        metric: HDBSCAN metric.
    """
    hdbscan_cls, _NearestNeighbors, normalize, backend = _require_deps()
    state["backend"] = backend
    vectors = state.get("vectors", [])
    if len(vectors) < 2:
        state["labels"] = [-1] * len(vectors)
        state["k"] = 0
        return

    X = np.stack(vectors, axis=0)
    X_used = normalize(X, norm="l2")

    clusterer = hdbscan_cls(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric=metric,
    )
    labels = clusterer.fit_predict(X_used)

    state["labels"] = labels.tolist()
    unique_labels = np.unique(labels)
    non_noise_labels = unique_labels[unique_labels != -1]
    state["k"] = int(len(non_noise_labels))


def _classify_with_knn(state: dict, vec_np: np.ndarray, k: int = 3, metric: str = "cosine") -> int:
    """Assign a new vector to an existing cluster by kNN majority vote. Verbatim from MEDS."""
    _hdbscan_cls, NearestNeighbors, _normalize, _backend = _require_deps()
    X = np.stack(state["vectors"], axis=0)
    labels = np.array(state["labels"], dtype=int)

    if len(X) == 0:
        return 0

    k_actual = min(k, len(X))
    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm="brute")
    nn.fit(X)

    distances, indices = nn.kneighbors(vec_np.reshape(1, -1))
    nn_labels = labels[indices[0]]

    unique, counts = np.unique(nn_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


# --------------------------------------------------------------------------------------
# Ours: the lifecycle a routing feature needs.
# --------------------------------------------------------------------------------------


@dataclass
class MEDSClusterer:
    """Fit clusters over observed behaviour, then assign new units to them.

    Two phases because routing needs both: the cluster structure is fitted once over a
    buffer of observed units, and each incoming unit is then assigned cheaply by kNN. Fitting
    per batch would give cluster ids that mean something different every step, which no
    downstream policy could learn from.

    Args:
        min_cluster_size: Passed to HDBSCAN. MEDS' default is 2.
        metric: HDBSCAN metric. MEDS' default is euclidean, after L2 normalisation.
        knn_k: Neighbours for assignment. MEDS' default is 3.
        knn_metric: Assignment metric. MEDS' default is cosine.
        max_buffer: Cap on retained vectors, so a long run does not grow without bound.

    Raises:
        ValueError: If any parameter is out of range.
    """

    min_cluster_size: int = 2
    metric: str = "euclidean"
    knn_k: int = 3
    knn_metric: str = "cosine"
    max_buffer: int = 4096

    _state: dict = field(default_factory=lambda: {"vectors": [], "labels": [], "k": 0}, repr=False)
    fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        if self.knn_k < 1:
            raise ValueError(f"knn_k must be >= 1, got {self.knn_k}")
        if self.max_buffer < 2:
            raise ValueError(f"max_buffer must be >= 2, got {self.max_buffer}")

    @property
    def backend(self) -> str:
        """Which HDBSCAN implementation the last fit used: ``hdbscan`` or ``sklearn``."""
        return str(self._state.get("backend", "none"))

    @property
    def n_clusters(self) -> int:
        """Number of non-noise clusters found by the last fit."""
        return int(self._state.get("k", 0))

    def add(self, vector: np.ndarray) -> None:
        """Buffer one behaviour representation for the next fit."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size == 0:
            raise ValueError("representation must be non-empty")
        if not np.isfinite(v).all():
            raise ValueError("representation contains NaN or inf; it would corrupt the fit")
        buf = self._state["vectors"]
        buf.append(v)
        if len(buf) > self.max_buffer:
            del buf[: len(buf) - self.max_buffer]

    def fit(self) -> int:
        """Cluster the buffer with MEDS' HDBSCAN settings.

        Returns:
            The number of non-noise clusters.

        Raises:
            ClusteringUnavailable: If hdbscan/scikit-learn are missing.
        """
        _cluster_with_hdbscan(
            self._state, min_cluster_size=self.min_cluster_size, metric=self.metric
        )
        self.fitted = True
        return self.n_clusters

    def assign(self, vector: np.ndarray) -> int:
        """Cluster id for one unit.

        Returns ``-1`` before any fit, which is HDBSCAN's own noise label -- a caller that
        forgets to fit gets the "no cluster" value rather than a plausible-looking 0 that
        would silently become a real feature.
        """
        if not self.fitted or not self._state["vectors"]:
            return -1
        v = np.asarray(vector, dtype=np.float64).ravel()
        return _classify_with_knn(self._state, v, k=self.knn_k, metric=self.knn_metric)
