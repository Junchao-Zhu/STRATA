"""External single-cell reference bank.

Loads (embedding, cell_type) pairs from a single-cell embedding directory,
runs KMeans to define cluster centroids, and tracks coverage as spots are
sampled. The bank reports how many new clusters each spot hits.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Each dataset points to a directory of single-cell embeddings.
DEFAULT_SC_ROOTS = {
    "HER2": os.environ.get("SC_BANK_HER2", "./data/sc_embeddings/her2"),
}


def _load_sc_root(root: str, max_cells: int = 100_000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, cell_type_strings)."""
    rng = np.random.default_rng(seed)
    embs: list = []
    cts: list = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(root, fn), allow_pickle=True)
        for d in arr:
            embs.append(d["embedding"])
            cts.append(str(d["cell_type"]))
    E = np.stack(embs).astype(np.float32)
    T = np.asarray(cts)
    if E.shape[0] > max_cells:
        idx = rng.choice(E.shape[0], size=max_cells, replace=False)
        E = E[idx]
        T = T[idx]
    return E, T


class SCBank:
    """Cluster the SC bank once, then track which clusters have been hit."""

    def __init__(self, dataset: str, n_clusters: int = 256, max_cells: int = 100_000, seed: int = 0):
        root = DEFAULT_SC_ROOTS.get(dataset)
        if root is None or not Path(root).exists():
            raise FileNotFoundError(f"No SC bank for dataset {dataset}")
        E, T = _load_sc_root(root, max_cells=max_cells, seed=seed)
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=3)
        km.fit(E)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.centroid_norm = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9
        self.n_clusters = n_clusters
        self.coverage = np.zeros(n_clusters, dtype=bool)

    def reset(self):
        self.coverage[:] = False

    def _nearest_cluster(self, mol: np.ndarray) -> int:
        m = mol / (np.linalg.norm(mol) + 1e-9)
        sims = self.centroids @ m / self.centroid_norm.squeeze(-1)
        return int(np.argmax(sims))

    def update(self, mol_embed: np.ndarray) -> float:
        """Add this spot's molecular embedding, return 1 if it hits a new cluster."""
        c = self._nearest_cluster(mol_embed.astype(np.float32))
        if not self.coverage[c]:
            self.coverage[c] = True
            return 1.0
        return 0.0

    @property
    def coverage_count(self) -> int:
        return int(self.coverage.sum())
