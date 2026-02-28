"""Step1 hyperbolic-vs-UMAP style metric sanity tests."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _knn_overlap_score(base: np.ndarray, emb: np.ndarray, k: int = 10) -> float:
    nn_base = NearestNeighbors(n_neighbors=k + 1).fit(base)
    nn_emb = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx_base = nn_base.kneighbors(return_distance=False)[:, 1:]
    idx_emb = nn_emb.kneighbors(return_distance=False)[:, 1:]
    overlap = [len(set(a) & set(b)) / float(k) for a, b in zip(idx_base, idx_emb)]
    return float(np.mean(overlap))


def test_step1_hyperbolic_like_embedding_better_than_random_projection():
    rng = np.random.default_rng(42)
    # two clearly separated clusters in "raw" space
    a = rng.normal(loc=-2.0, scale=0.4, size=(120, 6))
    b = rng.normal(loc=2.0, scale=0.4, size=(120, 6))
    raw = np.vstack([a, b])
    y = np.array([0] * len(a) + [1] * len(b))

    # "Hyperbolic-like" retains structure using first dims
    emb_good = raw[:, :2]
    # "UMAP-like poor case": shuffled/randomized embedding baseline
    emb_bad = rng.normal(size=(raw.shape[0], 2))

    sil_good = silhouette_score(emb_good, y)
    sil_bad = silhouette_score(emb_bad, y)
    knn_good = _knn_overlap_score(raw, emb_good, k=10)
    knn_bad = _knn_overlap_score(raw, emb_bad, k=10)

    assert sil_good > sil_bad
    assert knn_good > knn_bad
