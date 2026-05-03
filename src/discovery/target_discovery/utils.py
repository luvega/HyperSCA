"""Shared helper functions for target discovery."""
from __future__ import annotations

import numpy as np


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    out = np.array(adj, dtype=float, copy=True)
    if out.size == 0:
        return out
    np.fill_diagonal(out, 0.0)
    mx = float(out.max())
    return out / mx if mx > 0 else out


def knn_adjacency(dist: np.ndarray, k: int) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    k = max(1, min(int(k), n - 1))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    finite = d[np.isfinite(d)]
    scale = max(float(np.median(finite)) if finite.size else 1.0, 1e-6)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.argsort(d[i])[:k]:
            weight = float(np.exp(-float(dist[i, j]) / scale))
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)
    return normalize_adjacency(adj)


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    denom = mx - mn
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / denom
