"""Temporal-spatial propagation for intervention effects."""
from __future__ import annotations

import numpy as np


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square matrix")
    row_sum = a.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return a / row_sum


def simulate_temporal_spatial_propagation(
    base_effect: np.ndarray,
    time_grid: list[float],
    causal_adj: np.ndarray,
    *,
    spatial_adj: np.ndarray | None = None,
    diffusion: float = 0.35,
    temporal_decay: float = 0.08,
    max_effect: float = 1.0,
) -> np.ndarray:
    """Simulate effect(t, node) across time on graph."""
    t = np.asarray(time_grid, dtype=float)
    e0 = np.asarray(base_effect, dtype=float)
    if e0.ndim != 1:
        raise ValueError("base_effect must be 1D vector")
    k = len(e0)
    if k == 0:
        return np.zeros((len(t), 0), dtype=float)

    a_c = _normalize_adjacency(causal_adj)
    if spatial_adj is not None and spatial_adj.shape == causal_adj.shape:
        a_s = _normalize_adjacency(spatial_adj)
        a = 0.7 * a_c + 0.3 * a_s
    else:
        a = a_c

    out = np.zeros((len(t), k), dtype=float)
    out[0] = e0
    for i in range(1, len(t)):
        dt = max(float(t[i] - t[i - 1]), 1e-6)
        propagated = a.T @ out[i - 1]
        out[i] = (1.0 - temporal_decay * dt) * out[i - 1] + diffusion * propagated * dt
        out[i] = np.clip(out[i], 0.0, max_effect)
    return out
