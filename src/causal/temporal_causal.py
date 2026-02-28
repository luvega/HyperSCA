"""Temporal causal inference utilities."""
from __future__ import annotations

import numpy as np


def lagged_correlation_score(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Compute lagged correlation score corr(x[t], y[t+lag])."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    if len(x) <= lag or len(y) <= lag:
        return 0.0
    x0 = np.asarray(x[:-lag], dtype=float)
    y1 = np.asarray(y[lag:], dtype=float)
    if np.std(x0) < 1e-12 or np.std(y1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, y1)[0, 1])


def infer_temporal_causal_graph(
    temporal_effects: np.ndarray,
    *,
    max_lag: int = 2,
    threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer temporal causal adjacency and best-lag matrix from time series.

    temporal_effects: (T, K)
    """
    ts = np.asarray(temporal_effects, dtype=float)
    if ts.ndim != 2:
        raise ValueError("temporal_effects must be 2D [T, K]")
    t, k = ts.shape
    adj = np.zeros((k, k), dtype=float)
    best_lag = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            best = 0.0
            best_l = 0
            for lag in range(1, max_lag + 1):
                s = abs(lagged_correlation_score(ts[:, i], ts[:, j], lag))
                if s > best:
                    best = s
                    best_l = lag
            if best >= threshold:
                adj[i, j] = float(best)
                best_lag[i, j] = best_l
    return adj, best_lag
