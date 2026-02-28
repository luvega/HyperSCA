"""Tests for temporal causal inference helpers."""
from __future__ import annotations

import numpy as np

from src.causal.temporal_causal import infer_temporal_causal_graph, lagged_correlation_score


def test_lagged_correlation_score_positive():
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    y = np.roll(x, 1)
    y[0] = 0.0
    s = lagged_correlation_score(x, y, lag=1)
    assert s > 0.5


def test_infer_temporal_causal_graph_shape():
    t = 30
    x = np.zeros((t, 3), dtype=float)
    for i in range(1, t):
        x[i, 0] = 0.8 * x[i - 1, 0] + 1.0
        x[i, 1] = 0.6 * x[i - 1, 0] + 0.4 * x[i - 1, 1]
        x[i, 2] = 0.7 * x[i - 1, 1] + 0.2 * x[i - 1, 2]
    adj, lag = infer_temporal_causal_graph(x, max_lag=2, threshold=0.2)
    assert adj.shape == (3, 3)
    assert lag.shape == (3, 3)
    assert np.any(adj > 0)
