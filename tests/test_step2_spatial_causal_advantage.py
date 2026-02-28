"""Tests for Step2 spatial causal advantage against baseline communication."""
from __future__ import annotations

import numpy as np

from src.causal.baseline_communication import (
    build_baseline_communication_network,
    compare_spatial_causal_advantage,
)


def test_build_baseline_communication_network_shape():
    rng = np.random.default_rng(42)
    expr = rng.normal(size=(6, 30))
    adj = build_baseline_communication_network(expr, threshold_quantile=0.8)
    assert adj.shape == (6, 6)
    assert np.all(np.diag(adj) == 0)


def test_compare_spatial_causal_advantage_metrics():
    causal = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )
    base = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    spatial = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )
    m = compare_spatial_causal_advantage(causal, base, spatial_adj=spatial)
    assert "edge_overlap_jaccard" in m
    assert "spatial_consistency_gain" in m
    assert 0.0 <= m["edge_overlap_jaccard"] <= 1.0
