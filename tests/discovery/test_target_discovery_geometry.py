from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.target_discovery.config import TargetDiscoveryConfig
from src.discovery.target_discovery.geometry import blend_adjacencies, compute_geometry
from src.discovery.target_discovery.utils import knn_adjacency, minmax, normalize_adjacency


def test_default_config_resolves_output_base():
    cfg = TargetDiscoveryConfig.default()
    assert cfg.paths.output_base.name == "target_discovery"
    assert cfg.geometry.geometry_k == 4
    assert "hyperbolic" in cfg.geometry.modes


def test_minmax_constant_returns_zeros():
    out = minmax(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(out, np.zeros(3))


def test_normalize_adjacency_clears_diagonal_and_scales():
    adj = np.array([[10.0, 2.0], [4.0, 10.0]])
    out = normalize_adjacency(adj)
    assert np.allclose(np.diag(out), [0.0, 0.0])
    assert float(out.max()) == 1.0


def test_knn_adjacency_is_symmetric():
    dist = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )
    out = knn_adjacency(dist, k=1)
    assert out.shape == (3, 3)
    assert np.allclose(out, out.T)
    assert np.allclose(np.diag(out), [0.0, 0.0, 0.0])


def test_blend_adjacencies_normalizes_result():
    spatial = np.array([[0.0, 1.0], [1.0, 0.0]])
    geom = np.array([[0.0, 0.5], [0.5, 0.0]])
    out = blend_adjacencies(spatial, geom, blend=0.5)
    assert np.allclose(out, out.T)
    assert float(out.max()) == 1.0


def test_compute_geometry_euclidean_small_expression():
    expr = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [3.0, 2.0, 1.0],
            "C": [0.5, 0.7, 0.9],
        },
        index=["n1", "n2", "n3"],
    )
    out = compute_geometry(expr, ["n1", "n2", "n3"], mode="euclidean", k=1)
    assert out["embedding"].shape[0] == 3
    assert out["dist_matrix"].shape == (3, 3)
    assert out["adjacency"].shape == (3, 3)
    assert out["metrics"]["mode"] == "euclidean"
